from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from notifyme_bot.bot import (
    _list_recurrence_suffix,
    _t,
    help_command,
    list_reminders,
    on_text_message,
    set_timezone,
    start,
    stop,
)
from notifyme_bot.config import Settings
from notifyme_bot.llm.errors import LLMServiceError
from notifyme_bot.models import (
    CommandParseResult,
    ParsedDeleteRequest,
    ParsedReminder,
    Reminder,
)


@dataclass
class _User:
    id: int
    language_code: str | None = None


@dataclass
class _Chat:
    id: int


@dataclass
class _Message:
    text: str


class _FakeBot:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_message(self, **kwargs: int | str) -> None:
        self.messages.append(kwargs)


class _FakeRepo:
    def __init__(self) -> None:
        self.subscribed: dict[int, bool] = {}
        self.created: list[Reminder] = []
        self.pending: list[Reminder] = []
        self.timezones: dict[int, str] = {}
        self.deleted_ids: list[int] = []
        self.known_users: set[int] = set()

    async def user_exists(self, user_id: int) -> bool:
        return user_id in self.known_users

    async def upsert_user(self, user_id: int, chat_id: int) -> None:
        _ = chat_id
        self.known_users.add(user_id)
        self.subscribed[user_id] = True

    async def set_subscription(
        self, user_id: int, is_subscribed: bool
    ) -> None:
        self.subscribed[user_id] = is_subscribed

    async def is_subscribed(self, user_id: int) -> bool:
        return self.subscribed.get(user_id, False)

    async def set_user_timezone(self, user_id: int, timezone: str) -> None:
        self.timezones[user_id] = timezone

    async def get_user_timezone(self, user_id: int) -> str | None:
        return self.timezones.get(user_id)

    async def list_pending_for_user(self, user_id: int) -> list[Reminder]:
        _ = user_id
        return self.pending

    async def create_reminder(
        self,
        user_id: int,
        chat_id: int,
        source_text: str,
        reminder_text: str,
        notification_text: str,
        remind_at_utc: datetime,
        timezone: str,
        is_recurring: bool,
        recurrence_unit: str | None,
        recurrence_interval: int | None,
        recurrence_weekdays: list[int] | None,
    ) -> Reminder:
        _ = timezone
        reminder = Reminder(
            id=len(self.created) + 1,
            user_id=user_id,
            chat_id=chat_id,
            source_text=source_text,
            reminder_text=reminder_text,
            notification_text=notification_text,
            remind_at_utc=remind_at_utc,
            timezone="UTC",
            is_recurring=is_recurring,
            recurrence_unit=recurrence_unit,
            recurrence_interval=recurrence_interval,
            recurrence_weekdays=recurrence_weekdays,
            created_at_utc=datetime.now(UTC),
        )
        self.created.append(reminder)
        return reminder

    async def delete_reminder(self, reminder_id: int) -> None:
        self.deleted_ids.append(reminder_id)
        self.pending = [
            item for item in self.pending if item.id != reminder_id
        ]


class _FakeParser:
    """Stub ``LLMReminderParser`` used by ``on_text_message`` tests."""

    def __init__(
        self,
        result: ParsedReminder | None,
        should_raise: bool = False,
        service_error: bool = False,
    ) -> None:
        self._result = result
        self._should_raise = should_raise
        self._service_error = service_error
        self.last_timezone: str | None = None
        self.delete_result = None

    def _maybe_raise(self) -> None:
        if self._service_error:
            raise LLMServiceError("LLM request failed")
        if self._should_raise:
            raise RuntimeError("parser failed")

    async def parse_command(
        self,
        message_text: str,
        now_utc: datetime,
        user_timezone: str | None = None,
    ) -> CommandParseResult:
        _ = (message_text, now_utc)
        self.last_timezone = user_timezone
        self._maybe_raise()
        if self.delete_result is not None:
            return CommandParseResult(
                delete=self.delete_result, reminder=None
            )
        return CommandParseResult(delete=None, reminder=self._result)


class _FakeScheduler:
    def __init__(self) -> None:
        self.scheduled: list[Reminder] = []

    def schedule(self, reminder: Reminder) -> None:
        self.scheduled.append(reminder)


@dataclass
class _App:
    bot_data: dict
    bot: _FakeBot


@dataclass
class _Context:
    application: _App
    bot: _FakeBot
    args: list[str] | None = None


@dataclass
class _Update:
    effective_user: _User | None
    effective_chat: _Chat | None
    effective_message: _Message | None


def _settings(allow_new_users: bool = True) -> Settings:
    return Settings(
        telegram_bot_token="x",
        api_key="y",
        model_name="gemini-2.5-flash",
        database_path="d.sqlite3",
        default_timezone="UTC",
        allow_new_users=allow_new_users,
        llm_provider="gemini",
        llm_base_url=None,
    )


@pytest.mark.asyncio
async def test_start_stop_help_and_list() -> None:
    bot = _FakeBot()
    repo = _FakeRepo()
    app = _App(
        bot_data={"repository": repo, "settings": _settings()},
        bot=bot,
    )
    context = _Context(application=app, bot=bot)
    update = _Update(_User(1), _Chat(100), _Message("hello"))

    await start(update, context)
    assert repo.subscribed[1] is True
    assert "Subscription enabled" in bot.messages[-1]["text"]

    await help_command(update, context)
    assert "/list" in bot.messages[-1]["text"]
    assert "/set_timezone" in bot.messages[-1]["text"]

    await list_reminders(update, context)
    assert "No pending reminders." in bot.messages[-1]["text"]

    repo.pending = [
        Reminder(
            id=1,
            user_id=1,
            chat_id=100,
            source_text="s",
            reminder_text="send money",
            notification_text="Notification: send money",
            remind_at_utc=datetime.now(UTC) + timedelta(days=1),
            timezone="UTC",
            is_recurring=False,
            recurrence_unit=None,
            recurrence_interval=None,
            recurrence_weekdays=None,
            created_at_utc=datetime.now(UTC),
        )
    ]
    await list_reminders(update, context)
    assert "Pending reminders:" in bot.messages[-1]["text"]

    await stop(update, context)
    assert repo.subscribed[1] is False
    assert "Subscription disabled." in bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_on_text_message_unsubscribed() -> None:
    bot = _FakeBot()
    repo = _FakeRepo()
    parser = _FakeParser(None)
    scheduler = _FakeScheduler()
    app = _App(
        bot_data={
            "repository": repo,
            "parser": parser,
            "scheduler": scheduler,
            "settings": _settings(),
        },
        bot=bot,
    )
    context = _Context(application=app, bot=bot)
    update = _Update(_User(1), _Chat(100), _Message("remind me"))

    await on_text_message(update, context)
    assert "unsubscribed" in bot.messages[-1]["text"]
    assert scheduler.scheduled == []


@pytest.mark.asyncio
async def test_on_text_message_parse_none_and_error() -> None:
    bot = _FakeBot()
    repo = _FakeRepo()
    repo.subscribed[1] = True
    scheduler = _FakeScheduler()

    app_none = _App(
        bot_data={
            "repository": repo,
            "parser": _FakeParser(None),
            "scheduler": scheduler,
            "settings": _settings(),
        },
        bot=bot,
    )
    context_none = _Context(application=app_none, bot=bot)
    update = _Update(_User(1), _Chat(100), _Message("hello"))
    await on_text_message(update, context_none)
    assert "did not detect" in bot.messages[-1]["text"]

    app_err = _App(
        bot_data={
            "repository": repo,
            "parser": _FakeParser(None, should_raise=True),
            "scheduler": scheduler,
            "settings": _settings(),
        },
        bot=bot,
    )
    context_err = _Context(application=app_err, bot=bot)
    await on_text_message(update, context_err)
    assert "could not parse" in bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_on_text_message_llm_error_localized() -> None:
    bot = _FakeBot()
    repo = _FakeRepo()
    repo.subscribed[1] = True
    scheduler = _FakeScheduler()
    app = _App(
        bot_data={
            "repository": repo,
            "parser": _FakeParser(None, service_error=True),
            "scheduler": scheduler,
            "settings": _settings(),
        },
        bot=bot,
    )
    context = _Context(application=app, bot=bot)
    update_ru = _Update(
        _User(1), _Chat(100), _Message("Напомни мне позвонить завтра")
    )
    await on_text_message(update_ru, context)
    assert "Ошибка сервиса" in bot.messages[-1]["text"]


def test_i18n_translation_llm_error_non_english() -> None:
    assert "Errore del servizio promemoria" in _t("it", "llm_error")
    assert "Erreur du service" in _t("fr", "llm_error")
    assert "Fehler im Erinnerungsdienst" in _t("de", "llm_error")
    assert "Помилка сервісу" in _t("uk", "llm_error")
    assert "Error del servicio" in _t("es", "llm_error")


@pytest.mark.asyncio
async def test_start_localized_by_user_language() -> None:
    bot = _FakeBot()
    repo = _FakeRepo()
    app = _App(
        bot_data={"repository": repo, "settings": _settings()},
        bot=bot,
    )
    context = _Context(application=app, bot=bot)
    update = _Update(_User(2, language_code="ru"), _Chat(101), _Message("hi"))

    await start(update, context)

    assert "Подписка включена" in bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_start_localized_ukrainian_language_code() -> None:
    bot = _FakeBot()
    repo = _FakeRepo()
    app = _App(
        bot_data={"repository": repo, "settings": _settings()},
        bot=bot,
    )
    context = _Context(application=app, bot=bot)
    update = _Update(_User(3, language_code="uk"), _Chat(102), _Message("hi"))

    await start(update, context)

    assert "Підписку увімкнено" in bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_on_text_message_success_schedules() -> None:
    bot = _FakeBot()
    repo = _FakeRepo()
    repo.subscribed[1] = True
    repo.timezones[1] = "Europe/Rome"
    scheduler = _FakeScheduler()
    parsed = ParsedReminder(
        reminder_text="send 100$ to Alexei",
        notification_text="Notification: you must send 100$ to Alexei",
        remind_at_utc=datetime.now(UTC) + timedelta(days=2),
        is_recurring=False,
        recurrence_unit=None,
        recurrence_interval=None,
        recurrence_weekdays=None,
    )
    parser = _FakeParser(parsed)
    app = _App(
        bot_data={
            "repository": repo,
            "parser": parser,
            "scheduler": scheduler,
            "settings": _settings(),
        },
        bot=bot,
    )
    context = _Context(application=app, bot=bot)
    update = _Update(_User(1), _Chat(100), _Message("remind me"))

    await on_text_message(update, context)

    assert len(repo.created) == 1
    assert len(scheduler.scheduled) == 1
    assert "Reminder saved" in bot.messages[-1]["text"]
    assert parser.last_timezone == "Europe/Rome"


@pytest.mark.asyncio
async def test_on_text_message_delete_by_text() -> None:
    bot = _FakeBot()
    repo = _FakeRepo()
    repo.subscribed[1] = True
    repo.pending = [
        Reminder(
            id=10,
            user_id=1,
            chat_id=100,
            source_text="remind me to pay Alexei",
            reminder_text="pay Alexei",
            notification_text="Reminder: pay Alexei.",
            remind_at_utc=datetime.now(UTC) + timedelta(days=1),
            timezone="UTC",
            is_recurring=False,
            recurrence_unit=None,
            recurrence_interval=None,
            recurrence_weekdays=None,
            created_at_utc=datetime.now(UTC),
        ),
        Reminder(
            id=11,
            user_id=1,
            chat_id=100,
            source_text="remind me to buy milk",
            reminder_text="buy milk",
            notification_text="Reminder: buy milk.",
            remind_at_utc=datetime.now(UTC) + timedelta(days=1),
            timezone="UTC",
            is_recurring=False,
            recurrence_unit=None,
            recurrence_interval=None,
            recurrence_weekdays=None,
            created_at_utc=datetime.now(UTC),
        ),
    ]
    parser = _FakeParser(None)
    parser.delete_result = ParsedDeleteRequest(
        delete_all=False,
        target_text="Alexei",
    )
    scheduler = _FakeScheduler()
    app = _App(
        bot_data={
            "repository": repo,
            "parser": parser,
            "scheduler": scheduler,
            "settings": _settings(),
        },
        bot=bot,
    )
    context = _Context(application=app, bot=bot)
    update = _Update(_User(1), _Chat(100), _Message("cancel Alexei reminder"))

    await on_text_message(update, context)

    assert repo.deleted_ids == [10]
    assert "Deleted reminders: 1" in bot.messages[-1]["text"]
    assert scheduler.scheduled == []


@pytest.mark.asyncio
async def test_on_text_message_delete_none() -> None:
    bot = _FakeBot()
    repo = _FakeRepo()
    repo.subscribed[1] = True
    parser = _FakeParser(None)
    parser.delete_result = ParsedDeleteRequest(
        delete_all=True,
        target_text=None,
    )
    scheduler = _FakeScheduler()
    app = _App(
        bot_data={
            "repository": repo,
            "parser": parser,
            "scheduler": scheduler,
            "settings": _settings(),
        },
        bot=bot,
    )
    context = _Context(application=app, bot=bot)
    update = _Update(_User(1), _Chat(100), _Message("delete all reminders"))

    await on_text_message(update, context)

    assert "no pending reminders" in bot.messages[-1]["text"].lower()


@pytest.mark.asyncio
async def test_on_text_message_delete_not_found() -> None:
    bot = _FakeBot()
    repo = _FakeRepo()
    repo.subscribed[1] = True
    repo.pending = [
        Reminder(
            id=12,
            user_id=1,
            chat_id=100,
            source_text="remind me to buy milk",
            reminder_text="buy milk",
            notification_text="Reminder: buy milk.",
            remind_at_utc=datetime.now(UTC) + timedelta(days=1),
            timezone="UTC",
            is_recurring=False,
            recurrence_unit=None,
            recurrence_interval=None,
            recurrence_weekdays=None,
            created_at_utc=datetime.now(UTC),
        )
    ]
    parser = _FakeParser(None)
    parser.delete_result = ParsedDeleteRequest(
        delete_all=False,
        target_text="Alexei",
    )
    scheduler = _FakeScheduler()
    app = _App(
        bot_data={
            "repository": repo,
            "parser": parser,
            "scheduler": scheduler,
            "settings": _settings(),
        },
        bot=bot,
    )
    context = _Context(application=app, bot=bot)
    update = _Update(_User(1), _Chat(100), _Message("delete Alexei reminder"))

    await on_text_message(update, context)

    assert "could not find" in bot.messages[-1]["text"].lower()


@pytest.mark.asyncio
async def test_set_timezone_success_and_invalid() -> None:
    bot = _FakeBot()
    repo = _FakeRepo()
    repo.subscribed[1] = True
    app = _App(
        bot_data={"repository": repo, "settings": _settings()},
        bot=bot,
    )
    update = _Update(_User(1), _Chat(100), _Message(""))

    context_missing = _Context(application=app, bot=bot, args=[])
    await set_timezone(update, context_missing)
    assert "Please provide timezone" in bot.messages[-1]["text"]

    context_bad = _Context(application=app, bot=bot, args=["Bad/Timezone"])
    await set_timezone(update, context_bad)
    assert "Unknown timezone" in bot.messages[-1]["text"]

    context_ok = _Context(
        application=app,
        bot=bot,
        args=["Europe/Rome"],
    )
    await set_timezone(update, context_ok)
    assert repo.timezones[1] == "Europe/Rome"
    assert "Timezone saved: Europe/Rome" in bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_closed_registration_blocks_new_user() -> None:
    bot = _FakeBot()
    repo = _FakeRepo()
    app = _App(
        bot_data={
            "repository": repo,
            "settings": _settings(allow_new_users=False),
        },
        bot=bot,
    )
    context = _Context(application=app, bot=bot)
    update = _Update(_User(99), _Chat(100), _Message("hello"))

    await start(update, context)
    assert repo.subscribed.get(99) is None
    assert "not accepting new users" in bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_closed_registration_allows_whitelisted_user() -> None:
    bot = _FakeBot()
    repo = _FakeRepo()
    repo.known_users.add(1)
    repo.subscribed[1] = True
    app = _App(
        bot_data={
            "repository": repo,
            "settings": _settings(allow_new_users=False),
        },
        bot=bot,
    )
    context = _Context(application=app, bot=bot)
    update = _Update(_User(1), _Chat(100), _Message("hello"))

    await start(update, context)
    assert repo.subscribed[1] is True
    assert "Subscription enabled" in bot.messages[-1]["text"]


def _reminder_row(**overrides: object) -> Reminder:
    base = {
        "id": 1,
        "user_id": 1,
        "chat_id": 100,
        "source_text": "s",
        "reminder_text": "task",
        "notification_text": "n",
        "remind_at_utc": datetime.now(UTC) + timedelta(days=1),
        "timezone": "UTC",
        "is_recurring": True,
        "recurrence_unit": "minute",
        "recurrence_interval": 2,
        "recurrence_weekdays": None,
        "created_at_utc": datetime.now(UTC),
    }
    base.update(overrides)
    return Reminder(**base)


def test_list_recurrence_suffix_empty_for_one_shot() -> None:
    r = _reminder_row(
        is_recurring=False,
        recurrence_unit=None,
        recurrence_interval=None,
    )
    assert _list_recurrence_suffix("en", r) == ""


def test_list_recurrence_suffix_minute_and_weekly() -> None:
    assert "every 2 min" in _list_recurrence_suffix("en", _reminder_row())
    w = _reminder_row(
        recurrence_unit="week",
        recurrence_interval=1,
        recurrence_weekdays=[0, 2],
    )
    out = _list_recurrence_suffix("en", w)
    assert "Mon" in out and "Wed" in out
