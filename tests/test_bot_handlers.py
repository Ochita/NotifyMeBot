from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from conftest import (
    FakeApp,
    FakeBot,
    FakeChat,
    FakeContext,
    FakeMessage,
    FakeUpdate,
    FakeUser,
    make_reminder,
    make_settings,
)

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
from notifyme_bot.llm import LLMServiceError
from notifyme_bot.models import (
    CommandParseResult,
    ParsedDeleteRequest,
    ParsedReminder,
    Reminder,
)


class _FakeRepo:
    def __init__(self) -> None:
        self.subscribed: dict[int, bool] = {}
        self.created: list[Reminder] = []
        self.pending: list[Reminder] = []
        self.timezones: dict[int, str] = {}
        self.locales: dict[int, str] = {}
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

    async def set_user_locale(self, user_id: int, locale: str) -> None:
        self.locales[user_id] = locale

    async def get_user_locale(self, user_id: int) -> str | None:
        return self.locales.get(user_id)

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
        _ = (timezone, source_text)
        reminder = make_reminder(
            id=len(self.created) + 1,
            user_id=user_id,
            chat_id=chat_id,
            reminder_text=reminder_text,
            notification_text=notification_text,
            remind_at_utc=remind_at_utc,
            is_recurring=is_recurring,
            recurrence_unit=recurrence_unit,
            recurrence_interval=recurrence_interval,
            recurrence_weekdays=recurrence_weekdays,
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
        reminders: list | None = None,
    ) -> CommandParseResult:
        _ = (message_text, now_utc, reminders)
        self.last_timezone = user_timezone
        self._maybe_raise()
        if self.delete_result is not None:
            return CommandParseResult(delete=self.delete_result, reminder=None)
        return CommandParseResult(delete=None, reminder=self._result)


class _FakeScheduler:
    def __init__(self) -> None:
        self.scheduled: list[Reminder] = []

    def schedule(self, reminder: Reminder) -> None:
        self.scheduled.append(reminder)


@pytest.mark.asyncio
async def test_start_stop_help_and_list() -> None:
    bot = FakeBot()
    repo = _FakeRepo()
    app = FakeApp(
        bot_data={"repository": repo, "settings": make_settings()},
        bot=bot,
    )
    context = FakeContext(application=app, bot=bot)
    update = FakeUpdate(FakeUser(1), FakeChat(100), FakeMessage("hello"))

    await start(update, context)
    assert repo.subscribed[1] is True
    assert "Subscription enabled" in bot.messages[-1]["text"]

    await help_command(update, context)
    assert "/list" in bot.messages[-1]["text"]
    assert "/set_timezone" in bot.messages[-1]["text"]

    await list_reminders(update, context)
    assert "No pending reminders." in bot.messages[-1]["text"]

    repo.pending = [
        make_reminder(
            id=1,
            reminder_text="send money",
            notification_text="Notification: send money",
        )
    ]
    await list_reminders(update, context)
    assert "Pending reminders:" in bot.messages[-1]["text"]

    await stop(update, context)
    assert repo.subscribed[1] is False
    assert "Subscription disabled." in bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_on_text_message_unsubscribed() -> None:
    bot = FakeBot()
    repo = _FakeRepo()
    parser = _FakeParser(None)
    scheduler = _FakeScheduler()
    app = FakeApp(
        bot_data={
            "repository": repo,
            "parser": parser,
            "scheduler": scheduler,
            "settings": make_settings(),
        },
        bot=bot,
    )
    context = FakeContext(application=app, bot=bot)
    update = FakeUpdate(FakeUser(1), FakeChat(100), FakeMessage("remind me"))

    await on_text_message(update, context)
    assert "unsubscribed" in bot.messages[-1]["text"]
    assert scheduler.scheduled == []


@pytest.mark.asyncio
async def test_on_text_message_parse_none_and_error() -> None:
    bot = FakeBot()
    repo = _FakeRepo()
    repo.subscribed[1] = True
    scheduler = _FakeScheduler()

    app_none = FakeApp(
        bot_data={
            "repository": repo,
            "parser": _FakeParser(None),
            "scheduler": scheduler,
            "settings": make_settings(),
        },
        bot=bot,
    )
    context_none = FakeContext(application=app_none, bot=bot)
    update = FakeUpdate(FakeUser(1), FakeChat(100), FakeMessage("hello"))
    await on_text_message(update, context_none)
    assert "did not detect" in bot.messages[-1]["text"]

    app_err = FakeApp(
        bot_data={
            "repository": repo,
            "parser": _FakeParser(None, should_raise=True),
            "scheduler": scheduler,
            "settings": make_settings(),
        },
        bot=bot,
    )
    context_err = FakeContext(application=app_err, bot=bot)
    await on_text_message(update, context_err)
    assert "could not parse" in bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_on_text_message_llm_error_localized() -> None:
    bot = FakeBot()
    repo = _FakeRepo()
    repo.subscribed[1] = True
    scheduler = _FakeScheduler()
    app = FakeApp(
        bot_data={
            "repository": repo,
            "parser": _FakeParser(None, service_error=True),
            "scheduler": scheduler,
            "settings": make_settings(),
        },
        bot=bot,
    )
    context = FakeContext(application=app, bot=bot)
    update_ru = FakeUpdate(
        FakeUser(1, language_code="ru"),
        FakeChat(100),
        FakeMessage("Напомни мне позвонить завтра"),
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
    bot = FakeBot()
    repo = _FakeRepo()
    app = FakeApp(
        bot_data={"repository": repo, "settings": make_settings()},
        bot=bot,
    )
    context = FakeContext(application=app, bot=bot)
    update = FakeUpdate(
        FakeUser(2, language_code="ru"), FakeChat(101), FakeMessage("hi")
    )

    await start(update, context)

    assert "Подписка включена" in bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_start_localized_ukrainian_language_code() -> None:
    bot = FakeBot()
    repo = _FakeRepo()
    app = FakeApp(
        bot_data={"repository": repo, "settings": make_settings()},
        bot=bot,
    )
    context = FakeContext(application=app, bot=bot)
    update = FakeUpdate(
        FakeUser(3, language_code="uk"), FakeChat(102), FakeMessage("hi")
    )

    await start(update, context)

    assert "Підписку увімкнено" in bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_on_text_message_success_schedules() -> None:
    bot = FakeBot()
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
    app = FakeApp(
        bot_data={
            "repository": repo,
            "parser": parser,
            "scheduler": scheduler,
            "settings": make_settings(),
        },
        bot=bot,
    )
    context = FakeContext(application=app, bot=bot)
    update = FakeUpdate(FakeUser(1), FakeChat(100), FakeMessage("remind me"))

    await on_text_message(update, context)

    assert len(repo.created) == 1
    assert len(scheduler.scheduled) == 1
    assert "Reminder saved" in bot.messages[-1]["text"]
    assert parser.last_timezone == "Europe/Rome"


@pytest.mark.asyncio
async def test_on_text_message_delete_by_text() -> None:
    bot = FakeBot()
    repo = _FakeRepo()
    repo.subscribed[1] = True
    repo.pending = [
        make_reminder(
            id=10,
            reminder_text="pay Alexei",
            notification_text="Reminder: pay Alexei.",
        ),
        make_reminder(
            id=11,
            reminder_text="buy milk",
            notification_text="Reminder: buy milk.",
        ),
    ]
    parser = _FakeParser(None)
    parser.delete_result = ParsedDeleteRequest(
        delete_all=False,
        delete_ids=[10],
    )
    scheduler = _FakeScheduler()
    app = FakeApp(
        bot_data={
            "repository": repo,
            "parser": parser,
            "scheduler": scheduler,
            "settings": make_settings(),
        },
        bot=bot,
    )
    context = FakeContext(application=app, bot=bot)
    update = FakeUpdate(
        FakeUser(1), FakeChat(100), FakeMessage("cancel Alexei reminder")
    )

    await on_text_message(update, context)

    assert repo.deleted_ids == [10]
    assert "Deleted reminders: 1" in bot.messages[-1]["text"]
    assert scheduler.scheduled == []


@pytest.mark.asyncio
async def test_on_text_message_delete_none() -> None:
    bot = FakeBot()
    repo = _FakeRepo()
    repo.subscribed[1] = True
    parser = _FakeParser(None)
    parser.delete_result = ParsedDeleteRequest(
        delete_all=True,
        delete_ids=[],
    )
    scheduler = _FakeScheduler()
    app = FakeApp(
        bot_data={
            "repository": repo,
            "parser": parser,
            "scheduler": scheduler,
            "settings": make_settings(),
        },
        bot=bot,
    )
    context = FakeContext(application=app, bot=bot)
    update = FakeUpdate(
        FakeUser(1), FakeChat(100), FakeMessage("delete all reminders")
    )

    await on_text_message(update, context)

    assert "no pending reminders" in bot.messages[-1]["text"].lower()


@pytest.mark.asyncio
async def test_on_text_message_delete_not_found() -> None:
    bot = FakeBot()
    repo = _FakeRepo()
    repo.subscribed[1] = True
    repo.pending = [
        make_reminder(
            id=12,
            reminder_text="buy milk",
            notification_text="Reminder: buy milk.",
        )
    ]
    parser = _FakeParser(None)
    parser.delete_result = ParsedDeleteRequest(
        delete_all=False,
        delete_ids=[99],
    )
    scheduler = _FakeScheduler()
    app = FakeApp(
        bot_data={
            "repository": repo,
            "parser": parser,
            "scheduler": scheduler,
            "settings": make_settings(),
        },
        bot=bot,
    )
    context = FakeContext(application=app, bot=bot)
    update = FakeUpdate(
        FakeUser(1), FakeChat(100), FakeMessage("delete Alexei reminder")
    )

    await on_text_message(update, context)

    assert "could not find" in bot.messages[-1]["text"].lower()


@pytest.mark.asyncio
async def test_set_timezone_success_and_invalid() -> None:
    bot = FakeBot()
    repo = _FakeRepo()
    repo.subscribed[1] = True
    app = FakeApp(
        bot_data={"repository": repo, "settings": make_settings()},
        bot=bot,
    )
    update = FakeUpdate(FakeUser(1), FakeChat(100), FakeMessage(""))

    context_missing = FakeContext(application=app, bot=bot, args=[])
    await set_timezone(update, context_missing)
    assert "Please provide timezone" in bot.messages[-1]["text"]

    context_bad = FakeContext(application=app, bot=bot, args=["Bad/Timezone"])
    await set_timezone(update, context_bad)
    assert "Unknown timezone" in bot.messages[-1]["text"]

    context_ok = FakeContext(
        application=app,
        bot=bot,
        args=["Europe/Rome"],
    )
    await set_timezone(update, context_ok)
    assert repo.timezones[1] == "Europe/Rome"
    assert "Timezone saved: Europe/Rome" in bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_closed_registration_blocks_new_user() -> None:
    bot = FakeBot()
    repo = _FakeRepo()
    app = FakeApp(
        bot_data={
            "repository": repo,
            "settings": make_settings(allow_new_users=False),
        },
        bot=bot,
    )
    context = FakeContext(application=app, bot=bot)
    update = FakeUpdate(FakeUser(99), FakeChat(100), FakeMessage("hello"))

    await start(update, context)
    assert repo.subscribed.get(99) is None
    assert "not accepting new users" in bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_closed_registration_allows_whitelisted_user() -> None:
    bot = FakeBot()
    repo = _FakeRepo()
    repo.known_users.add(1)
    repo.subscribed[1] = True
    app = FakeApp(
        bot_data={
            "repository": repo,
            "settings": make_settings(allow_new_users=False),
        },
        bot=bot,
    )
    context = FakeContext(application=app, bot=bot)
    update = FakeUpdate(FakeUser(1), FakeChat(100), FakeMessage("hello"))

    await start(update, context)
    assert repo.subscribed[1] is True
    assert "Subscription enabled" in bot.messages[-1]["text"]


def test_list_recurrence_suffix_empty_for_one_shot() -> None:
    r = make_reminder(
        is_recurring=False,
        recurrence_unit=None,
        recurrence_interval=None,
    )
    assert _list_recurrence_suffix("en", r) == ""


def test_list_recurrence_suffix_minute_and_weekly() -> None:
    every_two = make_reminder(
        is_recurring=True,
        recurrence_unit="minute",
        recurrence_interval=2,
    )
    assert "every 2 min" in _list_recurrence_suffix("en", every_two)
    w = make_reminder(
        is_recurring=True,
        recurrence_unit="week",
        recurrence_interval=1,
        recurrence_weekdays=[0, 2],
    )
    out = _list_recurrence_suffix("en", w)
    assert "Mon" in out and "Wed" in out
