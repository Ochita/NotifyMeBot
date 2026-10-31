"""Handler tests against the real repository and fakes only at the edges.

The repository runs on a throwaway SQLite file and the LLM/Telegram edges are
fakes, so each test asserts what the user actually experiences: a row in the
database, a scheduled job, and a message in the chat.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import (
    FakeApp,
    FakeBot,
    FakeChat,
    FakeContext,
    FakeMessage,
    FakeUpdate,
    FakeUser,
    make_recurrence,
    make_reminder,
    make_settings,
)

from notifyme_bot.bot import (
    _list_recurrence_suffix,
    help_command,
    list_reminders,
    on_text_message,
    on_voice_message,
    set_locale,
    set_timezone,
    start,
    stop,
)
from notifyme_bot.db import ReminderRepository
from notifyme_bot.llm import LLMServiceError
from notifyme_bot.models import (
    CommandParseResult,
    ParsedDeleteRequest,
    ParsedReminder,
)


class _FakeParser:
    """Stub ``LLMReminderParser``: the extraction is scripted per test."""

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
        self.delete_result: ParsedDeleteRequest | None = None

    async def parse_command(
        self,
        message_text: str,
        now_utc: datetime,
        user_timezone: str | None = None,
        reminders: list | None = None,
    ) -> CommandParseResult:
        _ = (message_text, now_utc, reminders)
        self.last_timezone = user_timezone
        if self._service_error:
            raise LLMServiceError("LLM request failed")
        if self._should_raise:
            raise RuntimeError("parser failed")
        if self.delete_result is not None:
            return CommandParseResult(delete=self.delete_result, reminder=None)
        return CommandParseResult(delete=None, reminder=self._result)


class _FakeScheduler:
    def __init__(self) -> None:
        self.scheduled: list = []

    def schedule(self, reminder) -> None:
        self.scheduled.append(reminder)


async def _repo(tmp_path: Path) -> ReminderRepository:
    repo = ReminderRepository(str(tmp_path / "notifyme.sqlite3"))
    await repo.init()
    return repo


def _context(
    repo: ReminderRepository,
    *,
    bot: FakeBot | None = None,
    parser: _FakeParser | None = None,
    scheduler: _FakeScheduler | None = None,
    settings=None,
    llm=None,
) -> FakeContext:
    bot = bot or FakeBot()
    app = FakeApp(
        bot_data={
            "repository": repo,
            "parser": parser,
            "scheduler": scheduler,
            "settings": settings or make_settings(),
            "llm": llm,
        },
        bot=bot,
    )
    return FakeContext(application=app, bot=bot)


def _update(user_id: int = 1, chat_id: int = 100, text: str = "hello"):
    return FakeUpdate(FakeUser(user_id), FakeChat(chat_id), FakeMessage(text))


@pytest.mark.asyncio
async def test_closed_registration_blocks_command(tmp_path: Path) -> None:
    """When registration is closed, even /help needs a whitelisted user."""
    repo = await _repo(tmp_path)
    context = _context(repo, settings=make_settings(allow_new_users=False))

    await help_command(_update(user_id=99), context)

    assert "not accepting new users" in context.bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_start_stop_help_and_list(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    context = _context(repo)

    await start(_update(), context)
    assert await repo.is_subscribed(1) is True
    assert "Subscription enabled" in context.bot.messages[-1]["text"]

    await help_command(_update(), context)
    assert "/list" in context.bot.messages[-1]["text"]
    assert "/set_timezone" in context.bot.messages[-1]["text"]

    await list_reminders(_update(), context)
    assert "No pending reminders." in context.bot.messages[-1]["text"]

    await repo.create_reminder(
        user_id=1,
        chat_id=100,
        source_text="send money",
        reminder_text="send money",
        notification_text="Notification: send money",
        remind_at_utc=datetime.now(UTC) + timedelta(days=1),
        timezone="UTC",
    )
    await list_reminders(_update(), context)
    assert "Pending reminders:" in context.bot.messages[-1]["text"]
    assert "send money" in context.bot.messages[-1]["text"]

    await stop(_update(), context)
    assert await repo.is_subscribed(1) is False
    assert "Subscription disabled." in context.bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_on_text_message_unsubscribed(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    scheduler = _FakeScheduler()
    context = _context(repo, parser=_FakeParser(None), scheduler=scheduler)

    await on_text_message(_update(), context)

    assert "unsubscribed" in context.bot.messages[-1]["text"]
    assert scheduler.scheduled == []


@pytest.mark.asyncio
async def test_on_text_message_parse_none_and_error(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)
    scheduler = _FakeScheduler()

    context = _context(repo, parser=_FakeParser(None), scheduler=scheduler)
    await on_text_message(_update(), context)
    assert "did not detect" in context.bot.messages[-1]["text"]

    context = _context(
        repo,
        parser=_FakeParser(None, should_raise=True),
        scheduler=scheduler,
    )
    await on_text_message(_update(), context)
    assert "could not parse" in context.bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_on_text_message_llm_error_localized(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)
    context = _context(
        repo,
        parser=_FakeParser(None, service_error=True),
        scheduler=_FakeScheduler(),
    )
    update = FakeUpdate(
        FakeUser(1, language_code="ru"),
        FakeChat(100),
        FakeMessage("Напомни мне позвонить завтра"),
    )

    await on_text_message(update, context)

    assert "Ошибка сервиса" in context.bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_start_localized_by_user_language(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    context = _context(repo)
    update = FakeUpdate(
        FakeUser(2, language_code="ru"), FakeChat(101), FakeMessage("hi")
    )

    await start(update, context)

    assert "Подписка включена" in context.bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_stored_locale_wins_over_telegram_language(
    tmp_path: Path,
) -> None:
    """A /set_locale choice survives even when Telegram says otherwise."""
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=3, chat_id=102)
    await repo.set_user_locale(3, "uk")
    context = _context(repo)

    await start(
        FakeUpdate(FakeUser(3, "ru"), FakeChat(102), FakeMessage("hi")),
        context,
    )

    assert "Підписку увімкнено" in context.bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_on_text_message_success_schedules(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)
    await repo.set_user_timezone(1, "Europe/Rome")
    scheduler = _FakeScheduler()
    parsed = ParsedReminder(
        reminder_text="send 100$ to Alexei",
        notification_text="Notification: you must send 100$ to Alexei",
        remind_at_utc=datetime.now(UTC) + timedelta(days=2),
    )
    parser = _FakeParser(parsed)
    context = _context(repo, parser=parser, scheduler=scheduler)

    await on_text_message(_update(), context)

    stored = await repo.list_pending(1)
    assert [item.reminder_text for item in stored] == ["send 100$ to Alexei"]
    assert stored[0].timezone == "Europe/Rome"
    assert len(scheduler.scheduled) == 1
    assert "Reminder saved" in context.bot.messages[-1]["text"]
    assert parser.last_timezone == "Europe/Rome"


@pytest.mark.asyncio
async def test_on_text_message_delete_by_text(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)
    target = await repo.create_reminder(
        user_id=1,
        chat_id=100,
        source_text="pay Alexei",
        reminder_text="pay Alexei",
        notification_text="Reminder: pay Alexei.",
        remind_at_utc=datetime.now(UTC) + timedelta(days=1),
        timezone="UTC",
    )
    keep = await repo.create_reminder(
        user_id=1,
        chat_id=100,
        source_text="buy milk",
        reminder_text="buy milk",
        notification_text="Reminder: buy milk.",
        remind_at_utc=datetime.now(UTC) + timedelta(days=2),
        timezone="UTC",
    )
    parser = _FakeParser(None)
    parser.delete_result = ParsedDeleteRequest(
        delete_all=False, delete_ids=[target.id]
    )
    scheduler = _FakeScheduler()
    context = _context(repo, parser=parser, scheduler=scheduler)

    await on_text_message(_update(text="cancel Alexei reminder"), context)

    remaining = await repo.list_pending(1)
    assert [item.id for item in remaining] == [keep.id]
    assert "Deleted reminders: 1" in context.bot.messages[-1]["text"]
    assert scheduler.scheduled == []


@pytest.mark.asyncio
async def test_on_text_message_delete_all(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)
    for text in ("pay Alexei", "buy milk"):
        await repo.create_reminder(
            user_id=1,
            chat_id=100,
            source_text=text,
            reminder_text=text,
            notification_text=text,
            remind_at_utc=datetime.now(UTC) + timedelta(days=1),
            timezone="UTC",
        )
    parser = _FakeParser(None)
    parser.delete_result = ParsedDeleteRequest(delete_all=True, delete_ids=[])
    context = _context(repo, parser=parser, scheduler=_FakeScheduler())

    await on_text_message(_update(text="delete all reminders"), context)

    assert await repo.list_pending(1) == []
    assert "Deleted reminders: 2" in context.bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_list_shows_recurrence_and_user_timezone(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)
    await repo.set_user_timezone(1, "Europe/Rome")
    await repo.create_reminder(
        user_id=1,
        chat_id=100,
        source_text="standup",
        reminder_text="standup",
        notification_text="Reminder: standup.",
        remind_at_utc=datetime(2026, 11, 2, 8, 0, tzinfo=UTC),
        timezone="Europe/Rome",
        recurrence=make_recurrence("week", 1, weekdays=[0, 2]),
    )
    context = _context(repo)

    await list_reminders(_update(), context)

    text = context.bot.messages[-1]["text"]
    assert "09:00 CET" in text  # rendered in the user's timezone
    assert "Mon, Wed" in text


@pytest.mark.asyncio
async def test_on_text_message_delete_none(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)
    parser = _FakeParser(None)
    parser.delete_result = ParsedDeleteRequest(delete_all=True, delete_ids=[])
    context = _context(repo, parser=parser, scheduler=_FakeScheduler())

    await on_text_message(_update(text="delete all reminders"), context)

    assert "no pending reminders" in context.bot.messages[-1]["text"].lower()


@pytest.mark.asyncio
async def test_on_text_message_delete_not_found(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)
    await repo.create_reminder(
        user_id=1,
        chat_id=100,
        source_text="buy milk",
        reminder_text="buy milk",
        notification_text="Reminder: buy milk.",
        remind_at_utc=datetime.now(UTC) + timedelta(days=1),
        timezone="UTC",
    )
    parser = _FakeParser(None)
    parser.delete_result = ParsedDeleteRequest(
        delete_all=False, delete_ids=[99]
    )
    context = _context(repo, parser=parser, scheduler=_FakeScheduler())

    await on_text_message(_update(text="delete Alexei reminder"), context)

    assert "could not find" in context.bot.messages[-1]["text"].lower()
    assert len(await repo.list_pending(1)) == 1


@pytest.mark.asyncio
async def test_set_timezone_success_and_invalid(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)
    context = _context(repo)
    update = _update(text="")

    context.args = []
    await set_timezone(update, context)
    assert "Please provide timezone" in context.bot.messages[-1]["text"]

    context.args = ["Bad/Timezone"]
    await set_timezone(update, context)
    assert "Unknown timezone" in context.bot.messages[-1]["text"]

    context.args = ["Europe/Rome"]
    await set_timezone(update, context)
    assert await repo.get_user_timezone(1) == "Europe/Rome"
    assert "Timezone saved: Europe/Rome" in context.bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_set_locale_success_and_invalid(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)
    context = _context(repo)
    update = _update(text="")

    context.args = []
    await set_locale(update, context)
    assert "Provide a language code" in context.bot.messages[-1]["text"]

    context.args = ["zz"]
    await set_locale(update, context)
    assert "Unknown language" in context.bot.messages[-1]["text"]

    context.args = ["RU-it"]
    await set_locale(update, context)
    assert await repo.get_user_locale(1) == "ru"
    # The confirmation comes back in the language just chosen, not the old one.
    assert "Язык сохранен: ru" in context.bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_closed_registration_blocks_new_user(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    context = _context(repo, settings=make_settings(allow_new_users=False))

    await start(_update(user_id=99), context)

    assert await repo.user_exists(99) is False
    assert "not accepting new users" in context.bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_closed_registration_allows_whitelisted_user(
    tmp_path: Path,
) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)
    await repo.set_subscription(1, is_subscribed=False)
    context = _context(repo, settings=make_settings(allow_new_users=False))

    await start(_update(), context)

    assert await repo.is_subscribed(1) is True
    assert "Subscription enabled" in context.bot.messages[-1]["text"]


def test_list_recurrence_suffix_empty_for_one_shot() -> None:
    assert _list_recurrence_suffix("en", make_reminder()) == ""


def test_list_recurrence_suffix_minute_and_weekly() -> None:
    every_two = make_reminder(recurrence=make_recurrence("minute", 2))
    assert "every 2 min" in _list_recurrence_suffix("en", every_two)
    weekly = make_reminder(
        recurrence=make_recurrence("week", 1, weekdays=[0, 2])
    )
    out = _list_recurrence_suffix("en", weekly)
    assert "Mon" in out and "Wed" in out


class _FakeVoice:
    def __init__(self, file_id: str = "file-1") -> None:
        self.file_id = file_id
        self.mime_type = "audio/ogg"


class _FakeVoiceMessage(FakeMessage):
    def __init__(self, text: str | None = None) -> None:
        super().__init__(text)
        self.voice = _FakeVoice()
        self.replies: list[str] = []
        self.last_status: "_FakeStatus | None" = None

    async def reply_text(self, text: str) -> "_FakeStatus":
        status = _FakeStatus(text)
        self.replies.append(text)
        self.last_status = status
        return status


class _FakeStatus:
    def __init__(self, text: str) -> None:
        self.text = text
        self.edited: list[str] = []
        self.deleted = False

    async def edit_text(self, text: str) -> None:
        self.edited.append(text)

    async def delete(self) -> None:
        self.deleted = True


class _FakeTelegramFile:
    async def download_to_memory(self, buf) -> None:
        buf.write(b"OGG-DATA")


class _FakeVoiceBot(FakeBot):
    def __init__(self) -> None:
        super().__init__()
        self.file_requests: list[str] = []

    async def get_file(self, file_id: str) -> _FakeTelegramFile:
        self.file_requests.append(file_id)
        return _FakeTelegramFile()


class _FakeTranscriber:
    def __init__(self, text: str = "", error: Exception | None = None):
        self._text = text
        self._error = error
        self.calls: list[dict] = []

    async def transcribe(
        self, audio: bytes, mime_type: str, language: str | None = None
    ) -> str:
        self.calls.append(
            {"audio": audio, "mime": mime_type, "language": language}
        )
        if self._error is not None:
            raise self._error
        return self._text


def _voice_context(
    repo: ReminderRepository,
    transcriber: _FakeTranscriber,
    parser: _FakeParser | None,
    scheduler: _FakeScheduler | None,
):
    bot = _FakeVoiceBot()
    context = _context(
        repo, bot=bot, parser=parser, scheduler=scheduler, llm=transcriber
    )
    return context, bot


@pytest.mark.asyncio
async def test_voice_message_transcribes_and_schedules(tmp_path: Path) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)
    scheduler = _FakeScheduler()
    parsed = ParsedReminder(
        reminder_text="позвонить маме",
        notification_text="Напоминание: позвони маме.",
        remind_at_utc=datetime.now(UTC) + timedelta(hours=1),
    )
    transcriber = _FakeTranscriber("напомни позвонить маме")
    context, bot = _voice_context(
        repo, transcriber, _FakeParser(parsed), scheduler
    )
    update = FakeUpdate(FakeUser(1), FakeChat(100), _FakeVoiceMessage())

    await on_voice_message(update, context)

    assert bot.file_requests == ["file-1"]
    assert transcriber.calls[0]["audio"] == b"OGG-DATA"
    assert transcriber.calls[0]["mime"] == "audio/ogg"
    assert [item.reminder_text for item in await repo.list_pending(1)] == [
        "позвонить маме"
    ]
    assert len(scheduler.scheduled) == 1
    assert "Reminder saved" in bot.messages[-1]["text"]


@pytest.mark.asyncio
async def test_voice_message_empty_transcript_does_not_schedule(
    tmp_path: Path,
) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)
    scheduler = _FakeScheduler()
    context, _ = _voice_context(
        repo, _FakeTranscriber(""), _FakeParser(None), scheduler
    )
    update = FakeUpdate(FakeUser(1), FakeChat(100), _FakeVoiceMessage())

    await on_voice_message(update, context)

    assert await repo.list_pending(1) == []
    assert scheduler.scheduled == []
    assert update.effective_message.replies  # a status bubble was shown


@pytest.mark.asyncio
async def test_voice_message_transcription_error_is_reported(
    tmp_path: Path,
) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)
    transcriber = _FakeTranscriber(error=LLMServiceError("provider down"))
    context, _ = _voice_context(
        repo, transcriber, _FakeParser(None), _FakeScheduler()
    )
    message = _FakeVoiceMessage()
    update = FakeUpdate(FakeUser(1), FakeChat(100), message)

    await on_voice_message(update, context)

    assert await repo.list_pending(1) == []
    assert context.bot.messages == []
    status = message.last_status
    assert status is not None and "Could not transcribe" in status.edited[-1]


@pytest.mark.asyncio
async def test_voice_message_download_failure_is_reported(
    tmp_path: Path,
) -> None:
    repo = await _repo(tmp_path)
    await repo.upsert_user(user_id=1, chat_id=100)

    class _BrokenFile:
        async def download_to_memory(self, buf) -> None:
            raise RuntimeError("telegram file expired")

    class _BrokenBot(_FakeVoiceBot):
        async def get_file(self, file_id: str) -> _BrokenFile:
            self.file_requests.append(file_id)
            return _BrokenFile()

    bot = _BrokenBot()
    context = _context(
        repo, bot=bot, parser=_FakeParser(None), scheduler=_FakeScheduler()
    )
    message = _FakeVoiceMessage()
    update = FakeUpdate(FakeUser(1), FakeChat(100), message)

    await on_voice_message(update, context)

    assert await repo.list_pending(1) == []
    status = message.last_status
    assert status is not None and "Could not transcribe" in status.edited[-1]


@pytest.mark.asyncio
async def test_voice_message_unsubscribed_skips_transcription(
    tmp_path: Path,
) -> None:
    repo = await _repo(tmp_path)
    transcriber = _FakeTranscriber("ignored")
    context, _ = _voice_context(
        repo, transcriber, _FakeParser(None), _FakeScheduler()
    )
    update = FakeUpdate(FakeUser(1), FakeChat(100), _FakeVoiceMessage())

    await on_voice_message(update, context)

    assert transcriber.calls == []
    assert "unsubscribed" in context.bot.messages[-1]["text"]
