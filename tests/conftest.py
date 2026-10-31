"""Shared test doubles: Telegram fakes and a ``Reminder`` factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from notifyme_bot.config import AITUNNEL_BASE_URL, Settings
from notifyme_bot.llm import OpenAICompatibleService
from notifyme_bot.models import Recurrence, Reminder


@dataclass
class FakeUser:
    id: int
    language_code: str | None = None


@dataclass
class FakeChat:
    id: int


@dataclass
class FakeMessage:
    text: str | None
    voice: Any = None


@dataclass
class FakeUpdate:
    effective_user: FakeUser | None
    effective_chat: FakeChat | None
    effective_message: FakeMessage | None


class FakeBot:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_message(self, **kwargs: Any) -> None:
        self.messages.append(kwargs)


@dataclass
class FakeApp:
    bot_data: dict
    bot: FakeBot


@dataclass
class FakeContext:
    application: FakeApp
    bot: FakeBot
    args: list[str] | None = None


def make_reminder(**overrides: Any) -> Reminder:
    """One-shot ``Reminder`` by default; pass ``recurrence=`` for a rule."""
    base: dict[str, Any] = {
        "id": 1,
        "user_id": 1,
        "chat_id": 100,
        "reminder_text": "task",
        "notification_text": "Notification: task",
        "remind_at_utc": datetime.now(UTC) + timedelta(days=1),
        "timezone": "UTC",
        "recurrence": None,
    }
    base.update(overrides)
    return Reminder(**base)


def make_recurrence(
    unit: str = "day", interval: int = 1, **overrides: Any
) -> Recurrence:
    """Recurrence rule; keyword overrides keep the call sites readable."""
    return Recurrence(unit=unit, interval=interval, **overrides)


def make_service(
    fake: FakeOpenAI | None = None,
) -> OpenAICompatibleService:
    """Real service; ``fake`` replaces its client when given.

    With no argument the client comes from whatever ``AsyncOpenAI`` is patched
    to, so tests that monkeypatch the constructor keep working.
    """
    service = OpenAICompatibleService(
        api_key="test-key",
        model="gpt-4.1-nano",
        base_url=AITUNNEL_BASE_URL,
        stt_model="voxtral-small-24b-2507-stt",
    )
    if fake is not None:
        service._client = fake  # type: ignore[assignment]
    return service


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "telegram_bot_token": "x",
        "base_url": AITUNNEL_BASE_URL,
        "api_key": "y",
        "model_name": "gpt-4.1-nano",
        "stt_model": "voxtral-small-24b-2507-stt",
        "database_path": "d.sqlite3",
        "default_timezone": "UTC",
        "default_locale": "en",
        "allow_new_users": True,
    }
    base.update(overrides)
    return Settings(**base)


class FakeChatCompletions:
    """Minimal stand-in for ``client.chat.completions`` (``parse`` helper)."""

    def __init__(self, content: str, captured: dict | None = None) -> None:
        self._content = content
        self._captured = captured

    async def parse(self, **kwargs: Any) -> Any:
        if self._captured is not None:
            self._captured.update(kwargs)
        # The real SDK returns ``parsed`` already built; mirror that and keep
        # ``content`` identical to a gateway that honours response_format.
        model = kwargs["response_format"]
        parsed = model.model_validate_json(self._content)
        message = SimpleNamespace(parsed=parsed, content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeTranscriptions:
    """Minimal stand-in for ``client.audio.transcriptions``."""

    def __init__(self, text: str, captured: dict | None = None) -> None:
        self._text = text
        self._captured = captured

    async def create(self, **kwargs: Any) -> Any:
        if self._captured is not None:
            self._captured.update(kwargs)
        return SimpleNamespace(text=self._text)


@dataclass
class FakeOpenAI:
    """Duck-typed ``AsyncOpenAI`` with injectable sub-resources."""

    content: str = "{}"
    transcript: str = ""
    captured_chat: dict = field(default_factory=dict)
    captured_audio: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.chat = SimpleNamespace(
            completions=FakeChatCompletions(self.content, self.captured_chat)
        )
        self.audio = SimpleNamespace(
            transcriptions=FakeTranscriptions(
                self.transcript, self.captured_audio
            )
        )
