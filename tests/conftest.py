"""Shared test doubles: Telegram fakes and a ``Reminder`` factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from notifyme_bot.config import Settings
from notifyme_bot.models import Reminder


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
    """``Reminder`` with defaults; override only what a test cares about."""
    base: dict[str, Any] = {
        "id": 1,
        "user_id": 1,
        "chat_id": 100,
        "reminder_text": "task",
        "notification_text": "Notification: task",
        "remind_at_utc": datetime.now(UTC) + timedelta(days=1),
        "timezone": "UTC",
        "is_recurring": False,
        "recurrence_unit": None,
        "recurrence_interval": None,
        "recurrence_weekdays": None,
    }
    base.update(overrides)
    return Reminder(**base)


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "telegram_bot_token": "x",
        "base_url": None,
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
    """Minimal stand-in for ``client.chat.completions``."""

    def __init__(self, content: str, captured: dict | None = None) -> None:
        self._content = content
        self._captured = captured

    async def create(self, **kwargs: Any) -> Any:
        if self._captured is not None:
            self._captured.update(kwargs)
        message = SimpleNamespace(content=self._content)
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
