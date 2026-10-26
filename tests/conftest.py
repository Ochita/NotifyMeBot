"""Shared test doubles: Telegram fakes and a ``Reminder`` factory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
        "database_path": "d.sqlite3",
        "default_timezone": "UTC",
        "default_locale": "en",
        "allow_new_users": True,
    }
    base.update(overrides)
    return Settings(**base)
