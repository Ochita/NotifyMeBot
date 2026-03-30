from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class Reminder:
    id: int
    user_id: int
    chat_id: int
    source_text: str
    reminder_text: str
    notification_text: str
    remind_at_utc: datetime
    timezone: str
    is_recurring: bool
    recurrence_unit: str | None
    recurrence_interval: int | None
    recurrence_weekdays: list[int] | None
    created_at_utc: datetime


@dataclass(slots=True)
class ParsedReminder:
    reminder_text: str
    notification_text: str
    remind_at_utc: datetime
    is_recurring: bool
    recurrence_unit: str | None
    recurrence_interval: int | None
    recurrence_weekdays: list[int] | None


@dataclass(slots=True)
class ParsedDeleteRequest:
    delete_all: bool
    target_text: str | None


@dataclass(slots=True)
class CommandParseResult:
    """Result of a single LLM extraction pass."""

    delete: ParsedDeleteRequest | None
    reminder: ParsedReminder | None
