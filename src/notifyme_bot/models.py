from __future__ import annotations

import calendar
from datetime import UTC, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator

#: Recurrence rule units: the whole vocabulary, no "unknown" and no fallback.
RecurrenceUnit = Literal["minute", "hour", "day", "week", "month"]

#: Units that are a plain fixed-length step on the wall clock.
_FIXED_STEPS: dict[str, timedelta] = {
    "minute": timedelta(minutes=1),
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
}


class Recurrence(BaseModel):
    """A recurrence rule and the only place that knows how to advance it.

    ``interval`` means "every N units". ``weekdays`` turns a weekly rule into
    a pattern (Mon/Wed/Fri) instead of a fixed step; that is why the interval
    is not applied to it - one week is the cycle by definition.
    """

    unit: RecurrenceUnit
    interval: int = Field(default=1, ge=1)
    #: Monday=0 .. Sunday=6, matching ``datetime.weekday()``.
    weekdays: list[int] = Field(default_factory=list)
    #: Monthly anchor, 1..31. Kept on the rule because deriving it from the
    #: previous fire loses it: 31 Jan -> 28 Feb would then stay on the 28th.
    day_of_month: int | None = Field(default=None, ge=1, le=31)

    @field_validator("weekdays")
    @classmethod
    def _normalize_weekdays(cls, value: list[int]) -> list[int]:
        """Deduplicate and order, dropping days outside Monday..Sunday."""
        return sorted({day for day in value if 0 <= day <= 6})

    def next_after(
        self,
        *,
        previous: datetime,
        now: datetime,
        timezone: str,
    ) -> datetime:
        """First occurrence strictly after both ``now`` and ``previous``.

        ``previous`` is the UTC instant just delivered, so wall-clock math
        keeps the original time of day. Calendar rules (weeks by weekday,
        months) are resolved in ``timezone``, everything is returned in UTC.
        """
        tz = ZoneInfo(timezone)
        limit = max(now, previous).astimezone(tz)
        if self.unit == "month":
            return self._next_month(previous, limit, tz)
        if self.unit == "week" and self.weekdays:
            return self._next_weekday(previous, limit, tz)
        return self._next_step(previous, limit, tz)

    def _next_step(
        self, previous: datetime, limit: datetime, tz: ZoneInfo
    ) -> datetime:
        step = _FIXED_STEPS[self.unit] * self.interval
        base = previous.astimezone(tz)
        # Jump over whole missed periods instead of walking each one; the
        # expression is exactly the first ``base + k*step`` that beats limit.
        candidate = base + step * ((limit - base) // step + 1)
        return candidate.astimezone(UTC)

    def _next_weekday(
        self, previous: datetime, limit: datetime, tz: ZoneInfo
    ) -> datetime:
        base = previous.astimezone(tz)
        # A weekday pattern repeats within one week, so 7 days is enough.
        for offset in range(1, 8):
            candidate = base + timedelta(days=offset)
            if candidate.weekday() in self.weekdays and candidate > limit:
                return candidate.astimezone(UTC)
        raise AssertionError("unreachable: a weekday pattern covers a week")

    def _next_month(
        self, previous: datetime, limit: datetime, tz: ZoneInfo
    ) -> datetime:
        base = previous.astimezone(tz)
        # The anchor lives on the rule; months without that day use their last
        # one (Jan 31 -> Feb 28 -> Mar 31, not a permanent drift to the 28th).
        anchor_day = self.day_of_month or base.day
        year, month = limit.year, limit.month
        for _ in range(36):
            last_day = calendar.monthrange(year, month)[1]
            candidate = base.replace(
                year=year, month=month, day=min(anchor_day, last_day)
            )
            if candidate > limit:
                return candidate.astimezone(UTC)
            month += self.interval
            year += (month - 1) // 12
            month = ((month - 1) % 12) + 1
        raise AssertionError("unreachable: 36 months always pass the limit")


class Reminder(BaseModel):
    """One row of ``reminders`` as the bot and scheduler see it."""

    id: int
    user_id: int
    chat_id: int
    reminder_text: str
    notification_text: str
    remind_at_utc: datetime
    timezone: str = "UTC"
    #: ``None`` is a one-shot reminder; there is no separate flag.
    recurrence: Recurrence | None = None

    def next_run(self, now: datetime) -> datetime | None:
        """Next fire of a recurring reminder; ``None`` when one-shot."""
        if self.recurrence is None:
            return None
        return self.recurrence.next_after(
            previous=self.remind_at_utc,
            now=now,
            timezone=self.timezone,
        )


class ParsedReminder(BaseModel):
    """Schedule intent extracted by the LLM, time already resolved to UTC."""

    reminder_text: str = Field(min_length=1)
    notification_text: str = Field(min_length=1)
    remind_at_utc: datetime
    recurrence: Recurrence | None = None


class ParsedDeleteRequest(BaseModel):
    """Delete intent: everything pending, or a set of reminder ids."""

    delete_all: bool
    delete_ids: list[int] = Field(default_factory=list)


class CommandParseResult(BaseModel):
    """Result of a single LLM extraction pass."""

    delete: ParsedDeleteRequest | None = None
    reminder: ParsedReminder | None = None


class CommandExtraction(BaseModel):
    """Generation schema: what the model is allowed to answer.

    One source of truth for the contract, the prompt and the validation: the
    SDK turns this class into the strict JSON schema sent to the API
    (``chat.completions.parse``) and parses the reply back into it. Dates stay
    local wall-clock anchors on purpose - :class:`LLMReminderParser` resolves
    them, never the model.
    """

    action: Literal["store", "delete", "none"] = "none"
    delete_all: bool = False
    delete_ids: list[int] = Field(default_factory=list)
    reminder_text: str = ""
    notification_text: str = ""
    #: ``YYYY-MM-DD`` in the user's timezone; empty when not stated.
    local_date: str = ""
    #: ``HH:MM`` 24-hour local wall clock; empty when not stated.
    local_time: str = ""
    #: Monday=0 .. Sunday=6; -1 is the "no weekday" sentinel.
    weekday: int = -1
    #: today | tomorrow | day_after_tomorrow; empty when not stated.
    relative_day: str = ""
    #: Pure offset in minutes ("in 10 minutes"); 0 when not stated.
    delay_minutes: int = 0
    is_recurring: bool = False
    #: Null for a one-shot reminder; the schema offers the enum plus null, so
    #: an unknown unit cannot reach the parser in the first place.
    recurrence_unit: RecurrenceUnit | None = None
    recurrence_interval: int = 0
    recurrence_weekdays: list[int] = Field(default_factory=list)

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: object) -> str:
        """Lower-case the intent; unknown intents degrade to ``none``."""
        action = str(value).strip().lower()
        return action if action in {"store", "delete", "none"} else "none"

    @field_validator("recurrence_unit", mode="before")
    @classmethod
    def _blank_unit_is_null(cls, value: object) -> object:
        """``"none"``/``""`` mean "no recurrence"; let the enum see ``None``.

        The prompt asks for null, but models routinely answer with the string
        ``"none"`` they learned earlier, and a rejected answer would cost the
        whole extraction.
        """
        if value is None:
            return None
        text = str(value).strip().lower()
        return text if text and text != "none" else None
