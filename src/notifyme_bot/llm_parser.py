from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from notifyme_bot.llm import OpenAICompatibleService
from notifyme_bot.models import (
    CommandExtraction,
    CommandParseResult,
    ParsedDeleteRequest,
    ParsedReminder,
    Recurrence,
    Reminder,
)

LOGGER = logging.getLogger(__name__)

_DEFAULT_TIME = time(9, 0)
_RELATIVE_DAY_OFFSETS = {"today": 0, "tomorrow": 1, "day_after_tomorrow": 2}


class LLMReminderParser:
    """Reminder/delete parsing via one chat completion.

    The model only extracts what the user literally said (weekday, relative
    day, explicit date, local wall-clock time). Weekday arithmetic and
    timezone handling happen in :meth:`_resolve_fire_time`, because asking a
    small model for a UTC timestamp silently produced wrong days and shifted
    hours.
    """

    def __init__(
        self,
        provider: OpenAICompatibleService,
        default_timezone: str,
    ) -> None:
        self._provider = provider
        self._default_timezone = default_timezone

    async def parse_command(
        self,
        message_text: str,
        now_utc: datetime,
        user_timezone: str | None = None,
        reminders: list[Reminder] | None = None,
    ) -> CommandParseResult:
        """One LLM call: classify and extract delete and/or schedule."""
        now_utc = now_utc.astimezone(UTC)
        timezone_name = user_timezone or self._default_timezone
        local_now = now_utc.astimezone(ZoneInfo(timezone_name))
        data = await self._extract_command(
            message_text=message_text,
            local_now=local_now,
            timezone_name=timezone_name,
            reminders=reminders or [],
        )
        if data.action == "delete":
            return CommandParseResult(
                delete=ParsedDeleteRequest(
                    delete_all=data.delete_all,
                    delete_ids=[]
                    if data.delete_all
                    else self._unique_ids(data),
                )
            )
        return CommandParseResult(
            reminder=self._build_parsed_reminder_from_data(
                data, now_utc, timezone_name
            )
        )

    def _build_parsed_reminder_from_data(
        self,
        data: CommandExtraction,
        now_utc: datetime,
        timezone_name: str,
    ) -> ParsedReminder | None:
        if data.action != "store":
            return None

        reminder_text = data.reminder_text.strip()
        if not reminder_text:
            return None
        notification_text = data.notification_text.strip()

        recurrence = self._build_recurrence(data)
        if data.is_recurring and recurrence is None:
            # "every" something we cannot name: refuse instead of guessing.
            return None
        # A non-recurring message still carrying a unit is a one-shot: the
        # flag wins, and the rule is dropped rather than half-applied.
        if not data.is_recurring:
            recurrence = None

        remind_at = self._resolve_fire_time(
            data=data,
            now_utc=now_utc,
            timezone_name=timezone_name,
            recurrence=recurrence,
        )
        if remind_at is None:
            LOGGER.warning(
                "Schedule intent without resolvable local time: %s", data
            )
            return None

        if recurrence is not None and recurrence.unit == "month":
            # Anchor the monthly grid to the day the first fire lands on, in
            # the user's timezone; the scheduler reuses it from then on.
            local_day = remind_at.astimezone(ZoneInfo(timezone_name)).day
            recurrence = recurrence.model_copy(
                update={"day_of_month": local_day}
            )

        return ParsedReminder(
            reminder_text=reminder_text,
            notification_text=notification_text or reminder_text,
            remind_at_utc=remind_at,
            recurrence=recurrence,
        )

    @staticmethod
    def _build_recurrence(data: CommandExtraction) -> Recurrence | None:
        """Rule from the extraction, or ``None`` when it names no valid unit.

        ``weekdays`` only make sense for a weekly rule; on any other unit they
        are ignored, because "every 2 days on Monday" has no defined meaning.
        """
        if data.recurrence_unit is None:
            return None
        return Recurrence(
            unit=data.recurrence_unit,
            interval=max(data.recurrence_interval, 1),
            # Only a weekly rule uses a weekday list; on any other unit it
            # would be silently ignored, so drop it at the boundary.
            weekdays=(
                data.recurrence_weekdays
                if data.recurrence_unit == "week"
                else []
            ),
        )

    @staticmethod
    def _resolve_fire_time(
        data: CommandExtraction,
        now_utc: datetime,
        timezone_name: str,
        recurrence: Recurrence | None = None,
    ) -> datetime | None:
        """Resolve the first fire instant from the extracted local anchors.

        Precedence: explicit delay, then relative day, then weekday, then an
        explicit calendar date, then a bare clock time (today if still ahead,
        otherwise tomorrow). Everything is combined in the user's timezone and
        returned in UTC; the model never sees a timezone offset.
        """
        tz = ZoneInfo(timezone_name)
        local_now = now_utc.astimezone(tz)

        if data.delay_minutes > 0:
            return (
                local_now + timedelta(minutes=data.delay_minutes)
            ).astimezone(UTC)

        relative_day = data.relative_day.strip().lower()
        if relative_day not in _RELATIVE_DAY_OFFSETS:
            relative_day = ""

        weekday_given, weekday = LLMReminderParser._weekday_anchor(
            data.weekday
        )
        raw_date = data.local_date.strip()
        explicit_date = LLMReminderParser._parse_local_date(raw_date)
        # Garbage where a concrete date/weekday was expected must not silently
        # become "tomorrow": report failure so the bot asks again.
        if weekday_given and weekday is None:
            return None
        if raw_date and explicit_date is None:
            return None

        weekdays = list(recurrence.weekdays if recurrence else [])
        if weekday is not None:
            weekdays = sorted(set(weekdays) | {weekday})

        at = LLMReminderParser._parse_local_time(data.local_time)
        clock_given = bool(data.local_time.strip())
        if clock_given and at is None:
            # A clock time the user stated but we cannot read: never silently
            # substitute 09:00, let the bot ask again.
            return None
        calendar_anchor = bool(
            relative_day or weekdays or explicit_date is not None
        )
        if at is None and not calendar_anchor:
            unit = recurrence.unit if recurrence else None
            if unit in {"minute", "hour"}:
                # "каждые 2 часа": first fire one interval from now.
                assert recurrence is not None
                step = timedelta(**{f"{unit}s": recurrence.interval})
                return (local_now + step).astimezone(UTC)
            if unit is not None:
                # A calendar rule with no day would silently drift.
                return None
        if at is None:
            at = (
                _DEFAULT_TIME
                if calendar_anchor
                else local_now.time().replace(second=0, microsecond=0)
            )

        if relative_day:
            day = local_now.date() + timedelta(
                days=_RELATIVE_DAY_OFFSETS[relative_day]
            )
            candidate = datetime.combine(day, at, tzinfo=tz)
            if candidate <= local_now:
                # "today at 9" said at 10 means tomorrow, not the past.
                candidate += timedelta(days=1)
            return candidate.astimezone(UTC)

        if weekdays:
            return LLMReminderParser._next_weekday_fire(
                weekdays, at, local_now, tz
            )

        if explicit_date is not None:
            candidate = datetime.combine(explicit_date, at, tzinfo=tz)
            # Never guess a different day for a date stated in the past.
            if candidate <= local_now:
                return None
            return candidate.astimezone(UTC)

        if recurrence and recurrence.unit == "month" and not calendar_anchor:
            # Monthly grid needs an explicit day, else it drifts month to month.
            return None

        # Bare clock time ("напомни в 15:00", "каждый день в 15:00"):
        # today if the time is still ahead, tomorrow otherwise.
        candidate = datetime.combine(local_now.date(), at, tzinfo=tz)
        if candidate <= local_now:
            candidate += timedelta(days=1)
        return candidate.astimezone(UTC)

    @staticmethod
    def _weekday_anchor(value: object) -> tuple[bool, int | None]:
        """``(was provided, weekday)``; weekday is ``None`` when unusable."""
        text = str(value).strip() if value is not None else ""
        if not text:
            return False, None
        try:
            number = int(text)
        except (TypeError, ValueError):
            return True, None
        if number == -1:  # schema sentinel for "no weekday"
            return False, None
        return True, number if 0 <= number <= 6 else None

    @staticmethod
    def _next_weekday_fire(
        weekdays: list[int],
        at: time,
        local_now: datetime,
        tz: ZoneInfo,
    ) -> datetime:
        """Nearest upcoming occurrence among ``weekdays`` (0=Monday).

        ``weekdays`` is always non-empty and within 0..6: ``Recurrence``
        normalizes the rule's list and ``_weekday_anchor`` drops bad values.
        """

        def fire(weekday: int) -> datetime:
            days_ahead = (weekday - local_now.weekday()) % 7
            candidate = datetime.combine(
                local_now.date() + timedelta(days=days_ahead), at, tzinfo=tz
            )
            return (
                candidate
                if candidate > local_now
                else candidate + timedelta(days=7)
            )

        return min(map(fire, weekdays)).astimezone(UTC)

    @staticmethod
    def _parse_local_time(value: object) -> time | None:
        text = str(value or "").strip()
        if not text:
            return None
        for candidate in (text, f"0{text}"):
            try:
                return time.fromisoformat(candidate)
            except ValueError:
                continue
        return None

    @staticmethod
    def _parse_local_date(value: object) -> date | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None

    @staticmethod
    def _unique_ids(data: CommandExtraction) -> list[int]:
        """Ids in the order stated, duplicates dropped."""
        return list(dict.fromkeys(data.delete_ids))

    async def _extract_command(
        self,
        message_text: str,
        local_now: datetime,
        timezone_name: str,
        reminders: list[Reminder],
    ) -> CommandExtraction:
        extraction = await self._provider.chat_structured(
            system_prompt=self._command_prompt(
                local_now=local_now,
                timezone_name=timezone_name,
                reminders=reminders,
            ),
            user_prompt=message_text,
            response_model=CommandExtraction,
        )
        LOGGER.debug("LLM command payload: %s", extraction.model_dump())
        return extraction

    @staticmethod
    def _command_prompt(
        local_now: datetime,
        timezone_name: str,
        reminders: list[Reminder],
    ) -> str:
        if reminders:
            catalog = "\n".join(
                f"{item.id}: {item.reminder_text}" for item in reminders
            )
        else:
            catalog = "none"
        weekday_name = local_now.strftime("%A")
        return (
            "You extract reminder commands from user messages for a Telegram "
            "reminder bot. Return one strict JSON object with exactly the "
            "declared keys. Do not include examples or explanations.\n\n"
            "INTENT\n"
            "action=store when the user asks to be reminded of something. "
            "action=delete when the user asks to cancel or remove reminders. "
            "Otherwise action=none. If a message reads as both delete and "
            "schedule, prefer delete. delete_all=true only when the "
            "user clearly asks to remove all pending reminders. delete_ids "
            "must list only ids from Pending reminders; use [] when nothing "
            "matches. Do not invent ids, tasks or entities.\n\n"
            "TEXT\n"
            "reminder_text: the user task, verbatim and concise. "
            "notification_text: ready-to-send message that starts with a short "
            'label meaning "notification" in the SAME language and script as '
            "the user's message (English: Notification:, Russian: "
            "Напоминание: or Уведомление:, Ukrainian: Нагадування:, Spanish: "
            "Notificación:, German: Erinnerung: or Benachrichtigung:, Chinese: "
            "通知：), then one space unless the language normally omits it, then "
            "one imperative request in the user's language. Do not repeat "
            "reminder_text.\n\n"
            "WHEN (critical)\n"
            "Never convert to UTC, never compute offsets or timezone shifts. "
            "Report only what the user said, in local wall-clock terms.\n"
            f"Current local datetime: {local_now:%Y-%m-%d %H:%M} "
            f"({weekday_name}). User timezone: {timezone_name}.\n"
            'local_time: "HH:MM" 24-hour clock as stated. Use "09:00" when '
            "the user gave no time.\n"
            "Set exactly ONE date anchor and leave the others empty "
            '(delay_minutes 0, relative_day "", weekday -1, local_date ""):\n'
            "- delay_minutes > 0 only for pure offsets without a clock time: "
            '"через 10 минут", "через час", "in 2 hours". Then local_time '
            'must be "".\n'
            "- relative_day = today | tomorrow | day_after_tomorrow for "
            '"сегодня", "завтра", "послезавтра".\n'
            "- weekday for a named weekday, Monday=0 .. Sunday=6. A bare "
            '"в субботу" / "on Saturday" means the NEAREST upcoming such '
            "weekday: today only if its time is still ahead.\n"
            '- local_date = "YYYY-MM-DD" for an explicit calendar date '
            '("3 ноября", "on 4 April").\n'
            'If the user gave only a clock time ("напомни в 15:00"), leave '
            "every date anchor empty and keep local_time: the bot uses today, "
            "or tomorrow when that time already passed.\n"
            "Resolve weekday names and calendar dates against the current "
            "local datetime above.\n\n"
            "RECURRENCE\n"
            "recurrence_unit is one of minute, hour, day, week, month, and "
            "null for a one-shot. Weekly rules by weekday use "
            "recurrence_weekdays with Monday=0 .. Sunday=6. Monthly rules keep "
            "the resolved day of month (1..31). For one-shot reminders: "
            "is_recurring=false, recurrence_unit null, recurrence_interval 0, "
            "recurrence_weekdays [].\n\n"
            f"Pending reminders (id: text):\n{catalog}"
        )
