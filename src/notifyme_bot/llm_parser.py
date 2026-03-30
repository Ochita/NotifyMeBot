from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from notifyme_bot.llm.errors import LLMServiceError
from notifyme_bot.llm.json_utils import extract_json_object, loads_json_object
from notifyme_bot.llm.protocol import StructuredChatProvider
from notifyme_bot.models import (
    CommandParseResult,
    ParsedDeleteRequest,
    ParsedReminder,
)

# Backward-compatible name for callers and tests.
GeminiServiceError = LLMServiceError


class LLMReminderParser:
    """Reminder/delete parsing via ``StructuredChatProvider``."""

    def __init__(
        self,
        provider: StructuredChatProvider,
        default_timezone: str,
    ) -> None:
        self._provider = provider
        self._default_timezone = default_timezone

    _loads_json_object = staticmethod(loads_json_object)
    _extract_json_object = staticmethod(extract_json_object)

    async def parse_command(
        self,
        message_text: str,
        now_utc: datetime,
        user_timezone: str | None = None,
    ) -> CommandParseResult:
        """One LLM call: classify and extract delete and/or schedule."""
        now_utc = now_utc.astimezone(UTC)
        timezone_name = user_timezone or self._default_timezone
        local_now = now_utc.astimezone(ZoneInfo(timezone_name))
        data = await self._extract_command(
            message_text=message_text,
            local_now=local_now,
            timezone_name=timezone_name,
        )
        delete = self._build_parsed_delete_from_data(data)
        reminder = self._build_parsed_reminder_from_data(
            data, now_utc, timezone_name
        )
        if delete is not None and reminder is not None:
            return CommandParseResult(delete=delete, reminder=None)
        if delete is not None:
            return CommandParseResult(delete=delete, reminder=None)
        if reminder is not None:
            return CommandParseResult(delete=None, reminder=reminder)
        return CommandParseResult(delete=None, reminder=None)

    async def parse_reminder(
        self,
        message_text: str,
        now_utc: datetime,
        user_timezone: str | None = None,
    ) -> ParsedReminder | None:
        result = await self.parse_command(
            message_text, now_utc, user_timezone
        )
        return result.reminder

    async def parse_delete_request(
        self,
        message_text: str,
    ) -> ParsedDeleteRequest | None:
        result = await self.parse_command(
            message_text, datetime.now(UTC), None
        )
        return result.delete

    def _build_parsed_reminder_from_data(
        self,
        data: dict,
        now_utc: datetime,
        timezone_name: str,
    ) -> ParsedReminder | None:
        if not self._is_schedule_intent(data):
            return None

        reminder_text = str(data.get("reminder_text", "")).strip()
        notification_text = str(data.get("notification_text", "")).strip()
        remind_at_raw = str(data.get("remind_at_utc", "")).strip()
        if not reminder_text or not notification_text:
            return None

        is_recurring = bool(data.get("is_recurring"))
        recurrence_unit_raw = (
            str(data.get("recurrence_unit", "")).strip().lower()
        )
        recurrence_unit = (
            recurrence_unit_raw
            if recurrence_unit_raw in {"minute", "hour", "day", "week"}
            else None
        )
        recurrence_interval = self._positive_int_or_none(
            data.get("recurrence_interval")
        )
        recurrence_weekdays = self._normalize_weekdays(
            data.get("recurrence_weekdays")
        )

        if is_recurring:
            if recurrence_unit is None or recurrence_interval is None:
                return None
        else:
            recurrence_unit = None
            recurrence_interval = None
            recurrence_weekdays = None

        remind_at = self._first_fire_from_llm(
            remind_at_raw=remind_at_raw,
            timezone_name=timezone_name,
            now_utc=now_utc,
        )
        if remind_at is None:
            return None

        return ParsedReminder(
            reminder_text=reminder_text,
            notification_text=notification_text,
            remind_at_utc=remind_at,
            is_recurring=is_recurring,
            recurrence_unit=recurrence_unit,
            recurrence_interval=recurrence_interval,
            recurrence_weekdays=recurrence_weekdays,
        )

    def _build_parsed_delete_from_data(
        self,
        data: dict,
    ) -> ParsedDeleteRequest | None:
        if not self._is_delete_intent(data):
            return None

        delete_all = bool(data.get("delete_all"))
        target_text = str(data.get("target_text", "")).strip() or None
        if delete_all:
            target_text = None
        return ParsedDeleteRequest(
            delete_all=delete_all, target_text=target_text
        )

    @staticmethod
    def _command_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "should_schedule": {"type": "boolean"},
                "should_delete": {"type": "boolean"},
                "delete_all": {"type": "boolean"},
                "target_text": {"type": "string"},
                "reminder_text": {"type": "string"},
                "notification_text": {"type": "string"},
                "remind_at_utc": {"type": "string"},
                "is_recurring": {"type": "boolean"},
                "recurrence_unit": {"type": "string"},
                "recurrence_interval": {"type": "integer"},
                "recurrence_weekdays": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": [
                "action",
                "should_schedule",
                "should_delete",
                "delete_all",
                "target_text",
                "reminder_text",
                "notification_text",
                "remind_at_utc",
                "is_recurring",
                "recurrence_unit",
                "recurrence_interval",
                "recurrence_weekdays",
            ],
            "additionalProperties": False,
        }

    async def _extract_command(
        self,
        message_text: str,
        local_now: datetime,
        timezone_name: str,
    ) -> dict:
        system_prompt = self._command_prompt(
            local_now=local_now,
            timezone_name=timezone_name,
        )
        return await self._provider.chat_json(
            system_prompt=system_prompt,
            user_prompt=message_text,
            response_schema=self._command_schema(),
        )

    @staticmethod
    def _parse_remind_at_to_utc(value: str, timezone_name: str) -> datetime:
        """Parse LLM remind_at_utc string to UTC.

        Naive ISO wall time is treated as user-local (``timezone_name``), not
        UTC, so local clock values without ``Z`` schedule at the right instant.

        Values with ``Z`` or an explicit offset use that instant.
        """
        text = str(value).strip().replace("Z", "+00:00")
        parsed_dt = datetime.fromisoformat(text)
        if parsed_dt.tzinfo is None:
            local = parsed_dt.replace(tzinfo=ZoneInfo(timezone_name))
            return local.astimezone(UTC)
        return parsed_dt.astimezone(UTC)

    @staticmethod
    def _positive_int_or_none(value: object) -> int | None:
        try:
            int_value = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return int_value if int_value > 0 else None

    @staticmethod
    def _normalize_weekdays(value: object) -> list[int] | None:
        if not isinstance(value, list):
            return None
        result: list[int] = []
        for item in value:
            try:
                weekday = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= weekday <= 6 and weekday not in result:
                result.append(weekday)
        return result or None

    @staticmethod
    def _command_prompt(
        local_now: datetime,
        timezone_name: str,
    ) -> str:
        return (
            "You extract command intent from user messages for reminder bot. "
            "Return strict JSON object with keys: "
            "action, should_schedule, should_delete, delete_all, target_text, "
            "reminder_text, notification_text, remind_at_utc, is_recurring, "
            "recurrence_unit, recurrence_interval, recurrence_weekdays. "
            "Allowed action values: store, delete, none. "
            "Do not include examples in your response. "
            "Do not invent entities or tasks that are absent in user text. "
            "When message requests storing or creating a reminder, set "
            "action=store and should_schedule=true. "
            "When message requests deleting/canceling reminders, set "
            "action=delete and should_delete=true. "
            "When no relevant command, use action=none and both booleans "
            "false. delete_all must be true only if user clearly asks to "
            "remove all. target_text should be concise hint text for matching "
            "one reminder. "
            "If the message could be read as both delete and schedule, "
            "prefer delete (action=delete, should_delete=true). "
            "reminder_text should capture the user task verbatim and concise. "
            "notification_text must be ready-to-send: start with a short label meaning "
            "\"notification\" or \"reminder\" in the SAME language and script as the user's "
            "message (e.g. English: Notification:, Russian: Напоминание: or Уведомление:, "
            "Ukrainian: Нагадування:, Spanish: Notificación:, German: Erinnerung: or "
            "Benachrichtigung:, Chinese: 通知：). Do not use the English word Notification "
            "as the label when the user wrote in another language. "
            "After the label and separator (colon or fullwidth colon), one space unless the "
            "language normally omits it (e.g. Chinese 通知：). "
            "Then one natural-language request in the imperative mood in the same language "
            "as the user message, telling the user what to do. "
            "Do not repeat reminder_text; same meaning as reminder_text. "
            "If date is specified without time, assume 09:00 local time. "
            "recurrence_unit must be one of minute, hour, day, week, none. "
            "For non-recurring reminders: is_recurring=false, "
            "recurrence_unit=none, recurrence_interval=0, "
            "recurrence_weekdays=[]. "
            "Field remind_at_utc must be ISO 8601. Use Z or explicit offset "
            "for true UTC. If you omit timezone, wall time is interpreted as "
            "user local time (see User timezone below). "
            "Always output a valid remind_at_utc strictly after current local time: "
            "first fire for one-shot reminders, and for recurring rules the first "
            "occurrence (you compute it from the user's wording). "
            "For weekly recurrence by weekdays, use Monday=0..Sunday=6. "
            f"Current local time: {local_now.isoformat()}. "
            f"User timezone: {timezone_name}."
        )

    @staticmethod
    def _first_fire_from_llm(
        remind_at_raw: str,
        timezone_name: str,
        now_utc: datetime,
    ) -> datetime | None:
        """Parse LLM time only; must be strictly after ``now_utc``."""
        text = remind_at_raw.strip()
        if not text:
            return None
        try:
            dt = LLMReminderParser._parse_remind_at_to_utc(text, timezone_name)
        except ValueError:
            return None
        if dt <= now_utc:
            return None
        return dt

    @staticmethod
    def _normalize_action(value: object) -> str:
        action = str(value).strip().lower()
        return action if action in {"store", "delete", "none"} else "none"

    @classmethod
    def _is_schedule_intent(cls, data: dict) -> bool:
        action = cls._normalize_action(data.get("action"))
        return action == "store" or bool(data.get("should_schedule"))

    @classmethod
    def _is_delete_intent(cls, data: dict) -> bool:
        action = cls._normalize_action(data.get("action"))
        return action == "delete" or bool(data.get("should_delete"))
