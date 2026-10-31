"""Tests for ``LLMReminderParser`` (LLM extracts wording, code does dates).

The LLM is never asked for a UTC timestamp: weekday arithmetic and timezone
conversion are verified here because they used to be silently wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from conftest import (
    FakeOpenAI,
    make_recurrence,
    make_reminder,
    make_service,
)
from pydantic import ValidationError

from notifyme_bot.llm import LLMServiceError
from notifyme_bot.llm_parser import LLMReminderParser
from notifyme_bot.models import CommandExtraction, CommandParseResult

MOSCOW = ZoneInfo("Europe/Moscow")


def _parser(default_timezone: str = "UTC") -> LLMReminderParser:
    return LLMReminderParser(
        provider=make_service(),
        default_timezone=default_timezone,
    )


def _command(**overrides: object) -> CommandExtraction:
    """Model answer with documented defaults; override what a test needs."""
    base: dict[str, object] = {
        "action": "store",
        "delete_all": False,
        "delete_ids": [],
        "reminder_text": "посмотреть в окно",
        "notification_text": "Напоминание: посмотри в окно.",
        "local_date": "",
        "local_time": "12:00",
        "weekday": -1,
        "relative_day": "",
        "delay_minutes": 0,
        "is_recurring": False,
        "recurrence_unit": "none",
        "recurrence_interval": 0,
        "recurrence_weekdays": [],
    }
    base.update(overrides)
    return CommandExtraction(**base)


def _patch_llm(
    monkeypatch: pytest.MonkeyPatch,
    content: CommandExtraction | dict,
    captured: dict | None = None,
) -> FakeOpenAI:
    import json

    payload = (
        content.model_dump()
        if isinstance(content, CommandExtraction)
        else content
    )
    fake = FakeOpenAI(content=json.dumps(payload))
    monkeypatch.setattr("notifyme_bot.llm.AsyncOpenAI", lambda **kwargs: fake)
    if captured is not None:
        captured["fake"] = fake
    return fake


# 2026-10-23 is a Friday: 12:00 UTC == 15:00 Moscow.
FRIDAY = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


async def _parse(
    monkeypatch: pytest.MonkeyPatch,
    content: CommandExtraction,
    now_utc: datetime = FRIDAY,
    timezone_name: str = "Europe/Moscow",
):
    """Drive the public ``parse_command`` path, as the bot does."""
    _patch_llm(monkeypatch, content)
    result = await _parser(timezone_name).parse_command(
        "x", now_utc, timezone_name
    )
    return result.reminder


@pytest.mark.asyncio
async def test_weekday_uses_nearest_upcoming_local_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saturday said on Friday -> tomorrow at 12:00 Moscow, not today."""
    parsed = await _parse(monkeypatch, _command(weekday=5))

    assert parsed is not None
    assert parsed.remind_at_utc == datetime(2026, 10, 24, 9, 0, tzinfo=UTC)
    assert parsed.remind_at_utc.astimezone(MOSCOW).strftime("%A") == "Saturday"


@pytest.mark.asyncio
async def test_monday_after_weekend_is_not_today(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monday said on Friday -> next Monday, not the current week."""
    parsed = await _parse(monkeypatch, _command(weekday=0))

    assert parsed is not None
    local = parsed.remind_at_utc.astimezone(MOSCOW)
    assert local.strftime("%A") == "Monday"
    assert local.date() == datetime(2026, 10, 26).date()


@pytest.mark.asyncio
async def test_weekday_stated_on_sunday_rolls_to_next_week(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported bug: Saturday and Monday must not both fire on Sunday."""
    sunday = datetime(2026, 10, 25, 10, 0, tzinfo=UTC)

    saturday = await _parse(monkeypatch, _command(weekday=5), sunday)
    monday = await _parse(monkeypatch, _command(weekday=0), sunday)

    assert saturday is not None and monday is not None
    assert saturday.remind_at_utc.astimezone(MOSCOW).strftime("%A") == (
        "Saturday"
    )
    assert monday.remind_at_utc.astimezone(MOSCOW).strftime("%A") == "Monday"
    assert saturday.remind_at_utc != monday.remind_at_utc


@pytest.mark.asyncio
async def test_weekday_today_with_future_time_stays_today(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = await _parse(
        monkeypatch,
        _command(weekday=4, local_time="20:00"),
    )

    assert parsed is not None
    assert parsed.remind_at_utc == datetime(2026, 10, 23, 17, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_weekday_today_with_past_time_moves_a_week(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = await _parse(
        monkeypatch,
        _command(weekday=4, local_time="09:00"),
    )

    assert parsed is not None
    assert parsed.remind_at_utc == datetime(2026, 10, 30, 6, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_relative_day_is_local_midnight_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    late_evening = datetime(2026, 10, 23, 20, 0, tzinfo=UTC)
    parsed = await _parse(
        monkeypatch,
        _command(relative_day="tomorrow", local_time="09:00"),
        late_evening,
    )

    assert parsed is not None
    assert parsed.remind_at_utc == datetime(2026, 10, 24, 6, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_relative_day_today_in_the_past_moves_to_tomorrow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = await _parse(
        monkeypatch,
        _command(relative_day="today", local_time="09:00"),
    )

    assert parsed is not None
    local = parsed.remind_at_utc.astimezone(MOSCOW)
    assert local.date() == datetime(2026, 10, 24).date()
    assert local.hour == 9


@pytest.mark.asyncio
async def test_explicit_date_keeps_stated_calendar_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = await _parse(
        monkeypatch,
        _command(local_date="2026-12-31", local_time="23:30"),
    )

    assert parsed is not None
    assert parsed.remind_at_utc == datetime(2026, 12, 31, 20, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_explicit_past_date_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = await _parse(monkeypatch, _command(local_date="2020-01-01"))

    assert parsed is None


@pytest.mark.asyncio
async def test_explicit_date_survives_month_rollover_across_timezones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1st of next month 00:30 local is still the previous UTC day."""
    parsed = await _parse(
        monkeypatch,
        _command(local_date="2026-11-01", local_time="00:30"),
    )

    assert parsed is not None
    assert parsed.remind_at_utc == datetime(2026, 10, 31, 21, 30, tzinfo=UTC)
    assert parsed.remind_at_utc.astimezone(MOSCOW).day == 1


@pytest.mark.asyncio
async def test_delay_minutes_wins_over_clock_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = await _parse(
        monkeypatch,
        _command(delay_minutes=10, local_time=""),
    )

    assert parsed is not None
    assert parsed.remind_at_utc == FRIDAY + timedelta(minutes=10)


@pytest.mark.asyncio
async def test_missing_time_defaults_to_nine_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = await _parse(
        monkeypatch,
        _command(weekday=5, local_time=""),
    )

    assert parsed is not None
    local = parsed.remind_at_utc.astimezone(MOSCOW)
    assert (local.day, local.hour) == (24, 9)


@pytest.mark.asyncio
async def test_bare_clock_time_uses_today_or_tomorrow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'напомни в 15:00' with no date: today if ahead, else tomorrow."""
    future = await _parse(monkeypatch, _command(local_time="20:00"))
    past = await _parse(monkeypatch, _command(local_time="09:00"))

    assert future is not None and past is not None
    assert future.remind_at_utc.astimezone(MOSCOW).day == 23
    assert past.remind_at_utc.astimezone(MOSCOW).day == 24


@pytest.mark.asyncio
async def test_recurring_daily_without_date_uses_stated_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = await _parse(
        monkeypatch,
        _command(
            is_recurring=True,
            recurrence_unit="day",
            recurrence_interval=1,
            local_time="15:00",
        ),
    )

    assert parsed is not None
    local = parsed.remind_at_utc.astimezone(MOSCOW)
    assert (local.hour, local.minute) == (15, 0)
    assert parsed.remind_at_utc > FRIDAY


@pytest.mark.asyncio
async def test_recurring_interval_without_clock_uses_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = await _parse(
        monkeypatch,
        _command(
            is_recurring=True,
            recurrence_unit="hour",
            recurrence_interval=2,
            local_time="",
        ),
    )

    assert parsed is not None
    assert parsed.remind_at_utc == FRIDAY + timedelta(hours=2)


@pytest.mark.asyncio
async def test_recurring_month_without_date_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A monthly rule with no day would silently drift; refuse it."""
    parsed = await _parse(
        monkeypatch,
        _command(
            is_recurring=True,
            recurrence_unit="month",
            recurrence_interval=1,
            local_time="09:00",
        ),
    )

    assert parsed is None


@pytest.mark.asyncio
async def test_weekly_recurrence_uses_weekday_list_and_nearest_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = await _parse(
        monkeypatch,
        _command(
            is_recurring=True,
            recurrence_unit="week",
            recurrence_interval=1,
            recurrence_weekdays=[2, 5],
            local_time="12:00",
        ),
    )

    assert parsed is not None
    local = parsed.remind_at_utc.astimezone(MOSCOW)
    assert local.strftime("%A") == "Saturday"
    assert parsed.recurrence is not None
    assert parsed.recurrence.weekdays == [2, 5]


@pytest.mark.asyncio
async def test_notification_text_falls_back_to_reminder_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = await _parse(
        monkeypatch,
        _command(weekday=5, notification_text="  "),
    )

    assert parsed is not None
    assert parsed.notification_text == "посмотреть в окно"


@pytest.mark.asyncio
async def test_parse_command_single_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    fake = _patch_llm(monkeypatch, _command(weekday=5), captured)
    parser = LLMReminderParser(
        provider=make_service(fake), default_timezone="Europe/Moscow"
    )
    out = await parser.parse_command("напомни в субботу в 12", FRIDAY)

    assert isinstance(out, CommandParseResult)
    assert out.delete is None
    assert out.reminder is not None
    assert len(fake.captured_chat) > 0
    # The pydantic model is handed to the SDK as the schema.
    assert fake.captured_chat["response_format"] is CommandExtraction
    assert fake.captured_chat["extra_body"]["provider"]["sort"] == "price"


@pytest.mark.asyncio
async def test_parse_reminder_prompt_uses_command_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _patch_llm(monkeypatch, _command(weekday=5))
    parser = LLMReminderParser(
        provider=make_service(fake), default_timezone="Europe/Moscow"
    )
    await parser.parse_command("напомни в субботу", FRIDAY)

    messages = fake.captured_chat["messages"]
    system_prompt = messages[0]["content"]
    assert "Never convert to UTC" in system_prompt
    assert "Pending reminders" in system_prompt
    assert "Europe/Moscow" in system_prompt
    assert "Friday" in system_prompt
    # Schema drift is now impossible: the prompt keys come from the model.
    for key in ("local_date", "local_time", "weekday", "relative_day"):
        assert key in CommandExtraction.model_fields


@pytest.mark.asyncio
async def test_parse_reminder_llm_service_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(**kwargs: object) -> object:
        raise RuntimeError("down")

    fake = FakeOpenAI()
    fake.chat.completions.parse = boom  # type: ignore[method-assign]
    parser = LLMReminderParser(
        provider=make_service(fake), default_timezone="UTC"
    )

    with pytest.raises(LLMServiceError, match="down"):
        await parser.parse_command("Please remind me", FRIDAY)


@pytest.mark.asyncio
async def test_parse_delete_request_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _command(
        action="delete",
        delete_ids=[10],
    )
    _patch_llm(monkeypatch, payload)
    parsed = (
        await _parser().parse_command("Delete reminder about Alexei", FRIDAY)
    ).delete

    assert parsed is not None
    assert parsed.delete_all is False
    assert parsed.delete_ids == [10]


@pytest.mark.asyncio
async def test_delete_ids_are_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model listing the same id twice must not delete it twice."""
    _patch_llm(monkeypatch, _command(action="delete", delete_ids=[10, 11, 10]))
    result = await _parser().parse_command("delete these", FRIDAY)

    assert result.delete is not None
    assert result.delete.delete_ids == [10, 11]


@pytest.mark.asyncio
async def test_delete_wins_when_message_asks_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _command(
        action="delete",
        delete_all=True,
        delete_ids=[1],
    )
    _patch_llm(monkeypatch, payload)
    result = await _parser().parse_command("удали и напомни", FRIDAY)

    assert result.reminder is None
    assert result.delete is not None
    assert result.delete.delete_all is True
    assert result.delete.delete_ids == []


@pytest.mark.asyncio
async def test_parse_reminder_not_schedulable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_llm(
        monkeypatch,
        _command(
            action="none",
            reminder_text="",
            notification_text="",
        ),
    )
    parsed = (await _parser().parse_command("hello", FRIDAY)).reminder

    assert parsed is None


def test_parser_helpers() -> None:
    assert LLMReminderParser._weekday_anchor(7) == (True, None)
    assert LLMReminderParser._weekday_anchor("three") == (True, None)
    assert LLMReminderParser._weekday_anchor("3") == (True, 3)
    assert LLMReminderParser._weekday_anchor(-1) == (False, None)
    assert LLMReminderParser._weekday_anchor("") == (False, None)
    assert LLMReminderParser._parse_local_time("25:00") is None
    assert LLMReminderParser._parse_local_time("9:00") is not None
    assert LLMReminderParser._parse_local_date("31.12.2026") is None


@pytest.mark.asyncio
async def test_unreadable_clock_time_is_not_replaced_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stated but unparseable clock must fail, not silently become 09:00."""
    parsed = await _parse(
        monkeypatch,
        _command(weekday=5, local_time="вечером"),
    )

    assert parsed is None


@pytest.mark.asyncio
async def test_parse_reminder_returns_none_when_missing_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_llm(monkeypatch, _command(reminder_text="", weekday=5))
    assert (await _parser().parse_command("x", FRIDAY)).reminder is None


@pytest.mark.asyncio
async def test_parse_reminder_returns_none_when_no_schedulable_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schedule intent with only a broken date has no anchor."""
    _patch_llm(
        monkeypatch, _command(local_date="not-a-date", local_time="bad")
    )
    assert (await _parser().parse_command("x", FRIDAY)).reminder is None


@pytest.mark.asyncio
async def test_recurring_without_unit_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_llm(
        monkeypatch,
        _command(
            is_recurring=True,
            recurrence_unit="none",
            recurrence_interval=1,
            weekday=5,
        ),
    )
    assert (await _parser().parse_command("x", FRIDAY)).reminder is None


def test_recurrence_unit_is_a_nullable_enum() -> None:
    """The schema offers the five units plus null, nothing else."""

    def unit(raw: str) -> object:
        payload = '{"recurrence_unit": %s}' % raw
        return CommandExtraction.model_validate_json(payload).recurrence_unit

    for raw in ('"none"', '""', "null"):
        assert unit(raw) is None
    assert unit('"WEEK"') == "week"
    assert "recurrence_unit" in CommandExtraction.model_fields
    assert CommandExtraction().recurrence_unit is None
    with pytest.raises(ValidationError):
        unit('"fortnight"')


@pytest.mark.asyncio
async def test_recurring_defaults_interval_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_llm(
        monkeypatch,
        _command(
            is_recurring=True,
            recurrence_unit="week",
            recurrence_interval=0,
            recurrence_weekdays=[2, 5],
        ),
    )
    parsed = (await _parser().parse_command("x", FRIDAY)).reminder

    assert parsed is not None
    assert parsed.recurrence == make_recurrence("week", 1, weekdays=[2, 5])


@pytest.mark.asyncio
async def test_resolve_rejects_unknown_weekday_only_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = await _parse(
        monkeypatch, _command(weekday=9), timezone_name="UTC"
    )

    assert parsed is None


@pytest.mark.asyncio
async def test_resolve_monthly_first_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = await _parse(
        monkeypatch,
        _command(
            is_recurring=True,
            recurrence_unit="month",
            recurrence_interval=1,
            local_date="2026-11-03",
            local_time="09:00",
        ),
        timezone_name="UTC",
    )

    assert parsed is not None
    # The monthly anchor is captured from the first fire, so later months
    # cannot drift off the 3rd just because February is short.
    assert parsed.recurrence == make_recurrence("month", 1, day_of_month=3)
    assert parsed.remind_at_utc == datetime(2026, 11, 3, 9, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_weekly_recurrence_without_a_day_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A weekly rule with no weekday and no clock would drift; refuse it."""
    parsed = await _parse(
        monkeypatch,
        _command(
            is_recurring=True,
            recurrence_unit="week",
            recurrence_interval=1,
            local_time="",
        ),
    )

    assert parsed is None


@pytest.mark.asyncio
async def test_prompt_lists_pending_reminders_for_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delete-by-description needs the pending ids in the prompt."""
    fake = _patch_llm(monkeypatch, _command(weekday=5))
    parser = LLMReminderParser(
        provider=make_service(fake), default_timezone="UTC"
    )
    pending = [make_reminder(id=42, reminder_text="pay Alexei")]

    await parser.parse_command(
        "delete the Alexei one", FRIDAY, reminders=pending
    )

    system_prompt = fake.captured_chat["messages"][0]["content"]
    assert "42: pay Alexei" in system_prompt


@pytest.mark.asyncio
async def test_weekly_recurrence_ignores_weekdays_on_other_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'every 2 days on Monday' has no defined meaning: the day is dropped."""
    parsed = await _parse(
        monkeypatch,
        _command(
            is_recurring=True,
            recurrence_unit="day",
            recurrence_interval=2,
            recurrence_weekdays=[0],
            weekday=0,
            local_time="09:00",
        ),
    )

    assert parsed is not None
    assert parsed.recurrence == make_recurrence("day", 2)


@pytest.mark.asyncio
async def test_chat_completion_api_error_is_wrapped() -> None:
    async def boom(**kwargs: object) -> object:
        raise ValueError("upstream 401")

    fake = FakeOpenAI()
    fake.chat.completions.parse = boom  # type: ignore[method-assign]
    parser = LLMReminderParser(
        provider=make_service(fake), default_timezone="UTC"
    )

    with pytest.raises(LLMServiceError, match="gpt-4.1-nano") as exc:
        await parser.parse_command("hi", FRIDAY)
    assert exc.value.__cause__ is not None
    assert "401" in str(exc.value)


@pytest.mark.asyncio
async def test_chat_completion_bad_body_raises() -> None:
    fake = FakeOpenAI(content="not json")
    parser = LLMReminderParser(
        provider=make_service(fake), default_timezone="UTC"
    )

    with pytest.raises(LLMServiceError):
        await parser.parse_command("hi", FRIDAY)


@pytest.mark.asyncio
async def test_delete_all_not_inferred_from_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model delete_all=false wins over user wording like 'delete all'."""
    _patch_llm(
        monkeypatch,
        _command(
            action="delete",
            delete_all=False,
            delete_ids=[],
        ),
    )
    req = (
        await _parser().parse_command("please delete all reminders", FRIDAY)
    ).delete

    assert req is not None
    assert req.delete_all is False
