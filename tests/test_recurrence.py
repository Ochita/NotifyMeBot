"""The recurrence rule owns the date math, so it owns the edge cases.

Every case is expressed in UTC with a fixed ``now``: no sleeping, no reliance
on the machine clock, no DST surprises in the assertions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from notifyme_bot.models import Recurrence


def _next(
    rule: Recurrence,
    previous: datetime,
    *,
    now: datetime | None = None,
    timezone: str = "UTC",
) -> datetime:
    return rule.next_after(
        previous=previous, now=now or previous, timezone=timezone
    )


def test_fixed_units_advance_by_the_interval() -> None:
    start = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)

    assert _next(Recurrence(unit="minute"), start) == _next_minutes(start, 1)
    two_days = Recurrence(unit="day", interval=2)
    assert _next(two_days, start) == datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


def _next_minutes(start: datetime, minutes: int) -> datetime:
    from datetime import timedelta

    return start + timedelta(minutes=minutes)


def test_step_skips_a_long_outage() -> None:
    """After downtime the next fire is future, not the first missed one."""
    start = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    now = datetime(2026, 10, 23, 12, 0, tzinfo=UTC) + _hours(5)

    following = _next(Recurrence(unit="hour", interval=2), start, now=now)

    assert following == datetime(2026, 10, 23, 18, 0, tzinfo=UTC)


def _hours(n: int):
    from datetime import timedelta

    return timedelta(hours=n)


def test_weekly_pattern_picks_the_nearest_listed_day() -> None:
    # 2026-10-23 is a Friday; Mon/Wed is next Monday, never "yesterday".
    start = datetime(2026, 10, 23, 9, 0, tzinfo=UTC)

    following = _next(Recurrence(unit="week", weekdays=[0, 2]), start)

    assert following == datetime(2026, 10, 26, 9, 0, tzinfo=UTC)
    assert following.weekday() == 0


def test_weekly_pattern_keeps_the_wall_clock_across_dst() -> None:
    """08:00 Rome stays 08:00 Rome when the offset changes (CEST -> CET)."""
    rule = Recurrence(unit="week", weekdays=[0])  # Mondays
    start = datetime(2026, 10, 19, 6, 0, tzinfo=UTC)  # 08:00 CEST

    following = _next(rule, start, timezone="Europe/Rome")

    assert following == datetime(2026, 10, 26, 7, 0, tzinfo=UTC)  # 08:00 CET
    assert following.astimezone(ZoneInfo("Europe/Rome")).hour == 8


def test_weekly_pattern_ignores_the_interval() -> None:
    """One week is the cycle by definition; the interval cannot stretch it."""
    start = datetime(2026, 10, 23, 9, 0, tzinfo=UTC)
    rule = Recurrence(unit="week", interval=4, weekdays=[5])

    assert _next(rule, start) == datetime(2026, 10, 24, 9, 0, tzinfo=UTC)


def test_weekly_without_weekdays_is_a_plain_step() -> None:
    start = datetime(2026, 10, 23, 9, 0, tzinfo=UTC)

    following = _next(Recurrence(unit="week", interval=2), start)

    assert following == datetime(2026, 11, 6, 9, 0, tzinfo=UTC)


def test_monthly_keeps_the_day_and_clamps_short_months() -> None:
    """Jan 31 -> Feb 28 -> Mar 31, not a permanent drift to the 28th."""
    rule = Recurrence(unit="month", day_of_month=31)
    january = datetime(2027, 1, 31, 10, 0, tzinfo=UTC)

    february = _next(rule, january)
    march = _next(rule, february, now=february)

    assert february == datetime(2027, 2, 28, 10, 0, tzinfo=UTC)
    assert march == datetime(2027, 3, 31, 10, 0, tzinfo=UTC)


def test_monthly_rolls_over_the_year() -> None:
    start = datetime(2026, 11, 15, 10, 0, tzinfo=UTC)

    following = _next(Recurrence(unit="month", interval=3), start)

    assert following == datetime(2027, 2, 15, 10, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("unit", "interval"),
    [("minute", 1), ("hour", 1), ("day", 1), ("week", 1), ("month", 1)],
)
def test_every_unit_always_moves_forward(unit: str, interval: int) -> None:
    """No rule may return the instant it was given, or jobs loop forever."""
    start = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)

    following = _next(Recurrence(unit=unit, interval=interval), start)  # type: ignore[arg-type]

    assert following > start


def test_weekdays_are_sorted_deduplicated_and_bounded() -> None:
    rule = Recurrence(unit="week", weekdays=[4, 1, 1, 9, -1])

    assert rule.weekdays == [1, 4]


def test_interval_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Recurrence(unit="day", interval=0)


def test_unknown_unit_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Recurrence(unit="fortnight")  # type: ignore[arg-type]
