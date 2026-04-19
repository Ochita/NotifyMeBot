from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from notifyme_bot.db import ReminderRepository
from notifyme_bot.models import Reminder
from notifyme_bot.scheduler_service import ReminderScheduler


class _FakeBot:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.messages.append({"chat_id": chat_id, "text": text})


@pytest.mark.asyncio
async def test_scheduler_delivers_and_purges(tmp_path: Path) -> None:
    repo = ReminderRepository(str(tmp_path / "notifyme.sqlite3"))
    await repo.init()
    await repo.upsert_user(user_id=3, chat_id=30)

    reminder = await repo.create_reminder(
        user_id=3,
        chat_id=30,
        source_text="future reminder",
        reminder_text="pay Alexei",
        notification_text="Уведомление: отправьте деньги Алексею",
        remind_at_utc=datetime.now(UTC) + timedelta(minutes=2),
        timezone="UTC",
        is_recurring=False,
        recurrence_unit=None,
        recurrence_interval=None,
        recurrence_weekdays=None,
    )
    bot = _FakeBot()
    scheduler = ReminderScheduler(repository=repo, bot=bot)

    await scheduler._deliver_and_purge(reminder.id)

    assert len(bot.messages) == 1
    assert bot.messages[0]["chat_id"] == 30
    assert "Уведомление" in bot.messages[0]["text"]
    assert await repo.get_reminder(reminder.id) is None


@pytest.mark.asyncio
async def test_schedule_ignores_past_reminder(tmp_path: Path) -> None:
    repo = ReminderRepository(str(tmp_path / "notifyme.sqlite3"))
    await repo.init()
    bot = _FakeBot()
    scheduler = ReminderScheduler(repository=repo, bot=bot)
    scheduler._scheduler.start()

    reminder = await repo.create_reminder(
        user_id=4,
        chat_id=40,
        source_text="past",
        reminder_text="already late",
        notification_text="already late",
        remind_at_utc=datetime.now(UTC) - timedelta(minutes=2),
        timezone="UTC",
        is_recurring=False,
        recurrence_unit=None,
        recurrence_interval=None,
        recurrence_weekdays=None,
    )
    scheduler.schedule(reminder)

    assert scheduler._scheduler.get_job(f"reminder:{reminder.id}") is None
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_scheduler_recurring_reschedules(tmp_path: Path) -> None:
    repo = ReminderRepository(str(tmp_path / "notifyme.sqlite3"))
    await repo.init()
    await repo.upsert_user(user_id=5, chat_id=50)
    reminder = await repo.create_reminder(
        user_id=5,
        chat_id=50,
        source_text="every 2 days",
        reminder_text="drink water",
        notification_text="Reminder: drink water.",
        remind_at_utc=datetime.now(UTC) + timedelta(minutes=1),
        timezone="UTC",
        is_recurring=True,
        recurrence_unit="day",
        recurrence_interval=2,
        recurrence_weekdays=None,
    )
    bot = _FakeBot()
    scheduler = ReminderScheduler(repository=repo, bot=bot)
    scheduler._scheduler.start()

    await scheduler._deliver_and_purge(reminder.id)

    updated = await repo.get_reminder(reminder.id)
    assert updated is not None
    assert updated.remind_at_utc > reminder.remind_at_utc
    assert bot.messages[0]["chat_id"] == 50
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_scheduler_start_loads_pending_jobs(tmp_path: Path) -> None:
    repo = ReminderRepository(str(tmp_path / "notifyme.sqlite3"))
    await repo.init()
    await repo.upsert_user(user_id=6, chat_id=60)
    reminder = await repo.create_reminder(
        user_id=6,
        chat_id=60,
        source_text="future reminder",
        reminder_text="stand up",
        notification_text="Reminder: stand up.",
        remind_at_utc=datetime.now(UTC) + timedelta(minutes=3),
        timezone="UTC",
        is_recurring=False,
        recurrence_unit=None,
        recurrence_interval=None,
        recurrence_weekdays=None,
    )
    scheduler = ReminderScheduler(repository=repo, bot=_FakeBot())

    await scheduler.start()

    assert scheduler._scheduler.get_job(f"reminder:{reminder.id}") is not None
    await scheduler.shutdown()


def test_next_recurring_run_weekdays_and_fallback() -> None:
    base_dt = datetime.now(UTC) + timedelta(minutes=1)
    weekly = Reminder(
        id=1,
        user_id=1,
        chat_id=1,
        source_text="every wednesday and saturday",
        reminder_text="task",
        notification_text="task",
        remind_at_utc=base_dt,
        timezone="Europe/Rome",
        is_recurring=True,
        recurrence_unit="week",
        recurrence_interval=1,
        recurrence_weekdays=[2, 5],
        created_at_utc=base_dt,
    )
    fallback = Reminder(
        id=2,
        user_id=1,
        chat_id=1,
        source_text="broken recurrence",
        reminder_text="task",
        notification_text="task",
        remind_at_utc=base_dt,
        timezone="UTC",
        is_recurring=True,
        recurrence_unit="unknown",
        recurrence_interval=1,
        recurrence_weekdays=None,
        created_at_utc=base_dt,
    )

    weekly_next = ReminderScheduler._next_recurring_run(weekly)
    fallback_next = ReminderScheduler._next_recurring_run(fallback)

    assert weekly_next > base_dt
    assert weekly_next.weekday() in {2, 5}
    assert fallback_next.date() == (base_dt + timedelta(days=1)).date()


def test_next_recurring_run_minutes() -> None:
    base_dt = datetime.now(UTC) + timedelta(seconds=20)
    minute_reminder = Reminder(
        id=3,
        user_id=1,
        chat_id=1,
        source_text="every 2 minutes",
        reminder_text="task",
        notification_text="task",
        remind_at_utc=base_dt,
        timezone="UTC",
        is_recurring=True,
        recurrence_unit="minute",
        recurrence_interval=2,
        recurrence_weekdays=None,
        created_at_utc=base_dt,
    )

    next_dt = ReminderScheduler._next_recurring_run(minute_reminder)
    delta = next_dt - base_dt

    assert next_dt > base_dt
    assert timedelta(minutes=1, seconds=50) <= delta <= timedelta(
        minutes=2, seconds=10
    )


def test_next_recurring_run_monthly() -> None:
    base_dt = datetime.now(UTC).replace(day=3, hour=9, minute=0, second=0)
    if base_dt <= datetime.now(UTC):
        # Keep the same day/time, just move to future month baseline.
        month = base_dt.month + 1
        year = base_dt.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        base_dt = base_dt.replace(year=year, month=month)

    month_reminder = Reminder(
        id=4,
        user_id=1,
        chat_id=1,
        source_text="every month on day 3",
        reminder_text="pay rent",
        notification_text="Reminder: pay rent.",
        remind_at_utc=base_dt,
        timezone="UTC",
        is_recurring=True,
        recurrence_unit="month",
        recurrence_interval=1,
        recurrence_weekdays=None,
        created_at_utc=base_dt,
    )

    next_dt = ReminderScheduler._next_recurring_run(month_reminder)

    assert next_dt > base_dt
    assert next_dt.day == 3
