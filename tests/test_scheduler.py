from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import FakeBot, make_recurrence, make_reminder

from notifyme_bot.db import ReminderRepository
from notifyme_bot.scheduler_service import ReminderScheduler


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
    )
    bot = FakeBot()
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
    bot = FakeBot()
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
        recurrence=make_recurrence("day", 2),
    )
    bot = FakeBot()
    scheduler = ReminderScheduler(repository=repo, bot=bot)
    scheduler._scheduler.start()

    await scheduler._deliver_and_purge(reminder.id)

    updated = await repo.get_reminder(reminder.id)
    assert updated is not None
    assert updated.recurrence == reminder.recurrence
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
    )
    scheduler = ReminderScheduler(repository=repo, bot=FakeBot())

    await scheduler.start()

    assert scheduler._scheduler.get_job(f"reminder:{reminder.id}") is not None
    await scheduler.shutdown()


def test_recurring_reminder_moves_to_next_run() -> None:
    """Every 2 minutes from a fixed instant lands exactly 2 minutes later."""
    base_dt = datetime(2026, 10, 23, 12, 1, tzinfo=UTC)
    now = datetime(2026, 10, 23, 12, 2, tzinfo=UTC)
    reminder = make_reminder(
        remind_at_utc=base_dt,
        recurrence=make_recurrence("minute", 2),
    )

    assert reminder.next_run(now) == datetime(2026, 10, 23, 12, 3, tzinfo=UTC)


def test_recurring_reminder_catches_up_after_downtime() -> None:
    """Delivered late, the next fire stays in the future, not in the past."""
    base_dt = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    now = datetime(2026, 10, 23, 12, 9, tzinfo=UTC)
    reminder = make_reminder(
        remind_at_utc=base_dt,
        recurrence=make_recurrence("minute", 2),
    )

    assert reminder.next_run(now) == datetime(2026, 10, 23, 12, 10, tzinfo=UTC)


def test_one_shot_reminder_has_no_next_run() -> None:
    assert make_reminder().next_run(datetime.now(UTC)) is None
