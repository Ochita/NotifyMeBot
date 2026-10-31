from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import make_recurrence

from notifyme_bot.db import ReminderRepository


@pytest.mark.asyncio
async def test_repository_subscription_and_reminders(tmp_path: Path) -> None:
    db_path = tmp_path / "notifyme.sqlite3"
    repo = ReminderRepository(str(db_path))
    await repo.init()

    assert await repo.user_exists(1) is False
    await repo.upsert_user(user_id=1, chat_id=10)
    assert await repo.user_exists(1) is True
    assert await repo.is_subscribed(1) is True

    await repo.set_subscription(user_id=1, is_subscribed=False)
    assert await repo.is_subscribed(1) is False

    await repo.set_user_timezone(user_id=1, timezone="Europe/Rome")
    assert await repo.get_user_timezone(1) == "Europe/Rome"
    await repo.set_user_locale(user_id=1, locale="ru")
    assert await repo.get_user_locale(1) == "ru"

    await repo.set_subscription(user_id=1, is_subscribed=True)
    remind_at = datetime.now(UTC) + timedelta(minutes=10)
    created = await repo.create_reminder(
        user_id=1,
        chat_id=10,
        source_text="remind me soon",
        reminder_text="send payment",
        notification_text="Notification: send payment",
        remind_at_utc=remind_at,
        timezone="Europe/Rome",
    )

    fetched = await repo.get_reminder(created.id)
    assert fetched is not None
    assert fetched.reminder_text == "send payment"
    assert fetched.notification_text == "Notification: send payment"
    assert fetched.recurrence is None
    assert fetched.user_id == 1

    all_pending = await repo.list_pending()
    assert len(all_pending) == 1
    assert all_pending[0].id == created.id

    user_pending = await repo.list_pending(1)
    assert len(user_pending) == 1
    assert [item.id for item in await repo.list_pending(2)] == []

    await repo.delete_reminder(created.id)
    assert await repo.get_reminder(created.id) is None


@pytest.mark.asyncio
async def test_list_pending_filters_past(tmp_path: Path) -> None:
    db_path = tmp_path / "notifyme.sqlite3"
    repo = ReminderRepository(str(db_path))
    await repo.init()
    await repo.upsert_user(user_id=2, chat_id=20)

    past = datetime.now(UTC) - timedelta(minutes=5)
    future = datetime.now(UTC) + timedelta(minutes=5)

    await repo.create_reminder(
        user_id=2,
        chat_id=20,
        source_text="past",
        reminder_text="past reminder",
        notification_text="past reminder",
        remind_at_utc=past,
        timezone="UTC",
    )
    future_row = await repo.create_reminder(
        user_id=2,
        chat_id=20,
        source_text="future",
        reminder_text="future reminder",
        notification_text="future reminder",
        remind_at_utc=future,
        timezone="UTC",
    )

    pending = await repo.list_pending()
    assert [item.id for item in pending] == [future_row.id]


@pytest.mark.asyncio
async def test_recurrence_roundtrips_through_sqlite(tmp_path: Path) -> None:
    repo = ReminderRepository(str(tmp_path / "db.sqlite3"))
    await repo.init()
    await repo.upsert_user(user_id=7, chat_id=70)
    rule = make_recurrence("week", 1, weekdays=[4, 1, 1, 9, -1])

    created = await repo.create_reminder(
        user_id=7,
        chat_id=70,
        source_text="weekly",
        reminder_text="standup",
        notification_text="Reminder: standup",
        remind_at_utc=datetime.now(UTC) + timedelta(days=1),
        timezone="Europe/Rome",
        recurrence=rule,
    )

    fetched = await repo.get_reminder(created.id)

    assert fetched is not None
    # Sorted and cleaned by the model, not stored as the caller typed it.
    assert fetched.recurrence == make_recurrence("week", 1, weekdays=[1, 4])


@pytest.mark.asyncio
async def test_one_shot_stores_no_recurrence(tmp_path: Path) -> None:
    repo = ReminderRepository(str(tmp_path / "db.sqlite3"))
    await repo.init()
    await repo.upsert_user(user_id=8, chat_id=80)

    created = await repo.create_reminder(
        user_id=8,
        chat_id=80,
        source_text="once",
        reminder_text="once",
        notification_text="once",
        remind_at_utc=datetime.now(UTC) + timedelta(minutes=1),
        timezone="UTC",
    )

    assert created.recurrence is None
    assert (await repo.get_reminder(created.id)).recurrence is None  # type: ignore[union-attr]
