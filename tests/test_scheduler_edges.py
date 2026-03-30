from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from notifyme_bot.models import Reminder
from notifyme_bot.scheduler_service import ReminderScheduler


class _FakeBot:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.messages.append({"chat_id": chat_id, "text": text})


class _RepoNoReminder:
    async def get_reminder(self, reminder_id: int):
        return None

    async def delete_reminder(self, reminder_id: int) -> None:
        raise AssertionError("should not delete")

    async def update_reminder_next_run(self, **kwargs) -> None:
        raise AssertionError("should not update")


@pytest.mark.asyncio
async def test_deliver_purge_when_reminder_missing() -> None:
    bot = _FakeBot()
    scheduler = ReminderScheduler(repository=_RepoNoReminder(), bot=bot)
    await scheduler._deliver_and_purge(999)
    assert bot.messages == []


@pytest.mark.asyncio
async def test_deliver_uses_reminder_text_when_notification_blank(
    tmp_path: Path,
) -> None:
    from notifyme_bot.db import ReminderRepository

    repo = ReminderRepository(str(tmp_path / "db.sqlite3"))
    await repo.init()
    await repo.upsert_user(user_id=1, chat_id=10)
    reminder = await repo.create_reminder(
        user_id=1,
        chat_id=10,
        source_text="s",
        reminder_text="body only",
        notification_text="   ",
        remind_at_utc=datetime.now(UTC) + timedelta(seconds=30),
        timezone="UTC",
        is_recurring=False,
        recurrence_unit=None,
        recurrence_interval=None,
        recurrence_weekdays=None,
    )
    bot = _FakeBot()
    scheduler = ReminderScheduler(repository=repo, bot=bot)
    await scheduler._deliver_and_purge(reminder.id)
    assert bot.messages[0]["text"] == "body only"


@pytest.mark.asyncio
async def test_scheduler_start_idempotent(tmp_path: Path) -> None:
    from notifyme_bot.db import ReminderRepository

    repo = ReminderRepository(str(tmp_path / "db.sqlite3"))
    await repo.init()
    scheduler = ReminderScheduler(repository=repo, bot=_FakeBot())
    await scheduler.start()
    await scheduler.start()
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_scheduler_shutdown_when_not_running() -> None:
    scheduler = ReminderScheduler(
        repository=AsyncMock(),
        bot=_FakeBot(),
    )
    await scheduler.shutdown()


def test_next_recurring_run_fallback_unknown_unit() -> None:
    base = datetime.now(UTC) + timedelta(minutes=1)
    r = Reminder(
        id=1,
        user_id=1,
        chat_id=1,
        source_text="s",
        reminder_text="t",
        notification_text="n",
        remind_at_utc=base,
        timezone="UTC",
        is_recurring=True,
        recurrence_unit="not-a-known-unit",  # type: ignore[arg-type]
        recurrence_interval=1,
        recurrence_weekdays=None,
        created_at_utc=base,
    )
    nxt = ReminderScheduler._next_recurring_run(r)
    assert nxt > base
