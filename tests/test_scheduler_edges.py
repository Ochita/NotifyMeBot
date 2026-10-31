from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from conftest import FakeBot, make_recurrence
from pydantic import ValidationError

from notifyme_bot.db import ReminderRepository
from notifyme_bot.scheduler_service import ReminderScheduler


@pytest.mark.asyncio
async def test_deliver_purge_when_reminder_missing(tmp_path: Path) -> None:
    """A job whose row vanished (already deleted elsewhere) sends nothing."""
    repo = ReminderRepository(str(tmp_path / "db.sqlite3"))
    await repo.init()
    bot = FakeBot()

    await ReminderScheduler(repository=repo, bot=bot)._deliver_and_purge(999)

    assert bot.messages == []


@pytest.mark.asyncio
async def test_deliver_uses_reminder_text_when_notification_blank(
    tmp_path: Path,
) -> None:
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
    )
    bot = FakeBot()
    scheduler = ReminderScheduler(repository=repo, bot=bot)
    await scheduler._deliver_and_purge(reminder.id)
    assert bot.messages[0]["text"] == "body only"


@pytest.mark.asyncio
async def test_scheduler_start_idempotent(tmp_path: Path) -> None:
    repo = ReminderRepository(str(tmp_path / "db.sqlite3"))
    await repo.init()
    scheduler = ReminderScheduler(repository=repo, bot=FakeBot())
    await scheduler.start()
    await scheduler.start()
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_scheduler_shutdown_when_not_running() -> None:
    scheduler = ReminderScheduler(
        repository=AsyncMock(),
        bot=FakeBot(),
    )
    await scheduler.shutdown()


def test_unknown_unit_cannot_be_stored() -> None:
    """The vocabulary is closed: a bad unit fails at the model boundary."""
    with pytest.raises(ValidationError):
        make_recurrence("fortnight")  # type: ignore[arg-type]
