from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from notifyme_bot.db import ReminderRepository


def test_init_adds_missing_user_columns(tmp_path: Path) -> None:
    """A pre-locale ``users`` table gains the columns it lacks, keeps rows."""

    db_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY,
            chat_id INTEGER NOT NULL,
            is_subscribed INTEGER NOT NULL DEFAULT 1,
            created_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute("INSERT INTO users(user_id, chat_id) VALUES (1, 10);")
    conn.commit()
    conn.close()

    repo = ReminderRepository(str(db_path))
    asyncio.run(repo.init())

    conn = sqlite3.connect(str(db_path))
    user_cols = {row[1] for row in conn.execute("PRAGMA table_info(users);")}
    conn.close()

    assert {"timezone", "locale"} <= user_cols
    assert asyncio.run(repo.get_user_locale(1)) is None
    assert asyncio.run(repo.user_exists(1)) is True


def test_reminders_table_stores_one_recurrence_column(
    tmp_path: Path,
) -> None:
    """One-shot vs recurring is NULL vs a rule, not a parallel flag."""

    db_path = tmp_path / "fresh.sqlite3"
    asyncio.run(ReminderRepository(str(db_path)).init())

    conn = sqlite3.connect(str(db_path))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(reminders);")}
    conn.close()

    assert "recurrence" in cols
    assert not {"is_recurring", "recurrence_unit"} & cols
