from __future__ import annotations

import sqlite3
from pathlib import Path

from notifyme_bot.db import ReminderRepository


def test_init_migrates_legacy_reminders_table(tmp_path: Path) -> None:
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
    conn.execute(
        """
        CREATE TABLE reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            chat_id INTEGER NOT NULL,
            source_text TEXT NOT NULL,
            reminder_text TEXT NOT NULL,
            remind_at_utc TEXT NOT NULL,
            created_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()

    repo = ReminderRepository(str(db_path))

    async def run() -> None:
        await repo.init()

    import asyncio

    asyncio.run(run())

    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("PRAGMA table_info(reminders);")
    cols = {row[1] for row in cur.fetchall()}
    conn.close()
    assert "timezone" in cols
    assert "notification_text" in cols


def test_parse_weekdays_invalid_json_returns_none() -> None:
    assert ReminderRepository._parse_weekdays("{not json") is None
    assert ReminderRepository._parse_weekdays('{"x": 1}') is None
