from __future__ import annotations

from datetime import UTC, datetime

import aiosqlite

from notifyme_bot.models import Recurrence, Reminder

_COLUMNS = (
    "id, user_id, chat_id, reminder_text, notification_text, remind_at_utc, "
    "timezone, recurrence"
)


class ReminderRepository:
    def __init__(self, database_path: str) -> None:
        self._database_path = database_path

    async def init(self) -> None:
        async with aiosqlite.connect(self._database_path) as conn:
            await conn.execute("PRAGMA journal_mode=WAL;")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    chat_id INTEGER NOT NULL,
                    is_subscribed INTEGER NOT NULL DEFAULT 1,
                    timezone TEXT,
                    locale TEXT,
                    created_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            await self._ensure_columns(conn, "users", ("timezone", "locale"))
            # The rule is one JSON column: NULL means one-shot, so there is no
            # flag to fall out of sync and no schema change when the rule
            # grows a field (day_of_month, for one).
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    source_text TEXT NOT NULL,
                    reminder_text TEXT NOT NULL,
                    notification_text TEXT NOT NULL DEFAULT '',
                    remind_at_utc TEXT NOT NULL,
                    timezone TEXT NOT NULL DEFAULT 'UTC',
                    recurrence TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_due "
                "ON reminders(remind_at_utc);"
            )
            await conn.commit()

    async def user_exists(self, user_id: int) -> bool:
        async with aiosqlite.connect(self._database_path) as conn:
            async with conn.execute(
                "SELECT 1 FROM users WHERE user_id = ? LIMIT 1;",
                (user_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return row is not None

    async def upsert_user(self, user_id: int, chat_id: int) -> None:
        async with aiosqlite.connect(self._database_path) as conn:
            await conn.execute(
                """
                INSERT INTO users(user_id, chat_id, is_subscribed)
                VALUES (?, ?, 1)
                ON CONFLICT(user_id) DO UPDATE SET
                    chat_id = excluded.chat_id,
                    is_subscribed = 1;
                """,
                (user_id, chat_id),
            )
            await conn.commit()

    async def set_subscription(
        self, user_id: int, is_subscribed: bool
    ) -> None:
        async with aiosqlite.connect(self._database_path) as conn:
            await conn.execute(
                "UPDATE users SET is_subscribed = ? WHERE user_id = ?;",
                (1 if is_subscribed else 0, user_id),
            )
            await conn.commit()

    async def set_user_timezone(self, user_id: int, timezone: str) -> None:
        await self._set_user_field(user_id, "timezone", timezone)

    async def get_user_timezone(self, user_id: int) -> str | None:
        return await self._get_user_field(user_id, "timezone")

    async def set_user_locale(self, user_id: int, locale: str) -> None:
        await self._set_user_field(user_id, "locale", locale)

    async def get_user_locale(self, user_id: int) -> str | None:
        return await self._get_user_field(user_id, "locale")

    async def _set_user_field(
        self, user_id: int, column: str, value: str
    ) -> None:
        async with aiosqlite.connect(self._database_path) as conn:
            await conn.execute(
                f"UPDATE users SET {column} = ? WHERE user_id = ?;",  # noqa: S608
                (value, user_id),
            )
            await conn.commit()

    async def is_subscribed(self, user_id: int) -> bool:
        async with aiosqlite.connect(self._database_path) as conn:
            async with conn.execute(
                "SELECT is_subscribed FROM users WHERE user_id = ?;",
                (user_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return bool(row[0]) if row else False

    async def _get_user_field(self, user_id: int, column: str) -> str | None:
        async with aiosqlite.connect(self._database_path) as conn:
            async with conn.execute(
                f"SELECT {column} FROM users WHERE user_id = ?;",  # noqa: S608
                (user_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return str(row[0]) if row and row[0] else None

    async def create_reminder(
        self,
        user_id: int,
        chat_id: int,
        source_text: str,
        reminder_text: str,
        notification_text: str,
        remind_at_utc: datetime,
        timezone: str,
        recurrence: Recurrence | None = None,
    ) -> Reminder:
        remind_at = remind_at_utc.astimezone(UTC).isoformat()
        timezone_name = str(timezone).strip() or "UTC"
        async with aiosqlite.connect(self._database_path) as conn:
            cursor = await conn.execute(
                """
                INSERT INTO reminders(
                    user_id,
                    chat_id,
                    source_text,
                    reminder_text,
                    notification_text,
                    remind_at_utc,
                    timezone,
                    recurrence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    user_id,
                    chat_id,
                    # Kept for debugging; never read back by the bot.
                    source_text,
                    reminder_text,
                    notification_text,
                    remind_at,
                    timezone_name,
                    recurrence.model_dump_json() if recurrence else None,
                ),
            )
            await conn.commit()
            reminder_id = cursor.lastrowid

        return Reminder(
            id=int(reminder_id),
            user_id=user_id,
            chat_id=chat_id,
            reminder_text=reminder_text,
            notification_text=notification_text,
            remind_at_utc=self._parse_utc(remind_at),
            timezone=timezone_name,
            recurrence=recurrence,
        )

    async def get_reminder(self, reminder_id: int) -> Reminder | None:
        async with aiosqlite.connect(self._database_path) as conn:
            async with conn.execute(
                f"SELECT {_COLUMNS} FROM reminders WHERE id = ?;",
                (reminder_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return self._to_reminder(row) if row else None

    async def list_pending(self, user_id: int | None = None) -> list[Reminder]:
        """Due reminders, for one user when ``user_id`` is given."""
        where = "julianday(remind_at_utc) >= julianday('now')"
        params: tuple[int, ...] = ()
        if user_id is not None:
            where = f"user_id = ? AND {where}"
            params = (user_id,)
        async with aiosqlite.connect(self._database_path) as conn:
            async with conn.execute(
                f"SELECT {_COLUMNS} FROM reminders WHERE {where} "
                "ORDER BY remind_at_utc ASC;",
                params,
            ) as cursor:
                rows = await cursor.fetchall()
        return [self._to_reminder(row) for row in rows]

    async def delete_reminder(self, reminder_id: int) -> None:
        async with aiosqlite.connect(self._database_path) as conn:
            await conn.execute(
                "DELETE FROM reminders WHERE id = ?;", (reminder_id,)
            )
            await conn.commit()

    async def update_reminder_next_run(
        self,
        reminder_id: int,
        next_run_utc: datetime,
    ) -> None:
        async with aiosqlite.connect(self._database_path) as conn:
            await conn.execute(
                "UPDATE reminders SET remind_at_utc = ? WHERE id = ?;",
                (next_run_utc.astimezone(UTC).isoformat(), reminder_id),
            )
            await conn.commit()

    @staticmethod
    def _to_reminder(row: tuple) -> Reminder:
        return Reminder(
            id=int(row[0]),
            user_id=int(row[1]),
            chat_id=int(row[2]),
            reminder_text=str(row[3]),
            notification_text=str(row[4]),
            remind_at_utc=ReminderRepository._parse_utc(str(row[5])),
            timezone=str(row[6]) if row[6] else "UTC",
            recurrence=Recurrence.model_validate_json(str(row[7]))
            if row[7]
            else None,
        )

    @staticmethod
    def _parse_utc(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    @staticmethod
    async def _ensure_columns(
        conn: aiosqlite.Connection,
        table: str,
        columns: tuple[str, ...],
    ) -> None:
        async with conn.execute(f"PRAGMA table_info({table});") as cursor:
            have = {str(row[1]) for row in await cursor.fetchall()}
        for name in columns:
            if name not in have:
                await conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {name} TEXT;"
                )
