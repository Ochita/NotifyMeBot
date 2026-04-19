from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Bot

from notifyme_bot.db import ReminderRepository
from notifyme_bot.models import Reminder

LOGGER = logging.getLogger(__name__)


class ReminderScheduler:
    def __init__(self, repository: ReminderRepository, bot: Bot) -> None:
        self._repository = repository
        self._bot = bot
        self._scheduler = AsyncIOScheduler(timezone=UTC)

    async def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()
        pending = await self._repository.list_pending()
        for reminder in pending:
            self.schedule(reminder)
        LOGGER.info("Loaded %s pending reminders", len(pending))

    async def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    def schedule(self, reminder: Reminder) -> None:
        if reminder.remind_at_utc <= datetime.now(UTC):
            return
        self._scheduler.add_job(
            self._deliver_and_purge,
            trigger="date",
            run_date=reminder.remind_at_utc,
            kwargs={"reminder_id": reminder.id},
            id=self._job_id(reminder.id),
            replace_existing=True,
            misfire_grace_time=300,
        )

    async def _deliver_and_purge(self, reminder_id: int) -> None:
        reminder = await self._repository.get_reminder(reminder_id)
        if reminder is None:
            return

        notification_text = (
            reminder.notification_text.strip()
            or reminder.reminder_text.strip()
        )
        await self._bot.send_message(
            chat_id=reminder.chat_id,
            text=notification_text,
        )
        if reminder.is_recurring:
            next_run = self._next_recurring_run(reminder)
            await self._repository.update_reminder_next_run(
                reminder_id=reminder_id,
                next_run_utc=next_run,
            )
            updated = await self._repository.get_reminder(reminder_id)
            if updated:
                self.schedule(updated)
            LOGGER.info(
                "Delivered recurring reminder_id=%s next_run=%s",
                reminder_id,
                next_run.isoformat(),
            )
            return

        await self._repository.delete_reminder(reminder_id)
        LOGGER.info("Delivered and purged reminder_id=%s", reminder_id)

    @staticmethod
    def _job_id(reminder_id: int) -> str:
        return f"reminder:{reminder_id}"

    @staticmethod
    def _next_recurring_run(reminder: Reminder) -> datetime:
        now_utc = datetime.now(UTC)
        interval = max(reminder.recurrence_interval or 1, 1)
        tz = ZoneInfo(reminder.timezone or "UTC")
        now_local = now_utc.astimezone(tz)
        base_local = reminder.remind_at_utc.astimezone(tz)

        if reminder.recurrence_unit == "minute":
            candidate_local = base_local + timedelta(minutes=interval)
            while candidate_local <= now_local:
                candidate_local += timedelta(minutes=interval)
            return candidate_local.astimezone(UTC)

        if reminder.recurrence_unit == "hour":
            candidate_local = base_local + timedelta(hours=interval)
            while candidate_local <= now_local:
                candidate_local += timedelta(hours=interval)
            return candidate_local.astimezone(UTC)

        if reminder.recurrence_unit == "day":
            candidate_local = base_local + timedelta(days=interval)
            while candidate_local <= now_local:
                candidate_local += timedelta(days=interval)
            return candidate_local.astimezone(UTC)

        if reminder.recurrence_unit == "week":
            weekdays = sorted(set(reminder.recurrence_weekdays or []))
            if weekdays:
                candidate_local = base_local
                for _ in range(14):
                    candidate_local += timedelta(days=1)
                    candidate_utc = candidate_local.astimezone(UTC)
                    if (
                        candidate_local.weekday() in weekdays
                        and candidate_utc > now_utc
                    ):
                        return candidate_utc
            candidate_local = base_local + timedelta(weeks=interval)
            while candidate_local <= now_local:
                candidate_local += timedelta(weeks=interval)
            return candidate_local.astimezone(UTC)

        if reminder.recurrence_unit == "month":
            target_day = base_local.day
            year = now_local.year
            month = now_local.month
            for _ in range(36):
                last_day = ReminderScheduler._last_day_of_month(year, month)
                day = min(target_day, last_day)
                candidate_local = base_local.replace(
                    year=year,
                    month=month,
                    day=day,
                )
                if candidate_local > now_local:
                    return candidate_local.astimezone(UTC)
                month += interval
                year += (month - 1) // 12
                month = ((month - 1) % 12) + 1
            return (base_local + timedelta(days=31)).astimezone(UTC)

        # Fallback for unexpected recurrence configuration.
        return (base_local + timedelta(days=1)).astimezone(UTC)

    @staticmethod
    def _last_day_of_month(year: int, month: int) -> int:
        if month == 2:
            leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
            return 29 if leap else 28
        if month in {4, 6, 9, 11}:
            return 30
        return 31
