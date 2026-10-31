from __future__ import annotations

import logging
from datetime import UTC, datetime

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
            id=f"reminder:{reminder.id}",
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
        next_run = reminder.next_run(datetime.now(UTC))
        if next_run is None:
            await self._repository.delete_reminder(reminder_id)
            LOGGER.info("Delivered and purged reminder_id=%s", reminder_id)
            return

        await self._repository.update_reminder_next_run(
            reminder_id=reminder_id,
            next_run_utc=next_run,
        )
        self.schedule(reminder.model_copy(update={"remind_at_utc": next_run}))
        LOGGER.info(
            "Delivered recurring reminder_id=%s next_run=%s",
            reminder_id,
            next_run.isoformat(),
        )
