from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from notifyme_bot.config import Settings
from notifyme_bot.db import ReminderRepository
from notifyme_bot.i18n import detect_locale, supported_locales, translate
from notifyme_bot.llm import LLMServiceError, OpenAICompatibleService
from notifyme_bot.llm_parser import LLMReminderParser
from notifyme_bot.models import ParsedReminder, Reminder
from notifyme_bot.scheduler_service import ReminderScheduler

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _Command:
    """One update plus its services, already access-checked."""

    user_id: int
    chat_id: int
    locale: str
    repository: ReminderRepository
    settings: Settings
    context: ContextTypes.DEFAULT_TYPE

    async def reply(self, text: str) -> None:
        await self.context.bot.send_message(chat_id=self.chat_id, text=text)


async def _command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> _Command | None:
    """Shared prologue: skip updates without a user/chat, gate access.

    When registration is closed, only Telegram users already in ``users``
    may proceed.
    """
    user, chat = update.effective_user, update.effective_chat
    if user is None or chat is None:
        return None
    repository: ReminderRepository = context.application.bot_data["repository"]
    settings: Settings = context.application.bot_data["settings"]
    locale = await _locale_code(update, context)
    if not settings.allow_new_users and not await repository.user_exists(
        user.id
    ):
        await context.bot.send_message(
            chat_id=chat.id, text=translate(locale, "access_closed")
        )
        return None
    return _Command(user.id, chat.id, locale, repository, settings, context)


def build_application(settings: Settings) -> Application:
    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )

    repository = ReminderRepository(settings.database_path)
    service = OpenAICompatibleService(
        api_key=settings.api_key,
        model=settings.model_name,
        base_url=settings.base_url,
        stt_model=settings.stt_model,
    )
    parser = LLMReminderParser(
        provider=service,
        default_timezone=settings.default_timezone,
    )
    application.bot_data["settings"] = settings
    application.bot_data["repository"] = repository
    application.bot_data["parser"] = parser
    application.bot_data["llm"] = service

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("set_timezone", set_timezone))
    application.add_handler(CommandHandler("set_locale", set_locale))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("list", list_reminders))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_text_message)
    )
    application.add_handler(MessageHandler(filters.VOICE, on_voice_message))
    return application


async def _post_init(application: Application) -> None:
    repository: ReminderRepository = application.bot_data["repository"]
    await repository.init()
    scheduler = ReminderScheduler(repository=repository, bot=application.bot)
    await scheduler.start()
    application.bot_data["scheduler"] = scheduler
    LOGGER.info("Bot initialized")


async def _post_shutdown(application: Application) -> None:
    scheduler: ReminderScheduler | None = application.bot_data.get("scheduler")
    if scheduler:
        await scheduler.shutdown()
    LOGGER.info("Bot shutdown complete")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cmd = await _command(update, context)
    if cmd is None:
        return
    await cmd.repository.upsert_user(user_id=cmd.user_id, chat_id=cmd.chat_id)
    await cmd.reply(
        translate(
            cmd.locale,
            "start",
            default_timezone=cmd.settings.default_timezone,
        )
    )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cmd = await _command(update, context)
    if cmd is None:
        return
    await cmd.repository.set_subscription(cmd.user_id, is_subscribed=False)
    await cmd.reply(translate(cmd.locale, "stop"))


async def help_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    cmd = await _command(update, context)
    if cmd is None:
        return
    await cmd.reply(translate(cmd.locale, "help"))


async def list_reminders(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    cmd = await _command(update, context)
    if cmd is None:
        return
    timezone_name = await _effective_timezone_name(
        repository=cmd.repository,
        user_id=cmd.user_id,
        fallback=cmd.settings.default_timezone,
    )
    timezone = ZoneInfo(timezone_name)
    reminders = await cmd.repository.list_pending(cmd.user_id)
    if not reminders:
        await cmd.reply(translate(cmd.locale, "no_pending"))
        return

    lines = [translate(cmd.locale, "pending_header")]
    for reminder in reminders[:20]:
        local_dt = reminder.remind_at_utc.astimezone(timezone).strftime(
            "%Y-%m-%d %H:%M %Z"
        )
        row = f"- {local_dt}: {reminder.reminder_text}"
        recur = _list_recurrence_suffix(cmd.locale, reminder)
        if recur:
            row = f"{row} · {recur}"
        lines.append(row)
    await cmd.reply("\n".join(lines))


async def set_timezone(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    cmd = await _command(update, context)
    if cmd is None:
        return

    if not context.args:
        await cmd.reply(translate(cmd.locale, "set_timezone_missing"))
        return

    timezone_name = context.args[0].strip()
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        await cmd.reply(translate(cmd.locale, "set_timezone_invalid"))
        return

    await cmd.repository.upsert_user(user_id=cmd.user_id, chat_id=cmd.chat_id)
    await cmd.repository.set_user_timezone(cmd.user_id, timezone_name)
    await cmd.reply(
        translate(cmd.locale, "set_timezone_saved", timezone=timezone_name)
    )


async def set_locale(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    cmd = await _command(update, context)
    if cmd is None:
        return
    if not context.args:
        await cmd.reply(translate(cmd.locale, "set_locale_missing"))
        return
    chosen = context.args[0].strip().lower().split("-", maxsplit=1)[0]
    if chosen not in supported_locales():
        await cmd.reply(translate(cmd.locale, "set_locale_invalid"))
        return
    await cmd.repository.upsert_user(user_id=cmd.user_id, chat_id=cmd.chat_id)
    await cmd.repository.set_user_locale(cmd.user_id, chosen)
    await cmd.reply(translate(chosen, "set_locale_saved", locale=chosen))


async def on_text_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if update.effective_message is None:
        return
    message_text = (update.effective_message.text or "").strip()
    if not message_text:
        return
    await _handle_message_text(update, context, message_text)


async def on_voice_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Transcribe the voice note, then reuse the text pipeline unchanged."""
    message = update.effective_message
    if message is None or message.voice is None:
        return
    cmd = await _message_gate(update, context)
    if cmd is None:
        return
    locale = cmd.locale

    status = await message.reply_text(translate(locale, "status_transcribing"))
    try:
        downloaded = BytesIO()
        tg_file = await context.bot.get_file(message.voice.file_id)
        await tg_file.download_to_memory(downloaded)
        service: OpenAICompatibleService = context.application.bot_data["llm"]
        transcript = await service.transcribe(
            downloaded.getvalue(),
            message.voice.mime_type or "audio/ogg",
            language=locale,
        )
    except LLMServiceError:
        LOGGER.exception("Transcription failed")
        await status.edit_text(translate(locale, "transcribe_error"))
        return
    except Exception:
        LOGGER.exception("Voice download failed")
        await status.edit_text(translate(locale, "transcribe_error"))
        return

    if not transcript:
        await status.edit_text(translate(locale, "transcribe_empty"))
        return
    await status.delete()
    await _handle_message_text(update, context, transcript)


async def _message_gate(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> _Command | None:
    """Gate for text and voice: access plus subscription."""
    cmd = await _command(update, context)
    if cmd is None:
        return None
    if await cmd.repository.is_subscribed(cmd.user_id):
        return cmd
    await cmd.reply(translate(cmd.locale, "unsubscribed"))
    return None


async def _handle_message_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    message_text: str,
) -> None:
    """Schedule or delete from one piece of text, whatever its origin."""
    cmd = await _message_gate(update, context)
    if cmd is None:
        return
    parser: LLMReminderParser = context.application.bot_data["parser"]
    scheduler: ReminderScheduler = context.application.bot_data["scheduler"]

    try:
        timezone_name = await _effective_timezone_name(
            repository=cmd.repository,
            user_id=cmd.user_id,
            fallback=cmd.settings.default_timezone,
        )
        pending = await cmd.repository.list_pending(cmd.user_id)
        parsed_result = await parser.parse_command(
            message_text=message_text,
            now_utc=datetime.now(UTC),
            user_timezone=timezone_name,
            reminders=pending,
        )
        if parsed_result.delete is not None:
            await _handle_delete_request(
                cmd,
                parsed_result.delete.delete_all,
                parsed_result.delete.delete_ids,
            )
            return
        parsed = parsed_result.reminder
        if parsed is None:
            await cmd.reply(translate(cmd.locale, "not_schedulable"))
            return
    except LLMServiceError:
        LOGGER.exception("LLM provider is unavailable")
        await cmd.reply(translate(cmd.locale, "llm_error"))
        return
    except Exception:
        LOGGER.exception("Reminder parse failed")
        await cmd.reply(translate(cmd.locale, "parse_error"))
        return

    await _save_and_schedule_reminder(
        cmd=cmd,
        scheduler=scheduler,
        timezone_name=timezone_name,
        source_text=message_text,
        parsed=parsed,
    )


async def _save_and_schedule_reminder(
    cmd: _Command,
    scheduler: ReminderScheduler,
    timezone_name: str,
    source_text: str,
    parsed: ParsedReminder,
) -> None:
    reminder = await cmd.repository.create_reminder(
        user_id=cmd.user_id,
        chat_id=cmd.chat_id,
        source_text=source_text,
        reminder_text=parsed.reminder_text,
        notification_text=parsed.notification_text,
        remind_at_utc=parsed.remind_at_utc,
        timezone=timezone_name,
        recurrence=parsed.recurrence,
    )
    scheduler.schedule(reminder)

    local_dt = parsed.remind_at_utc.astimezone(ZoneInfo(timezone_name))
    await cmd.reply(
        translate(
            cmd.locale,
            "saved",
            when=local_dt.strftime("%Y-%m-%d %H:%M %Z"),
            text=parsed.reminder_text,
        )
    )


async def _handle_delete_request(
    cmd: _Command,
    delete_all: bool,
    delete_ids: list[int],
) -> None:
    reminders = await cmd.repository.list_pending(cmd.user_id)
    if not reminders:
        await cmd.reply(translate(cmd.locale, "delete_none"))
        return

    if delete_all:
        to_delete = reminders
    else:
        wanted = set(delete_ids)
        to_delete = [
            reminder for reminder in reminders if reminder.id in wanted
        ]

    if not to_delete:
        await cmd.reply(translate(cmd.locale, "delete_not_found"))
        return

    for reminder in to_delete:
        await cmd.repository.delete_reminder(reminder.id)

    await cmd.reply(
        translate(cmd.locale, "delete_done", count=str(len(to_delete)))
    )


async def _locale_code(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> str:
    settings: Settings = context.application.bot_data["settings"]
    repository: ReminderRepository = context.application.bot_data["repository"]
    user = update.effective_user
    if user is not None:
        stored = await repository.get_user_locale(user.id)
        if stored in supported_locales():
            return stored
    language_code = user.language_code if user is not None else None
    return detect_locale(language_code, settings.default_locale)


def _list_recurrence_suffix(locale: str, reminder: Reminder) -> str:
    """Short translated recurrence fragment for /list (empty if one-shot)."""
    rule = reminder.recurrence
    if rule is None:
        return ""
    if rule.unit == "week" and rule.weekdays:
        days = ", ".join(calendar.day_abbr[day] for day in rule.weekdays)
        return translate(locale, "list_recur_weekly", days=days)
    return translate(locale, f"list_recur_{rule.unit}", n=str(rule.interval))


async def _effective_timezone_name(
    repository: ReminderRepository,
    user_id: int,
    fallback: str,
) -> str:
    user_timezone = await repository.get_user_timezone(user_id)
    return user_timezone or fallback
