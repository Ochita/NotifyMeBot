from __future__ import annotations

import logging
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from notifyme_bot.config import AITUNNEL_BASE_URL, Settings
from notifyme_bot.db import ReminderRepository
from notifyme_bot.i18n import detect_locale, supported_locales, translate
from notifyme_bot.llm import (
    LLMServiceError,
    OpenAICompatibleStructuredChatProvider,
)
from notifyme_bot.llm_parser import (
    LLMReminderParser,
)
from notifyme_bot.models import ParsedReminder, Reminder
from notifyme_bot.scheduler_service import ReminderScheduler

LOGGER = logging.getLogger(__name__)


async def _is_access_denied(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    repository: ReminderRepository,
    settings: Settings,
    locale: str,
) -> bool:
    """When registration is closed, only Telegram users in ``users`` may proceed."""  # noqa: E501
    if settings.allow_new_users:
        return False
    if update.effective_user is None or update.effective_chat is None:
        return False
    if await repository.user_exists(update.effective_user.id):
        return False
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=_t(locale, "access_closed"),
    )
    return True


def build_application(settings: Settings) -> Application:
    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )

    repository = ReminderRepository(settings.database_path)
    base = (settings.base_url or AITUNNEL_BASE_URL).strip().rstrip("/")
    parser = LLMReminderParser(
        provider=OpenAICompatibleStructuredChatProvider(
            api_key=settings.api_key,
            model=settings.model_name,
            base_url=base,
        ),
        default_timezone=settings.default_timezone,
    )
    application.bot_data["settings"] = settings
    application.bot_data["repository"] = repository
    application.bot_data["parser"] = parser

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("set_timezone", set_timezone))
    application.add_handler(CommandHandler("set_locale", set_locale))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("list", list_reminders))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_text_message)
    )
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
    if update.effective_user is None or update.effective_chat is None:
        return
    repository: ReminderRepository = context.application.bot_data["repository"]
    settings: Settings = context.application.bot_data["settings"]
    locale = await _locale_code(update, context)
    if await _is_access_denied(update, context, repository, settings, locale):
        return
    await repository.upsert_user(
        user_id=update.effective_user.id,
        chat_id=update.effective_chat.id,
    )
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=_t(
            locale,
            "start",
            default_timezone=settings.default_timezone,
        ),
    )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.effective_chat is None:
        return
    repository: ReminderRepository = context.application.bot_data["repository"]
    settings: Settings = context.application.bot_data["settings"]
    locale = await _locale_code(update, context)
    if await _is_access_denied(update, context, repository, settings, locale):
        return
    await repository.set_subscription(
        update.effective_user.id, is_subscribed=False
    )
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=_t(locale, "stop"),
    )


async def help_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if update.effective_user is None or update.effective_chat is None:
        return
    repository: ReminderRepository = context.application.bot_data["repository"]
    settings: Settings = context.application.bot_data["settings"]
    locale = await _locale_code(update, context)
    if await _is_access_denied(update, context, repository, settings, locale):
        return
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=_t(locale, "help"),
    )


async def list_reminders(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if update.effective_user is None or update.effective_chat is None:
        return
    locale = await _locale_code(update, context)
    repository: ReminderRepository = context.application.bot_data["repository"]
    settings: Settings = context.application.bot_data["settings"]
    if await _is_access_denied(update, context, repository, settings, locale):
        return
    timezone_name = await _effective_timezone_name(
        repository=repository,
        user_id=update.effective_user.id,
        fallback=settings.default_timezone,
    )
    timezone = ZoneInfo(timezone_name)
    reminders = await repository.list_pending_for_user(
        update.effective_user.id
    )
    if not reminders:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_t(locale, "no_pending"),
        )
        return

    lines = [_t(locale, "pending_header")]
    for reminder in reminders[:20]:
        local_dt = reminder.remind_at_utc.astimezone(timezone).strftime(
            "%Y-%m-%d %H:%M %Z"
        )
        row = f"- {local_dt}: {reminder.reminder_text}"
        recur = _list_recurrence_suffix(locale, reminder)
        if recur:
            row = f"{row} · {recur}"
        lines.append(row)
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="\n".join(lines)
    )


async def set_timezone(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if update.effective_user is None or update.effective_chat is None:
        return
    locale = await _locale_code(update, context)
    repository: ReminderRepository = context.application.bot_data["repository"]
    settings: Settings = context.application.bot_data["settings"]
    if await _is_access_denied(update, context, repository, settings, locale):
        return

    if not context.args:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_t(locale, "set_timezone_missing"),
        )
        return

    timezone_name = context.args[0].strip()
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_t(locale, "set_timezone_invalid"),
        )
        return

    await repository.upsert_user(
        user_id=update.effective_user.id,
        chat_id=update.effective_chat.id,
    )
    await repository.set_user_timezone(update.effective_user.id, timezone_name)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=_t(locale, "set_timezone_saved", timezone=timezone_name),
    )


async def set_locale(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if update.effective_user is None or update.effective_chat is None:
        return
    locale = await _locale_code(update, context)
    repository: ReminderRepository = context.application.bot_data["repository"]
    settings: Settings = context.application.bot_data["settings"]
    if await _is_access_denied(update, context, repository, settings, locale):
        return
    if not context.args:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_t(locale, "set_locale_missing"),
        )
        return
    chosen = context.args[0].strip().lower().split("-", maxsplit=1)[0]
    if chosen not in supported_locales():
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_t(locale, "set_locale_invalid"),
        )
        return
    await repository.upsert_user(
        user_id=update.effective_user.id,
        chat_id=update.effective_chat.id,
    )
    await repository.set_user_locale(update.effective_user.id, chosen)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=_t(chosen, "set_locale_saved", locale=chosen),
    )


async def on_text_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if (
        update.effective_user is None
        or update.effective_chat is None
        or update.effective_message is None
    ):
        return
    message_text = (update.effective_message.text or "").strip()
    if not message_text:
        return
    locale = await _locale_code(update, context)

    repository, parser, scheduler, settings = _runtime_services(context)
    if await _is_access_denied(update, context, repository, settings, locale):
        return
    if not await repository.is_subscribed(update.effective_user.id):
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_t(locale, "unsubscribed"),
        )
        return

    try:
        timezone_name = await _effective_timezone_name(
            repository=repository,
            user_id=update.effective_user.id,
            fallback=settings.default_timezone,
        )
        pending = await repository.list_pending_for_user(
            update.effective_user.id
        )
        parsed_result = await parser.parse_command(
            message_text=message_text,
            now_utc=datetime.now(UTC),
            user_timezone=timezone_name,
            reminders=pending,
        )
        if parsed_result.delete is not None:
            await _handle_delete_request(
                update=update,
                context=context,
                repository=repository,
                locale=locale,
                delete_all=parsed_result.delete.delete_all,
                delete_ids=parsed_result.delete.delete_ids,
            )
            return
        parsed = parsed_result.reminder
        if parsed is None:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=_t(locale, "not_schedulable"),
            )
            return
    except LLMServiceError:
        LOGGER.exception("LLM provider is unavailable")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_t(locale, "llm_error"),
        )
        return
    except Exception:
        LOGGER.exception("Reminder parse failed")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_t(locale, "parse_error"),
        )
        return

    await _save_and_schedule_reminder(
        update=update,
        context=context,
        repository=repository,
        scheduler=scheduler,
        locale=locale,
        timezone_name=timezone_name,
        source_text=message_text,
        parsed=parsed,
    )


def _runtime_services(
    context: ContextTypes.DEFAULT_TYPE,
) -> tuple[ReminderRepository, LLMReminderParser, ReminderScheduler, Settings]:
    return (
        context.application.bot_data["repository"],
        context.application.bot_data["parser"],
        context.application.bot_data["scheduler"],
        context.application.bot_data["settings"],
    )


async def _save_and_schedule_reminder(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    repository: ReminderRepository,
    scheduler: ReminderScheduler,
    locale: str,
    timezone_name: str,
    source_text: str,
    parsed: ParsedReminder,
) -> None:
    reminder = await repository.create_reminder(
        user_id=update.effective_user.id,
        chat_id=update.effective_chat.id,
        source_text=source_text,
        reminder_text=parsed.reminder_text,
        notification_text=parsed.notification_text,
        remind_at_utc=parsed.remind_at_utc,
        timezone=timezone_name,
        is_recurring=parsed.is_recurring,
        recurrence_unit=parsed.recurrence_unit,
        recurrence_interval=parsed.recurrence_interval,
        recurrence_weekdays=parsed.recurrence_weekdays,
    )
    scheduler.schedule(reminder)

    local_dt = parsed.remind_at_utc.astimezone(ZoneInfo(timezone_name))
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=_t(
            locale,
            "saved",
            when=local_dt.strftime("%Y-%m-%d %H:%M %Z"),
            text=parsed.reminder_text,
        ),
    )


async def _handle_delete_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    repository: ReminderRepository,
    locale: str,
    delete_all: bool,
    delete_ids: list[int],
) -> None:
    if update.effective_user is None or update.effective_chat is None:
        return
    reminders = await repository.list_pending_for_user(
        update.effective_user.id
    )
    if not reminders:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_t(locale, "delete_none"),
        )
        return

    if delete_all:
        to_delete = reminders
    else:
        wanted = set(delete_ids)
        to_delete = [
            reminder for reminder in reminders if reminder.id in wanted
        ]

    if not to_delete:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_t(locale, "delete_not_found"),
        )
        return

    for reminder in to_delete:
        await repository.delete_reminder(reminder.id)

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=_t(locale, "delete_done", count=str(len(to_delete))),
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


def _t(locale: str, key: str, **kwargs: str) -> str:
    return translate(locale, key, **kwargs)


_WEEKDAY_SHORT_EN = (
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
    "Sun",
)


def _list_recurrence_suffix(locale: str, reminder: Reminder) -> str:
    """Short translated recurrence fragment for /list (empty if one-shot)."""
    if not reminder.is_recurring or not reminder.recurrence_unit:
        return ""
    n = max(reminder.recurrence_interval or 1, 1)
    unit = reminder.recurrence_unit
    if unit == "week" and reminder.recurrence_weekdays:
        days = ", ".join(
            _WEEKDAY_SHORT_EN[d]
            for d in sorted(set(reminder.recurrence_weekdays))
            if 0 <= d <= 6
        )
        if days:
            return _t(locale, "list_recur_weekly", days=days)
    key = f"list_recur_{unit}"
    return _t(locale, key, n=str(n))


async def _effective_timezone_name(
    repository: ReminderRepository,
    user_id: int,
    fallback: str,
) -> str:
    user_timezone = await repository.get_user_timezone(user_id)
    return user_timezone or fallback
