from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from notifyme_bot.bot import (
    _handle_delete_request,
    _post_init,
    _post_shutdown,
    build_application,
    help_command,
    list_reminders,
    on_text_message,
    set_timezone,
    start,
)
from notifyme_bot.config import Settings
from notifyme_bot.models import Reminder


def _settings(db_path: Path) -> Settings:
    return Settings(
        telegram_bot_token="123456:ABCDEF",
        api_key="k",
        model_name="gemini-2.5-flash",
        database_path=str(db_path),
        default_timezone="UTC",
        allow_new_users=True,
        llm_provider="gemini",
        llm_base_url=None,
    )


def test_build_application_registers_services(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    app = build_application(_settings(db))
    assert "settings" in app.bot_data
    assert "repository" in app.bot_data
    assert "parser" in app.bot_data


@pytest.mark.asyncio
async def test_post_init_and_shutdown(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    app = build_application(_settings(db))
    app.bot = MagicMock()
    await _post_init(app)
    assert "scheduler" in app.bot_data
    await _post_shutdown(app)


@pytest.mark.asyncio
async def test_post_shutdown_without_scheduler(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    app = build_application(_settings(db))
    app.bot = MagicMock()
    app.bot_data.pop("scheduler", None)
    await _post_shutdown(app)


@dataclass
class _User:
    id: int
    language_code: str | None = None


@dataclass
class _Chat:
    id: int


@dataclass
class _Message:
    text: str | None


@dataclass
class _Update:
    effective_user: object | None
    effective_chat: object | None
    effective_message: object | None


class _Ctx:
    def __init__(self, bot_data: dict) -> None:
        self.application = MagicMock()
        self.application.bot_data = bot_data
        self.bot = AsyncMock()
        self.args: list[str] | None = None


@pytest.mark.asyncio
async def test_start_stop_help_early_return_no_user(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    settings = _settings(db)
    repo = MagicMock()
    ctx = _Ctx({"repository": repo, "settings": settings})
    await start(_Update(None, _Chat(1), _Message("x")), ctx)  # type: ignore[arg-type]
    repo.upsert_user.assert_not_called()


@pytest.mark.asyncio
async def test_help_early_return_no_chat(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    ctx = _Ctx({"repository": MagicMock(), "settings": _settings(db)})
    await help_command(_Update(_User(1), None, _Message("x")), ctx)  # type: ignore[arg-type]
    ctx.bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_list_reminders_early_return(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    ctx = _Ctx({"repository": MagicMock(), "settings": _settings(db)})
    await list_reminders(_Update(None, _Chat(1), _Message("x")), ctx)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_on_text_message_empty_message(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    app = build_application(_settings(db))
    ctx = MagicMock()
    ctx.application = MagicMock()
    ctx.application.bot_data = app.bot_data
    ctx.bot = AsyncMock()
    await on_text_message(
        _Update(_User(1), _Chat(1), _Message("   ")),  # type: ignore[arg-type]
        ctx,
    )


@pytest.mark.asyncio
async def test_set_timezone_early_return_no_user(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    ctx = _Ctx({"repository": MagicMock(), "settings": _settings(db)})
    await set_timezone(_Update(None, _Chat(1), _Message("")), ctx)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_handle_delete_request_early_return(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    ctx = _Ctx({"repository": MagicMock(), "settings": _settings(db)})
    await _handle_delete_request(
        _Update(None, _Chat(1), _Message("x")),  # type: ignore[arg-type]
        ctx,
        repository=MagicMock(),
        locale="en",
        delete_all=False,
        target_text=None,
    )


@pytest.mark.asyncio
async def test_handle_delete_request_delete_nearest_without_target(
    tmp_path: Path,
) -> None:
    db = tmp_path / "db.sqlite3"
    repo = AsyncMock()
    reminder = Reminder(
        id=9,
        user_id=1,
        chat_id=10,
        source_text="s",
        reminder_text="only",
        notification_text="n",
        remind_at_utc=datetime.now(UTC) + timedelta(days=1),
        timezone="UTC",
        is_recurring=False,
        recurrence_unit=None,
        recurrence_interval=None,
        recurrence_weekdays=None,
        created_at_utc=datetime.now(UTC),
    )
    repo.list_pending_for_user = AsyncMock(return_value=[reminder])
    repo.delete_reminder = AsyncMock()
    ctx = _Ctx({"repository": repo, "settings": _settings(db)})
    await _handle_delete_request(
        _Update(_User(1), _Chat(10), _Message("cancel")),  # type: ignore[arg-type]
        ctx,
        repository=repo,
        locale="en",
        delete_all=False,
        target_text=None,
    )
    repo.delete_reminder.assert_called_once_with(9)
