from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import (
    FakeChat,
    FakeMessage,
    FakeUpdate,
    FakeUser,
    make_reminder,
    make_settings,
)

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
from notifyme_bot.config import AITUNNEL_BASE_URL


def test_build_application_registers_services(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    app = build_application(make_settings(database_path=str(db)))
    assert "settings" in app.bot_data
    assert "repository" in app.bot_data
    assert "parser" in app.bot_data
    assert app.bot_data["parser"]._provider._base_url == AITUNNEL_BASE_URL


@pytest.mark.asyncio
async def test_post_init_and_shutdown(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    app = build_application(make_settings(database_path=str(db)))
    app.bot = MagicMock()
    await _post_init(app)
    assert "scheduler" in app.bot_data
    await _post_shutdown(app)


@pytest.mark.asyncio
async def test_post_shutdown_without_scheduler(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    app = build_application(make_settings(database_path=str(db)))
    app.bot = MagicMock()
    app.bot_data.pop("scheduler", None)
    await _post_shutdown(app)


def _ctx(bot_data: dict) -> MagicMock:
    ctx = MagicMock()
    ctx.application.bot_data = bot_data
    ctx.bot = AsyncMock()
    ctx.args = None
    return ctx


@pytest.mark.asyncio
async def test_start_stop_help_early_return_no_user(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    settings = make_settings(database_path=str(db))
    repo = MagicMock()
    ctx = _ctx({"repository": repo, "settings": settings})
    await start(FakeUpdate(None, FakeChat(1), FakeMessage("x")), ctx)  # type: ignore[arg-type]
    repo.upsert_user.assert_not_called()


@pytest.mark.asyncio
async def test_help_early_return_no_chat(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    ctx = _ctx(
        {
            "repository": MagicMock(),
            "settings": make_settings(database_path=str(db)),
        }
    )
    await help_command(FakeUpdate(FakeUser(1), None, FakeMessage("x")), ctx)  # type: ignore[arg-type]
    ctx.bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_list_reminders_early_return(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    ctx = _ctx(
        {
            "repository": MagicMock(),
            "settings": make_settings(database_path=str(db)),
        }
    )
    await list_reminders(FakeUpdate(None, FakeChat(1), FakeMessage("x")), ctx)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_on_text_message_empty_message(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    app = build_application(make_settings(database_path=str(db)))
    ctx = MagicMock()
    ctx.application = MagicMock()
    ctx.application.bot_data = app.bot_data
    ctx.bot = AsyncMock()
    await on_text_message(
        FakeUpdate(FakeUser(1), FakeChat(1), FakeMessage("   ")),  # type: ignore[arg-type]
        ctx,
    )


@pytest.mark.asyncio
async def test_set_timezone_early_return_no_user(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    ctx = _ctx(
        {
            "repository": MagicMock(),
            "settings": make_settings(database_path=str(db)),
        }
    )
    await set_timezone(FakeUpdate(None, FakeChat(1), FakeMessage("")), ctx)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_handle_delete_request_early_return(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    ctx = _ctx(
        {
            "repository": MagicMock(),
            "settings": make_settings(database_path=str(db)),
        }
    )
    await _handle_delete_request(
        FakeUpdate(None, FakeChat(1), FakeMessage("x")),  # type: ignore[arg-type]
        ctx,
        repository=MagicMock(),
        locale="en",
        delete_all=False,
        delete_ids=[],
    )


@pytest.mark.asyncio
async def test_handle_delete_request_delete_nearest_without_target(
    tmp_path: Path,
) -> None:
    db = tmp_path / "db.sqlite3"
    repo = AsyncMock()
    reminder = make_reminder(
        id=9, chat_id=10, reminder_text="only", notification_text="n"
    )
    repo.list_pending_for_user = AsyncMock(return_value=[reminder])
    repo.delete_reminder = AsyncMock()
    ctx = _ctx(
        {"repository": repo, "settings": make_settings(database_path=str(db))}
    )
    await _handle_delete_request(
        FakeUpdate(FakeUser(1), FakeChat(10), FakeMessage("cancel")),  # type: ignore[arg-type]
        ctx,
        repository=repo,
        locale="en",
        delete_all=False,
        delete_ids=[],
    )
    repo.delete_reminder.assert_not_called()
    ctx.bot.send_message.assert_awaited()
