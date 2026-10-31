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
    _Command,
    _handle_delete_request,
    _post_init,
    _post_shutdown,
    build_application,
    help_command,
    list_reminders,
    on_text_message,
    on_voice_message,
    set_locale,
    set_timezone,
    start,
    stop,
)
from notifyme_bot.config import AITUNNEL_BASE_URL


def test_build_application_registers_services(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    app = build_application(make_settings(database_path=str(db)))
    assert "settings" in app.bot_data
    assert "repository" in app.bot_data
    assert "parser" in app.bot_data
    assert "llm" in app.bot_data
    client = app.bot_data["parser"]._provider._client
    assert str(client.base_url).rstrip("/") == AITUNNEL_BASE_URL


def test_settings_base_url_has_no_trailing_slash(tmp_path: Path) -> None:
    """Normalized once in config, so no consumer has to strip it again."""
    app = build_application(
        make_settings(database_path=str(tmp_path / "db.sqlite3"))
    )
    assert not app.bot_data["settings"].base_url.endswith("/")


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
async def test_stop_and_set_locale_early_return_no_user(
    tmp_path: Path,
) -> None:
    repo = MagicMock()
    ctx = _ctx({"repository": repo, "settings": make_settings()})

    await stop(FakeUpdate(None, FakeChat(1), FakeMessage("x")), ctx)  # type: ignore[arg-type]
    await set_locale(FakeUpdate(None, FakeChat(1), FakeMessage("x")), ctx)  # type: ignore[arg-type]

    repo.set_subscription.assert_not_called()
    repo.set_user_locale.assert_not_called()
    ctx.bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_on_voice_message_without_voice_does_nothing(
    tmp_path: Path,
) -> None:
    repo = MagicMock()
    ctx = _ctx({"repository": repo, "settings": make_settings()})

    await on_voice_message(
        FakeUpdate(FakeUser(1), FakeChat(1), FakeMessage("plain text")),  # type: ignore[arg-type]
        ctx,
    )
    await on_voice_message(FakeUpdate(FakeUser(1), FakeChat(1), None), ctx)  # type: ignore[arg-type]

    repo.is_subscribed.assert_not_called()
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
    ctx.bot.send_message.assert_not_called()


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
    ctx.bot.send_message.assert_not_called()


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
    ctx.bot.send_message.assert_not_called()


def _command(bot: AsyncMock, repo: MagicMock, locale: str = "en") -> _Command:
    return _Command(
        user_id=1,
        chat_id=10,
        locale=locale,
        repository=repo,
        settings=make_settings(),
        context=MagicMock(bot=bot),
    )


@pytest.mark.asyncio
async def test_handle_delete_request_no_pending() -> None:
    bot = AsyncMock()
    repo = MagicMock()
    repo.list_pending = AsyncMock(return_value=[])

    await _handle_delete_request(_command(bot, repo), True, [])

    bot.send_message.assert_awaited_once()
    repo.delete_reminder.assert_not_called()


@pytest.mark.asyncio
async def test_handle_delete_request_delete_nearest_without_target() -> None:
    """delete_all=false with no matching id deletes nothing and says so."""
    bot = AsyncMock()
    repo = MagicMock()
    repo.list_pending = AsyncMock(
        return_value=[make_reminder(id=9, chat_id=10, reminder_text="only")]
    )
    repo.delete_reminder = AsyncMock()

    await _handle_delete_request(_command(bot, repo), False, [])

    repo.delete_reminder.assert_not_called()
    assert "could not find" in bot.send_message.await_args.kwargs["text"]
