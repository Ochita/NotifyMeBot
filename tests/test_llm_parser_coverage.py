from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from chat_payloads import chat_completion_response
from http_fakes import JsonHttpResponse
from test_llm_parser import FRIDAY, _command

from notifyme_bot.llm import (
    LLMServiceError,
    OpenAICompatibleStructuredChatProvider,
)
from notifyme_bot.llm_parser import LLMReminderParser


def _provider(model: str = "m") -> OpenAICompatibleStructuredChatProvider:
    return OpenAICompatibleStructuredChatProvider(
        api_key="k",
        model=model,
        base_url="https://api.aitunnel.ru/v1",
    )


def _parser() -> LLMReminderParser:
    return LLMReminderParser(provider=_provider(), default_timezone="UTC")


def _patch(monkeypatch: pytest.MonkeyPatch, content: dict) -> None:
    async def fake_post(self, url: str, json: dict, headers: dict):
        _ = (url, json, headers)
        return JsonHttpResponse(chat_completion_response(content))

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)


@pytest.mark.asyncio
async def test_parse_reminder_returns_none_when_missing_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, _command(reminder_text="", weekday=5))
    assert (await _parser().parse_command("x", FRIDAY)).reminder is None


@pytest.mark.asyncio
async def test_parse_reminder_returns_none_when_no_schedulable_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schedule intent with only a broken clock time has no anchor."""
    _patch(monkeypatch, _command(local_date="not-a-date", local_time="bad"))
    assert (await _parser().parse_command("x", FRIDAY)).reminder is None


@pytest.mark.asyncio
async def test_recurring_without_unit_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(
        monkeypatch,
        _command(
            is_recurring=True,
            recurrence_unit="none",
            recurrence_interval=1,
            weekday=5,
        ),
    )
    assert (await _parser().parse_command("x", FRIDAY)).reminder is None


@pytest.mark.asyncio
async def test_recurring_defaults_interval_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(
        monkeypatch,
        _command(
            is_recurring=True,
            recurrence_unit="week",
            recurrence_interval=0,
            recurrence_weekdays=[2, 5],
        ),
    )
    parsed = (await _parser().parse_command("x", FRIDAY)).reminder

    assert parsed is not None
    assert parsed.recurrence_interval == 1
    assert parsed.recurrence_weekdays == [2, 5]


def test_resolve_rejects_unknown_weekday_only_input() -> None:
    parser = _parser()
    assert (
        parser._build_parsed_reminder_from_data(
            _command(weekday=9), FRIDAY, "UTC"
        )
        is None
    )


def test_resolve_monthly_first_fire() -> None:
    parser = _parser()
    parsed = parser._build_parsed_reminder_from_data(
        _command(
            is_recurring=True,
            recurrence_unit="month",
            recurrence_interval=1,
            local_date="2026-11-03",
            local_time="09:00",
        ),
        FRIDAY,
        "UTC",
    )

    assert parsed is not None
    assert parsed.recurrence_unit == "month"
    assert parsed.remind_at_utc == datetime(2026, 11, 3, 9, 0, tzinfo=UTC)


def test_next_weekday_fire_raises_on_empty_weekdays() -> None:
    with pytest.raises(ValueError, match="weekdays"):
        LLMReminderParser._next_weekday_fire(
            [11], datetime.min.time(), FRIDAY, UTC
        )


@pytest.mark.asyncio
async def test_chat_completion_http_status_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = httpx.Request("POST", "https://example.com/test")
    resp = httpx.Response(401, request=req, text="unauthorized")

    async def fake_post(self, url: str, json: dict, headers: dict):
        _ = (url, json, headers)
        raise httpx.HTTPStatusError(
            "Unauthorized",
            request=req,
            response=resp,
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    parser = LLMReminderParser(
        provider=_provider("test-model"),
        default_timezone="UTC",
    )
    with pytest.raises(LLMServiceError, match="model=test-model") as exc:
        await parser.parse_command("hi", FRIDAY)
    assert exc.value.__cause__ is not None
    assert "401" in str(exc.value)


@pytest.mark.asyncio
async def test_chat_completion_bad_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post(self, url: str, json: dict, headers: dict):
        _ = (url, json, headers)
        return JsonHttpResponse({})

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    with pytest.raises(LLMServiceError):
        await _parser().parse_command("hi", FRIDAY)


@pytest.mark.asyncio
async def test_delete_all_not_inferred_from_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model delete_all=false wins over user wording like 'delete all'."""
    _patch(
        monkeypatch,
        _command(
            action="delete",
            should_schedule=False,
            should_delete=True,
            delete_all=False,
            delete_ids=[],
        ),
    )
    req = (
        await _parser().parse_command(
            "please delete all reminders", FRIDAY
        )
    ).delete

    assert req is not None
    assert req.delete_all is False
