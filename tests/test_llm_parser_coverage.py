from __future__ import annotations

import json
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import httpx
import pytest
from http_fakes import JsonHttpResponse

from notifyme_bot.llm.errors import LLMServiceError
from notifyme_bot.llm.openai_compatible import (
    OpenAICompatibleStructuredChatProvider,
)
from notifyme_bot.llm_parser import LLMReminderParser


def _provider(model: str = "m") -> OpenAICompatibleStructuredChatProvider:
    return OpenAICompatibleStructuredChatProvider(
        api_key="k",
        model=model,
        base_url="https://api.aitunnel.ru/v1",
    )


@pytest.mark.asyncio
async def test_parse_reminder_returns_none_when_missing_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "should_schedule": True,
                            "reminder_text": "",
                            "notification_text": "x",
                            "remind_at_utc": "",
                            "is_recurring": False,
                            "recurrence_unit": "none",
                            "recurrence_interval": 0,
                            "recurrence_weekdays": [],
                        }
                    )
                }
            }
        ]
    }

    async def fake_post(self, url: str, json: dict, headers: dict):
        return JsonHttpResponse(payload)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    parser = LLMReminderParser(
        provider=_provider(),
        default_timezone="UTC",
    )
    out = (await parser.parse_command("x", datetime.now(UTC))).reminder
    assert out is None


@pytest.mark.asyncio
async def test_parse_reminder_returns_none_when_no_schedulable_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-recurring with invalid remind_at and no fallback -> None."""
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "should_schedule": True,
                            "reminder_text": "a",
                            "notification_text": "b",
                            "remind_at_utc": "not-a-date",
                            "is_recurring": False,
                            "recurrence_unit": "none",
                            "recurrence_interval": 0,
                            "recurrence_weekdays": [],
                        }
                    )
                }
            }
        ]
    }

    async def fake_post(self, url: str, json: dict, headers: dict):
        return JsonHttpResponse(payload)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    parser = LLMReminderParser(
        provider=_provider(),
        default_timezone="UTC",
    )
    out = (await parser.parse_command("x", datetime.now(UTC))).reminder
    assert out is None


def test_parse_remind_at_naive_is_user_local() -> None:
    dt = LLMReminderParser._parse_remind_at_to_utc(
        "2025-01-15T12:00:00",
        "Europe/Moscow",
    )
    noon_moscow = datetime(
        2025,
        1,
        15,
        12,
        0,
        0,
        tzinfo=ZoneInfo("Europe/Moscow"),
    )
    assert dt == noon_moscow.astimezone(UTC)


def test_parse_remind_at_explicit_z_is_utc() -> None:
    dt = LLMReminderParser._parse_remind_at_to_utc(
        "2025-01-15T12:00:00Z",
        "Europe/Moscow",
    )
    assert dt == datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


def test_first_fire_from_llm_rejects_invalid_and_past() -> None:
    now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
    assert (
        LLMReminderParser._first_fire_from_llm(
            "not-a-date", "UTC", now
        )
        is None
    )
    past = "2025-06-15T11:00:00Z"
    assert LLMReminderParser._first_fire_from_llm(past, "UTC", now) is None


@pytest.mark.asyncio
async def test_chat_completion_http_status_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = httpx.Request("POST", "https://example.com/test")
    resp = httpx.Response(401, request=req, text="unauthorized")

    async def fake_post(self, url: str, json: dict, headers: dict):
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
        await parser.parse_command("hi", datetime.now(UTC))
    assert exc.value.__cause__ is not None
    assert "401" in str(exc.value)


@pytest.mark.asyncio
async def test_chat_completion_bad_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post(self, url: str, json: dict, headers: dict):
        return JsonHttpResponse({})

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    parser = LLMReminderParser(
        provider=_provider(),
        default_timezone="UTC",
    )
    with pytest.raises(LLMServiceError):
        await parser.parse_command("hi", datetime.now(UTC))


@pytest.mark.asyncio
async def test_parse_delete_request_delete_all_from_llm_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """delete_all follows model JSON only (no phrase heuristics)."""
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "should_delete": True,
                            "delete_all": True,
                            "target_text": "",
                        }
                    )
                }
            }
        ]
    }

    async def fake_post(self, url: str, json: dict, headers: dict):
        return JsonHttpResponse(payload)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    parser = LLMReminderParser(
        provider=_provider(),
        default_timezone="UTC",
    )
    req = (
        await parser.parse_command(
            "please delete all reminders", datetime.now(UTC)
        )
    ).delete
    assert req is not None
    assert req.delete_all is True
    assert req.target_text is None


@pytest.mark.asyncio
async def test_parse_delete_request_delete_all_not_inferred_from_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model delete_all=false wins over user wording like 'delete all'."""
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "should_delete": True,
                            "delete_all": False,
                            "target_text": "",
                        }
                    )
                }
            }
        ]
    }

    async def fake_post(self, url: str, json: dict, headers: dict):
        return JsonHttpResponse(payload)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    parser = LLMReminderParser(
        provider=_provider(),
        default_timezone="UTC",
    )
    req = (
        await parser.parse_command(
            "please delete all reminders", datetime.now(UTC)
        )
    ).delete
    assert req is not None
    assert req.delete_all is False
