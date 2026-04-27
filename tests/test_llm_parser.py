"""Unit tests for ``LLMReminderParser`` (LLM JSON is source of truth)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from gemini_payloads import gemini_choices_response
from http_fakes import JsonHttpResponse

from notifyme_bot.llm.gemini import (
    GeminiStructuredChatProvider,
    extract_gemini_text,
)
from notifyme_bot.llm_parser import (
    GeminiServiceError,
    LLMReminderParser,
)
from notifyme_bot.models import CommandParseResult


def _parser() -> LLMReminderParser:
    return LLMReminderParser(
        provider=GeminiStructuredChatProvider(
            api_key="test-key",
            model="test/model",
        ),
        default_timezone="UTC",
    )


@pytest.mark.asyncio
async def test_parse_reminder_success(monkeypatch: pytest.MonkeyPatch) -> None:
    now_utc = datetime.now(UTC)
    remind_at = (
        (now_utc + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    )
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "should_schedule": True,
                            "reminder_text": "send 100$ to Alexei",
                            "notification_text": (
                                "Notification: send 100$ to Alexei"
                            ),
                            "remind_at_utc": remind_at,
                            "is_recurring": True,
                            "recurrence_unit": "week",
                            "recurrence_interval": 1,
                            "recurrence_weekdays": [0, 2],
                        }
                    )
                }
            }
        ]
    }

    async def fake_post(
        self, url: str, json: dict, headers: dict
    ) -> JsonHttpResponse:
        assert "models/test/model:generateContent" in url
        assert "x-goog-api-key" in headers
        assert (
            json["generationConfig"]["responseMimeType"]
            == "application/json"
        )
        return JsonHttpResponse(payload)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    parser = _parser()
    parsed = await parser.parse_reminder("Please remind me tomorrow", now_utc)

    assert parsed is not None
    assert parsed.reminder_text == "send 100$ to Alexei"
    assert "Notification:" in parsed.notification_text
    assert parsed.remind_at_utc > now_utc
    assert parsed.is_recurring is True
    assert parsed.recurrence_unit == "week"
    assert parsed.recurrence_interval == 1
    assert parsed.recurrence_weekdays == [0, 2]


@pytest.mark.asyncio
async def test_parse_reminder_not_schedulable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "should_schedule": False,
                            "reminder_text": "",
                            "notification_text": "",
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

    async def fake_post(
        self, url: str, json: dict, headers: dict
    ) -> JsonHttpResponse:
        _ = (url, json, headers)
        return JsonHttpResponse(payload)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    parser = _parser()
    parsed = await parser.parse_reminder("hello", datetime.now(UTC))
    assert parsed is None


def test_loads_json_object_from_fenced_json() -> None:
    text = (
        "```json\n"
        '{"should_schedule":false,"reminder_text":"",'
        '"notification_text":"","remind_at_utc":"",'
        '"is_recurring":false,"recurrence_unit":"none",'
        '"recurrence_interval":0,"recurrence_weekdays":[]}\n'
        "```"
    )
    result = LLMReminderParser._loads_json_object(text)
    assert result["should_schedule"] is False


@pytest.mark.asyncio
async def test_parse_reminder_llm_service_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post(
        self, url: str, json: dict, headers: dict
    ) -> JsonHttpResponse:
        _ = (self, url, json, headers)
        raise httpx.ConnectError("down")

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    parser = _parser()

    with pytest.raises(GeminiServiceError):
        await parser.parse_reminder("Please remind me", datetime.now(UTC))


@pytest.mark.asyncio
async def test_parse_delete_request_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "should_delete": True,
                            "delete_all": False,
                            "target_text": "Alexei",
                        }
                    )
                }
            }
        ]
    }

    async def fake_post(
        self, url: str, json: dict, headers: dict
    ) -> JsonHttpResponse:
        _ = (self, url, json, headers)
        return JsonHttpResponse(payload)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    parser = _parser()
    parsed = await parser.parse_delete_request("Delete reminder about Alexei")
    assert parsed is not None
    assert parsed.delete_all is False
    assert parsed.target_text == "Alexei"


@pytest.mark.asyncio
async def test_parse_reminder_recurring_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_utc = datetime.now(UTC)
    remind_at = (
        (now_utc + timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    )
    # Recurring rows require positive recurrence_interval from the model.
    payload = gemini_choices_response(
        {
            "should_schedule": True,
            "reminder_text": "check status",
            "notification_text": "Reminder: check status",
            "remind_at_utc": remind_at,
            "is_recurring": True,
            "recurrence_unit": "week",
            "recurrence_interval": 1,
            "recurrence_weekdays": [8, "x", 2],
        }
    )

    async def fake_post(
        self, url: str, json: dict, headers: dict
    ) -> JsonHttpResponse:
        _ = (self, url, json, headers)
        return JsonHttpResponse(payload)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    parser = _parser()
    parsed = await parser.parse_reminder(
        "Every Wednesday check status",
        now_utc,
    )

    assert parsed is not None
    assert parsed.is_recurring is True
    assert parsed.recurrence_unit == "week"
    assert parsed.recurrence_interval == 1
    assert parsed.recurrence_weekdays == [2]


def test_parser_helpers() -> None:
    assert LLMReminderParser._positive_int_or_none(2) == 2
    assert LLMReminderParser._positive_int_or_none(0) is None
    assert LLMReminderParser._positive_int_or_none("bad") is None
    assert LLMReminderParser._normalize_weekdays([0, "2", 9, "x", 0]) == [0, 2]
    assert LLMReminderParser._normalize_weekdays("bad") is None
    assert LLMReminderParser._normalize_action(" store ") == "store"
    assert LLMReminderParser._normalize_action("DELETE") == "delete"
    assert LLMReminderParser._normalize_action("nope") == "none"


def test_extract_gemini_text_from_parts() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": '{"ok":true}'},
                        {"type": "image", "text": "ignored"},
                    ]
                }
            }
        ]
    }
    assert extract_gemini_text(body) == '{"ok":true}'


@pytest.mark.asyncio
async def test_parse_reminder_prompt_uses_command_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_prompt: dict[str, str] = {}
    now_utc = datetime.now(UTC)
    remind_at = (
        (now_utc + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    )
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "should_schedule": True,
                            "reminder_text": "посмотреть в окно",
                            "notification_text": (
                                "Напоминание: посмотри в окно."
                            ),
                            "remind_at_utc": remind_at,
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

    async def fake_post(
        self, url: str, json: dict, headers: dict
    ) -> JsonHttpResponse:
        _ = (self, url, headers)
        captured_prompt["system"] = json["system_instruction"]["parts"][0][
            "text"
        ]
        return JsonHttpResponse(payload)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    parser = _parser()
    await parser.parse_reminder("напомни посмотреть в окно", now_utc)

    system_prompt = captured_prompt["system"]
    assert "Allowed action values: store, delete, none." in system_prompt
    assert "Do not include examples in your response." in system_prompt


@pytest.mark.asyncio
async def test_parse_reminder_recurring_with_llm_first_fire_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_utc = datetime.now(UTC)
    first_fire = (
        (now_utc + timedelta(minutes=3))
        .isoformat()
        .replace("+00:00", "Z")
    )
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "should_schedule": True,
                            "reminder_text": "посмотреть в окно",
                            "notification_text": (
                                "Напоминание: посмотри в окно."
                            ),
                            "remind_at_utc": first_fire,
                            "is_recurring": True,
                            "recurrence_unit": "minute",
                            "recurrence_interval": 2,
                            "recurrence_weekdays": [],
                        }
                    )
                }
            }
        ]
    }

    async def fake_post(
        self, url: str, json: dict, headers: dict
    ) -> JsonHttpResponse:
        _ = (self, url, json, headers)
        return JsonHttpResponse(payload)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    parser = _parser()
    parsed = await parser.parse_reminder(
        "Напоминай каждые 2 минуты посмотреть в окно",
        now_utc,
    )

    assert parsed is not None
    assert parsed.is_recurring is True
    assert parsed.recurrence_unit == "minute"
    assert parsed.recurrence_interval == 2
    assert parsed.remind_at_utc > now_utc


@pytest.mark.asyncio
async def test_parse_reminder_uses_notification_text_from_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_utc = datetime.now(UTC)
    remind_at = (
        (now_utc + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    )
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "should_schedule": True,
                            "reminder_text": "встать и размяться",
                            "notification_text": (
                                "Напоминание: посмотри в окно."
                            ),
                            "remind_at_utc": remind_at,
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

    async def fake_post(
        self, url: str, json: dict, headers: dict
    ) -> JsonHttpResponse:
        _ = (self, url, json, headers)
        return JsonHttpResponse(payload)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    parser = _parser()
    parsed = await parser.parse_reminder("напомни встать", now_utc)

    assert parsed is not None
    assert parsed.notification_text == "Напоминание: посмотри в окно."


@pytest.mark.asyncio
async def test_parse_reminder_monthly_from_llm_first_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_utc = datetime.now(UTC)
    remind_at = (
        (now_utc + timedelta(days=10))
        .replace(day=17, hour=9, minute=0, second=0, microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "should_schedule": True,
                            "reminder_text": "заплатить за квартиру",
                            "notification_text": "Напоминание: заплати за квартиру.",
                            "remind_at_utc": remind_at,
                            "is_recurring": True,
                            "recurrence_unit": "month",
                            "recurrence_interval": 1,
                            "recurrence_weekdays": [],
                        }
                    )
                }
            }
        ]
    }

    async def fake_post(
        self, url: str, json: dict, headers: dict
    ) -> JsonHttpResponse:
        _ = (self, url, json, headers)
        return JsonHttpResponse(payload)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    parser = _parser()
    parsed = await parser.parse_reminder(
        "Напоминай каждый месяц третьего числа заплатить за квартиру",
        now_utc,
    )

    assert parsed is not None
    assert parsed.is_recurring is True
    assert parsed.recurrence_unit == "month"
    assert parsed.recurrence_interval == 1
    local_dt = parsed.remind_at_utc.astimezone(UTC)
    assert local_dt.day == 17
    assert parsed.remind_at_utc > now_utc


@pytest.mark.asyncio
async def test_parse_command_single_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``parse_command`` must issue exactly one generateContent request."""
    posts: list[int] = []

    async def fake_post(
        self, url: str, json: dict, headers: dict
    ) -> JsonHttpResponse:
        posts.append(1)
        return JsonHttpResponse(
            gemini_choices_response(
                {
                    "should_schedule": True,
                    "should_delete": False,
                    "reminder_text": "a",
                    "notification_text": "Notification: a",
                    "remind_at_utc": (
                        datetime.now(UTC) + timedelta(hours=1)
                    ).isoformat().replace("+00:00", "Z"),
                    "is_recurring": False,
                    "recurrence_unit": "none",
                    "recurrence_interval": 0,
                    "recurrence_weekdays": [],
                }
            )
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    parser = _parser()
    out = await parser.parse_command("remind me a", datetime.now(UTC))
    assert isinstance(out, CommandParseResult)
    assert out.delete is None
    assert out.reminder is not None
    assert len(posts) == 1
