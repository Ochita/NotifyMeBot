"""Unit tests for ``LLMReminderParser`` (LLM JSON is source of truth)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from chat_payloads import chat_completion_response
from http_fakes import JsonHttpResponse

from notifyme_bot.llm import (
    LLMServiceError,
    OpenAICompatibleStructuredChatProvider,
)
from notifyme_bot.llm_parser import LLMReminderParser
from notifyme_bot.models import CommandParseResult


def _parser() -> LLMReminderParser:
    return LLMReminderParser(
        provider=OpenAICompatibleStructuredChatProvider(
            api_key="test-key",
            model="gpt-4.1-nano",
            base_url="https://api.aitunnel.ru/v1",
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
        assert url == "https://api.aitunnel.ru/v1/chat/completions"
        assert headers["Authorization"] == "Bearer test-key"
        assert json["model"] == "gpt-4.1-nano"
        assert json["response_format"]["type"] == "json_schema"
        assert json["provider"]["sort"] == "price"
        return JsonHttpResponse(payload)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    parser = _parser()
    parsed = (
        await parser.parse_command("Please remind me tomorrow", now_utc)
    ).reminder

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
    parsed = (await parser.parse_command("hello", datetime.now(UTC))).reminder
    assert parsed is None


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

    with pytest.raises(LLMServiceError):
        await parser.parse_command("Please remind me", datetime.now(UTC))


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
                            "delete_ids": [10],
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
    parsed = (
        await parser.parse_command(
            "Delete reminder about Alexei", datetime.now(UTC)
        )
    ).delete
    assert parsed is not None
    assert parsed.delete_all is False
    assert parsed.delete_ids == [10]


@pytest.mark.asyncio
async def test_parse_reminder_recurring_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_utc = datetime.now(UTC)
    remind_at = (
        (now_utc + timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    )
    # Recurring rows require positive recurrence_interval from the model.
    payload = chat_completion_response(
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
    parsed = (
        await parser.parse_command(
            "Every Wednesday check status",
            now_utc,
        )
    ).reminder

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
        captured_prompt["system"] = json["messages"][0]["content"]
        return JsonHttpResponse(payload)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    parser = _parser()
    await parser.parse_command("напомни посмотреть в окно", now_utc)

    system_prompt = captured_prompt["system"]
    assert "Allowed action values: store, delete, none." in system_prompt
    assert "delete_ids" in system_prompt
    assert "Pending reminders" in system_prompt


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
    parsed = (
        await parser.parse_command(
            "Напоминай каждые 2 минуты посмотреть в окно",
            now_utc,
        )
    ).reminder

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
    parsed = (await parser.parse_command("напомни встать", now_utc)).reminder

    assert parsed is not None
    assert parsed.notification_text == "Напоминание: посмотри в окно."


@pytest.mark.asyncio
async def test_parse_reminder_monthly_from_llm_first_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_utc = datetime.now(UTC)
    # Day 17 of next month is always strictly after now.
    month = now_utc.month % 12 + 1
    year = now_utc.year + (1 if now_utc.month == 12 else 0)
    remind_at = (
        now_utc.replace(
            year=year,
            month=month,
            day=17,
            hour=9,
            minute=0,
            second=0,
            microsecond=0,
        )
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
                            "notification_text": (
                                "Напоминание: заплати за квартиру."
                            ),
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
    parsed = (
        await parser.parse_command(
            "Напоминай каждый месяц третьего числа заплатить за квартиру",
            now_utc,
        )
    ).reminder

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
    """``parse_command`` must issue exactly one chat completion."""
    posts: list[int] = []

    async def fake_post(
        self, url: str, json: dict, headers: dict
    ) -> JsonHttpResponse:
        posts.append(1)
        return JsonHttpResponse(
            chat_completion_response(
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
