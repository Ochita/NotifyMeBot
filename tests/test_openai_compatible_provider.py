"""Smoke tests for OpenAI-compatible HTTP wiring."""

from __future__ import annotations

import json as json_stdlib

import httpx
import pytest

from notifyme_bot.llm.openai_compatible import (
    OpenAICompatibleStructuredChatProvider,
)


class _OkResponse:
    def __init__(self, body: dict) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


@pytest.mark.asyncio
async def test_chat_json_posts_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    async def fake_post(
        self,
        url: str,
        json: dict,
        headers: dict,
    ) -> _OkResponse:
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _OkResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json_stdlib.dumps({"action": "none"}),
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    prov = OpenAICompatibleStructuredChatProvider(
        api_key="sk",
        model="gpt-4o-mini",
        base_url="https://api.example.com/v1",
    )
    schema = {"type": "object", "properties": {"action": {"type": "string"}}}
    out = await prov.chat_json("sys", "user", schema)
    assert out == {"action": "none"}
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk"
    assert captured["json"]["model"] == "gpt-4o-mini"
    assert captured["json"]["response_format"]["type"] == "json_schema"
