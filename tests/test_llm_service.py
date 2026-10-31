"""Tests for the OpenAI-SDK backed chat/transcription service."""

from __future__ import annotations

import json

import pytest
from conftest import FakeOpenAI

from notifyme_bot.llm import LLMServiceError, OpenAICompatibleService


def _service(fake: FakeOpenAI, **overrides: object) -> OpenAICompatibleService:
    kwargs: dict = {
        "api_key": "sk",
        "model": "gpt-4.1-nano",
        "base_url": "https://api.example.com/v1/",
        "stt_model": "voxtral-small-24b-2507-stt",
    }
    kwargs.update(overrides)
    service = OpenAICompatibleService(**kwargs)
    service._client = fake  # type: ignore[assignment]
    return service


def test_strips_trailing_slash_from_base_url() -> None:
    # Build a real client (no request is made) to inspect the parsed URL.
    service = OpenAICompatibleService(
        api_key="k",
        model="m",
        base_url="https://api.example.com/v1/",
        stt_model="stt",
    )
    assert str(service.client.base_url).rstrip("/") == (
        "https://api.example.com/v1"
    )


def test_rejects_empty_model_stt_and_base_url() -> None:
    with pytest.raises(ValueError, match="Model name"):
        OpenAICompatibleService(
            api_key="k",
            model=" ",
            base_url="https://x/v1",
            stt_model="stt",
        )
    with pytest.raises(ValueError, match="stt_model"):
        OpenAICompatibleService(
            api_key="k",
            model="m",
            base_url="https://x/v1",
            stt_model="",
        )
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleService(
            api_key="k",
            model="m",
            base_url=" ",
            stt_model="stt",
        )


@pytest.mark.asyncio
async def test_chat_json_sends_schema_and_provider_hint() -> None:
    fake = FakeOpenAI(content=json.dumps({"action": "none"}))
    service = _service(fake)
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}

    out = await service.chat_json("sys", "user", schema)

    assert out == {"action": "none"}
    sent = fake.captured_chat
    assert sent["model"] == "gpt-4.1-nano"
    assert sent["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["response_format"]["json_schema"]["schema"] == schema
    assert sent["extra_body"]["provider"] == {
        "sort": "price",
        "require_parameters": True,
    }


@pytest.mark.asyncio
async def test_chat_json_rejects_non_object_payload() -> None:
    fake = FakeOpenAI(content="[1, 2]")
    with pytest.raises(LLMServiceError, match="not a JSON object"):
        await _service(fake).chat_json("s", "u", {})


@pytest.mark.asyncio
async def test_chat_json_empty_content_raises() -> None:
    fake = FakeOpenAI(content="   ")
    with pytest.raises(LLMServiceError):
        await _service(fake).chat_json("s", "u", {})


@pytest.mark.asyncio
async def test_transcribe_passes_language_and_returns_text() -> None:
    fake = FakeOpenAI(transcript="  напомни позвонить  ")
    out = await _service(fake).transcribe(b"audio", "audio/ogg", language="ru")

    assert out == "напомни позвонить"
    sent = fake.captured_audio
    assert sent["model"] == "voxtral-small-24b-2507-stt"
    assert sent["language"] == "ru"
    name, stream, mime = sent["file"]
    assert name == "voice.ogg"
    assert mime == "audio/ogg"
    assert stream.read() == b"audio"


@pytest.mark.asyncio
async def test_transcribe_omits_language_when_unknown() -> None:
    fake = FakeOpenAI(transcript="hi")
    await _service(fake).transcribe(b"a", "audio/ogg")

    assert "language" not in fake.captured_audio


@pytest.mark.asyncio
async def test_transcribe_maps_mpeg_to_mp3() -> None:
    fake = FakeOpenAI(transcript="hi")
    await _service(fake).transcribe(b"a", "audio/mpeg")

    assert fake.captured_audio["file"][0] == "voice.mp3"


@pytest.mark.asyncio
async def test_transcribe_wraps_errors() -> None:
    async def boom(**kwargs: object) -> object:
        raise RuntimeError("no speech endpoint")

    fake = FakeOpenAI()
    fake.audio.transcriptions.create = boom  # type: ignore[method-assign]

    with pytest.raises(LLMServiceError, match="Transcription failed"):
        await _service(fake).transcribe(b"a", "audio/ogg")


@pytest.mark.asyncio
async def test_transcribe_does_not_swallow_cancellation() -> None:
    import asyncio

    async def cancelled(**kwargs: object) -> object:
        raise asyncio.CancelledError

    fake = FakeOpenAI()
    fake.audio.transcriptions.create = cancelled  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        await _service(fake).transcribe(b"a", "audio/ogg")
