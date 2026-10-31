"""Tests for the OpenAI-SDK backed chat/transcription service."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from conftest import FakeOpenAI, make_service
from pydantic import BaseModel

from notifyme_bot.llm import LLMServiceError, OpenAICompatibleService


class _Answer(BaseModel):
    """Stand-in structured response for schema/parsing assertions."""

    action: str = "none"


def test_strips_trailing_slash_from_base_url() -> None:
    # Build a real client (no request is made) to inspect the parsed URL.
    service = OpenAICompatibleService(
        api_key="k",
        model="m",
        base_url="https://api.example.com/v1/",
        stt_model="stt",
    )
    client = service._client
    assert str(client.base_url).rstrip("/") == "https://api.example.com/v1"


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
async def test_chat_structured_sends_model_and_provider_hint() -> None:
    fake = FakeOpenAI(content=json.dumps({"action": "none"}))
    service = make_service(fake)

    out = await service.chat_structured("sys", "user", _Answer)

    assert out == _Answer(action="none")
    sent = fake.captured_chat
    assert sent["model"] == "gpt-4.1-nano"
    assert sent["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    # The pydantic class is the schema; the SDK renders it as strict JSON.
    assert sent["response_format"] is _Answer
    assert sent["extra_body"]["provider"] == {
        "sort": "price",
        "require_parameters": True,
    }


@pytest.mark.asyncio
async def test_chat_structured_rejects_malformed_payload() -> None:
    fake = FakeOpenAI(content="[1, 2]")
    with pytest.raises(LLMServiceError, match="Chat completion failed"):
        await make_service(fake).chat_structured("s", "u", _Answer)


@pytest.mark.asyncio
async def test_chat_structured_empty_content_raises() -> None:
    fake = FakeOpenAI(content="   ")
    with pytest.raises(LLMServiceError, match="Chat completion failed"):
        await make_service(fake).chat_structured("s", "u", _Answer)


@pytest.mark.asyncio
async def test_chat_structured_falls_back_to_text_content() -> None:
    """Gateways that ignore response_format only fill ``content``."""

    class _NoParsed:
        async def parse(self, **kwargs: object) -> object:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            parsed=None, content='{"action": "store"}'
                        )
                    )
                ]
            )

    fake = FakeOpenAI()
    fake.chat.completions = _NoParsed()  # type: ignore[assignment]

    out = await make_service(fake).chat_structured("s", "u", _Answer)

    assert out == _Answer(action="store")


@pytest.mark.asyncio
async def test_transcribe_passes_language_and_returns_text() -> None:
    fake = FakeOpenAI(transcript="  напомни позвонить  ")
    out = await make_service(fake).transcribe(
        b"audio", "audio/ogg", language="ru"
    )

    assert out == "напомни позвонить"
    sent = fake.captured_audio
    assert sent["model"] == "voxtral-small-24b-2507-stt"
    assert sent["language"] == "ru"
    name, stream, mime = sent["file"]
    assert name == "voice.ogg"
    assert mime == "audio/ogg"
    assert stream.read() == b"audio"


@pytest.mark.asyncio
async def test_transcribe_defaults_to_ogg_when_mime_is_blank() -> None:
    """Telegram can report an empty content type; assume voice OGG."""
    fake = FakeOpenAI(transcript="hi")
    await make_service(fake).transcribe(b"a", "   ")

    name, _, mime = fake.captured_audio["file"]
    assert name == "voice.ogg"
    assert mime == "audio/ogg"


@pytest.mark.asyncio
async def test_transcribe_omits_language_when_unknown() -> None:
    fake = FakeOpenAI(transcript="hi")
    await make_service(fake).transcribe(b"a", "audio/ogg")

    assert "language" not in fake.captured_audio


@pytest.mark.asyncio
async def test_transcribe_maps_mpeg_to_mp3() -> None:
    fake = FakeOpenAI(transcript="hi")
    await make_service(fake).transcribe(b"a", "audio/mpeg")

    assert fake.captured_audio["file"][0] == "voice.mp3"


@pytest.mark.asyncio
async def test_transcribe_wraps_errors() -> None:
    async def boom(**kwargs: object) -> object:
        raise RuntimeError("no speech endpoint")

    fake = FakeOpenAI()
    fake.audio.transcriptions.create = boom  # type: ignore[method-assign]

    with pytest.raises(LLMServiceError, match="Transcription failed"):
        await make_service(fake).transcribe(b"a", "audio/ogg")


@pytest.mark.asyncio
async def test_transcribe_does_not_swallow_cancellation() -> None:
    import asyncio

    async def cancelled(**kwargs: object) -> object:
        raise asyncio.CancelledError

    fake = FakeOpenAI()
    fake.audio.transcriptions.create = cancelled  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        await make_service(fake).transcribe(b"a", "audio/ogg")
