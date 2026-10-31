from __future__ import annotations

import asyncio
import json
from io import BytesIO

from openai import AsyncOpenAI


class LLMServiceError(RuntimeError):
    """Raised when an LLM request fails before a usable structured response."""


class OpenAICompatibleService:
    """Chat + speech-to-text on one OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        stt_model: str,
    ) -> None:
        self._model_name = str(model).strip()
        if not self._model_name:
            raise ValueError("Model name must be non-empty.")
        self._stt_model = str(stt_model).strip()
        if not self._stt_model:
            raise ValueError("stt_model must be non-empty.")
        base = str(base_url).strip().rstrip("/")
        if not base:
            raise ValueError("base_url must be non-empty.")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict,
        schema_name: str = "notifyme_command",
    ) -> dict:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": response_schema,
                    },
                },
                # Cheapest upstream that accepts response_format.
                extra_body={
                    "provider": {"sort": "price", "require_parameters": True}
                },
            )
            raw = completion.choices[0].message.content
            return _loads_json(raw)
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalized for callers
            raise LLMServiceError(
                f"Chat completion failed (model={self._model_name}): {exc}"
            ) from exc

    async def transcribe(
        self,
        audio: bytes,
        mime_type: str,
        language: str | None = None,
    ) -> str:
        """Speech to text; empty string when the model heard nothing."""
        kwargs: dict = {
            "model": self._stt_model,
            "file": _audio_file(audio, mime_type),
        }
        if language:
            kwargs["language"] = language
        try:
            result = await self._client.audio.transcriptions.create(**kwargs)
            return (getattr(result, "text", "") or "").strip()
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalized for callers
            raise LLMServiceError(f"Transcription failed: {exc}") from exc


def _audio_file(audio: bytes, mime_type: str) -> tuple[str, BytesIO, str]:
    mime = (mime_type or "audio/ogg").split(";", 1)[0].strip().lower()
    if not mime:
        mime = "audio/ogg"
    ext = "mp3" if mime == "audio/mpeg" else mime.rsplit("/", 1)[-1]
    return (f"voice.{ext}", BytesIO(audio), mime)


def _loads_json(raw: object) -> dict:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty response")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed
