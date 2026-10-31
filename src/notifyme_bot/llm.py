from __future__ import annotations

import asyncio
from io import BytesIO
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

#: Structured-response model the endpoint must fill in.
StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


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

    async def chat_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredModel],
    ) -> StructuredModel:
        """One completion whose answer is validated into ``response_model``.

        The pydantic model *is* the schema: the SDK converts it to a strict
        JSON schema and parses the reply back, so the prompt and the contract
        cannot drift apart.
        """
        try:
            completion = await self._client.chat.completions.parse(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format=response_model,
                # Cheapest upstream that accepts response_format.
                extra_body={
                    "provider": {"sort": "price", "require_parameters": True}
                },
            )
            return _parsed(completion, response_model)
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
    mime = (mime_type or "").split(";", 1)[0].strip().lower() or "audio/ogg"
    ext = "mp3" if mime == "audio/mpeg" else mime.rsplit("/", 1)[-1]
    return (f"voice.{ext}", BytesIO(audio), mime)


def _parsed(
    completion: object, response_model: type[StructuredModel]
) -> StructuredModel:
    """Pull the validated object out, or retry the text as a last resort.

    Routers that ignore ``response_format`` still answer with the JSON body in
    ``content``; only the parsed object is missing. Anything else is an error
    the caller has to see.
    """
    message = completion.choices[0].message  # type: ignore[attr-defined]
    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, response_model):
        return parsed
    text = str(getattr(message, "content", "") or "").strip()
    if not text:
        raise ValueError("empty structured response")
    return response_model.model_validate_json(text)
