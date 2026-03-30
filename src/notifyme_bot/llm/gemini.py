from __future__ import annotations

import json

import httpx

from notifyme_bot.llm.errors import LLMServiceError
from notifyme_bot.llm.json_utils import loads_json_object


class GeminiStructuredChatProvider:
    """Gemini ``generateContent`` with JSON schema output."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model_name = str(model).strip()
        if not self._model_name:
            raise ValueError("Gemini model name must be non-empty.")

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict,
    ) -> dict:
        base_payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseJsonSchema": response_schema,
            },
        }
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {**base_payload}
            try:
                response = await client.post(
                    "https://generativelanguage.googleapis.com/v1beta/"
                    f"models/{self._model_name}:generateContent",
                    json=payload,
                    headers={
                        "x-goog-api-key": self._api_key,
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                body = response.json()
                return loads_json_object(extract_gemini_text(body))
            except httpx.HTTPStatusError as exc:
                error_body = (
                    exc.response.text[:400] if exc.response else ""
                )
                status = (
                    exc.response.status_code
                    if exc.response
                    else "unknown"
                )
                last_error = LLMServiceError(
                    "Gemini request failed "
                    f"(status={status}, "
                    f"model={self._model_name}, "
                    f"body={error_body})"
                )
            except (
                httpx.HTTPError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                last_error = exc

        raise LLMServiceError(
            f"Gemini request failed (model={self._model_name})"
        ) from last_error


def extract_gemini_text(body: dict) -> str:
    if "candidates" in body:
        parts = body["candidates"][0]["content"]["parts"]
        if isinstance(parts, list):
            return "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict)
            )
        return str(parts)

    # Backward-compatible fallback for legacy test fixtures.
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)
