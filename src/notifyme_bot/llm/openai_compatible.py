from __future__ import annotations

import json

import httpx

from notifyme_bot.llm.errors import LLMServiceError
from notifyme_bot.llm.json_utils import loads_json_object


class OpenAICompatibleStructuredChatProvider:
    """OpenAI-style ``/chat/completions`` with JSON schema."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
    ) -> None:
        self._api_key = api_key
        self._model_name = str(model).strip()
        if not self._model_name:
            raise ValueError("Model name must be non-empty.")
        base = str(base_url).strip().rstrip("/")
        if not base:
            raise ValueError("base_url must be non-empty.")
        self._base_url = base

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict,
    ) -> dict:
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "notifyme_command",
                    "strict": True,
                    "schema": response_schema,
                },
            },
        }
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                body = response.json()
                raw = body["choices"][0]["message"]["content"]
                if isinstance(raw, list):
                    text = "".join(
                        part.get("text", "")
                        for part in raw
                        if isinstance(part, dict)
                        and part.get("type") == "text"
                    )
                else:
                    text = str(raw)
                return loads_json_object(text)
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
                    "OpenAI-compatible request failed "
                    f"(status={status}, "
                    f"model={self._model_name}, "
                    f"base_url={self._base_url}, "
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
            "OpenAI-compatible request failed "
            f"(model={self._model_name}, base_url={self._base_url})"
        ) from last_error
