from __future__ import annotations

import json

import httpx

from notifyme_bot.llm.errors import LLMServiceError


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
        url = f"{self._base_url}/chat/completions"
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
            # Cheapest upstream that accepts response_format.
            "provider": {
                "sort": "price",
                "require_parameters": True,
            },
        }
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
                return json.loads(str(raw))
            except httpx.HTTPStatusError as exc:
                error_body = exc.response.text[:400] if exc.response else ""
                status = (
                    exc.response.status_code if exc.response else "unknown"
                )
                raise LLMServiceError(
                    "OpenAI-compatible request failed "
                    f"(status={status}, "
                    f"model={self._model_name}, "
                    f"base_url={self._base_url}, "
                    f"body={error_body})"
                ) from exc
            except (
                httpx.HTTPError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                raise LLMServiceError(
                    "OpenAI-compatible request failed "
                    f"(model={self._model_name}, "
                    f"base_url={self._base_url})"
                ) from exc
