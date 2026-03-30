from __future__ import annotations

from typing import Protocol


class StructuredChatProvider(Protocol):
    """Async structured JSON from a chat model (provider-specific HTTP)."""

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict,
    ) -> dict:
        """Return a JSON object matching ``response_schema``."""
        ...
