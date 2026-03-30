from __future__ import annotations

from notifyme_bot.config import Settings
from notifyme_bot.llm.gemini import GeminiStructuredChatProvider
from notifyme_bot.llm.openai_compatible import (
    OpenAICompatibleStructuredChatProvider,
)
from notifyme_bot.llm.protocol import StructuredChatProvider

_DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"


def llm_provider_from_settings(settings: Settings) -> StructuredChatProvider:
    """Build the configured provider (Gemini or OpenAI-compatible)."""
    kind = settings.llm_provider.strip().lower()
    if kind == "gemini":
        return GeminiStructuredChatProvider(
            api_key=settings.api_key,
            model=settings.model_name,
        )
    if kind == "openai_compatible":
        base = (settings.llm_base_url or _DEFAULT_OPENAI_BASE).strip().rstrip(
            "/"
        )
        return OpenAICompatibleStructuredChatProvider(
            api_key=settings.api_key,
            model=settings.model_name,
            base_url=base,
        )
    raise ValueError(
        f"Unsupported llm_provider: {settings.llm_provider!r}. "
        "Use 'gemini' or 'openai_compatible'."
    )
