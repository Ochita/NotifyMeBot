"""LLM provider abstractions (Gemini, OpenAI-compatible APIs, etc.)."""

from notifyme_bot.llm.errors import LLMServiceError
from notifyme_bot.llm.factory import llm_provider_from_settings
from notifyme_bot.llm.protocol import StructuredChatProvider

__all__ = [
    "LLMServiceError",
    "StructuredChatProvider",
    "llm_provider_from_settings",
]
