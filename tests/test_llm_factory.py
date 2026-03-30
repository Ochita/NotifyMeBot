from __future__ import annotations

import dataclasses

import pytest

from notifyme_bot.config import Settings
from notifyme_bot.llm.factory import llm_provider_from_settings
from notifyme_bot.llm.gemini import GeminiStructuredChatProvider
from notifyme_bot.llm.openai_compatible import (
    OpenAICompatibleStructuredChatProvider,
)


def _base_settings(**kwargs: object) -> Settings:
    defaults = {
        "telegram_bot_token": "t",
        "api_key": "k",
        "model_name": "m",
        "database_path": "d.db",
        "default_timezone": "UTC",
        "allow_new_users": True,
        "llm_provider": "gemini",
        "llm_base_url": None,
    }
    defaults.update(kwargs)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_factory_returns_gemini() -> None:
    p = llm_provider_from_settings(_base_settings())
    assert isinstance(p, GeminiStructuredChatProvider)


def test_factory_returns_openai_compatible() -> None:
    p = llm_provider_from_settings(
        _base_settings(
            llm_provider="openai_compatible",
            llm_base_url="https://example.com/v1",
        )
    )
    assert isinstance(p, OpenAICompatibleStructuredChatProvider)


def test_factory_rejects_unknown_provider() -> None:
    bad = dataclasses.replace(
        _base_settings(),
        llm_provider="not-a-real-provider",
    )
    with pytest.raises(ValueError, match="Unsupported llm_provider"):
        llm_provider_from_settings(bad)
