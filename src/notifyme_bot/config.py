from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class Settings:
    telegram_bot_token: str
    api_key: str
    model_name: str
    database_path: str
    default_timezone: str
    allow_new_users: bool
    #: ``gemini`` or ``openai_compatible`` (OpenAI-compatible HTTP APIs).
    llm_provider: str
    #: API base for ``openai_compatible`` only (e.g. OpenRouter).
    llm_base_url: str | None


def load_settings(config_path: str = "config/settings.yaml") -> Settings:
    path = Path(config_path)
    if not path.exists():
        raise ValueError(
            f"Missing config file: {path}. "
            "Copy config/settings.example.yaml to config/settings.yaml."
        )

    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}

    token = _required(data, "telegram_bot_token")
    api_key = _required(data, "api_key")
    if not token:
        raise ValueError("Missing telegram_bot_token in YAML config.")
    if not api_key:
        raise ValueError("Missing api_key in YAML config.")

    model_name = _model_name(data)
    allow_new_users = data.get("allow_new_users", True)
    if not isinstance(allow_new_users, bool):
        raise ValueError(
            "allow_new_users must be true or false in YAML config."
        )
    llm_provider = _llm_provider(data)
    llm_base_url = _optional_str(data, "llm_base_url")
    return Settings(
        telegram_bot_token=token,
        api_key=api_key,
        model_name=model_name,
        database_path=str(data.get("database_path", "notifyme.sqlite3")),
        default_timezone=str(data.get("default_timezone", "UTC")),
        allow_new_users=allow_new_users,
        llm_provider=llm_provider,
        llm_base_url=llm_base_url,
    )


def _required(data: dict, key: str) -> str:
    value = data.get(key, "")
    return str(value).strip()


def _optional_str(data: dict, key: str) -> str | None:
    raw = data.get(key)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _llm_provider(data: dict) -> str:
    raw = data.get("llm_provider", "gemini")
    name = str(raw).strip().lower()
    if name in {"gemini", "openai_compatible"}:
        return name
    raise ValueError(
        "llm_provider must be 'gemini' or 'openai_compatible' in YAML config."
    )


def _model_name(data: dict) -> str:
    raw = data.get("model_name")
    if raw is None:
        raw = "gemini-2.5-flash"
    if isinstance(raw, list):
        raise ValueError("model_name must be a single string, not a list.")
    model = str(raw).strip()
    if not model:
        raise ValueError("Missing model_name in YAML config.")
    return model
