from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class Settings:
    telegram_bot_token: str
    gemini_api_key: str
    gemini_model: str
    database_path: str
    default_timezone: str
    allow_new_users: bool


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
    gemini_key = _required(data, "gemini_api_key")
    if not token:
        raise ValueError("Missing telegram_bot_token in YAML config.")
    if not gemini_key:
        raise ValueError("Missing gemini_api_key in YAML config.")

    if data.get("gemini_models") is not None:
        raise ValueError(
            "gemini_models is no longer supported; set gemini_model to a "
            "single model id instead."
        )

    gemini_model = _gemini_model(data)
    allow_new_users = data.get("allow_new_users", True)
    if not isinstance(allow_new_users, bool):
        raise ValueError("allow_new_users must be true or false in YAML config.")
    return Settings(
        telegram_bot_token=token,
        gemini_api_key=gemini_key,
        gemini_model=gemini_model,
        database_path=str(data.get("database_path", "notifyme.sqlite3")),
        default_timezone=str(data.get("default_timezone", "UTC")),
        allow_new_users=allow_new_users,
    )


def _required(data: dict, key: str) -> str:
    value = data.get(key, "")
    return str(value).strip()


def _gemini_model(data: dict) -> str:
    raw = data.get("gemini_model")
    if raw is None:
        raw = "gemini-2.5-flash"
    if isinstance(raw, list):
        raise ValueError(
            "gemini_model must be a single string, not a list. "
            "Remove gemini_models and set gemini_model only."
        )
    model = str(raw).strip()
    if not model:
        raise ValueError("Missing gemini_model in YAML config.")
    return model
