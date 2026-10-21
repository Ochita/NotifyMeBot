from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from notifyme_bot.i18n import supported_locales

AITUNNEL_BASE_URL = "https://api.aitunnel.ru/v1"
DEFAULT_MODEL_NAME = "gpt-4.1-nano"


@dataclass(frozen=True, slots=True)
class Settings:
    telegram_bot_token: str
    #: Override for ``AITUNNEL_BASE_URL`` (mirror or tests).
    base_url: str | None
    api_key: str
    model_name: str
    database_path: str
    default_timezone: str
    default_locale: str
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
    base_url = _optional_str(data, "base_url")
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
    default_locale = _default_locale(data)
    return Settings(
        telegram_bot_token=token,
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        database_path=str(data.get("database_path", "notifyme.sqlite3")),
        default_timezone=str(data.get("default_timezone", "UTC")),
        default_locale=default_locale,
        allow_new_users=allow_new_users,
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


def _default_locale(data: dict) -> str:
    raw = data.get("default_locale", "en")
    locale = str(raw).strip().lower().split("-", maxsplit=1)[0]
    if locale not in supported_locales():
        known = ", ".join(sorted(supported_locales()))
        raise ValueError(f"default_locale must be one of: {known}.")
    return locale


def _model_name(data: dict) -> str:
    raw = data.get("model_name")
    if raw is None:
        raw = DEFAULT_MODEL_NAME
    if isinstance(raw, list):
        raise ValueError("model_name must be a single string, not a list.")
    model = str(raw).strip()
    if not model:
        raise ValueError("Missing model_name in YAML config.")
    return model
