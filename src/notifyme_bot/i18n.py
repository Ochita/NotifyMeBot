from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULT_LOCALE = "en"
_MESSAGES_PATH = Path(__file__).with_name("locales") / "messages.yaml"


def _load_messages() -> dict[str, dict[str, str]]:
    with _MESSAGES_PATH.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return {
        str(locale): {str(key): str(value) for key, value in values.items()}
        for locale, values in data.items()
        if isinstance(values, dict)
    }


_MESSAGES = _load_messages()


def translate(locale: str, key: str, **kwargs: str) -> str:
    messages = _MESSAGES.get(locale, _MESSAGES[_DEFAULT_LOCALE])
    template = messages.get(key, _MESSAGES[_DEFAULT_LOCALE][key])
    return template.format(**kwargs)


def detect_locale(
    user_language_code: str | None,
    message_text: str = "",
) -> str:
    if user_language_code:
        base = user_language_code.lower().split("-", maxsplit=1)[0]
        if base in _MESSAGES:
            return base

    lowered = message_text.lower()
    if any(char in lowered for char in ("ї", "є", "і", "ґ")):
        return "uk"
    has_cyrillic = any(
        "а" <= char <= "я" or "А" <= char <= "Я" for char in message_text
    )
    if has_cyrillic:
        return "ru"
    if any(
        word in lowered
        for word in ("ricord", "promemoria", "per favore", "domani", "oggi")
    ):
        return "it"
    if any(
        word in lowered
        for word in ("rappelle", "rappel", "demain", "aujourd", "s'il te")
    ):
        return "fr"
    if any(
        word in lowered for word in ("erinner", "morgen", "heute", "bitte")
    ):
        return "de"
    if any(
        word in lowered
        for word in ("recu", "recorda", "mañana", "hoy", "por favor")
    ):
        return "es"
    return _DEFAULT_LOCALE
