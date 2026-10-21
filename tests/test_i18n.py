from __future__ import annotations

from notifyme_bot.i18n import detect_locale, translate


def test_translate_unknown_locale_falls_back_to_en() -> None:
    text = translate("xx", "start", default_timezone="UTC")
    assert "Subscription enabled" in text


def test_detect_locale_from_telegram_language_code() -> None:
    assert detect_locale("it-IT") == "it"
    assert detect_locale("fr") == "fr"
    assert detect_locale(None) == "en"
    assert detect_locale("zh") == "en"
