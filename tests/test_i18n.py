from __future__ import annotations

from notifyme_bot.i18n import detect_locale, translate


def test_translate_unknown_locale_falls_back_to_en() -> None:
    text = translate("xx", "start", default_timezone="UTC")
    assert "Subscription enabled" in text


def test_detect_locale_from_telegram_language_code() -> None:
    assert detect_locale("it-IT") == "it"
    assert detect_locale("fr") == "fr"


def test_detect_locale_ukrainian_letters() -> None:
    assert detect_locale(None, "Привіт їжак") == "uk"


def test_detect_locale_russian_cyrillic() -> None:
    assert detect_locale(None, "Напомни завтра") == "ru"


def test_detect_locale_italian_keywords() -> None:
    assert detect_locale(None, "Per favore ricordami domani") == "it"


def test_detect_locale_french_keywords() -> None:
    assert detect_locale(None, "Rappelle-moi demain") == "fr"


def test_detect_locale_german_keywords() -> None:
    assert detect_locale(None, "Erinner mich morgen bitte") == "de"


def test_detect_locale_spanish_keywords() -> None:
    assert detect_locale(None, "Recuérdame mañana por favor") == "es"


def test_detect_locale_defaults_to_en() -> None:
    assert detect_locale(None, "plain English text here") == "en"
