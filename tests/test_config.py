from __future__ import annotations

from pathlib import Path

import pytest

from notifyme_bot.config import load_settings


def test_load_settings_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        "\n".join(
            [
                'telegram_bot_token: "tg-token"',
                'gemini_api_key: "gm-key"',
                'gemini_model: "gemini-2.5-flash"',
                'database_path: "db.sqlite3"',
                'default_timezone: "Europe/Kyiv"',
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(str(config_file))

    assert settings.telegram_bot_token == "tg-token"
    assert settings.gemini_api_key == "gm-key"
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.database_path == "db.sqlite3"
    assert settings.default_timezone == "Europe/Kyiv"
    assert settings.allow_new_users is True


def test_load_settings_allow_new_users_false(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        "\n".join(
            [
                'telegram_bot_token: "tg-token"',
                'gemini_api_key: "gm-key"',
                'gemini_model: "gemini-2.5-flash"',
                "allow_new_users: false",
            ]
        ),
        encoding="utf-8",
    )
    settings = load_settings(str(config_file))
    assert settings.allow_new_users is False


def test_load_settings_allow_new_users_non_bool(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        "\n".join(
            [
                'telegram_bot_token: "tg-token"',
                'gemini_api_key: "gm-key"',
                'gemini_model: "gemini-2.5-flash"',
                'allow_new_users: "no"',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="allow_new_users must be"):
        load_settings(str(config_file))


def test_load_settings_rejects_gemini_models_key(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        "\n".join(
            [
                'telegram_bot_token: "tg-token"',
                'gemini_api_key: "gm-key"',
                "gemini_models:",
                '  - "gemini-2.5-flash"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="gemini_models is no longer"):
        load_settings(str(config_file))


def test_load_settings_missing_file() -> None:
    with pytest.raises(ValueError, match="Missing config file"):
        load_settings("does-not-exist.yaml")


def test_load_settings_missing_required_values(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text('telegram_bot_token: "ok"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="gemini_api_key"):
        load_settings(str(config_file))


def test_load_settings_missing_token(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text('gemini_api_key: "ok"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="telegram_bot_token"):
        load_settings(str(config_file))


def test_load_settings_rejects_gemini_model_as_list(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        "\n".join(
            [
                'telegram_bot_token: "tg-token"',
                'gemini_api_key: "gm-key"',
                "gemini_model:",
                '  - "gemini-2.5-flash"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="single string"):
        load_settings(str(config_file))


def test_load_settings_rejects_empty_gemini_model(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        "\n".join(
            [
                'telegram_bot_token: "tg-token"',
                'gemini_api_key: "gm-key"',
                'gemini_model: ""',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing gemini_model"):
        load_settings(str(config_file))
