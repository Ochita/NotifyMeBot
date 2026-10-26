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
                'api_key: "gm-key"',
                'model_name: "gpt-4.1-nano"',
                'database_path: "db.sqlite3"',
                'default_timezone: "Europe/Kyiv"',
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(str(config_file))

    assert settings.telegram_bot_token == "tg-token"
    assert settings.api_key == "gm-key"
    assert settings.model_name == "gpt-4.1-nano"
    assert settings.database_path == "db.sqlite3"
    assert settings.default_timezone == "Europe/Kyiv"
    assert settings.allow_new_users is True
    assert settings.base_url is None


def test_load_settings_allow_new_users_false(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        "\n".join(
            [
                'telegram_bot_token: "tg-token"',
                'api_key: "gm-key"',
                'model_name: "gpt-4.1-nano"',
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
                'api_key: "gm-key"',
                'model_name: "gpt-4.1-nano"',
                'allow_new_users: "no"',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="allow_new_users must be"):
        load_settings(str(config_file))


def test_load_settings_missing_file() -> None:
    with pytest.raises(ValueError, match="Missing config file"):
        load_settings("does-not-exist.yaml")


def test_load_settings_missing_required_values(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text('telegram_bot_token: "ok"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="api_key"):
        load_settings(str(config_file))


def test_load_settings_missing_token(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text('api_key: "ok"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="telegram_bot_token"):
        load_settings(str(config_file))


def test_load_settings_rejects_model_name_as_list(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        "\n".join(
            [
                'telegram_bot_token: "tg-token"',
                'api_key: "gm-key"',
                "model_name:",
                '  - "gpt-4.1-nano"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="single non-empty string"):
        load_settings(str(config_file))


def test_load_settings_base_url_override(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        "\n".join(
            [
                'telegram_bot_token: "tg-token"',
                'base_url: "https://ru-api.aitunnel.ru/v1"',
                'api_key: "sk-test"',
                'model_name: "gpt-4.1-nano"',
            ]
        ),
        encoding="utf-8",
    )
    settings = load_settings(str(config_file))
    assert settings.model_name == "gpt-4.1-nano"
    assert settings.base_url == "https://ru-api.aitunnel.ru/v1"


def test_load_settings_defaults_model_name(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        'telegram_bot_token: "tg-token"\napi_key: "sk-test"\n',
        encoding="utf-8",
    )
    settings = load_settings(str(config_file))
    assert settings.model_name == "gpt-4.1-nano"
    assert settings.base_url is None
    assert settings.default_locale == "en"


def test_load_settings_default_locale(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        "\n".join(
            [
                'telegram_bot_token: "tg-token"',
                'api_key: "sk-test"',
                'default_locale: "RU"',
            ]
        ),
        encoding="utf-8",
    )
    assert load_settings(str(config_file)).default_locale == "ru"

    config_file.write_text(
        "\n".join(
            [
                'telegram_bot_token: "tg-token"',
                'api_key: "sk-test"',
                'default_locale: "zz"',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="default_locale must be one of"):
        load_settings(str(config_file))


def test_load_settings_rejects_empty_model_name(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        "\n".join(
            [
                'telegram_bot_token: "tg-token"',
                'api_key: "gm-key"',
                'model_name: ""',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="single non-empty string"):
        load_settings(str(config_file))
