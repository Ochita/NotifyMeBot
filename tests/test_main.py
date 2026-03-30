from __future__ import annotations

import logging

from notifyme_bot.main import configure_logging, main


class _FakeApp:
    def __init__(self) -> None:
        self.called = False

    def run_polling(self, allowed_updates: object) -> None:
        _ = allowed_updates
        self.called = True


def test_configure_logging_sets_httpx_warning() -> None:
    configure_logging()
    assert logging.getLogger("httpx").level == logging.WARNING


def test_main_uses_config_argument(monkeypatch) -> None:
    fake_app = _FakeApp()

    def fake_load_settings(config_path: str):
        assert config_path == "custom.yaml"
        return object()

    def fake_build_application(settings: object) -> _FakeApp:
        _ = settings
        return fake_app

    monkeypatch.setattr(
        "notifyme_bot.main.load_settings",
        fake_load_settings,
    )
    monkeypatch.setattr(
        "notifyme_bot.main.build_application",
        fake_build_application,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--config", "custom.yaml"],
    )

    main()
    assert fake_app.called is True
