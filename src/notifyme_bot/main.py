from __future__ import annotations

import argparse
import logging

from telegram import Update

from notifyme_bot.bot import build_application
from notifyme_bot.config import load_settings


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Run NotifyMe Telegram bot.",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    settings = load_settings(config_path=args.config)
    application = build_application(settings)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
