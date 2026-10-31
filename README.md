# NotifyMeBot

Telegram reminder bot. Reminders are parsed by an OpenAI-compatible
chat model on [AITunnel](https://aitunnel.ru) (`gpt-4.1-nano` by default)
and stored in SQLite.

## What it does

- User subscribes with `/start`
- User sends natural language reminder text, or a voice message
  (transcribed by the configured speech-to-text model, then handled by the
  exact same pipeline)
- User can schedule recurring reminders (for example: every Monday,
  every 2 days, every week, every Wednesday and Saturday)
- User can also delete reminders in natural language
- Bot asks the configured LLM to extract the reminder content and the local
  wall-clock anchors (weekday, relative day, date, time); the code, not the
  model, resolves them to a UTC instant
- Bot stores reminder in SQLite
- Bot schedules one-time notification
- At due time, bot sends:
  - language-aware `notification_text` produced by LLM
- Bot deletes the reminder after sending
- If the LLM is unavailable, bot returns a clear retry message
- System messages are localized (English, Russian, Italian, French,
  German, Ukrainian, Spanish)

## Setup

1. Create and activate venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -e .
```

3. Configure YAML settings

```bash
cp config/settings.example.yaml config/settings.yaml
```

Edit `config/settings.yaml` with your secrets.

## Run

```bash
python -m notifyme_bot
```

Or use a custom config path:

```bash
python -m notifyme_bot --config /run/secrets/notifyme/settings.yaml
```

## Run with Docker Compose

```bash
cp config/settings.docker.example.yaml config/settings.yaml
```

Edit `config/settings.yaml` and set `telegram_bot_token`, `api_key`, and
optionally `base_url`, `model_name`, `stt_model`, `default_timezone`,
`default_locale` (`en`, `ru`, `uk`, `it`, `fr`, `de`, `es`). Then:

```bash
docker compose up -d --build
```

The SQLite database is persisted in Docker volume `notifyme_data`.

Stop:

```bash
docker compose down
```

## Commands

- `/start` subscribe user
- `/stop` unsubscribe user
- `/set_timezone <Area/City>` set user timezone (example: `Europe/Rome`)
- `/set_locale <code>` set your language (example: `ru`)
- `/list` list pending reminders
- `/help` show help
- Natural text like "cancel reminder about Alexei" also works

## Notes

- Datetimes are normalized to UTC in the database.
- Each user can override timezone via `/set_timezone`.
- If a message includes date but no time, parser assumes `09:00` in
  user timezone (or `default_timezone` if user timezone is not set).
- On restart, pending reminders are reloaded from SQLite and re-scheduled.
- Config is read from YAML via `yaml.safe_load` (no `.env` loading).

## Development checks

Install dev tools:

```bash
pip install -e ".[dev]"
```

Run linting:

```bash
ruff check .
ruff format --check .
```

Run tests with coverage:

```bash
pytest
```

