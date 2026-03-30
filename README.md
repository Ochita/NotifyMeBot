# NotifyMeBot

Telegram reminder bot powered by Google Gemini + SQLite.

## What it does

- User subscribes with `/start`
- User sends natural language reminder text
- User can schedule recurring reminders (for example: every Monday,
  every 2 days, every week, every Wednesday and Saturday)
- User can also delete reminders in natural language
- Bot asks Gemini LLM to extract:
  - reminder content
  - UTC due datetime
- Bot stores reminder in SQLite
- Bot schedules one-time notification
- At due time, bot sends:
  - language-aware `notification_text` produced by LLM
- Bot deletes the reminder after sending
- If Gemini is unavailable, bot returns a clear retry message
- System messages are localized (English, Russian, Italian, French,
  German, Ukrainian, Spanish)

## Project structure

```text
src/notifyme_bot/
  __main__.py
  main.py
  config.py
  models.py
  db.py
  llm_parser.py
  scheduler_service.py
  bot.py
```

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

## Run with Docker

1. Prepare Docker config file:

```bash
cp config/settings.docker.example.yaml config/settings.yaml
```

Edit `config/settings.yaml` and set:
- `telegram_bot_token`
- `gemini_api_key`
- optionally `default_timezone` and `gemini_model`

2. Build image:

```bash
docker build -t notifyme-bot:latest .
```

3. Run container:

```bash
docker run -d \
  --name notifyme-bot \
  --restart unless-stopped \
  -v "$(pwd)/config/settings.yaml:/run/secrets/notifyme/settings.yaml:ro" \
  -v notifyme_data:/app/data \
  notifyme-bot:latest
```

The SQLite database is persisted in Docker volume `notifyme_data`.

## Run with Docker Compose

```bash
docker compose up -d --build
```

Stop:

```bash
docker compose down
```

## Commands

- `/start` subscribe user
- `/stop` unsubscribe user
- `/set_timezone <Area/City>` set user timezone (example: `Europe/Rome`)
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

