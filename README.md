# Chatalogue

A local indexing, transcription, and search system for YouTube channels.

## Features
- **Channel Tracking**: Monitor multiple YouTube channels for new content.
- **Efficient Archiving**: Downloads audio-only (m4a) to save space.
- **AI-Powered Analysis**:
  - **Transcription**: Modular engine (`auto | whisper | parakeet`) with automatic Whisper fallback.
  - **Diarization**: Speaker identification using `pyannote.audio`.
  - **Speaker Profiles**: Track unique speakers across videos and assign names/thumbnails.
- **Search**: Full-text search across all transcripts.
- **Clipping**: Download specific video segments on-demand without downloading the full video.
- **Database**: Embedded PostgreSQL (binary-managed, cross-platform) with automatic SQLite migration support.

## Setup

1. **Prerequisites**:
   - Python 3.10+
   - NVIDIA GPU (CUDA 11.8+ recommended)
   - [FFmpeg](https://ffmpeg.org/download.html) installed and in PATH.

2. **Installation**:
   ```bash
   run.bat
   ```
   *This script creates a virtual environment and installs all dependencies.*
   Optional Parakeet dependencies during install: set `INSTALL_PARAKEET=1` before running `run.bat`.

3. **Running**:
   The `run.bat` script will start the backend server automatically after installation.
   
   Access the API docs at: http://localhost:8011/docs

## Development (Ralph Loop)
This project follows the Ralph Loop agentic workflow.
- `prd.json`: Machine-readable product specs.
- `plan.md`: Implementation roadmap.
- `activity.md`: Development log.

## Environment Variables
Create a `.env` file in `backend/` if needed (e.g., for Hugging Face tokens).
- `HF_TOKEN`: Required for access to gated `pyannote` models.
- `TRANSCRIPTION_ENGINE`: `auto` (default), `whisper`, or `parakeet`.
- `PARAKEET_MODEL`: default `nvidia/parakeet-tdt-0.6b-v2`.
- `PARAKEET_BATCH_SIZE`: default `16`.
- `PARAKEET_REQUIRE_WORD_TIMESTAMPS`: `true` (default). If Parakeet output has poor word timing coverage, Chatalogue falls back to Whisper.
- `DB_PROVIDER`: `postgres` (default) or `sqlite`.
- `DATABASE_URL`: Optional explicit SQLAlchemy URL. If set, embedded DB bootstrap is skipped.
- `DB_MIGRATE_SQLITE_ON_START`: `true` (default) to auto-copy existing SQLite data into Postgres on first boot.

### Optional Parakeet install
Parakeet dependencies are optional so base installs remain stable across hosts.

```bash
python -m pip install -r backend/requirements-parakeet.txt
```

Optional cache warm-up (useful in installers):
```bash
python backend/preload_models.py --engine auto --whisper-model medium --parakeet-model nvidia/parakeet-tdt-0.6b-v2
```

### Embedded PostgreSQL (default)
By default, Chatalogue starts an embedded local PostgreSQL runtime and downloads platform-matching binaries on first run.

- Data path: `backend/data/postgres`
- Binary cache path: `backend/bin/postgres`
- Default connection: `postgresql+psycopg://chatalogue@127.0.0.1:55432/chatalogue`

Optional runtime variables:
- `EMBEDDED_PG_ENABLED=true`
- `EMBEDDED_PG_AUTO_DOWNLOAD=true`
- `EMBEDDED_PG_VERSION=17.6.0`
- `EMBEDDED_PG_HOST=127.0.0.1`
- `EMBEDDED_PG_PORT=55432`
- `EMBEDDED_PG_USER=chatalogue`
- `EMBEDDED_PG_DATABASE=chatalogue`
- `EMBEDDED_PG_BIN_DIR` (override binary folder if you bundle your own binaries)

### Manual migration command
Run once if you want an explicit migration command:
```bash
cd backend
python migrate_sqlite_to_postgres.py --force
```

Optional runtime control:
```bash
cd backend
python manage_embedded_postgres.py start
python manage_embedded_postgres.py stop
```
