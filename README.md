# Chatalogue

Chatalogue is a local-first YouTube channel intelligence app for long-form podcast/video workflows:
- ingest channels and keep episodes updated
- transcribe + diarize at scale
- search transcripts across episodes
- manage speaker profiles and merges
- detect/explain funny moments
- generate summary/chapters
- create and export clips

## Technology
- Backend: FastAPI, SQLModel, Uvicorn
- Frontend: React + Vite + TypeScript + Tailwind
- Database: embedded PostgreSQL runtime (auto-managed binaries) with SQLite migration support
- Transcription: modular ASR engine (`auto`, `whisper`, `parakeet`) with fallback logic
- Diarization: `pyannote.audio`
- LLM integrations: local (Ollama) and remote providers
- Media tooling: `yt-dlp`, `ffmpeg`

## Repository layout
- `backend/`: API, queue workers, ingestion/transcription/diarization pipeline, DB layer
- `frontend/`: web UI
- `install_windows.bat`: Windows install/bootstrap (venv + deps + model preload)
- `install_mac.sh`: macOS install/bootstrap (venv + deps + model preload)
- `run.bat`: Windows dev runner (starts backend + frontend)

## Prerequisites
- Python 3.10+
- Node.js 18+
- `ffmpeg` available on PATH
- Optional but recommended: NVIDIA GPU for faster transcription/diarization

## Windows installation
From repo root in `cmd.exe`:

```bat
install_windows.bat
```

Then run the app:

```bat
run.bat
```

App URLs:
- Frontend: `http://localhost:5173`
- Backend API/docs: `http://localhost:8011/docs`

## macOS installation
From repo root:

```bash
chmod +x install_mac.sh
./install_mac.sh
```

Start both backend + frontend together:

```bash
chmod +x run_mac.sh
./run_mac.sh
```

Or start manually:

```bash
cd backend
./.venv/bin/python -m uvicorn src.main:app --app-dir . --host 0.0.0.0 --port 8011
```

Start frontend (new terminal):

```bash
cd frontend
npm run dev
```

## Installer options
Both installers support optional environment variables.

- `INSTALL_PARAKEET=1|0`
  - Default: `1`
  - Installs optional NeMo Parakeet dependencies when enabled.
- `SKIP_MODEL_PRELOAD=1|0`
  - Default: `0`
  - When `0`, runs `backend/preload_models.py` after dependency install.
- `PRELOAD_ENGINE=auto|whisper|parakeet`
  - Default: `auto`
  - Controls which ASR model cache is preloaded.
- `OLLAMA_MODELS="<model1> <model2> ..."` (optional)
  - If set and `ollama` exists in PATH, installer will run `ollama pull` for each model.

Examples:

Windows:
```bat
set INSTALL_PARAKEET=0
set SKIP_MODEL_PRELOAD=1
install_windows.bat
```

macOS:
```bash
INSTALL_PARAKEET=1 PRELOAD_ENGINE=auto OLLAMA_MODELS="qwen2.5:7b qwen3.5:27b" ./install_mac.sh
```

## Configuration
Copy `backend/.env.example` to `backend/.env` and set values as needed.

Common variables:
- `HF_TOKEN`: required for gated pyannote models
- `TRANSCRIPTION_ENGINE`: `auto` (default), `whisper`, or `parakeet`
- `PARAKEET_MODEL`: default `nvidia/parakeet-tdt-0.6b-v2`
- `PARAKEET_BATCH_SIZE`: default `16`
- `PARAKEET_REQUIRE_WORD_TIMESTAMPS`: default `true`
- `DB_PROVIDER`: default `postgres`
- `DATABASE_URL`: explicit SQLAlchemy URL (if set, embedded bootstrap is bypassed)

## Embedded PostgreSQL defaults
- Data dir: `backend/data/postgres`
- Binary cache: `backend/bin/postgres`
- Default DSN: `postgresql+psycopg://chatalogue@127.0.0.1:55432/chatalogue`

Optional runtime flags:
- `EMBEDDED_PG_ENABLED`
- `EMBEDDED_PG_AUTO_DOWNLOAD`
- `EMBEDDED_PG_VERSION`
- `EMBEDDED_PG_HOST`
- `EMBEDDED_PG_PORT`
- `EMBEDDED_PG_USER`
- `EMBEDDED_PG_DATABASE`
- `EMBEDDED_PG_BIN_DIR`

## Troubleshooting
- `No module named 'pkg_resources'`
  - Ensure installer completed or run:
    - `pip install "setuptools<81"`
- `Sign in to confirm your age` from `yt-dlp`
  - Configure YouTube cookies per yt-dlp docs.
- Backend unreachable
  - Confirm backend is running on port `8011`, then retry frontend.

## Contributing
See `CONTRIBUTING.md`.

## License
MIT (`LICENSE`).
