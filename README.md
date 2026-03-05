# Chatalogue

![Chatalogue Logo](frontend/public/chatalogue-logo.svg)

Chatalogue is a local-first YouTube analysis workstation for long-form video/podcast workflows:
- channel ingest + refresh
- transcription + diarization pipeline
- channel-level transcript search
- speaker profile management/merging
- funny-moment detection + explanation
- AI summary + chapter generation
- clip editing/export/upload flows

## Stack
- Backend: FastAPI + SQLModel + Uvicorn
- Frontend: React + Vite + TypeScript + Tailwind
- Database: embedded PostgreSQL runtime (auto-managed binaries)
- ASR engines: Whisper + NVIDIA Parakeet (configurable/fallback)
- Diarization: pyannote.audio
- LLM providers: local Ollama + remote API providers
- Media tooling: yt-dlp + ffmpeg

## Prerequisites
- Python 3.10+
- Node.js 18+
- `ffmpeg` on PATH (recommended)
- Optional: NVIDIA GPU for fastest transcription/diarization

## Quick Start

### Windows
From repo root:

```bat
run.bat
```

`run.bat` now auto-runs `install_windows.bat` if dependencies are missing, waits for backend readiness, then starts frontend.

### macOS
From repo root:

```bash
chmod +x run_mac.sh install_mac.sh
./run_mac.sh
```

`run_mac.sh` now auto-runs `install_mac.sh` when venv or frontend dependencies are missing.

### App URLs
- Frontend: `http://localhost:5173`
- Backend API docs: `http://localhost:8011/docs`

## Install Scripts
- `install_windows.bat`: creates backend venv, installs backend/frontend deps, optional Parakeet deps, optional model preloading.
- `install_mac.sh`: same flow for macOS.

Installer env vars:
- `INSTALL_PARAKEET=1|0` (default `1`)
- `SKIP_MODEL_PRELOAD=1|0` (default `0`)
- `PRELOAD_ENGINE=auto|whisper|parakeet` (default `auto`)
- `OLLAMA_MODELS="model1 model2 ..."` (optional Ollama pulls)

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

## Startup Scripts
- `run.bat`: full Windows startup (backend + frontend, readiness-gated)
- `run_mac.sh`: full macOS startup (backend + frontend, readiness-gated)
- `run_frontend.bat`: frontend-only startup
- `backend/run_worker.bat`: backend debug worker helper

## Configuration
Copy `backend/.env.example` to `backend/.env` and set keys as needed.

Common keys:
- `HF_TOKEN`
- `TRANSCRIPTION_ENGINE=auto|whisper|parakeet`
- `PARAKEET_MODEL` (default `nvidia/parakeet-tdt-0.6b-v2`)
- `DB_PROVIDER=postgres`
- `DATABASE_URL` (optional explicit DB URL)

## Embedded PostgreSQL Defaults
- Data: `backend/data/postgres`
- Binaries cache: `backend/bin/postgres`
- Default DSN: `postgresql+psycopg://chatalogue@127.0.0.1:55432/chatalogue`

## Brand
- Primary: `#FF5252`
- Secondary: `#ED3B3B`
- Accent: `#FF8A8A`
- Surface tint: `#FFF1F1`

## Troubleshooting
- `No module named 'pkg_resources'`:
  - `pip install "setuptools<81"` in backend venv.
- yt-dlp age/cookie errors:
  - configure YouTube cookies per yt-dlp docs.
- Backend unreachable:
  - verify `http://localhost:8011/system/worker-status`.

## License
MIT (`LICENSE`).
