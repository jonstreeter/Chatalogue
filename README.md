# Chatalogue

|  |  |
|---|---|
| <img src="frontend/public/chatalogue-logo.svg" width="44" alt="Chatalogue logo" /> | **Chatalogue**<br/>*Dialogue -> Data* |

Chatalogue is a local-first workstation for long-form spoken content:
- channel ingest + refresh
- transcription + diarization pipeline
- channel-level transcript search
- speaker profile management/merging
- funny-moment detection + explanation
- AI summaries + chapter generation
- clip editing/export/upload

## Installation (First)

### One-file bootstrap installers
Put one installer file into the folder where you want Chatalogue installed, then run it.

The installer will:
1. Check prerequisites (`git`, Python, Node, optional `ffmpeg`)
2. Clone/update the GitHub repo
3. Create backend `.venv`
4. Install backend + frontend dependencies
5. Install optional Parakeet dependencies
6. Preload transcription models

By default, repo is installed to `./Chatalogue` relative to the installer script location.

### Windows installer
- Direct `.bat`:  
  [install_windows.bat](./install_windows.bat)
- Zipped `.bat` (if browser/security policy blocks script download):  
  [install_windows.zip](./installers/install_windows.zip)

### macOS installer
- Direct `.sh`:  
  [install_mac.sh](./install_mac.sh)
- Zipped `.sh` (if needed):  
  [install_mac.zip](./installers/install_mac.zip)

### Optional installer environment variables
- `CHATALOGUE_REPO_URL` (default: `https://github.com/jonstreeter/Chatalogue.git`)
- `CHATALOGUE_REPO_BRANCH` (default: `main`)
- `CHATALOGUE_REPO_DIR` (default: `Chatalogue`)
- `PYTHON_BIN` (macOS/Linux installer only, default: `python3`)
- `INSTALL_PARAKEET=1|0` (default: `1`)
- `SKIP_MODEL_PRELOAD=1|0` (default: `0`)
- `PRELOAD_ENGINE=auto|whisper|parakeet` (default: `auto`)
- `OLLAMA_MODELS="model1 model2 ..."` (optional model pulls)

Examples:

Windows:
```bat
set INSTALL_PARAKEET=0
set SKIP_MODEL_PRELOAD=1
install_windows.bat
```

macOS:
```bash
INSTALL_PARAKEET=1 PRELOAD_ENGINE=auto OLLAMA_MODELS="qwen2.5:7b qwen3.5:27b" bash ./install_mac.sh
```

If you prefer running with `./install_mac.sh`, make it executable first:
```bash
chmod +x install_mac.sh
```

## Run

After install:

### Windows
```bat
Chatalogue\run.bat
```

### macOS
```bash
chmod +x Chatalogue/run_mac.sh
./Chatalogue/run_mac.sh
```

`run.bat` and `run_mac.sh` wait for backend readiness before launching frontend.

### App URLs
- Frontend: `http://localhost:5173`
- Backend API docs: `http://localhost:8011/docs`

## Stack
- Backend: FastAPI + SQLModel + Uvicorn
- Frontend: React + Vite + TypeScript + Tailwind
- Database: embedded PostgreSQL runtime (auto-managed binaries)
- ASR engines: Whisper + NVIDIA Parakeet (configurable/fallback)
- Diarization: pyannote.audio
- LLM providers: local Ollama + hosted APIs
- Media tooling: yt-dlp + ffmpeg

## Configuration
Copy `backend/.env.example` to `backend/.env` and set keys as needed.

Common keys:
- `HF_TOKEN`
- `TRANSCRIPTION_ENGINE=auto|whisper|parakeet`
- `PARAKEET_MODEL` (default `nvidia/parakeet-tdt-0.6b-v2`)
- `PARAKEET_BATCH_SIZE` (requested batch size before auto-capping)
- `PARAKEET_OOM_CHUNK_RETRY=true|false`
- `PARAKEET_OOM_CHUNK_SECONDS` (default `600`)
- `PARAKEET_OOM_MIN_CHUNK_SECONDS` (default `30`)
- `PARAKEET_OOM_CHUNK_OVERLAP_SECONDS` (default `0.35`)
- `DB_PROVIDER=postgres`
- `DATABASE_URL` (optional explicit DB URL)

## Security
- Local secret scanning: `pre-commit` + `gitleaks` (`.pre-commit-config.yaml`)
- CI secret scanning: GitHub Actions workflow (`.github/workflows/gitleaks.yml`)

## Embedded PostgreSQL defaults
- Data: `backend/data/postgres`
- Binaries cache: `backend/bin/postgres`
- Default DSN: `postgresql+psycopg://chatalogue@127.0.0.1:55432/chatalogue`

## License
MIT (`LICENSE`).

