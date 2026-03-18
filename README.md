# Chatalogue

|  |  |
|---|---|
| <img src="frontend/public/chatalogue-logo.svg" width="44" alt="Chatalogue logo" /> | **Chatalogue**<br/>*Dialogue -> Data* |

Chatalogue is a local-first workstation for ingesting, transcribing, diarizing, searching, and editing long-form spoken content.

## Core Features
- Channel ingest and automatic refreshing
- High-performance transcription and speaker diarization pipeline
- Channel-level transcript search
- Speaker profile management and merging
- Funny-moment detection and explanation
- AI-powered summaries and chapter generation
- Video clip editing, exporting, and uploading

---

## Quick Start (Installation)

Installers clone the repo, create the backend virtual environment, install dependencies, and preload the transcription stack.

### Prerequisites
Before installing, make sure your system has:
1. [Git](https://git-scm.com/downloads)
2. [Python 3.10+](https://www.python.org/downloads/)
   - On Windows, enable **Add Python to PATH** during install.
3. [Node.js 18+](https://nodejs.org/en/download/)

### Windows Installation
1. Download the Windows installer: [install_windows.zip](https://github.com/jonstreeter/Chatalogue/raw/main/installers/install_windows.zip)
2. Extract the zip into the folder where you want Chatalogue installed.
3. Open the extracted folder and run `install_windows.bat`.

### macOS Installation
1. Download the macOS installer: [install_mac.zip](https://github.com/jonstreeter/Chatalogue/raw/main/installers/install_mac.zip)
2. Extract the zip into the folder where you want Chatalogue installed.
3. Open Terminal, `cd` into the extracted folder, then run:

```bash
chmod +x install_mac.sh
./install_mac.sh
```

### Optional installer environment variables
Set these before running the installer if you need custom behavior:
- `INSTALL_PARAKEET=1|0` default `1`
- `SKIP_MODEL_PRELOAD=1|0` default `0`
- `PRELOAD_ENGINE=auto|whisper|parakeet` default `auto`
- `OLLAMA_MODELS="model1 model2 ..."`
- `CHATALOGUE_REPO_URL`
- `CHATALOGUE_REPO_BRANCH`

Example on Windows:

```bat
set INSTALL_PARAKEET=0
set SKIP_MODEL_PRELOAD=1
install_windows.bat
```

---

## Running the App

Once installation is complete, start Chatalogue at any time.

### Windows
Double-click `run_windows.bat` inside the `Chatalogue` folder.

### macOS
Open Terminal, `cd` into the `Chatalogue` folder, then run:

```bash
./run_mac.sh
```

The frontend opens at `http://localhost:5173`.
Backend API docs are available at `http://localhost:8011/docs`.

---

## Configuration

To override backend settings, copy `backend/.env.example` to `backend/.env` and adjust values as needed.

Common settings:
- `HF_TOKEN`: Hugging Face token for gated diarization models
- `TRANSCRIPTION_ENGINE`: `auto`, `whisper`, or `parakeet`
- `PARAKEET_MODEL`: default `nvidia/parakeet-tdt-0.6b-v2`
- `PARAKEET_ALLOW_WHISPER_FALLBACK`: `true` or `false`
- `PIPELINE_EXECUTION_MODE`: `sequential` or `staged`
- `DB_PROVIDER`: default `postgres`

---

## Technology Stack
- Backend: FastAPI, SQLModel, Uvicorn, embedded PostgreSQL
- Frontend: React, Vite, TypeScript, Tailwind
- AI/ML: Whisper, NVIDIA Parakeet, pyannote.audio, Ollama
- Media Tools: yt-dlp, FFmpeg

## License
MIT
