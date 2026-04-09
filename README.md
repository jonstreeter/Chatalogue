# Chatalogue

|  |  |
|---|---|
| <img src="frontend/public/chatalogue-logo.svg" width="44" alt="Chatalogue logo" /> | **Chatalogue**<br/>*Dialogue -> Data* |

Chatalogue is a local-first workstation for ingesting, transcribing, diarizing, optimizing, searching, and editing long-form spoken content.

## Core Features
- Channel ingest and automatic refreshing
- High-performance transcription and speaker diarization pipeline
- Transcript optimization workbench with repair, rebuild, retranscribe, benchmarks, and campaigns
- Channel-level transcript search
- Speaker profile management and merging
- AI episode cloning workbench with multi-variant generation
- Funny-moment detection and explanation
- AI-powered summaries and chapter generation
- Cleanup, ClearVoice, VoiceFixer, and conversation reconstruction runtimes
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

## Test-Ready Development Setup

The normal installers create a runtime-ready environment. If you also want local test execution:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

On macOS/Linux:

```bash
cd backend
.venv/bin/python -m pip install -r requirements-dev.txt
```

Then run tests with:

```bash
python -m pytest backend/src -q
```

---

## Configuration

To override backend settings, copy `backend/.env.example` to `backend/.env` and adjust values as needed.

Common settings:
- `HF_TOKEN`: Hugging Face token for gated diarization models
- `TRANSCRIPTION_ENGINE`: `auto`, `whisper`, or `parakeet`
- `WHISPER_BACKEND`: `faster_whisper` or `insanely_fast_whisper`
- `PARAKEET_MODEL`: default `nvidia/parakeet-tdt-0.6b-v2`
- `PARAKEET_ALLOW_WHISPER_FALLBACK`: `true` or `false`
- `MULTILINGUAL_ROUTING_ENABLED`: route likely non-English episodes to Whisper automatically
- `MULTILINGUAL_WHISPER_MODEL`: multilingual Whisper model used for language-routed jobs
- `LANGUAGE_DETECTION_SAMPLE_SECONDS`: opening-audio probe window for language detection
- `LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD`: threshold for metadata/audio language routing
- `PIPELINE_EXECUTION_MODE`: `sequential` or `staged`
- `YOUTUBE_DATA_API_KEY`: optional YouTube Data API key for view-count/popularity metadata and API-key test
- `DB_PROVIDER`: default `postgres`

---

## Recommended Post-Install Setup

After first launch, open `Settings` and review these sections:

- `Transcription`
  - choose `TRANSCRIPTION_ENGINE`
  - choose `WHISPER_BACKEND`
  - test the engine/runtime path
- `YouTube`
  - add a `YouTube Data API Key` if you want stable popularity/view-count backfill
  - use `Test API Key` to verify connectivity
- `Runtimes`
  - install and self-test optional local runtimes such as `ClearVoice`, `VoiceFixer`, and `Conversation Reconstruction`

The YouTube Data API key is the preferred path for public view-count metadata. Browser-cookie scraping remains optional fallback behavior, not the recommended default.

---

## Technology Stack
- Backend: FastAPI, SQLModel, Uvicorn, embedded PostgreSQL
- Frontend: React, Vite, TypeScript, Tailwind
- AI/ML: Whisper, NVIDIA Parakeet, pyannote.audio, Ollama
- Media Tools: yt-dlp, FFmpeg

## License
MIT
