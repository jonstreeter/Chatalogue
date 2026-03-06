# Chatalogue

|  |  |
|---|---|
| <img src="frontend/public/chatalogue-logo.svg" width="44" alt="Chatalogue logo" /> | **Chatalogue**<br/>*Dialogue -> Data* |

Chatalogue is a local-first workstation designed to help you process, search, and edit long-form spoken content.

### Core Features:
- Channel ingest & automatic refreshing
- High-performance transcription & speaker diarization pipeline
- Channel-level transcript search
- Speaker profile management and merging
- Funny-moment detection & explanation
- AI-powered summaries and chapter generation
- Video clip editing, exporting, and uploading

---

## 🚀 Quick Start (Installation)

We've made installation as simple as possible. Ensure you have the prerequisites, then download and run the installer for your system.

### Prerequisites
Before installing, make sure your system has the following installed:
1. **[Git](https://git-scm.com/downloads)**
2. **[Python 3.10+](https://www.python.org/downloads/)** *(Important: Make sure to check **"Add Python to PATH"** during installation)*
3. **[Node.js 18+](https://nodejs.org/en/download/)**

### Windows Installation
1. Download the Windows installer: **[⏬ Download install_windows.zip](https://github.com/jonstreeter/Chatalogue/raw/main/installers/install_windows.zip)**
2. Extract the downloaded `.zip` file into the folder where you want Chatalogue to live. 
3. Open the extracted folder and double-click `install_windows.bat` to run the installation script.

### macOS Installation
1. Download the macOS installer: **[⏬ Download install_mac.zip](https://github.com/jonstreeter/Chatalogue/raw/main/installers/install_mac.zip)**
2. Extract the downloaded `.zip` file into the folder where you want Chatalogue to live.
3. Open your Terminal, navigate to the extracted folder, and run:
   ```bash
   chmod +x install_mac.sh
   ./install_mac.sh
   ```

*Note: The installers will automatically clone the repository, set up the Python virtual environment, install all dependencies, and begin preloading transcription models. This may take a few minutes depending on your internet connection.*

---

## 💻 Running the App

Once installation is complete, you can start Chatalogue at any time:

**Windows:**
Double-click the `run.bat` file inside your newly created `Chatalogue` folder.

**macOS:**
Open your Terminal, navigate to your `Chatalogue` folder, and run:
```bash
./run_mac.sh
```

The application will launch automatically in your default web browser at `http://localhost:5173`. 
*(Backend API docs are available at `http://localhost:8011/docs`)*

---

## ⚙️ Advanced Configuration (Optional)

Chatalogue is highly configurable. To change backend settings, copy the `backend/.env.example` file to `backend/.env` and adjust the keys as needed.

**Common Configurations:**
- `HF_TOKEN`: Your HuggingFace token for accessing gated diarization models.
- `TRANSCRIPTION_ENGINE`: Choose between `auto`, `whisper`, or `parakeet`.
- `PARAKEET_MODEL`: (Default: `nvidia/parakeet-tdt-0.6b-v2`).
- `DB_PROVIDER`: (Default: `postgres`)

### Advanced Installer Setup
If you need custom behavior during the initial installation script, you can set the following environment variables *before* running the installer:

- `INSTALL_PARAKEET=1|0` (Default: `1`)
- `SKIP_MODEL_PRELOAD=1|0` (Default: `0`) 
- `PRELOAD_ENGINE=auto|whisper|parakeet` (Default: `auto`)
- `OLLAMA_MODELS="model1 model2 ..."` (Pull specific Ollama models during install)
- `CHATALOGUE_REPO_URL` / `CHATALOGUE_REPO_BRANCH` (Install from a custom fork or branch)

**Example (Windows):**
```bat
set INSTALL_PARAKEET=0
set SKIP_MODEL_PRELOAD=1
install_windows.bat
```

---

## Technology Stack
- **Backend**: FastAPI, SQLModel, Uvicorn, PostgreSQL (Embedded)
- **Frontend**: React, Vite, TypeScript, Tailwind
- **AI/ML**: Whisper, NVIDIA Parakeet, pyannote.audio, Ollama
- **Media Tools**: yt-dlp, FFmpeg

**License:** MIT
