#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" != "--internal" ]; then
    echo "[Chatalogue] Initializing installer with logging (saving to install.log)..."
    bash "$0" --internal 2>&1 | tee "install_$(date +%Y%m%d%H%M).log"
    exit ${PIPESTATUS[0]}
fi

INSTALLER_VERSION="2026-03-19.1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_URL="${CHATALOGUE_REPO_URL:-https://github.com/jonstreeter/Chatalogue.git}"
REPO_BRANCH="${CHATALOGUE_REPO_BRANCH:-main}"
REPO_DIR_NAME="${CHATALOGUE_REPO_DIR:-Chatalogue}"

echo
echo " ============================================"
echo "   Chatalogue Installer  v${INSTALLER_VERSION}"
echo " ============================================"
echo

# ===================================================================
#  PREREQUISITE CHECKS
# ===================================================================

PREREQS_OK=1

ask_yes_no() {
  local prompt="$1"
  local answer
  read -r -p "      $prompt [Y/n]: " answer </dev/tty
  answer="${answer:-Y}"
  [[ "$answer" =~ ^[Yy]$ ]]
}

# --- Homebrew (macOS package manager) ---
echo "[Prerequisites] Checking Homebrew..."
if ! command -v brew >/dev/null 2>&1; then
  echo "  [!] Homebrew is NOT installed."
  echo "      Homebrew is needed to install missing prerequisites (Git, Python, Node.js, ffmpeg)."
  echo
  if ask_yes_no "Would you like to install Homebrew now?"; then
    echo "  Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for this session (Apple Silicon vs Intel)
    if [ -f /opt/homebrew/bin/brew ]; then
      eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -f /usr/local/bin/brew ]; then
      eval "$(/usr/local/bin/brew shellenv)"
    fi
    if command -v brew >/dev/null 2>&1; then
      echo "  [OK] Homebrew installed successfully."
    else
      echo "  [!] Homebrew was installed but is not in your PATH."
      echo "      Close this terminal, open a NEW terminal, and re-run this installer."
      PREREQS_OK=0
    fi
  else
    echo "  [WARN] Skipping Homebrew. You may need to manually install missing prerequisites."
  fi
else
  echo "  [OK] Homebrew $(brew --version | head -1 | awk '{print $2}')"
fi

HAS_BREW=0
command -v brew >/dev/null 2>&1 && HAS_BREW=1

# --- Git ---
echo "[Prerequisites] Checking Git..."
if ! command -v git >/dev/null 2>&1; then
  echo "  [!] Git is NOT installed."
  if [ "$HAS_BREW" = "1" ]; then
    echo
    if ask_yes_no "Would you like to install Git via Homebrew?"; then
      echo "  Installing Git..."
      brew install git
      if command -v git >/dev/null 2>&1; then
        echo "  [OK] Git installed successfully."
      else
        echo "  [!] Git install completed but 'git' is not in your PATH."
        echo "      Try: xcode-select --install  (or restart your terminal)"
        PREREQS_OK=0
      fi
    else
      echo "  Please install Git: https://git-scm.com/downloads"
      echo "  Or run: xcode-select --install"
      PREREQS_OK=0
    fi
  else
    echo "  Please install Git: https://git-scm.com/downloads"
    echo "  Or run: xcode-select --install"
    PREREQS_OK=0
  fi
else
  echo "  [OK] $(git --version)"
fi

# --- Python ---
PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "[Prerequisites] Checking Python..."
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "  [!] Python is NOT installed ($PYTHON_BIN not found)."
  if [ "$HAS_BREW" = "1" ]; then
    echo
    if ask_yes_no "Would you like to install Python 3.12 via Homebrew?"; then
      echo "  Installing Python 3.12..."
      brew install python@3.12
      # Update PYTHON_BIN to use newly installed Python
      if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
        echo "  [OK] Python installed successfully."
      else
        echo "  [!] Python install completed but 'python3' is not in PATH."
        echo "      Close this terminal, open a NEW terminal, and re-run this installer."
        PREREQS_OK=0
      fi
    else
      echo "  Please install Python 3.10+: https://www.python.org/downloads/"
      PREREQS_OK=0
    fi
  else
    echo "  Please install Python 3.10+: https://www.python.org/downloads/"
    PREREQS_OK=0
  fi
else
  echo "  [OK] $($PYTHON_BIN --version)"
fi

# --- Node.js ---
echo "[Prerequisites] Checking Node.js..."
if ! command -v node >/dev/null 2>&1; then
  echo "  [!] Node.js is NOT installed."
  if [ "$HAS_BREW" = "1" ]; then
    echo
    if ask_yes_no "Would you like to install Node.js LTS via Homebrew?"; then
      echo "  Installing Node.js LTS..."
      brew install node
      if command -v node >/dev/null 2>&1; then
        echo "  [OK] Node.js installed successfully."
      else
        echo "  [!] Node.js install completed but 'node' is not in PATH."
        echo "      Close this terminal, open a NEW terminal, and re-run this installer."
        PREREQS_OK=0
      fi
    else
      echo "  Please install Node.js 18+: https://nodejs.org/en/download/"
      PREREQS_OK=0
    fi
  else
    echo "  Please install Node.js 18+: https://nodejs.org/en/download/"
    PREREQS_OK=0
  fi
else
  echo "  [OK] Node.js $(node --version)"
fi

# --- ffmpeg ---
echo "[Prerequisites] Checking ffmpeg..."
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "  [!] ffmpeg is NOT installed (optional but recommended)."
  if [ "$HAS_BREW" = "1" ]; then
    echo
    if ask_yes_no "Would you like to install ffmpeg via Homebrew?"; then
      echo "  Installing ffmpeg..."
      brew install ffmpeg
      if command -v ffmpeg >/dev/null 2>&1; then
        echo "  [OK] ffmpeg installed successfully."
      else
        echo "  [WARN] ffmpeg install completed but 'ffmpeg' is not in PATH."
        echo "         Media conversion features may fail."
      fi
    else
      echo "  [WARN] ffmpeg not installed. Media conversion features may fail."
      echo "         Download from: https://ffmpeg.org/download.html"
    fi
  else
    echo "  [WARN] ffmpeg not installed. Media conversion features may fail."
    echo "         Download from: https://ffmpeg.org/download.html"
  fi
else
  echo "  [OK] $(ffmpeg -version 2>/dev/null | head -1)"
fi

echo "[Prerequisites] Checking SoX..."
if ! command -v sox >/dev/null 2>&1; then
  echo "  [!] SoX is NOT installed (required for conversation reconstruction)."
  if [ "$HAS_BREW" = "1" ]; then
    echo
    if ask_yes_no "Would you like to install SoX via Homebrew?"; then
      echo "  Installing SoX..."
      brew install sox
      if command -v sox >/dev/null 2>&1; then
        echo "  [OK] SoX installed successfully."
      else
        echo "  [WARN] SoX install completed but 'sox' is not in PATH."
        echo "         Conversation reconstruction may not work until PATH is refreshed."
      fi
    else
      echo "  [WARN] SoX not installed. Conversation reconstruction will be unavailable."
      echo "         Download from: https://formulae.brew.sh/formula/sox"
    fi
  else
    echo "  [WARN] SoX not installed. Conversation reconstruction will be unavailable."
    echo "         Download from: https://formulae.brew.sh/formula/sox"
  fi
else
  echo "  [OK] $(sox --version 2>/dev/null | head -1)"
fi

echo
if [ "$PREREQS_OK" -ne 1 ]; then
  echo "[ERROR] One or more required prerequisites are missing."
  echo "        Install the missing tools listed above, then re-run this installer."
  exit 1
fi
echo "[Prerequisites] All checks passed."
echo

# ===================================================================
#  REPOSITORY SETUP
# ===================================================================

ROOT_DIR="$SCRIPT_DIR"
if [[ ! -f "$ROOT_DIR/backend/src/main.py" || ! -f "$ROOT_DIR/frontend/package.json" ]]; then
  echo "[Bootstrap] Local repo not detected in \"$SCRIPT_DIR\"."
  if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] Git is required for bootstrap install."
    exit 1
  fi

  CLONE_TARGET="$SCRIPT_DIR/$REPO_DIR_NAME"
  if [[ -d "$CLONE_TARGET/.git" ]]; then
    echo "[Bootstrap] Existing repo found at \"$CLONE_TARGET\". Pulling latest $REPO_BRANCH..."
    (
      cd "$CLONE_TARGET"
      git checkout "$REPO_BRANCH" >/dev/null 2>&1 || true
      git pull --ff-only origin "$REPO_BRANCH" || echo "[WARN] git pull failed. Continuing with existing checkout."
    )
  else
    if [[ -e "$CLONE_TARGET" ]]; then
      echo "[ERROR] \"$CLONE_TARGET\" exists but is not a git repository."
      echo "Delete it or set CHATALOGUE_REPO_DIR to a different folder name."
      exit 1
    fi
    echo "[Bootstrap] Cloning $REPO_URL ($REPO_BRANCH)..."
    git clone --branch "$REPO_BRANCH" --depth 1 "$REPO_URL" "$CLONE_TARGET"
  fi
  ROOT_DIR="$CLONE_TARGET"
fi

# ===================================================================
#  INSTALL DEPENDENCIES
# ===================================================================

BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/.venv"

INSTALL_PARAKEET="${INSTALL_PARAKEET:-1}"
SKIP_MODEL_PRELOAD="${SKIP_MODEL_PRELOAD:-0}"
PRELOAD_ENGINE="${PRELOAD_ENGINE:-auto}"

echo "[1/8] Creating backend virtual environment..."
if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
PIP_CMD="$VENV_DIR/bin/pip"

echo "[2/8] Upgrading pip tooling..."
"$VENV_PYTHON" -m pip install --upgrade pip wheel "setuptools<81"

echo "[3/8] Installing torch + torchaudio..."
"$PIP_CMD" install -r "$BACKEND_DIR/requirements-macos.txt"

echo "[4/8] Installing core backend dependencies..."
"$PIP_CMD" install -r "$BACKEND_DIR/requirements.txt"

echo "[5/8] Installing pyannote stack..."
"$PIP_CMD" install pyannote.audio==4.0.4 --no-deps

if [ "$INSTALL_PARAKEET" = "1" ]; then
  echo "[6/8] Installing optional Parakeet dependencies..."
  "$PIP_CMD" install -r "$BACKEND_DIR/requirements-parakeet.txt"
else
  echo "[6/8] Skipping Parakeet dependencies (INSTALL_PARAKEET=$INSTALL_PARAKEET)."
fi

echo "[7/8] Installing frontend dependencies..."
(
  cd "$FRONTEND_DIR"
  npm install --fund=false --audit=false --loglevel=warn
)

if [ "$SKIP_MODEL_PRELOAD" = "1" ]; then
  echo "[8/8] Skipping model preload (SKIP_MODEL_PRELOAD=1)."
else
  echo "[8/8] Preloading ASR models (engine=$PRELOAD_ENGINE)..."
  (
    cd "$BACKEND_DIR"
    "$VENV_PYTHON" -u preload_models.py --engine "$PRELOAD_ENGINE"
  )
fi

if [ -n "${OLLAMA_MODELS:-}" ]; then
  if command -v ollama >/dev/null 2>&1; then
    echo "Pulling Ollama models from OLLAMA_MODELS..."
    for model in $OLLAMA_MODELS; do
      echo "  ollama pull $model"
      ollama pull "$model"
    done
  else
    echo "[WARN] OLLAMA_MODELS provided but ollama binary not found in PATH."
  fi
fi

# ===================================================================
#  DONE — SUMMARY
# ===================================================================

echo
echo " ============================================"
echo "   Installation complete!"
echo " ============================================"
echo
echo " Project root : $ROOT_DIR"
echo " Backend venv : $VENV_DIR"
echo
echo " HOW TO START THE APP:"
echo " ---------------------"
if [[ -x "$ROOT_DIR/run_mac.sh" ]]; then
  echo "   cd \"$ROOT_DIR\""
  echo "   ./run_mac.sh"
else
  echo "   Start backend/frontend manually from $ROOT_DIR."
fi
echo
echo " Once running, open your browser to:"
echo "   Frontend : http://localhost:5173"
echo "   API docs : http://localhost:8011/docs"
echo
echo " HOW TO UPDATE LATER:"
echo " --------------------"
if [[ -x "$ROOT_DIR/update_mac.sh" ]]; then
  echo "   cd \"$ROOT_DIR\""
  echo "   ./update_mac.sh"
else
  echo "   cd into $ROOT_DIR and run: git pull"
fi
echo
echo " CONFIGURATION:"
echo " --------------"
echo "  Copy backend/.env.example to backend/.env and set:"
echo "    HF_TOKEN  = your Hugging Face token (for diarization models)"
echo
