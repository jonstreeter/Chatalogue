#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" != "--internal" ]; then
    echo "[Chatalogue] Initializing installer with logging (saving to install.log)..."
    bash "$0" --internal 2>&1 | tee "install_$(date +%Y%m%d%H%M).log"
    exit ${PIPESTATUS[0]}
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_URL="${CHATALOGUE_REPO_URL:-https://github.com/jonstreeter/Chatalogue.git}"
REPO_BRANCH="${CHATALOGUE_REPO_BRANCH:-main}"
REPO_DIR_NAME="${CHATALOGUE_REPO_DIR:-Chatalogue}"

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

BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/.venv"

PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_PARAKEET="${INSTALL_PARAKEET:-1}"
SKIP_MODEL_PRELOAD="${SKIP_MODEL_PRELOAD:-0}"
PRELOAD_ENGINE="${PRELOAD_ENGINE:-auto}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python 3.10+ is required and was not found: $PYTHON_BIN"
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "[ERROR] Node.js 18+ is required and was not found in PATH."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[WARN] ffmpeg was not found in PATH. Media conversion features may fail until ffmpeg is installed."
fi

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

echo
echo "Installation complete."
echo "Project root: $ROOT_DIR"
echo "Backend venv: $VENV_DIR"
if [[ -x "$ROOT_DIR/run_mac.sh" ]]; then
  echo "Start app with: $ROOT_DIR/run_mac.sh"
fi
