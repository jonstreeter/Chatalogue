#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
PIP_CMD="$VENV_DIR/bin/pip"

echo "[Chatalogue Update]"
echo

# Verify we're in a git repo with the expected structure
if [ ! -d "$ROOT_DIR/.git" ]; then
  echo "[ERROR] Not a git repository. Run this script from the Chatalogue project root."
  exit 1
fi
if [ ! -f "$BACKEND_DIR/src/main.py" ]; then
  echo "[ERROR] Backend not found. Run this script from the Chatalogue project root."
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] Git is not installed or not in PATH."
  exit 1
fi

# Pull latest changes
echo "[1/4] Pulling latest changes from GitHub..."
cd "$ROOT_DIR"
if ! git pull --ff-only; then
  echo
  echo "[ERROR] git pull failed. You may have local changes that conflict."
  echo "Try: git stash, then re-run this script, then: git stash pop"
  exit 1
fi

# Update backend dependencies
echo
echo "[2/4] Updating backend dependencies..."
if [ ! -x "$VENV_PYTHON" ]; then
  echo "[WARN] Backend venv not found. Run install_mac.sh first."
  exit 1
fi
"$PIP_CMD" install --upgrade fastapi uvicorn yt-dlp python-dotenv sqlmodel aiosqlite psycopg[binary] "setuptools<81" faster-whisper "ctranslate2<4.6" python-multipart sympy

# Update frontend dependencies
echo
echo "[3/4] Updating frontend dependencies..."
(
  cd "$FRONTEND_DIR"
  npm install
)

# Done
echo
echo "[4/4] Update complete!"
echo
echo "Run $ROOT_DIR/run_mac.sh to start Chatalogue."
