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
echo "[1/7] Pulling latest changes from GitHub..."
cd "$ROOT_DIR"
if ! git pull --ff-only; then
  echo
  echo "[ERROR] git pull failed. You may have local changes that conflict."
  echo "Try: git stash, then re-run this script, then: git stash pop"
  exit 1
fi

# Update backend dependencies
echo
echo "[2/7] Updating torch + torchaudio..."
if [ ! -x "$VENV_PYTHON" ]; then
  echo "[WARN] Backend venv not found. Run install_mac.sh first."
  exit 1
fi
"$PIP_CMD" install --upgrade --quiet -r "$BACKEND_DIR/requirements-macos.txt"
echo "   Torch stack up to date."

echo
echo "[3/7] Updating shared backend dependencies..."
"$PIP_CMD" install --upgrade --quiet -r "$BACKEND_DIR/requirements.txt"
echo "   Shared backend dependencies up to date."

echo
echo "[4/7] Updating pyannote stack..."
"$PIP_CMD" install --upgrade --quiet pyannote.audio==4.0.4 --no-deps
echo "   Pyannote stack up to date."

echo
echo "[5/7] Updating optional Parakeet dependencies..."
if "$PIP_CMD" show nemo-toolkit >/dev/null 2>&1; then
  "$PIP_CMD" install --upgrade --quiet -r "$BACKEND_DIR/requirements-parakeet.txt"
  echo "   Optional Parakeet dependencies up to date."
else
  echo "   Parakeet not installed; skipping optional NeMo stack."
fi

# Update frontend dependencies
echo
echo "[6/7] Updating frontend dependencies..."
(
  cd "$FRONTEND_DIR"
  npm install --fund=false --audit=false --loglevel=warn
)

# Done
echo
echo "[7/7] Update complete!"
echo
echo "Run $ROOT_DIR/run_mac.sh to start Chatalogue."
