#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
"$PIP_CMD" install torch torchaudio

echo "[4/8] Installing core backend dependencies..."
"$PIP_CMD" install fastapi uvicorn yt-dlp python-dotenv sqlmodel aiosqlite psycopg[binary] "setuptools<81" faster-whisper "ctranslate2<4.6" python-multipart sympy

echo "[5/8] Installing pyannote stack..."
"$PIP_CMD" install pyannote.audio --no-deps
"$PIP_CMD" install asteroid-filterbanks einops huggingface-hub lightning matplotlib opentelemetry-api opentelemetry-exporter-otlp opentelemetry-sdk pyannote-core pyannote-database pyannote-metrics pyannote-pipeline pytorch-metric-learning rich safetensors soundfile torch-audiomentations torchmetrics torchcodec pyannoteai-sdk av onnxruntime tokenizers

if [ "$INSTALL_PARAKEET" = "1" ]; then
  echo "[6/8] Installing optional Parakeet dependencies..."
  "$PIP_CMD" install -r "$BACKEND_DIR/requirements-parakeet.txt"
else
  echo "[6/8] Skipping Parakeet dependencies (INSTALL_PARAKEET=$INSTALL_PARAKEET)."
fi

echo "[7/8] Installing frontend dependencies..."
(
  cd "$FRONTEND_DIR"
  npm install
)

if [ "$SKIP_MODEL_PRELOAD" = "1" ]; then
  echo "[8/8] Skipping model preload (SKIP_MODEL_PRELOAD=1)."
else
  echo "[8/8] Preloading ASR models (engine=$PRELOAD_ENGINE)..."
  (
    cd "$BACKEND_DIR"
    "$VENV_PYTHON" preload_models.py --engine "$PRELOAD_ENGINE"
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
echo "Backend venv: $VENV_DIR"
echo "Start backend: $VENV_DIR/bin/python -m uvicorn src.main:app --app-dir \"$BACKEND_DIR\" --host 0.0.0.0 --port 8011"
echo "Start frontend: cd \"$FRONTEND_DIR\" && npm run dev"
