@echo off
SETLOCAL EnableDelayedExpansion

SET "PROJECT_ROOT=%~dp0"
SET "BACKEND_DIR=%PROJECT_ROOT%backend"
SET "FRONTEND_DIR=%PROJECT_ROOT%frontend"
SET "VENV_DIR=%BACKEND_DIR%\.venv"
SET "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
SET "PIP_CMD=%VENV_DIR%\Scripts\pip.exe"

SET "INSTALL_PARAKEET=%INSTALL_PARAKEET%"
IF "%INSTALL_PARAKEET%"=="" SET "INSTALL_PARAKEET=1"

SET "SKIP_MODEL_PRELOAD=%SKIP_MODEL_PRELOAD%"
IF "%SKIP_MODEL_PRELOAD%"=="" SET "SKIP_MODEL_PRELOAD=0"

SET "PRELOAD_ENGINE=%PRELOAD_ENGINE%"
IF "%PRELOAD_ENGINE%"=="" SET "PRELOAD_ENGINE=auto"

SET "PYTHON_CMD="
where py >nul 2>&1 && SET "PYTHON_CMD=py -3"
IF "!PYTHON_CMD!"=="" (
  where python >nul 2>&1 && SET "PYTHON_CMD=python"
)
IF "!PYTHON_CMD!"=="" (
  echo [ERROR] Python 3.10+ is required and was not found in PATH.
  EXIT /B 1
)

where node >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Node.js 18+ is required and was not found in PATH.
  EXIT /B 1
)

where ffmpeg >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
  echo [WARN] ffmpeg was not found in PATH. Download/extract will work, but media conversion may fail until ffmpeg is installed.
)

echo [1/8] Creating backend virtual environment...
IF NOT EXIST "%VENV_PYTHON%" (
  cd /d "%BACKEND_DIR%"
  %PYTHON_CMD% -m venv .venv
)

echo [2/8] Upgrading pip tooling...
"%VENV_PYTHON%" -m pip install --upgrade pip wheel "setuptools<81"

echo [3/8] Installing PyTorch nightly cu128 (RTX 50xx friendly)...
"%PIP_CMD%" install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
IF %ERRORLEVEL% NEQ 0 (
  echo [WARN] Nightly cu128 install failed. Falling back to default torch wheels...
  "%PIP_CMD%" install torch torchaudio
)

echo [4/8] Installing core backend dependencies...
"%PIP_CMD%" install fastapi uvicorn yt-dlp python-dotenv sqlmodel aiosqlite psycopg[binary] "setuptools<81" faster-whisper "ctranslate2<4.6" python-multipart sympy

echo [5/8] Installing pyannote stack...
"%PIP_CMD%" install pyannote.audio --no-deps
"%PIP_CMD%" install asteroid-filterbanks einops huggingface-hub lightning matplotlib opentelemetry-api opentelemetry-exporter-otlp opentelemetry-sdk pyannote-core pyannote-database pyannote-metrics pyannote-pipeline pytorch-metric-learning rich safetensors soundfile torch-audiomentations torchmetrics torchcodec pyannoteai-sdk av onnxruntime tokenizers

IF /I "%INSTALL_PARAKEET%"=="1" (
  echo [6/8] Installing optional Parakeet dependencies...
  "%PIP_CMD%" install -r "%BACKEND_DIR%\requirements-parakeet.txt"
) ELSE (
  echo [6/8] Skipping Parakeet dependencies (INSTALL_PARAKEET=%INSTALL_PARAKEET%).
)

echo [7/8] Installing frontend dependencies...
cd /d "%FRONTEND_DIR%"
npm install

IF "%SKIP_MODEL_PRELOAD%"=="1" (
  echo [8/8] Skipping model preload (SKIP_MODEL_PRELOAD=1).
) ELSE (
  echo [8/8] Preloading ASR models (engine=%PRELOAD_ENGINE%)...
  cd /d "%BACKEND_DIR%"
  "%VENV_PYTHON%" preload_models.py --engine "%PRELOAD_ENGINE%"
)

IF DEFINED OLLAMA_MODELS (
  where ollama >nul 2>&1
  IF %ERRORLEVEL% EQU 0 (
    echo Pulling Ollama models from OLLAMA_MODELS...
    FOR %%M IN (%OLLAMA_MODELS%) DO (
      echo   ollama pull %%M
      ollama pull %%M
    )
  ) ELSE (
    echo [WARN] OLLAMA_MODELS was provided but `ollama` was not found in PATH.
  )
)

echo.
echo Installation complete.
echo Backend venv: %VENV_DIR%
echo Start app with: run.bat
echo.
ENDLOCAL
