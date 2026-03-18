@echo off
if "%~1"=="--internal" goto :core_install

SET "LOGFILE=install_%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%.log"
echo [Chatalogue] Installing with log: %LOGFILE%
REM Use Python for unbuffered tee (PowerShell Tee-Object buffers and causes hangs)
where py >nul 2>&1 && (
  cmd /c ""%~f0" --internal" 2>&1 | py -3 -u -c "import sys; f=open('%LOGFILE%','w',encoding='utf-8'); [((sys.stdout.write(l),sys.stdout.flush(),f.write(l),f.flush()) if l else None) for l in sys.stdin]; f.close()"
  exit /b %ERRORLEVEL%
)
where python >nul 2>&1 && (
  cmd /c ""%~f0" --internal" 2>&1 | python -u -c "import sys; f=open('%LOGFILE%','w',encoding='utf-8'); [((sys.stdout.write(l),sys.stdout.flush(),f.write(l),f.flush()) if l else None) for l in sys.stdin]; f.close()"
  exit /b %ERRORLEVEL%
)
REM Fallback: no Python available yet, run without logging
echo [WARN] Python not found in PATH. Install log will not be created.
call "%~f0" --internal
exit /b %ERRORLEVEL%

:core_install
SETLOCAL EnableDelayedExpansion
SET "INSTALLER_VERSION=2026-03-09.2"

echo [Chatalogue Installer] Windows bootstrap v%INSTALLER_VERSION%
echo.

SET "SCRIPT_DIR=%~dp0"
IF "%SCRIPT_DIR:~-1%"=="\" SET "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

SET "REPO_URL=%CHATALOGUE_REPO_URL%"
IF "%REPO_URL%"=="" SET "REPO_URL=https://github.com/jonstreeter/Chatalogue.git"

SET "REPO_BRANCH=%CHATALOGUE_REPO_BRANCH%"
IF "%REPO_BRANCH%"=="" SET "REPO_BRANCH=main"

SET "REPO_DIR_NAME=%CHATALOGUE_REPO_DIR%"
IF "%REPO_DIR_NAME%"=="" SET "REPO_DIR_NAME=Chatalogue"

SET "PROJECT_ROOT=%SCRIPT_DIR%"
IF EXIST "%PROJECT_ROOT%\backend\src\main.py" IF EXIST "%PROJECT_ROOT%\frontend\package.json" GOTO :project_ready

echo [Bootstrap] Local repo not detected in "%SCRIPT_DIR%".
where git >nul 2>&1
IF ERRORLEVEL 1 (
  echo [ERROR] Git is required for bootstrap install. Please install Git and retry.
  EXIT /B 1
)

SET "CLONE_TARGET=%SCRIPT_DIR%\%REPO_DIR_NAME%"
IF EXIST "%CLONE_TARGET%\.git" (
  echo [Bootstrap] Existing repo found at "%CLONE_TARGET%". Pulling latest %REPO_BRANCH%...
  git -C "%CLONE_TARGET%" checkout "%REPO_BRANCH%" >nul 2>&1
  git -C "%CLONE_TARGET%" pull --ff-only origin "%REPO_BRANCH%"
  IF ERRORLEVEL 1 (
    echo [WARN] git pull failed. Continuing with existing checkout.
  )
) ELSE (
  IF EXIST "%CLONE_TARGET%" (
    echo [ERROR] "%CLONE_TARGET%" exists but is not a git repository.
    echo Delete it or set CHATALOGUE_REPO_DIR to a different folder name.
    EXIT /B 1
  )
  echo [Bootstrap] Cloning %REPO_URL% ^(%REPO_BRANCH%^)...
  git clone --branch "%REPO_BRANCH%" --depth 1 "%REPO_URL%" "%CLONE_TARGET%"
  IF ERRORLEVEL 1 (
    echo [ERROR] Failed to clone repository.
    EXIT /B 1
  )
)
SET "PROJECT_ROOT=%CLONE_TARGET%"

:project_ready
SET "BACKEND_DIR=%PROJECT_ROOT%\backend"
SET "FRONTEND_DIR=%PROJECT_ROOT%\frontend"
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
IF ERRORLEVEL 1 (
  echo [ERROR] Node.js 18+ is required and was not found in PATH.
  EXIT /B 1
)

where ffmpeg >nul 2>&1
IF ERRORLEVEL 1 (
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
IF ERRORLEVEL 1 (
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
  echo [6/8] Skipping Parakeet dependencies ^(INSTALL_PARAKEET=%INSTALL_PARAKEET%^).
)

echo [7/8] Installing frontend dependencies...
cd /d "%FRONTEND_DIR%"
call npm install --fund=false --audit=false --loglevel=warn

IF "%SKIP_MODEL_PRELOAD%"=="1" (
  echo [8/8] Skipping model preload ^(SKIP_MODEL_PRELOAD=1^).
) ELSE (
  echo [8/8] Preloading ASR models ^(engine=%PRELOAD_ENGINE%^)...
  cd /d "%BACKEND_DIR%"
  "%VENV_PYTHON%" -u preload_models.py --engine "%PRELOAD_ENGINE%"
)

IF DEFINED OLLAMA_MODELS (
  where ollama >nul 2>&1
  IF ERRORLEVEL 1 (
    echo [WARN] OLLAMA_MODELS was provided but `ollama` was not found in PATH.
  ) ELSE (
    echo Pulling Ollama models from OLLAMA_MODELS...
    FOR %%M IN (%OLLAMA_MODELS%) DO (
      echo   ollama pull %%M
      ollama pull %%M
    )
  )
)

echo.
echo Installation complete.
echo Project root: %PROJECT_ROOT%
echo Backend venv: %VENV_DIR%
IF EXIST "%PROJECT_ROOT%\run_windows.bat" (
  echo Start app with: "%PROJECT_ROOT%\run_windows.bat"
) ELSE (
  echo Start app by running backend/frontend manually from %PROJECT_ROOT%.
)
echo.
ENDLOCAL
