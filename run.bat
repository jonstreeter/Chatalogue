@echo off
SETLOCAL EnableDelayedExpansion

:: Define paths
SET "PROJECT_ROOT=%~dp0"
SET "PROJECT_NAME=Chatalogue"
SET "BACKEND_DIR=%PROJECT_ROOT%backend"
SET "FRONTEND_DIR=%PROJECT_ROOT%frontend"
SET "VENV_DIR=%BACKEND_DIR%\.venv"
SET "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
SET "BACKEND_PORT=8011"

:: Kill any existing instances
echo Stopping any existing servers...
taskkill /F /T /FI "WINDOWTITLE eq %PROJECT_NAME% Backend*" >nul 2>&1
taskkill /F /T /FI "WINDOWTITLE eq %PROJECT_NAME% Frontend*" >nul 2>&1
:: Kill backend uvicorn processes by command line (catches --reload parent/child even if not listening)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$backend='%BACKEND_DIR%'; Get-CimInstance Win32_Process | Where-Object { ($_.Name -ieq 'python.exe' -or $_.Name -ieq 'pythonw.exe') -and $_.CommandLine -like '*uvicorn*src.main:app*' -and $_.CommandLine -like ('*' + $backend + '*') } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {} }" >nul 2>&1
:: Kill frontend dev server wrappers/processes by command line (vite/npm dev)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$frontend='%FRONTEND_DIR%'; Get-CimInstance Win32_Process | Where-Object { (($_.Name -ieq 'node.exe') -or ($_.Name -ieq 'cmd.exe')) -and $_.CommandLine -like ('*' + $frontend + '*') -and ($_.CommandLine -like '*vite*' -or $_.CommandLine -like '*npm*run*dev*') } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {} }" >nul 2>&1
:: Also kill by port in case window titles don't match
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%BACKEND_PORT%.*LISTENING" 2^>nul') do (
    taskkill /F /T /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173.*LISTENING" 2^>nul') do (
    taskkill /F /T /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

:: If venv already exists, skip Python detection
IF EXIST "%VENV_PYTHON%" goto :venv_ready

:: Find Python - try "python", then "py" (Windows Launcher)
SET "PYTHON_CMD="
where python >nul 2>&1 && SET "PYTHON_CMD=python"
IF "!PYTHON_CMD!"=="" (
    where py >nul 2>&1 && SET "PYTHON_CMD=py"
)
IF "!PYTHON_CMD!"=="" (
    echo Python is not installed or not in PATH.
    echo Install Python 3.10+ and ensure "Add to PATH" is checked.
    pause
    exit /b 1
)

echo Creating Python virtual environment...
cd /d "%BACKEND_DIR%"
!PYTHON_CMD! -m venv .venv
echo Installing dependencies (this may take a while on first run)...

:: Step 1: PyTorch nightly with CUDA 12.8 (required for RTX 5090 Blackwell)
echo [1/4] Installing PyTorch nightly cu128...
"%VENV_DIR%\Scripts\pip" install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

:: Step 2: App dependencies
echo [2/4] Installing app dependencies...
"%VENV_DIR%\Scripts\pip" install fastapi uvicorn yt-dlp python-dotenv sqlmodel aiosqlite psycopg[binary] "setuptools<81" faster-whisper "ctranslate2<4.6" python-multipart

:: Step 3: pyannote (--no-deps to avoid torch version conflict)
echo [3/4] Installing pyannote.audio...
"%VENV_DIR%\Scripts\pip" install pyannote.audio --no-deps

:: Step 4: pyannote's other dependencies
echo [4/4] Installing pyannote dependencies...
"%VENV_DIR%\Scripts\pip" install asteroid-filterbanks einops huggingface-hub lightning matplotlib opentelemetry-api opentelemetry-exporter-otlp opentelemetry-sdk pyannote-core pyannote-database pyannote-metrics pyannote-pipeline pytorch-metric-learning rich safetensors soundfile torch-audiomentations torchmetrics torchcodec pyannoteai-sdk av onnxruntime tokenizers

:: Optional Step: Parakeet engine dependencies (set INSTALL_PARAKEET=1 before running)
IF /I "%INSTALL_PARAKEET%"=="1" (
    echo [optional] Installing Parakeet dependencies...
    "%VENV_DIR%\Scripts\pip" install -r "%BACKEND_DIR%\requirements-parakeet.txt"
)

:venv_ready
echo Python venv OK.

:: Check if Node.js is installed
where node >nul 2>&1
IF !ERRORLEVEL! NEQ 0 (
    echo Node.js is not installed or not in PATH. Please install Node.js 18+.
    pause
    exit /b 1
)

:: Install frontend dependencies if needed
IF NOT EXIST "%FRONTEND_DIR%\node_modules" (
    echo Installing frontend dependencies...
    cd /d "%FRONTEND_DIR%"
    npm install
)

:: Start Backend (worker thread starts automatically inside the API server)
echo Starting Backend Server on http://localhost:%BACKEND_PORT% ...
start "%PROJECT_NAME% Backend" "%VENV_PYTHON%" -m uvicorn src.main:app --app-dir "%BACKEND_DIR%" --host 0.0.0.0 --port %BACKEND_PORT%

:: Start Frontend
echo Starting Frontend on http://localhost:5173 ...
start "%PROJECT_NAME% Frontend" cmd /c "cd /d "%FRONTEND_DIR%" && npm run dev"

:: Wait for servers to start, then open browser
timeout /t 3 /nobreak >nul
start http://localhost:5173

echo.
echo Backend:  http://localhost:%BACKEND_PORT%
echo Frontend: http://localhost:5173
echo.
echo Close this window to stop both servers.
pause
