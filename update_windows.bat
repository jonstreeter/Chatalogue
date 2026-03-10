@echo off
SETLOCAL EnableDelayedExpansion

SET "PROJECT_ROOT=%~dp0"
IF "%PROJECT_ROOT:~-1%"=="\" SET "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
SET "BACKEND_DIR=%PROJECT_ROOT%\backend"
SET "FRONTEND_DIR=%PROJECT_ROOT%\frontend"
SET "VENV_DIR=%BACKEND_DIR%\.venv"
SET "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
SET "PIP_CMD=%VENV_DIR%\Scripts\pip.exe"

echo [Chatalogue Update]
echo.

:: Verify we're in a git repo with the expected structure
IF NOT EXIST "%PROJECT_ROOT%\.git" (
    echo [ERROR] Not a git repository. Run this script from the Chatalogue project root.
    pause
    exit /b 1
)
IF NOT EXIST "%BACKEND_DIR%\src\main.py" (
    echo [ERROR] Backend not found. Run this script from the Chatalogue project root.
    pause
    exit /b 1
)

:: Check for git
where git >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Git is not installed or not in PATH.
    pause
    exit /b 1
)

:: Pull latest changes
echo [1/4] Pulling latest changes from GitHub...
cd /d "%PROJECT_ROOT%"
git pull --ff-only
IF !ERRORLEVEL! NEQ 0 (
    echo.
    echo [ERROR] git pull failed. You may have local changes that conflict.
    echo Try: git stash, then re-run this script, then: git stash pop
    pause
    exit /b 1
)

:: Update backend dependencies
echo.
echo [2/4] Updating backend dependencies...
IF NOT EXIST "%VENV_PYTHON%" (
    echo [WARN] Backend venv not found. Run install_windows.bat first.
    pause
    exit /b 1
)
"%PIP_CMD%" install --upgrade fastapi uvicorn yt-dlp python-dotenv sqlmodel aiosqlite psycopg[binary] "setuptools<81" faster-whisper "ctranslate2<4.6" python-multipart sympy

:: Update frontend dependencies
echo.
echo [3/4] Updating frontend dependencies...
cd /d "%FRONTEND_DIR%"
call npm install

:: Done
echo.
echo [4/4] Update complete!
echo.
echo Run "%PROJECT_ROOT%\run.bat" to start Chatalogue.
echo.
pause
ENDLOCAL
