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
echo [1/7] Pulling latest changes from GitHub...
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
echo [2/7] Updating torch + torchaudio...
IF NOT EXIST "%VENV_PYTHON%" (
    echo [WARN] Backend venv not found. Run install_windows.bat first.
    pause
    exit /b 1
)
"%PIP_CMD%" install --upgrade --quiet --pre -r "%BACKEND_DIR%\requirements-windows-cu128.txt"
IF ERRORLEVEL 1 (
    echo [WARN] Nightly cu128 update failed. Falling back to stable torch pins...
    "%PIP_CMD%" install --upgrade --quiet -r "%BACKEND_DIR%\requirements-macos.txt"
)
echo    Validating torch stack...
"%VENV_PYTHON%" "%BACKEND_DIR%\tools\check_torch_stack.py"
IF ERRORLEVEL 1 (
    echo [ERROR] Torch validation failed.
    echo         An NVIDIA GPU was detected, but this venv cannot use CUDA.
    echo         Updating further would leave transcription running on CPU.
    pause
    exit /b 1
)
echo    Torch stack up to date.

echo.
echo [3/7] Updating shared backend dependencies...
"%PIP_CMD%" install --upgrade --quiet -r "%BACKEND_DIR%\requirements.txt"
echo    Shared backend dependencies up to date.

echo.
echo [4/7] Updating pyannote stack...
"%PIP_CMD%" install --upgrade --quiet pyannote.audio==4.0.4 --no-deps
IF ERRORLEVEL 1 (
    echo [ERROR] pyannote.audio update failed.
    pause
    exit /b 1
)
echo    Normalizing pyannote package metadata...
"%VENV_PYTHON%" "%BACKEND_DIR%\tools\repair_pyannote_metadata.py"
IF ERRORLEVEL 1 (
    echo [ERROR] pyannote metadata normalization failed.
    pause
    exit /b 1
)
echo    Pyannote stack up to date.

echo.
echo [5/7] Updating optional Parakeet dependencies...
"%PIP_CMD%" show nemo-toolkit >nul 2>&1
IF ERRORLEVEL 1 (
    echo    Parakeet not installed; skipping optional NeMo stack.
) ELSE (
    "%PIP_CMD%" install --upgrade --quiet -r "%BACKEND_DIR%\requirements-parakeet.txt"
    echo    Optional Parakeet dependencies up to date.
)

:: Update frontend dependencies
echo.
echo [6/7] Updating frontend dependencies...
cd /d "%FRONTEND_DIR%"
call npm install --fund=false --audit=false --loglevel=warn

:: Done
echo.
echo [7/7] Update complete!
echo.
echo Run "%PROJECT_ROOT%\run_windows.bat" to start Chatalogue.
echo.
pause
ENDLOCAL
