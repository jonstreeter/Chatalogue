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
SET "INSTALLER_VERSION=2026-03-19.1"

echo.
echo  ============================================
echo    Chatalogue Installer  v%INSTALLER_VERSION%
echo  ============================================
echo.

SET "SCRIPT_DIR=%~dp0"
IF "%SCRIPT_DIR:~-1%"=="\" SET "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

SET "REPO_URL=%CHATALOGUE_REPO_URL%"
IF "%REPO_URL%"=="" SET "REPO_URL=https://github.com/jonstreeter/Chatalogue.git"

SET "REPO_BRANCH=%CHATALOGUE_REPO_BRANCH%"
IF "%REPO_BRANCH%"=="" SET "REPO_BRANCH=main"

SET "REPO_DIR_NAME=%CHATALOGUE_REPO_DIR%"
IF "%REPO_DIR_NAME%"=="" SET "REPO_DIR_NAME=Chatalogue"

REM ===================================================================
REM  PREREQUISITE CHECKS
REM ===================================================================

SET "HAS_WINGET=0"
where winget >nul 2>&1 && SET "HAS_WINGET=1"

SET "PREREQS_OK=1"

REM --- Git ---
echo [Prerequisites] Checking Git...
where git >nul 2>&1
IF ERRORLEVEL 1 (
  echo   [!] Git is NOT installed.
  IF "!HAS_WINGET!"=="1" (
    echo.
    SET /P "INSTALL_GIT=      Would you like to install Git automatically via winget? [Y/n]: "
    IF /I "!INSTALL_GIT!"=="" SET "INSTALL_GIT=Y"
    IF /I "!INSTALL_GIT!"=="Y" (
      echo   Installing Git...
      winget install --id Git.Git -e --source winget --accept-source-agreements --accept-package-agreements
      REM Refresh PATH for this session
      FOR /F "tokens=2*" %%A IN ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') DO SET "SYS_PATH=%%B"
      FOR /F "tokens=2*" %%A IN ('reg query "HKCU\Environment" /v Path 2^>nul') DO SET "USR_PATH=%%B"
      SET "PATH=!SYS_PATH!;!USR_PATH!"
      where git >nul 2>&1
      IF ERRORLEVEL 1 (
        echo   [!] Git was installed but is not yet in your PATH.
        echo       Close this terminal, open a NEW terminal, and re-run this installer.
        SET "PREREQS_OK=0"
      ) ELSE (
        echo   [OK] Git installed successfully.
      )
    ) ELSE (
      echo   Please install Git manually: https://git-scm.com/downloads
      SET "PREREQS_OK=0"
    )
  ) ELSE (
    echo   [!] winget is not available for automatic installation.
    echo       Please install Git manually: https://git-scm.com/downloads
    SET "PREREQS_OK=0"
  )
) ELSE (
  FOR /F "tokens=3" %%V IN ('git --version 2^>nul') DO echo   [OK] Git %%V
)

REM --- Python ---
echo [Prerequisites] Checking Python...
SET "PYTHON_CMD="
where py >nul 2>&1 && SET "PYTHON_CMD=py -3"
IF "!PYTHON_CMD!"=="" (
  where python >nul 2>&1 && SET "PYTHON_CMD=python"
)
IF "!PYTHON_CMD!"=="" (
  echo   [!] Python is NOT installed.
  IF "!HAS_WINGET!"=="1" (
    echo.
    SET /P "INSTALL_PY=      Would you like to install Python 3.12 automatically via winget? [Y/n]: "
    IF /I "!INSTALL_PY!"=="" SET "INSTALL_PY=Y"
    IF /I "!INSTALL_PY!"=="Y" (
      echo   Installing Python 3.12...
      winget install --id Python.Python.3.12 -e --source winget --accept-source-agreements --accept-package-agreements
      REM Refresh PATH
      FOR /F "tokens=2*" %%A IN ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') DO SET "SYS_PATH=%%B"
      FOR /F "tokens=2*" %%A IN ('reg query "HKCU\Environment" /v Path 2^>nul') DO SET "USR_PATH=%%B"
      SET "PATH=!SYS_PATH!;!USR_PATH!"
      where py >nul 2>&1 && SET "PYTHON_CMD=py -3"
      IF "!PYTHON_CMD!"=="" (
        where python >nul 2>&1 && SET "PYTHON_CMD=python"
      )
      IF "!PYTHON_CMD!"=="" (
        echo   [!] Python was installed but is not yet in your PATH.
        echo       Close this terminal, open a NEW terminal, and re-run this installer.
        SET "PREREQS_OK=0"
      ) ELSE (
        echo   [OK] Python installed successfully.
      )
    ) ELSE (
      echo   Please install Python 3.10+ manually: https://www.python.org/downloads/
      echo   IMPORTANT: Check "Add Python to PATH" during installation.
      SET "PREREQS_OK=0"
    )
  ) ELSE (
    echo   [!] winget is not available for automatic installation.
    echo       Please install Python 3.10+ manually: https://www.python.org/downloads/
    echo       IMPORTANT: Check "Add Python to PATH" during installation.
    SET "PREREQS_OK=0"
  )
) ELSE (
  FOR /F "tokens=*" %%V IN ('!PYTHON_CMD! --version 2^>nul') DO echo   [OK] %%V
)

REM --- Node.js ---
echo [Prerequisites] Checking Node.js...
where node >nul 2>&1
IF ERRORLEVEL 1 (
  echo   [!] Node.js is NOT installed.
  IF "!HAS_WINGET!"=="1" (
    echo.
    SET /P "INSTALL_NODE=      Would you like to install Node.js LTS automatically via winget? [Y/n]: "
    IF /I "!INSTALL_NODE!"=="" SET "INSTALL_NODE=Y"
    IF /I "!INSTALL_NODE!"=="Y" (
      echo   Installing Node.js LTS...
      winget install --id OpenJS.NodeJS.LTS -e --source winget --accept-source-agreements --accept-package-agreements
      REM Refresh PATH
      FOR /F "tokens=2*" %%A IN ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') DO SET "SYS_PATH=%%B"
      FOR /F "tokens=2*" %%A IN ('reg query "HKCU\Environment" /v Path 2^>nul') DO SET "USR_PATH=%%B"
      SET "PATH=!SYS_PATH!;!USR_PATH!"
      where node >nul 2>&1
      IF ERRORLEVEL 1 (
        echo   [!] Node.js was installed but is not yet in your PATH.
        echo       Close this terminal, open a NEW terminal, and re-run this installer.
        SET "PREREQS_OK=0"
      ) ELSE (
        echo   [OK] Node.js installed successfully.
      )
    ) ELSE (
      echo   Please install Node.js 18+ manually: https://nodejs.org/en/download/
      SET "PREREQS_OK=0"
    )
  ) ELSE (
    echo   [!] winget is not available for automatic installation.
    echo       Please install Node.js 18+ manually: https://nodejs.org/en/download/
    SET "PREREQS_OK=0"
  )
) ELSE (
  FOR /F "tokens=*" %%V IN ('node --version 2^>nul') DO echo   [OK] Node.js %%V
)

REM --- ffmpeg ---
echo [Prerequisites] Checking ffmpeg...
where ffmpeg >nul 2>&1
IF ERRORLEVEL 1 (
  echo   [!] ffmpeg is NOT installed (optional but recommended).
  IF "!HAS_WINGET!"=="1" (
    echo.
    SET /P "INSTALL_FF=      Would you like to install ffmpeg automatically via winget? [Y/n]: "
    IF /I "!INSTALL_FF!"=="" SET "INSTALL_FF=Y"
    IF /I "!INSTALL_FF!"=="Y" (
      echo   Installing ffmpeg...
      winget install --id Gyan.FFmpeg -e --source winget --accept-source-agreements --accept-package-agreements
      REM Refresh PATH
      FOR /F "tokens=2*" %%A IN ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') DO SET "SYS_PATH=%%B"
      FOR /F "tokens=2*" %%A IN ('reg query "HKCU\Environment" /v Path 2^>nul') DO SET "USR_PATH=%%B"
      SET "PATH=!SYS_PATH!;!USR_PATH!"
      where ffmpeg >nul 2>&1
      IF ERRORLEVEL 1 (
        echo   [WARN] ffmpeg was installed but is not yet in your PATH.
        echo          Media conversion features may fail until PATH is refreshed.
      ) ELSE (
        echo   [OK] ffmpeg installed successfully.
      )
    ) ELSE (
      echo   [WARN] ffmpeg not installed. Download/extract will work, but media conversion may fail.
      echo          Download from: https://ffmpeg.org/download.html
    )
  ) ELSE (
    echo   [WARN] ffmpeg not installed. Download/extract will work, but media conversion may fail.
    echo          Download from: https://ffmpeg.org/download.html
  )
) ELSE (
  FOR /F "tokens=1-3" %%A IN ('ffmpeg -version 2^>nul') DO (
    IF "%%A"=="ffmpeg" echo   [OK] ffmpeg %%C
    goto :ffmpeg_done
  )
  :ffmpeg_done
)

echo [Prerequisites] Checking SoX...
where sox >nul 2>&1
IF ERRORLEVEL 1 (
  echo   [!] SoX is NOT installed (required for conversation reconstruction).
  IF "!HAS_WINGET!"=="1" (
    echo.
    SET /P "INSTALL_SOX=      Would you like to install SoX automatically via winget? [Y/n]: "
    IF /I "!INSTALL_SOX!"=="" SET "INSTALL_SOX=Y"
    IF /I "!INSTALL_SOX!"=="Y" (
      echo   Installing SoX...
      winget install --id ChrisBagwell.SoX -e --source winget --accept-source-agreements --accept-package-agreements
      FOR /F "tokens=2*" %%A IN ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') DO SET "SYS_PATH=%%B"
      FOR /F "tokens=2*" %%A IN ('reg query "HKCU\Environment" /v Path 2^>nul') DO SET "USR_PATH=%%B"
      SET "PATH=!SYS_PATH!;!USR_PATH!"
      where sox >nul 2>&1
      IF ERRORLEVEL 1 (
        echo   [WARN] SoX was installed but is not yet in your PATH.
        echo          Conversation reconstruction may not work until PATH is refreshed.
      ) ELSE (
        echo   [OK] SoX installed successfully.
      )
    ) ELSE (
      echo   [WARN] SoX not installed. Conversation reconstruction will be unavailable.
      echo          Download from: https://sourceforge.net/projects/sox/files/sox/
    )
  ) ELSE (
    echo   [WARN] SoX not installed. Conversation reconstruction will be unavailable.
    echo          Download from: https://sourceforge.net/projects/sox/files/sox/
  )
) ELSE (
  FOR /F "tokens=1-3" %%A IN ('sox --version 2^>nul') DO (
    IF /I "%%A"=="sox:" echo   [OK] SoX %%C
    IF /I "%%A"=="SoX" echo   [OK] SoX %%B
    goto :sox_done
  )
  :sox_done
)

echo.
IF "!PREREQS_OK!"=="0" (
  echo [ERROR] One or more required prerequisites are missing.
  echo         Install the missing tools listed above, then re-run this installer.
  ENDLOCAL
  EXIT /B 1
)
echo [Prerequisites] All checks passed.
echo.

REM ===================================================================
REM  REPOSITORY SETUP
REM ===================================================================

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

REM ===================================================================
REM  INSTALL DEPENDENCIES
REM ===================================================================

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

echo [1/8] Creating backend virtual environment...
IF NOT EXIST "%VENV_PYTHON%" (
  cd /d "%BACKEND_DIR%"
  %PYTHON_CMD% -m venv .venv
)

echo [2/8] Upgrading pip tooling...
"%VENV_PYTHON%" -m pip install --upgrade pip wheel "setuptools<81"

echo [3/8] Installing PyTorch nightly cu128 (RTX 50xx friendly)...
"%PIP_CMD%" install --pre -r "%BACKEND_DIR%\requirements-windows-cu128.txt"
IF ERRORLEVEL 1 (
  echo [WARN] Nightly cu128 install failed. Falling back to stable torch pins...
  "%PIP_CMD%" install -r "%BACKEND_DIR%\requirements-macos.txt"
)
echo     Validating torch stack...
"%VENV_PYTHON%" "%BACKEND_DIR%\tools\check_torch_stack.py"
IF ERRORLEVEL 1 (
  echo [ERROR] Torch validation failed.
  echo         An NVIDIA GPU was detected, but this venv cannot use CUDA.
  echo         Continuing would force transcription onto CPU and make jobs much slower.
  ENDLOCAL
  EXIT /B 1
)

echo [4/8] Installing core backend dependencies...
"%PIP_CMD%" install -r "%BACKEND_DIR%\requirements.txt"

echo [5/8] Installing pyannote stack...
"%PIP_CMD%" install pyannote.audio==4.0.4 --no-deps
IF ERRORLEVEL 1 (
  echo [ERROR] Failed to install pyannote.audio.
  ENDLOCAL
  EXIT /B 1
)
echo     Normalizing pyannote package metadata...
"%VENV_PYTHON%" "%BACKEND_DIR%\tools\repair_pyannote_metadata.py"
IF ERRORLEVEL 1 (
  echo [ERROR] pyannote metadata normalization failed.
  ENDLOCAL
  EXIT /B 1
)

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

REM ===================================================================
REM  DONE — SUMMARY
REM ===================================================================

echo.
echo  ============================================
echo    Installation complete!
echo  ============================================
echo.
echo  Project root : %PROJECT_ROOT%
echo  Backend venv : %VENV_DIR%
echo.
echo  HOW TO START THE APP:
echo  ---------------------
IF EXIST "%PROJECT_ROOT%\run_windows.bat" (
  echo   Double-click or run:
  echo     "%PROJECT_ROOT%\run_windows.bat"
) ELSE (
  echo   Start backend/frontend manually from %PROJECT_ROOT%.
)
echo.
echo  Once running, open your browser to:
echo    Frontend : http://localhost:5173
echo    API docs : http://localhost:8011/docs
echo.
echo  HOW TO UPDATE LATER:
echo  --------------------
IF EXIST "%PROJECT_ROOT%\update_windows.bat" (
  echo   Run: "%PROJECT_ROOT%\update_windows.bat"
) ELSE (
  echo   cd into %PROJECT_ROOT% and run: git pull
)
echo.
echo  CONFIGURATION:
echo  --------------
echo   Copy backend\.env.example to backend\.env and set:
echo     HF_TOKEN  = your Hugging Face token (for diarization models)
echo.
ENDLOCAL
