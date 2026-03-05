@echo off
setlocal
set "ROOT=%~dp0"
set "VENV_PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [ERROR] Backend venv not found at "%VENV_PY%".
  echo Run install_windows.bat from repo root first.
  exit /b 1
)

set "PYTHONUNBUFFERED=1"
"%VENV_PY%" debug_process.py
pause
