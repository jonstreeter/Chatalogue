@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "FRONTEND_DIR=%PROJECT_ROOT%frontend"

where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Node.js 18+ is required and was not found in PATH.
  exit /b 1
)

if not exist "%FRONTEND_DIR%\node_modules" (
  echo Frontend dependencies missing. Running npm install...
  cd /d "%FRONTEND_DIR%"
  npm install
)

cd /d "%FRONTEND_DIR%"
npm run dev
