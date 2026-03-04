@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "FRONTEND_DIR=%PROJECT_ROOT%frontend"
cd /d "%FRONTEND_DIR%"
npm run dev
