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
SET "BACKEND_LOG=%BACKEND_DIR%\uvicorn.out.log"
SET "FRONTEND_LOG=%FRONTEND_DIR%\vite.out.log"

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

:: Ensure backend/frontend dependencies exist
IF NOT EXIST "%VENV_PYTHON%" (
    echo Backend venv missing. Running install_windows.bat...
    call "%PROJECT_ROOT%install_windows.bat"
    IF !ERRORLEVEL! NEQ 0 (
        echo install_windows.bat failed. Aborting startup.
        pause
        exit /b 1
    )
)

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
    call npm install --fund=false --audit=false --loglevel=warn
)

:: Start Backend (worker thread starts automatically inside the API server)
echo Starting Backend Server on http://localhost:%BACKEND_PORT% ...
start /B "" cmd.exe /c ""%VENV_PYTHON%" -m uvicorn src.main:app --app-dir "%BACKEND_DIR%" --host 0.0.0.0 --port %BACKEND_PORT% 1>"%BACKEND_LOG%" 2>&1"

:: Wait for backend readiness before starting frontend
echo Waiting for backend to become ready (first run may take a few minutes to set up PostgreSQL)...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$url='http://127.0.0.1:%BACKEND_PORT%/system/worker-status'; $deadline=(Get-Date).AddSeconds(300); $ok=$false; while((Get-Date)-lt $deadline){ try { $r=Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2; if($r.StatusCode -ge 200 -and $r.StatusCode -lt 300){ $ok=$true; break } } catch {} Start-Sleep -Milliseconds 800 }; if(-not $ok){ exit 1 }"
IF !ERRORLEVEL! NEQ 0 (
    echo Backend did not become ready in time.
    echo Check backend logs/terminal, then re-run this script.
    pause
    exit /b 1
)

:: Start Frontend
echo Starting Frontend on http://localhost:5173 ...
start /B "" cmd.exe /c "cd /d "%FRONTEND_DIR%" && npm run dev 1>"%FRONTEND_LOG%" 2>&1"

:: Wait for frontend dev server to bind, then open browser
timeout /t 2 /nobreak >nul
start http://localhost:5173

echo.
echo ===================================================
echo   %PROJECT_NAME% is running
echo.
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:%BACKEND_PORT%
echo.
echo   Logs:
echo     Backend:  %BACKEND_LOG%
echo     Frontend: %FRONTEND_LOG%
echo ===================================================
echo.
echo Press any key to stop all servers...
pause >nul
echo.
echo Stopping servers...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%BACKEND_PORT%.*LISTENING" 2^>nul') do (
    taskkill /F /T /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173.*LISTENING" 2^>nul') do (
    taskkill /F /T /PID %%a >nul 2>&1
)
echo Done.
