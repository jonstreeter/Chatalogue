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

:: Clean up any previous Chatalogue processes from an earlier run.
echo Checking for a previous Chatalogue session...
echo   This only closes older Chatalogue backend/frontend processes if they are still running.
taskkill /F /T /FI "WINDOWTITLE eq %PROJECT_NAME% Backend*" >nul 2>&1
taskkill /F /T /FI "WINDOWTITLE eq %PROJECT_NAME% Frontend*" >nul 2>&1
:: Kill backend uvicorn processes by command line (catches --reload parent/child even if not listening)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$backend='%BACKEND_DIR%'; Get-CimInstance Win32_Process | Where-Object { ($_.Name -ieq 'python.exe' -or $_.Name -ieq 'pythonw.exe') -and $_.CommandLine -like '*uvicorn*src.main:app*' -and $_.CommandLine -like ('*' + $backend + '*') } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {} }" >nul 2>&1
:: Kill backend python processes that are running this repo app even if uvicorn commandline matching misses them
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$backend='%BACKEND_DIR%'; Get-CimInstance Win32_Process | Where-Object { ($_.Name -ieq 'python.exe' -or $_.Name -ieq 'pythonw.exe') -and $_.CommandLine -like ('*' + $backend + '*') -and ($_.CommandLine -like '*src.main:app*' -or $_.CommandLine -like '*--app-dir*' + $backend + '*') } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {} }" >nul 2>&1
:: Kill frontend dev server wrappers/processes by command line (vite/npm dev)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$frontend='%FRONTEND_DIR%'; Get-CimInstance Win32_Process | Where-Object { (($_.Name -ieq 'node.exe') -or ($_.Name -ieq 'cmd.exe')) -and $_.CommandLine -like ('*' + $frontend + '*') -and ($_.CommandLine -like '*vite*' -or $_.CommandLine -like '*npm*run*dev*') } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {} }" >nul 2>&1
:: Kill stale embedded postgres processes tied to this repo data dir
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$pgData='%BACKEND_DIR%\data\postgres'; $pgDataFwd=($pgData -replace '\\','/'); Get-CimInstance Win32_Process | Where-Object { $_.Name -ieq 'postgres.exe' -and (($_.CommandLine -like ('*' + $pgData + '*')) -or ($_.CommandLine -like ('*' + $pgDataFwd + '*')) -or ($_.CommandLine -like '*embedded-postgres-binaries*')) } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {} }" >nul 2>&1
:: Also kill by port in case window titles don't match
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%BACKEND_PORT%.*LISTENING" 2^>nul') do (
    taskkill /F /T /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173.*LISTENING" 2^>nul') do (
    taskkill /F /T /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

:: Reset per-run launcher logs so stale tracebacks from an older session do not
:: trip the readiness probe for the current startup attempt.
if exist "%BACKEND_LOG%" del /f /q "%BACKEND_LOG%" >nul 2>&1
if exist "%FRONTEND_LOG%" del /f /q "%FRONTEND_LOG%" >nul 2>&1

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

:: Start Backend with auto-restart on CUDA faults (exit code 75)
echo Starting Backend Server on http://localhost:%BACKEND_PORT% ...
SET "BACKEND_RESTART_SCRIPT=%BACKEND_DIR%\runtime\backend_restart_loop.bat"
(
echo @echo off
echo :restart_loop
echo del /f /q "%BACKEND_DIR%\runtime\cuda_restart_requested" ^>nul 2^>^&1
echo del /f /q "%BACKEND_DIR%\runtime\backend.instance.lock" ^>nul 2^>^&1
echo "%VENV_PYTHON%" -m uvicorn src.main:app --app-dir "%BACKEND_DIR%" --host 0.0.0.0 --port %BACKEND_PORT% 1^>^>"%BACKEND_LOG%" 2^>^&1
echo if %%ERRORLEVEL%% EQU 75 ^(
echo     echo [%%date%% %%time%%] Backend exited for CUDA auto-restart ^(code 75^). Respawning... ^>^>"%BACKEND_LOG%"
echo     timeout /t 3 /nobreak ^>nul
echo     goto restart_loop
echo ^)
) > "%BACKEND_RESTART_SCRIPT%"
start /B "" cmd.exe /c ""%BACKEND_RESTART_SCRIPT%""

:: Wait for backend readiness before starting frontend
echo Waiting for backend to become ready...
echo   If PostgreSQL was interrupted previously, crash recovery can take several minutes.
echo   Do not close this window while recovery is in progress or startup will restart from the beginning.
set "PGLOG_PATH=%BACKEND_DIR%\runtime\postgres\postgres.log"
set "PGLOG_SIZE_BEFORE=0"
for %%I in ("%PGLOG_PATH%") do if exist "%%~fI" set "PGLOG_SIZE_BEFORE=%%~zI"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$url='http://127.0.0.1:%BACKEND_PORT%/system/worker-status'; $log='%BACKEND_LOG%'; $pglog='%PGLOG_PATH%'; $pgStartOffset=[int64]('%PGLOG_SIZE_BEFORE%'); $deadline=(Get-Date).AddSeconds(900); $ok=$false; $fatal=$false; $lastRecoveryNotice=''; while((Get-Date)-lt $deadline){ try { $r=Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2; if($r.StatusCode -ge 200 -and $r.StatusCode -lt 300){ $ok=$true; break } } catch {}; if(Test-Path $log){ try { $tail=((Get-Content -Path $log -Tail 120 -ErrorAction SilentlyContinue) -join [Environment]::NewLine); if(($tail -match 'Traceback \(most recent call last\)') -or ($tail -match 'EmbeddedPostgresError') -or ($tail -match 'ConnectionTimeout') -or ($tail -match 'connection timeout expired') -or ($tail -match 'did not become query-ready in time') -or ($tail -match 'pre-existing shared memory block is still in use') -or ($tail -match 'error while attempting to bind on address')){ $fatal=$true; break } } catch {} }; if(Test-Path $pglog){ try { $content=(Get-Content -Path $pglog -Raw -Encoding UTF8 -ErrorAction SilentlyContinue); if($null -ne $content){ if($content.Length -gt $pgStartOffset){ $recent=$content.Substring([Math]::Min($pgStartOffset, $content.Length)); $recoverLines=$recent -split \"`r?`n\" | Where-Object { $_ -match 'syncing data directory \(fsync\), elapsed time:|database system is ready to accept connections|automatic recovery in progress' }; $msg=$recoverLines | Select-Object -Last 1; if($msg){ $msg=$msg.Trim(); if($msg -ne $lastRecoveryNotice){ Write-Host ('  [postgres] ' + $msg); $lastRecoveryNotice=$msg } } } } } catch {} }; Start-Sleep -Milliseconds 800 }; if($ok){ exit 0 }; if($fatal){ exit 2 }; exit 1"
set "BACKEND_READY_RC=%ERRORLEVEL%"
if not "%BACKEND_READY_RC%"=="0" goto :backend_start_failed
goto :backend_start_ready

:backend_start_failed
if "%BACKEND_READY_RC%"=="2" (
    echo Backend failed during startup.
    echo Recent backend log:
    powershell -NoProfile -ExecutionPolicy Bypass -Command "if(Test-Path '%BACKEND_LOG%'){ Get-Content -Path '%BACKEND_LOG%' -Tail 80 }"
) else (
    echo Backend did not become ready in time.
)
echo Check backend logs/terminal, then re-run this script.
pause
exit /b 1

:backend_start_ready

:: Start Frontend
echo Starting Frontend on http://localhost:5173 ...
start "Chatalogue Frontend" /B cmd.exe /c "cd /d ""%FRONTEND_DIR%"" && npm run dev 1>""%FRONTEND_LOG%"" 2>&1"

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
