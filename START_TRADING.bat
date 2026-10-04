@echo off
REM ============================================================================
REM  MarketLab - start the 24/7 PAPER trading tournament.
REM
REM  Starts: Ollama (local AI), the read-only dashboard, and the daemon (collectors,
REM  Polymarket tracker, Kalshi feeds, strategy workers, AI tiers, paper exchange,
REM  monitoring). The daemon runs under a watchdog: if it crashes or the network drops
REM  it is restarted after 30s, resuming every sleeve's bankroll from the database.
REM
REM  PAPER ONLY. Real Kalshi order placement is hard-disabled.
REM  (ping is the delay: `timeout` refuses to run without an interactive console.)
REM  Emergency stop: double-click STOP_TRADING.bat (or run: uv run marketlab kill)
REM ============================================================================
setlocal
cd /d "%~dp0"
title MarketLab daemon (PAPER) - close this window to stop

where uv >nul 2>nul
if errorlevel 1 (
  echo [!] uv is not installed. Install it with:  pip install uv
  pause
  exit /b 1
)

echo [1/4] Syncing Python environment...
uv sync --extra dev --quiet
if errorlevel 1 (
  echo [!] uv sync failed.
  pause
  exit /b 1
)

echo [2/4] Checking local AI (Ollama)...
powershell -NoProfile -Command "try { Invoke-RestMethod -TimeoutSec 3 http://localhost:11434/api/tags | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    echo       starting Ollama...
    start "" /min "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
    ping -n 7 127.0.0.1 >nul
  ) else (
    echo       Ollama not found - AI arms will be skipped, everything else runs.
  )
)

if exist "data\KILL_SWITCH" (
  echo [!] The kill switch was engaged:
  type "data\KILL_SWITCH"
  echo     Starting the tournament clears it.
  uv run marketlab unkill
)

echo [3/4] Starting dashboard at http://127.0.0.1:8765/
start "MarketLab dashboard" /min cmd /c "uv run marketlab dashboard --port 8765"
ping -n 4 127.0.0.1 >nul
start "" http://127.0.0.1:8765/

echo [4/4] Starting the paper-trading daemon (watchdog mode)...
:loop
uv run marketlab run
if exist "data\KILL_SWITCH" (
  echo Kill switch engaged - not restarting. Run START_TRADING.bat to resume.
  goto end
)
echo Daemon exited (code %errorlevel%). Restarting in 30 seconds... (close this window to stop)
ping -n 31 127.0.0.1 >nul
goto loop

:end
pause
endlocal
