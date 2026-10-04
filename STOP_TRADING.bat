@echo off
REM ============================================================================
REM  MarketLab EMERGENCY STOP.
REM  Engages the global kill switch (the risk engine refuses every new position, for
REM  every strategy, regardless of any model or config) and stops the daemon. The
REM  START_TRADING.bat watchdog sees the switch and does not restart.
REM  Resume later with START_TRADING.bat.
REM ============================================================================
setlocal
cd /d "%~dp0"
uv run marketlab kill --reason "STOP_TRADING.bat"
echo.
echo Trading stopped. The dashboard (if open) keeps showing the last state.
pause
endlocal
