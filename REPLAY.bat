@echo off
REM ============================================================================
REM  Replay a recorded session through the strategies (no look-ahead).
REM  Usage:  REPLAY.bat 2026-10-04        (a UTC date with recorded data)
REM          REPLAY.bat                   (lists the dates available)
REM ============================================================================
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  uv run marketlab replay --list
) else (
  uv run marketlab replay %*
)
pause
endlocal
