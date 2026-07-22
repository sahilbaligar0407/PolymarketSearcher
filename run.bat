@echo off
REM Double-click to run the collector.
cd /d "%~dp0"
node collect.js
echo.
pause
