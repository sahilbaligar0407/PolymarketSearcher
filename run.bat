@echo off
REM Double-click to launch the dashboard at http://localhost:5173
cd /d "%~dp0"
start "" http://localhost:5173
node server.js
pause
