@echo off
setlocal
cd /d "%~dp0.."

set PY=python

echo Stopping old servers on port 1234...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 1234 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo Starting Neon Arcade...
start "Neon Arcade Server" /MIN "%PY%" src\server.py --port 1234

timeout /t 3 /nobreak >nul
start http://localhost:1234
echo Open http://localhost:1234 if browser did not launch.
