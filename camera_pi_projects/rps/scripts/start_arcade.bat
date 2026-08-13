@echo off
setlocal
cd /d "%~dp0.."

set PY=python

echo Stopping old servers on port 8080...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo Starting Neon Arcade...
start "Neon Arcade Server" /MIN "%PY%" src\server.py --port 8080

timeout /t 3 /nobreak >nul
start http://localhost:8080
echo Open http://localhost:8080 if browser did not launch.
