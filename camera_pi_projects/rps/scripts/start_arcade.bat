@echo off
setlocal
cd /d "%~dp0.."

set PY=E:\SoftComputing\sketch-matcher-pi\sketch_matcher_env\Scripts\python.exe
if not exist "%PY%" set PY=python

echo Stopping old servers on port 8123...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8123 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo Starting Neon Arcade (wait ~15s for model load)...
start "Neon Arcade Server" /MIN "%PY%" src\server.py --port 8123

timeout /t 15 /nobreak >nul
start http://127.0.0.1:8123
echo Open http://127.0.0.1:8123 if browser did not launch.
