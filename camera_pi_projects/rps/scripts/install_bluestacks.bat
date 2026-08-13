@echo off
echo Installing BlueStacks 5 via winget...
echo This may take several minutes. Accept any prompts that appear.
winget install --id BlueStack.BlueStacks -e --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
  echo.
  echo If winget failed, download manually: https://www.bluestacks.com
  pause
  exit /b 1
)
echo.
echo BlueStacks installed. Next steps:
echo   1. Open BlueStacks and sign in to Play Store
echo   2. Install Subway Surfers and/or Temple Run 2
echo   3. Settings - Advanced - Enable Android Debug Bridge
echo   4. Open http://127.0.0.1:8123 and start a game
pause
