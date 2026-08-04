@echo off
echo ============================================================
echo Sketch Matcher - Local GPU Training Launcher
echo ============================================================
echo.
echo Installing dependencies (first time only)...
pip install -r requirements-gpu.txt
if errorlevel 1 (
    echo Failed to install dependencies
    pause
    exit /b 1
)
echo.
echo Starting full training pipeline...
echo This will run: download - preprocess - train(300ep) - evaluate - export
echo.
python run_training.py
if errorlevel 1 (
    echo Training pipeline failed
    pause
    exit /b 1
)
echo.
echo ============================================================
echo ALL DONE! pi_deploy.zip is ready for your Raspberry Pi.
echo ============================================================
pause
