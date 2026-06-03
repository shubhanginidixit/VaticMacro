@echo off
REM VaticMacro - Web Application Startup Script

echo.
echo =========================================
echo    VaticMacro Inflation Forecasting
echo         Web Application Starter
echo =========================================
echo.

REM Check if .venv exists
if not exist ".venv" (
    echo Error: Virtual environment not found
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if app.py exists
if not exist "app.py" (
    echo Error: app.py not found
    pause
    exit /b 1
)

REM Start Flask app
echo.
echo =========================================
echo Starting Flask Application...
echo =========================================
echo.
echo Open browser to: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
echo =========================================
echo.

python app.py

pause
