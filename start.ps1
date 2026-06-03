# VaticMacro - Web Application Startup Script (PowerShell)

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   VaticMacro Inflation Forecasting" -ForegroundColor Cyan
Write-Host "       Web Application Starter" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if .venv exists
if (-not (Test-Path ".venv")) {
    Write-Host "Error: Virtual environment not found" -ForegroundColor Red
    Write-Host "Please run: python -m venv .venv" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1

# Check if app.py exists
if (-not (Test-Path "app.py")) {
    Write-Host "Error: app.py not found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Start Flask app
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Starting Flask Application..." -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Open browser to: http://localhost:5000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

python app.py

Read-Host "Press Enter to exit"
