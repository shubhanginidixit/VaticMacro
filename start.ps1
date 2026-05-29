# Create and use a .venv, install requirements, then run the app (PowerShell)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venv = Join-Path $root ".venv"

if (-Not (Test-Path $venv)) {
    Write-Host "Creating virtual environment at $venv..."
    python -m venv $venv
}

$python = Join-Path $venv "Scripts\python.exe"
if (-Not (Test-Path $python)) {
    Write-Host "Could not find Python in the created venv. Ensure 'python' is on PATH and try again.";
    exit 1
}

Write-Host "Upgrading pip and installing requirements..."
& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $root "requirements.txt")

Write-Host "Starting VaticMacro app..."
& $python (Join-Path $root "app.py")
