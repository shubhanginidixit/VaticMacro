@echo off
REM Create a .venv if missing, install requirements, and run app (Windows CMD)
if not exist .venv (
    python -m venv .venv
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
