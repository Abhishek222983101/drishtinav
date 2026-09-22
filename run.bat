@echo off
title DrishtiNav - Intelligent Dead Reckoning
echo.
echo  ==========================================
echo   DRISHTINAV  -  ISRO / SIH 2026 / PS 26168
echo  ==========================================
echo.
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment...
  py -3 -m venv .venv 2>nul || python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip >nul
  .venv\Scripts\python -m pip install -r requirements.txt
)
echo Dashboard:  http://localhost:5000
echo Phone app:  http://localhost:5000/app/   (phones: see README for HTTPS)
start "" "http://localhost:5000"
.venv\Scripts\python -m drishtinav serve --port 5000
pause
