@echo off
title DrishtiNav - AI Dead Reckoning System
echo.
echo  ==========================================
echo   DRISHTINAV - AI Dead Reckoning System
echo   ISRO / SIH 2026 / PS-168
echo  ==========================================
echo.
echo Starting server...
echo.
start "" "http://localhost:5000"
C:\Python314\python.exe "%~dp0backend\app.py"
pause
