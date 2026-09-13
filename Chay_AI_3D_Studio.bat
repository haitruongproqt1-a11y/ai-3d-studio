@echo off
chcp 65001 >nul
title AI 3D Studio - NVIDIA RTX 3050 (100% Offline)
cd /d "%~dp0"

echo ========================================================
echo   AI 3D STUDIO v1.6.0 - NVIDIA RTX 3050 OFFLINE
echo ========================================================
echo   Dang khoi chay ung dung AI 3D Studio...
echo.

start "" "TripoSR\.venv\Scripts\pythonw.exe" "app.py"
exit
