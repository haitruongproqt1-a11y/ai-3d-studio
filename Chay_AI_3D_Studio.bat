@echo off
chcp 65001 >nul
title AI 3D Studio - NVIDIA RTX 3050 (SSD H:)
cd /d "%~dp0"

echo ========================================================
echo   AI 3D STUDIO v1.7.0 - NVIDIA RTX 3050 OFFLINE (SSD H:)
echo ========================================================
echo   Dang khoi chay ung dung AI 3D Studio...
echo.

set HF_HOME=%~dp0.cache\huggingface
set TORCH_HOME=%~dp0.cache\torch
set U2NET_HOME=%~dp0.cache\u2net

start "" "%~dp0TripoSR\.venv\Scripts\pythonw.exe" "%~dp0app.py"
exit
