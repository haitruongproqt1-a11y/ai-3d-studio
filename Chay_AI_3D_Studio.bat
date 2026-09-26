@echo off
chcp 65001 >nul
title AI 3D Studio - NVIDIA RTX 3050 (SSD H:)
cd /d "%~dp0"

echo ========================================================
echo   AI 3D STUDIO v2.1.1 - NVIDIA RTX 3050 ^& Hunyuan3D Turbo
echo ========================================================
echo   Dang khoi chay ung dung AI 3D Studio...
echo.

taskkill /f /im pythonw.exe >nul 2>&1

set HF_HOME=%~dp0.cache\huggingface
set TORCH_HOME=%~dp0.cache\torch
set U2NET_HOME=%~dp0.cache\u2net
set HY3DGEN_MODELS=%~dp0.cache\hy3dgen

start "" "%~dp0TripoSR\.venv\Scripts\pythonw.exe" "%~dp0app.py"
exit
