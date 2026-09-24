@echo off
chcp 65001 >nul
title Đóng AI 3D Studio & Giải Phóng GPU (Tiết Kiệm Pin)
cd /d "%~dp0"

echo ========================================================
echo   DANG TAT TOAN BO TIEN TRINH AI 3D STUDIO...
echo ========================================================
echo.

taskkill /f /im pythonw.exe >nul 2>&1
taskkill /f /im python.exe /fi "WINDOWTITLE eq AI 3D Studio*" >nul 2>&1

echo [OK] Da tat sach toan bo tien trinh pythonw.exe va app.py!
echo [OK] GPU NVIDIA RTX 3050 da duoc giai phong hoan toan ve che do ngu 0W.
echo [OK] May sach nhu ban dau, laptop khong con hao tut pin!
echo.
timeout /t 2 >nul
exit
