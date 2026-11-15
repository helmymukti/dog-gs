@echo off
setlocal
title DOG-GS installer
cd /d "%~dp0.."
call "%~dp0_find_python.bat" || goto :fail
echo Using Python: %PY%
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -e ".[all]"
if errorlevel 1 goto :fail
echo.
"%PY%" -m doggs.cli info
echo.
echo Instalasi selesai. Seret file .ply atau folder GeoSplat ke "DOG-GS_render.bat".
pause
exit /b 0
:fail
echo.
echo Instalasi gagal. Pastikan Python 3.9+ atau Anaconda terpasang dan ada koneksi internet.
pause
exit /b 1
