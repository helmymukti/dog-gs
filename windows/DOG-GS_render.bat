@echo off
setlocal EnableDelayedExpansion
title DOG-GS - Direct Orthophoto Generation
rem Seret (drag-and-drop) file .ply hasil Gaussian Splatting, atau folder proyek GeoSplat, ke file ini.
call "%~dp0_find_python.bat" || (pause & exit /b 1)
if "%~1"=="" (
  echo Seret file .ply atau folder GeoSplat ke file ini, atau ketik path-nya:
  set /p "INPUT=Path: "
) else (
  set "INPUT=%~1"
)
set "INPUT=!INPUT:"=!"
echo.
echo Input : !INPUT!
echo.
set "GSD=0.06"
set /p "GSD=GSD orthophoto dalam meter [0.06]: "
set "ZMIN="
set /p "ZMIN=Buang Gaussian di bawah tinggi (m, kosong = tidak) []: "
set "ZMAX="
set /p "ZMAX=Buang Gaussian di atas tinggi / floater (m, kosong = tidak) []: "
set "COL=sh"
set /p "COL=Warna: sh (nadir) atau dc [sh]: "
set "ARGS=--gsd !GSD! --colour !COL!"
if not "!ZMIN!"=="" set "ARGS=!ARGS! --zmin !ZMIN!"
if not "!ZMAX!"=="" set "ARGS=!ARGS! --zmax !ZMAX!"
echo.
"%PY%" -m doggs.cli render "!INPUT!" !ARGS!
if errorlevel 1 (
  echo.
  echo Proses gagal. Jalankan install_DOG-GS.bat terlebih dahulu bila belum.
  pause
  exit /b 1
)
echo.
echo Selesai.
if exist "!INPUT!\11_DOG-GS" (start "" explorer "!INPUT!\11_DOG-GS") else (for %%F in ("!INPUT!") do if exist "%%~dpFDOG-GS_output" start "" explorer "%%~dpFDOG-GS_output")
pause
