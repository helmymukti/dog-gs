@echo off
setlocal EnableDelayedExpansion
title DOG-GS command-line render
rem Command-line alternative to the app: drag a Gaussian .PLY or a GeoSplat project folder onto this file.
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  call "%~dp0..\setup\install.bat" || (pause & exit /b 1)
)
if "%~1"=="" (
  set /p "INPUT=Path to .ply or GeoSplat folder: "
) else (
  set "INPUT=%~1"
)
set "INPUT=!INPUT:"=!"
set "GSD=0.06"
set /p "GSD=Orthophoto GSD in metres [0.06]: "
set "ZMIN=auto"
set /p "ZMIN=Lower height limit in metres, auto or none [auto]: "
set "ZMAX=auto"
set /p "ZMAX=Upper height limit in metres (floaters), auto or none [auto]: "
set "ARGS=--gsd !GSD!"
if /i not "!ZMIN!"=="none" set "ARGS=!ARGS! --zmin !ZMIN!"
if /i not "!ZMAX!"=="none" set "ARGS=!ARGS! --zmax !ZMAX!"
".venv\Scripts\python.exe" -m doggs.cli render "!INPUT!" !ARGS!
pause
