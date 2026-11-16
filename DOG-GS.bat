@echo off
setlocal
REM DOG-GS - one-click launcher. First run prepares everything inside this folder (.venv).
REM Tip: drag a GeoSplat project folder or a Gaussian .PLY onto this file to open it directly.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo [DOG-GS] First-run setup - please wait...
  call "%~dp0setup\install.bat"
  if errorlevel 1 ( echo. & echo [DOG-GS] Setup failed - see the messages above. & pause & exit /b 1 )
)
start "" ".venv\Scripts\pythonw.exe" -m doggs.gui %*
endlocal
