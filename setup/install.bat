@echo off
setlocal
REM Creates .venv in the DOG-GS folder and installs DOG-GS with Numba, OpenCV and PySide6.
cd /d "%~dp0.."
set "PY="
where py >nul 2>nul && (for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)"') do set "PY=%%i")
if not defined PY (where python >nul 2>nul && (for /f "delims=" %%i in ('python -c "import sys;print(sys.executable)"') do set "PY=%%i"))
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY (
  echo [DOG-GS] Python 3.9+ was not found. Install it from https://www.python.org/downloads/ and run DOG-GS.bat again.
  exit /b 1
)
echo [DOG-GS] Using %PY%
"%PY%" -m venv .venv || exit /b 1
".venv\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1
".venv\Scripts\python.exe" -m pip install -e ".[all,gui]" || exit /b 1
".venv\Scripts\python.exe" -m doggs.cli info
echo [DOG-GS] Setup complete.
exit /b 0
