@echo off
rem Sets PY to a usable Python interpreter (Anaconda first, because it ships numba).
set "PY="
if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%ProgramData%\anaconda3\python.exe" set "PY=%ProgramData%\anaconda3\python.exe"
if not defined PY (where py >nul 2>nul && for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)"') do set "PY=%%i")
if not defined PY (where python >nul 2>nul && for /f "delims=" %%i in ('python -c "import sys;print(sys.executable)"') do set "PY=%%i")
if not defined PY (echo Python tidak ditemukan. & exit /b 1)
exit /b 0
