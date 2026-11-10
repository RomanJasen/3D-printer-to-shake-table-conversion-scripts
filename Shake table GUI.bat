@echo off
rem Double-click to open the shake-table window. This console shows errors, if any.
cd /d "%~dp0"
python shake_gui.py
if errorlevel 1 (
    echo.
    echo The window could not start. It needs Python 3 with numpy and matplotlib:
    echo     pip install numpy matplotlib
    pause
)
