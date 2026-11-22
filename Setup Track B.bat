@echo off
rem Double-click once: installs everything Track B needs on this PC (on D:, not C:).
rem Safe to run again; it skips finished steps. Takes 10-30 min (mostly the PyTorch download).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_track_b.ps1"
pause
