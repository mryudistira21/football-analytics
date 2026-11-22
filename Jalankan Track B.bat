@echo off
rem Double-click: runs Track B on every match in the config that has period times,
rem in the background-friendly "light" profile (idle CPU priority, 2 CPU cores,
rem half the GPU memory, short GPU rests, RAM safety stop). Resumable: close the
rem window any time and double-click again later to continue.
rem Change PROFILE to balanced (faster) or full (fastest, PC busy).
setlocal
set PROFILE=light
set ROOT=D:\Video-Bola-Analis
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found at .venv - see MATCH_WORKFLOW.md, "Laptop" section.
  pause
  exit /b 1
)
title Track B (%PROFILE%) - you can minimise this window
".venv\Scripts\python.exe" scripts\run_queue.py --config matches\persiba_2026-27.json --matches ready --root "%ROOT%" --download --profile %PROFILE%
echo.
echo Finished. Results: %ROOT%\runs\^<match_id^>\^<match_id^>_trackb_bundle.zip
pause
