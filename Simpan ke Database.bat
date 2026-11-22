@echo off
rem Double-click: saves every match into D:\Video-Bola-Analis\football.db
rem Picks up Track B results (runs folder or *_trackb_bundle.zip) and your Tagger
rem downloads (<match_id>_events.csv, <match_id>_review.json) from your Downloads folder.
rem Safe to run again: each match is replaced, never duplicated.
setlocal
set ROOT=D:\Video-Bola-Analis
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run "Setup Track B.bat" first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\save_to_db.py --config matches\persiba_2026-27.json --root "%ROOT%" --all --inbox "%USERPROFILE%\Downloads" --inbox "%ROOT%"
echo.
echo Database: %ROOT%\football.db  (open it with "DB Browser for SQLite" or Python)
pause
