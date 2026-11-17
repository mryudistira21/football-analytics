@echo off
rem Double-click to open the match downloader window (no terminal typing).
rem Uses the project's own Python venv so yt-dlp is available.
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" "src\download_gui.pyw"
) else (
  echo Virtual environment not found at .venv
  echo Create it first:  py -3.10 -m venv .venv  ^&^&  .\.venv\Scripts\Activate.ps1  ^&^&  pip install -U "yt-dlp[default]"
  pause
)
