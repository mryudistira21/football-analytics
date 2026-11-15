# Football Analytics

Analyzing football matches from video: **passing, shooting, crossing, hot areas, xG and xA**.

First match: **PERSIBA Bantul vs Gresik United** (Liga Nusantara 2026/27).

## Two ways we get the numbers

| Track | What it gives | How |
|---|---|---|
| **A — Manual tagging** | passing, shooting, crossing, xG, xA, xT, progressive passes, pass networks, PPDA, field tilt, shot/pass maps, hot areas | Watch the video and click events on a pitch. Runs in a browser, no GPU. |
| **B — Automatic tracking** | player/ball positions, hot areas, later event detection | `roboflow/sports` CV models (players, ball, pitch) on the RTX 2070. |

Track A works today and needs nothing installed. Track B is the harder, GPU-based pipeline we build up over time.

## Who does what

This is a two-AI project. Keep the lanes separate so instructions don't collide.

- **Claude** — the tagging app (Pitchside Tagger), the xG/xA model (`src/xg_model.py`), heatmaps from position CSVs, and code review.
- **ChatGPT** — the local CV pipeline: environment setup on Windows, detection + tracking, exporting per-frame positions to CSV.
- **Rizki (Yudhi)** — runs everything locally, decides direction, moves files and messages between the two AIs via this repo.

The current plan, open questions and decisions live in [`TASKS.md`](TASKS.md) and [`DECISIONS.md`](DECISIONS.md). The message to hand to ChatGPT is [`collab/status.md`](collab/status.md).

## Hardware

- GPU: NVIDIA RTX 2070, 8 GB VRAM, driver 526.98 (CUDA 11.8 build of PyTorch).
- Disk: C: is nearly full, so **everything installs on D:** (`D:\Codingan\football-analytics`).

## Repo layout

```
src/        reusable code (xG + xT models, crop, heatmaps)
models/     fitted model files (xT grid)
scripts/    how the models were fitted (re-runnable)
tagger/     source of the Pitchside Tagger web app
collab/     the hand-off note for ChatGPT, kept up to date
notebooks/  experiments
data/       local only — videos, model weights, CSVs (git-ignored)
```

## Quick start (Track A)

1. Get the video (720p, H.264 so the browser can play it):
   ```powershell
   pip install -U yt-dlp
   python src/download_match.py "https://www.youtube.com/live/nCVF6TTJGXw" --name persiba_vs_gresik
   # videos save to a sibling folder (D:\Codingan\football-video), NOT inside the repo
   # only the first 15 minutes, to try things out:
   python src/download_match.py "https://www.youtube.com/live/nCVF6TTJGXw" --start 00:00 --end 15:00 --name test15
   ```
2. Open the Pitchside Tagger, set up the match, choose the video from `data/`, and tag.
3. Download the CSV into `data/`. The scripts in `src/` read it.
