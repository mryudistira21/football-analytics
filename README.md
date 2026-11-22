# Football Analytics

Analyzing football matches from video: **passing, shooting, crossing, hot areas, xG and xA**.

First match: **PERSIBA Bantul vs Gresik United** (Liga Nusantara 2026/27).

## Two ways we get the numbers

| Track | What it gives | How |
|---|---|---|
| **A — Manual tagging** | passing, shooting, crossing, xG, xA, xT, progressive passes, pass networks, PPDA, field tilt, shot/pass maps, hot areas, **per-player & team moment reels, telestration, confidence flags** | Watch the video and click events on a pitch. Runs in a browser, no GPU. |
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
hub/        League Hub web app (clubs, squads, fixtures, standings, scorers, your analysis)
collab/     the hand-off note for ChatGPT, kept up to date
notebooks/  Colab notebooks (match_colab.ipynb = production runner)
matches/    match configs: period times, direction, video URL (persiba_2026-27.json)
tests/      Track B unit tests
data/       league data (clubs/players/fixtures/standings/scorers *.json — tracked)
            + local media (videos, model weights, CSVs — git-ignored)
```

## Quick start (Track A)

One-time setup (PowerShell, in the repo folder):
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U "yt-dlp[default]"
winget install Gyan.FFmpeg DenoLand.Deno      # then open a NEW window
setx FOOTBALL_VIDEO_DIR D:\Video-Bola-Analis   # videos live outside the repo
```

Each match:
1. Download (720p, 30 fps, no re-encode):
   `python src/download_match.py "URL" --start 17:00 --end 37:00 --name test20`
2. Squads: `data/clubs.json` holds all 23 Liga Nusantara rosters (built by `scripts/build_clubs_json.py`; or `python src/export_squads.py` from your own `club_db.py`). In the Tagger setup use **Load squads JSON** → **From squads → Pick XI** (tap GK, pick 11).
3. Open the Pitchside Tagger, set up the match, choose the video, tag.
4. Download the CSV (format in `DATA_CONTRACT.md`), then import it in the **League Hub** (`hub/`) to see the match in the league context.

## Quick start (Track B)
**Real matches:** follow [MATCH_WORKFLOW.md](MATCH_WORKFLOW.md) — `notebooks/match_colab.ipynb` on Colab, or `scripts/run_match.py` locally, driven by `matches/persiba_2026-27.json`.

Automatic video tracking is in [TRACK_B.md](TRACK_B.md): model downloads, the 30-second smoke test, GPU/VRAM settings, CSV output and known limitations. Heavy full-match inference runs on Colab (`notebooks/track_b_colab.ipynb`); Track B then exports per-track cut-lists (`scripts/build_player_moments.py`) that import into the Tagger's Reel ("Imported (Track B)").
