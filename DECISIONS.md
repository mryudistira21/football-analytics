# Decisions

Short log of choices, so neither AI re-opens a settled question. Newest first.

## 2026-11-15 — After ChatGPT's second review (integration pass)
- **xT methodology (Finding E):** now excludes WHOLE set-piece possessions via StatsBomb `play_pattern`, not just the delivery, for passes, carries and shots. Kept patterns: Regular Play, From Counter, From Keeper, From Kick Off. Training moves 411k → 224k; near-goal peak 0.25 → 0.32. "Open play only" is now literally true. Grid refitted; Tagger and models/xt_grid.json stay identical (192 cells).
- **Stable match_id (Finding B):** event CSV now starts `match_id, match_name, ...`. `match_id` is the Tagger's immutable id; joins use it, never the display name.
- **video_seconds always captured (Finding C):** recorded whenever a video is loaded, in both time modes; empty only with no video. This is the join key to tracking.
- **Downloader output path (Finding A):** `resolve_output()` prefers the merged `filepath`, then falls back to the newest matching file on disk. Smoke-tested with a mock.
- **Storage edge cases (Finding F1):** corrupted local match-list/events now warn instead of looking like an empty match; a failed load blocks saves so unreadable data is never overwritten.
- **Assist by match time (Finding F2):** candidateAssist orders by clock, not tag order, so seeking back in the video and tagging late still links correctly.
- **Extra-time direction (Finding D):** contract now carries `home_attacks_right_by_period`; periods 3–4 get a fresh coin toss, never inferred from period 1.
- **Join tolerance:** contract ties it to `frame_stride / fps`, not a fixed ±0.5 s.
- **30 fps is best-effort:** documented; if no ≤30 fps stream exists the download keeps the higher rate (plays fine) and the note says how to bring it down.

## 2026-11-15 — After ChatGPT's source review
- Accepted and fixed: set-piece filter in `fit_xt.py` (refit: open play only, 412k moves), `fit_xt.py` main() + mkdir, Tagger localStorage error surfacing, xA = pass whose receiver is the shooter, CSV import validation, exact xG coefficients in the Tagger, PPDA labelled as approximation, downloader real output path, measured crop.
- Not changed: StatsBomb coordinates are already oriented with the acting team attacking toward x = 120, so `fit_xt.py` needs no per-team flip.
- **`DATA_CONTRACT.md` is the shared format** between tracks. Tracking = camera frame; events = team frame; `video_seconds` joins them; `track_id` ≠ `shirt_number`.
- Event CSV gains a `side` column (`home`/`away`).

## 2026-11-15 — Light downloads and processing
- Download 720p at **30 fps** (YouTube's own 30 fps stream, no re-encode). 60 fps adds nothing for tagging and doubles frames.
- Trim with `--start/--end` **without re-encoding** (keyframe cut, a few seconds of slack). `--exact` re-encodes and is opt-in.
- Any re-encode (crop, exact trim) uses the `veryfast` preset and **4 threads** by default, so the PC stays usable.
- Disk space is checked before downloading. Videos live outside the repo.
- yt-dlp needs a JS runtime: install **Deno** (`winget install DenoLand.Deno`) and `yt-dlp[default]`.
- yt-dlp warns that Python 3.10 is deprecated. It still works; if a future yt-dlp drops 3.10, keep downloads in a separate small 3.13 venv and leave Track B on 3.10.

## 2026-11-15 — Club and player database is local only
`src/club_db.py` (SQLite) holds clubs, squads per season and matches. The database file (`data/football.db` or `FOOTBALL_DB`) and exported lineups are git-ignored and never pushed. Only the script is shared.

## 2026-11-15 — Tagger v2 scope and definitions
Full-match tagging is the plan, so speed matters: digits set player # (and receiver # after the end click), "Follow the ball" makes the receiver the next passer and switches team after a lost pass, `A` starts the next event where the ball last was.
- **xT:** 16 x 12 grid, StatsBomb open data (230 matches), value iteration over passes and carries. Completed passes earn xT(end) − xT(start); failed passes earn 0.
- **Progressive pass:** ≥30 m closer to goal inside own half, ≥15 m crossing halfway, ≥10 m in opponent half (Wyscout-style).
- **PPDA:** opponent passes in their own 60% ÷ our tackles + interceptions + fouls in that zone.
- **Field tilt:** share of both teams' final-third passes.
- **Possession:** pass share (we don't tag ball-in-play time).

## 2026-11-15 — Research: what to adopt
- **socceraction** (SPADL format, VAEP, xT) and **kloppy** (format conversion) are the target ecosystem; our CSV should export to SPADL.
- **mplsoccer** for publication-quality plots.
- **SoccerNet ball-action spotting** can detect Pass/Cross/Shot/Tackle from broadcast, but top scores are ~0.45–0.55 F1 (2026). Not reliable enough to replace tagging; maybe later as a pre-fill.

## 2026-11-14 — Python version: 3.10 or 3.11, not 3.13
Use **Python 3.10** (3.11 is also fine). Reason: the Track B stack (PyTorch cu118, ultralytics, and roboflow/sports' `umap-learn` → `numba`/`llvmlite`) has the most reliable Windows wheels on 3.10/3.11. 3.12 usually works. **Avoid 3.13 for now** — some transitive wheels still lag on it and you'd hit build errors. 3.13 is not worth the debugging time for this project.

## 2026-11-14 — CUDA build: cu118
Driver is 526.98, which is too old for CUDA 12.x minor-version compatibility. Install the **cu118** PyTorch build (`--index-url https://download.pytorch.org/whl/cu118`). Alternative: update the NVIDIA driver to ≥527.41, then cu124 also works. ChatGPT was right on this.

## 2026-11-14 — Models: roboflow/sports, not generic YOLO
Use the **football-trained** models from `roboflow/sports` (players + goalkeepers + referees + ball, plus pitch-keypoint detection for homography). Generic COCO YOLO has no "ball"/"referee" split and no pitch keypoints, so it can't map pixels to real pitch coordinates. This matters for hot areas and xG geometry.

## 2026-11-14 — Collaboration: shared GitHub repo
Claude and ChatGPT don't talk directly. This repo is the shared office. Rizki moves work and messages between them through it. `collab/status.md` is the always-current brief to paste into ChatGPT.

## 2026-11-14 — xG model
Logistic regression on 5,606 non-penalty StatsBomb open-data shots (WC 2018/2022, Euro 2020/2024). Features: distance, goal-mouth angle, header, direct free kick. Penalty = 0.76 fixed. Coefficients frozen in `src/xg_model.py`. Broadcast-level only — no keeper position or pressure, so treat single-shot values as rough.
