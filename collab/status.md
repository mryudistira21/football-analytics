# Collaboration status — 22 November 2026 (afternoon)

Paste this into ChatGPT to bring it up to date.

## Repo state
- **`main` is the only branch that matters.** It contains all of Track A and all of Track B.
- **PR #1 (Track B v1) is closed as superseded:** every Track B file on `track-b/automatic-tracking-v1` is byte-identical to `main` except `src/heatmap_from_positions.py`, whose extra feature (directions read from `<csv>.meta.json`) is now ported. **PR #2 is merged.** Please base new work on `main` and open PRs against `main`.
- CI (`.github/workflows/track-b-tests.yml`) runs on pushes to `main` and on PRs (43 tests).

## How Track B runs now (production)
- `scripts/run_match.py` + `matches/persiba_2026-27.json` + `notebooks/match_colab.ipynb`; runbook `MATCH_WORKFLOW.md`.
- One command per match: download (reuses earlier downloads found by YouTube id) → models → `track_video.py` in 10-min chunks, each a subprocess, resumable via chunk metadata → combine → `build_player_moments.py` → `scripts/team_report.py` → bundle zip.
- `track_video.py` (CSV schema unchanged): `grab()`/`retrieve()` for skipped frames; batched FP16 YOLO (`--batch`, auto 8 on CUDA); `--pitch-every N`; `--team-sample-start` with seek-based sampling for intervals ≥ 2 s. Output is byte-identical to the previous version at batch 1 on a test clip.
- Driver defaults: frame stride `auto` = round(fps/10) (the pilot clip was 58 fps); team fit from ~100 frames spread over the 1st half (every 15 s, up to 25 min, ≤ 600 crops); `--team-classify-every 10`; track IDs `chunk × 100000 + n`; CUDA OOM → retry at batch 2.
- Home/away swap (`swap_teams`) and attack direction (`home_dir_1h`) are applied **after** tracking — no GPU rerun to fix them.
- Stoppage time stays in its half (the pilot's "P3/P4" were added time, now merged into periods 1–2).

## Matches in the config
| match_id | Fixture | Score | Scope |
|---|---|---|---|
| `persiba_gresik_20261113` | PERSIBA – Gresik United | 0–0 | PERSIBA |
| `persipa_persiba_20261117` | Persipa Pati – PERSIBA | 2–2 | PERSIBA |
| `persiba_persika_20261121` | PERSIBA – Persika Karanganyar | 0–1 | PERSIBA (video link pending) |
| `persika_persinga_20261113` | Persika – Persinga Ngawi | 1–0 | scouting |
| `gresik_persika_20261117` | Gresik United – Persika | 1–1 | scouting |
| `persinga_persibangga_20261122` | Persinga – Persibangga | — | scouting (video pending) |

Next PERSIBA opponent: **Persinga Ngawi (away, 26 Nov)**.

## Contracts — please don't change without a note here
- Tracking CSV columns: `match_id, frame, video_seconds, period, track_id, role, team, x, y, confidence, camera_state` (x, y in metres on 105 × 68, camera frame).
- Cut-list (7 columns): `moment, start_s, end_s, duration_s, start_clock, events, label`.
- Team report JSON: `schema: "trackb_team_report_v1"` (Hub imports it; the Hub club page aggregates it for scouting).
- `run_match.py` CLI flags and the config keys.

## Track B tasks for you (in priority order)
1. **Team-classification check on real footage.** After Rizki's first smoke test, look at `runs/<id>/smoke/still_*.jpg` and `smoke_summary.json` (`player_rows_by_team`). If `unknown` is still high, options: torso-only crops (upper 55% of the box) to drop shorts/grass; a minimum box height for fit crops; requiring 2 agreeing votes before labelling. Keep CLI + schema.
2. **Throughput numbers** from the first real T4 run (the driver prints per-chunk minutes and × real time). If a T4 does < 1× real time, test `--pitch-every 2` and `--imgsz 800` on one chunk and compare `calibration_median_error_m` and row counts.
3. **Defensive moments** in `build_player_moments.py` (ball in a team's own third, nearest 2 defenders), same 7-column cut-list, label `ok/review`.
4. **Pressing proxy for `team_report.py`:** share of opponent ball-near samples in the opponent's own third where a player of this team is within 5 m. Add as new keys; don't rename existing ones.
5. Later: shirt-number OCR → `track_map.csv`.
