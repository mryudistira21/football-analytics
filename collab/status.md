# Collaboration status — 22 November 2026

Paste this into ChatGPT to bring it up to date.

## What changed (main)
- **PR #2 merged** (fast-forward): four-period Tagger/Hub/heatmap work is on `main`.
- **Production runner:** `scripts/run_match.py` + `matches/persiba_2026-27.json` + `notebooks/match_colab.ipynb`. One command per match: download → models → tracking in 10-min chunks (each a subprocess, resumable via chunk metadata) → combine → `build_player_moments.py` → `scripts/team_report.py` → bundle zip. Same driver runs on Colab and on Rizki's RTX 2070. The runbook is `MATCH_WORKFLOW.md`.
- The pilot notebooks shared in chat (download / split-halves / T4 / memory-safe variants) are **superseded** by `match_colab.ipynb`. `notebooks/track_b_colab.ipynb` stays for reference.
- **Periods:** the "P3/P4" ranges in the pilot were stoppage time (2–3 min after each half), not extra time. Merged into periods 1 and 2 in the config. League matches use 1–2 only.
- **`track_video.py` (schema unchanged):** `grab()`/`retrieve()` so skipped frames aren't converted; batched YOLO (`--batch`, auto 8 on CUDA, FP16); `--pitch-every N` reuses the last homography between pitch-model calls (default 1 = unchanged). New metadata keys `batch`, `pitch_every`.
- **Frame stride auto** = round(fps/10). Your 20-min pilot clip was **58 fps**, so stride 3 sampled ~19/s (twice the work).
- **Track IDs** = `chunk × 100000 + n` (deterministic; replaces scanning previous CSVs for the max).
- **Home/away swap and attack direction moved to post-processing** (`swap_teams`, `home_dir_1h` in the config) — no GPU rerun to fix either.
- **Team report** `trackb_team_report_v1`: per period possession proxy (ball proximity ≤ 3 m), territory thirds, block height overall / in / out of possession, back line (drops the single deepest player per frame — keepers often come through as `player`), width, length, 21×14 heatmaps, data-quality shares. Hub imports it beside the Tagger CSV.
- CI now also runs on pushes to `main`; new tests `test_run_match.py`, `test_team_report.py`, notebook compile test.

## The three matches
| match_id | Fixture | Score |
|---|---|---|
| `persiba_gresik_20261113` | PERSIBA – Gresik United | 0–0 |
| `persipa_persiba_20261117` | Persipa Pati – PERSIBA | 2–2 |
| `persiba_persika_20261121` | PERSIBA – Persika Karanganyar | 0–1 |

## Useful next Track B work (pick any)
1. **Team classification quality.** In the pilot, ~43% of tracks ended `unknown`. Ideas: fit crops sampled across the whole first half (not only the first 4 min), exclude goalkeeper/referee crops from the fit, and vote by majority over more refreshes. Keep the CLI and schema.
2. **Throughput numbers** from the first real T4 run (`runs/<id>/logs/*.log` + chunk timings printed by the driver) to set expectations per match.
3. **Defensive moments** in `build_player_moments.py` (ball in own third, nearest defenders), same 7-column cut-list.
4. Later: shirt-number OCR → `track_map.csv`.

Please don't change the cut-list columns, the tracking CSV columns, or the report schema without a note here.
