# Collaboration status — 22 November 2026 (afternoon)

Paste this into ChatGPT to bring it up to date.

## Repo state
- **`main` is the only branch that matters.** It contains all of Track A and all of Track B.
- **PR #1 (Track B v1) is closed as superseded:** every Track B file on `track-b/automatic-tracking-v1` is byte-identical to `main` except `src/heatmap_from_positions.py`, whose extra feature (directions read from `<csv>.meta.json`) is now ported. **PR #2 is merged.** Please base new work on `main` and open PRs against `main`.
- CI (`.github/workflows/track-b-tests.yml`) runs on pushes to `main` and on PRs (53 tests).

## How Track B runs now (production)
- `scripts/run_match.py` + `matches/persiba_2026-27.json` + `notebooks/match_colab.ipynb`; runbook `MATCH_WORKFLOW.md`.
- One command per match: download (reuses earlier downloads found by YouTube id) → models → `track_video.py` in 10-min chunks, each a subprocess, resumable via chunk metadata → combine → `build_player_moments.py` → `scripts/team_report.py` → bundle zip.
- `track_video.py` (CSV schema unchanged): `grab()`/`retrieve()` for skipped frames; batched FP16 YOLO (`--batch`, auto 8 on CUDA); `--pitch-every N`; `--team-sample-start` with seek-based sampling for intervals ≥ 2 s. Output is byte-identical to the previous version at batch 1 on a test clip.
- Driver defaults: frame stride `auto` = round(fps/10) (the pilot clip was 58 fps); team fit from ~100 frames spread over the 1st half (every 15 s, up to 25 min, ≤ 600 crops); `--team-classify-every 10`; track IDs `chunk × 100000 + n`; CUDA OOM → retry at batch 2.
- **Unattended runs:** `scripts/run_queue.py` (one worker per GPU) + `notebooks/match_kaggle.ipynb` (Kaggle commit, 2 × T4, two matches at once). Smoke outputs land in `deliverables/<match>_still_*.jpg` and `<match>_smoke_summary.json`.
- **Resource profiles** for local runs: `--profile light|balanced|full` (CPU threads/cores/priority, VRAM fraction, GPU rest per batch, RAM watchdog). Chunk metadata records `resource_limits`; the driver logs peak RAM per chunk.
- **Review pipeline:** `scripts/build_event_candidates.py` → `<id>_candidates.csv` (contract v1, DATA_CONTRACT §6b) → Tagger *Review (Track B)* tab (accept / fix / reject, shirt numbers remembered per track) → `<id>_events.csv` + `<id>_review.json` → `scripts/save_to_db.py` → SQLite `football.db` (§6d).
- Home/away swap (`swap_teams`) and attack direction (`home_dir_1h`) are applied **after** tracking — no GPU rerun to fix them.
- Stoppage time stays in its half (the pilot's "P3/P4" were added time, now merged into periods 1–2).

## Matches in the config
| match_id | Fixture | Score | Scope |
|---|---|---|---|
| `persiba_gresik_20261113` | PERSIBA – Gresik United | 0–0 | PERSIBA |
| `persipa_persiba_20261117` | Persipa Pati – PERSIBA | 2–2 | PERSIBA |
| `persiba_persika_20261121` | PERSIBA – Persika Karanganyar | 0–1 | PERSIBA |
| `persika_persinga_20261113` | Persika – Persinga Ngawi | 1–0 | scouting |
| `gresik_persika_20261117` | Gresik United – Persika | 1–1 | scouting |
| `persinga_psf_20261118` | Persinga – Putra Sang Fajar | 1–2 | scouting |
| `persinga_persibangga_20261122` | Persinga – Persibangga | — | scouting (video pending) |

Next PERSIBA opponent: **Persinga Ngawi (away, 26 Nov)**.

## Contracts — please don't change without a note here
- Tracking CSV columns: `match_id, frame, video_seconds, period, track_id, role, team, x, y, confidence, camera_state` (x, y in metres on 105 × 68, camera frame).
- Cut-list (7 columns): `moment, start_s, end_s, duration_s, start_clock, events, label`.
- Team report JSON: `schema: "trackb_team_report_v1"` (Hub imports it; the Hub club page aggregates it for scouting).
- Event candidates CSV v1 and the review JSON (`tagger_review_v1`); SQLite table/column names.
- `run_match.py` CLI flags and the config keys.

## Fixed since last note (23 Nov, from your review)
- **Moment labels bug fixed.** `build_player_moments.py` now reads calibration from BOTH combined-metadata shapes — the old `segments[].details` and run_match's flat `chunks[]` (fps/stride at top level). Before, every moment got `calibration_unavailable` → all `review`. Now PERSIBA–Persika is 230 ok / 897 review and Persika–Persinga 152 ok / 643, exactly as you predicted. Manifest now carries `label_counts`. Regression test added (`test_calibration_read_from_run_match_chunks_metadata`).
- **Both matches post-processed.** Directions/swaps resolved from goalkeeper positions + kit colours (PERSIBA home kit is red → swap_teams=true, home_dir_1h=L; Persika at home in blue → swap_teams=false, home_dir_1h=R). Reports, candidates and a combined `football.db` built; both coherent against the scorelines.
- **Kaggle export:** the committed `match_kaggle.ipynb` cell 4 already iterates `MATCHES` (collects a bundle per match). The single-match zip was an ad-hoc cell, not the notebook.
- **Throughput (your benchmark):** ~1.0× real-time per match on one T4 (2 matches in parallel), peak RAM ~2.9 GB, GPU ~2.3 GB. `--pitch-every 2` / `--imgsz 800` not yet A/B-tested — still open.

## Track B tasks for you (in priority order)
1. **Team-classification check on real footage.** After Rizki's first smoke test, look at `runs/<id>/smoke/still_*.jpg` and `smoke_summary.json` (`player_rows_by_team`). If `unknown` is still high, options: torso-only crops (upper 55% of the box) to drop shorts/grass; a minimum box height for fit crops; requiring 2 agreeing votes before labelling. Keep CLI + schema.
2. **Throughput numbers** from the first real T4 run (the driver prints per-chunk minutes and × real time). If a T4 does < 1× real time, test `--pitch-every 2` and `--imgsz 800` on one chunk and compare `calibration_median_error_m` and row counts.
3. **Candidate quality:** after the first real review, compare `candidates.review_status` per `type`/`confidence` in `football.db` (`v_review_progress`) and tune `build_event_candidates.py` thresholds (HOLD_M, PASS_MIN_M, LOOSE_*). Better ball detection = more candidates.
4. **Defensive moments** in `build_player_moments.py` (ball in a team's own third, nearest 2 defenders), same 7-column cut-list, label `ok/review`.
5. **Pressing proxy for `team_report.py`:** share of opponent ball-near samples in the opponent's own third where a player of this team is within 5 m. Add as new keys; don't rename existing ones.
6. **Shirt-number OCR is in** (`--read-numbers`, easyocr) → `<id>_track_map.csv`, pre-fills the Tagger review and the DB `track_map` (source=ocr). Contract §6e. **Needs a real-match accuracy check**: run one match with numbers on, then look at `v_track_numbers` (coverage, confidence) and the "OCR suggests" hits in review. Tune crop region / `--numbers-*` / consider a jersey-number model if easyocr coverage is low. → `track_map.csv`.
