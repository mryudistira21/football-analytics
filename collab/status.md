# Collaboration status — 25 November 2026

Paste this into ChatGPT to bring it up to date.

## Repo state
- **`main` is the only branch that matters** (Track A + Track B). Base new work on `main`, open PRs against `main`.
- CI (`.github/workflows/track-b-tests.yml`) runs on pushes to `main` and on PRs (77 tests).

## How Track B runs (production)
- `scripts/run_match.py` + `matches/persiba_2026-27.json`; notebooks `match_colab.ipynb` / `match_kaggle.ipynb` (Kaggle = unattended, 2 × T4, two matches at once); laptop: `--profile light|balanced|full`. Runbook `MATCH_WORKFLOW.md`.
- One command per match: download → models → `track_video.py` in 10-min chunks (subprocess each, resumable) → combine (+`swap_teams`) → **`resolve_direction.py` (new)** → `build_player_moments.py` → `team_report.py` → `build_event_candidates.py` → bundle zip → (local) `save_to_db.py`.
- Review: candidates CSV → Tagger *Review (Track B)* → `<id>_events.csv` + `<id>_review.json` → SQLite `football.db`.

## New today (Claude) — please read before touching these files
1. **Possession is a time share now** (`team_report.possession_frames`): holder = nearest known-team player ≤ 2 m; the owner keeps the ball until the opponent holds it 3 samples in a row; nobody owns it after 5 s without a holder or across a camera cut. New keys `possession_share`, `quality.possession_known_share`; `ball_proximity_share` kept as diagnostic. In/out-of-possession block heights use this owner. Real matches: PERSIBA–Persika 50/50 (the old 50/50 was real), Persika–Persinga 61/39.
2. **Attack direction from the data** (`team_report.infer_direction` + `scripts/resolve_direction.py`): kick-off positions (first 10 s of each period) + possession shift (players higher when their team has the ball). `home_dir_1h: null` now works; a config value wins but a confident contradiction prints a WARNING and lands in the report (`direction_check`, `warnings`). Writes `<id>_direction.json`.
3. **PERSIBA–Persika direction was wrong** (`L` from keeper positions). Kick-off + possession shift say `R` in 4/4 votes, and under `L` both teams' in-possession blocks were *lower* than out-of-possession. Config fixed to `R`. Lesson: keeper team labels are near random (keeper kits), never derive direction from them.
4. **Goalkeeper teams fixed post hoc**: once the direction is known, every goalkeeper track gets the team defending the half it stands in; the combined CSV is rewritten (streamed). ~12k / ~8.6k keeper rows changed on the two matches.
5. **Review queue −55 %** (candidates v1.1, two extra columns `priority`, `linked`): contested-ball blips dropped; priority 3 (recovery after ball out of view, blip possessions) hidden by default in the Tagger; interception linked to its failed pass = one card. PERSIBA–Persika 878 → 389 cards, Persika–Persinga 512 → 234. DB `candidates` gets `priority`, `linked` (auto-migrated).
6. **`scripts/evaluate_candidates.py`**: hand-tagged window vs candidates → recall/precision per type, per priority, recall of the default queue, wrong-team matches, pass-outcome agreement.

## Contracts — please don't change without a note here
- Tracking CSV columns: `match_id, frame, video_seconds, period, track_id, role, team, x, y, confidence, camera_state` (camera frame, 105 × 68 m).
- Cut-list (7 columns): `moment, start_s, end_s, duration_s, start_clock, events, label`.
- Team report JSON `trackb_team_report_v1` — add keys, never rename (Hub reads `possession_share` with fallback to `ball_proximity_share`).
- Candidates CSV v1.1 (DATA_CONTRACT §6b), direction JSON (§6f), review JSON `tagger_review_v1`, SQLite names (§6d), `run_match.py` CLI flags and config keys.

## Matches in the config
| match_id | Fixture | Score | Scope | State |
|---|---|---|---|---|
| `persiba_gresik_20261113` | PERSIBA – Gresik United | 0–0 | PERSIBA | periods set, not run yet |
| `persipa_persiba_20261117` | Persipa Pati – PERSIBA | 2–2 | PERSIBA | periods set, not run yet |
| `persiba_persika_20261121` | PERSIBA – Persika | 0–1 | PERSIBA | **done** (dir R, swap) |
| `persika_persinga_20261113` | Persika – Persinga Ngawi | 1–0 | scouting | **done** (dir R) |
| `gresik_persika_20261117` | Gresik – Persika | 1–1 | scouting | periods set, not run |
| `persinga_psf_20261118` | Persinga – Putra Sang Fajar | 1–2 | scouting | needs period times |
| `persinga_persibangga_20261122` | Persinga – Persibangga | — | scouting | video pending |

Next PERSIBA opponent: **Persinga Ngawi (away, 26 Nov)**.

## New since PR #3 (Claude, on main) — physical, counter-press, summary
From the hublabdev reference app, all built on the tracking we already produce (no schema removals):
- **`scripts/player_physical.py`** — per-team and (with OCR numbers) per-player distance, high-intensity distance, sprint distance, top speed, touches. Team totals reliable; per-track are chunk fragments; per-player needs `track_map.csv`. DATA_CONTRACT §6g.
- **Counter-press** in `team_report` — per turnover, the losing team's pressers within 5 m in 3 s and whether it regained within 5 s; `<id>_counterpress.csv` + per-team keys. §6h.
- **Match summary** block (report + Hub card).
- DB: `team_physical`, `player_physical`, `counter_press` + views `v_team_physical`, `v_counter_press`.
- `run_match` now runs player-physical into `report/` before the team report (`--physical`). 77 tests green.

## Matches to run (this batch)
`match_kaggle.ipynb` on `main` is set to **PERSIBA match 1 (`persiba_gresik_20261113`), match 2 (`persipa_persiba_20261117`), and Persinga–Putra Sang Fajar (`persinga_psf_20261118`)**, READ_NUMBERS=True (also the OCR check). PERSIBA 1–2 have period times; Persinga–PSF needs its 4 timestamps filled in OVERRIDES. Claude processes the bundles when they return.

## Tasks for you (ChatGPT), in priority order
1. **Shirt-number OCR, real-match check.** Rizki runs the next Kaggle batch with `READ_NUMBERS = True` (PERSIBA's matches 1–2: `persiba_gresik_20261113`, `persipa_persiba_20261117`). From the bundles / `football.db`: coverage (`v_track_numbers`: share of tracks with a number, share of candidate `track_id`s with `track_player`), confidence distribution, and plausibility (numbers outside each squad list in the Hub = wrong reads; the same number on both teams at once = suspicious). If coverage < ~20 % or many impossible numbers: tune `NumberReader` (crop band, `min_box_px`, `min_conf`, reads per track) or propose a jersey-number model (e.g. a small fine-tuned classifier on the back crop). Keep the `<id>_track_map.csv` contract (§6e). Deliver: a short note in `collab/` + a PR.
2. **Throughput A/B on one chunk** (still open): same 10-min chunk with defaults vs `--pitch-every 2` vs `--imgsz 800`. Compare minutes, `calibration_median_error_m`, rows, ball rows, and the team report numbers for that chunk. Recommend a default for T4 and for Rizki's laptop (`light` profile, 8 GB RAM).
3. **Pressing proxy + defensive moments — MERGED (PR #3).** Adds `press_share` and sample-count diagnostics to `trackb_team_report_v1`; adds `defensive_nearest_2` per-track moments using resolved direction metadata and the existing seven-column cut-list. No schema changes. `ppda_proxy` is deferred until a hand-tagged evaluation window validates candidate pass / defensive-action counts; do not label raw candidate counts as PPDA.
4. **Candidate tuning — after** Rizki has one hand-tagged window and `evaluate_candidates.py` output. Use it to tune `HOLD_M`, `PASS_MIN_M`, `LOOSE_*`, `FLICKER_*` and the priority-3 rules. Report before/after recall/precision; never tune without that file.
5. **Direction restart fallback — MERGED (PR #3; Claude fixed a NumPy-2 crash in the restart split before merge).** If the first 10 seconds do not give a valid kick-off-position vote, `infer_direction` scans up to the first 10 minutes for a clear centre-ball restart (kick-off / after-goal restart) with both teams in opposite halves. At most one strongest restart vote is contributed per period. If neither restart evidence nor possession shift is strong, direction remains unknown.

## Tasks for Claude (in progress / next)
- Physical / counter-press / summary built and validated on the two real matches; Rizki has the rebuilt reports + DB.
- Process the three new Kaggle bundles when they arrive; Persinga scouting summary.
- **Note for ChatGPT:** physical running, counter-press and the match summary are done — don't rebuild them; your OCR-coverage check (task 1) now also feeds the per-player physical table.


## ChatGPT implementation note
- See `collab/chatgpt-task-update-2026-11-23.md` for definitions, test scope, limitations, and remaining data requests.
- Shirt-number OCR and throughput A/B remain dependent on the next READ_NUMBERS Kaggle bundles and same-chunk variant runs; candidate tuning remains blocked until a hand-tagged evaluation window is available.
