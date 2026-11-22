# Decisions

Short log of choices, so neither AI re-opens a settled question. Newest first.

## 2026-11-22 (e) — Review pipeline and database
- **Track B proposes, Track A disposes.** `scripts/build_event_candidates.py` turns possession changes into event candidates (pass/cross, pass-fail + interception, tackle, recovery, clearance, shot) in the Tagger's team frame. Possession = nearest known-team player within 2 m of the ball on wide frames, ≥ 2 samples. On the pilot clip: ~12 candidates/min, ~half the real on-ball actions (ball visible ~53% of wide frames). Candidates contract v1 in DATA_CONTRACT §6b.
- **Tagger Review tab:** accept (Enter) / reject (X) / skip / back / play clip; team, event, outcome and shirt numbers editable; changing team mirrors the coordinates. Shirt numbers are remembered per track (`match.tmap`) and pre-fill later candidates; bulk-accept of OK passes whose tracks are mapped. Accepted candidates become normal events with `source=trackb`, `cand`, `tr`, `tro`; rejecting an accepted one deletes its event. Candidates persist per match (db chunks of 400 / localStorage). Downloads: `<match_id>_events.csv` (+ source, candidate_id, track_id, track_to columns) and `<match_id>_review.json`.
- **Database = SQLite** (`<root>/football.db`): built into Python, one file, no server; same tables can move to Supabase/Postgres later. `scripts/save_to_db.py` ingests run folders, bundle zips and Tagger downloads (Downloads folder as inbox), one transaction per match, re-runs replace. Local runs save automatically; Colab/Kaggle pass `--no-db` and the PC builds the DB from the bundle (which now also carries the tracking CSV and the candidates).

## 2026-11-22 (d) — Resource limits for running on Rizki's own PC
- **Profiles** in `run_match.py` (`--profile full|balanced|light`, passed through by `run_queue.py`): CPU threads (env + `torch.set_num_threads` + `cv2.setNumThreads`), CPU affinity, process priority (Windows IDLE / BELOW_NORMAL; Linux nice 19 / 10), `torch.cuda.set_per_process_memory_fraction`, a GPU rest after each batch, and a RAM watchdog (psutil) that stops a chunk above the limit and retries it at batch 1. Limits are applied inside `track_video.py` before thread pools start, and recorded in each chunk's metadata (`resource_limits`).
- GPU utilisation can't be hard-capped on consumer NVIDIA cards; the duty-cycle pause is the lever. Power limits (`nvidia-smi -pl`) need admin and are often locked on laptops, so not used.
- `Jalankan Track B.bat`: double-click runner, light profile, all ready matches, resumable.
- `psutil` added to requirements.
- **8 GB laptop:** RAM stop = total − reserve (light 3.5 GB → 4.5 GB stop; balanced 2.5 GB) instead of a share; light/balanced put SigLIP on the GPU (`--team-min-free-vram-gb 1.5`) to keep ~1 GB out of system RAM.
- **`Setup Track B.bat` / `setup_track_b.ps1`:** one-click, re-runnable Windows setup (PS 5.1-safe), pip temp/cache on D:, torch pinned via a constraints file so pip can't swap in the CPU-only wheel, GPU check, models.
- **No duplicate video copies on a single disk** (`--work` = `--root` → use the master in place); `video_aliases` find the pilot downloads by their old names.

## 2026-11-22 (c) — Unattended runs (Kaggle, laptop)
- **`scripts/run_queue.py`:** several matches, one worker per GPU (`--gpus 0,1`), smoke test (only when no team state yet) + full run each, per-match logs and `queue/status.json`. `--matches ready` = every match with a video URL and both period times.
- **`notebooks/match_kaggle.ipynb`:** Kaggle commits run with the browser closed; 2 × T4 → two matches in parallel. Videos and scratch live in `/tmp` (not saved); runs, bundles, stills and logs in `/kaggle/working`. Resume by attaching the previous version's output. Uploaded mp4 datasets are picked up when YouTube blocks the download.
- **`--video-dir`** on the driver: where masters live; when set, no second local copy is made and downloads land there directly.
- Colab background execution is a Pro+ feature only, so Kaggle or the laptop are the "close the browser" options.

## 2026-11-22 (b) — Track B parity, team-colour fit, scouting
- **PR #1 is fully contained in `main`.** Every Track B file on `track-b/automatic-tracking-v1` is byte-identical to `main` except `src/heatmap_from_positions.py`, whose one extra feature (read `home_attacks_right_by_period` from `<csv>.meta.json`) is now ported. PR #1 closed as superseded.
- **Team-colour fit spread over the 1st half:** ~100 frames (every 15 s across up to 25 min, reached by seeking), up to 10 player crops each, max 600 crops — instead of the first 4 minutes, which are often walk-outs, close-ups and replays. New `--team-sample-start`; intervals ≥ 2 s seek instead of decoding every frame.
- **New tracks are coloured sooner:** `team_classify_every` 10 sampled frames (≈1 s) instead of 30. Only tracks without a stable vote are re-embedded, so the cost stays small. Aims at the ~43% `unknown` tracks seen in the pilot.
- **Opponent scouting** lives in the same config (`"scope": "scouting"`); the Hub club page aggregates every imported Track B report for that club (own perspective, averages, half-to-half change, average heatmap). No extra file format.

## 2026-11-22 — Production run for the PERSIBA matches (Pekan 1–3)
- **One driver, one notebook.** `scripts/run_match.py` runs a whole match from `matches/persiba_2026-27.json`: download → models → tracking in 10-min chunks → combine → moments → team report → bundle zip. Resumable (finished chunks are skipped, checked against their metadata). Same command on Colab and on the RTX 2070. `notebooks/match_colab.ipynb` is a thin front end; it supersedes the pilot notebooks that were passed around (download/split/T4/memory-safe variants).
- **Stoppage time stays in its half.** The pilot "P3/P4" ranges were 2–3 min of added time, not extra time; merged into periods 1 and 2. League matches use periods 1–2 only.
- **Memory/GPU:** the notebook kernel never imports torch; each chunk is a subprocess; skipped frames are `grab()`bed (no BGR decode); YOLO runs batched (`--batch`, auto 8 on CUDA) in FP16; `--pitch-every` can thin the pitch model; `MALLOC_ARENA_MAX=2`, `expandable_segments`; CUDA OOM retries the chunk at batch 2. No re-encoding anywhere — half clips are stream copies.
- **Frame stride `auto`** = round(fps/10): the pilot clip was 58 fps, where stride 3 meant ~19 samples/s (double the work).
- **Track IDs** are `chunk × 100000 + n` — deterministic, so a resumed run never collides.
- **Direction and team swap are post-processing.** Tracking doesn't need `home_dir_1h`; the report does. A swapped home/away cluster is fixed with `swap_teams` at combine time — no GPU rerun.
- **Team fit once per match** (`team-state.joblib`, from the first ~4 min of P1, made by the smoke test) and reused by all chunks.
- **Team report** (`scripts/team_report.py`, schema `trackb_team_report_v1`): per period — ball-proximity share (possession proxy), territory thirds, block height overall / in / out of possession, back line (deepest outfield after dropping the single deepest per frame — keepers are often detected as players), width, length, 21×14 heatmaps, plus data-quality shares. Wide-camera frames only. Proxies, for comparing halves/matches.
- **Hub** imports the report JSON next to the Tagger CSV; same match (id or fixture) merges into one entry.
- **Tagger:** optional *Match ID* on new matches (use the config id), and *kick-off times in the video* — the match clock then skips pre-match and half-time on a full broadcast, and the 1st/2nd half button follows the video.

## 2026-11-19 — Periods (incl. extra time) and analysis-by-half
- **Tagger periods:** the tagging toggle is now 1st / 2nd / ET1 / ET2 (event `half` = 1–4). Extra-time direction is a separate match setting ("Home attacks in 1st extra-time half", defaults to the 1st-half direction; ET2 flips it) — a fresh coin toss per DATA_CONTRACT `home_attacks_right_by_period`. Injury/added time is tagged under its own half (no separate control needed).
- **Analysis-by-half:** a single Period selector (Full / 1st / 2nd / ET1 / ET2 / Extra time) above the analysis tabs filters every view (Summary, Players, Shot/Pass maps, Network, Hot areas, Moments, Log, xG race). The scoreboard stays full-match. Replaced the Network tab's own half filter. Implemented via `AEV()` (period-filtered event set).
- **Hub:** the imported-match view gets a Full / 1st / 2nd / ET toggle that recomputes score, xG, shots and the shot map per half (reads the CSV `half` column).
- **Downloader GUI:** added Extra Time 1 / Extra Time 2 download modes; the half time-range hint now says to include injury time.
- **Track B:** periods 3/4 (extra time) go in the notebook `PERIODS` config like 1st/2nd; each period's source-video boundary should include that half's injury time.

## 2026-11-18 — Tagger v3: moment reels, confidence flags, telestration, lineup chips
Driven by a peer prototype (midblockstats) doing the same auto-cut + reel concept. Added to the Tagger, all on the manual side so they work for match day without the CV pipeline:
- **Moments / reel tab.** Pick a team + player (or whole team) and every tagged action becomes a clip; overlapping clips merge (gap ≤ 1.2 s). Play them back-to-back in the video, toggle any off, export a cut list. The **Actions** filter gives per-player, team-attack and team-defence reels. Needs `video_seconds`, so tag with a video loaded.
- **Confidence / review flags.** Each event cycles unmarked → review → wrong → ok in the Log; badges show in the Log and on each Moment (worst-of). New `review` column in the events CSV (import + export).
- **Telestration.** Pause the video, grab the frame, draw arrows / circles / line / pen / spotlight, save a PNG. The Metrica-style piece the peer tool lacks.
- **Lineup quick-pick chips.** Tap a player from the match lineup instead of typing the number (works for receiver too). Reads `match.lu`.

## 2026-11-18 — Track B cut-list importer (A↔B join, layer 1)
The reel's "Imported (Track B)" source imports Track B per-track cut-lists. **Contract:** exactly 7 columns `moment,start_s,end_s,duration_s,start_clock,events,label`; `start_s`/`end_s` are `video_seconds` and drive playback; `label` is read as confidence (`ok`/`review`/`wrong`); `events` is the moment descriptor; `track` id comes from the `track_<id>_###` prefix in `moment`. One cut-list per track, multi-file load, deduped by `moment`. `start_clock`/`duration_s` display-only. Manifest (team, diagnostics) stays a sidecar — never extra columns in the cut-list. Implemented in `scripts/build_player_moments.py` on Track B.

## 2026-11-18 — How Track A and Track B combine
Two layers joined by `video_seconds`, not merged into one blob — Track B stays regeneratable, Track A events stay the source of truth for xG/xA/xT. Stages: (1) shared timeline — auto-moments + manual events in the same reel (done); (2) verify — B's confidence flags + A's review/edit correct the auto output, minimap sanity-checks; (3) event pre-fill — B auto-detects pass/shot and writes them as Tagger events pre-marked `review` for a human to confirm (later; needs event detection, harder than tracking).

## 2026-11-18 — League Hub + full roster scrape
Beyond the Tagger, added a second web app — the **League Hub** (`hub/index.html` + `hub/data.json`) — a neutral Liga Nusantara browser (standings, fixtures, clubs, squads, players search, top scorers) that also integrates the user's own analysis: import a Tagger events CSV and it shows score, xG, shots and a shot map, stored in the artifact's `db`, and tags the matching fixture. Data scraped from ileague.id (2026-11-18): all **23 clubs / 660 players** via WebFetch (`scripts/build_clubs_json.py` → `data/clubs.json`), plus groups A/B/C, standings (points only — ileague lists no W/D/L/GF/GA), Pekan 2 results and top scorers (`scripts/build_hub_data.py` → `hub/data.json`). ileague gives no positions or kit colours, so positions are blank (GK tapped in the Tagger) and colours are placeholders. Snapshot model: re-run the build scripts to refresh. The sandbox shell can't reach ileague.id (proxy 403); scraping is done through WebFetch.

## 2026-11-18 — Club & player DB: local JSON (not Supabase)
Considered Supabase, then chose a plain local JSON instead — simpler, no keys, nothing leaves the device, fits the "local only" stance. Scope: `clubs` + `players` (+ short code and kit colour), in `data/clubs.json`. `src/export_squads.py` builds that file from `src/club_db.py`'s SQLite. The Tagger match settings load it ("Load squads JSON", remembered in the browser). `data/clubs.json` holds the FULL squad per club ({number, name, position}); "From squads -> Pick XI" selects the starting eleven with a rule of **exactly 11 and exactly 1 GK** (GK detected from the player's position — GK/Kiper/keeper), or "Whole squad" loads everyone. It also sets the team name/colour. Match events and Track B moments stay on their current paths.

## 2026-11-18 — Track B exports per-track moments for the Tagger Reel
- Build player moments as a pure post-processor over the tracking CSV; do not rerun YOLO or SigLIP.
- Each anonymous track gets a separate cut-list CSV with exactly `moment,start_s,end_s,duration_s,start_clock,events,label`; source video seconds remain the playback/join values.
- `events` is the descriptor (v1: `near_ball`); `label` is `ok`, `review`, or `wrong`. V1 emits only `ok` and `review`; never infer `wrong` from missing/weak evidence.
- Keep `match_id`, `track_id`, team and diagnostics in a sidecar manifest, not as extra cut-list columns. Track IDs remain distinct from shirt numbers.
- Segment status is explicit: `complete`, `partial_eof`, or `no_frames`. Resume accepts only complete metadata whose actual start/end coverage matches the requested interval; early EOF must not be treated as a completed segment.

## 2026-11-17 — Track B moves full-match inference to hosted Colab and runs resumable segments
- The local Windows RTX 2070 run took about 39 minutes for 60 seconds of source video and the laptop froze twice during full-length attempts. Heavy Track B inference will use a hosted Colab GPU (T4 when available); Colab's hosted runtime is separate from the laptop's local CPU/GPU.
- The notebook `notebooks/track_b_colab.ipynb` processes each half in 900-second segments by default and stores each segment CSV, metadata and log in Google Drive. Complete segments are skipped after reconnect; incomplete segments are rerun; segment CSVs are concatenated only after completion.
- Segment runs use a persistent `--team-model-state` file (fitted UMAP + KMeans) so the arbitrary cluster 0/1 labels remain consistent between segments. Verify the first segment's cluster-to-home/away assignment before using it as ground truth.
- Person `track_id` values receive a `--track-id-offset` per segment to avoid collisions in the merged CSV; ball `track_id=0` remains reserved.
- `--team-device auto` selects CUDA only when free VRAM after both YOLO models are loaded is at least 4 GB; otherwise it uses CPU. Replace deprecated Ultralytics `half=True` with `quantize="fp16"` for CUDA.
- `--cluster-0-side` maps raw cluster numbering to home/away at run time; the notebook reruns completed segments when that mapping changes so merged CSVs stay consistent. Per-segment CSVs are concatenated as a row stream to avoid loading the full match into memory.

## 2026-11-17 — No-terminal downloader
Added src/download_gui.pyw (Tkinter) + "Unduh Video.bat" so videos can be downloaded without the terminal, with Full Pertandingan / Babak Pertama / Babak Kedua options (halves cut at keyframes, no re-encode). Reuses src/download_match.py's logic; Track A.

## 2026-11-16 — Track B tracker API follows Supervision 0.31+
- Supervision 0.31.0 removed `sv.ByteTrack`; use Roboflow's external `trackers.ByteTrackTracker` with `tracker.update(detections, timestamp=...)`.
- Pin Supervision to the 0.31.x API range and add `trackers>=2.7.0,<3.0.0` to the Track B requirements.
- Filter tracker ID `-1` (tentative/unmatched detections) before team votes, CSV output and preview labels so it is never treated as a persistent player identity.
- Pass source-video PTS timestamps and source FPS to ByteTrackTracker so frame-stride sampling does not masquerade as full-rate updates.

## 2026-11-15 — Track B pitch geometry aligned with Pitchside Tagger
- **Homography target geometry:** Track B maps Roboflow's 32 ordered soccer-pitch keypoints to the exact 105 × 68 m landmark coordinates used by Pitchside Tagger. Do not axis-scale the 120 × 70 m source template independently; this distorts internal markings while a homography can still have a low reprojection residual. Identified in tracking metadata as `pitchside_tagger_105x68_v1` and guarded by tests for all 32 landmarks.
- **Camera-state contract for v1:** only `wide` and `closeup` are emitted. `closeup` is a fallback for invalid/unavailable pitch calibration, not a confirmed semantic close-up; replay/graphics classification remains future work.

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
