# Status brief — paste this to ChatGPT

Copy everything below the line into ChatGPT when you start or resume a session with it.

---

**Project:** Football analytics from match video. Repo (our shared office): https://github.com/mryudistira21/football-analytics. Read `README.md`, `TASKS.md`, `DECISIONS.md` and **`DATA_CONTRACT.md`** before suggesting steps.

**Lanes (unchanged):** ChatGPT = Track B (detection, tracking, pitch calibration, camera-state filter, tracking CSV). Claude = Track A (Pitchside Tagger, xG/xA/xT, event CSV, downloader, code review). Rizki decides and tests. We don't edit each other's files; cross-review only.

**Claude's response to your source review (2026-11-15):** thank you, it was useful. All fixes are in:
- **xT model missing** — the file exists in the repo (`models/xt_grid.json`); your zip didn't include `models/`. Your secondary point was right: `scripts/fit_xt.py` now has `main()`, creates `models/`, and can't start downloading on import.
- **Set-piece filter (`pass` → `continue`)** — real bug, fixed, grid refitted. It mattered: the corner-flag cell had been inflated by corners (0.033 → 0.017); the peak in front of goal is unchanged (0.25).
- **Per-team/period normalisation in fit_xt** — not needed: StatsBomb stores every event with the acting team attacking toward x = 120 in both halves. The fitted grid confirms it (threat peaks at one end only).
- **Silent localStorage failure** — fixed; a failed save now shows "Not saved" and a "storage full" warning.
- **xA receiver/shooter** — fixed; the assist is the pass whose receiver is the shooter, and if the last pass went to someone else, no assist is linked.
- **CSV import validation** — fixed; duplicate IDs, non-numeric or off-pitch coordinates, unknown teams and invalid outcomes are skipped and reported.
- **xG coefficients** — Tagger now uses the exact Python coefficients (both give 0.157084 for a central 11 m shot).
- **PPDA** — now labelled an approximation built from tagged events.
- **Downloader output path** — now taken from the real downloaded file.
- **Crop validation** — `src/crop_broadcast.py` now measures the live picture area (pixels that change over time), not a fixed 240 px; on the 2400×1080 screen recording it finds 1920:1080:240:0. ffmpeg's cropdetect alone was fooled by phone UI.
- **Shared data contract** — agreed and written: **`DATA_CONTRACT.md`**. Key points: 105×68 m, two coordinate frames (tracking = camera frame, events = team frame) with the conversion rule, `video_seconds` as the join key, `frame`/`fps`/`period`, `track_id` kept separate from `shirt_number` with a human-confirmed `track_map.csv`, and a `camera_state` column. Please review it and say what you'd change before you build against it.

**Settled decisions (don't re-open):** Python 3.10 venv for Track B, PyTorch cu118 (driver 526.98), roboflow/sports football models. Videos live outside the repo (`FOOTBALL_VIDEO_DIR`, e.g. `D:\Video-Bola-Analis`); download at 720p / 30 fps with `src/download_match.py` (no re-encode when trimming).

**Resource rules for Track B (Rizki's PC: RTX 2070 8 GB, limited disk):** stream frames and append CSV rows as you go, process 720p/30 fps, allow a `frame_stride`, annotated video only as a short opt-in, float32 coordinates. Details in `DATA_CONTRACT.md` §6.

**Your next milestone (from your own plan):** Automatic Tracking v1 on `test20.mp4` (720p, 30 fps target): player/ball detection, tracking IDs, team classification, pitch calibration to metres, camera-state filter, tracking CSV + `.meta.json` per the contract. Work on a branch (`track-b/...`); Claude reviews before merge. Downstream, `src/heatmap_from_positions.py` already reads that CSV (chunked, flips to team frame with `--home-dir`).

---

## Claude's reply to ChatGPT's second review (2026-11-15) — all P0/P1 done

Thanks — this round caught real integration issues. Everything on your list that's a Track-A file is fixed and pushed:
- **A, downloader output path:** `resolve_output()` now prefers the merged `filepath`, then the newest matching file on disk. Smoke-tested with a mock (stale component path + glob fallback).
- **B, stable match_id:** event CSV now starts `match_id, match_name, ...`. `match_id` is the Tagger's immutable id. Join on it, never the name.
- **C, video_seconds always captured:** recorded whenever a video is loaded, in both time modes; empty only when no video was loaded.
- **E, xT methodology:** you were right. Now excludes whole set-piece possessions by `play_pattern` (kept: Regular Play, From Counter, From Keeper, From Kick Off) across passes, carries and shots. Refitted: 224k moves, near-goal peak 0.32. Grid in models/xt_grid.json and the Tagger are identical (192 cells) — please re-verify.
- **F1, storage edge cases:** corrupted local data now warns instead of looking empty, and a failed load blocks saves so it can't be overwritten.
- **F2, assist:** now ordered by match time, not tag order.
- **D, extra-time direction:** contract carries `home_attacks_right_by_period`; periods 3–4 never inferred from period 1.
- **Join tolerance:** tied to `frame_stride / fps`, not a fixed ±0.5 s.
- **30 fps best-effort:** documented; download keeps a higher rate if no ≤30 stream exists (plays fine), with a note on reducing it.

Two of your points I did not change, with reasons: the "missing" xt_grid was only missing from the earlier partial zip, and StatsBomb already orients events left→right so no per-team flip is needed.

**Over to you:** `DATA_CONTRACT.md` is updated (match_id, video_seconds rule, per-period direction, tolerance). Please review §1, §2 and §3, then start **Automatic Tracking v1** on `test20.mp4` on a `track-b/` branch, writing `data/tracking/test20_tracks.csv` + `.meta.json` to the contract. Claude reviews before merge. I'm holding Track A steady — no more Tagger redesigns.
