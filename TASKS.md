# Tasks

Status board. `[ ]` open · `[~]` in progress · `[x]` done. Owner in **bold**.

## Track A — Manual tagging (Claude)
- [x] **Claude** — Pitchside Tagger v1 (pass/cross/shot, xG, xA, shot map, pass map, hot areas, CSV export)
- [x] **Claude** — xG model fitted on StatsBomb open data → `src/xg_model.py`
- [x] **Claude** — Tagger v2: xT + progressive passes + passes into box; receiver # + pass network; defensive events (tackle, interception, recovery, clearance, foul) → PPDA, field tilt; xG race chart; lineups
- [x] **Claude** — xT grid fitted on StatsBomb open data → `src/xt_model.py`, `models/xt_grid.json`, `scripts/fit_xt.py`
- [ ] **Rizki** — download the full match at 720p (yt-dlp), tag the first 15 minutes, export CSV to `data/`
- [ ] **Claude** — check the first real CSV: tagging speed, missing receivers, any stat that looks off
- [ ] **Claude** — match report page from a finished CSV (one shareable page per match)
- [ ] **Claude** — export to SPADL so socceraction (VAEP) and mplsoccer work on our data

## Track B — Automatic tracking (ChatGPT, review by Claude)
- [x] **Rizki** — Git installed on D:\Codingan\Git
- [~] **Rizki** — Python 3.10 env on D:, PyTorch cu118, ultralytics — CUDA must report True
- [ ] **Rizki** — download roboflow/sports models (players, ball, pitch) into `data/`
- [ ] **ChatGPT** — run PLAYER_TRACKING and RADAR on a 720p clip
- [ ] **ChatGPT** — export per-frame player positions (metres, per team) to CSV
- [ ] **Claude** — review the export script; heatmaps via `src/heatmap_from_positions.py`
- [ ] **both** — camera-state filter so close-ups/replays don't poison pitch mapping
- [ ] **later** — try SoccerNet ball-action spotting (lRomul/ballaction-spotting) to pre-fill passes automatically; humans fix the rest

## Open questions
- Which competitions should the xG/xT models eventually be refit on? Indonesian league data isn't public; StatsBomb open data is the best free proxy for now.
