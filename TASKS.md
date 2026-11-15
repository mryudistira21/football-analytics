# Tasks

Status board. `[ ]` open · `[~]` in progress · `[x]` done. Owner in **bold**.

## Track A — Manual tagging (Claude)
- [x] **Claude** — Pitchside Tagger built (pass/cross/shot, xG, xA, shot map, pass map, hot areas, CSV export)
- [x] **Claude** — xG model fitted on StatsBomb open data → `src/xg_model.py`
- [ ] **Rizki** — tag the wide-play segments of the clip, export CSV to `data/`
- [ ] **Claude** — once a real CSV exists, sanity-check the numbers and add a match report

## Track B — Automatic tracking (ChatGPT, review by Claude)
- [~] **Rizki** — install env on D: (Python, PyTorch cu118, ultralytics) — CUDA must report True
- [ ] **Rizki** — download roboflow/sports models (players, ball, pitch) into `data/`
- [ ] **Rizki** — crop broadcast picture with `src/crop_broadcast.py` (2400→1920)
- [ ] **ChatGPT** — run PLAYER_TRACKING and RADAR on the cropped clip
- [ ] **ChatGPT** — export per-frame player positions (metres, per team) to CSV
- [ ] **Claude** — review the export script; turn position CSV into heatmaps (`src/heatmap_from_positions.py`)
- [ ] **both** — add a camera-state filter so close-ups/replays don't poison pitch mapping

## Open questions
- Use roboflow/sports' football-trained models, or generic YOLO? → see DECISIONS.md (sports models chosen)
- Which Python version? → see DECISIONS.md (3.10 or 3.11)
