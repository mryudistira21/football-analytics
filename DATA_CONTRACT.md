# Data contract

The shared format between **Track A** (manual tagging, Claude) and **Track B** (automatic tracking, ChatGPT). Both tracks build against this file. Change it only with a note in `DECISIONS.md`.

## 1. Pitch and coordinates
- Pitch: **105 × 68 metres**. Units are metres everywhere, never pixels.
- Track B homography targets must use the **same 32 internal landmark coordinates as Pitchside Tagger**. Do not independently scale Roboflow's 120 × 70 m template axes to 105 × 68 m: that changes penalty-area, goal-area and penalty-spot geometry even when the homography residual is small. The validated mapping is versioned as `pitchside_tagger_105x68_v1` in tracking metadata.
- `y = 0` is the **top touchline as seen on the broadcast camera**; `y` grows downward (same as the screen and the Tagger).
- Two frames, never mixed in one column:

| Frame | Used by | `x = 0` is… |
|---|---|---|
| **camera** | tracking CSV | the goal on the **left of the broadcast picture**, all match long |
| **team** | event CSV (Tagger) | the acting team's **own goal**; every team attacks toward `x = 105` |

- Converting camera → team: if the team attacks **left** on screen in that period, `x' = 105 − x`, `y' = 68 − y`.
- Direction per period is explicit, not assumed. The match metadata carries `home_attacks_right_by_period` (e.g. `{"1": true, "2": false}`). Teams swap at half-time, but extra time (periods 3–4) gets a fresh coin toss (IFAB Law 8), so never infer period 3 from period 1. For a normal two-half match, `home_dir_1h` (`R`/`L`) is enough and periods 3–4 are absent.

## 2. Time
| Field | Type | Meaning |
|---|---|---|
| `video_seconds` | float | presentation time in seconds from the start of the **video file** (the join key between tracks). The Tagger records the video's current time whenever a video is loaded, in **either** time mode; empty only when no video was loaded, and such events cannot be joined to tracking. Tracking should use decoder presentation timestamps (PTS) where available. |
| `frame` | int | 0-based decoded frame index in the video file (tracking only) |
| `fps` | float | nominal/average video frame rate, stored once per run (tracking metadata); fallback only for deriving timestamps when PTS is unavailable |
| `period` | int | 1 = first half, 2 = second half, 3/4 = extra time |
| `match_seconds` | float | match clock in seconds (as on the scoreboard) |

For constant-frame-rate files, `video_seconds` is approximately `frame / fps`; for variable-frame-rate files, use the frame's presentation timestamp (PTS) rather than assuming a constant frame interval. Join events to tracking by `video_seconds`, taking the nearest tracking frame within one actual sampling interval (approximately `frame_stride / fps` for constant-frame-rate video), not a fixed half-second. A tagged event's `video_seconds` is a human click and can lag the true moment by a second or so, so when several candidate players fall inside the window, record the ambiguity rather than forcing one match.

## 3. Identity
- `match_id`: one immutable id per recording/analysis, shared by both tracks. Two recordings of the "same" fixture (e.g. a replay) get different `match_id`s.
- Event CSV `side`: always `home` or `away`; event CSV `team` is the display name.
- Tracking CSV `team`: `home`, `away`, or `unknown` (unknown when classification is unavailable or uncertain).
- `shirt_number` (events) and `track_id` (tracking) are **separate identifiers**. A `track_id` is anonymous and can change after occlusions or camera cuts. They are linked only through an explicit mapping table that a human confirms:
  `track_map.csv`: `match_id, track_id, video_seconds_from, video_seconds_to, team, shirt_number`

## 4. Event CSV (Track A, Pitchside Tagger export)
One row per event; coordinates in the **team** frame.

`match_id, match_name, id, half, match_clock, match_seconds, video_seconds, team, side, player, player_name, event, outcome, receiver, receiver_name, x, y, end_x, end_y, body_part, situation, assist_event_id, xg, xt_added, progressive, into_box, review, source, candidate_id, track_id, track_to`

- `source`: `manual` (tagged by hand) or `trackb` (an accepted Track B candidate). `candidate_id`, `track_id`, `track_to` link a `trackb` event back to its candidate and Track B tracks. The Tagger downloads this as `<match_id>_events.csv`.

- `half` is the match period: `1` = first half, `2` = second half, `3` = extra-time first half, `4` = extra-time second half. Preserve this integer through Tagger import/export and Hub analysis; do not collapse periods 3 and 4 into period 1 or 2.

- `match_id` is the stable id both tracks share (the Tagger's internal match id, e.g. `m8f3k2ab`); `match_name` is the display name only. Join across tracks on `match_id`, never on the name.
- `team` is the team name; `side` is `home` or `away` (use `side` to join with tracking).
- `event`: `pass, cross, shot, tackle, interception, recovery, clearance, foul`
- `outcome`: `complete, incomplete` (pass/cross) · `goal, on target, off target, blocked` (shot) · `won, lost` (tackle) · `complete` (others)
- `player`, `receiver`: shirt numbers as text; empty when unknown.

## 5. Tracking CSV (Track B)
One row per detected object per processed frame; coordinates in the **camera** frame.

`match_id, frame, video_seconds, period, track_id, role, team, x, y, confidence, camera_state`

- `role`: `player, goalkeeper, referee, ball`
- `team`: `home, away, unknown` (`unknown` for referees, ball, unclassified)
- `x`, `y`: metres; empty when the frame has no valid pitch calibration.
- `camera_state`: Track B v1 emits `wide` or `closeup`. `wide` means the frame's pitch-keypoint homography passed calibration validation and its coordinates are usable. `closeup` is the fallback when calibration does not pass; it is **not** a semantic guarantee that the footage is a close-up and may also include replay/graphics frames or weak keypoint detections. Only `wide` rows carry `x`, `y`. Add `replay` or `graphics` only when a future version actually detects and tests those states.
- Metadata file next to it, `<name>.meta.json`: at minimum `match_id, video_file, fps, width, height, home_dir_1h, home_attacks_right_by_period, frame_stride, model_versions`. It should also record the processed `period`, source-video start/end seconds, completion status, and relevant tracking configuration.

## 6. Automatic moment cut-list (Track B → Tagger Reel importer)

Track B can export one cut-list CSV per anonymous `track_id`. The Reel importer accepts exactly these seven columns, with no additional columns:

`moment,start_s,end_s,duration_s,start_clock,events,label`

- `start_s` and `end_s` are source-video `video_seconds` and drive playback. They are the time join key; they are not match-clock seconds.
- `moment` is unique within the match and uses `track_<track_id>_<sequence>` (for example, `track_137_001`). The importer parses `track_id` from this prefix.
- `events` is the moment descriptor, such as `near_ball`.
- `label` is the confidence badge: `ok`, `review`, or `wrong`. Track B v1 assigns only `ok` and `review` automatically; `wrong` is reserved for human correction.
- `duration_s` is display-only and may be recomputed by the importer. `start_clock` is display-only and is not parsed by the importer.
- Multiple per-track files may be imported together; the importer deduplicates by `moment`. A sidecar manifest may carry `match_id`, team and diagnostics; it is not required by the importer.
- This cut-list is distinct from the 26-column event CSV. It does not change the event importer.
- `track_id` is not `shirt_number`; no player-number identity is implied.

## 6b. Event candidates (Track B → Tagger Review tab) — v1.1

`<match_id>_candidates.csv`, built by `scripts/build_event_candidates.py` from the combined tracking CSV once the attack direction is known (config `home_dir_1h`, or read from the data — §6f):

`cand_id, period, t, t_end, side, type, outcome, track_id, track_to, x, y, ex, ey, confidence, reason, track_player, track_to_player, priority, linked`

- `type` uses the Tagger's event names (`pass, cross, shot, tackle, interception, recovery, clearance`); `outcome` uses the Tagger's internal codes (`ok, fail, on, won`; empty = the reviewer picks).
- `t` = `video_seconds` of the action (release for passes, gain for interceptions/tackles/recoveries); `t_end` = when the ball arrived.
- `x, y, ex, ey`: **team frame** metres (the acting `side` attacks left → right), so an accepted candidate becomes an event unchanged.
- `track_id` = acting track, `track_to` = receiver (passes) or the player who lost it (tackles). Anonymous Track B ids, not shirt numbers.
- `confidence`: `ok` or `review`; `reason` is a short human explanation.
- `priority` (v1.1): `1` Track B confident, `2` needs a look, `3` probably noise (hidden in the Tagger's default queue, never deleted). Missing column = 2.
- `linked` (v1.1): on an interception, the `cand_id` of the failed pass / clearance it answers; the Tagger decides both together (accepted for a turnover, rejected otherwise). Empty elsewhere.
- `track_player`, `track_to_player`: optional shirt-number suggestions for the acting / receiving track, filled from `<match_id>_track_map.csv` (§6e) when shirt-number OCR ran; empty otherwise. The Tagger review pre-fills from them.

## 6f. Attack direction — `<match_id>_direction.json` (+ team report keys)

Written by `scripts/resolve_direction.py` (run by `run_match` after combining): `match_id, used (R|L|null), source (config|inferred|unknown), config, inferred, confidence (high|medium|conflict|none), votes [{signal: kickoff|possession_shift, period, metres, home_right_1h}], goalkeeper_rows_relabelled, warning`. The combined `.meta.json` gets `home_attacks_right_by_period` and a `direction` block. Goalkeeper rows in the combined tracking CSV carry the team defending the half they stand in once the direction is known.

Team report (`trackb_team_report_v1`, additive keys only): per side `possession_share` (time share, §DECISIONS 2026-11-23 evening), `quality.possession_known_share`, top-level `direction_check` and `warnings`. `method.possession` replaces `method.possession_hold_s`.

## 6e. Shirt-number map (Track B OCR, optional) — `<match_id>_track_map.csv`

Written when Track B runs with `--read-numbers` (easyocr). One row per track that got a reading:

`track_id, team, number, votes, reads, confidence, attempts`

- `number` is the most-voted shirt number for that anonymous `track_id`; `confidence` = votes / reads. A track spans one chunk only (ids are `chunk × 100000 + n`), so a player appears under several track ids.
- Suggestions, not ground truth: broadcast numbers are small and often hidden, so coverage and accuracy vary. The Tagger review uses `confidence ≥ 0.5, votes ≥ 2` as a pre-fill; the human always confirms. In the DB these land in `track_map` with `source='ocr'`; confirmed numbers from review are `source='review'`.

## 6c. Tagger review export — `<match_id>_review.json`

`{"schema": "tagger_review_v1", "match_id": ..., "track_map": {"<track_id>": {"team": 0|1, "p": "<shirt>"}}, "candidates": [{"cand_id", "status": "accepted|rejected|pending", "event_id"}]}` — `team` 0 = home, 1 = away.

## 6d. Database — `<root>/football.db` (SQLite)

Built by `scripts/save_to_db.py` (called by `run_match.py` at the end of a local run). Tables: `matches, tracking, moments, candidates (+ review_status, event_id, priority, linked — older databases get the two new columns automatically), track_map, events, team_metrics, team_heatmaps, ingests`; views `v_review_progress, v_player_events, v_team_half`. Re-saving a match replaces that match's rows. Column names follow sections 4–6c.

## 7. Resource rules (both tracks)
- **Stream, don't hoard:** read video frame by frame, write CSV rows as you go (append mode), never hold the whole match in memory.
- **Process at 720p / 30 fps** (download that way; no need for 60 fps). Use `frame_stride` (e.g. every 2nd or 3rd frame) for position heatmaps.
- **Annotated videos are opt-in** and short (a few minutes); a full annotated match costs gigabytes.
- **float32** for coordinates; round to 0.01 m in CSVs.
- Videos and model weights live **outside the repo** (`FOOTBALL_VIDEO_DIR`), never committed.
