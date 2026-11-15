# Data contract

The shared format between **Track A** (manual tagging, Claude) and **Track B** (automatic tracking, ChatGPT). Both tracks build against this file. Change it only with a note in `DECISIONS.md`.

## 1. Pitch and coordinates
- Pitch: **105 × 68 metres**. Units are metres everywhere, never pixels.
- `y = 0` is the **top touchline as seen on the broadcast camera**; `y` grows downward (same as the screen and the Tagger).
- Two frames, never mixed in one column:

| Frame | Used by | `x = 0` is… |
|---|---|---|
| **camera** | tracking CSV | the goal on the **left of the broadcast picture**, all match long |
| **team** | event CSV (Tagger) | the acting team's **own goal**; every team attacks toward `x = 105` |

- Converting camera → team: if the team attacks **left** on screen in that period, `x' = 105 − x`, `y' = 68 − y`. Teams swap ends at half-time. The match's `home_dir_1h` (`R` or `L`) says which way home attacks on screen in period 1.

## 2. Time
| Field | Type | Meaning |
|---|---|---|
| `video_seconds` | float | seconds from the start of the **video file** (the join key between tracks) |
| `frame` | int | 0-based frame index in the video file (tracking only) |
| `fps` | float | video frame rate, stored once per run (tracking metadata) |
| `period` | int | 1 = first half, 2 = second half, 3/4 = extra time |
| `match_seconds` | float | match clock in seconds (as on the scoreboard) |

`video_seconds = frame / fps`. Join events to tracking by `video_seconds` (nearest frame within ±0.5 s).

## 3. Identity
- `team`: always `home` or `away` (names live in match metadata).
- `shirt_number` (events) and `track_id` (tracking) are **separate identifiers**. A `track_id` is anonymous and can change after occlusions or camera cuts. They are linked only through an explicit mapping table that a human confirms:
  `track_map.csv`: `match_id, track_id, video_seconds_from, video_seconds_to, team, shirt_number`

## 4. Event CSV (Track A, Pitchside Tagger export)
One row per event; coordinates in the **team** frame.

`match, id, half, match_clock, match_seconds, video_seconds, team, side, player, player_name, event, outcome, receiver, receiver_name, x, y, end_x, end_y, body_part, situation, assist_event_id, xg, xt_added, progressive, into_box`

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
- `camera_state`: `wide` (usable for positions), `closeup`, `replay`, `graphics`. Only `wide` rows carry `x`, `y`.
- Metadata file next to it, `<name>.meta.json`: `match_id, video_file, fps, width, height, home_dir_1h, frame_stride, model_versions`.

## 6. Resource rules (both tracks)
- **Stream, don't hoard:** read video frame by frame, write CSV rows as you go (append mode), never hold the whole match in memory.
- **Process at 720p / 30 fps** (download that way; no need for 60 fps). Use `frame_stride` (e.g. every 2nd or 3rd frame) for position heatmaps.
- **Annotated videos are opt-in** and short (a few minutes); a full annotated match costs gigabytes.
- **float32** for coordinates; round to 0.01 m in CSVs.
- Videos and model weights live **outside the repo** (`FOOTBALL_VIDEO_DIR`), never committed.
