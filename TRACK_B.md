# Track B — automatic tracking v1

First implementation of video-to-position tracking. It is deliberately separate from the Pitchside Tagger and Track A's xG/xT work.

## What v1 does

- Runs Roboflow Sports' football player detector, which has the football classes: ball, goalkeeper, player and referee.
- Tracks people using Roboflow's external `trackers.ByteTrackTracker` with native `supervision.Detections`; Supervision 0.31+ no longer ships `sv.ByteTrack`.
- Passes decoder timestamps to the tracker so frame-stride sampling is reflected in its motion timing; tentative tracker ID `-1` rows are omitted from exported player tracks.
- Optionally learns two kit-colour clusters from a capped sample of player crops using Roboflow Sports' SigLIP + UMAP + KMeans TeamClassifier.
- Detects pitch keypoints on each sampled frame and estimates a homography.
- Maps the 32 ordered Roboflow pitch keypoints to the Tagger's actual 105 × 68 m landmark geometry (`pitchside_tagger_105x68_v1`); it does not independently scale the source pitch axes.
- Converts bottom-centre player locations and ball centres into the 105 x 68 metre camera coordinate frame.
- Streams rows to a temporary CSV, promoting it to the final CSV only after a successful run.
- Writes a metadata JSON beside the CSV.
- Writes no annotated video unless explicitly requested; previews are limited by --annotated-max-seconds.

## Known limitations

1. **Team clusters are arbitrary.** `--cluster-0-side home` maps cluster 0 to home; it does not prove that cluster 0 is home. Check the short annotated preview, then rerun with `--cluster-0-side away` if reversed.
2. **Track IDs are not shirt numbers.** IDs can change after occlusions/camera cuts. Use a human-confirmed `track_map.csv` as described in DATA_CONTRACT.md.
3. **Camera-state classification is calibration-based, not semantic.** V1 emits only `wide` and `closeup`: `wide` means the pitch homography passed validation; `closeup` is the fallback when it did not. The fallback can include genuine close-ups, replays, graphics, or poor keypoint detections. V1 does not reliably identify replay/graphics footage and must not claim to deduplicate it.
4. **Homography is per sampled frame.** There is no temporal interpolation or camera-cut smoothing yet. Frames with weak calibration have blank coordinates.
5. **VFR timing needs validation.** The script uses OpenCV's decoder timestamp when it increases monotonically, otherwise frame/FPS. Validate event joins for variable-frame-rate screen recordings.
6. **Goalkeepers may remain unknown.** V1 team-colour votes are accumulated from outfield players; the goalkeeper is not confidently assigned to a side yet.
7. **Extra time requires explicit direction metadata.** Periods 3 and 4 are accepted only with `--period-direction 3:R|L` / `--period-direction 4:R|L`; never infer them from period 1.
8. **Ball detection is a first-pass baseline.** It uses the ball class from the football player detector. Very small or blurred balls can be missed; the dedicated ball model is downloaded but not yet used in the main loop.

## Install (Windows / Python 3.10)

Use the project venv on D:. Install the cu118 PyTorch wheel first as recorded in DECISIONS.md, then install the repository requirements and Roboflow Sports.

```powershell
cd D:\Codingan\football-analytics
.\.venv\Scripts\Activate.ps1

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install git+https://github.com/roboflow/sports.git

# requirements.txt installs the external Trackers package used for ByteTrack
```

Install pytest only for tests:

```powershell
pip install pytest
```

Keep video, model weights and Hugging Face cache outside Git:

```powershell
setx FOOTBALL_VIDEO_DIR D:\Video-Bola-Analis
setx FOOTBALL_MODEL_DIR D:\Video-Bola-Analis\models
```

Open a new PowerShell after setx.

## Download model weights

```powershell
python scripts\download_sports_models.py
```

The script stores player, pitch and ball weights in the external model folder. The main v1 tracker uses the player detector (including its ball class) and the pitch-keypoint detector. The dedicated ball model is present for a future tiled ball-detection stage.

The downloader uses Google Drive file IDs directly and is compatible with gdown 6.x; do not add the removed `fuzzy` argument.

The tracking pipeline uses `ByteTrackTracker` with a 0.25 activation threshold to retain the player's previous ByteTrack confidence behavior, and confirms IDs after three sampled observations.

## First smoke test

Use the immutable `match_id` from the event CSV exported by Pitchside Tagger.

```powershell
python scripts\track_video.py `
  --input D:\Video-Bola-Analis\test20.mp4 `
  --match-id YOUR_TAGGER_MATCH_ID `
  --period 1 `
  --home-dir-1h R `
  --frame-stride 3 `
  --limit-seconds 30 `
  --annotated-output D:\Video-Bola-Analis\tracking\test20_preview.mp4
```

Change `R` to `L` if the home team attacks toward screen-left in the first half. Check the preview to validate cluster mapping.

Outputs:

```text
D:\Video-Bola-Analis\tracking\
  test20_p1_tracks.csv
  test20_p1_tracks.meta.json
  test20_preview.mp4
```

Check `complete: true` in metadata. The CSV must have the same `match_id` as the event CSV.

## Full match / separate periods

The current script assigns one period value per run. For a full 90-minute file, run each half separately with source-video time ranges. For a file that already contains only one half, omit the range flags.

Example for a 90-minute video (assuming the half-time boundary is at 45:00):

```powershell
# First half
python scripts\track_video.py `
  --input D:\Video-Bola-Analis\match.mp4 `
  --match-id YOUR_TAGGER_MATCH_ID `
  --period 1 `
  --home-dir-1h R `
  --start-seconds 0 --end-seconds 2700 `
  --frame-stride 3

# Second half — home direction automatically reverses from the recorded first-half direction
python scripts\track_video.py `
  --input D:\Video-Bola-Analis\match.mp4 `
  --match-id YOUR_TAGGER_MATCH_ID `
  --period 2 `
  --home-dir-1h R `
  --start-seconds 2700 --end-seconds 5400 `
  --frame-stride 3
```

For extra time, specify the actual direction separately; do not infer it from period 1:

```powershell
python scripts\track_video.py --input match.mp4 --match-id YOUR_ID --period 3 --home-dir-1h R --period-direction 3:L
```

The default output filenames include the period suffix (for example, match_p1_tracks.csv and match_p2_tracks.csv). Source-video timestamps and frame indices remain relative to the original file, so the period CSVs can still be joined to Tagger events.

Recommended initial frame stride is 3 (about 10 observations/sec on 30 FPS video). Use stride 2 for denser sampling at higher cost. Use `--no-team-classification` for the fastest pipeline test, with `team=unknown`.

## Output schema

The tracking CSV follows DATA_CONTRACT.md:

```text
match_id,frame,video_seconds,period,track_id,role,team,x,y,confidence,camera_state
```

- Coordinates are camera-frame metres: x=0 at the screen-left goal; y=0 at the top touchline.
- Coordinates are only filled when `camera_state=wide`.
- `track_id=0` is reserved for the single ball row; use `role` to distinguish it from person tracks.
- People use anonymous ByteTrack IDs; do not equate these to shirt numbers.
- Period 1 and 2 directions are derived from `--home-dir-1h`; periods 3 and 4 require explicit `--period-direction`. Tracking coordinates themselves are not flipped by period.

## Resource behaviour

Video is decoded one frame at a time. Team-crop sampling is capped (300 by default). The SigLIP team classifier runs on CPU to leave VRAM available for player and pitch models on the RTX 2070 8 GB. CSV rows are written incrementally. Annotated video is opt-in and short. Model weights and Hugging Face cache live outside the repo.

## Test

```powershell
python -m pytest tests\test_track_video.py -q
```

## Upstream code and licenses

The model and soccer-pipeline foundation comes from [Roboflow Sports](https://github.com/roboflow/sports/tree/main/examples/soccer). Its soccer example uses Ultralytics YOLO, Supervision, SigLIP, UMAP and KMeans. The upstream repository states the Sports code is MIT-licensed and the YOLOv8 dependency is AGPL-3.0. Review the applicable licenses before redistributing a product or providing a hosted service.
