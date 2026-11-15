# Collaboration status — 15 November 2026

## Current repo state
- main remains Claude's Track A / shared baseline.
- ChatGPT's Track B work is isolated on track-b/automatic-tracking-v1.
- Track B branch: https://github.com/mryudistira21/football-analytics/tree/track-b/automatic-tracking-v1
- Draft PR: https://github.com/mryudistira21/football-analytics/pull/1
- Track B setup and usage: [TRACK_B.md](../TRACK_B.md)
- Track A review for Claude: [track-a-review-2026-11-15.md](track-a-review-2026-11-15.md)

## Track B v1 scope
- Roboflow Sports soccer player detector + pitch-keypoint detector.
- Supervision ByteTrack for anonymous person IDs.
- Team-colour clustering via SigLIP/UMAP/KMeans (CPU by default to conserve the RTX 2070's 8 GB VRAM).
- Pitch homography to 105 x 68 m; invalid calibration emits blank x/y.
- CSV + metadata are streamed to an external video/tracking folder; no full-match annotated render by default.
- Source start/end time arguments allow a full match file to be processed as separate period runs while retaining original frame numbers and video timestamps.
- Per-period home attack directions are written as metadata; camera-frame tracking x/y are not flipped.
- Camera state is conservative: wide only when pitch calibration passes; closeup otherwise. Replay/graphics classification is not claimed in v1.

## First local validation needed
Run the Track B 30-second smoke test from TRACK_B.md, inspect test20_preview.mp4, then check:
1. Player boxes stay on actual players during a wide live shot.
2. Team clusters map to the correct home/away kits.
3. Pitch coordinates move smoothly and remain within 0..105 x 0..68.
4. Close-up/replay/graphic segments produce blank coordinates when calibration is invalid.
5. The output CSV match_id exactly matches the Pitchside Tagger event CSV.

## Collaboration rule
Do not merge Track B into main until the Windows smoke test passes. Claude should address the actionable Track A issues in collab/track-a-review-2026-11-15.md on main or a dedicated Track A branch.
