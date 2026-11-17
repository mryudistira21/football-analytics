# Collaboration status — 15 November 2026

## Current repo state
- main remains Claude's Track A / shared baseline.
- ChatGPT's Track B work is isolated on track-b/automatic-tracking-v1.
- Track B branch: https://github.com/mryudistira21/football-analytics/tree/track-b/automatic-tracking-v1
- Draft PR: https://github.com/mryudistira21/football-analytics/pull/1
- Latest Track B fix: [commit `12fc09d`](https://github.com/mryudistira21/football-analytics/commit/12fc09d26f8406e31c64ab4efc76b810e1cc4337) aligns all 32 Roboflow pitch keypoints with the Tagger's 105 × 68 m landmark geometry (`pitchside_tagger_105x68_v1`); adds geometry/order tests; clarifies v1 camera-state semantics; removes the committed ZIP and ignores `*.zip`.
- Latest runtime fix: external `trackers.ByteTrackTracker` replaces removed `sv.ByteTrack` for Supervision 0.31+; adds the package dependency, timestamp-aware updates, and excludes tentative ID `-1` detections. See the latest branch commit.
- CI: an earlier Track B unit-test run passed; verify the latest branch head's checks before merging. Full Colab model/video inference still needs validation.
- Track B setup and usage: [TRACK_B.md](../TRACK_B.md)
- Track A review for Claude: [track-a-review-2026-11-15.md](track-a-review-2026-11-15.md)

## Track B v1 scope
- Roboflow Sports soccer player detector + pitch-keypoint detector.
- External `trackers.ByteTrackTracker` for anonymous person IDs, with source timestamps and tentative ID `-1` excluded.
- Team-colour clustering via SigLIP/UMAP/KMeans; `--team-device auto` selects GPU when at least 4 GB free VRAM remains after detector models load, otherwise CPU.
- Pitch homography to 105 x 68 m using explicit 32-point Tagger-aligned geometry; unexpected upstream landmark geometry/order fails fast; invalid calibration emits blank x/y.
- CSV + metadata are streamed to an external video/tracking folder; no full-match annotated render by default.
- Source start/end time arguments allow a full match file to be processed as separate period runs while retaining original frame numbers and video timestamps.
- Per-period home attack directions are written as metadata; camera-frame tracking x/y are not flipped.
- Camera state is conservative: wide only when pitch calibration passes; closeup otherwise. Replay/graphics classification is not claimed in v1.

## Colab validation needed (remaining merge gate)
Run at least one short segment in `notebooks/track_b_colab.ipynb`, inspect its log and `.meta.json`, then check:
1. SigLIP selected CUDA on the T4 when free VRAM met the threshold (or recorded CPU fallback).
2. Player boxes stay on actual players during a wide live shot.
3. Team clusters map to the correct home/away kits; cluster zero is arbitrary and must be checked.
4. Pitch coordinates remain within 0..105 x 0..68 when `camera_state=wide`.
5. The CSV columns and `match_id` still match the shared `DATA_CONTRACT.md`.

## Collaboration rule
Do not merge Track B into main until the hosted Colab segment smoke test and output review pass. Keep Claude's Track A work on main or a dedicated Track A branch; Track B changes must not redesign the Tagger.


## Track B performance and Colab update — 17 November 2026
- The latest Windows diagnostic logs showed the no-team test completed with 145 sampled frames / 1,615 rows for a 15-second window. The team-enabled test completed with 290 sampled frames / 4,230 rows for a 30-second window; initial SigLIP extraction for 300 training crops took about 67 seconds. The separate Colab notebook uses a hosted runtime; the laptop is not the compute host.
- Track B now bounds the initial team-crop sampling pass to the effective segment window, including `--limit-seconds`, and samples from `--start-seconds`. Sampling reuses the resident player detector instead of instantiating another YOLO copy.
- Added `--team-device auto|cuda:0|cpu`. Auto checks free VRAM after the player and pitch models are loaded; it selects CUDA when at least 4 GB is free, else CPU. SigLIP is loaded after crop sampling to avoid overlapping its model allocation with the sampling pass.
- SigLIP features are cached per player track. New tracks are classified; stable assignments are not continuously re-embedded, while uncertain tracks can be refreshed after the configured interval. Default `--team-classify-every` is now 30.
- Both YOLO inference calls now use `quantize="fp16"` on CUDA instead of deprecated `half=True`.
- Added `--team-model-state` to persist/reuse fitted UMAP + KMeans state across segments, preserving cluster numbering. Cluster-to-home/away remains arbitrary and must be checked in the first segment.
- Added `--track-id-offset` for unique person IDs across merged segments. Ball ID 0 is unchanged; track IDs remain anonymous and are not shirt numbers.
- Added `notebooks/track_b_colab.ipynb`: hosted-GPU setup, optional yt-dlp video download, model installation/download, Google Drive output/log persistence, 15-minute segments, completion-aware resume, and concatenation.
- Automated tests were expanded for team-device selection, stable/unstable track embedding refresh, cluster transforms, and saved cluster-state mapping. CI status must be checked for the latest branch commit before calling it green.
- Keep Track A/Tagger, xG/xA/xT, and downloader behavior untouched. Do not merge the draft PR until CI and a Colab segment smoke test plus output validation pass.
