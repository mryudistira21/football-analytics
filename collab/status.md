# Collaboration status — 18 November 2026

Paste this into ChatGPT to bring it up to date.

## Repo reconciled into one branch
History got rewritten earlier (a force-push to `main` left an old lineage), so the work was split across branches. It is now reconciled: **`main` is `520bb6f`**, holding BOTH tracks in one coherent tree:
- **Track A**: Pitchside Tagger v3, League Hub, `data/` roster DB, downloader GUI.
- **Track B** (from `track-b/automatic-tracking-v1`): `scripts/track_video.py`, `scripts/build_player_moments.py`, `notebooks/track_b_colab.ipynb`, `tests/`, `TRACK_B.md`.
- Shared docs merged (DATA_CONTRACT, DECISIONS, requirements with `trackers`/`joblib`).

**Done:** `main` was force-updated to this unified tree (over the old diverged lineage). One source of truth now. The old feature branches and the draft **PR #1** are redundant — close PR #1 rather than merging it.

## Track A — done
- **Tagger v3:** moment reels (per-player / team-attack / team-defence), confidence flags (`review` CSV column), telestration, lineup chips, and a **starting-XI picker** (pick 11, tap GK, enforced 11 + 1 GK) that loads from `data/clubs.json`.
- **Cut-list importer** in the reel ("Imported (Track B)") — accepts your exact 7-column format. Ready for your moment CSVs now.
- **League Hub** — a Liga Nusantara browser (standings, fixtures, clubs, squads, 660-player search, scorers) that also imports a tagged-match CSV to show score/xG/shot map. Data scraped from ileague.id into `data/*.json`.

## Track B — next steps
1. **Cheap first test (no GPU):** run `scripts/build_player_moments.py` on the existing 20-min combined CSV → per-track cut-lists → Rizki imports them into the Tagger reel. Validates the A↔B loop. This is the priority.
2. **Full match:** first real match is **PERSIBA Bantul vs Gresik United (13 Nov, `nCVF6TTJGXw`)** — process the full match on Colab (two periods), not the 20-min clip.
3. **Team mapping gate:** make a short `--annotated-output` segment, confirm PERSIBA (home) = cluster side; flip `--cluster-0-side` if swapped.
4. Later: **2D pitch radar/minimap** (Phase 2), **defensive moments** (ball-far / defensive-third pressing, not just near-ball), **shirt-number OCR** → `track_map.csv` (Phase 3).

## Video download — 30 fps is correct
Download **720p at 30 fps** (yt-dlp, no re-encode) — confirmed as the standard for both tracks. 60 fps doubles frames for no tagging or tracking benefit; 30 fps source with `--frame-stride 3` gives ~10 fps effective sampling, which is plenty. Use the downloader GUI (full match) for PERSIBA–Gresik.

## Open gates before merging the Track B PR
- Team home/away mapping verified against real jerseys.
- A real full-match run (the 20-min clip only validated the pipeline, not a match).
