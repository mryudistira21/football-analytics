# Match workflow — PERSIBA 2026-27 (production)

**The pipeline:** get the video → Track B tracks it automatically → Track A reviews Track B's proposals in the Tagger, fixes them and adds what it missed → results in the Tagger and the League Hub → everything saved in `football.db`.

| Step | Tool | Output |
|---|---|---|
| 1 Video | `run_match.py --download` (any runner) | `videos/<match_id>.mp4` |
| 2 Track B | laptop (`Jalankan Track B.bat`), Kaggle or Colab | tracking, moments, team report, **event candidates**, bundle zip |
| 3 Track A review | Tagger → **Review (Track B)** tab | `<match_id>_events.csv`, `<match_id>_review.json` |
| 4 Results | Tagger analysis tabs; League Hub *My analysis* + club *Scouting profile* | |
| 5 Database | `Simpan ke Database.bat` | `D:\Video-Bola-Analis\football.db` |

One page to analyse a match end to end. Track B (automatic tracking) and Track A (manual tagging) run in parallel on the **same original video**, joined on `video_seconds` and `match_id`.

| match_id | Fixture | Score | Video | Period times |
|---|---|---|---|---|
| `persiba_gresik_20261113` | PERSIBA – Gresik United (H), Pekan 1 | 0–0 | [nCVF6TTJGXw](https://www.youtube.com/watch?v=nCVF6TTJGXw) | filled |
| `persipa_persiba_20261117` | Persipa Pati – PERSIBA (A), Pekan 2 | 2–2 | [v3LlHgczPZA](https://www.youtube.com/watch?v=v3LlHgczPZA) | filled |
| `persiba_persika_20261121` | PERSIBA – Persika Karanganyar (H), Pekan 3 | 0–1 | [4u7t742kGsU](https://www.youtube.com/watch?v=4u7t742kGsU) | to fill |

Opponent scouting (same config, `"scope": "scouting"`):

| match_id | Fixture | Score | Video | Period times |
|---|---|---|---|---|
| `persika_persinga_20261113` | Persika – Persinga Ngawi | 1–0 | [wSEbj4TNGr4](https://www.youtube.com/watch?v=wSEbj4TNGr4) | filled |
| `gresik_persika_20261117` | Gresik United – Persika | 1–1 | [1IYP9I4Atyo](https://www.youtube.com/watch?v=1IYP9I4Atyo) | filled |
| `persinga_psf_20261118` | Persinga – Putra Sang Fajar | 1–2 | [FUOhJTwq5a8](https://www.youtube.com/watch?v=FUOhJTwq5a8) | to fill |
| `persinga_persibangga_20261122` | Persinga – Persibangga | — | add `video_url` | to fill |

Next opponent: **Persinga Ngawi (away, 26 Nov)**.

Config: [`matches/persiba_2026-27.json`](matches/persiba_2026-27.json). On Colab the live copy is `MyDrive/football-analysis/matches/persiba_2026-27.json` (step 2 of the notebook edits it).

## Rules that keep the data joinable

- **One master video per match**, never re-encoded. Tag it, track it, review it — all the same file.
- **Period times** = elapsed time in the master, real kick-off → real final whistle, **stoppage time included**. League matches have no extra time, so only periods 1 and 2. (The earlier "P3/P4" ranges were stoppage time; they are merged back into their halves.)
- **match_id** from the config is typed into the Tagger's *Match ID* field when you create the match.
- **Direction**: `home_dir_1h` = which way the *home* team attacks on screen in the 1st half (R = left→right). For away matches, "home" is the opponent. **You can leave it `null`**: after tracking, the run reads it from the data (kick-off positions + who stands higher with the ball) and prints `Direction: data says R (high; ...)`. Only set it to override; if your value contradicts a confident reading you get a `WARNING` in the log and at the top of the report — check the video then.

## A · Track B on Colab — `notebooks/match_colab.ipynb`

| Step | Runtime | Time | What happens |
|---|---|---|---|
| 0–2 Setup + details | CPU | 3 min | Drive, code, packages; edit period times / URL / direction |
| 3 Download | CPU | 5–15 min | 720p ≤30 fps, no re-encode; re-uses the pilot downloads in `videos/pilot_2026/` |
| 4 Smoke test | **GPU** | 3–5 min | 60 s with team colours → stills; prints a full-match time estimate |
| 5 Full match | **GPU** | ~1–1.5 h on T4 (estimate) | 10-min chunks, each its own process; resumable |
| 6 Results | any | 1 min | report preview + `<id>_trackb_bundle.zip` download |
| 7 Post-only | CPU | 2 min | rebuild report/moments after changing direction or swap |

**If Colab drops you** (RAM crash, "GPU limit", idle timeout): reconnect, run 0–1, then the step you were on. Finished chunks are skipped; at most one 10-min chunk is redone.

**Why it no longer runs out of RAM**
- The notebook kernel never imports PyTorch; every chunk is a fresh subprocess (memory returned to the OS every 10 min of video).
- Frames the tracker skips are only `grab()`bed, not converted; only 1 in N is decoded to an image.
- YOLO runs in batches of 8 on the GPU (fewer, larger GPU calls), FP16.
- The team-colour model is fitted once per match (`team-state.joblib`) and re-used by every chunk.
- No re-encoding anywhere. Half clips (step 8) are stream copies that take seconds.
- `MALLOC_ARENA_MAX=2` and `expandable_segments` for PyTorch stop slow memory creep; a CUDA out-of-memory retries the chunk with batch 2.

**Saving compute units**: do steps 1–3 on a CPU runtime; switch to GPU only for 4–5. `--max-chunks N` (driver flag) stops after N chunks if you want to spread a match over two sessions.

**Local alternative (no Colab limits)**: the RTX 2070 laptop runs the same driver.
One-time setup: double-click **`Setup Track B.bat`** (Python 3.10 venv, PyTorch CUDA 11.8, packages, FFmpeg/Deno, models — all on D:, safe to re-run). Earlier downloads in `D:\Video-Bola-Analis` (`<slug>.mp4` from the PowerShell pilot) are found through `video_aliases` in the config, so nothing is downloaded twice. See *Laptop: keep the PC usable* below for running it.

## A2 · Runs that keep going with the browser closed

| Where | Cost | Background | Notes |
|---|---|---|---|
| **Kaggle** — `notebooks/match_kaggle.ipynb` | free (weekly GPU quota) | yes: *Save Version → Save & Run All (Commit)* | 2 × T4 → two matches at once; ≤ 12 h per run; needs a phone-verified account for Internet + GPU |
| **Laptop (RTX 2070)** — `scripts/run_queue.py` | free | yes, as long as the laptop is on | no quota, no time limit; best overnight |
| Colab Pro+ | paid | yes (Pro+ only) | the Colab notebook as is |

Kaggle, step by step: import `match_kaggle.ipynb` → *Session options*: GPU T4 x2, Internet On → edit cell 1 (match ids, period times) → **Save Version → Save & Run All (Commit)** → close the tab. Results: the version's *Output* → `deliverables/` (bundle zips, smoke stills, logs). To resume a failed run, add the previous version's output as an input and commit again.

### Laptop: keep the PC usable while it runs

Double-click **`Jalankan Track B.bat`** (repo root): it runs every match that has period times with the **light** profile and can be minimised; close it any time and double-click again to resume. Or from PowerShell:

```powershell
python scripts\run_queue.py --config matches\persiba_2026-27.json --matches ready --root D:\Video-Bola-Analis --download --profile light
```

| Profile | CPU | GPU | RAM | Use when |
|---|---|---|---|---|
| `light` | 2 threads on 2 cores, **idle priority** (yields to anything you open) | batch 2, max 50% VRAM, 0.25 s rest after each batch | stops a chunk above total RAM − 3.5 GB (4.5 GB on an 8 GB laptop), retries at batch 1; team-colour model kept on the GPU | you are working on the PC |
| `balanced` | 4 threads, half the cores, below-normal priority | batch 4, max 70% VRAM, 0.05 s rest | stop at total − 2.5 GB | light use (browsing) |
| `full` (default) | all | batch 8, no cap | no stop | overnight / Colab / Kaggle |

Fine-tune any value on top of a profile: `--cpu-threads`, `--cpu-cores`, `--priority normal|below|idle`, `--vram-fraction 0.4`, `--gpu-pause 0.5`, `--max-ram-gb 6`. Each finished chunk prints its peak RAM, so you can set `--max-ram-gb` from real numbers. GPU *load* itself can't be hard-capped on a GeForce card; the rests between batches are what keep it (and the fan) down. Expect `light` to take roughly 2–3× longer than `full`.

## B · Track A: review Track B, then fill the gaps — [Pitchside Tagger](https://claude.ai/artifact/9d6kEj3CJLuLTadYPLgNBp)

1. **New match** → *Match ID* = the config `match_id`; teams; *Home attacks in 1st half*; **Kick-off times in the video** = the period starts from the config (e.g. `17:00` and `1:20:26`). The match clock then skips pre-match and half-time, and the 1st/2nd-half button switches by itself as the video plays.
2. Lineups: **From squads → Pick XI** (11 players, tap the keeper). Load the master video.
3. **Review (Track B)** tab → *Import candidates CSV* → `<match_id>_candidates.csv` from the bundle. For each proposal (pass, cross, shot, tackle, interception, recovery, clearance):
   - **V** plays the clip (3 s before → 2 s after), the mini pitch shows Track B's start/end.
   - Type the shirt number(s), fix team / event / outcome if needed → **Enter** = accept. **X** = reject (not a real event), **N** = skip, **B** = back.
   - Numbers are remembered per Track B track, so later candidates from the same track arrive pre-filled. Once a few tracks are mapped, **Accept mapped OK passes** clears the easy ones in bulk.
   - The queue shows the **Main queue** (priority 1–2, about 40–45 % of all candidates). A failed pass and the interception that follows are **one card**: type the passer and the "Intercepted by #" number, Enter decides both. *Low priority* (ball was out of view, blips) is one dropdown away if you have time left.
   - **+ Tag something missed here** jumps the video to that moment; tag it with the normal panel (shots, fouls, anything Track B can't see).
4. Optional, once per few matches — **measure Track B**: pick a 10–15 min stretch you have NOT reviewed yet, tag it fully by hand (normal tagging panel), download the CSV, then run
   `python scripts/evaluate_candidates.py --candidates <id>_candidates.csv --events <id>_events.csv`
   → recall / precision per event type. That tells us which thresholds to tune.
5. Optional extra pass: **Moments → Imported (Track B)** with `moments/track_*_moments.csv` to watch a player's ball-near moments.
6. **Download CSV** (`<match_id>_events.csv`) and **Download review (JSON)** (`<match_id>_review.json`). Both land in Downloads; the database step picks them up from there.

Track B finds roughly half of the real on-ball actions (the ball is often not visible), so expect to accept/fix most candidates and still add some by hand. Fastest order: shots first (filter *Type = Shot*), then passes in the final third, then the rest.

### Shirt numbers (optional)

Turn on `READ_NUMBERS = True` in the notebook (or `--read-numbers` locally) to read shirt numbers with OCR. Track B then writes `<match_id>_track_map.csv`, and the Tagger review pre-fills the player number ("OCR suggests #7") so you confirm instead of type. It adds GPU time and accuracy on broadcast footage is hit-and-miss — it's a time-saver, not a replacement for the review. Numbers land in the DB `track_map` table (`source='ocr'`).

## D · Save to the database

Double-click **`Simpan ke Database.bat`**. It writes every match into **`D:\Video-Bola-Analis\football.db`** (SQLite: one file, nothing to install — it is part of Python). It reads the Track B results (run folder, or a `*_trackb_bundle.zip` from Kaggle/Colab placed in Downloads) and your two Tagger downloads. Run it again after more review; each match is replaced, never duplicated.

Tables: `matches, tracking, moments, candidates (with your accept/reject), track_map, events, team_metrics, team_heatmaps`. Ready-made views: `v_player_events` (per player and event: count, xG, xT), `v_team_half`, `v_review_progress`. To look inside: *DB Browser for SQLite* (free), or Python `pandas.read_sql`. Local laptop runs save Track B into it automatically at the end.

## C · Bring it together — [League Hub](https://claude.ai/artifact/8cgsEsf9yeqHh2fmrnkLCT) → My analysis

Import the Tagger CSV **and** `report/<id>_team_report.json` (one or both, any order — same match merges into one entry). A **Match summary** card (possession, press, counter-press regain, width/length, team distance / high-intensity / sprint / top speed; per-player physical once OCR numbers exist). Per half: score, xG, shot map (Track A); possession (time share), block height in/out of possession, back line, width/length, territory and position heatmaps (Track B), each with a data-quality line.

**Scouting an opponent:** open the club in the Hub (Clubs → club). Every imported Track B report involving that club is combined into a *Scouting profile*: per-match row from that club's point of view, averages, 1st→2nd half change, and an average position heatmap. Run Track B on 2–3 of the opponent's matches (they don't need tagging) and import the reports.

**Reading the Track B numbers**: they are proxies from a broadcast camera — compare halves and matches, not official stats. If a team's back line comes out near 5–10 m and its block very deep for the whole match, check `home_dir_1h` first (a wrong direction mirrors everything).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `fill periods 1 and 2 in the config` | Step 2: enter P1/P2 start and end |
| Home team shown in orange on the stills | Step 2: `SWAP_TEAMS = yes`, then step 7 if the run is already done |
| Team colours wrong (both teams one colour, keepers dominate) | Delete `runs/<id>/team-state.joblib`, rerun step 4 (re-fits from ~100 frames spread over the 1st half) |
| `Period end ... is past the video end` | The period time is wrong or the download is incomplete |
| Chunk failed | The last log lines are printed; full log in `runs/<id>/logs/`. Re-run step 5 |
| Report skipped ("direction could not be read") | Set `home_dir_1h` (R/L) in step 2, run step 7 |
| `WARNING: Config says home_dir_1h=… but the tracking says …` | Watch the first seconds of the 1st half: which goal does the home team defend? Fix `home_dir_1h` (or set it to null), run step 7 |
| 60 fps download | Fine: sampling auto-adjusts to ~10 frames/s (stride 6) |
