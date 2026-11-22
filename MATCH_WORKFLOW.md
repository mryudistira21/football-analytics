# Match workflow — PERSIBA 2026-27 (production)

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
- **Direction**: `home_dir_1h` = which way the *home* team attacks on screen in the 1st half (R = left→right). For away matches, "home" is the opponent.

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

## B · Track A tagging — [Pitchside Tagger](https://claude.ai/artifact/9d6kEj3CJLuLTadYPLgNBp)

1. **New match** → *Match ID* = the config `match_id`; teams; *Home attacks in 1st half*; **Kick-off times in the video** = the period starts from the config (e.g. `17:00` and `1:20:26`). The match clock then skips pre-match and half-time, and the 1st/2nd-half button switches by itself as the video plays.
2. Lineups: **From squads → Pick XI** (11 players, tap the keeper).
3. Load the master video. Tag in three levels — stop at the level time allows:

| Level | Tag | Gives you | Time / match |
|---|---|---|---|
| 1 | every **shot** + the **pass/cross** that set it up | xG, xA, shot map, chance creators | 30–45 min |
| 2 | + all **crosses**, **recoveries/interceptions** | crossing map, where PERSIBA wins the ball | +45 min |
| 3 | + all passes | pass network, xT, progressive passes | 3–4 h |

   Use 1.5–2× playback outside the final third; set the half buttons (1st/2nd) correctly — stoppage time stays in its half.
4. **Reel → Imported (Track B)**: load `moments/track_*_moments.csv` from the bundle to jump through ball-near moments; mark wrong ones.
5. **Download CSV**.

## C · Bring it together — [League Hub](https://claude.ai/artifact/8cgsEsf9yeqHh2fmrnkLCT) → My analysis

Import the Tagger CSV **and** `report/<id>_team_report.json` (one or both, any order — same match merges into one entry). Per half: score, xG, shot map (Track A); possession proxy, block height in/out of possession, back line, width/length, territory and position heatmaps (Track B), each with a data-quality line.

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
| Report skipped | Set `home_dir_1h`, run step 7 |
| 60 fps download | Fine: sampling auto-adjusts to ~10 frames/s (stride 6) |
