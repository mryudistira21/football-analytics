# Status brief — paste this to ChatGPT

Copy everything below the line into ChatGPT whenever you start or resume a session with it.
Update it as the project moves. Keep it short.

---

**Project:** Football analytics from match video — passing, shooting, crossing, hot areas, xG, xA.
**First match:** PERSIBA Bantul vs Gresik United (Liga Nusantara 2026/27), 67-second test clip.

**Shared repo (our office):** https://github.com/mryudistira21/football-analytics
Read README.md, TASKS.md and DECISIONS.md there before suggesting steps. Put code and notes in the repo, not only in chat.

**Two AIs, separate lanes:**
- You (ChatGPT): the local CV pipeline — Windows env setup, detection + tracking, exporting per-frame player/ball positions to CSV.
- Claude: the tagging app, the xG/xA model (`src/xg_model.py`), heatmaps from position CSVs, code review.
Please don't give environment steps for the tagging app, and I'll keep Claude out of your tracking pipeline. If you change a shared decision, note it in DECISIONS.md so we don't collide.

**Hardware (settled):** RTX 2070, 8 GB VRAM, driver 526.98 → PyTorch **cu118** build. Everything installs on **D:** (`D:\Codingan\football-analytics`), C: is nearly full.

**Decisions already made (don't re-open):**
- Python **3.10** (3.11 ok, avoid 3.13).
- Models: **roboflow/sports** football-trained models (players, ball, pitch keypoints), not generic YOLO — we need pitch keypoints for real-pitch coordinates.
- cu118 PyTorch (driver too old for CUDA 12.x).

**Where we are:** environment install on D: in progress; CUDA must report True. Next: download the sports models into `data/`, crop the broadcast picture (the clip is 2400×1080 with phone bars — real picture is 1920×1080 starting at x=240; use `src/crop_broadcast.py`), then run tracking.

**What I need from you next:** help finish the env install and get PLAYER_TRACKING running on the cropped clip, then an export script that writes per-frame player positions (in metres, split by team) to a CSV. Claude will review that script and build the heatmaps from its output.
