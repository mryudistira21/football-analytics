#!/usr/bin/env python3
"""Run several matches unattended: one worker per GPU, matches taken from a queue.

Each match = smoke test (only when no team-colour state exists yet) + full run,
exactly as scripts/run_match.py does them; everything stays resumable.
Matches without period times are skipped with a message.

    # Kaggle (2 x T4): two matches at the same time
    python scripts/run_queue.py --config matches/persiba_2026-27.json \
        --matches persiba_persika_20261121,persika_persinga_20261113 \
        --root /kaggle/working/fa --work /tmp/work --video-dir /tmp/videos --gpus 0,1 --download

    # Laptop (one GPU, all matches that have period times), light on the PC while you use it
    python scripts/run_queue.py --config matches/persiba_2026-27.json --matches ready \
        --root D:/Video-Bola-Analis --download --profile light

Any extra flag (e.g. --profile, --max-ram-gb, --gpu-pause) is passed to scripts/run_match.py.

Progress: <root>/queue/<match>.log per match and <root>/queue/status.json.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "run_match.py"


def has_periods(m: dict) -> bool:
    p = m.get("periods") or {}
    return bool(p.get("1")) and bool(p.get("2"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run several matches unattended (one worker per GPU).")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--matches", required=True,
                    help="comma-separated match_ids, or 'ready' = every match with a video URL and period times")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--video-dir", type=Path)
    ap.add_argument("--gpus", default="0", help="GPU indices, e.g. 0,1 (one match per GPU at a time)")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--no-smoke", action="store_true", help="skip the 60 s smoke test")
    ap.add_argument("--no-teams", action="store_true", help="pass --no-teams to every run (testing)")
    a, passthrough = ap.parse_known_args()  # e.g. --profile light --max-ram-gb 6 go to every run

    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    by_id = {m["match_id"]: m for m in cfg["matches"]}
    if a.matches.strip() == "ready":
        ids = [m["match_id"] for m in cfg["matches"] if m.get("video_url") and has_periods(m)]
    else:
        ids = [x.strip() for x in a.matches.split(",") if x.strip()]
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise SystemExit(f"Not in the config: {unknown}")
    todo = [i for i in ids if has_periods(by_id[i])]
    for i in ids:
        if i not in todo:
            print(f"[skip] {i}: no period times yet (fill periods 1 and 2 in the config).", flush=True)
    if not todo:
        return 1

    qdir = a.root / "queue"
    qdir.mkdir(parents=True, exist_ok=True)
    status = {i: {"state": "queued"} for i in todo}
    lock = threading.Lock()

    def save() -> None:
        with lock:
            (qdir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    common = ["--config", str(a.config), "--root", str(a.root)]
    if a.work:
        common += ["--work", str(a.work)]
    if a.video_dir:
        common += ["--video-dir", str(a.video_dir)]
    if a.download:
        common += ["--download"]
    if a.no_teams:
        common += ["--no-teams"]
    common += passthrough

    jobs: queue.Queue[str] = queue.Queue()
    for i in todo:
        jobs.put(i)

    def worker(gpu: str) -> None:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
        while True:
            try:
                mid = jobs.get_nowait()
            except queue.Empty:
                return
            log = qdir / f"{mid}.log"
            t0 = time.time()
            status[mid] = {"state": "running", "gpu": gpu, "started": time.strftime("%H:%M:%S")}
            save()
            print(f"[gpu {gpu}] start {mid}", flush=True)
            steps = []
            if not a.no_smoke and not (a.root / "runs" / mid / "team-state.joblib").exists():
                steps.append(["--smoke"])
            steps.append([])
            code = 0
            with log.open("a", encoding="utf-8") as fh:
                for extra in steps:
                    fh.write(f"\n===== {' '.join(['run_match', mid, *extra])} =====\n")
                    fh.flush()
                    code = subprocess.run([sys.executable, str(DRIVER), *common, "--match", mid, *extra],
                                          stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).returncode
                    if code != 0:
                        break
            mins = (time.time() - t0) / 60
            status[mid] = {"state": "done" if code == 0 else f"failed (exit {code})", "gpu": gpu,
                           "minutes": round(mins, 1), "log": str(log)}
            save()
            print(f"[gpu {gpu}] {'done' if code == 0 else 'FAILED'} {mid} in {mins:.0f} min"
                  + ("" if code == 0 else f" — see {log}"), flush=True)

    save()
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        with lock:
            line = " | ".join(f"{k}: {v['state']}" for k, v in status.items())
        print(f"[{time.strftime('%H:%M')}] {line}", flush=True)
    failed = [k for k, v in status.items() if v["state"] != "done"]
    print("All done." if not failed else f"Failed: {failed}", flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
