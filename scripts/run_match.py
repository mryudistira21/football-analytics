#!/usr/bin/env python3
"""Run Track B for one whole match, resumably, from a match config file.

One command does: video download (if missing) -> models -> chunked tracking
(each chunk is its own subprocess, so RAM/VRAM is released between chunks)
-> combine -> moment cut-lists -> team report -> a small bundle zip.

Re-run the same command after a Colab disconnect: finished chunks are skipped.
This driver never imports torch, so the notebook kernel stays light.

    # 1-minute check with an annotated preview + team-colour stills
    python scripts/run_match.py --config matches/persiba_2026-27.json \
        --match persiba_gresik_20261113 --root /content/drive/MyDrive/football-analysis \
        --work /content/work --smoke

    # the full match (resumable)
    python scripts/run_match.py --config matches/persiba_2026-27.json \
        --match persiba_gresik_20261113 --root /content/drive/MyDrive/football-analysis \
        --work /content/work

    # only rebuild combine/moments/report (no GPU), e.g. after setting swap_teams
    python scripts/run_match.py ... --post-only

Layout under --root (persistent; Google Drive on Colab, D:\\ on Windows):
    videos/<match_id>.mp4
    models/
    runs/<match_id>/chunks/*.csv + *.meta.json   (one per finished chunk)
    runs/<match_id>/logs/*.log
    runs/<match_id>/team-state.joblib            (fitted once, reused by every chunk)
    runs/<match_id>/smoke/                       (preview mp4 + stills)
    runs/<match_id>/<match_id>_tracking.csv      (combined)
    runs/<match_id>/moments/                     (Tagger cut-lists)
    runs/<match_id>/report/                      (team report json + md)
    runs/<match_id>/<match_id>_trackb_bundle.zip (what you download)
--work is fast local scratch (/content/work on Colab). Optional; defaults to --root.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TRACKER = REPO / "scripts" / "track_video.py"
MOMENTS = REPO / "scripts" / "build_player_moments.py"
REPORT = REPO / "scripts" / "team_report.py"
DOWNLOADER = REPO / "src" / "download_match.py"
MODELS_SCRIPT = REPO / "scripts" / "download_sports_models.py"
MODEL_FILES = ("football-player-detection.pt", "football-pitch-detection.pt")
TRACK_ID_BLOCK = 100_000  # chunk k uses person IDs k*100000 + n: deterministic, resume-safe


# ---------------------------------------------------------------- config
def parse_clock(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Use MM:SS or H:MM:SS, got {value!r}")
    nums = [float(p) for p in parts]
    if any(n < 0 for n in nums):
        raise ValueError(f"Negative time {value!r}")
    return nums[0] * 60 + nums[1] if len(nums) == 2 else nums[0] * 3600 + nums[1] * 60 + nums[2]


def fmt_clock(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def load_match(config_path: Path, match_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for m in cfg.get("matches", []):
        if m.get("match_id") == match_id:
            settings = {**cfg.get("defaults", {}), **m.get("settings", {})}
            return m, settings
    ids = ", ".join(m.get("match_id", "?") for m in cfg.get("matches", []))
    raise SystemExit(f"Match {match_id!r} not in {config_path}. Known: {ids}")


def match_periods(match: dict[str, Any]) -> list[dict[str, Any]]:
    """Validated periods sorted by source time. Stoppage time stays in its half."""
    raw = match.get("periods") or {}
    out = []
    for key, rng in raw.items():
        if rng is None:
            continue
        p = int(key)
        if p not in (1, 2, 3, 4):
            raise SystemExit(f"Period {key} is not 1-4")
        start, end = parse_clock(rng[0]), parse_clock(rng[1])
        if end <= start:
            raise SystemExit(f"Period {p}: end {rng[1]} is not after start {rng[0]}")
        out.append({"period": p, "start": start, "end": end})
    have = {p["period"] for p in out}
    if not {1, 2} <= have:
        raise SystemExit(
            f"{match['match_id']}: fill periods 1 and 2 in the config (real kick-off -> final whistle, "
            "stoppage time included, as elapsed time in the downloaded video)."
        )
    if (3 in have) != (4 in have):
        raise SystemExit("Extra time needs both periods 3 and 4.")
    out.sort(key=lambda p: p["start"])
    for a, b in zip(out, out[1:]):
        if b["start"] < a["end"]:
            raise SystemExit(f"Period {b['period']} overlaps period {a['period']}; check the whistle times.")
    return out


def home_right_by_period(match: dict[str, Any], periods: list[dict[str, Any]]) -> dict[str, bool] | None:
    d = match.get("home_dir_1h")
    if d not in ("R", "L"):
        return None
    out = {"1": d == "R", "2": d != "R"}
    et = match.get("home_dir_et1")
    if any(p["period"] >= 3 for p in periods):
        if et not in ("R", "L"):
            return None
        out.update({"3": et == "R", "4": et != "R"})
    return out


# ---------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def child_env(models: Path, cache: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "FOOTBALL_MODEL_DIR": str(models),
        "HF_HOME": str(cache),
        "HUGGINGFACE_HUB_CACHE": str(cache / "hub"),
        # Fewer glibc malloc arenas: stops slow RSS creep in long multi-threaded runs.
        "MALLOC_ARENA_MAX": "2",
        # Lets PyTorch return fragmented VRAM instead of failing with OOM.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONUNBUFFERED": "1",
    })
    return env


def video_info(path: Path) -> tuple[float, float]:
    """(duration_seconds, fps) via OpenCV; no frames are decoded."""
    import cv2  # light; no torch
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (frames / fps if frames else 0.0), fps


def resources() -> str:
    parts = []
    try:
        import psutil
        vm = psutil.virtual_memory()
        parts.append(f"RAM {vm.available / 2**30:.1f}/{vm.total / 2**30:.1f} GB free")
    except Exception:
        pass
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if q.returncode == 0 and q.stdout.strip():
            name, used, total = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
            parts.append(f"GPU {name} {used}/{total} MiB used")
    except Exception:
        parts.append("GPU none")
    return " | ".join(parts) or "n/a"


def copy_if_needed(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.copying")  # unique: parallel workers may copy at once
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def youtube_id(url: str) -> str | None:
    import re
    m = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/)([A-Za-z0-9_-]{11})", url or "")
    return m.group(1) if m else None


# ---------------------------------------------------------------- steps
def ensure_video(match: dict[str, Any], root: Path, work: Path, download: bool,
                 video_dir: Path | None = None) -> Path:
    """Persistent copy in root/videos; a local working copy in work/ for fast decoding."""
    mid = match["match_id"]
    vids = video_dir or root / "videos"
    persistent = next((p for p in sorted(vids.glob(mid + ".*"))
                       if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")), None) if vids.exists() else None
    if persistent is None and vids.exists():
        # Re-use an earlier download named after the YouTube id (e.g. videos/pilot_2026/<slug>/<id>.mp4).
        yid = youtube_id(match.get("video_url") or "")
        if yid:
            hits = [p for p in vids.rglob(yid + ".*")
                    if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov") and "clips" not in p.parts]
            if hits:
                persistent = max(hits, key=lambda p: p.stat().st_size)
                log(f"Using the earlier download {persistent}")
    if persistent is None:
        if not download:
            raise SystemExit(f"No video for {mid} in {vids}. Pass --download or put {mid}.mp4 there.")
        url = match.get("video_url")
        if not url:
            raise SystemExit(f"{mid}: set video_url in the config first.")
        stage = vids if video_dir is not None else work / "videos"
        log(f"Downloading {url} (720p, <=30 fps, no re-encode) ...")
        r = subprocess.run([sys.executable, str(DOWNLOADER), url, "--height", "720", "--fps", "30",
                            "--name", mid, "--out", str(stage)])
        if r.returncode != 0:
            raise SystemExit("Download failed; see the messages above.")
        got = next(p for p in sorted(stage.glob(mid + ".*")) if p.suffix.lower() in (".mp4", ".mkv", ".webm"))
        persistent = vids / got.name
        if persistent.resolve() != got.resolve():
            log(f"Saving the master to {persistent} ...")
            copy_if_needed(got, persistent)
    if video_dir is not None:
        return persistent  # --video-dir is already fast local disk (Kaggle/laptop): no second copy
    local = work / "videos" / persistent.name
    if local.resolve() != persistent.resolve():
        if not (local.exists() and local.stat().st_size == persistent.stat().st_size):
            log("Copying the master video to fast local disk ...")
        copy_if_needed(persistent, local)
    return local


def ensure_models(root: Path) -> Path:
    models = root / "models"
    if all((models / f).is_file() and (models / f).stat().st_size > 1_000_000 for f in MODEL_FILES):
        return models
    log("Downloading the Roboflow Sports models ...")
    subprocess.run([sys.executable, str(MODELS_SCRIPT), "--out", str(models)], check=True)
    for f in MODEL_FILES:
        if not (models / f).is_file():
            raise SystemExit(f"Model missing after download: {models / f}")
    return models


def resolve_stride(settings: dict[str, Any], fps: float) -> None:
    """'auto' -> about 10 samples per second whatever the source fps (30 fps -> 3, 60 fps -> 6)."""
    if settings.get("frame_stride") in (None, 0, "auto"):
        settings["frame_stride"] = max(1, int(round(fps / 10.0)))


def stride_from_chunks(run: Path, settings: dict[str, Any]) -> None:
    if settings.get("frame_stride") not in (None, 0, "auto"):
        return
    for m in sorted((run / "chunks").glob("*.meta.json")):
        try:
            settings["frame_stride"] = int(json.loads(m.read_text(encoding="utf-8"))["frame_stride"])
            return
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            continue
    raise SystemExit("No finished chunks found to read the frame stride from.")


def plan_chunks(periods: list[dict[str, Any]], chunk_seconds: float) -> list[dict[str, Any]]:
    chunks, k = [], 0
    for p in sorted(periods, key=lambda x: x["period"]):
        s = p["start"]
        while s < p["end"] - 1e-6:
            e = min(p["end"], s + chunk_seconds)
            if p["end"] - e < 30:  # never leave a sliver chunk at the whistle
                e = p["end"]
            chunks.append({"k": k, "period": p["period"], "start": round(s, 3), "end": round(e, 3)})
            k += 1
            s = e
    return chunks


def chunk_name(mid: str, c: dict[str, Any]) -> str:
    return f"{mid}_p{c['period']}_c{c['k']:02d}_{int(c['start']):05d}-{int(c['end']):05d}"


def chunk_done(csv_path: Path, c: dict[str, Any], mid: str, settings: dict[str, Any]) -> bool:
    meta_path = csv_path.with_suffix(".meta.json")
    if not (csv_path.is_file() and meta_path.is_file()):
        return False
    try:
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        tol = max(0.5, 1.5 * int(settings["frame_stride"]) / float(m.get("fps") or 30))
        return (
            m.get("status") == "complete" and m.get("complete") is True
            and m.get("match_id") == mid and int(m.get("period", -1)) == c["period"]
            and int(m.get("frame_stride", -1)) == int(settings["frame_stride"])
            and int(m.get("imgsz", -1)) == int(settings["imgsz"])
            and int(m.get("track_id_offset", -1)) == c["k"] * TRACK_ID_BLOCK
            and abs(float(m.get("requested_start_seconds", -1)) - c["start"]) < 0.01
            and abs(float(m.get("requested_end_seconds", -1)) - c["end"]) < 0.01
            and m.get("actual_end_seconds") is not None
            and float(m["actual_end_seconds"]) >= c["end"] - tol - 1.0
        )
    except (ValueError, TypeError, OSError, json.JSONDecodeError):
        return False


def tracker_cmd(video: Path, models: Path, cache: Path, mid: str, c: dict[str, Any],
                settings: dict[str, Any], out_csv: Path, team_state: Path, batch: int,
                home_dir: str, extra: list[str]) -> list[str]:
    return [
        sys.executable, str(TRACKER), "--input", str(video), "--models", str(models),
        "--cache", str(cache), "--match-id", mid, "--period", str(c["period"]),
        "--home-dir-1h", home_dir,
        *(["--period-direction", f"{c['period']}:R"] if c["period"] >= 3 else []),
        "--start-seconds", str(c["start"]), "--end-seconds", str(c["end"]),
        "--frame-stride", str(settings["frame_stride"]), "--imgsz", str(settings["imgsz"]),
        "--pitch-imgsz", str(settings["pitch_imgsz"]), "--pitch-every", str(settings["pitch_every"]),
        "--batch", str(batch), "--team-model-state", str(team_state),
        "--cluster-0-side", "home", "--track-id-offset", str(c["k"] * TRACK_ID_BLOCK),
        # Team colours are fitted once (first run without a saved state) from crops spread
        # over the first half, then reused by every chunk.
        "--team-sample-start", str(settings["_team_sample_start"]),
        "--team-sample-seconds", str(settings["_team_sample_seconds"]),
        "--team-sample-interval", str(settings.get("team_sample_interval", 15)),
        "--team-sample-max-crops", str(settings.get("team_sample_max_crops", 600)),
        "--team-classify-every", str(settings.get("team_classify_every", 10)),
        "--output", str(out_csv), "--overwrite", *extra,
    ]


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, str]:
    """Run a child, tee a few progress lines, keep the full log on disk."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail: list[str] = []
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env, cwd=str(REPO), bufsize=1)
        assert proc.stdout is not None
        n = 0
        for line in proc.stdout:
            fh.write(line)
            tail.append(line)
            tail = tail[-60:]
            if line.startswith("frames "):
                n += 1
                if n % 10 == 0:  # every 1000 sampled frames
                    print("    " + line.rstrip(), flush=True)
            elif any(k in line for k in ("Error", "error", "Traceback", "Team classifier", "Loaded team", "Saved team")):
                print("    " + line.rstrip(), flush=True)
        proc.wait()
    return proc.returncode, "".join(tail)


def extract_stills(preview: Path, out_dir: Path, count: int = 4) -> list[Path]:
    import cv2
    cap = cv2.VideoCapture(str(preview))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    paths = []
    for i in range(count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (i + 1) / (count + 1)))
        ok, frame = cap.read()
        if ok:
            frame = cv2.resize(frame, (960, int(960 * frame.shape[0] / frame.shape[1])))
            p = out_dir / f"still_{i + 1}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            paths.append(p)
    cap.release()
    return paths


def summarize_smoke(csv_path: Path) -> dict[str, Any]:
    teams, roles, wide, n = {}, {}, 0, 0
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n += 1
            roles[r["role"]] = roles.get(r["role"], 0) + 1
            if r["role"] == "player":
                teams[r["team"]] = teams.get(r["team"], 0) + 1
            wide += r["camera_state"] == "wide"
    return {"rows": n, "roles": roles, "player_rows_by_team": teams, "wide_share": round(wide / n, 3) if n else 0}


def combine(run: Path, mid: str, chunks: list[dict[str, Any]], swap: bool,
            out_csv: Path, periods: list[dict[str, Any]], dirs: dict[str, bool] | None) -> dict[str, Any]:
    swap_map = {"home": "away", "away": "home"}
    rows, metas, header = 0, [], None
    tmp = out_csv.with_name(out_csv.name + ".partial")
    with tmp.open("w", newline="", encoding="utf-8") as fo:
        writer = None
        for c in chunks:
            p = run / "chunks" / (chunk_name(mid, c) + ".csv")
            with p.open(newline="", encoding="utf-8") as fi:
                rd = csv.DictReader(fi)
                if header is None:
                    header = rd.fieldnames
                    writer = csv.DictWriter(fo, fieldnames=header)
                    writer.writeheader()
                elif rd.fieldnames != header:
                    raise SystemExit(f"Column mismatch in {p}")
                for r in rd:
                    if swap and r.get("team") in swap_map:
                        r["team"] = swap_map[r["team"]]
                    writer.writerow(r)
                    rows += 1
            metas.append(json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8")))
    tmp.replace(out_csv)
    fps = float(metas[0].get("fps") or 30)
    meta = {
        "match_id": mid, "status": "complete", "complete": True, "rows": rows,
        "fps": fps, "frame_stride": metas[0].get("frame_stride"),
        "periods": [{**p, "start_clock": fmt_clock(p["start"]), "end_clock": fmt_clock(p["end"])} for p in periods],
        "home_attacks_right_by_period": dirs,
        "teams_swapped_at_combine": swap,
        "pitch_geometry": "pitchside_tagger_105x68_v1",
        "chunks": [{"name": chunk_name(mid, c), "period": c["period"], "start": c["start"], "end": c["end"],
                    "processed_frames": m.get("processed_frames"), "camera_state_counts": m.get("camera_state_counts"),
                    "calibration_median_error_m": m.get("calibration_median_error_m")}
                   for c, m in zip(chunks, metas)],
        "note": "video_seconds are seconds in the original master video. Person track IDs are chunk*100000+n and are not shirt numbers.",
    }
    out_csv.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def bundle(run: Path, mid: str) -> Path:
    z = run / f"{mid}_trackb_bundle.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in ("moments", "report"):
            for f in sorted((run / sub).glob("*")) if (run / sub).exists() else []:
                if f.is_file():
                    zf.write(f, f"{sub}/{f.name}")
        meta = run / f"{mid}_tracking.meta.json"
        if meta.exists():
            zf.write(meta, meta.name)
    return z


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Resumable Track B run for one match from a config file.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--match", required=True, help="match_id from the config")
    ap.add_argument("--root", type=Path, required=True, help="persistent folder (Drive on Colab)")
    ap.add_argument("--work", type=Path, help="fast local scratch (default: --root)")
    ap.add_argument("--video-dir", type=Path,
                    help="where master videos are kept (default: <root>/videos). On Kaggle use local disk "
                         "so 2 GB videos don't count against the saved output.")
    ap.add_argument("--download", action="store_true", help="download the video if it is missing")
    ap.add_argument("--download-only", action="store_true",
                    help="only fetch the video (+ models) into --root; no GPU needed")
    ap.add_argument("--smoke", action="store_true", help="60 s check with preview + stills, then stop")
    ap.add_argument("--smoke-seconds", type=float, default=60)
    ap.add_argument("--post-only", action="store_true", help="skip tracking; rebuild combine/moments/report")
    ap.add_argument("--chunk-seconds", type=float)
    ap.add_argument("--batch", type=int, help="frames per YOLO call (0 = auto)")
    ap.add_argument("--no-teams", action="store_true",
                    help="skip home/away classification (faster; team = unknown everywhere)")
    ap.add_argument("--max-chunks", type=int, help="stop after this many new chunks (budget control)")
    a = ap.parse_args()

    match, settings = load_match(a.config, a.match)
    if a.chunk_seconds:
        settings["chunk_seconds"] = a.chunk_seconds
    if a.batch is not None:
        settings["batch"] = a.batch
    no_teams = ["--no-team-classification"] if a.no_teams else []
    mid = match["match_id"]
    root = a.root.expanduser().resolve()
    work = (a.work or a.root).expanduser().resolve()
    run = root / "runs" / mid
    for d in (run / "chunks", run / "logs", work):
        d.mkdir(parents=True, exist_ok=True)
    if a.download_only:
        video = ensure_video(match, root, work, True, a.video_dir)
        duration, fps = video_info(video)
        ensure_models(root)
        log(f"Ready: {video} ({fmt_clock(duration)}, {fps:.2f} fps). "
            "Fill the period times for this match, then run the smoke test.")
        return 0

    periods = match_periods(match)
    dirs = home_right_by_period(match, periods)
    home_dir = match.get("home_dir_1h") if match.get("home_dir_1h") in ("R", "L") else "R"
    cache = work / "hf-cache"
    team_state = run / "team-state.joblib"
    chunks = plan_chunks(periods, float(settings["chunk_seconds"]))
    first = min(periods, key=lambda x: x["start"])
    settings["_team_sample_start"] = round(first["start"], 3)
    settings["_team_sample_seconds"] = round(min(first["end"] - first["start"],
                                                 float(settings.get("team_sample_span_seconds", 1500))), 3)
    combined = run / f"{mid}_tracking.csv"

    log(f"{match['home']} vs {match['away']} ({match.get('date', '')}) -> {run}")
    for p in sorted(periods, key=lambda x: x["period"]):
        log(f"  period {p['period']}: {fmt_clock(p['start'])} -> {fmt_clock(p['end'])} "
            f"({(p['end'] - p['start']) / 60:.1f} min)")
    log(f"  {len(chunks)} chunks of <= {settings['chunk_seconds'] / 60:.0f} min | {resources()}")

    if not a.post_only:
        video = ensure_video(match, root, work, a.download, a.video_dir)
        duration, fps = video_info(video)
        last_end = max(p["end"] for p in periods)
        if duration and last_end > duration + 1:
            raise SystemExit(f"Period end {fmt_clock(last_end)} is past the video end {fmt_clock(duration)}.")
        resolve_stride(settings, fps)
        log(f"Video {fmt_clock(duration)} at {fps:.2f} fps -> sampling every {settings['frame_stride']} frames "
            f"(~{fps / settings['frame_stride']:.1f} per second)")
        models = ensure_models(root)
        env = child_env(models, cache)

        if a.smoke:
            sm = run / "smoke"
            sm.mkdir(parents=True, exist_ok=True)
            p1 = min(periods, key=lambda x: x["start"])
            c = {"k": 0, "period": p1["period"], "start": p1["start"],
                 "end": min(p1["end"], p1["start"] + a.smoke_seconds)}
            out_csv, preview = sm / "smoke.csv", sm / "smoke_preview.mp4"
            cmd = tracker_cmd(video, models, cache, mid, c, settings, out_csv, team_state,
                              int(settings["batch"]), home_dir,
                              ["--annotated-output", str(preview), "--annotated-max-seconds", str(a.smoke_seconds), *no_teams])
            log("Smoke test (first run also fits the team colours from frames spread over the 1st half) ...")
            t0 = time.time()
            code, tail = run_logged(cmd, run / "logs" / "smoke.log", env)
            if code != 0:
                print(tail)
                raise SystemExit("Smoke test failed; full log in runs/<match>/logs/smoke.log")
            secs = time.time() - t0
            info = summarize_smoke(out_csv)
            stills = extract_stills(preview, sm)
            speed = a.smoke_seconds / secs if secs else 0
            total_min = sum(p["end"] - p["start"] for p in periods) / 60
            (sm / "smoke_summary.json").write_text(json.dumps({**info, "seconds": round(secs, 1)}, indent=2))
            log(f"Smoke OK in {secs:.0f} s (includes model loading). {info}")
            log(f"Full-match time, upper bound: ~{total_min / max(speed, 1e-6):.0f} min "
                "(this smoke run also paid for model loading and the team-colour fit, so expect less).")
            log("Open the stills: GREEN boxes = 'home', ORANGE = 'away'. "
                f"If {match['home']} is ORANGE, set \"swap_teams\": true for this match in the config.")
            for s in stills:
                print("   ", s)
            return 0

        todo = [c for c in chunks if not chunk_done(run / "chunks" / (chunk_name(mid, c) + ".csv"), c, mid, settings)]
        log(f"{len(chunks) - len(todo)} of {len(chunks)} chunks already done.")
        new = 0
        t_start = time.time()
        for c in todo:
            if a.max_chunks is not None and new >= a.max_chunks:
                log(f"Stopping after {new} new chunks (--max-chunks). Re-run to continue.")
                return 0
            name = chunk_name(mid, c)
            local_csv = work / "chunks" / mid / (name + ".csv")
            local_csv.parent.mkdir(parents=True, exist_ok=True)
            batch = int(settings["batch"])
            for attempt in (1, 2):
                log(f"chunk {c['k'] + 1}/{len(chunks)} · period {c['period']} · "
                    f"{fmt_clock(c['start'])}-{fmt_clock(c['end'])} · batch {batch or 'auto'} · {resources()}")
                t0 = time.time()
                code, tail = run_logged(
                    tracker_cmd(video, models, cache, mid, c, settings, local_csv, team_state, batch, home_dir, no_teams),
                    run / "logs" / (name + ".log"), env)
                if code == 0:
                    break
                oom = "out of memory" in tail.lower() or "cuda error" in tail.lower() or code in (-9, 137)
                if attempt == 1 and oom:
                    batch = 2
                    log("Out of memory -> retrying this chunk with batch 2.")
                    continue
                print(tail)
                raise SystemExit(f"Chunk {name} failed (exit {code}). Log: {run / 'logs' / (name + '.log')}. "
                                 "Re-run the same command to resume.")
            for suffix in (".meta.json", ".csv"):  # csv last: its presence marks the copy as finished
                copy_if_needed(local_csv.with_suffix(suffix) if suffix != ".csv" else local_csv,
                               run / "chunks" / (name + suffix))
            local_csv.unlink(missing_ok=True)
            new += 1
            el = time.time() - t0
            rate = (c["end"] - c["start"]) / el if el else 0
            left = sum(x["end"] - x["start"] for x in todo[todo.index(c) + 1:])
            log(f"  done in {el / 60:.1f} min ({rate:.2f}x real time); ~{left / max(rate, 1e-6) / 60:.0f} min left")
        log(f"Tracking finished: {new} new chunks in {(time.time() - t_start) / 60:.1f} min.")

    else:
        stride_from_chunks(run, settings)

    missing = [chunk_name(mid, c) for c in chunks
               if not chunk_done(run / "chunks" / (chunk_name(mid, c) + ".csv"), c, mid, settings)]
    if missing:
        raise SystemExit(f"{len(missing)} chunks are not finished yet (first: {missing[0]}). Run without --post-only.")

    log("Combining chunks ...")
    meta = combine(run, mid, chunks, bool(match.get("swap_teams")), combined, periods, dirs)
    log(f"  {meta['rows']:,} rows -> {combined}")

    log("Building moment cut-lists ...")
    subprocess.run([sys.executable, str(MOMENTS), "--input", str(combined),
                    "--metadata", str(combined.with_suffix(".meta.json")),
                    "--output-dir", str(run / "moments")], check=True, cwd=str(REPO))

    if dirs is None:
        log("Team report skipped: set home_dir_1h (R or L) for this match in the config, then run with --post-only.")
    else:
        log("Building the team report ...")
        subprocess.run([sys.executable, str(REPORT), "--input", str(combined), "--config", str(a.config),
                        "--match", mid, "--output-dir", str(run / "report")], check=True, cwd=str(REPO))
    z = bundle(run, mid)
    log(f"Bundle: {z}  ({z.stat().st_size / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
