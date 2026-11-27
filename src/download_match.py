"""Download a match video from YouTube with yt-dlp, ready for the Tagger.

Light on CPU, memory and disk by default:
  * 720p, H.264 (avc1) in .mp4, so every browser plays it in the Tagger.
  * 30 fps cap: picks YouTube's own 30 fps stream when there is one. Half the
    frames of 60 fps, half the file size, and no re-encoding.
  * --start/--end trims WITHOUT re-encoding: the cut snaps to the nearest
    keyframe (usually within a few seconds), so the CPU stays idle.
    Add --exact only if you need frame-precise cuts (re-encodes, slower).
  * Checks free disk space before downloading.
  * --crop auto removes black bars (only needed for screen recordings).

WHERE VIDEOS GO
  Outside the repo, so the editor never indexes them. Order of preference:
  --out, then the FOOTBALL_VIDEO_DIR environment variable, then a sibling
  "football-video" folder next to the repo.
    setx FOOTBALL_VIDEO_DIR D:\\Video-Bola-Analis     (then open a NEW window)

JAVASCRIPT RUNTIME
  YouTube now needs a JS runtime for some formats. Install Deno once:
      winget install DenoLand.Deno
  and the yt-dlp extras:  pip install -U "yt-dlp[default]"
  Node or Bun also work if Deno isn't there; the script picks what it finds.

Usage:
    python src/download_match.py URL --name persiba_vs_gresik
    python src/download_match.py URL --start 17:00 --end 37:00 --name test20
    python src/download_match.py URL --height 1080 --fps 60
    python src/download_match.py URL --list
For personal analysis only: the footage belongs to the league or broadcaster.
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path(os.environ.get("FOOTBALL_VIDEO_DIR") or REPO.parent / "football-video")


def to_seconds(t: str) -> float:
    secs = 0.0
    for p in t.strip().split(":"):
        secs = secs * 60 + float(p)
    return secs


def js_runtimes() -> dict:
    """Use whichever JS runtime is installed (yt-dlp only enables Deno by default)."""
    for name in ("deno", "node", "bun"):
        if shutil.which(name):
            return {name: {}}
    return {}


def format_selector(height: int, fps: int) -> str:
    f = f"[fps<={fps}]" if fps else ""
    h = f"[height<={height}]"
    return "/".join([
        f"bv*{h}{f}[vcodec^=avc1]+ba[ext=m4a]",   # H.264 + AAC at the fps cap: no re-encode, plays everywhere
        f"bv*{h}{f}+ba",                          # any codec at the fps cap
        f"bv*{h}[vcodec^=avc1]+ba[ext=m4a]",      # no stream at the cap: take H.264 at any fps
        f"bv*{h}+ba",
        f"b{h}",                                  # last resort: combined stream
    ])


def estimate_bytes(info: dict) -> float | None:
    fmts = info.get("requested_formats") or [info]
    total = 0.0
    for f in fmts:
        size = f.get("filesize") or f.get("filesize_approx")
        if not size:
            return None
        total += size
    return total


def gb(n: float) -> str:
    return f"{n / 1024**3:.2f} GB"


def resolve_output(info: dict, out_dir: Path, stem: str) -> Path | None:
    """The final merged file, not an intermediate video/audio component.

    yt-dlp's top-level `filepath` is set after merge + post-processing; a
    `requested_downloads` entry can still point at a component. Try both, then
    fall back to the newest matching file actually on disk.
    """
    cands: list[str] = []
    rd = info.get("requested_downloads") or []
    if rd:
        cands += [rd[0].get("filepath"), rd[0].get("_filename")]
    cands += [info.get("filepath"), info.get("_filename")]
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    if "%" not in stem:
        hits = sorted(out_dir.glob(stem + ".*"), key=lambda p: p.stat().st_mtime, reverse=True)
        vids = [h for h in hits if h.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")]
        if vids or hits:
            return (vids or hits)[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Download a match video for tagging.")
    ap.add_argument("url")
    ap.add_argument("--height", type=int, default=720, help="max height: 480, 720 (default), 1080")
    ap.add_argument("--fps", type=int, default=30, help="max frame rate (default 30; 0 = no cap)")
    ap.add_argument("--start", help="start time mm:ss or hh:mm:ss (download only a part)")
    ap.add_argument("--end", help="end time mm:ss or hh:mm:ss")
    ap.add_argument("--exact", action="store_true", help="frame-precise cuts (re-encodes: slow, CPU-heavy)")
    ap.add_argument("--crop", default="none", help="'none' (default), 'auto', or w:h:x:y")
    ap.add_argument("--keep-original", action="store_true", help="keep the uncropped file after --crop")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads when re-encoding (default 4)")
    ap.add_argument("--name", help="output file name without extension (default: video title)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output folder (default: {DEFAULT_OUT})")
    ap.add_argument("--cookies", type=Path, help="Netscape cookies.txt for YouTube "
                    "(gets past the 'confirm you're not a bot' check on datacenter IPs)")
    ap.add_argument("--list", action="store_true", help="list available qualities and exit")
    a = ap.parse_args()

    try:
        import yt_dlp
    except ImportError:
        print('yt-dlp is not installed. Run:  pip install -U "yt-dlp[default]"')
        return 1
    if shutil.which("ffmpeg") is None:
        print("ffmpeg was not found on PATH. Install it (winget install Gyan.FFmpeg) and open a new window.")
        return 1

    runtimes = js_runtimes()
    if not runtimes:
        print("Note: no JavaScript runtime found, so some YouTube formats may be missing.\n"
              "      Fix once with:  winget install DenoLand.Deno   (then open a new window)\n")
    base_opts = {"js_runtimes": runtimes} if runtimes else {}
    if a.cookies:
        ck = a.cookies
        if ck.is_dir():   # a Kaggle dataset folder was passed: find the cookie file inside it
            files = [p for p in sorted(ck.rglob("*")) if p.is_file()]
            ck = next((p for p in files if "cookie" in p.name.lower()), None) \
                or next((p for p in files if p.suffix.lower() == ".txt"), None)
        if not ck or not ck.is_file():
            print(f"--cookies: no cookie file found at {a.cookies}")
            return 1
        base_opts["cookiefile"] = str(ck)
        print(f"Using cookies from {ck}")

    if a.list:
        with yt_dlp.YoutubeDL({**base_opts, "listformats": True}) as ydl:
            ydl.download([a.url])
        return 0

    out_dir: Path = a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = a.name or "%(title).80B"
    opts = {
        **base_opts,
        "format": format_selector(a.height, a.fps),
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / f"{stem}.%(ext)s"),
        "restrictfilenames": a.name is None,
        "noplaylist": True,
    }
    start = to_seconds(a.start) if a.start else 0.0
    end = to_seconds(a.end) if a.end else None
    if a.start or a.end:
        from yt_dlp.utils import download_range_func
        opts["download_ranges"] = download_range_func(None, [(start, end if end else float("inf"))])
        opts["force_keyframes_at_cuts"] = a.exact          # False = stream copy, no re-encode
        if a.exact:
            opts["external_downloader_args"] = {"ffmpeg_o": ["-threads", str(a.threads), "-preset", "veryfast"]}

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(a.url, download=False)

        # Disk check before a single byte is written.
        est = estimate_bytes(info)
        duration = info.get("duration") or 0
        if est and duration and (a.start or a.end):
            span = (end or duration) - start
            est *= max(0.0, min(1.0, span / duration))
        free = shutil.disk_usage(out_dir).free
        if est:
            need = est * (2.1 if a.crop != "none" else 1.1)   # cropping briefly keeps two copies
            print(f"Estimated download {gb(est)}; free space on {out_dir.anchor or out_dir} {gb(free)}")
            if need > free:
                print(f"Not enough disk space (needs about {gb(need)}). Free some space or use --out on another drive.")
                return 1

        info = ydl.process_ie_result(info, download=True)

    path = resolve_output(info, out_dir, stem)
    if not path or not path.exists():
        print("Download finished but the merged output file could not be located in", out_dir)
        return 1
    fps = info.get("fps")
    print(f"\nSaved: {path}")
    print(f"{info.get('width')}x{info.get('height')} @ {fps} fps, codec {info.get('vcodec')}, {gb(path.stat().st_size)}")
    over_fps = a.fps and fps and fps > a.fps + 1
    if over_fps:
        print(f"Note: YouTube had no {a.fps} fps stream at this height, so this is {fps} fps. "
              f"Add --crop auto, or --fps {a.fps} on a re-encode, to bring it down; "
              "the Tagger plays it fine either way.")

    if a.crop != "none":
        from crop_broadcast import crop_video, detect_crop
        crop = detect_crop(path) if a.crop == "auto" else a.crop
        if not crop:
            print("No black bars found: no crop needed.")
        else:
            cropped = path.with_name(path.stem + "_crop.mp4")
            cap = a.fps if over_fps else None   # re-encoding anyway: apply the fps cap too
            crop_video(path, cropped, crop, a.threads, cap)
            if not a.keep_original:
                path.unlink()
                cropped.rename(path)
                cropped = path
            print(f"Cropped to {crop}: {cropped}")

    print("\nIn the Tagger, click 'Choose video file' and pick this file.")
    if a.start and not a.exact:
        print("The cut snapped to the nearest keyframe, so the clip may begin a few seconds before "
              f"{a.start}. Read the scoreboard on the first frame for 'Match clock at video 0:00'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
