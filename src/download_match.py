"""Download a match video from YouTube with yt-dlp, ready for the Tagger.

Why not a website downloader: YouTube only serves one combined video+audio
stream, and it stops at 360p. 480p and up come as separate video and audio
streams that must be merged. yt-dlp downloads both and merges them with ffmpeg.

It prefers H.264 (avc1) video in an .mp4, because every browser plays that,
so the file opens straight in the Pitchside Tagger.

Usage (from the repo folder, with the venv active):
    python src/download_match.py URL
    python src/download_match.py URL --height 1080
    python src/download_match.py URL --start 00:00 --end 15:00      # just a part
    python src/download_match.py URL --list                          # show qualities
    python src/download_match.py URL --name persiba_vs_gresik

Needs:  pip install -U yt-dlp   and   ffmpeg on PATH.
Saves to data/ (git-ignored). For personal analysis only: the footage belongs
to the league or broadcaster.
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def to_seconds(t: str) -> float:
    parts = [float(p) for p in t.strip().split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def main() -> int:
    ap = argparse.ArgumentParser(description="Download a match video for tagging.")
    ap.add_argument("url")
    ap.add_argument("--height", type=int, default=720, help="max height: 480, 720 (default), 1080")
    ap.add_argument("--start", help="start time mm:ss or hh:mm:ss (download only a part)")
    ap.add_argument("--end", help="end time mm:ss or hh:mm:ss")
    ap.add_argument("--name", help="output file name without extension (default: video title)")
    ap.add_argument("--list", action="store_true", help="list available qualities and exit")
    a = ap.parse_args()

    try:
        import yt_dlp
    except ImportError:
        print("yt-dlp is not installed. Run:  pip install -U yt-dlp")
        return 1
    if shutil.which("ffmpeg") is None and not a.list:
        print("ffmpeg was not found on PATH. Install it (winget install Gyan.FFmpeg),"
              " open a new PowerShell window, and try again.")
        return 1

    if a.list:
        with yt_dlp.YoutubeDL({"listformats": True}) as ydl:
            ydl.download([a.url])
        return 0

    h = a.height
    fmt = (f"bv*[height<={h}][vcodec^=avc1]+ba[ext=m4a]/"   # H.264 + AAC: plays everywhere
           f"bv*[height<={h}]+ba/"                            # any codec, merged
           f"b[height<={h}]")                                 # last resort: combined stream
    DATA.mkdir(parents=True, exist_ok=True)
    stem = a.name or "%(title).80B"
    opts = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": str(DATA / f"{stem}.%(ext)s"),
        "restrictfilenames": a.name is None,
        "noplaylist": True,
    }
    if a.start or a.end:
        from yt_dlp.utils import download_range_func
        start = to_seconds(a.start) if a.start else 0
        end = to_seconds(a.end) if a.end else float("inf")
        opts["download_ranges"] = download_range_func(None, [(start, end)])
        opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(a.url, download=True)
        path = ydl.prepare_filename(info)
    out = Path(path).with_suffix(".mp4")
    print(f"\nSaved: {out}")
    print(f"Resolution: {info.get('width')}x{info.get('height')}  video codec: {info.get('vcodec')}")
    if a.start:
        print(f"This clip starts at {a.start} of the video. Enter that as 'Match clock at"
              " video 0:00' in the Tagger if the video starts at kick-off.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
