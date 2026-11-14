"""Crop the real broadcast picture out of a screen recording.

The test clip is 2400x1080: a 1920x1080 broadcast picture with the phone's
black bars / UI on the sides. The picture starts at x=240 (measured from the
clip). Cropping first makes the CV models see a normal 16:9 broadcast frame.

Usage:
    python src/crop_broadcast.py data/1000023455.mp4 data/btl_grs.mp4
    python src/crop_broadcast.py IN OUT --x 240 --w 1920

If your own recording has different bars, run ffmpeg's cropdetect to measure:
    ffmpeg -i IN -vf cropdetect=24:2:0 -t 20 -f null -
"""
from __future__ import annotations
import argparse
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inp")
    p.add_argument("out")
    p.add_argument("--x", type=int, default=240, help="left edge of the picture")
    p.add_argument("--w", type=int, default=1920, help="picture width")
    p.add_argument("--h", type=int, default=1080, help="picture height")
    p.add_argument("--y", type=int, default=0, help="top edge of the picture")
    a = p.parse_args()

    cmd = [
        "ffmpeg", "-y", "-i", a.inp,
        "-vf", f"crop={a.w}:{a.h}:{a.x}:{a.y}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "copy", a.out,
    ]
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
