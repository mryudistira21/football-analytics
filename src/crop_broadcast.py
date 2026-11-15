"""Find and crop the real broadcast picture out of a video.

Screen recordings (like the first test clip, 2400x1080 with phone bars and
on-screen buttons) need this; a normal YouTube download usually doesn't. The
crop is MEASURED from the video (the area that moves over time), never assumed,
and the script tells you when there is nothing to crop.

Cropping means re-encoding, which is CPU-heavy. This script keeps it gentle:
"veryfast" preset and a thread limit (default 4) so your PC stays usable.

Usage:
    python src/crop_broadcast.py IN.mp4                      # detect + crop -> IN_crop.mp4
    python src/crop_broadcast.py IN.mp4 OUT.mp4 --threads 2
    python src/crop_broadcast.py IN.mp4 --detect-only        # just print the crop
    python src/crop_broadcast.py IN.mp4 --crop 1920:1080:240:0
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def probe(path: str | Path) -> dict:
    """Width, height, fps and duration of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    st = d["streams"][0]

    def rate(v):
        num, den = (v or "0/1").split("/")
        return float(num) / float(den) if float(den or 0) else 0.0
    fps = rate(st.get("avg_frame_rate"))          # real average; screen recordings report odd r_frame_rate
    if not 1 <= fps <= 240:
        fps = rate(st.get("r_frame_rate"))
    return {"width": int(st["width"]), "height": int(st["height"]), "fps": fps,
            "duration": float(d.get("format", {}).get("duration") or 0)}


def _gray_frames(path, times, width=480):
    """Grab small grayscale frames at the given times (a few MB of RAM at most)."""
    import numpy as np
    info = probe(path)
    h = int(round(info["height"] * width / info["width"] / 2)) * 2
    frames = []
    for t in times:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
             "-vf", f"scale={width}:{h},format=gray", "-f", "rawvideo", "-"],
            capture_output=True).stdout
        if len(raw) == width * h:
            frames.append(np.frombuffer(raw, np.uint8).reshape(h, width).astype(np.float32))
    return frames, info, width, h


def _span(profile, frac=0.25):
    """First/last index where the profile rises above frac of its high level."""
    import numpy as np
    hi = np.percentile(profile, 90)
    idx = np.where(profile > hi * frac)[0]
    return (int(idx[0]), int(idx[-1])) if len(idx) else (0, len(profile) - 1)


def detect_crop(path: str | Path, samples: int = 12) -> str | None:
    """Return 'w:h:x:y' for the live picture area, or None if nothing to crop.

    The broadcast picture changes over time; black bars and a phone's static
    on-screen buttons don't. So it samples frames across the video and keeps
    the rows/columns that move. (ffmpeg's cropdetect only finds pure black
    bars and is fooled by on-screen UI.)
    """
    import numpy as np
    info = probe(path)
    dur = info["duration"] or 60
    times = [dur * (k + 0.5) / samples for k in range(samples)]
    frames, info, w, h = _gray_frames(path, times)
    if len(frames) < 3:
        return None
    stack = np.stack(frames)
    # How OFTEN each pixel changes between samples (not how much): the live picture
    # changes almost every time; bars and UI that pop up once (player controls) don't.
    changed = (np.abs(np.diff(stack, axis=0)) > 12).mean(axis=0)
    x0, x1 = _span(changed.mean(axis=0), 0.5)
    y0, y1 = _span(changed.mean(axis=1), 0.5)
    sx, sy = info["width"] / w, info["height"] / h
    X0, X1 = int(x0 * sx), int(min(info["width"], (x1 + 1) * sx))
    Y0, Y1 = int(y0 * sy), int(min(info["height"], (y1 + 1) * sy))
    cw, ch = X1 - X0, Y1 - Y0
    # Snap to 16:9 when it's close (broadcasts are 16:9) and to even numbers for H.264.
    if abs(cw / ch - 16 / 9) < 0.06:
        cw2 = int(round(ch * 16 / 9))
        X0 = max(0, min(info["width"] - cw2, X0 + (cw - cw2) // 2))
        cw = cw2
    cw, ch, X0, Y0 = cw // 2 * 2, ch // 2 * 2, X0 // 2 * 2, Y0 // 2 * 2
    if info["width"] - cw < 16 and info["height"] - ch < 16:
        return None
    return f"{cw}:{ch}:{X0}:{Y0}"


def crop_video(src: str | Path, dst: str | Path, crop: str, threads: int = 4,
               fps: float | None = None) -> None:
    vf = f"crop={crop}"
    if fps:
        vf += f",fps={fps}"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats",
           "-i", str(src), "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", str(threads),
           "-c:a", "copy", "-movflags", "+faststart", str(dst)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect and crop the broadcast picture.")
    ap.add_argument("inp")
    ap.add_argument("out", nargs="?", help="output file (default: IN_crop.mp4)")
    ap.add_argument("--crop", help="force a crop w:h:x:y instead of detecting it")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for encoding (default 4)")
    ap.add_argument("--detect-only", action="store_true")
    a = ap.parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ffmpeg/ffprobe not found on PATH.")
        return 1

    info = probe(a.inp)
    print(f"input {info['width']}x{info['height']} @ {info['fps']:.2f} fps, {info['duration']/60:.1f} min")
    crop = a.crop or detect_crop(a.inp)
    if not crop:
        print("No black bars found: nothing to crop. Use the file as it is.")
        return 0
    print(f"picture area: {crop}")
    if a.detect_only:
        return 0
    out = Path(a.out) if a.out else Path(a.inp).with_name(Path(a.inp).stem + "_crop.mp4")
    crop_video(a.inp, out, crop, a.threads)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
