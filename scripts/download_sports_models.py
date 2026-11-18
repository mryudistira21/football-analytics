from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def default_video_dir() -> Path:
    return Path(os.environ.get("FOOTBALL_VIDEO_DIR") or REPO_ROOT.parent / "football-video")


def default_model_dir() -> Path:
    return Path(os.environ.get("FOOTBALL_MODEL_DIR") or default_video_dir() / "models")


MODELS = {
    "football-player-detection.pt": "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q",
    "football-pitch-detection.pt": "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf",
    "football-ball-detection.pt": "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V",
}


def download_models(model_dir: Path, force: bool = False) -> int:
    try:
        import gdown
    except ImportError:
        print("Missing dependency. Activate the project venv and run: pip install gdown")
        return 1

    model_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for filename, file_id in MODELS.items():
        destination = model_dir / filename
        if destination.exists() and destination.stat().st_size > 1_000_000 and not force:
            print(f"[present] {destination} ({destination.stat().st_size / 1024**2:.1f} MB)")
            continue

        temporary = destination.with_suffix(destination.suffix + ".download")
        temporary.unlink(missing_ok=True)
        print(f"[download] {filename}")
        try:
            result = gdown.download(
                id=file_id,
                output=str(temporary),
                quiet=False,
            )
            if not result or not temporary.exists() or temporary.stat().st_size < 1_000_000:
                raise RuntimeError("Google Drive did not return a valid model file.")
            temporary.replace(destination)
            print(f"[saved] {destination} ({destination.stat().st_size / 1024**2:.1f} MB)")
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            failed.append((filename, str(exc)))
            print(f"[failed] {filename}: {exc}", file=sys.stderr)

    if failed:
        print("\nSome models could not be downloaded:")
        for filename, reason in failed:
            print(f" - {filename}: {reason}")
        return 1

    print(f"\nAll three Roboflow Sports models are ready in: {model_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Roboflow Sports soccer model weights outside the Git repository."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_model_dir(),
        help=f"model directory (default: {default_model_dir()})",
    )
    parser.add_argument("--force", action="store_true", help="download again even if a model exists")
    args = parser.parse_args()
    return download_models(args.out, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
