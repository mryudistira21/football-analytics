from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import math
import os
import joblib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "match_id", "frame", "video_seconds", "period", "track_id",
    "role", "team", "x", "y", "confidence", "camera_state",
]
MODEL_FILES = {
    "player": "football-player-detection.pt",
    "pitch": "football-pitch-detection.pt",
    "ball": "football-ball-detection.pt",
}


def video_dir() -> Path:
    return Path(os.environ.get("FOOTBALL_VIDEO_DIR") or REPO_ROOT.parent / "football-video")


def model_dir() -> Path:
    return Path(os.environ.get("FOOTBALL_MODEL_DIR") or video_dir() / "models")


# Roboflow Sports SoccerPitchConfiguration keypoint order (centimetres).
# Keep this reference in sync with the upstream 32-vertex configuration. The
# runtime check below makes an upstream geometry/order change fail loudly.
_ROBOFLOW_PITCH_VERTICES_CM = np.asarray([
    [0, 0], [0, 1450], [0, 2584], [0, 4416], [0, 5550], [0, 7000],
    [550, 2584], [550, 4416], [1100, 3500],
    [2015, 1450], [2015, 2584], [2015, 4416], [2015, 5550],
    [6000, 0], [6000, 2585], [6000, 4415], [6000, 7000],
    [9985, 1450], [9985, 2584], [9985, 4416], [9985, 5550],
    [10900, 3500], [11450, 2584], [11450, 4416],
    [12000, 0], [12000, 1450], [12000, 2584], [12000, 4416],
    [12000, 5550], [12000, 7000], [5085, 3500], [6915, 3500],
], dtype=np.float32)

# The matching 32 landmarks in Pitchside Tagger's 105 x 68 metre geometry.
# These are true pitch dimensions/markings, not an independent axis rescale.
_TAGGER_PITCH_VERTICES_M = np.asarray([
    [0, 0], [0, 13.84], [0, 24.84], [0, 43.16], [0, 54.16], [0, 68],
    [5.5, 24.84], [5.5, 43.16], [11, 34],
    [16.5, 13.84], [16.5, 24.84], [16.5, 43.16], [16.5, 54.16],
    [52.5, 0], [52.5, 24.85], [52.5, 43.15], [52.5, 68],
    [88.5, 13.84], [88.5, 24.84], [88.5, 43.16], [88.5, 54.16],
    [94, 34], [99.5, 24.84], [99.5, 43.16],
    [105, 0], [105, 13.84], [105, 24.84], [105, 43.16],
    [105, 54.16], [105, 68], [43.35, 34], [61.65, 34],
], dtype=np.float32)


def pitch_targets(vertices: Any) -> np.ndarray:
    """Map Roboflow's ordered 32 pitch landmarks to the Tagger's 105 x 68 m pitch."""
    v = np.asarray(vertices, dtype=np.float32)
    if v.shape != _ROBOFLOW_PITCH_VERTICES_CM.shape:
        raise ValueError(
            "Expected the 32 ordered Roboflow soccer pitch keypoints; "
            f"received shape {v.shape}"
        )
    if not np.allclose(v, _ROBOFLOW_PITCH_VERTICES_CM, atol=1e-3, rtol=0):
        raise ValueError(
            "Unexpected Roboflow pitch landmark geometry/order. "
            "Update the validated source and Tagger landmark arrays together."
        )
    return _TAGGER_PITCH_VERTICES_M.copy()


def fit_homography(
    pixel_xy: Any,
    pitch_xy: Any,
    confidence: Any = None,
    min_points: int = 6,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    src = np.asarray(pixel_xy, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(pitch_xy, dtype=np.float32).reshape(-1, 2)
    if src.shape != dst.shape:
        return None, {"valid_points": 0, "inliers": 0, "median_error_m": None}

    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    valid &= (src[:, 0] > 1) & (src[:, 1] > 1)
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        if len(conf) == len(valid):
            valid &= np.isfinite(conf) & (conf >= 0.20)

    src, dst = src[valid], dst[valid]
    if len(src) < min_points:
        return None, {
            "valid_points": int(len(src)), "inliers": 0, "median_error_m": None
        }

    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return None, {
            "valid_points": int(len(src)), "inliers": 0, "median_error_m": None
        }

    inliers = mask.ravel().astype(bool)
    if int(inliers.sum()) < min_points:
        return None, {
            "valid_points": int(len(src)),
            "inliers": int(inliers.sum()),
            "median_error_m": None,
        }

    predicted = cv2.perspectiveTransform(src[inliers].reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(predicted - dst[inliers], axis=1)
    median = float(np.median(errors))
    p90 = float(np.percentile(errors, 90))
    # Avoid accepting numerically plausible but geometrically unusable mappings.
    if not math.isfinite(median) or median > 4.5 or p90 > 11.25:
        return None, {
            "valid_points": int(len(src)),
            "inliers": int(inliers.sum()),
            "median_error_m": round(median, 4),
        }
    return matrix.astype(np.float32), {
        "valid_points": int(len(src)),
        "inliers": int(inliers.sum()),
        "median_error_m": round(median, 4),
    }


def map_points(pixel_xy: Any, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(pixel_xy, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    valid = np.isfinite(mapped).all(axis=1)
    valid &= (mapped[:, 0] >= -2) & (mapped[:, 0] <= 107)
    valid &= (mapped[:, 1] >= -2) & (mapped[:, 1] <= 70)
    mapped[~valid] = np.nan
    mapped[valid, 0] = np.clip(mapped[valid, 0], 0, 105)
    mapped[valid, 1] = np.clip(mapped[valid, 1], 0, 68)
    return mapped.astype(np.float32)


def classify_segment_status(
    processed_frames: int,
    requested_end_seconds: float | None,
    reached_requested_end: bool,
    last_decoded_seconds: float | None,
    end_tolerance_seconds: float,
) -> str:
    """Classify normal EOF against the requested source-video interval."""
    if processed_frames <= 0:
        return "no_frames"
    if requested_end_seconds is None or reached_requested_end:
        return "complete"
    if (
        last_decoded_seconds is not None
        and last_decoded_seconds >= requested_end_seconds - end_tolerance_seconds
    ):
        return "complete"
    return "partial_eof"


def class_ids_from_names(names: Any) -> dict[str, int]:
    pairs = (
        [(int(k), str(v).lower()) for k, v in names.items()]
        if isinstance(names, dict)
        else [(i, str(v).lower()) for i, v in enumerate(names)]
    )
    aliases = {
        "ball": ("ball", "soccer ball"),
        "goalkeeper": ("goalkeeper", "goal keeper"),
        "player": ("player",),
        "referee": ("referee", "official"),
    }
    result: dict[str, int] = {}
    for role, terms in aliases.items():
        for idx, name in pairs:
            if any(term in name for term in terms):
                result[role] = idx
                break
    # Official Roboflow Sports soccer model class order.
    for role, fallback in {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}.items():
        result.setdefault(role, fallback)
    return result


def extract_crop(frame: np.ndarray, box: Any) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=int)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 16 or y2 - y1 < 32:
        return None
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (128, 256), interpolation=cv2.INTER_AREA) if crop.size else None


def sample_team_crops(
    video: Path,
    model: Any,
    device: str,
    class_id: int,
    seconds: float,
    interval: float,
    limit: int,
    imgsz: int,
    start_seconds: float = 0.0,
) -> list[np.ndarray]:
    """Bounded first pass: keep <= limit crops, never keep frames or full video in RAM."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * interval)))
    if start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)
    crops: list[np.ndarray] = []
    frame_no = max(0, int(round(start_seconds * fps)))
    try:
        while len(crops) < limit:
            ok, frame = cap.read()
            if not ok or frame_no / fps >= start_seconds + seconds:
                break
            if frame_no % step == 0:
                result = model.predict(frame, imgsz=imgsz, conf=0.25, device=device,
                                       verbose=False)[0]
                detections = __import__("supervision").Detections.from_ultralytics(result)
                if detections.class_id is not None:
                    boxes = detections[detections.class_id == class_id].xyxy
                    if len(boxes) > 8:
                        boxes = boxes[np.linspace(0, len(boxes) - 1, 8, dtype=int)]
                    for box in boxes:
                        crop = extract_crop(frame, box)
                        if crop is not None:
                            crops.append(crop)
                            if len(crops) >= limit:
                                break
            frame_no += 1
    finally:
        cap.release()
        gc.collect()
    return crops




def choose_team_device(
    torch_module: Any,
    requested: str = "auto",
    min_free_vram_gb: float = 4.0,
) -> tuple[str, dict[str, float] | None]:
    """Choose SigLIP's device using free VRAM after both YOLO detectors are loaded."""
    if requested == "cpu":
        return "cpu", None
    if not torch_module.cuda.is_available():
        if requested == "cuda:0":
            raise RuntimeError("--team-device cuda:0 requested, but CUDA is unavailable")
        return "cpu", None

    # Release unused cached inference buffers before measuring true free VRAM.
    empty_cache = getattr(torch_module.cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(0)
    memory = {
        "free_gb": round(float(free_bytes) / (1024 ** 3), 3),
        "total_gb": round(float(total_bytes) / (1024 ** 3), 3),
    }
    if requested == "cuda:0" or memory["free_gb"] >= min_free_vram_gb:
        return "cuda:0", memory
    return "cpu", memory


def should_refresh_team_embedding(
    vote_counts: Counter[str],
    last_refresh_frame: int | None,
    current_frame: int,
    refresh_after_frames: int,
    stability_ratio: float = 0.75,
) -> bool:
    """Refresh a track's SigLIP crop only when new, awaiting validation, or unstable."""
    if last_refresh_frame is None:
        return True
    if current_frame - last_refresh_frame < refresh_after_frames:
        return False
    total_votes = sum(vote_counts.values())
    if total_votes <= 1:
        return True
    return max(vote_counts.values()) / total_votes < stability_ratio


def team_clusters_from_embeddings(team_classifier: Any, embeddings: np.ndarray) -> np.ndarray:
    """Apply fitted UMAP + KMeans stages to already-extracted SigLIP features."""
    features = np.asarray(embeddings, dtype=np.float32)
    if features.ndim != 2 or len(features) == 0:
        return np.empty((0,), dtype=np.int64)
    projections = team_classifier.reducer.transform(features)
    return np.asarray(team_classifier.cluster_model.predict(projections), dtype=np.int64)


def load_team_cluster_state(path: Path) -> tuple[Any, Any]:
    """Load fitted UMAP/KMeans state; home/away mapping is applied at prediction time."""
    state = joblib.load(path)
    if state.get("state_version") != 1:
        raise ValueError(f"Unsupported team classifier state version in {path}")
    return state["reducer"], state["cluster_model"]


def save_team_cluster_state(path: Path, team_classifier: Any, cluster_0_side: str) -> None:
    """Persist fitted UMAP/KMeans state atomically for reuse by later segments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    joblib.dump({
        "state_version": 1,
        "cluster_0_side": cluster_0_side,
        "reducer": team_classifier.reducer,
        "cluster_model": team_classifier.cluster_model,
    }, temporary)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Track football players and ball on broadcast video.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", type=Path, default=model_dir())
    parser.add_argument("--cache", type=Path, default=video_dir() / "cache" / "huggingface")
    parser.add_argument("--device", choices=["auto", "cuda:0", "cpu"], default="auto")
    parser.add_argument("--period", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--home-dir-1h", choices=["R", "L"], required=True)
    parser.add_argument(
        "--period-direction", action="append", default=[], metavar="PERIOD:R|L",
        help="explicit period direction override; required for periods 3 and 4",
    )
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--pitch-imgsz", type=int, default=960)
    parser.add_argument("--start-seconds", type=float, default=0.0,
                        help="source-video start time; use one run per match period")
    parser.add_argument("--end-seconds", type=float,
                        help="source-video end time (exclusive)")
    parser.add_argument("--limit-seconds", type=float,
                        help="maximum processing duration from --start-seconds (smoke test)")
    parser.add_argument("--no-team-classification", action="store_true")
    parser.add_argument("--team-sample-seconds", type=float, default=240)
    parser.add_argument("--team-sample-interval", type=float, default=1.5)
    parser.add_argument("--team-sample-max-crops", type=int, default=300)
    parser.add_argument("--team-classify-every", type=int, default=30)
    parser.add_argument("--team-device", choices=["auto", "cuda:0", "cpu"], default="auto")
    parser.add_argument("--team-min-free-vram-gb", type=float, default=4.0)
    parser.add_argument("--team-reclassify-after", type=int, default=150)
    parser.add_argument("--team-stability-ratio", type=float, default=0.75)
    parser.add_argument("--team-model-state", type=Path,
                        help="persist/reuse fitted UMAP + KMeans state across video segments")
    parser.add_argument("--track-id-offset", type=int, default=0,
                        help="add this offset to person track IDs; ball ID 0 remains reserved")
    parser.add_argument("--cluster-0-side", choices=["home", "away"], default="home")
    parser.add_argument("--annotated-output", type=Path)
    parser.add_argument("--annotated-max-seconds", type=float, default=180)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.frame_stride < 1 or args.team_classify_every < 1:
        parser.error("frame stride and team classification interval must be positive")
    if args.team_reclassify_after < 1 or args.team_min_free_vram_gb < 0:
        parser.error("team reclassification interval must be positive and VRAM threshold non-negative")
    if not 0.5 <= args.team_stability_ratio <= 1.0:
        parser.error("--team-stability-ratio must be between 0.5 and 1.0")
    if args.track_id_offset < 0:
        parser.error("--track-id-offset must be non-negative")
    directions = {
        "1": args.home_dir_1h == "R",
        "2": args.home_dir_1h != "R",
    }
    for item in args.period_direction:
        try:
            period_text, side = item.split(":", 1)
            period = int(period_text)
        except (ValueError, TypeError):
            parser.error(f"Invalid --period-direction {item!r}; use PERIOD:R or PERIOD:L")
        side = side.upper()
        if period not in (1, 2, 3, 4) or side not in ("R", "L"):
            parser.error(f"Invalid --period-direction {item!r}; use period 1..4 and R/L")
        directions[str(period)] = side == "R"
    if str(args.period) not in directions:
        parser.error(f"Period {args.period} needs --period-direction {args.period}:R or {args.period}:L.")
    if args.start_seconds < 0:
        parser.error("--start-seconds must be >= 0")
    if args.end_seconds is not None and args.end_seconds <= args.start_seconds:
        parser.error("--end-seconds must be greater than --start-seconds")
    if args.limit_seconds is not None and args.limit_seconds <= 0:
        parser.error("--limit-seconds must be positive")

    video = args.input.expanduser().resolve()
    if not video.is_file():
        parser.error(f"Video not found: {video}")
    models = args.models.expanduser().resolve()
    player_weights, pitch_weights = models / MODEL_FILES["player"], models / MODEL_FILES["pitch"]
    for path in (player_weights, pitch_weights):
        if not path.is_file():
            parser.error(f"Missing {path}. Run python scripts/download_sports_models.py")

    args.cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.cache.resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((args.cache / "hub").resolve()))

    try:
        import torch
        import supervision as sv
        from trackers import ByteTrackTracker
        from ultralytics import YOLO
        from sports.common.team import TeamClassifier
        from sports.configs.soccer import SoccerPitchConfiguration
    except ImportError as exc:
        print(f"Track B dependencies missing: {exc}", file=sys.stderr)
        print("Install requirements.txt and roboflow/sports into the Python 3.10 venv.")
        return 1

    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA selected but torch.cuda.is_available() is false.", file=sys.stderr)
        return 1
    half = device.startswith("cuda")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"Cannot open video: {video}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output = (args.output or (
        video_dir() / "tracking" / f"{video.stem}_p{args.period}_tracks.csv"
    )).expanduser().resolve()
    meta_path = output.with_suffix(".meta.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and (output.exists() or meta_path.exists()):
        print("Output exists; pass --overwrite or choose another --output.", file=sys.stderr)
        return 1

    stop_at = args.end_seconds
    if args.limit_seconds is not None:
        duration_limit = args.start_seconds + args.limit_seconds
        stop_at = min(stop_at, duration_limit) if stop_at is not None else duration_limit

    team_sample_seconds = args.team_sample_seconds
    if stop_at is not None:
        team_sample_seconds = min(team_sample_seconds, max(0.0, stop_at - args.start_seconds))

    period_dirs = directions
    metadata: dict[str, Any] = {
        "match_id": args.match_id, "video_file": str(video),
        "fps": float(fps), "width": width, "height": height,
        "home_dir_1h": args.home_dir_1h,
        "home_attacks_right_by_period": period_dirs,
        "frame_stride": args.frame_stride,
        "imgsz": args.imgsz,
        "pitch_imgsz": args.pitch_imgsz,
        "period": args.period,
        "track_id_offset": args.track_id_offset,
        "team_device_requested": args.team_device,
        "team_min_free_vram_gb": args.team_min_free_vram_gb,
        "team_classify_every": args.team_classify_every,
        "team_reclassify_after": args.team_reclassify_after,
        "team_stability_ratio": args.team_stability_ratio,
        "team_sample_seconds_requested": args.team_sample_seconds,
        "team_sample_interval": args.team_sample_interval,
        "team_sample_max_crops": args.team_sample_max_crops,
        "source_start_seconds": args.start_seconds,
        "source_end_seconds": args.end_seconds,
        "model_versions": {
            "ultralytics": importlib.metadata.version("ultralytics"),
            "supervision": importlib.metadata.version("supervision"),
            "trackers": importlib.metadata.version("trackers"),
            "sports": "roboflow/sports",
        },
        "team_cluster_0_side": args.cluster_0_side,
        "pitch_geometry": "pitchside_tagger_105x68_v1",
        "camera_state_method": "wide when pitch-keypoint homography passes validation; otherwise closeup (v1 fallback state)",
        "complete": False,
    }
    temporary_csv = output.with_suffix(output.suffix + ".partial")
    write_meta = lambda: meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_meta()

    try:
        team_classifier = None
        votes: dict[int, Counter[str]] = defaultdict(Counter)
        team_embeddings: dict[int, np.ndarray] = {}
        last_embedding_refresh: dict[int, int] = {}
        team_state_loaded = False
        cluster_to_side = {
            0: args.cluster_0_side,
            1: "away" if args.cluster_0_side == "home" else "home",
        }

        # Keep one copy of each YOLO detector resident. Reuse player_model during
        # team-crop sampling; do not instantiate a third detector copy.
        player_model = YOLO(str(player_weights)).to(device)
        class_ids = class_ids_from_names(player_model.names)
        pitch_model = YOLO(str(pitch_weights)).to(device)

        if not args.no_team_classification:
            requested_state = args.team_model_state.expanduser().resolve() if args.team_model_state else None

            if requested_state is not None and requested_state.is_file():
                # State reuse does not need another player crop pass. Check free VRAM
                # now, after both YOLO detectors are resident, before loading SigLIP.
                team_device, team_vram = choose_team_device(
                    torch, args.team_device, args.team_min_free_vram_gb
                )
                team_classifier = TeamClassifier(device=team_device, batch_size=16)
                reducer, cluster_model = load_team_cluster_state(requested_state)
                team_classifier.reducer = reducer
                team_classifier.cluster_model = cluster_model
                team_state_loaded = True
                print(f"Loaded team cluster state: {requested_state}")
            else:
                # Sample with the resident player detector before loading SigLIP, so
                # the feature encoder does not overlap this GPU workload.
                print("Sampling player crops for team classification...")
                crops = sample_team_crops(
                    video, player_model, device, class_ids["player"],
                    team_sample_seconds, args.team_sample_interval,
                    args.team_sample_max_crops, args.imgsz,
                    start_seconds=args.start_seconds,
                )
                if len(crops) >= 30:
                    # Choose the classifier device after crop inference has completed,
                    # using remaining free VRAM rather than a pre-sampling estimate.
                    team_device, team_vram = choose_team_device(
                        torch, args.team_device, args.team_min_free_vram_gb
                    )
                    team_classifier = TeamClassifier(device=team_device, batch_size=16)
                    team_classifier.fit(crops)
                    print(
                        f"Team classifier fitted on {len(crops)} crops ({team_device})."
                    )
                    if requested_state is not None:
                        save_team_cluster_state(
                            requested_state, team_classifier, args.cluster_0_side
                        )
                        print(f"Saved team cluster state: {requested_state}")
                else:
                    team_device, team_vram = "not_used", None
                    print(f"Only {len(crops)} usable crops; teams will remain unknown.")
                del crops
                gc.collect()

            metadata["team_device"] = team_device
            metadata["team_vram_at_selection"] = team_vram
            if team_device != "not_used":
                print(
                    f"Team classifier device: {team_device}"
                    + (
                        f" (free VRAM {team_vram['free_gb']:.2f}/{team_vram['total_gb']:.2f} GB)"
                        if team_vram else ""
                    )
                )
            metadata["team_classifier_state_path"] = (
                str(requested_state) if requested_state is not None else None
            )
            metadata["team_classifier_state_loaded"] = team_state_loaded
            metadata["team_embedding_cache"] = "per-track SigLIP feature vectors"
            write_meta()

        pitch_vertices = pitch_targets(SoccerPitchConfiguration().vertices)
        tracker = ByteTrackTracker(
            track_activation_threshold=0.25,
            minimum_consecutive_frames=3,
            frame_rate=float(fps),
        )
        ball_tracker = __import__("sports.common.ball", fromlist=["BallTracker"]).BallTracker(buffer_size=12)

        writer = cv2.VideoWriter()
        if args.annotated_output:
            preview = args.annotated_output.expanduser().resolve()
            preview.parent.mkdir(parents=True, exist_ok=True)
            writer.open(str(preview), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.frame_stride, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create annotated video: {preview}")
            metadata["annotated_video"] = str(preview)
            metadata["annotated_max_seconds"] = args.annotated_max_seconds
            write_meta()

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot reopen video: {video}")
        if args.start_seconds > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000.0)

        metadata["effective_end_seconds"] = stop_at
        write_meta()

        processed = 0
        last_timestamp = -1.0
        last_decoded_timestamp: float | None = None
        first_processed_timestamp: float | None = None
        last_processed_timestamp: float | None = None
        reached_requested_end = False
        rows_written = 0
        camera_counts: Counter[str] = Counter()
        errors: list[float] = []

        with temporary_csv.open("w", newline="", encoding="utf-8") as fp:
            out = csv.DictWriter(fp, fieldnames=FIELDS)
            out.writeheader()
            decoded_count = max(0, int(round(args.start_seconds * fps)))
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # POS_FRAMES reports the next frame after read.
                    raw_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    frame_index = max(0, int(raw_frame_pos) - 1) if raw_frame_pos > 0 else decoded_count
                    decoded_count += 1

                    # Prefer the decoder's presentation timestamp (PTS).
                    raw_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    timestamp = raw_msec / 1000.0
                    if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= last_timestamp:
                        timestamp = frame_index / fps
                    last_timestamp = timestamp
                    last_decoded_timestamp = timestamp
                    if timestamp < args.start_seconds:
                        continue
                    if stop_at is not None and timestamp >= stop_at:
                        reached_requested_end = True
                        break
                    if frame_index % args.frame_stride:
                        continue
                    processed += 1
                    if first_processed_timestamp is None:
                        first_processed_timestamp = timestamp
                    last_processed_timestamp = timestamp

                    result = player_model.predict(
                        frame, imgsz=args.imgsz, conf=0.08,
                        device=device, quantize="fp16" if half else None, verbose=False,
                    )[0]
                    detections = sv.Detections.from_ultralytics(result)
                    if detections.class_id is None:
                        detections = sv.Detections(
                            xyxy=np.empty((0, 4), dtype=np.float32),
                            confidence=np.empty((0,), dtype=np.float32),
                            class_id=np.empty((0,), dtype=int),
                        )
                    else:
                        keep_ids = np.array(list(class_ids.values()), dtype=int)
                        confidence = (
                            detections.confidence
                            if detections.confidence is not None
                            else np.ones(len(detections), dtype=np.float32)
                        )
                        thresholds = np.where(
                            detections.class_id == class_ids["ball"], 0.08, 0.25
                        )
                        keep = np.isin(detections.class_id, keep_ids) & (confidence >= thresholds)
                        detections = detections[keep]

                    people = detections[np.isin(
                        detections.class_id,
                        [class_ids["player"], class_ids["goalkeeper"], class_ids["referee"]],
                    )]
                    tracked = tracker.update(people, timestamp=timestamp)
                    # ByteTrackTracker uses -1 for tentative/unmatched detections.
                    # Do not export those as if they were persistent player identities.
                    if tracked.tracker_id is not None and len(tracked):
                        tracked = tracked[tracked.tracker_id >= 0]

                    pitch_result = pitch_model.predict(
                        frame, imgsz=args.pitch_imgsz, conf=0.10,
                        device=device, quantize="fp16" if half else None, verbose=False,
                    )[0]
                    matrix = None
                    stats = {"median_error_m": None, "inliers": 0}
                    keypoints = getattr(pitch_result, "keypoints", None)
                    if keypoints is not None and keypoints.xy is not None and len(keypoints.xy):
                        xy = keypoints.xy[0].detach().cpu().numpy()
                        conf = (
                            keypoints.conf[0].detach().cpu().numpy()
                            if keypoints.conf is not None else None
                        )
                        matrix, stats = fit_homography(xy, pitch_vertices, conf)
                    camera_state = "wide" if matrix is not None else "closeup"
                    camera_counts[camera_state] += 1
                    if stats["median_error_m"] is not None:
                        errors.append(float(stats["median_error_m"]))

                    # Extract SigLIP features only for new tracks, delayed validation,
                    # or tracks whose accumulated team votes remain unstable.
                    if (
                        team_classifier is not None
                        and processed % args.team_classify_every == 0
                        and tracked.tracker_id is not None
                        and tracked.class_id is not None
                    ):
                        players = tracked[tracked.class_id == class_ids["player"]]
                        crops_now, tids = [], []
                        for box, tid in zip(players.xyxy, players.tracker_id):
                            if tid is None:
                                continue
                            tid_int = int(tid)
                            if not should_refresh_team_embedding(
                                votes[tid_int],
                                last_embedding_refresh.get(tid_int),
                                processed,
                                args.team_reclassify_after,
                                args.team_stability_ratio,
                            ):
                                continue
                            crop = extract_crop(frame, box)
                            if crop is not None:
                                crops_now.append(crop)
                                tids.append(tid_int)
                        if crops_now:
                            features = np.asarray(
                                team_classifier.extract_features(crops_now),
                                dtype=np.float32,
                            )
                            clusters = team_clusters_from_embeddings(
                                team_classifier, features
                            )
                            for tid, feature, cluster in zip(tids, features, clusters):
                                team_embeddings[tid] = feature.copy()
                                last_embedding_refresh[tid] = processed
                                votes[tid][cluster_to_side[int(cluster)]] += 1

                    positions = (
                        map_points(tracked.get_anchors_coordinates(
                            anchor=sv.Position.BOTTOM_CENTER
                        ), matrix)
                        if matrix is not None and len(tracked)
                        else np.full((len(tracked), 2), np.nan, dtype=np.float32)
                    )

                    rows = []
                    if tracked.tracker_id is not None and tracked.class_id is not None:
                        for i, (tid, cid) in enumerate(zip(tracked.tracker_id, tracked.class_id)):
                            if tid is None:
                                continue
                            role = (
                                "goalkeeper" if int(cid) == class_ids["goalkeeper"]
                                else "referee" if int(cid) == class_ids["referee"]
                                else "player"
                            )
                            team = "unknown"
                            if role != "referee" and votes[int(tid)]:
                                ordered = votes[int(tid)].most_common()
                                if len(ordered) == 1 or ordered[0][1] > ordered[1][1]:
                                    team = ordered[0][0]
                            x, y = positions[i]
                            row = {
                                "match_id": args.match_id, "frame": frame_index,
                                "video_seconds": round(timestamp, 3), "period": args.period,
                                "track_id": int(tid) + args.track_id_offset, "role": role, "team": team,
                                "x": round(float(x), 2) if np.isfinite(x) else "",
                                "y": round(float(y), 2) if np.isfinite(y) else "",
                                "confidence": round(float(tracked.confidence[i]), 4)
                                if tracked.confidence is not None else 1.0,
                                "camera_state": camera_state,
                            }
                            rows.append(row)

                    ball = ball_tracker.update(
                        detections[detections.class_id == class_ids["ball"]]
                    )
                    if len(ball):
                        ball_pixel = ball.get_anchors_coordinates(anchor=sv.Position.CENTER)
                        ball_xy = map_points(ball_pixel, matrix)[0] if matrix is not None else [np.nan, np.nan]
                        rows.append({
                            "match_id": args.match_id, "frame": frame_index,
                            "video_seconds": round(timestamp, 3), "period": args.period,
                            "track_id": 0, "role": "ball", "team": "unknown",
                            "x": round(float(ball_xy[0]), 2) if np.isfinite(ball_xy[0]) else "",
                            "y": round(float(ball_xy[1]), 2) if np.isfinite(ball_xy[1]) else "",
                            "confidence": round(float(ball.confidence[0]), 4)
                            if ball.confidence is not None else 1.0,
                            "camera_state": camera_state,
                        })

                    out.writerows(rows)
                    rows_written += len(rows)
                    if processed % 25 == 0:
                        fp.flush()
                    if processed % 100 == 0:
                        print(f"frames {processed:,}; video {timestamp:.1f}s; rows {rows_written:,}; camera={camera_state}")

                    if (
                        writer.isOpened()
                        and timestamp >= args.start_seconds
                        and timestamp - args.start_seconds <= args.annotated_max_seconds
                    ):
                        annotated = frame.copy()
                        if tracked.tracker_id is not None and tracked.class_id is not None:
                            for box, tid, cid in zip(tracked.xyxy, tracked.tracker_id, tracked.class_id):
                                if tid is None:
                                    continue
                                role = (
                                    "GK" if int(cid) == class_ids["goalkeeper"]
                                    else "REF" if int(cid) == class_ids["referee"]
                                    else "P"
                                )
                                team = "unknown"
                                if role != "REF" and votes[int(tid)]:
                                    ordered = votes[int(tid)].most_common()
                                    if len(ordered) == 1 or ordered[0][1] > ordered[1][1]:
                                        team = ordered[0][0]
                                color = (
                                    (40, 210, 70) if team == "home"
                                    else (40, 145, 255) if team == "away"
                                    else (175, 175, 175)
                                )
                                if role == "REF":
                                    color = (190, 190, 190)
                                x1, y1, x2, y2 = np.asarray(box, dtype=int)
                                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                                label = f"{role} {int(tid) + args.track_id_offset} {team}"
                                cv2.putText(
                                    annotated, label, (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                                    cv2.LINE_AA,
                                )
                        if len(ball):
                            ball_center = ball.get_anchors_coordinates(
                                anchor=sv.Position.CENTER
                            )[0].astype(int)
                            cv2.circle(
                                annotated, tuple(ball_center), 7,
                                (40, 40, 240), 2, cv2.LINE_AA,
                            )
                            cv2.putText(
                                annotated, "BALL", (int(ball_center[0]) + 8, int(ball_center[1]) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 240), 1,
                                cv2.LINE_AA,
                            )
                        cv2.putText(annotated, f"CAMERA: {camera_state.upper()}", (12, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        writer.write(annotated)

            finally:
                cap.release()
                if writer.isOpened():
                    writer.release()

        # EOF before the requested stop is a partial segment, not a successful full segment.
        sample_interval_seconds = args.frame_stride / float(fps)
        end_tolerance_seconds = max(1.0 / float(fps), sample_interval_seconds * 1.5)
        completion_status = classify_segment_status(
            processed,
            stop_at,
            reached_requested_end,
            last_decoded_timestamp,
            end_tolerance_seconds,
        )

        # Atomic promotion keeps even a normal-EOF partial CSV available for diagnosis.
        temporary_csv.replace(output)
        metadata.update({
            "status": completion_status,
            "complete": completion_status == "complete",
            "requested_start_seconds": args.start_seconds,
            "requested_end_seconds": stop_at,
            "actual_start_seconds": (
                round(first_processed_timestamp, 3)
                if first_processed_timestamp is not None else None
            ),
            "actual_end_seconds": (
                round(last_processed_timestamp, 3)
                if last_processed_timestamp is not None else None
            ),
            "last_decoded_seconds": (
                round(last_decoded_timestamp, 3)
                if last_decoded_timestamp is not None else None
            ),
            "processed_coverage_seconds": (
                round(max(0.0, last_processed_timestamp - first_processed_timestamp), 3)
                if first_processed_timestamp is not None and last_processed_timestamp is not None
                else 0.0
            ),
            "end_tolerance_seconds": round(end_tolerance_seconds, 4),
            "termination_reason": (
                "requested_end_reached" if reached_requested_end
                else "end_of_file_no_bound" if stop_at is None and processed > 0
                else "end_of_file"
            ),
            "processed_frames": processed,
            "rows_written": rows_written,
            "camera_state_counts": dict(camera_counts),
            "calibration_median_error_m": round(float(np.median(errors)), 3) if errors else None,
            "track_count_with_team_votes": len(votes),
            "team_embedding_cache_tracks": len(team_embeddings),
            "track_id_offset": args.track_id_offset,
            "unknown_team_track_count": sum(
                1 for v in votes.values()
                if len(v) == 0 or (len(v.most_common()) > 1 and v.most_common()[0][1] == v.most_common()[1][1])
            ),
        })
        write_meta()
        print(f"\nTracking output written ({completion_status}): {output}")
        print(f"Metadata: {meta_path}")
        print(
            f"Processed frames: {processed:,}; rows: {rows_written:,}; "
            f"status: {completion_status}"
        )
        return 0 if completion_status == "complete" else 2

    except KeyboardInterrupt:
        metadata["interrupted"] = True
        metadata["status"] = "interrupted"
        metadata["complete"] = False
        write_meta()
        print("Interrupted: partial CSV retained with .partial suffix.", file=sys.stderr)
        return 130
    except Exception as exc:
        metadata["error"] = str(exc)
        metadata["status"] = "failed"
        metadata["complete"] = False
        write_meta()
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
