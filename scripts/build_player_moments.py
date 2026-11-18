#!/usr/bin/env python3
"""Build Tagger-compatible per-track ball-proximity moments from a tracking CSV.

This is a pure post-processor: it does not load video, YOLO, SigLIP, or OCR models.
The output cut-lists deliberately contain exactly the seven columns consumed by
Track A's automatic-moment importer.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

CUTLIST_FIELDS = [
    "moment", "start_s", "end_s", "duration_s",
    "start_clock", "events", "label",
]
TRACKING_REQUIRED_FIELDS = {
    "match_id", "frame", "video_seconds", "track_id", "role",
    "team", "x", "y", "confidence", "camera_state",
}
PLAYER_ROLES = {"player", "goalkeeper"}


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def finite_int(value: Any) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return number


def format_video_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def infer_sampling_interval(
    metadata: dict[str, Any] | None,
    observed_deltas: Iterable[float] = (),
    override: float | None = None,
) -> float:
    """Infer one sample interval from segment metadata or consecutive CSV frames."""
    if override is not None:
        if override <= 0:
            raise ValueError("--sampling-interval-seconds must be positive")
        return float(override)

    metadata = metadata or {}
    segment_details = []
    for segment in metadata.get("segments", []):
        details = segment.get("details", {}) if isinstance(segment, dict) else {}
        if isinstance(details, dict):
            segment_details.append(details)
    if not segment_details:
        segment_details = [metadata]

    intervals = []
    for details in segment_details:
        stride = finite_float(details.get("frame_stride"))
        fps = finite_float(details.get("fps"))
        if stride and fps and stride > 0 and fps > 0:
            intervals.append(stride / fps)
    if intervals:
        return float(statistics.median(intervals))

    deltas = [float(value) for value in observed_deltas if value > 0 and math.isfinite(value)]
    if deltas:
        return float(statistics.median(deltas))
    raise ValueError(
        "Cannot infer the tracking sample interval. Supply metadata or "
        "--sampling-interval-seconds."
    )


def match_ball_observation(
    player_frame: int | None,
    player_seconds: float,
    ball_by_frame: dict[int, tuple[float, float, float]],
    ball_by_time: list[tuple[float, int, float, float]],
    max_time_delta: float,
) -> tuple[tuple[float, float, float] | None, float | None]:
    """Return (ball_x, ball_y, confidence), preferring exact frame then nearest PTS."""
    if player_frame is not None and player_frame in ball_by_frame:
        return ball_by_frame[player_frame], 0.0
    if not ball_by_time:
        return None, None

    times = [item[0] for item in ball_by_time]
    index = bisect.bisect_left(times, player_seconds)
    candidates = []
    if index < len(ball_by_time):
        candidates.append(ball_by_time[index])
    if index > 0:
        candidates.append(ball_by_time[index - 1])
    if not candidates:
        return None, None

    nearest = min(candidates, key=lambda item: abs(item[0] - player_seconds))
    delta = abs(nearest[0] - player_seconds)
    if delta > max_time_delta:
        return None, None
    return (nearest[2], nearest[3], 1.0), delta


def merge_candidate_observations(
    observations: list[dict[str, Any]],
    merge_gap_seconds: float,
    sample_interval_seconds: float,
) -> list[dict[str, Any]]:
    """Merge near-ball observations when the uncovered gap is within the threshold."""
    if not observations:
        return []
    ordered = sorted(observations, key=lambda item: item["video_seconds"])
    windows: list[dict[str, Any]] = []
    current = [ordered[0]]
    last_seconds = ordered[0]["video_seconds"]

    for observation in ordered[1:]:
        seconds = observation["video_seconds"]
        uncovered_gap = seconds - (last_seconds + sample_interval_seconds)
        if uncovered_gap <= merge_gap_seconds + 1e-9:
            current.append(observation)
        else:
            windows.append(_window_from_observations(current, sample_interval_seconds))
            current = [observation]
        last_seconds = seconds

    windows.append(_window_from_observations(current, sample_interval_seconds))
    return windows


def _window_from_observations(
    observations: list[dict[str, Any]], sample_interval_seconds: float
) -> dict[str, Any]:
    start = float(observations[0]["video_seconds"])
    last = float(observations[-1]["video_seconds"])
    return {
        "start_s": start,
        "end_s": last + sample_interval_seconds,
        "candidate_observations": observations,
    }


def confidence_for_moment(
    *,
    calibration_error_m: float | None,
    wide_share: float,
    unknown_team_share: float,
    ball_observation_share: float,
    continuity_max_gap_s: float,
    observation_count: int,
    duration_s: float,
    sampling_interval_s: float,
) -> tuple[str, list[str]]:
    """Conservative automatic confidence; v1 never labels a moment 'wrong'."""
    reasons: list[str] = []
    if calibration_error_m is None:
        reasons.append("calibration_unavailable")
    elif calibration_error_m > 1.0:
        reasons.append("calibration_error_high")
    if wide_share < 0.90:
        reasons.append("wide_camera_share_low")
    if unknown_team_share > 0.25:
        reasons.append("unknown_team_share_high")
    if ball_observation_share < 0.50:
        reasons.append("ball_observation_share_low")
    if observation_count < 3 or duration_s < 2.0:
        reasons.append("short_or_sparse_track")
    max_allowed_gap = max(0.75, sampling_interval_s * 3.0)
    if continuity_max_gap_s > max_allowed_gap:
        reasons.append("track_continuity_gap")
    return ("review" if reasons else "ok"), reasons


def _metadata_segments(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    metadata = metadata or {}
    segments = []
    for segment in metadata.get("segments", []):
        details = segment.get("details", {}) if isinstance(segment, dict) else {}
        if isinstance(details, dict):
            segments.append(details)
    if not segments:
        segments = [metadata]
    return segments


def calibration_error_for_window(
    metadata: dict[str, Any] | None, start_s: float, end_s: float
) -> float | None:
    """Median of calibration summaries for source segments overlapping a moment."""
    errors = []
    for details in _metadata_segments(metadata):
        error = finite_float(details.get("calibration_median_error_m"))
        if error is None:
            continue
        seg_start = finite_float(details.get("source_start_seconds"))
        seg_end = finite_float(
            details.get("effective_end_seconds", details.get("source_end_seconds"))
        )
        if seg_start is None or seg_end is None:
            errors.append(error)
            continue
        if start_s < seg_end and end_s >= seg_start:
            errors.append(error)
    if errors:
        return float(statistics.median(errors))
    return None


def load_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Metadata must contain a JSON object: {path}")
    return value


def read_ball_index(
    tracking_csv: Path,
) -> tuple[
    dict[int, tuple[float, float, float]],
    list[tuple[float, int, float, float]],
    set[str],
    list[float],
]:
    """First pass: index valid wide-ball positions and infer sampled-frame deltas."""
    ball_by_frame: dict[int, tuple[float, float, float]] = {}
    frame_times: list[float] = []
    match_ids: set[str] = set()
    observed_deltas: list[float] = []
    last_frame: int | None = None
    last_seconds: float | None = None

    with tracking_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Tracking CSV has no header: {tracking_csv}")
        missing = TRACKING_REQUIRED_FIELDS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Tracking CSV missing required columns: {sorted(missing)}")

        for row in reader:
            match_id = (row.get("match_id") or "").strip()
            if match_id:
                match_ids.add(match_id)
            frame = finite_int(row.get("frame"))
            seconds = finite_float(row.get("video_seconds"))
            if frame is not None and seconds is not None and frame != last_frame:
                if (
                    last_frame is not None and last_seconds is not None
                    and frame > last_frame and seconds > last_seconds
                    and len(observed_deltas) < 10000
                ):
                    observed_deltas.append(seconds - last_seconds)
                last_frame, last_seconds = frame, seconds

            if (row.get("role") or "").strip().lower() != "ball":
                continue
            if (row.get("camera_state") or "").strip().lower() != "wide":
                continue
            if frame is None or seconds is None:
                continue
            x, y = finite_float(row.get("x")), finite_float(row.get("y"))
            if x is None or y is None or not (0 <= x <= 105 and 0 <= y <= 68):
                continue
            confidence = finite_float(row.get("confidence")) or 0.0
            previous = ball_by_frame.get(frame)
            candidate = (x, y, confidence)
            if previous is None or confidence > previous[2]:
                ball_by_frame[frame] = candidate

    ball_by_time = []
    # Re-read the ball-only records in timestamp order from the frame index.
    # This avoids keeping the original CSV rows in memory.
    with tracking_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("role") or "").strip().lower() != "ball":
                continue
            if (row.get("camera_state") or "").strip().lower() != "wide":
                continue
            frame = finite_int(row.get("frame"))
            seconds = finite_float(row.get("video_seconds"))
            if frame is None or seconds is None or frame not in ball_by_frame:
                continue
            x, y, confidence = ball_by_frame[frame]
            ball_by_time.append((seconds, frame, x, y))
    ball_by_time.sort(key=lambda item: item[0])
    # Deduplicate frames in the fallback index.
    deduplicated = []
    seen_frames = set()
    for item in ball_by_time:
        if item[1] not in seen_frames:
            deduplicated.append(item)
            seen_frames.add(item[1])
    return ball_by_frame, deduplicated, match_ids, observed_deltas


def _candidate_observations(
    tracking_csv: Path,
    ball_by_frame: dict[int, tuple[float, float, float]],
    ball_by_time: list[tuple[float, int, float, float]],
    max_time_delta: float,
    ball_distance_m: float,
) -> dict[int, list[dict[str, Any]]]:
    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with tracking_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            role = (row.get("role") or "").strip().lower()
            if role not in PLAYER_ROLES:
                continue
            if (row.get("camera_state") or "").strip().lower() != "wide":
                continue
            track_id = finite_int(row.get("track_id"))
            frame = finite_int(row.get("frame"))
            seconds = finite_float(row.get("video_seconds"))
            x, y = finite_float(row.get("x")), finite_float(row.get("y"))
            if track_id is None or seconds is None or x is None or y is None:
                continue
            if not (0 <= x <= 105 and 0 <= y <= 68):
                continue
            ball, time_delta = match_ball_observation(
                frame, seconds, ball_by_frame, ball_by_time, max_time_delta
            )
            if ball is None:
                continue
            distance = math.hypot(x - ball[0], y - ball[1])
            if distance > ball_distance_m:
                continue
            candidates[track_id].append({
                "video_seconds": seconds,
                "frame": frame,
                "team": (row.get("team") or "unknown").strip().lower() or "unknown",
                "confidence": finite_float(row.get("confidence")),
                "ball_time_delta_s": time_delta,
                "ball_distance_m": distance,
            })
    return candidates


def _moment_quality(
    tracking_csv: Path,
    track_id: int,
    start_s: float,
    end_s: float,
    ball_by_frame: dict[int, tuple[float, float, float]],
    ball_by_time: list[tuple[float, int, float, float]],
    max_time_delta: float,
) -> dict[str, Any]:
    total = wide = unknown = ball_observations = 0
    times: list[float] = []
    team_counts: Counter[str] = Counter()

    with tracking_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("role") or "").strip().lower() not in PLAYER_ROLES:
                continue
            if finite_int(row.get("track_id")) != track_id:
                continue
            seconds = finite_float(row.get("video_seconds"))
            if seconds is None or seconds < start_s or seconds > end_s:
                continue
            total += 1
            times.append(seconds)
            camera_state = (row.get("camera_state") or "").strip().lower()
            if camera_state == "wide":
                wide += 1
            team = (row.get("team") or "unknown").strip().lower() or "unknown"
            team_counts[team] += 1
            if team == "unknown":
                unknown += 1
            frame = finite_int(row.get("frame"))
            _, _, x = (None, None, None)
            del x
            # Count a ball observation independently of whether it is close enough.
            matched_ball, _ = match_ball_observation(
                frame, seconds, ball_by_frame, ball_by_time, max_time_delta
            )
            if matched_ball is not None:
                ball_observations += 1

    times.sort()
    max_gap = max(
        (right - left for left, right in zip(times, times[1:])),
        default=0.0,
    )
    return {
        "observation_count": total,
        "wide_share": wide / total if total else 0.0,
        "unknown_team_share": unknown / total if total else 1.0,
        "ball_observation_share": ball_observations / total if total else 0.0,
        "continuity_max_gap_s": max_gap,
        "team": team_counts.most_common(1)[0][0] if team_counts else "unknown",
    }


def write_moment_cut_lists(
    tracking_csv: Path,
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
    *,
    ball_distance_m: float = 2.5,
    merge_gap_seconds: float = 1.2,
    min_duration_seconds: float = 0.5,
    sampling_interval_seconds: float | None = None,
) -> dict[str, Any]:
    if ball_distance_m <= 0:
        raise ValueError("ball_distance_m must be positive")
    if merge_gap_seconds < 0:
        raise ValueError("merge_gap_seconds must be non-negative")
    if min_duration_seconds < 0:
        raise ValueError("min_duration_seconds must be non-negative")

    ball_by_frame, ball_by_time, match_ids, observed_deltas = read_ball_index(tracking_csv)
    if len(match_ids) != 1:
        raise ValueError(
            "Input must contain exactly one non-empty match_id; "
            f"found {sorted(match_ids)}"
        )
    match_id = next(iter(match_ids))
    if metadata and metadata.get("match_id") not in (None, match_id):
        raise ValueError(
            f"Metadata match_id {metadata.get('match_id')!r} does not match CSV {match_id!r}"
        )

    sample_interval = infer_sampling_interval(
        metadata, observed_deltas, sampling_interval_seconds
    )
    candidates = _candidate_observations(
        tracking_csv, ball_by_frame, ball_by_time, sample_interval, ball_distance_m
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_track: list[dict[str, Any]] = []
    total_moments = 0

    for track_id in sorted(candidates):
        windows = merge_candidate_observations(
            candidates[track_id], merge_gap_seconds, sample_interval
        )
        accepted = [
            window for window in windows
            if window["end_s"] - window["start_s"] >= min_duration_seconds
        ]
        if not accepted:
            continue

        cutlist_path = output_dir / f"track_{track_id}_moments.csv"
        moment_diagnostics = []
        with cutlist_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CUTLIST_FIELDS)
            writer.writeheader()
            for sequence, window in enumerate(accepted, start=1):
                start_s, end_s = window["start_s"], window["end_s"]
                quality = _moment_quality(
                    tracking_csv, track_id, start_s, end_s,
                    ball_by_frame, ball_by_time, sample_interval,
                )
                calibration_error = calibration_error_for_window(
                    metadata, start_s, end_s
                )
                duration = end_s - start_s
                label, reasons = confidence_for_moment(
                    calibration_error_m=calibration_error,
                    wide_share=quality["wide_share"],
                    unknown_team_share=quality["unknown_team_share"],
                    ball_observation_share=quality["ball_observation_share"],
                    continuity_max_gap_s=quality["continuity_max_gap_s"],
                    observation_count=quality["observation_count"],
                    duration_s=duration,
                    sampling_interval_s=sample_interval,
                )
                moment_id = f"track_{track_id}_{sequence:03d}"
                writer.writerow({
                    "moment": moment_id,
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "duration_s": round(duration, 3),
                    "start_clock": format_video_clock(start_s),
                    "events": "near_ball",
                    "label": label,
                })
                moment_diagnostics.append({
                    "moment": moment_id,
                    "track_id": track_id,
                    "team": quality["team"],
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "calibration_error_m": calibration_error,
                    "wide_share": round(quality["wide_share"], 4),
                    "unknown_team_share": round(quality["unknown_team_share"], 4),
                    "ball_observation_share": round(quality["ball_observation_share"], 4),
                    "continuity_max_gap_s": round(quality["continuity_max_gap_s"], 3),
                    "observation_count": quality["observation_count"],
                    "label": label,
                    "review_reasons": reasons,
                })

        total_moments += len(moment_diagnostics)
        per_track.append({
            "track_id": track_id,
            "cut_list": cutlist_path.name,
            "moment_count": len(moment_diagnostics),
            "moments": moment_diagnostics,
        })

    manifest = {
        "schema_version": 1,
        "match_id": match_id,
        "source_tracking_csv": str(tracking_csv),
        "cut_list_columns": CUTLIST_FIELDS,
        "parameters": {
            "ball_distance_m": ball_distance_m,
            "merge_gap_seconds": merge_gap_seconds,
            "min_duration_seconds": min_duration_seconds,
            "sampling_interval_seconds": round(sample_interval, 6),
            "ball_match_max_time_delta_seconds": round(sample_interval, 6),
        },
        "confidence_policy": {
            "ok": (
                "calibration <= 1.0 m, wide share >= 0.90, unknown-team share <= 0.25, "
                "ball observation share >= 0.50, at least 3 observations, duration >= 2 s, "
                "and no track-observation gap above max(0.75 s, 3 sample intervals)"
            ),
            "review": "one or more quality checks fail or required calibration is unavailable",
            "wrong": "never assigned automatically in v1; reserved for human correction",
        },
        "track_count": len(per_track),
        "moment_count": total_moments,
        "tracks": per_track,
    }
    manifest_path = output_dir / f"{tracking_csv.stem}_moments.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Tagger-compatible per-track near-ball moment cut-lists."
    )
    parser.add_argument("--input", type=Path, required=True, help="Track B tracking CSV")
    parser.add_argument("--output-dir", type=Path, help="Output folder (default: <input parent>/moments)")
    parser.add_argument("--metadata", type=Path, help="Metadata JSON (default: adjacent .meta.json when present)")
    parser.add_argument("--ball-distance-m", type=float, default=2.5)
    parser.add_argument("--merge-gap-seconds", type=float, default=1.2)
    parser.add_argument("--min-duration-seconds", type=float, default=0.5)
    parser.add_argument(
        "--sampling-interval-seconds", type=float,
        help="Override sample interval; otherwise inferred from metadata or CSV timestamps",
    )
    args = parser.parse_args()

    tracking_csv = args.input.expanduser().resolve()
    if not tracking_csv.is_file():
        parser.error(f"Tracking CSV not found: {tracking_csv}")
    metadata_path = (
        args.metadata.expanduser().resolve()
        if args.metadata
        else tracking_csv.with_suffix(".meta.json")
    )
    metadata = load_metadata(metadata_path) if metadata_path.is_file() else None
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else tracking_csv.parent / "moments"
    )

    manifest = write_moment_cut_lists(
        tracking_csv,
        output_dir,
        metadata,
        ball_distance_m=args.ball_distance_m,
        merge_gap_seconds=args.merge_gap_seconds,
        min_duration_seconds=args.min_duration_seconds,
        sampling_interval_seconds=args.sampling_interval_seconds,
    )
    print(
        f"Built {manifest['moment_count']} moments across "
        f"{manifest['track_count']} tracks for match {manifest['match_id']}."
    )
    print(f"Cut-lists and manifest: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
