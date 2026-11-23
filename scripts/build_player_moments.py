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

import pandas as pd

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

    segment_details = _normalise_segments(metadata)

    # Consecutive sampled-frame PTS deltas take precedence, which also handles
    # variable-frame-rate sources. Nominal stride/FPS is only a fallback.
    deltas = [float(value) for value in observed_deltas if value > 0 and math.isfinite(value)]
    if deltas:
        return float(statistics.median(deltas))

    intervals = []
    for details in segment_details:
        stride = finite_float(details.get("frame_stride"))
        fps = finite_float(details.get("fps"))
        if stride and fps and stride > 0 and fps > 0:
            intervals.append(stride / fps)
    if intervals:
        return float(statistics.median(intervals))
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

    # Tuples are sorted by their first element (video_seconds), so bisect
    # directly without rebuilding a full timestamp list for every player row.
    index = bisect.bisect_left(
        ball_by_time, (player_seconds, -1, float("-inf"), float("-inf"))
    )
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


def _normalise_segments(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Per-source-segment calibration/timing, from either combined-metadata shape.

    Old Colab combiner: metadata["segments"][i]["details"] with source_start/end_seconds.
    run_match combiner:  metadata["chunks"][i] with flat start/end and calibration; fps and
    frame_stride live at the top level. Returns a list of detail dicts in a single shape.
    """
    metadata = metadata or {}
    out: list[dict[str, Any]] = []
    for segment in metadata.get("segments", []):
        details = segment.get("details", {}) if isinstance(segment, dict) else {}
        if isinstance(details, dict) and details:
            out.append(details)
    if not out:
        top_fps = metadata.get("fps")
        top_stride = metadata.get("frame_stride")
        for chunk in metadata.get("chunks", []):
            if not isinstance(chunk, dict):
                continue
            out.append({
                "calibration_median_error_m": chunk.get("calibration_median_error_m"),
                "source_start_seconds": chunk.get("start", chunk.get("source_start_seconds")),
                "source_end_seconds": chunk.get("end", chunk.get("source_end_seconds")),
                "fps": chunk.get("fps", top_fps),
                "frame_stride": chunk.get("frame_stride", top_stride),
            })
    if not out:
        out = [metadata]
    return out


def _count_labels(per_track: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for track in per_track:
        for moment in track.get("moments", []):
            counts[moment["label"]] = counts.get(moment["label"], 0) + 1
    return counts


def _metadata_segments(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    metadata = metadata or {}
    return _normalise_segments(metadata)


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
    expected_frame_stride: int | None = None,
) -> tuple[
    dict[int, tuple[float, float, float]],
    list[tuple[float, int, float, float]],
    set[str],
    list[float],
]:
    """First pass: index valid wide-ball positions and infer sampled-frame deltas."""
    ball_by_frame: dict[int, tuple[float, float, float]] = {}
    ball_timestamps: dict[int, float] = {}
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
                frame_delta = frame - last_frame if last_frame is not None else None
                is_consecutive_sample = (
                    frame_delta is not None
                    and frame_delta > 0
                    and (
                        expected_frame_stride is None
                        or frame_delta == expected_frame_stride
                    )
                )
                if (
                    is_consecutive_sample
                    and last_seconds is not None
                    and seconds > last_seconds
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
                ball_timestamps[frame] = seconds

    ball_by_time = sorted(
        (seconds, frame, ball_by_frame[frame][0], ball_by_frame[frame][1])
        for frame, seconds in ball_timestamps.items()
    )
    return ball_by_frame, ball_by_time, match_ids, observed_deltas


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
                "events": "near_ball",
            })
    return candidates


def _empty_quality() -> dict[str, Any]:
    return {
        "observation_count": 0,
        "wide_count": 0,
        "unknown_count": 0,
        "ball_observation_count": 0,
        "last_seconds": None,
        "continuity_max_gap_s": 0.0,
        "team_counts": Counter(),
    }


def _collect_quality_for_windows(
    tracking_csv: Path,
    windows_by_track: dict[int, list[dict[str, Any]]],
    ball_by_frame: dict[int, tuple[float, float, float]],
    ball_by_time: list[tuple[float, int, float, float]],
    max_time_delta: float,
) -> None:
    """Accumulate all moment quality metrics in one streaming pass over the CSV."""
    starts = {
        track_id: [window["start_s"] for window in windows]
        for track_id, windows in windows_by_track.items()
    }
    for windows in windows_by_track.values():
        for window in windows:
            window["quality"] = _empty_quality()

    with tracking_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("role") or "").strip().lower() not in PLAYER_ROLES:
                continue
            track_id = finite_int(row.get("track_id"))
            seconds = finite_float(row.get("video_seconds"))
            if track_id is None or seconds is None or track_id not in windows_by_track:
                continue
            track_windows = windows_by_track[track_id]
            index = bisect.bisect_right(starts[track_id], seconds) - 1
            if index < 0:
                continue
            window = track_windows[index]
            if seconds < window["start_s"] or seconds > window["end_s"]:
                continue

            quality = window["quality"]
            quality["observation_count"] += 1
            previous_seconds = quality["last_seconds"]
            if previous_seconds is not None:
                quality["continuity_max_gap_s"] = max(
                    quality["continuity_max_gap_s"], seconds - previous_seconds
                )
            quality["last_seconds"] = seconds

            if (row.get("camera_state") or "").strip().lower() == "wide":
                quality["wide_count"] += 1
            team = (row.get("team") or "unknown").strip().lower() or "unknown"
            quality["team_counts"][team] += 1
            if team == "unknown":
                quality["unknown_count"] += 1

            frame = finite_int(row.get("frame"))
            matched_ball, _ = match_ball_observation(
                frame, seconds, ball_by_frame, ball_by_time, max_time_delta
            )
            if matched_ball is not None:
                quality["ball_observation_count"] += 1

    for windows in windows_by_track.values():
        for window in windows:
            quality = window["quality"]
            total = quality["observation_count"]
            teams = quality["team_counts"]
            window["quality"] = {
                "observation_count": total,
                "wide_share": quality["wide_count"] / total if total else 0.0,
                "unknown_team_share": quality["unknown_count"] / total if total else 1.0,
                "ball_observation_share": (
                    quality["ball_observation_count"] / total if total else 0.0
                ),
                "continuity_max_gap_s": quality["continuity_max_gap_s"],
                "team": teams.most_common(1)[0][0] if teams else "unknown",
            }


def defensive_candidate_observations(
    tracking_csv: Path,
    metadata: dict[str, Any] | None,
) -> dict[int, list[dict[str, Any]]]:
    """Find the two nearest outfield defenders while the opponent holds the ball in their own third.

    Direction is read from the resolved combined-run metadata. If direction is
    unavailable for a period, defensive moments for that period are skipped rather
    than guessed. The possession owner/holder timeline is shared with team_report.
    """
    directions = (metadata or {}).get("home_attacks_right_by_period") or {}
    if not isinstance(directions, dict) or not directions:
        return {}

    required = [
        "frame", "video_seconds", "period", "track_id", "role", "team", "x", "y", "camera_state"
    ]
    frame_df = pd.read_csv(
        tracking_csv,
        usecols=required,
        dtype={"x": "float32", "y": "float32", "period": "int8"},
    )
    for col in ("role", "team", "camera_state"):
        frame_df[col] = frame_df[col].astype(str)

    try:
        from team_report import possession_frames
    except ImportError:  # imported as scripts.build_player_moments in tests
        from scripts.team_report import possession_frames

    possession = possession_frames(frame_df)
    possession_by_frame = possession.set_index("frame")
    players = frame_df[
        (frame_df.camera_state == "wide")
        & (frame_df.role == "player")
        & frame_df.team.isin(["home", "away"])
    ].dropna(subset=["x", "y"])
    players_by_frame = {
        int(frame): group
        for frame, group in players.groupby("frame", sort=False)
    }
    balls = frame_df[
        (frame_df.camera_state == "wide") & (frame_df.role == "ball")
    ].dropna(subset=["x", "y", "video_seconds", "period"])

    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ball in balls.itertuples(index=False):
        frame = int(ball.frame)
        state = possession_by_frame.loc[frame] if frame in possession_by_frame.index else None
        if state is None or state.owner not in ("home", "away") or state.holder != state.owner:
            continue
        owner = str(state.owner)
        defending_team = "away" if owner == "home" else "home"
        home_right = directions.get(str(int(ball.period)))
        if not isinstance(home_right, bool):
            continue
        defending_attacks_right = home_right if defending_team == "home" else not home_right
        ball_team_x = float(ball.x) if defending_attacks_right else 105.0 - float(ball.x)
        if ball_team_x > 35.0:
            continue

        frame_players = players_by_frame.get(frame)
        if frame_players is None or frame_players.empty:
            continue
        ranked = []
        for player in frame_players.itertuples(index=False):
            if player.team != defending_team:
                continue
            distance = math.hypot(float(player.x) - float(ball.x), float(player.y) - float(ball.y))
            ranked.append((distance, int(float(player.track_id)) if hasattr(player, "track_id") else None))
        ranked = [(distance, tid) for distance, tid in ranked if tid is not None]
        ranked.sort(key=lambda item: item[0])
        for distance, track_id in ranked[:2]:
            result[track_id].append({
                "video_seconds": float(ball.video_seconds),
                "frame": frame,
                "team": defending_team,
                "confidence": None,
                "ball_time_delta_s": 0.0,
                "ball_distance_m": distance,
                "events": "defensive_nearest_2",
            })
    return result


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

    stride_values = [
        finite_int(details.get("frame_stride"))
        for details in _metadata_segments(metadata)
        if finite_int(details.get("frame_stride")) is not None
        and finite_int(details.get("frame_stride")) > 0
    ]
    expected_stride = (
        int(statistics.median(stride_values)) if stride_values else None
    )
    ball_by_frame, ball_by_time, match_ids, observed_deltas = read_ball_index(
        tracking_csv, expected_frame_stride=expected_stride
    )
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
    defensive = defensive_candidate_observations(tracking_csv, metadata)
    for track_id, observations in defensive.items():
        candidates.setdefault(track_id, []).extend(observations)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{tracking_csv.stem}_moments.manifest.json"
    previous_cutlists: set[str] = set()
    if manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_cutlists = {
                str(track["cut_list"])
                for track in previous_manifest.get("tracks", [])
                if isinstance(track, dict) and isinstance(track.get("cut_list"), str)
                and Path(track["cut_list"]).name == track["cut_list"]
                and track["cut_list"].startswith("track_")
                and track["cut_list"].endswith("_moments.csv")
            }
        except (OSError, json.JSONDecodeError, TypeError):
            previous_cutlists = set()

    windows_by_track: dict[int, list[dict[str, Any]]] = {}
    for track_id in sorted(candidates):
        windows = merge_candidate_observations(
            candidates[track_id], merge_gap_seconds, sample_interval
        )
        for window in windows:
            event_types = sorted({
                str(observation.get("events") or "near_ball")
                for observation in window["candidate_observations"]
            })
            window["events"] = "+".join(event_types)
        accepted = [
            window for window in windows
            if window["end_s"] - window["start_s"] >= min_duration_seconds
        ]
        if accepted:
            windows_by_track[track_id] = accepted

    # One streaming quality pass for all generated windows; no repeated full-CSV scans.
    _collect_quality_for_windows(
        tracking_csv, windows_by_track, ball_by_frame, ball_by_time, sample_interval
    )

    per_track: list[dict[str, Any]] = []
    total_moments = 0
    for track_id, accepted in sorted(windows_by_track.items()):
        cutlist_path = output_dir / f"track_{track_id}_moments.csv"
        moment_diagnostics = []
        with cutlist_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CUTLIST_FIELDS)
            writer.writeheader()
            for sequence, window in enumerate(accepted, start=1):
                start_s, end_s = window["start_s"], window["end_s"]
                quality = window["quality"]
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
                    "events": window.get("events", "near_ball"),
                    "label": label,
                })
                moment_diagnostics.append({
                    "moment": moment_id,
                    "track_id": track_id,
                    "team": quality["team"],
                    "events": window.get("events", "near_ball"),
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
            "defensive": {
                "enabled": bool((metadata or {}).get("home_attacks_right_by_period")),
                "definition": "two nearest outfield defenders to the ball while the opponent is the current "
                              "possession holder and the ball is in the defending team's own third",
                "nearest_defenders": 2,
                "own_third_m": 35.0,
            },
        },
        "defensive_observation_count": sum(len(v) for v in defensive.values()),
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
        "label_counts": _count_labels(per_track),
        "tracks": per_track,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    current_cutlists = {track["cut_list"] for track in per_track}
    # Remove only stale cut-lists previously declared by this script's manifest.
    for stale_name in previous_cutlists - current_cutlists:
        (output_dir / stale_name).unlink(missing_ok=True)
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
