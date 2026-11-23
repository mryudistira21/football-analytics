import csv
import json
from pathlib import Path

import pytest

from scripts.build_player_moments import (
    CUTLIST_FIELDS,
    confidence_for_moment,
    infer_sampling_interval,
    match_ball_observation,
    merge_candidate_observations,
    write_moment_cut_lists,
)
from scripts.track_video import classify_segment_status


def write_tracking_csv(
    path: Path, *, include_ball: bool = True, unknown: bool = False,
    player_x: float = 50.0, ball_x: float = 51.0,
):
    fields = [
        "match_id", "frame", "video_seconds", "period", "track_id",
        "role", "team", "x", "y", "confidence", "camera_state",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(25):
            frame = index * 3
            seconds = 10.0 + index * 0.1
            writer.writerow({
                "match_id": "match-demo", "frame": frame,
                "video_seconds": round(seconds, 3), "period": 1,
                "track_id": 17, "role": "player",
                "team": "unknown" if unknown else "home",
                "x": player_x, "y": 34.0, "confidence": 0.9,
                "camera_state": "wide",
            })
            if include_ball:
                writer.writerow({
                    "match_id": "match-demo", "frame": frame,
                    "video_seconds": round(seconds, 3), "period": 1,
                    "track_id": 0, "role": "ball", "team": "unknown",
                    "x": ball_x, "y": 34.0, "confidence": 0.8,
                    "camera_state": "wide",
                })


def test_classify_segment_status_detects_no_frames():
    assert classify_segment_status(0, 900.0, False, None, 0.15) == "no_frames"


def test_classify_segment_status_marks_early_eof_partial():
    assert classify_segment_status(100, 900.0, False, 300.0, 0.15) == "partial_eof"


def test_classify_segment_status_marks_boundary_reached_complete():
    assert classify_segment_status(100, 900.0, True, 899.9, 0.15) == "complete"


def test_classify_segment_status_allows_full_video_to_end_at_eof():
    assert classify_segment_status(100, None, False, 899.9, 0.15) == "complete"


def test_match_ball_prefers_exact_frame_then_nearest_time():
    by_frame = {30: (51.0, 34.0, 0.8)}
    by_time = [(1.0, 30, 51.0, 34.0), (1.1, 33, 52.0, 34.0)]
    exact, exact_delta = match_ball_observation(30, 1.05, by_frame, by_time, 0.1)
    assert exact == (51.0, 34.0, 0.8)
    assert exact_delta == 0.0

    nearest, nearest_delta = match_ball_observation(None, 1.08, by_frame, by_time, 0.1)
    assert nearest == (52.0, 34.0, 1.0)
    assert nearest_delta == pytest.approx(0.02)


def test_merge_candidate_observations_merges_within_gap_and_splits_larger_gap():
    observations = [
        {"video_seconds": 0.0},
        {"video_seconds": 0.1},
        {"video_seconds": 1.3},  # uncovered gap is 1.1 s
        {"video_seconds": 3.0},  # uncovered gap is 1.6 s
    ]
    windows = merge_candidate_observations(
        observations, merge_gap_seconds=1.2, sample_interval_seconds=0.1
    )
    assert len(windows) == 2
    assert windows[0]["start_s"] == 0.0
    assert windows[0]["end_s"] == pytest.approx(1.4)
    assert windows[1]["start_s"] == 3.0


def test_infer_sampling_interval_uses_stride_over_fps():
    interval = infer_sampling_interval(
        {"frame_stride": 3, "fps": 30.0},
        observed_deltas=[0.1, 0.1],
    )
    assert interval == pytest.approx(0.1)


def test_infer_sampling_interval_prefers_observed_pts_for_vfr():
    interval = infer_sampling_interval(
        {"frame_stride": 3, "fps": 30.0},
        observed_deltas=[0.09, 0.11, 0.12],
    )
    assert interval == pytest.approx(0.11)


def test_confidence_is_conservative_and_never_auto_labels_wrong():
    label, reasons = confidence_for_moment(
        calibration_error_m=0.64,
        wide_share=1.0,
        unknown_team_share=0.0,
        ball_observation_share=1.0,
        continuity_max_gap_s=0.1,
        observation_count=8,
        duration_s=2.0,
        sampling_interval_s=0.1,
    )
    assert label == "ok"
    assert reasons == []

    review, reasons = confidence_for_moment(
        calibration_error_m=None,
        wide_share=0.5,
        unknown_team_share=0.8,
        ball_observation_share=0.2,
        continuity_max_gap_s=1.5,
        observation_count=2,
        duration_s=0.2,
        sampling_interval_s=0.1,
    )
    assert review == "review"
    assert "calibration_unavailable" in reasons
    assert review != "wrong"


def test_build_player_moments_writes_exact_seven_column_cutlist(tmp_path):
    tracking_csv = tmp_path / "tracking.csv"
    write_tracking_csv(tracking_csv)
    metadata = {
        "match_id": "match-demo",
        "fps": 30.0,
        "frame_stride": 3,
        "calibration_median_error_m": 0.64,
        "source_start_seconds": 0.0,
        "effective_end_seconds": 60.0,
    }
    output_dir = tmp_path / "moments"

    result = write_moment_cut_lists(tracking_csv, output_dir, metadata)

    assert result["match_id"] == "match-demo"
    assert result["moment_count"] == 1
    assert result["track_count"] == 1

    cutlist = output_dir / "track_17_moments.csv"
    with cutlist.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CUTLIST_FIELDS
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["moment"] == "track_17_001"
    assert rows[0]["events"] == "near_ball"
    assert rows[0]["label"] == "ok"
    assert float(rows[0]["start_s"]) == pytest.approx(10.0)
    assert float(rows[0]["end_s"]) == pytest.approx(12.5)

    manifest_path = output_dir / "tracking_moments.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["match_id"] == "match-demo"
    assert manifest["tracks"][0]["track_id"] == 17
    assert manifest["tracks"][0]["moments"][0]["team"] == "home"


def test_players_far_from_ball_do_not_generate_moments(tmp_path):
    tracking_csv = tmp_path / "tracking.csv"
    write_tracking_csv(tracking_csv, player_x=60.0, ball_x=51.0)
    metadata = {
        "match_id": "match-demo",
        "fps": 30.0,
        "frame_stride": 3,
        "calibration_median_error_m": 0.64,
    }

    result = write_moment_cut_lists(
        tracking_csv, tmp_path / "moments", metadata, ball_distance_m=2.5
    )
    assert result["moment_count"] == 0


def test_eof_within_sampling_tolerance_can_complete():
    assert classify_segment_status(100, 900.0, False, 899.9, 0.15) == "complete"


def test_rerun_removes_only_stale_cutlists_from_previous_manifest(tmp_path):
    tracking_csv = tmp_path / "tracking.csv"
    output_dir = tmp_path / "moments"
    metadata = {
        "match_id": "match-demo",
        "fps": 30.0,
        "frame_stride": 3,
        "calibration_median_error_m": 0.64,
    }

    write_tracking_csv(tracking_csv)
    first = write_moment_cut_lists(tracking_csv, output_dir, metadata)
    cutlist = output_dir / "track_17_moments.csv"
    assert first["moment_count"] == 1
    assert cutlist.is_file()

    write_tracking_csv(tracking_csv, player_x=60.0, ball_x=51.0)
    second = write_moment_cut_lists(tracking_csv, output_dir, metadata)
    assert second["moment_count"] == 0
    assert not cutlist.exists()


def test_missing_ball_observations_produce_no_false_moments(tmp_path):
    tracking_csv = tmp_path / "tracking.csv"
    write_tracking_csv(tracking_csv, include_ball=False)
    metadata = {
        "match_id": "match-demo",
        "fps": 30.0,
        "frame_stride": 3,
        "calibration_median_error_m": 0.64,
    }

    result = write_moment_cut_lists(
        tracking_csv, tmp_path / "moments", metadata
    )
    assert result["moment_count"] == 0
    assert result["track_count"] == 0


def test_unknown_team_moments_are_review_not_wrong(tmp_path):
    tracking_csv = tmp_path / "tracking.csv"
    write_tracking_csv(tracking_csv, unknown=True)
    metadata = {
        "match_id": "match-demo",
        "fps": 30.0,
        "frame_stride": 3,
        "calibration_median_error_m": 0.64,
    }

    result = write_moment_cut_lists(
        tracking_csv, tmp_path / "moments", metadata
    )
    assert result["moment_count"] == 1
    moment = result["tracks"][0]["moments"][0]
    assert moment["label"] == "review"
    assert "unknown_team_share_high" in moment["review_reasons"]


def test_calibration_read_from_run_match_chunks_metadata():
    from scripts.build_player_moments import _normalise_segments, calibration_error_for_window
    # run_match combine() metadata shape: flat "chunks", fps/stride at top level
    meta = {"fps": 30.0, "frame_stride": 3, "chunks": [
        {"start": 0.0, "end": 60.0, "calibration_median_error_m": 0.5},
        {"start": 60.0, "end": 120.0, "calibration_median_error_m": 0.7}]}
    segs = _normalise_segments(meta)
    assert len(segs) == 2 and segs[0]["source_start_seconds"] == 0.0
    assert segs[0]["fps"] == 30.0 and segs[0]["frame_stride"] == 3
    # a moment inside the first chunk gets that chunk's calibration, not "unavailable"
    assert calibration_error_for_window(meta, 10.0, 20.0) == 0.5
    # the old "segments"/"details" shape still works
    old = {"segments": [{"details": {"calibration_median_error_m": 0.4, "source_start_seconds": 0.0,
                                      "source_end_seconds": 60.0, "fps": 30.0, "frame_stride": 3}}]}
    assert calibration_error_for_window(old, 10.0, 20.0) == 0.4


def test_defensive_moments_select_two_nearest_outfield_defenders_and_keep_cutlist_contract(tmp_path):
    tracking_csv = tmp_path / "defensive.csv"
    fields = [
        "match_id", "frame", "video_seconds", "period", "track_id",
        "role", "team", "x", "y", "confidence", "camera_state",
    ]
    with tracking_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(40):
            frame = i * 3
            seconds = 10.0 + i * 0.1
            rows = [
                (0, "ball", "unknown", 30.0, 34.0, 0.9),
                (20, "player", "away", 30.5, 34.0, 0.9),
                (10, "player", "home", 33.0, 34.0, 0.9),
                (11, "player", "home", 34.0, 34.0, 0.9),
                (12, "player", "home", 38.0, 34.0, 0.9),
            ]
            for track_id, role, team, x, y, confidence in rows:
                writer.writerow({
                    "match_id": "defensive-demo", "frame": frame,
                    "video_seconds": seconds, "period": 1, "track_id": track_id,
                    "role": role, "team": team, "x": x, "y": y,
                    "confidence": confidence, "camera_state": "wide",
                })

    metadata = {
        "match_id": "defensive-demo",
        "fps": 30.0,
        "frame_stride": 3,
        "calibration_median_error_m": 0.5,
        "home_attacks_right_by_period": {"1": True, "2": False},
    }
    output_dir = tmp_path / "moments"
    result = write_moment_cut_lists(tracking_csv, output_dir, metadata, ball_distance_m=0.4)

    assert result["defensive_observation_count"] == 80
    assert result["moment_count"] == 2
    cutlist = output_dir / "track_11_moments.csv"
    with cutlist.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CUTLIST_FIELDS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["events"] == "defensive_nearest_2"
    assert rows[0]["label"] == "ok"


def test_defensive_moments_are_skipped_without_resolved_direction(tmp_path):
    tracking_csv = tmp_path / "no-direction.csv"
    write_tracking_csv(tracking_csv)
    metadata = {
        "match_id": "match-demo",
        "fps": 30.0,
        "frame_stride": 3,
        "calibration_median_error_m": 0.5,
    }
    result = write_moment_cut_lists(tracking_csv, tmp_path / "moments", metadata)
    assert result["defensive_observation_count"] == 0
