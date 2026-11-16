import numpy as np
import pytest

from scripts.track_video import fit_homography, map_points, pitch_targets


def roboflow_pitch_vertices():
    return np.asarray([
        [0, 0], [0, 1450], [0, 2584], [0, 4416], [0, 5550], [0, 7000],
        [550, 2584], [550, 4416], [1100, 3500],
        [2015, 1450], [2015, 2584], [2015, 4416], [2015, 5550],
        [6000, 0], [6000, 2585], [6000, 4415], [6000, 7000],
        [9985, 1450], [9985, 2584], [9985, 4416], [9985, 5550],
        [10900, 3500], [11450, 2584], [11450, 4416],
        [12000, 0], [12000, 1450], [12000, 2584], [12000, 4416],
        [12000, 5550], [12000, 7000], [5085, 3500], [6915, 3500],
    ], dtype=np.float32)


def test_pitch_targets_match_all_32_tagger_landmarks():
    vertices = roboflow_pitch_vertices()
    target = pitch_targets(vertices)
    expected = np.asarray([
        [0, 0], [0, 13.84], [0, 24.84], [0, 43.16], [0, 54.16], [0, 68],
        [5.5, 24.84], [5.5, 43.16], [11, 34],
        [16.5, 13.84], [16.5, 24.84], [16.5, 43.16], [16.5, 54.16],
        [52.5, 0], [52.5, 24.85], [52.5, 43.15], [52.5, 68],
        [88.5, 13.84], [88.5, 24.84], [88.5, 43.16], [88.5, 54.16],
        [94, 34], [99.5, 24.84], [99.5, 43.16],
        [105, 0], [105, 13.84], [105, 24.84], [105, 43.16],
        [105, 54.16], [105, 68], [43.35, 34], [61.65, 34],
    ], dtype=np.float32)

    assert target.shape == (32, 2)
    np.testing.assert_allclose(target, expected, atol=1e-5)
    # Explicitly guard the markings that were distorted by axis scaling.
    np.testing.assert_allclose(target[[9, 17], 0], [16.5, 88.5], atol=1e-5)
    np.testing.assert_allclose(target[[8, 21], 0], [11, 94], atol=1e-5)
    np.testing.assert_allclose(target[[14, 15], 1], [24.85, 43.15], atol=1e-5)


def test_pitch_targets_reject_unexpected_keypoint_order_or_geometry():
    vertices = roboflow_pitch_vertices()
    vertices[[9, 10]] = vertices[[10, 9]]
    with pytest.raises(ValueError, match="geometry/order"):
        pitch_targets(vertices)


def test_homography_maps_image_coordinates_to_pitch_metres():
    source = np.array(
        [[100, 100], [1100, 100], [1100, 700], [100, 700],
         [350, 250], [850, 250], [850, 550], [350, 550]],
        dtype=np.float32,
    )
    target = np.array(
        [[0, 0], [105, 0], [105, 68], [0, 68],
         [26.25, 17], [78.75, 17], [78.75, 51], [26.25, 51]],
        dtype=np.float32,
    )
    matrix, diagnostics = fit_homography(source, target, np.ones(len(source)))
    assert matrix is not None
    assert diagnostics["inliers"] >= 6
    mapped = map_points(np.array([[100, 100], [1100, 700]], dtype=np.float32), matrix)
    np.testing.assert_allclose(mapped, [[0, 0], [105, 68]], atol=0.05)


def test_homography_rejects_too_few_keypoints():
    source = np.array([[10, 10], [100, 10], [100, 100]], dtype=np.float32)
    matrix, diagnostics = fit_homography(source, source, min_points=4)
    assert matrix is None
    assert diagnostics["inliers"] == 0


def test_projection_marks_points_outside_pitch_as_missing():
    mapped = map_points(
        np.array([[20, 20], [400, 400]], dtype=np.float32),
        np.eye(3, dtype=np.float32),
    )
    assert np.isfinite(mapped[0]).all()
    assert np.isnan(mapped[1]).all()



def test_external_bytetrack_tracker_accepts_supervision_detections_and_timestamps():
    import supervision as sv
    from trackers import ByteTrackTracker

    tracker = ByteTrackTracker(
        track_activation_threshold=0.25,
        minimum_consecutive_frames=2,
        frame_rate=30.0,
    )
    empty = sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=np.int64),
    )

    first = tracker.update(empty, timestamp=0.0)
    second = tracker.update(empty, timestamp=0.1)

    assert isinstance(first, sv.Detections)
    assert isinstance(second, sv.Detections)
    assert len(first) == 0
    assert len(second) == 0
