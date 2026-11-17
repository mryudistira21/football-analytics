import numpy as np
import pytest

from collections import Counter
from types import SimpleNamespace

from scripts.track_video import (
    choose_team_device,
    fit_homography,
    load_team_cluster_state,
    map_points,
    pitch_targets,
    save_team_cluster_state,
    should_refresh_team_embedding,
    team_clusters_from_embeddings,
)


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



def test_choose_team_device_uses_cuda_when_free_vram_meets_threshold():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda device: (6 * 1024**3, 16 * 1024**3),
        )
    )
    device, memory = choose_team_device(fake_torch, "auto", 4.0)
    assert device == "cuda:0"
    assert memory == {"free_gb": 6.0, "total_gb": 16.0}


def test_choose_team_device_falls_back_to_cpu_when_free_vram_is_low():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda device: (3 * 1024**3, 8 * 1024**3),
        )
    )
    device, memory = choose_team_device(fake_torch, "auto", 4.0)
    assert device == "cpu"
    assert memory["free_gb"] == 3.0


def test_choose_team_device_auto_uses_cpu_without_cuda():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    )
    assert choose_team_device(fake_torch, "auto") == ("cpu", None)


def test_choose_team_device_rejects_explicit_cuda_without_cuda():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    )
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        choose_team_device(fake_torch, "cuda:0")


def test_team_embedding_refresh_skips_stable_tracks():
    votes = Counter({"home": 4})
    assert should_refresh_team_embedding(votes, 100, 120, 150) is False
    assert should_refresh_team_embedding(votes, 100, 260, 150) is False


def test_team_embedding_refresh_rechecks_unstable_tracks_after_interval():
    votes = Counter({"home": 2, "away": 2})
    assert should_refresh_team_embedding(votes, 100, 249, 150) is False
    assert should_refresh_team_embedding(votes, 100, 250, 150) is True


def test_team_embedding_refresh_classifies_new_tracks():
    assert should_refresh_team_embedding(Counter(), None, 1, 150) is True


def test_team_clusters_from_cached_embeddings():
    class Reducer:
        def transform(self, features):
            return features[:, :1]

    class ClusterModel:
        def predict(self, projections):
            return (projections[:, 0] > 0.5).astype(int)

    classifier = SimpleNamespace(reducer=Reducer(), cluster_model=ClusterModel())
    clusters = team_clusters_from_embeddings(
        classifier, np.asarray([[0.2, 1.0], [0.8, 1.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(clusters, [0, 1])


def test_team_cluster_state_round_trip_is_independent_of_home_away_mapping(tmp_path):
    classifier = SimpleNamespace(
        reducer={"fitted": True},
        cluster_model={"clusters": 2},
    )
    path = tmp_path / "team-clusters.joblib"
    save_team_cluster_state(path, classifier, "home")
    reducer, cluster_model = load_team_cluster_state(path)
    assert reducer == {"fitted": True}
    assert cluster_model == {"clusters": 2}
    # Home/away mapping can be changed in a later run without refitting clusters.
    reducer_again, model_again = load_team_cluster_state(path)
    assert reducer_again == reducer
    assert model_again == cluster_model
