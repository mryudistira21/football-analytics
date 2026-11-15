import numpy as np
import pytest

from scripts.track_video import fit_homography, map_points, pitch_targets


def test_pitch_target_points_match_shared_pitch_dimensions():
    vertices = np.array(
        [[0, 0], [12000, 0], [12000, 7000], [0, 7000]],
        dtype=np.float32,
    )
    target = pitch_targets(vertices)
    np.testing.assert_allclose(
        target,
        [[0, 0], [105, 0], [105, 68], [0, 68]],
        atol=1e-5,
    )


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
