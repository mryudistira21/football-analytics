from src.heatmap_from_positions import attacks_right


def test_period_directions_are_read_explicitly():
    directions = {"1": True, "2": False, "3": False, "4": True}
    assert attacks_right("home", 1, directions) is True
    assert attacks_right("away", 1, directions) is False
    assert attacks_right("home", 3, directions) is False
    assert attacks_right("away", 3, directions) is True
    assert attacks_right("home", 4, directions) is True


def test_extra_time_direction_is_not_inferred():
    try:
        attacks_right("home", 3, {"1": True, "2": False})
    except ValueError as exc:
        assert "must not be inferred" in str(exc)
    else:
        raise AssertionError("period 3 should need its own direction")


def test_directions_are_read_from_tracking_metadata(tmp_path):
    import json
    from src.heatmap_from_positions import directions_from_meta
    meta = tmp_path / "t.meta.json"
    meta.write_text(json.dumps({"home_attacks_right_by_period": {"1": False, "2": "R"}}))
    assert directions_from_meta(meta) == {"1": False, "2": True}
    meta.write_text(json.dumps({"home_dir_1h": "L"}))
    assert directions_from_meta(meta) == {"1": False, "2": True}
    assert directions_from_meta(tmp_path / "missing.json") is None
