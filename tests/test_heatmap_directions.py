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
