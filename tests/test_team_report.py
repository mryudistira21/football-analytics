import csv
import json

import numpy as np

from scripts.team_report import build, markdown


def _synthetic(path, home_right_1h=True):
    """Home sits high (attacks right in P1), away sits deep; home always nearest the ball."""
    fields = ["match_id", "frame", "video_seconds", "period", "track_id", "role", "team", "x", "y",
              "confidence", "camera_state"]
    rng = np.random.default_rng(0)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for period in (1, 2):
            home_r = home_right_1h if period == 1 else not home_right_1h
            for f in range(200):
                frame = period * 10_000 + f * 3
                def screen(x_team, team):
                    # team frame (attacking right) -> screen frame
                    right = home_r if team == "home" else not home_r
                    return x_team if right else 105 - x_team
                for i in range(9):
                    for team, base in (("home", 60.0), ("away", 25.0)):
                        x = screen(base + rng.normal(0, 4), team)
                        w.writerow({"match_id": "t", "frame": frame, "video_seconds": frame / 30, "period": period,
                                    "track_id": i + (100 if team == "away" else 0), "role": "player", "team": team,
                                    "x": round(x, 2), "y": round(10 + i * 6, 2), "confidence": 0.9,
                                    "camera_state": "wide"})
                bx = screen(60.5, "home")
                w.writerow({"match_id": "t", "frame": frame, "video_seconds": frame / 30, "period": period,
                            "track_id": 0, "role": "ball", "team": "unknown", "x": bx, "y": 10.5,
                            "confidence": 0.5, "camera_state": "wide"})


def test_report_normalises_direction_and_halves(tmp_path):
    p = tmp_path / "t.csv"
    _synthetic(p, home_right_1h=False)
    match = {"match_id": "t", "home": "H", "away": "A", "score": [1, 0], "home_dir_1h": "L"}
    rep = build(p, match, {"fps": 30, "frame_stride": 3})
    for key in ("1", "2", "all"):
        b = rep["periods"][key]
        assert 55 < b["home"]["block_height_m"] < 66, b["home"]
        assert 20 < b["away"]["block_height_m"] < 31, b["away"]
        assert b["home"]["ball_proximity_share"] == 1.0
        assert b["territory_home_view"]["middle"] == 1.0
        assert b["quality"]["wide_share"] == 1.0
        assert len(b["home"]["heatmap"]) == 21 and len(b["home"]["heatmap"][0]) == 14
    assert "| Block height" in markdown(rep)
    json.dumps(rep)  # serialisable


def test_wrong_direction_shows_up_as_implausible_shape(tmp_path):
    p = tmp_path / "t.csv"
    _synthetic(p, home_right_1h=True)
    rep = build(p, {"match_id": "t", "home": "H", "away": "A", "home_dir_1h": "L"}, {"fps": 30, "frame_stride": 3})
    assert rep["periods"]["1"]["home"]["block_height_m"] < 50  # flipped: home now looks deep


def test_press_share_uses_opponent_possession_in_own_third_and_five_metre_holder_pressure():
    import pandas as pd
    from scripts.team_report import pressing_stats

    rows = []
    for frame, presser_x in ((1, 94.0), (2, 101.0), (3, None)):
        rows.append({"frame": frame, "role": "ball", "team": "unknown",
                     "x": 90.0, "y": 34.0, "tx": 90.0, "camera_state": "wide"})
        rows.append({"frame": frame, "role": "player", "team": "away",
                     "x": 90.5, "y": 34.0, "tx": 90.5, "camera_state": "wide"})
        if presser_x is not None:
            rows.append({"frame": frame, "role": "player", "team": "home",
                         "x": presser_x, "y": 34.0, "tx": presser_x, "camera_state": "wide"})
    df = pd.DataFrame(rows)
    poss = pd.DataFrame({
        "frame": [1, 2, 3],
        "near": ["away", "away", "away"],
        "holder": ["away", "away", "away"],
        "owner": ["away", "away", "away"],
    })

    result = pressing_stats(df, poss)
    assert result["home"]["press_eligible_samples"] == 3
    assert result["home"]["pressing_samples"] == 1
    assert result["home"]["press_share"] == 0.333
    assert result["away"]["press_share"] is None
