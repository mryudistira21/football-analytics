"""Physical metrics: distance, speed thresholds, team totals, per-player aggregation."""
import numpy as np
import pandas as pd

from scripts.player_physical import build, per_track, team_totals, _track_metric, HI_MS, SPRINT_MS


def _straight_track(speed_ms, seconds=12.0, hz=10.0):
    n = int(seconds * hz)
    t = np.arange(n) / hz
    x = speed_ms * t          # straight line along x at constant speed
    y = np.full(n, 34.0)
    return t, x, y


def test_distance_and_speed_of_a_known_run():
    t, x, y = _straight_track(3.0, seconds=12)
    m = _track_metric(t, x, y)
    assert abs(m["distance_m"] - 36.0) < 4.0          # 3 m/s * 12 s, minus smoothing edge
    assert abs(m["top_speed_kmh"] - 3.0 * 3.6) < 1.5
    assert m["hi_distance_m"] == 0.0 and m["sprint_distance_m"] == 0.0   # 3 m/s < HI


def test_sprint_distance_counts_only_fast_metres():
    t, x, y = _straight_track(SPRINT_MS + 0.5, seconds=10)   # above the sprint threshold throughout
    m = _track_metric(t, x, y)
    assert m["sprint_distance_m"] > 0.5 * m["distance_m"]
    assert m["hi_distance_m"] >= m["sprint_distance_m"]      # sprint speed is also high-intensity


def test_jitter_does_not_invent_distance():
    rng = np.random.default_rng(0)
    n = 120
    t = np.arange(n) / 10.0
    x = 20.0 + rng.normal(0, 0.4, n)    # standing still, 0.4 m position noise
    y = 34.0 + rng.normal(0, 0.4, n)
    m = _track_metric(t, x, y)
    assert m["distance_m"] < 8.0        # smoothing keeps jitter from reading as metres run
    assert m["sprint_distance_m"] == 0.0


def _match_df():
    rows = []
    for tid, team, speed in [(1, "home", 3.0), (2, "home", 7.5), (3, "away", 2.0)]:
        t, x, y = _straight_track(speed, seconds=12)
        for ti, xi, yi in zip(t, x, y):
            rows.append(dict(video_seconds=ti, period=1, track_id=tid, role="player",
                             team=team, x=xi % 105, y=yi, camera_state="wide"))
    return pd.DataFrame(rows)


def test_team_totals_sum_tracks():
    pt = per_track(_match_df())
    tt = team_totals(pt)
    assert tt["home"]["distance_km"] > tt["away"]["distance_km"]   # home has two runners incl. a sprinter
    assert tt["home"]["sprint_distance_km"] > 0
    assert tt["away"]["sprint_distance_km"] == 0


def test_by_player_aggregates_with_track_map(tmp_path):
    df = _match_df()
    csv = tmp_path / "m_tracking.csv"
    df.assign(match_id="m").to_csv(csv, index=False)
    # map both home tracks to shirt #9, away track to #4
    tm = tmp_path / "m_track_map.csv"
    pd.DataFrame([{"track_id": 1, "number": "9", "confidence": 0.9, "votes": 5},
                  {"track_id": 2, "number": "9", "confidence": 0.9, "votes": 5},
                  {"track_id": 3, "number": "4", "confidence": 0.9, "votes": 5}]).to_csv(tm, index=False)
    res = build(csv, tm)
    bp = res["by_player"]
    assert not bp.empty
    nine = bp[(bp.team == "home") & (bp.number == "9")].iloc[0]
    assert nine["tracks"] == 2
    # #9's distance is the sum of both home tracks
    pt = res["per_track"]
    assert abs(nine["distance_m"] - pt[pt.team == "home"].distance_m.sum()) < 0.1
