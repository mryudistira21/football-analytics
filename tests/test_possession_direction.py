"""Possession timeline (hysteresis), attack-direction inference and goalkeeper teams."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.team_report import fix_goalkeepers, infer_direction, possession_frames

REPO = Path(__file__).resolve().parents[1]


def _holders(seq, period=1, t0=0.0, camera=None):
    """seq: list of 'home' / 'away' / None (who holds the ball at each 0.1 s sample)."""
    rows = []
    for i, h in enumerate(seq):
        f = int(period * 100000 + i * 3)
        t = t0 + i * 0.1
        cam = camera[i] if camera else "wide"
        rows.append(dict(frame=f, video_seconds=t, period=period, track_id=0, role="ball", team="unknown",
                         x=50.0, y=34.0, camera_state=cam))
        rows.append(dict(frame=f, video_seconds=t, period=period, track_id=1, role="player", team="home",
                         x=50.5 if h == "home" else 40.0, y=34.0, camera_state=cam))
        rows.append(dict(frame=f, video_seconds=t, period=period, track_id=2, role="player", team="away",
                         x=50.5 if h == "away" else 60.0, y=34.0 if h == "away" else 20.0, camera_state=cam))
    return pd.DataFrame(rows)


def test_flicker_does_not_flip_possession():
    seq = ["home"] * 10 + ["away", "away"] + ["home"] * 10      # 2-sample blip
    p = possession_frames(_holders(seq))
    assert set(p.owner) == {"home"}


def test_three_samples_win_the_ball():
    seq = ["home"] * 10 + ["away"] * 10
    owner = list(possession_frames(_holders(seq)).owner)
    assert owner[:12] == ["home"] * 12           # away needs 3 samples in a row
    assert owner[12:] == ["away"] * 8


def test_loose_ball_resets_after_five_seconds():
    seq = ["home"] * 5 + [None] * 60 + ["away"]
    owner = list(possession_frames(_holders(seq)).owner)
    assert owner[10] == "home"                    # still home's ball 0.5 s later
    assert pd.isna(owner[60])                     # > 5 s without a holder
    assert owner[-1] == "away"                    # nobody owned it -> first holder takes it


def test_camera_cut_breaks_the_timeline():
    seq = ["home"] * 5 + [None] * 60 + [None]
    cam = ["wide"] * 5 + ["close"] * 60 + ["wide"]
    owner = list(possession_frames(_holders(seq, camera=cam)).owner)
    assert pd.isna(owner[-1])
    assert all(pd.isna(o) for o in owner[5:65])    # replays/close-ups are never possession time


def _match(home_right_1h, seconds=60, rng=np.random.default_rng(1)):
    """Kick-off: each team in its own half; then home has the ball and pushes forward half the time."""
    rows = []
    for period in (1, 2):
        hr = home_right_1h if period == 1 else not home_right_1h
        t0 = 1000.0 * period
        for i in range(seconds * 10):
            t = t0 + i * 0.1
            f = int(t * 30)
            home_ball = (i // 50) % 2 == 0 and i >= 100
            for k in range(8):
                for side in ("home", "away"):
                    attacks_right = hr if side == "home" else not hr
                    if i < 100:      # kick-off phase: own half
                        tx = 35 + rng.normal(0, 4)
                    else:
                        has = home_ball == (side == "home")
                        tx = (65 if has else 40) + rng.normal(0, 4)
                    x = tx if attacks_right else 105 - tx
                    rows.append(dict(frame=f, video_seconds=t, period=period, track_id=k + (50 if side == "away" else 0),
                                     role="player", team=side, x=x, y=10 + 6 * k, camera_state="wide"))
            if i >= 100:
                holder = 0 if home_ball else 50
                hx = next(r["x"] for r in rows[-16:] if r["track_id"] == holder)
                rows.append(dict(frame=f, video_seconds=t, period=period, track_id=0, role="ball", team="unknown",
                                 x=hx + 0.3, y=10.0, camera_state="wide"))
            # one goalkeeper per team, labelled at random
            for side, gx_right in (("home", 3.0), ("away", 102.0)):
                defends_left = (hr if side == "home" else not hr)
                gx = 3.0 if defends_left else 102.0
                rows.append(dict(frame=f, video_seconds=t, period=period, track_id=900 + (side == "away"),
                                 role="goalkeeper", team=rng.choice(["home", "away", "unknown"]), x=gx, y=34.0,
                                 camera_state="wide"))
    return pd.DataFrame(rows)


def test_direction_inferred_both_ways():
    for hr1, want in ((True, "R"), (False, "L")):
        res = infer_direction(_match(hr1))
        assert res["home_dir_1h"] == want, res
        assert res["confidence"] == "high"
        assert {v["signal"] for v in res["votes"]} == {"kickoff", "possession_shift"}


def test_goalkeepers_get_the_team_of_their_goal():
    df = _match(True, seconds=12)
    fixed, changed = fix_goalkeepers(df, {1: True, 2: False})
    gk = fixed[fixed.role == "goalkeeper"]
    assert changed > 0
    for (tid, p), g in gk.groupby(["track_id", "period"]):
        x = g.x.iloc[0]
        home_defends_left = p == 1
        want = "home" if (x < 52.5) == home_defends_left else "away"
        assert set(g.team) == {want}


def test_resolve_direction_script_rewrites_csv(tmp_path):
    df = _match(False, seconds=15)
    df.insert(0, "match_id", "m")
    df["confidence"] = 0.9
    csv_path = tmp_path / "m_tracking.csv"
    df.to_csv(csv_path, index=False)
    csv_path.with_suffix(".meta.json").write_text(json.dumps({"match_id": "m"}))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"matches": [{"match_id": "m", "home": "H", "away": "A", "home_dir_1h": None}]}))
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "resolve_direction.py"), "--input", str(csv_path),
                          "--config", str(cfg), "--match", "m"], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "L"
    res = json.loads((tmp_path / "m_direction.json").read_text())
    assert res["source"] == "inferred" and res["goalkeeper_rows_relabelled"] > 0
    meta = json.loads(csv_path.with_suffix(".meta.json").read_text())
    assert meta["home_attacks_right_by_period"] == {"1": False, "2": True}
    back = pd.read_csv(csv_path)
    assert len(back) == len(df) and list(back.columns) == list(df.columns)
    gk = back[(back.role == "goalkeeper") & (back.period == 1)]
    assert set(gk[gk.x < 52.5].team) == {"away"}       # home attacks left -> defends the right goal
    # with a contradicting config the direction is kept but a warning is printed
    cfg.write_text(json.dumps({"matches": [{"match_id": "m", "home": "H", "away": "A", "home_dir_1h": "R"}]}))
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "resolve_direction.py"), "--input", str(csv_path),
                          "--config", str(cfg), "--match", "m"], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "R" and "WARNING" in out.stderr


def _delayed_restart_match(home_right_1h=True):
    """The configured period begins in pre-roll; centre restarts occur 30 s later."""
    rows = []
    for period in (1, 2):
        hr = home_right_1h if period == 1 else not home_right_1h
        t0 = 1000.0 * period
        # Pre-roll: both teams are visible, but clustered around halfway and no ball is tracked.
        for i in range(200):
            t = t0 + i * 0.1
            f = period * 100000 + i * 3
            for k in range(8):
                for side in ("home", "away"):
                    rows.append(dict(frame=f, video_seconds=t, period=period,
                                     track_id=k + (50 if side == "away" else 0),
                                     role="player", team=side, x=50 + (k - 3.5) * 0.3,
                                     y=10 + 6 * k, camera_state="wide"))
        # A centre restart: the ball is at centre while both teams take their own halves.
        for j in range(40):
            t = t0 + 30.0 + j * 0.1
            f = period * 100000 + (200 + j) * 3
            for k in range(8):
                for side in ("home", "away"):
                    stands_left = (side == "home" and hr) or (side == "away" and not hr)
                    x = (25.0 + k * 0.4) if stands_left else (80.0 - k * 0.4)
                    rows.append(dict(frame=f, video_seconds=t, period=period,
                                     track_id=k + (50 if side == "away" else 0),
                                     role="player", team=side, x=x,
                                     y=10 + 6 * k, camera_state="wide"))
            rows.append(dict(frame=f, video_seconds=t, period=period, track_id=0,
                             role="ball", team="unknown", x=52.5, y=34.0,
                             camera_state="wide"))
    return pd.DataFrame(rows)


def test_direction_falls_back_to_delayed_centre_restarts_when_opening_window_is_preroll():
    res = infer_direction(_delayed_restart_match(True))
    assert res["home_dir_1h"] == "R", res
    assert res["confidence"] == "high"
    assert [v["signal"] for v in res["votes"]] == ["center_restart", "center_restart"]
