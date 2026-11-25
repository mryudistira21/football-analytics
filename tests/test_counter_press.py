"""Counter-press: presser counting and regain detection on a synthetic turnover."""
import numpy as np
import pandas as pd

from scripts.team_report import counter_press_events, counter_press_agg, possession_frames


def _rows(seq):
    """seq: list of (ball_holder_side, [extra home players xy]) at 0.1 s steps; ball at x=50."""
    out = []
    for i, (holder, extras) in enumerate(seq):
        f = 100000 + i * 3
        t = i * 0.1
        bx, by = 50.0, 34.0
        out.append(dict(frame=f, video_seconds=t, period=1, track_id=0, role="ball", team="unknown",
                        x=bx, y=by, camera_state="wide"))
        # the holder sits on the ball
        out.append(dict(frame=f, video_seconds=t, period=1, track_id=1 if holder == "home" else 2,
                        role="player", team=holder, x=bx, y=by, camera_state="wide"))
        # a second player of the other team somewhere
        other = "away" if holder == "home" else "home"
        out.append(dict(frame=f, video_seconds=t, period=1, track_id=11 if other == "home" else 12,
                        role="player", team=other, x=10.0, y=10.0, camera_state="wide"))
        for j, (x, y) in enumerate(extras):
            out.append(dict(frame=f, video_seconds=t, period=1, track_id=20 + j, role="player",
                            team="home", x=x, y=y, camera_state="wide"))
    return pd.DataFrame(out)


def test_turnover_press_and_regain():
    # home holds, loses to away (3+ samples to confirm), presses with 2 players, wins it back
    seq = [("home", [])] * 6
    seq += [("away", [(51, 34), (49, 35)])] * 8    # away has it; two home players within 5 m of the ball
    seq += [("home", [])] * 6                       # home regains within 5 s
    ev = counter_press_events(_rows(seq), possession_frames(_rows(seq)))
    home_loss = ev[ev.side == "home"]
    assert len(home_loss) >= 1
    r = home_loss.iloc[0]
    assert r["pressers"] >= 2            # the two chasing players were counted
    assert bool(r["regained"]) is True   # home owned it again within 5 s
    agg = counter_press_agg(ev, "home")
    assert agg["counter_press_losses"] >= 1 and agg["counter_press_regain_share"] == 1.0


def test_no_regain_when_opponent_keeps_it():
    seq = [("home", [])] * 6 + [("away", [])] * 40   # home loses it and never gets it back
    ev = counter_press_events(_rows(seq), possession_frames(_rows(seq)))
    home_loss = ev[ev.side == "home"]
    assert len(home_loss) >= 1 and bool(home_loss.iloc[0]["regained"]) is False
