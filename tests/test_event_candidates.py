import pandas as pd

from scripts.build_event_candidates import build


def _rows(frames):
    out = []
    for f, (bx, by, holders) in enumerate(frames):
        t = f * 0.1
        out.append({"frame": f * 3, "video_seconds": t, "period": 1, "track_id": 0, "role": "ball",
                    "team": "unknown", "x": bx, "y": by, "camera_state": "wide"})
        for tid, side, x, y in holders:
            out.append({"frame": f * 3, "video_seconds": t, "period": 1, "track_id": tid, "role": "player",
                        "team": side, "x": x, "y": y, "camera_state": "wide"})
    return pd.DataFrame(out)


def test_pass_then_turnover_in_team_frame():
    seq = []
    seq += [(30.0, 34.0, [(1, "home", 30.5, 34.0), (2, "home", 45.0, 30.0), (3, "away", 60.0, 34.0)])] * 5
    seq += [(38.0, 32.0, [(1, "home", 31.0, 34.0), (2, "home", 45.0, 30.0), (3, "away", 60.0, 34.0)])] * 3
    seq += [(45.0, 30.5, [(1, "home", 31.0, 34.0), (2, "home", 45.3, 30.5), (3, "away", 60.0, 34.0)])] * 5
    seq += [(52.0, 32.0, [(1, "home", 31.0, 34.0), (2, "home", 46.0, 30.5), (3, "away", 60.0, 34.0)])] * 3
    seq += [(59.8, 34.0, [(1, "home", 31.0, 34.0), (2, "home", 46.0, 30.5), (3, "away", 60.0, 34.0)])] * 5
    cands = build(_rows(seq), {1: False, 2: True})   # home attacks right-to-left in period 1
    kinds = [(c["type"], c["side"], c["outcome"]) for c in cands]
    assert ("pass", "home", "ok") in kinds
    assert ("pass", "home", "fail") in kinds
    assert ("interception", "away", "ok") in kinds
    ok_pass = next(c for c in cands if c["type"] == "pass" and c["outcome"] == "ok")
    # home attacks left in P1 -> team frame mirrors the camera frame
    assert abs(ok_pass["x"] - (105 - 30.0)) < 0.2 and abs(ok_pass["ex"] - (105 - 45.0)) < 0.2
    assert ok_pass["track_id"] == 1 and ok_pass["track_to"] == 2
    inter = next(c for c in cands if c["type"] == "interception")
    assert abs(inter["x"] - 59.8) < 0.2   # away attacks right in P1: camera frame
    assert [c["cand_id"] for c in cands] == sorted(c["cand_id"] for c in cands)


def test_no_candidates_without_a_ball():
    df = _rows([(float("nan"), float("nan"), [(1, "home", 30.0, 34.0)])] * 5).dropna(subset=["x"])
    assert build(df, {1: True, 2: False}) == []


def test_candidates_pick_up_track_map_numbers(tmp_path):
    import csv as _csv
    import json
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    # tracking: one home pass from track 5 to track 6
    tracking = tmp_path / "m_tracking.csv"
    with tracking.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["match_id", "frame", "video_seconds", "period", "track_id", "role", "team",
                    "x", "y", "camera_state"])
        seq = [(30, 34, [(5, "home", 30, 34), (6, "home", 50, 30), (9, "away", 70, 34)])] * 5 \
            + [(50, 30, [(5, "home", 31, 34), (6, "home", 50, 30), (9, "away", 70, 34)])] * 5
        for f, (bx, by, hs) in enumerate(seq):
            w.writerow(["m", f * 3, f * 0.1, 1, 0, "ball", "unknown", bx, by, "wide"])
            for tid, side, x, y in hs:
                w.writerow(["m", f * 3, f * 0.1, 1, tid, "player", side, x, y, "wide"])
    tmap = tmp_path / "m_track_map.csv"
    tmap.write_text("track_id,team,number,votes,reads,confidence,attempts\n"
                    "5,home,10,4,5,0.8,5\n6,home,8,1,3,0.33,3\n", encoding="utf-8")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"matches": [{"match_id": "m", "home": "H", "away": "A", "home_dir_1h": "R"}]}))
    out = tmp_path / "m_candidates.csv"
    r = subprocess.run([sys.executable, str(repo / "scripts" / "build_event_candidates.py"),
                        "--input", str(tracking), "--config", str(cfg), "--match", "m", "--output", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = list(_csv.DictReader(out.open()))
    p = next(r for r in rows if r["type"] == "pass")
    assert p["track_player"] == "10"        # track 5, confident -> suggested
    assert p["track_to_player"] == ""        # track 6 below the confidence/vote floor -> no suggestion
