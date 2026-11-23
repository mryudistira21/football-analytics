import csv
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_save_run_folder_and_tagger_files(tmp_path):
    mid = "m1"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"season": "S", "matches": [
        {"match_id": mid, "home": "H", "away": "A", "score": [1, 0], "periods": {"1": ["0:00", "45:00"]}}]}))
    run = tmp_path / "root" / "runs" / mid
    (run / "report").mkdir(parents=True)
    with (run / f"{mid}_tracking.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["match_id", "frame", "video_seconds", "period", "track_id", "role", "team", "x", "y",
                    "confidence", "camera_state"])
        w.writerow([mid, 3, 0.1, 1, 5, "player", "home", 10.5, 20, 0.9, "wide"])
    with (run / f"{mid}_candidates.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cand_id", "period", "t", "t_end", "side", "type", "outcome", "track_id", "track_to",
                    "x", "y", "ex", "ey", "confidence", "reason"])
        w.writerow(["c00001", 1, 1.2, 2.0, "home", "pass", "ok", "5", "6", 10, 20, 30, 25, "ok", "r"])
        w.writerow(["c00002", 1, 3.0, 3.0, "away", "recovery", "ok", "7", "", 50, 30, "", "", "review", "r"])
    (run / "report" / f"{mid}_team_report.json").write_text(json.dumps({"periods": {"1": {
        "quality": {"wide_share": 0.8}, "territory_home_view": {"middle": 0.5},
        "home": {"block_height_m": 50.0, "heatmap": [[0.1]]}, "away": {"block_height_m": 30.0}}}}))
    inbox = tmp_path / "Downloads"
    inbox.mkdir()
    (inbox / f"{mid}_review.json").write_text(json.dumps({"match_id": mid, "track_map": {"5": {"team": 0, "p": "10"}},
        "candidates": [{"cand_id": "c00001", "status": "accepted", "event_id": "e1"},
                       {"cand_id": "c00002", "status": "rejected", "event_id": None}]}))
    (inbox / f"{mid}_events.csv").write_text(
        "match_id,id,half,event,outcome,player,receiver,x,y,end_x,end_y,source,candidate_id,track_id\n"
        f"{mid},e1,1,pass,complete,10,6,10,20,30,25,trackb,c00001,5\n"
        f"{mid},e2,1,shot,goal,9,,95,34,,,manual,,\n", encoding="utf-8")
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "save_to_db.py"), "--config", str(cfg),
                        "--root", str(tmp_path / "root"), "--match", mid, "--inbox", str(inbox)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    con = sqlite3.connect(tmp_path / "root" / "football.db")
    q = lambda s: con.execute(s).fetchall()
    assert q("select home_score, away_score from matches") == [(1, 0)]
    assert q("select count(*) from tracking") == [(1,)]
    assert dict(q("select cand_id, review_status from candidates")) == {"c00001": "accepted", "c00002": "rejected"}
    assert q("select track_id, side, player from track_map") == [("5", "home", "10")]
    assert sorted(q("select event_id, source, player from events")) == [("e1", "trackb", "10"), ("e2", "manual", "9")]
    assert q("select value from team_metrics where side='home' and metric='block_height_m'") == [(50.0,)]
    assert q("select count(*) from team_heatmaps") == [(1,)]
    # re-running replaces, never duplicates
    subprocess.run([sys.executable, str(REPO / "scripts" / "save_to_db.py"), "--config", str(cfg),
                    "--root", str(tmp_path / "root"), "--match", mid, "--inbox", str(inbox)], check=True,
                   capture_output=True)
    assert q("select count(*) from events") == [(2,)] and q("select count(*) from tracking") == [(1,)]


def test_old_database_gets_new_candidate_columns(tmp_path):
    import sqlite3
    from scripts.save_to_db import migrate
    con = sqlite3.connect(tmp_path / "old.db")
    con.execute("CREATE TABLE candidates (match_id TEXT, cand_id TEXT, review_status TEXT)")
    con.execute("INSERT INTO candidates VALUES ('m', 'c1', 'pending')")
    migrate(con)
    migrate(con)   # idempotent
    cols = [r[1] for r in con.execute("PRAGMA table_info(candidates)")]
    assert cols[-2:] == ["priority", "linked"]
    assert con.execute("SELECT cand_id FROM candidates").fetchone() == ("c1",)
