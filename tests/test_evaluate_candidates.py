import csv
import subprocess
import sys
from pathlib import Path

from scripts.evaluate_candidates import evaluate, load_cands, load_truth

REPO = Path(__file__).resolve().parents[1]


def test_precision_recall_and_window():
    truth = load_truth([
        {"event": "pass", "side": "home", "video_seconds": "100", "outcome": "complete", "source": "manual"},
        {"event": "pass", "side": "home", "video_seconds": "110", "outcome": "incomplete", "source": "manual"},
        {"event": "tackle", "side": "away", "video_seconds": "120", "outcome": "won", "source": "manual"},
        {"event": "pass", "side": "home", "video_seconds": "130", "outcome": "complete", "source": "trackb"},
    ], include_accepted=False)
    cands = load_cands([
        {"cand_id": "c1", "type": "pass", "side": "home", "t": "100.8", "outcome": "ok", "priority": "1"},
        {"cand_id": "c2", "type": "pass", "side": "home", "t": "111.5", "outcome": "ok", "priority": "3"},
        {"cand_id": "c3", "type": "tackle", "side": "home", "t": "120.2", "outcome": "won", "priority": "2"},
        {"cand_id": "c4", "type": "recovery", "side": "away", "t": "105", "outcome": "ok", "priority": "2"},
        {"cand_id": "c5", "type": "pass", "side": "home", "t": "500", "outcome": "ok", "priority": "1"},
    ])
    res = evaluate(truth, cands, 100, 120, 2.0)
    p = res["per_type"]["pass"]
    assert res["tagged_events"] == 3                       # the trackb event is not truth
    assert p["tagged"] == 2 and p["matched"] == 2 and p["recall"] == 1.0
    assert p["recall_default_queue"] == 0.5                # c2 is priority 3 (hidden by default)
    assert p["pass_outcome_agreement"] == 0.5
    assert res["per_type"]["tackle"]["recall"] == 0.0 and res["wrong_team_matches"] == 1
    assert res["per_type"]["all"]["candidates"] == 4      # c5 is outside the window


def test_cli(tmp_path):
    ev, ca = tmp_path / "e.csv", tmp_path / "c.csv"
    with ev.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "event", "side", "video_seconds", "outcome", "source"])
        w.writerow(["e1", "pass", "home", "10", "complete", "manual"])
    with ca.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cand_id", "type", "side", "t", "outcome", "priority"])
        w.writerow(["c1", "pass", "home", "10.5", "ok", "1"])
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "evaluate_candidates.py"), "--candidates", str(ca),
                          "--events", str(ev), "--output", str(tmp_path / "r.json")],
                         capture_output=True, text=True, check=True)
    assert "| pass | 1 | 1 | 100%" in out.stdout
