#!/usr/bin/env python3
"""Attack direction + goalkeeper teams for a combined Track B tracking CSV.

1. Infers which way the HOME team attacks in the 1st half from the tracking itself
   (kick-off positions and the possession shift — see team_report.infer_direction).
2. Decides the direction to use: the config's home_dir_1h when set, otherwise the
   inferred one (high or medium confidence). Warns when a confident inference
   disagrees with the config.
3. With a direction, relabels every goalkeeper track to the team defending the half
   it stands in (keeper kits make colour clustering unreliable) and rewrites the CSV
   in place (streamed, low RAM). Idempotent.
4. Writes <match>_direction.json next to the CSV and records the result in the
   CSV's .meta.json (home_attacks_right_by_period, direction).

    python scripts/resolve_direction.py --input runs/<id>/<id>_tracking.csv \
        --config matches/persiba_2026-27.json --match <id>
Prints the direction used (R, L or ?) on the last line.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:  # run as a script (scripts/ on sys.path) or imported as scripts.resolve_direction
    from team_report import PL, USECOLS, infer_direction, load_match, possession_frames
except ImportError:  # pragma: no cover
    from scripts.team_report import PL, USECOLS, infer_direction, load_match, possession_frames


def goalkeeper_sides(df: pd.DataFrame, home_dir: str) -> dict[tuple[str, str], str]:
    """(track_id, period) -> team for every goalkeeper track, from its median x."""
    hr = {1: home_dir == "R", 2: home_dir != "R"}
    g = df[df.role == "goalkeeper"].dropna(subset=["x"]).groupby(["track_id", "period"]).x.median()
    out = {}
    for (tid, p), gx in g.items():
        if int(p) not in hr:
            continue
        defends_left = gx < PL / 2
        out[(str(int(tid)), str(int(p)))] = "home" if defends_left == hr[int(p)] else "away"
    return out


def rewrite_goalkeepers(csv_path: Path, sides: dict[tuple[str, str], str]) -> int:
    tmp = csv_path.with_name(csv_path.name + ".partial")
    changed = 0
    with csv_path.open(newline="", encoding="utf-8") as fi, tmp.open("w", newline="", encoding="utf-8") as fo:
        rd = csv.DictReader(fi)
        w = csv.DictWriter(fo, fieldnames=rd.fieldnames)
        w.writeheader()
        for r in rd:
            if r.get("role") == "goalkeeper":
                tid = r.get("track_id", "")
                tid = str(int(float(tid))) if tid not in ("", None) else tid
                new = sides.get((tid, str(int(float(r.get("period") or 0)))))
                if new and new != r.get("team"):
                    r["team"] = new
                    changed += 1
            w.writerow(r)
    tmp.replace(csv_path)
    return changed


def resolve(csv_path: Path, match: dict[str, Any], fix_goalkeepers: bool = True) -> dict[str, Any]:
    df = pd.read_csv(csv_path, usecols=USECOLS, dtype={"x": "float32", "y": "float32"})
    for c in ("role", "team", "camera_state"):
        df[c] = df[c].astype(str)
    inferred = infer_direction(df, possession_frames(df))
    cfg_dir = match.get("home_dir_1h") if match.get("home_dir_1h") in ("R", "L") else None
    if cfg_dir:
        used, source = cfg_dir, "config"
    elif inferred["confidence"] in ("high", "medium"):
        used, source = inferred["home_dir_1h"], "inferred"
    else:
        used, source = None, "unknown"
    res: dict[str, Any] = {
        "match_id": match["match_id"], "used": used, "source": source, "config": cfg_dir,
        "inferred": inferred["home_dir_1h"], "confidence": inferred["confidence"], "votes": inferred["votes"],
        "goalkeeper_rows_relabelled": 0, "warning": None,
    }
    if cfg_dir and inferred["confidence"] == "high" and inferred["home_dir_1h"] != cfg_dir:
        res["warning"] = (f"Config says home_dir_1h={cfg_dir} but the tracking says {inferred['home_dir_1h']} "
                          f"({len(inferred['votes'])} agreeing votes: kick-off positions and possession shift). "
                          f"Check the video and fix the config, then run --post-only again.")
    del df  # free before the streamed rewrite
    if used and fix_goalkeepers:
        df = pd.read_csv(csv_path, usecols=["track_id", "period", "role", "x"], dtype={"x": "float32"})
        res["goalkeeper_rows_relabelled"] = rewrite_goalkeepers(csv_path, goalkeeper_sides(df, used))
    meta_path = csv_path.with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if used:
            meta["home_attacks_right_by_period"] = {"1": used == "R", "2": used != "R"}
        meta["direction"] = {k: res[k] for k in ("used", "source", "inferred", "confidence",
                                                  "goalkeeper_rows_relabelled")}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Infer attack direction and fix goalkeeper teams.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--match", required=True)
    ap.add_argument("--output", type=Path, help="default: <input dir>/<match>_direction.json")
    ap.add_argument("--no-goalkeeper-fix", action="store_true")
    a = ap.parse_args()
    match = load_match(a.config, a.match)
    res = resolve(a.input, match, not a.no_goalkeeper_fix)
    out = a.output or a.input.with_name(f"{a.match}_direction.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    votes = ", ".join(f"{v['signal']} P{v['period']} {'R' if v['home_right_1h'] else 'L'}" for v in res["votes"]) or "none"
    print(f"Direction: data says {res['inferred'] or '?'} ({res['confidence']}; {votes}); "
          f"using {res['used'] or '?'} ({res['source']}); goalkeeper rows relabelled: {res['goalkeeper_rows_relabelled']}")
    if res["warning"]:
        print("WARNING:", res["warning"], file=sys.stderr)
    print(res["used"] or "?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
