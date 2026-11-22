#!/usr/bin/env python3
"""Per-half team report from a Track B tracking CSV — no manual tagging needed.

Everything here is a *proxy* computed from broadcast tracking, so each number
comes with a data-quality block (wide-camera share, known-team share, ball
visibility). Use it for trends and comparisons between halves and matches,
not as official stats.

Metrics per period (1, 2, [3, 4], all) and team:
  ball_proximity_share  share of ball-near moments where this team's player is
                        nearest the ball (<= 3 m) — a possession proxy
  territory             ball location split into thirds, seen from the home team
  block_height_m        median mean-x of the team's outfield players (0 = own goal line)
  def_line_m            median x of the deepest outfield player (after dropping the single
                        deepest one per frame, which is often a keeper detected as 'player')
  width_m / length_m    median spread of the outfield players
  in/out of possession  block height split by the (smoothed) proximity owner
  heatmap               21 x 14 grid (5 m cells), team attacking left -> right

    python scripts/team_report.py --input runs/<id>/<id>_tracking.csv \
        --config matches/persiba_2026-27.json --match <id> --output-dir runs/<id>/report
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PL, PW = 105.0, 68.0
GRID = (21, 14)
NEAR_BALL_M = 3.0
MIN_OUTFIELD = 7          # a frame counts for team shape only if this many outfield players are mapped
POSSESSION_HOLD_S = 3.0   # carry the last proximity owner forward this long
USECOLS = ["frame", "video_seconds", "period", "track_id", "role", "team", "x", "y", "camera_state"]


def load_match(config: Path, match_id: str) -> dict[str, Any]:
    cfg = json.loads(config.read_text(encoding="utf-8"))
    for m in cfg.get("matches", []):
        if m.get("match_id") == match_id:
            return m
    raise SystemExit(f"{match_id} not in {config}")


def home_right(match: dict[str, Any], periods: list[int]) -> dict[int, bool]:
    d = match.get("home_dir_1h")
    if d not in ("R", "L"):
        raise SystemExit("Set home_dir_1h (R or L) for this match in the config first.")
    out = {1: d == "R", 2: d != "R"}
    if any(p >= 3 for p in periods):
        et = match.get("home_dir_et1")
        if et not in ("R", "L"):
            raise SystemExit("Extra time present: set home_dir_et1 (R or L) in the config.")
        out.update({3: et == "R", 4: et != "R"})
    return out


def r1(v: Any, nd: int = 1) -> float | None:
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), nd)


def possession_frames(df: pd.DataFrame, interval: float) -> pd.DataFrame:
    """Per sampled frame: nearest known-team player to the ball within NEAR_BALL_M."""
    xy = df.dropna(subset=["x", "y"])
    ball = xy[xy.role == "ball"][["frame", "x", "y"]].drop_duplicates("frame")
    ppl = xy[xy.role.isin(["player", "goalkeeper"]) & xy.team.isin(["home", "away"])][["frame", "team", "x", "y"]]
    m = ppl.merge(ball, on="frame", suffixes=("", "_b"))
    if m.empty:
        return pd.DataFrame(columns=["frame", "near", "owner"])
    m["d"] = np.hypot(m.x - m.x_b, m.y - m.y_b)
    best = m.loc[m.groupby("frame").d.idxmin()]
    best = best[best.d <= NEAR_BALL_M][["frame", "team"]].rename(columns={"team": "near"})
    frames = df[["frame"]].drop_duplicates().sort_values("frame")
    out = frames.merge(best, on="frame", how="left")
    hold = max(1, int(round(POSSESSION_HOLD_S / interval)))
    out["owner"] = out["near"].ffill(limit=hold)
    return out


def team_frame(df: pd.DataFrame, hr: dict[int, bool]) -> pd.DataFrame:
    """Add tx, ty: coordinates with each team attacking left -> right (ball: home frame)."""
    home_r = df.period.map(hr).astype(bool)
    flip = np.where(df.team == "away", home_r, ~home_r)  # away attacks right when home attacks left
    flip = np.where(df.role == "ball", ~home_r, flip)
    df = df.copy()
    df["tx"] = np.where(flip, PL - df.x, df.x)
    df["ty"] = np.where(flip, PW - df.y, df.y)
    return df


def shape_stats(out: pd.DataFrame) -> dict[str, Any]:
    """out = one team's outfield rows (team frame) in qualifying frames."""
    if out.empty:
        return {"frames": 0}
    # Goalkeepers are often detected as 'player' and coloured into a team; drop the
    # single deepest player per frame so the keeper cannot pose as the back line.
    deepest = out.groupby("frame").tx.idxmin()
    out = out.drop(index=deepest.to_numpy())
    g = out.groupby("frame")
    per = pd.DataFrame({
        "mean_x": g.tx.mean(), "min_x": g.tx.min(),
        "width": g.ty.max() - g.ty.min(), "length": g.tx.max() - g.tx.min(),
    })
    return {
        "frames": int(len(per)),
        "block_height_m": r1(per.mean_x.median()),
        "def_line_m": r1(per.min_x.median()),
        "width_m": r1(per.width.median()),
        "length_m": r1(per.length.median()),
    }


def heat(rows: pd.DataFrame) -> list[list[float]]:
    if rows.empty:
        return [[0.0] * GRID[1] for _ in range(GRID[0])]
    h, _, _ = np.histogram2d(rows.tx, rows.ty, bins=GRID, range=[[0, PL], [0, PW]])
    mx = h.max() or 1.0
    return [[round(float(v) / mx, 3) for v in col] for col in h]


def period_block(df: pd.DataFrame, poss: pd.DataFrame) -> dict[str, Any]:
    wide = df[df.camera_state == "wide"]
    mapped = wide.dropna(subset=["x", "y"])
    frames_all = df.frame.nunique()
    frames_wide = wide.frame.nunique()
    players = df[df.role == "player"]
    known = players.team.isin(["home", "away"]).mean() if len(players) else 0.0
    ball_frames = mapped[mapped.role == "ball"].frame.nunique()

    pf = poss[poss.frame.isin(mapped.frame.unique())]
    near = pf.near.value_counts()
    near_tot = int(near.get("home", 0) + near.get("away", 0))

    ball = mapped[mapped.role == "ball"]
    thirds = [0.0, 0.0, 0.0]
    if len(ball):
        b = pd.cut(ball.tx, [-1, 35, 70, PL + 1], labels=False).value_counts(normalize=True)
        thirds = [round(float(b.get(i, 0.0)), 3) for i in range(3)]

    owner = pf.set_index("frame").owner
    res: dict[str, Any] = {
        "quality": {
            "sampled_frames": int(frames_all),
            "wide_share": round(frames_wide / frames_all, 3) if frames_all else 0.0,
            "known_team_share": round(float(known), 3),
            "ball_visible_share": round(ball_frames / frames_wide, 3) if frames_wide else 0.0,
            "ball_near_frames": near_tot,
        },
        "territory_home_view": {"def_third": thirds[0], "middle": thirds[1], "att_third": thirds[2]},
    }
    for side in ("home", "away"):
        o = mapped[(mapped.role == "player") & (mapped.team == side)]
        counts = o.groupby("frame").size()
        good = counts[counts >= MIN_OUTFIELD].index
        o = o[o.frame.isin(good)]
        own = owner.reindex(o.frame).to_numpy()
        res[side] = {
            "ball_proximity_share": round(float(near.get(side, 0)) / near_tot, 3) if near_tot else None,
            **shape_stats(o),
            "block_height_in_possession_m": shape_stats(o[own == side]).get("block_height_m"),
            "block_height_out_of_possession_m": shape_stats(o[(own != side) & pd.notna(own)]).get("block_height_m"),
        }
        res[side]["heatmap"] = heat(mapped[(mapped.role == "player") & (mapped.team == side)])
    return res


def build(csv_path: Path, match: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    df = pd.read_csv(csv_path, usecols=USECOLS,
                     dtype={"x": "float32", "y": "float32", "role": "category", "team": "category",
                            "camera_state": "category", "period": "int8"})
    for c in ("role", "team", "camera_state"):
        df[c] = df[c].astype(str)
    periods = sorted(int(p) for p in df.period.unique())
    hr = home_right(match, periods)
    df = team_frame(df, hr)
    fps = float(meta.get("fps") or 30)
    interval = float(meta.get("frame_stride") or 3) / fps
    poss = possession_frames(df, interval)

    report: dict[str, Any] = {
        "schema": "trackb_team_report_v1",
        "match_id": match["match_id"], "date": match.get("date"), "round": match.get("round"),
        "home": match["home"], "away": match["away"], "score": match.get("score"),
        "home_attacks_right_by_period": {str(k): v for k, v in hr.items()},
        "grid": list(GRID), "pitch": [PL, PW],
        "method": {
            "near_ball_m": NEAR_BALL_M, "min_outfield_players_for_shape": MIN_OUTFIELD,
            "possession_hold_s": POSSESSION_HOLD_S,
            "caveat": "Broadcast-tracking proxies. Only wide-camera frames are used for positions; "
                      "replays/close-ups are excluded. Compare halves and matches, not with official stats.",
        },
        "periods": {},
    }
    for p in periods:
        report["periods"][str(p)] = period_block(df[df.period == p], poss[poss.frame.isin(df[df.period == p].frame)])
    report["periods"]["all"] = period_block(df, poss)
    return report


def markdown(rep: dict[str, Any]) -> str:
    h, a = rep["home"], rep["away"]
    lines = [f"# {h} {rep['score'][0] if rep.get('score') else ''}–{rep['score'][1] if rep.get('score') else ''} {a}",
             f"{rep.get('round') or ''} · {rep.get('date') or ''} · Track B team report (proxies)", ""]
    names = {"1": "1st half", "2": "2nd half", "3": "ET 1", "4": "ET 2", "all": "Full match"}
    for key in [k for k in ("1", "2", "3", "4", "all") if k in rep["periods"]]:
        b = rep["periods"][key]
        q = b["quality"]
        t = b["territory_home_view"]
        lines += [f"## {names[key]}", "",
                  f"Data: wide camera {q['wide_share']:.0%} · team known {q['known_team_share']:.0%} · "
                  f"ball visible {q['ball_visible_share']:.0%} · {q['ball_near_frames']} ball-near samples", "",
                  f"| Metric | {h} | {a} |", "|---|---:|---:|"]
        def v(side: str, k: str, unit: str = "") -> str:
            x = b[side].get(k)
            return "—" if x is None else (f"{x:.0%}" if unit == "%" else f"{x} {unit}".strip())
        for label, k, u in [("Ball-proximity share (possession proxy)", "ball_proximity_share", "%"),
                            ("Block height (median, from own goal)", "block_height_m", "m"),
                            ("  in possession", "block_height_in_possession_m", "m"),
                            ("  out of possession", "block_height_out_of_possession_m", "m"),
                            ("Back line (deepest outfield, keeper-robust)", "def_line_m", "m"),
                            ("Width", "width_m", "m"), ("Length", "length_m", "m")]:
            lines.append(f"| {label} | {v('home', k, u)} | {v('away', k, u)} |")
        lines += ["", f"Ball territory (from {h}'s view): own third {t['def_third']:.0%} · "
                      f"middle {t['middle']:.0%} · attacking third {t['att_third']:.0%}", ""]
    lines.append(rep["method"]["caveat"])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Team report (per half) from a Track B tracking CSV.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--match", required=True)
    ap.add_argument("--metadata", type=Path, help="default: <input>.meta.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    match = load_match(a.config, a.match)
    meta_path = a.metadata or a.input.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    rep = build(a.input, match, meta)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    j = a.output_dir / f"{match['match_id']}_team_report.json"
    md = a.output_dir / f"{match['match_id']}_team_report.md"
    j.write_text(json.dumps(rep, separators=(",", ":")), encoding="utf-8")
    md.write_text(markdown(rep), encoding="utf-8")
    print("wrote", j)
    print("wrote", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
