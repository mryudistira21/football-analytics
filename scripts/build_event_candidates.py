#!/usr/bin/env python3
"""Event candidates from Track B tracking, for review in the Pitchside Tagger.

Track B knows where the ball and the players are, not what happened. This turns
ball possession changes into *proposals* that a person confirms, fixes or rejects
in the Tagger's Review tab:

  pass / cross   ball goes from one player to a team-mate (ok) or an opponent (fail)
  interception   opponent receives a ball that travelled >= 3 m
  tackle         opponent takes the ball within 3 m (outcome 'won' for the taker)
  recovery       a team picks up a loose ball after >= 1.5 s with no holder
  clearance      a long ball forward out of a team's own third that no team-mate receives
  shot           ball leaves a player in the final third and is next seen at the goal

Possession = the nearest player with a known team within 2 m of the ball, on
wide-camera frames only, held for >= 2 samples. Coordinates are written in the
Tagger's team frame (105 x 68 m, the acting team attacking left -> right), so an
accepted candidate becomes an event without conversion.

Output columns (candidates contract v1):
  cand_id, period, t, t_end, side, type, outcome, track_id, track_to,
  x, y, ex, ey, confidence, reason

    python scripts/build_event_candidates.py --input runs/<id>/<id>_tracking.csv \
        --config matches/persiba_2026-27.json --match <id> --output runs/<id>/<id>_candidates.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PL, PW = 105.0, 68.0
HOLD_M = 2.0            # ball within this distance of a player = that player has it
MIN_SAMPLES = 2         # a possession needs at least this many samples
MERGE_GAP_S = 0.4       # same player, gap up to this long = still the same possession
MAX_LINK_S = 4.0        # possessions further apart than this are not linked into one event
LOOSE_S = 1.5           # gap after which the next holder 'recovers' a loose ball
LOOSE_MAX_S = 8.0       # longer gaps are usually the ball being out of view, not a loose ball
PASS_MIN_M = 3.0
CLEAR_MIN_M = 25.0
FIELDS = ["cand_id", "period", "t", "t_end", "side", "type", "outcome", "track_id", "track_to",
          "x", "y", "ex", "ey", "confidence", "reason"]


@dataclass
class Seg:
    track: int
    side: str
    role: str
    period: int
    t0: float
    t1: float
    bx0: float
    by0: float
    bx1: float
    by1: float
    n: int


def load_match(config: Path, match_id: str) -> dict[str, Any]:
    for m in json.loads(config.read_text(encoding="utf-8")).get("matches", []):
        if m.get("match_id") == match_id:
            return m
    raise SystemExit(f"{match_id} not in {config}")


def home_right(match: dict[str, Any]) -> dict[int, bool]:
    d = match.get("home_dir_1h")
    if d not in ("R", "L"):
        raise SystemExit("Set home_dir_1h (R or L) for this match in the config first.")
    out = {1: d == "R", 2: d != "R"}
    et = match.get("home_dir_et1")
    if et in ("R", "L"):
        out.update({3: et == "R", 4: et != "R"})
    return out


def team_frame(side: str, period: int, x: float, y: float, hr: dict[int, bool]) -> tuple[float, float]:
    right = hr.get(period, True) if side == "home" else not hr.get(period, True)
    return (x, y) if right else (PL - x, PW - y)


def holders(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sampled frame that has a ball: nearest known-team player within HOLD_M."""
    w = df[(df.camera_state == "wide")].dropna(subset=["x", "y"])
    ball = w[w.role == "ball"][["frame", "video_seconds", "period", "x", "y"]].drop_duplicates("frame")
    ppl = w[w.role.isin(["player", "goalkeeper"]) & w.team.isin(["home", "away"])][
        ["frame", "track_id", "team", "role", "x", "y"]]
    m = ppl.merge(ball, on="frame", suffixes=("", "_b"))
    if m.empty:
        return pd.DataFrame(columns=["frame", "t", "period", "bx", "by", "track", "side", "role"])
    m["d"] = np.hypot(m.x - m.x_b, m.y - m.y_b)
    best = m.loc[m.groupby("frame").d.idxmin()]
    best = best[best.d <= HOLD_M]
    out = ball.rename(columns={"video_seconds": "t", "x": "bx", "y": "by"}).merge(
        best[["frame", "track_id", "team", "role"]].rename(columns={"track_id": "track", "team": "side"}),
        on="frame", how="left")
    return out.sort_values("t").reset_index(drop=True)


def segments(h: pd.DataFrame) -> list[Seg]:
    segs: list[Seg] = []
    cur: Seg | None = None
    for r in h.itertuples(index=False):
        if pd.isna(r.track):
            continue
        tr, period = int(r.track), int(r.period)
        if cur and cur.track == tr and cur.period == period and r.t - cur.t1 <= MERGE_GAP_S:
            cur.t1, cur.bx1, cur.by1, cur.n = r.t, r.bx, r.by, cur.n + 1
            continue
        if cur and cur.n >= MIN_SAMPLES:
            segs.append(cur)
        cur = Seg(tr, str(r.side), str(r.role), period, r.t, r.t, r.bx, r.by, r.bx, r.by, 1)
    if cur and cur.n >= MIN_SAMPLES:
        segs.append(cur)
    # merge consecutive segments of the same track that were split by a short loss
    merged: list[Seg] = []
    for s in segs:
        if merged and merged[-1].track == s.track and merged[-1].period == s.period and s.t0 - merged[-1].t1 <= 1.0:
            p = merged[-1]
            p.t1, p.bx1, p.by1, p.n = s.t1, s.bx1, s.by1, p.n + s.n
        else:
            merged.append(s)
    return merged


def ball_after(h: pd.DataFrame, period: int, t0: float, t1: float) -> pd.DataFrame:
    sel = h[(h.period == period) & (h.t > t0) & (h.t < t1)]
    return sel[["t", "bx", "by"]]


def build(df: pd.DataFrame, hr: dict[int, bool]) -> list[dict[str, Any]]:
    h = holders(df)
    segs = segments(h)
    out: list[dict[str, Any]] = []

    def add(kind: str, side: str, period: int, t: float, t_end: float, out_: str, tr: int, tr_to: Any,
            p0: tuple[float, float], p1: tuple[float, float] | None, conf: str, reason: str) -> None:
        x, y = team_frame(side, period, *p0, hr)
        ex = ey = None
        if p1 is not None:
            ex, ey = team_frame(side, period, *p1, hr)
        out.append({
            "period": period, "t": round(t, 2), "t_end": round(t_end, 2), "side": side, "type": kind,
            "outcome": out_, "track_id": tr, "track_to": "" if tr_to is None else tr_to,
            "x": round(x, 1), "y": round(y, 1),
            "ex": "" if ex is None else round(ex, 1), "ey": "" if ey is None else round(ey, 1),
            "confidence": conf, "reason": reason,
        })

    for a, b in zip(segs, segs[1:]):
        if a.period != b.period:
            continue
        gap = b.t0 - a.t1
        start, end = (a.bx1, a.by1), (b.bx0, b.by0)
        dist = math.hypot(end[0] - start[0], end[1] - start[1])
        conf = "ok" if gap <= 1.5 and a.n >= 3 and b.n >= 3 else "review"
        ax, ay = team_frame(a.side, a.period, *start, hr)

        # shot: ball leaves the final third and is next seen at the opponent goal
        if ax >= 70 and gap <= MAX_LINK_S:
            near_goal = False
            for r in ball_after(h, a.period, a.t1, min(b.t0, a.t1 + 2.0)).itertuples(index=False):
                gx, gy = team_frame(a.side, a.period, r.bx, r.by, hr)
                if gx >= 101 and 20 <= gy <= 48:
                    near_goal = True
                    break
            bx_, _ = team_frame(a.side, a.period, *end, hr)
            if near_goal or (b.side != a.side and b.role == "goalkeeper" and bx_ >= 94):
                outcome = "on" if (b.side != a.side and b.role == "goalkeeper") else ""
                add("shot", a.side, a.period, a.t1, b.t0, outcome, a.track, None, start, None, "review",
                    "ball reached the goal area" if near_goal else "keeper took the ball")
                continue

        if gap > MAX_LINK_S:
            if b.side in ("home", "away") and gap <= LOOSE_MAX_S:
                add("recovery", b.side, b.period, b.t0, b.t0, "ok", b.track, None, end, None, "review",
                    f"first holder after {gap:.1f} s without one")
            continue

        if b.side == a.side:
            if b.track == a.track or dist < PASS_MIN_M:
                continue
            ex_, ey_ = team_frame(a.side, a.period, *end, hr)
            in_box = ex_ >= 88.5 and 13.84 <= ey_ <= 54.16
            wide = ay < 13.84 or ay > 54.16
            kind = "cross" if (ax >= 70 and wide and in_box) else "pass"
            add(kind, a.side, a.period, a.t1, b.t0, "ok", a.track, b.track, start, end, conf,
                f"{dist:.0f} m to a team-mate")
            continue

        # possession changed team
        if gap > LOOSE_S:
            add("recovery", b.side, b.period, b.t0, b.t0, "ok", b.track, None, end, None, "review",
                f"loose ball for {gap:.1f} s")
            continue
        if dist < PASS_MIN_M:
            add("tackle", b.side, b.period, b.t0, b.t0, "won", b.track, a.track, end, None, conf,
                "ball changed team within 3 m")
            continue
        fwd = team_frame(a.side, a.period, *end, hr)[0] - ax
        if ax <= 35 and fwd >= CLEAR_MIN_M:
            add("clearance", a.side, a.period, a.t1, b.t0, "ok", a.track, None, start, None, conf,
                f"long ball out of own third ({fwd:.0f} m)")
        else:
            add("pass", a.side, a.period, a.t1, b.t0, "fail", a.track, b.track, start, end, conf,
                f"{dist:.0f} m to an opponent")
        add("interception", b.side, b.period, b.t0, b.t0, "ok", b.track, None, end, None, conf,
            "received an opponent's ball")

    out.sort(key=lambda r: (r["t"], r["type"] != "pass"))
    for i, r in enumerate(out, 1):
        r["cand_id"] = f"c{i:05d}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Event candidates from a Track B tracking CSV.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--match", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    match = load_match(a.config, a.match)
    hr = home_right(match)
    df = pd.read_csv(a.input, usecols=["frame", "video_seconds", "period", "track_id", "role", "team",
                                       "x", "y", "camera_state"],
                     dtype={"x": "float32", "y": "float32", "role": "category", "team": "category",
                            "camera_state": "category", "period": "int8"})
    for c in ("role", "team", "camera_state"):
        df[c] = df[c].astype(str)
    rows = build(df, hr)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by[r["type"]] = by.get(r["type"], 0) + 1
    print(f"{len(rows)} candidates -> {a.output}  {by}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
