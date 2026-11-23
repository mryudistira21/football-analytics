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

Review workload (v1.1, extra columns only):
  - contested balls: A -> opponent (<= 0.3 s) -> A's team again within 1.5 s is a duel
    or detection flicker, not two changes of possession: both are dropped (a pass
    between two team-mates around it is kept, confidence review)
  - priority 1 = Track B confident (ok), 2 = needs a look, 3 = probably noise (recovery
    after the ball was out of view, very short possessions). The Tagger hides 3 by default.
  - linked: an interception carries the cand_id of the failed pass / clearance it
    answers; the Tagger decides both with one keypress.

Output columns (candidates contract v1, + v1.1 additions at the end):
  cand_id, period, t, t_end, side, type, outcome, track_id, track_to,
  x, y, ex, ey, confidence, reason, track_player, track_to_player, priority, linked

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
FLICKER_N = 3           # an opponent 'possession' of at most this many samples ...
FLICKER_S = 1.5         # ... that the first team gets back within this long = contested ball
PASS_MIN_M = 3.0
CLEAR_MIN_M = 25.0
FIELDS = ["cand_id", "period", "t", "t_end", "side", "type", "outcome", "track_id", "track_to",
          "x", "y", "ex", "ey", "confidence", "reason", "track_player", "track_to_player",
          "priority", "linked"]


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


def home_right(match: dict[str, Any], override: str | None = None) -> dict[int, bool]:
    d = override or match.get("home_dir_1h")
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
            p0: tuple[float, float], p1: tuple[float, float] | None, conf: str, reason: str,
            priority: int | None = None) -> dict[str, Any]:
        x, y = team_frame(side, period, *p0, hr)
        ex = ey = None
        if p1 is not None:
            ex, ey = team_frame(side, period, *p1, hr)
        row = {
            "period": period, "t": round(t, 2), "t_end": round(t_end, 2), "side": side, "type": kind,
            "outcome": out_, "track_id": tr, "track_to": "" if tr_to is None else tr_to,
            "x": round(x, 1), "y": round(y, 1),
            "ex": "" if ex is None else round(ex, 1), "ey": "" if ey is None else round(ey, 1),
            "confidence": conf, "reason": reason,
            "priority": priority if priority is not None else (1 if conf == "ok" else 2), "linked": "",
        }
        out.append(row)
        return row

    # contested balls: a -> b (short, other team) -> c (a's team again, soon)
    contested: set[int] = set()
    for i in range(len(segs) - 2):
        a, b, c = segs[i], segs[i + 1], segs[i + 2]
        if (a.period == b.period == c.period and b.side != a.side and c.side == a.side
                and b.n <= FLICKER_N and c.t0 - a.t1 <= FLICKER_S):
            contested.update((i, i + 1))
            dist = math.hypot(c.bx0 - a.bx1, c.by0 - a.by1)
            if c.track != a.track and dist >= PASS_MIN_M:
                add("pass", a.side, a.period, a.t1, c.t0, "ok", a.track, c.track, (a.bx1, a.by1), (c.bx0, c.by0),
                    "review", f"{dist:.0f} m to a team-mate through a contested ball")

    for i, (a, b) in enumerate(zip(segs, segs[1:])):
        if a.period != b.period or i in contested:
            continue
        gap = b.t0 - a.t1
        start, end = (a.bx1, a.by1), (b.bx0, b.by0)
        dist = math.hypot(end[0] - start[0], end[1] - start[1])
        conf = "ok" if gap <= 1.5 and a.n >= 3 and b.n >= 3 else "review"
        short = min(a.n, b.n) <= MIN_SAMPLES       # one of the two possessions is a blip
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
                    "ball reached the goal area" if near_goal else "keeper took the ball", priority=2)
                continue

        if gap > MAX_LINK_S:
            if b.side in ("home", "away") and gap <= LOOSE_MAX_S:
                add("recovery", b.side, b.period, b.t0, b.t0, "ok", b.track, None, end, None, "review",
                    f"first holder after {gap:.1f} s without one (ball probably out of view)", priority=3)
            continue

        if b.side == a.side:
            if b.track == a.track or dist < PASS_MIN_M:
                continue
            ex_, ey_ = team_frame(a.side, a.period, *end, hr)
            in_box = ex_ >= 88.5 and 13.84 <= ey_ <= 54.16
            wide = ay < 13.84 or ay > 54.16
            kind = "cross" if (ax >= 70 and wide and in_box) else "pass"
            add(kind, a.side, a.period, a.t1, b.t0, "ok", a.track, b.track, start, end, conf,
                f"{dist:.0f} m to a team-mate", priority=3 if (short and gap > 1.0 and kind == "pass") else None)
            continue

        # possession changed team
        if gap > LOOSE_S:
            add("recovery", b.side, b.period, b.t0, b.t0, "ok", b.track, None, end, None, "review",
                f"loose ball for {gap:.1f} s", priority=3 if b.n <= MIN_SAMPLES else 2)
            continue
        if dist < PASS_MIN_M:
            add("tackle", b.side, b.period, b.t0, b.t0, "won", b.track, a.track, end, None, conf,
                "ball changed team within 3 m", priority=3 if (short and conf != "ok") else None)
            continue
        fwd = team_frame(a.side, a.period, *end, hr)[0] - ax
        if ax <= 35 and fwd >= CLEAR_MIN_M:
            parent = add("clearance", a.side, a.period, a.t1, b.t0, "ok", a.track, None, start, None, conf,
                         f"long ball out of own third ({fwd:.0f} m)")
        else:
            parent = add("pass", a.side, a.period, a.t1, b.t0, "fail", a.track, b.track, start, end, conf,
                         f"{dist:.0f} m to an opponent", priority=3 if (short and conf != "ok") else None)
        child = add("interception", b.side, b.period, b.t0, b.t0, "ok", b.track, None, end, None, conf,
                    "received an opponent's ball", priority=parent["priority"])
        child["_parent"] = parent

    out.sort(key=lambda r: (r["t"], r["type"] != "pass"))
    for i, r in enumerate(out, 1):
        r["cand_id"] = f"c{i:05d}"
    for r in out:
        parent = r.pop("_parent", None)
        if parent is not None:
            r["linked"] = parent["cand_id"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Event candidates from a Track B tracking CSV.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--match", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--track-map", type=Path,
                    help="optional <match>_track_map.csv from shirt-number OCR; pre-fills track_player columns")
    ap.add_argument("--home-dir", choices=["R", "L"], help="override home_dir_1h (run_match passes the resolved one)")
    a = ap.parse_args()
    match = load_match(a.config, a.match)
    numbers: dict[str, str] = {}
    tmap_path = a.track_map or a.input.with_name(f"{a.match}_track_map.csv")
    if tmap_path.is_file():
        with tmap_path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                # keep only reasonably confident reads as a suggestion
                if float(r.get("confidence") or 0) >= 0.5 and int(float(r.get("votes") or 0)) >= 2:
                    numbers[str(r["track_id"])] = r["number"]
    hr = home_right(match, a.home_dir)
    df = pd.read_csv(a.input, usecols=["frame", "video_seconds", "period", "track_id", "role", "team",
                                       "x", "y", "camera_state"],
                     dtype={"x": "float32", "y": "float32", "role": "category", "team": "category",
                            "camera_state": "category", "period": "int8"})
    for c in ("role", "team", "camera_state"):
        df[c] = df[c].astype(str)
    rows = build(df, hr)
    for r in rows:
        r["track_player"] = numbers.get(str(r["track_id"]), "")
        r["track_to_player"] = numbers.get(str(r["track_to"]), "") if r["track_to"] else ""
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by[r["type"]] = by.get(r["type"], 0) + 1
    n_named = sum(1 for r in rows if r.get("track_player"))
    queue = sum(1 for r in rows if r["priority"] in (1, 2) and not r["linked"])
    print(f"{len(rows)} candidates -> {a.output}  {by}" + (f"  ({n_named} with a suggested number)" if numbers else ""))
    print(f"  review queue: {queue} cards (priority 1-2, interceptions decided with their pass); "
          f"{sum(1 for r in rows if r['priority'] == 3)} low-priority hidden by default")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
