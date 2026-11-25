#!/usr/bin/env python3
"""Physical running metrics from a Track B tracking CSV — distance, speed, sprints, touches.

Everything is a broadcast-tracking *proxy*: a player is only measured while on the
wide camera with a mapped position, so distances under-count (off-ball, replays and
close-ups are excluded), and Track B's anonymous ids restart every 10-minute chunk,
so one physical player is spread over many short tracks. Therefore:

- **Team totals** (sum over the team's tracks) are the reliable headline — every bit
  of measured running belongs to some track, so the team sum approximates the team's
  on-camera running.
- **Per track**: distance, top speed, sprints etc. for each anonymous track.
- **Per player**: only when a shirt-number map (`<id>_track_map.csv`, from OCR) is
  given — tracks are grouped by (team, number) and summed. Without it there are no
  per-player rows, just per-track.

Method (per track, wide-camera samples sorted by time):
  - split into contiguous runs wherever the gap between samples exceeds GAP_S;
  - smooth x, y with a short centred window (position noise would otherwise invent speed);
  - step speed = distance / dt, with steps above MAX_SPEED_MS dropped as id jumps;
  - distance = sum of kept steps; top speed = 95th percentile (robust to spikes);
  - high-intensity distance = distance covered at >= HI_MS; a sprint = a run at >= SPRINT_MS
    held for at least SPRINT_MIN_S;
  - touches = times this track was the nearest player (<= HOLD_M) to the ball.

    python scripts/player_physical.py --input runs/<id>/<id>_tracking.csv \
        [--track-map runs/<id>/<id>_track_map.csv] --output-dir runs/<id>
Thresholds follow common sports-tracking conventions (HI 19.8 km/h, sprint 25.2 km/h).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HOLD_M = 2.0              # ball within this of a player = a touch
GAP_S = 0.4              # samples further apart than this start a new run (don't bridge gaps)
SMOOTH_WINDOW = 13        # centred samples (~1.3 s at 10 Hz) — broadcast homography jitter is large
VEL_HALF_WINDOW = 5       # speed = displacement over +/- this many samples (~1 s), not per step
MAX_SPEED_MS = 9.0        # ~32 km/h; faster is id-swap jitter, not a run — clipped
HI_MS = 5.5               # 19.8 km/h — high-intensity running
SPRINT_MS = 7.0           # 25.2 km/h — sprint
MIN_SAMPLES = 8           # a track shorter than this (~0.8 s) is not measured
USECOLS = ["video_seconds", "period", "track_id", "role", "team", "x", "y", "camera_state"]


def _speed(rt: np.ndarray, rx: np.ndarray, ry: np.ndarray, w: int) -> np.ndarray:
    """Central-difference speed over +/- w samples (vectorised), clipped to MAX_SPEED_MS."""
    n = len(rt)
    hi = np.minimum(np.arange(n) + w, n - 1)
    lo = np.maximum(np.arange(n) - w, 0)
    dt = rt[hi] - rt[lo]
    disp = np.hypot(rx[hi] - rx[lo], ry[hi] - ry[lo])
    v = np.divide(disp, dt, out=np.zeros(n), where=dt > 0)
    return np.clip(v, 0.0, MAX_SPEED_MS)


def _track_metric(times: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> dict[str, float]:
    """One track's physical metrics from its time-ordered wide-camera samples.

    Distance-based metrics (distance, high-intensity, sprint distance) are the trustworthy
    ones; counting discrete sprint efforts on fragmented, jittery tracks is not, so we report
    distance at speed, not effort counts.
    """
    order = np.argsort(times)
    t, x, y = times[order], xs[order], ys[order]
    brk = np.flatnonzero(np.diff(t) > GAP_S) + 1       # break where the sampling gap is too big
    bounds = [0, *brk.tolist(), len(t)]
    dist = hi_dist = sprint_dist = moving_s = 0.0
    tops: list[float] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < 3:
            continue
        rt, rx, ry = t[a:b], x[a:b], y[a:b]
        k = min(SMOOTH_WINDOW, b - a)
        rx = pd.Series(rx).rolling(k, center=True, min_periods=1).mean().to_numpy()
        ry = pd.Series(ry).rolling(k, center=True, min_periods=1).mean().to_numpy()
        dt = np.diff(rt)
        step = np.hypot(np.diff(rx), np.diff(ry))
        v = _speed(rt, rx, ry, VEL_HALF_WINDOW)
        vstep = np.minimum(v[1:], v[:-1])              # speed attributed to each step
        keep = step <= MAX_SPEED_MS * dt               # drop id-jump steps from distance
        dist += float(step[keep].sum())
        hi_dist += float(step[keep & (vstep >= HI_MS)].sum())
        sprint_dist += float(step[keep & (vstep >= SPRINT_MS)].sum())
        moving_s += float(dt[keep].sum())
        tops.append(float(np.percentile(v, 99)))
    top = max(tops) if tops else 0.0
    return {
        "minutes": round(moving_s / 60.0, 2),
        "distance_m": round(dist, 1),
        "hi_distance_m": round(hi_dist, 1),
        "sprint_distance_m": round(sprint_dist, 1),
        "top_speed_kmh": round(top * 3.6, 1),
    }


def touches_by_track(df: pd.DataFrame) -> pd.Series:
    """Touch count per track: how many separate times it was the nearest player (<= HOLD_M) to the ball."""
    w = df[df.camera_state == "wide"].dropna(subset=["x", "y"])
    ball = w[w.role == "ball"][["video_seconds", "x", "y"]].drop_duplicates("video_seconds")
    ppl = w[w.role.isin(["player", "goalkeeper"]) & w.team.isin(["home", "away"])][
        ["video_seconds", "track_id", "x", "y"]]
    if ball.empty or ppl.empty:
        return pd.Series(dtype=int)
    m = ppl.merge(ball, on="video_seconds", suffixes=("", "_b"))
    m["d"] = np.hypot(m.x - m.x_b, m.y - m.y_b)
    nearest = m.loc[m.groupby("video_seconds").d.idxmin()]
    nearest = nearest[nearest.d <= HOLD_M].sort_values("video_seconds")
    # count a new touch whenever the holder track changes
    holder = nearest.track_id.to_numpy()
    new = np.r_[True, holder[1:] != holder[:-1]]
    return pd.Series(holder[new]).value_counts()


def per_track(df: pd.DataFrame) -> pd.DataFrame:
    w = df[(df.camera_state == "wide") & df.role.isin(["player", "goalkeeper"])
           & df.team.isin(["home", "away"])].dropna(subset=["x", "y"])
    rows = []
    for (tid, team, role), g in w.groupby(["track_id", "team", "role"], sort=False):
        if len(g) < MIN_SAMPLES:
            continue
        met = _track_metric(g.video_seconds.to_numpy(float), g.x.to_numpy(float), g.y.to_numpy(float))
        if met["minutes"] <= 0:
            continue
        rows.append({"track_id": int(tid), "team": team, "role": role, **met})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    touches = touches_by_track(df)
    out["touches"] = out.track_id.map(touches).fillna(0).astype(int)
    out["dist_per_min"] = (out.distance_m / out.minutes.where(out.minutes > 0)).round(0)
    return out.sort_values("distance_m", ascending=False).reset_index(drop=True)


def team_totals(pt: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Team running totals from the per-track table (sum of the team's tracks)."""
    out: dict[str, dict[str, Any]] = {}
    for side in ("home", "away"):
        s = pt[pt.team == side]
        out[side] = {
            "distance_km": round(float(s.distance_m.sum()) / 1000.0, 2),
            "hi_distance_km": round(float(s.hi_distance_m.sum()) / 1000.0, 2),
            "sprint_distance_km": round(float(s.sprint_distance_m.sum()) / 1000.0, 2),
            "top_speed_kmh": round(float(s.top_speed_kmh.max()), 1) if len(s) else None,
            "tracked_player_seconds": int(round(float(s.minutes.sum()) * 60)),
            "touches": int(s.touches.sum()) if "touches" in s else 0,
        }
    return out


def by_player(pt: pd.DataFrame, track_map: Path | None) -> pd.DataFrame:
    """Aggregate tracks to players using a shirt-number map; empty if no usable map."""
    if pt.empty or not track_map or not Path(track_map).is_file():
        return pd.DataFrame()
    tm = pd.read_csv(track_map)
    tm = tm[(tm.get("confidence").fillna(0) >= 0.5) & (tm.get("votes").fillna(0) >= 2)]
    if tm.empty:
        return pd.DataFrame()
    num = dict(zip(tm.track_id.astype(int), tm.number.astype(str)))
    j = pt.copy()
    j["number"] = j.track_id.map(num)
    j = j.dropna(subset=["number"])
    if j.empty:
        return pd.DataFrame()
    agg = j.groupby(["team", "number"]).agg(
        minutes=("minutes", "sum"), distance_m=("distance_m", "sum"),
        hi_distance_m=("hi_distance_m", "sum"), sprint_distance_m=("sprint_distance_m", "sum"),
        top_speed_kmh=("top_speed_kmh", "max"), touches=("touches", "sum"),
        tracks=("track_id", "nunique")).reset_index()
    agg["dist_per_min"] = (agg.distance_m / agg.minutes.where(agg.minutes > 0)).round(0)
    return agg.sort_values(["team", "distance_m"], ascending=[True, False]).reset_index(drop=True)


def build(csv_path: Path, track_map: Path | None = None) -> dict[str, Any]:
    df = pd.read_csv(csv_path, usecols=USECOLS,
                     dtype={"x": "float32", "y": "float32", "period": "int16"})
    for c in ("role", "team", "camera_state"):
        df[c] = df[c].astype(str)
    pt = per_track(df)
    players = by_player(pt, track_map)
    return {
        "schema": "trackb_player_physical_v1",
        "thresholds": {"hi_kmh": round(HI_MS * 3.6, 1), "sprint_kmh": round(SPRINT_MS * 3.6, 1),
                       "max_speed_kmh": round(MAX_SPEED_MS * 3.6, 1)},
        "caveat": "Broadcast-tracking proxy: on-camera running only; Track B ids restart each "
                  "10-min chunk, so team totals are reliable but per-track rows are fragments. "
                  "Per-player rows need shirt-number OCR (track_map).",
        "team_totals": team_totals(pt) if not pt.empty else {},
        "per_track": pt, "by_player": players,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-player / per-team physical metrics from Track B tracking.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--track-map", type=Path, help="default: <input dir>/<match>_track_map.csv if present")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--match", help="match id for output names; default: tracking stem without _tracking")
    a = ap.parse_args()
    mid = a.match or a.input.stem.replace("_tracking", "")
    tmap = a.track_map or a.input.with_name(f"{mid}_track_map.csv")
    res = build(a.input, tmap)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    pt, players = res.pop("per_track"), res.pop("by_player")
    pt_path = a.output_dir / f"{mid}_player_physical.csv"
    pt.to_csv(pt_path, index=False)
    res["per_track_rows"] = int(len(pt))
    res["by_player"] = players.to_dict("records")   # small (<= squad size); embedded for the Hub
    if not players.empty:
        players.to_csv(a.output_dir / f"{mid}_player_physical_byplayer.csv", index=False)
    (a.output_dir / f"{mid}_player_physical.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    tt = res["team_totals"]
    print(f"wrote {pt_path} ({len(pt)} tracks, {len(players)} players)")
    for side in ("home", "away"):
        if side in tt:
            d = tt[side]
            print(f"  {side}: {d['distance_km']} km total, {d['hi_distance_km']} km high-intensity, "
                  f"{d['sprint_distance_km']} km sprinting, top {d['top_speed_kmh']} km/h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
