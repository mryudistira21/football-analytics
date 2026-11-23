#!/usr/bin/env python3
"""Per-half team report from a Track B tracking CSV — no manual tagging needed.

Everything here is a *proxy* computed from broadcast tracking, so each number
comes with a data-quality block (wide-camera share, known-team share, ball
visibility). Use it for trends and comparisons between halves and matches,
not as official stats.

Metrics per period (1, 2, [3, 4], all) and team:
  possession_share      share of possession TIME (wide-camera samples with a known owner).
                        Owner = team of the ball holder (nearest known-team player within
                        2 m), with hysteresis: the owner keeps the ball through duels and
                        flicker until the opponent holds it in 3 samples in a row; nobody
                        owns it after 5 s without a holder or across a camera cut.
  ball_proximity_share  diagnostic: share of ball-near samples (<= 3 m) where this team's
                        player is nearest — frame counting, flickers in duels
  territory             ball location split into thirds, seen from the home team
  block_height_m        median mean-x of the team's outfield players (0 = own goal line)
  def_line_m            median x of the deepest outfield player (after dropping the single
                        deepest one per frame, which is often a keeper detected as 'player')
  width_m / length_m    median spread of the outfield players
  in/out of possession  block height split by the possession owner above
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
HOLDER_M = 2.0            # ball holder: nearest known-team player within this distance
CONFIRM_SAMPLES = 3       # the opponent must hold the ball this many samples in a row to win it
LOOSE_RESET_S = 5.0       # no holder for this long (or a camera cut/gap this long) -> nobody owns it
USECOLS = ["frame", "video_seconds", "period", "track_id", "role", "team", "x", "y", "camera_state"]


def load_match(config: Path, match_id: str) -> dict[str, Any]:
    cfg = json.loads(config.read_text(encoding="utf-8"))
    for m in cfg.get("matches", []):
        if m.get("match_id") == match_id:
            return m
    raise SystemExit(f"{match_id} not in {config}")


def home_right(match: dict[str, Any], periods: list[int], override: str | None = None) -> dict[int, bool]:
    d = override or match.get("home_dir_1h")
    if d not in ("R", "L"):
        raise SystemExit("Attack direction unknown: set home_dir_1h (R or L) for this match in the config, "
                         "or run scripts/resolve_direction.py (run_match does this automatically).")
    out = {1: d == "R", 2: d != "R"}
    if any(p >= 3 for p in periods):
        et = match.get("home_dir_et1")
        if et not in ("R", "L"):
            raise SystemExit("Extra time present: set home_dir_et1 (R or L) in the config.")
        out.update({3: et == "R", 4: et != "R"})
    return out


KICKOFF_WINDOW_S = 10.0   # first seconds of each period: both teams in their own half
KICKOFF_MIN_ROWS = 15
KICKOFF_MIN_GAP_M = 8.0
SHIFT_MIN_M = 6.0         # (home shift - away shift) needed for a possession-shift vote


def center_restart_vote(period_rows: pd.DataFrame, period: int, scan_seconds: float = 600.0) -> dict[str, Any] | None:
    """Find one delayed kick-off / after-goal restart vote in the first part of a period.

    A valid window needs the ball near the centre circle and both outfield teams
    clearly arranged in opposite halves. Contiguous centre-ball runs are evaluated
    once each to avoid repeatedly scanning the full player table.
    """
    players = period_rows[
        (period_rows.camera_state == "wide")
        & (period_rows.role == "player")
        & period_rows.team.isin(["home", "away"])
    ].dropna(subset=["x"])
    ball = period_rows[
        (period_rows.camera_state == "wide")
        & (period_rows.role == "ball")
    ].dropna(subset=["x", "y", "video_seconds"])[
        ["frame", "video_seconds", "x", "y"]
    ].drop_duplicates("frame").sort_values("video_seconds")
    if players.empty or ball.empty:
        return None

    t0 = float(period_rows.video_seconds.min())
    t1 = min(float(period_rows.video_seconds.max()), t0 + scan_seconds)
    ball = ball[(ball.video_seconds >= t0) & (ball.video_seconds <= t1)]
    radius = np.hypot(ball.x - PL / 2.0, ball.y - PW / 2.0)
    center_ball = ball[radius <= 12.0]
    if len(center_ball) < 3:
        return None

    # A restart keeps the ball near the centre for several sampled frames.
    # Split distant centre-ball appearances into independent candidate restarts.
    center_ball = center_ball.sort_values("video_seconds")
    times = center_ball.video_seconds.to_numpy(dtype=float)
    breaks = np.flatnonzero(np.diff(times) > 1.5) + 1
    bounds = [0, *breaks.tolist(), len(center_ball)]
    # slice with iloc: np.split on a DataFrame returns bare arrays on newer NumPy
    runs = [center_ball.iloc[a:b] for a, b in zip(bounds, bounds[1:])]
    candidates = []
    window_s = 3.0
    for run in runs:
        if len(run) < 3 or float(run.video_seconds.iloc[-1] - run.video_seconds.iloc[0]) < 0.15:
            continue
        start = float(run.video_seconds.iloc[0])
        b = ball[(ball.video_seconds >= start) & (ball.video_seconds < start + window_s)]
        if len(b) < 3:
            continue
        center = np.hypot(b.x - PL / 2.0, b.y - PW / 2.0) <= 12.0
        center_frames = set(b.loc[center, "frame"].astype(int).tolist())
        center_share = float(center.mean())
        if len(center_frames) < 3 or center_share < 0.15:
            continue

        p = players[players.frame.isin(center_frames)]
        home = p[p.team == "home"].x
        away = p[p.team == "away"].x
        if len(home) < 15 or len(away) < 15:
            continue
        right_consistency = (float((home < PL / 2).mean())
                             + float((away > PL / 2).mean())) / 2
        left_consistency = (float((home > PL / 2).mean())
                            + float((away < PL / 2).mean())) / 2
        if right_consistency >= left_consistency:
            consistency, gap, home_right_period = (
                right_consistency, float(away.mean() - home.mean()), True
            )
        else:
            consistency, gap, home_right_period = (
                left_consistency, float(home.mean() - away.mean()), False
            )
        if consistency >= 0.68 and abs(gap) >= KICKOFF_MIN_GAP_M:
            candidates.append({
                "signal": "center_restart",
                "period": int(period),
                "seconds_from_period_start": round(start - t0, 1),
                "metres": round(abs(gap), 1),
                "center_ball_share": round(center_share, 3),
                "half_separation": round(consistency, 3),
                "home_right_1h": home_right_period if period == 1 else not home_right_period,
            })

    if not candidates:
        return None
    candidates.sort(key=lambda v: (
        v["half_separation"], v["center_ball_share"], -v["seconds_from_period_start"]
    ), reverse=True)
    return candidates[0]


def infer_direction(df: pd.DataFrame, poss: pd.DataFrame | None = None) -> dict[str, Any]:
    """Which way the home team attacks in the 1st half, from the tracking itself.

    Two independent signals, both from OUTFIELD players' team labels (keepers' kits
    are unreliable for team colour):
      kickoff   — in the first seconds of each period each team stands in its own half
      shift     — a team's players stand further forward when it has the ball than
                  when the opponent has it (camera x, possession timeline below)
    Each period gives up to one vote per signal, expressed as 'home attacks right in
    the 1st half' (a 2nd-half vote is mirrored). Agreeing votes -> R/L.
    """
    votes: list[dict[str, Any]] = []
    wide = df[(df.camera_state == "wide") & (df.role == "player") & df.team.isin(["home", "away"])].dropna(subset=["x"])
    if poss is None:
        poss = possession_frames(df)
    for p in sorted(int(x) for x in df.period.unique() if int(x) in (1, 2)):
        period_rows = df[df.period == p]
        d = wide[wide.period == p]
        if d.empty:
            continue
        t0 = float(period_rows.video_seconds.min())
        k = d[d.video_seconds < t0 + KICKOFF_WINDOW_S]
        n = k.team.value_counts()
        # Keep the simple first-10-second kick-off vote when it is usable.
        kickoff_added = False
        if n.get("home", 0) >= KICKOFF_MIN_ROWS and n.get("away", 0) >= KICKOFF_MIN_ROWS:
            gap = float(k[k.team == "away"].x.mean() - k[k.team == "home"].x.mean())
            if abs(gap) >= KICKOFF_MIN_GAP_M:
                right_p = gap > 0
                votes.append({"signal": "kickoff", "period": p, "metres": round(gap, 1),
                              "home_right_1h": right_p if p == 1 else not right_p})
                kickoff_added = True

        # If the configured period start contains pre-roll / graphics, look for
        # a later centre restart (kick-off or after-goal restart) instead.
        if not kickoff_added:
            restart = center_restart_vote(period_rows, p)
            if restart:
                votes.append(restart)

        m = d.merge(poss[["frame", "owner"]], on="frame")
        shift = {}
        for side in ("home", "away"):
            o = m[m.team == side]
            fin = o[o.owner == side].groupby("frame").x.mean().median()
            fout = o[o.owner.notna() & (o.owner != side)].groupby("frame").x.mean().median()
            shift[side] = float(fin - fout) if pd.notna(fin) and pd.notna(fout) else None
        if shift["home"] is not None and shift["away"] is not None:
            net = shift["home"] - shift["away"]
            if abs(net) >= SHIFT_MIN_M:
                right_p = net > 0
                votes.append({"signal": "possession_shift", "period": p, "metres": round(net, 1),
                              "home_right_1h": right_p if p == 1 else not right_p})
    yes = sum(v["home_right_1h"] for v in votes)
    no = len(votes) - yes
    if votes and (yes == 0 or no == 0):
        d1 = "R" if yes else "L"
        conf = "high" if len(votes) >= 2 else "medium"
    else:
        d1, conf = None, ("conflict" if votes else "none")
    return {"home_dir_1h": d1, "confidence": conf, "votes": votes}


def fix_goalkeepers(df: pd.DataFrame, hr: dict[int, bool]) -> tuple[pd.DataFrame, int]:
    """Give every goalkeeper track the team that defends the half it stands in.

    Keeper kits differ from the outfield kit, so colour clustering labels keepers
    almost at random. With the attack direction known, the half decides: per
    (track, period) the median x picks the goal. Returns (df, rows changed).
    """
    gk = df.role == "goalkeeper"
    if not gk.any():
        return df, 0
    g = df[gk].dropna(subset=["x"]).groupby(["track_id", "period"]).x.median().rename("gx").reset_index()
    if g.empty:
        return df, 0
    home_r = g.period.map(lambda p: hr.get(int(p))).astype("boolean")
    # home attacks right -> home defends the LEFT goal (x < 52.5)
    left = g.gx < PL / 2
    g["new_team"] = np.where(home_r.isna(), None, np.where(left == home_r.fillna(False), "home", "away"))
    g = g.dropna(subset=["new_team"])
    df = df.copy()
    if isinstance(df["team"].dtype, pd.CategoricalDtype):
        df["team"] = df["team"].astype(str)
    key = df.loc[gk, ["track_id", "period"]].merge(g[["track_id", "period", "new_team"]], how="left",
                                                    on=["track_id", "period"])
    new = key.new_team.to_numpy()
    old = df.loc[gk, "team"].to_numpy()
    take = pd.notna(new)
    changed = int((take & (new != old)).sum())
    upd = np.where(take, new, old)
    df.loc[gk, "team"] = upd
    return df, changed


def r1(v: Any, nd: int = 1) -> float | None:
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), nd)


def possession_frames(df: pd.DataFrame, interval: float = 0.1) -> pd.DataFrame:
    """Per sampled frame: `near` (nearest team within NEAR_BALL_M, diagnostic),
    `holder` (nearest team within HOLDER_M) and `owner` (possession with hysteresis).

    Only wide-camera samples carry positions; other samples break the timeline once
    they last LOOSE_RESET_S (a replay or close-up is not possession time).
    """
    cols = ["frame", "near", "holder", "owner"]
    wide = df[df.camera_state == "wide"] if "camera_state" in df else df
    xy = wide.dropna(subset=["x", "y"])
    ball = xy[xy.role == "ball"][["frame", "x", "y"]].drop_duplicates("frame")
    ppl = xy[xy.role.isin(["player", "goalkeeper"]) & xy.team.isin(["home", "away"])][["frame", "team", "x", "y"]]
    m = ppl.merge(ball, on="frame", suffixes=("", "_b"))
    frames = (df[["frame", "video_seconds"]] if "video_seconds" in df else df[["frame"]]) \
        .drop_duplicates("frame").sort_values("frame").reset_index(drop=True)
    if "video_seconds" not in frames:
        frames["video_seconds"] = frames.frame * interval
    if m.empty:
        out = frames[["frame"]].copy()
        for c in cols[1:]:
            out[c] = None
        return out
    m["d"] = np.hypot(m.x - m.x_b, m.y - m.y_b)
    best = m.loc[m.groupby("frame").d.idxmin(), ["frame", "team", "d"]]
    out = frames.merge(best, on="frame", how="left")
    out["near"] = out.team.where(out.d <= NEAR_BALL_M)
    out["holder"] = out.team.where(out.d <= HOLDER_M)
    wide_frames = set(wide.frame.unique())

    owners: list[Any] = []
    owner, challenger, streak, last_hold_t, last_wide_t = None, None, 0, -1e9, -1e9
    for f, t, h in zip(out.frame.to_numpy(), out.video_seconds.to_numpy(dtype=float), out.holder.to_numpy()):
        if f not in wide_frames:
            if t - last_wide_t > LOOSE_RESET_S:
                owner, challenger, streak = None, None, 0
            owners.append(None)
            continue
        if t - last_wide_t > LOOSE_RESET_S:          # camera cut / gap: start fresh
            owner, challenger, streak = None, None, 0
        last_wide_t = t
        if isinstance(h, str):
            last_hold_t = t
            if h == owner:
                challenger, streak = None, 0
            elif owner is None:
                owner, challenger, streak = h, None, 0
            else:
                streak = streak + 1 if h == challenger else 1
                challenger = h
                if streak >= CONFIRM_SAMPLES:
                    owner, challenger, streak = h, None, 0
        elif t - last_hold_t > LOOSE_RESET_S:
            owner, challenger, streak = None, None, 0
        owners.append(owner)
    out["owner"] = owners
    return out[cols]


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


def pressing_stats(df: pd.DataFrame, poss: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Estimate pressure in the opponent's own third from wide-camera tracking.

    Denominator: opponent-owned samples in the opponent's own third. Numerator:
    those samples with at least one outfield presser within 5 m of the current
    ball holder. Samples without a current holder stay in the denominator.
    This is a tracking proxy, not event-data PPDA.
    """
    empty = {side: {"press_share": None, "press_eligible_samples": 0, "pressing_samples": 0}
             for side in ("home", "away")}
    if df.empty or poss.empty:
        return empty
    wide = df[df.camera_state == "wide"]
    ball = wide[wide.role == "ball"].dropna(subset=["x", "y", "tx"])[
        ["frame", "x", "y", "tx"]].drop_duplicates("frame")
    if ball.empty:
        return empty
    people = wide[wide.role.isin(["player", "goalkeeper"]) & wide.team.isin(["home", "away"])].dropna(
        subset=["x", "y"])[["frame", "team", "x", "y"]]
    if people.empty:
        return empty
    near = people.merge(ball[["frame", "x", "y"]], on="frame", suffixes=("", "_ball"))
    near["holder_distance_m"] = np.hypot(near.x - near.x_ball, near.y - near.y_ball)
    nearest_idx = near.groupby("frame").holder_distance_m.idxmin()
    holders = near.loc[nearest_idx, ["frame", "team", "x", "y", "holder_distance_m"]]
    holders = holders[holders.holder_distance_m <= HOLDER_M].rename(
        columns={"team": "holder_team", "x": "holder_x", "y": "holder_y"})
    states = (poss[["frame", "owner", "holder"]]
              .merge(ball[["frame", "tx"]], on="frame", how="inner")
              .merge(holders[["frame", "holder_team", "holder_x", "holder_y"]],
                     on="frame", how="left"))
    for side in ("home", "away"):
        opponent = "away" if side == "home" else "home"
        # tx is the home team's attacking-left-to-right view.
        own_third = states.tx >= 70.0 if opponent == "away" else states.tx <= 35.0
        eligible = states[(states.owner == opponent) & own_third]
        denominator = int(eligible.frame.nunique())
        if denominator == 0:
            continue
        held = eligible[(eligible.holder == opponent) & (eligible.holder_team == opponent)][
            ["frame", "holder_x", "holder_y"]].drop_duplicates("frame")
        pressers = wide[(wide.role == "player") & (wide.team == side)].dropna(
            subset=["x", "y"])[["frame", "x", "y"]]
        pairs = pressers.merge(held, on="frame", how="inner")
        if pairs.empty:
            numerator = 0
        else:
            distance = np.hypot(pairs.x - pairs.holder_x, pairs.y - pairs.holder_y)
            numerator = int(pairs.loc[distance <= 5.0, "frame"].nunique())
        empty[side] = {"press_share": round(numerator / denominator, 3),
                       "press_eligible_samples": denominator, "pressing_samples": numerator}
    return empty


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
    own_counts = pf.owner.value_counts()
    own_tot = int(own_counts.get("home", 0) + own_counts.get("away", 0))

    ball = mapped[mapped.role == "ball"]
    thirds = [0.0, 0.0, 0.0]
    if len(ball):
        b = pd.cut(ball.tx, [-1, 35, 70, PL + 1], labels=False).value_counts(normalize=True)
        thirds = [round(float(b.get(i, 0.0)), 3) for i in range(3)]

    owner = pf.set_index("frame").owner
    press = pressing_stats(df, poss)
    res: dict[str, Any] = {
        "quality": {
            "sampled_frames": int(frames_all),
            "wide_share": round(frames_wide / frames_all, 3) if frames_all else 0.0,
            "known_team_share": round(float(known), 3),
            "ball_visible_share": round(ball_frames / frames_wide, 3) if frames_wide else 0.0,
            "ball_near_frames": near_tot,
            "possession_known_share": round(own_tot / frames_wide, 3) if frames_wide else 0.0,
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
            "possession_share": round(float(own_counts.get(side, 0)) / own_tot, 3) if own_tot else None,
            "ball_proximity_share": round(float(near.get(side, 0)) / near_tot, 3) if near_tot else None,
            **shape_stats(o),
            "block_height_in_possession_m": shape_stats(o[own == side]).get("block_height_m"),
            "block_height_out_of_possession_m": shape_stats(o[(own != side) & pd.notna(own)]).get("block_height_m"),
            **press[side],
            "heatmap": heat(mapped[(mapped.role == "player") & (mapped.team == side)]),
        }
    return res


def build(csv_path: Path, match: dict[str, Any], meta: dict[str, Any], home_dir: str | None = None) -> dict[str, Any]:
    df = pd.read_csv(csv_path, usecols=USECOLS,
                     dtype={"x": "float32", "y": "float32", "role": "category", "team": "category",
                            "camera_state": "category", "period": "int8"})
    for c in ("role", "team", "camera_state"):
        df[c] = df[c].astype(str)
    periods = sorted(int(p) for p in df.period.unique())
    fps = float(meta.get("fps") or 30)
    interval = float(meta.get("frame_stride") or 3) / fps
    poss = possession_frames(df, interval)  # one timeline; resets at camera cuts and half time
    inferred = infer_direction(df, poss)
    cfg_dir = match.get("home_dir_1h") if match.get("home_dir_1h") in ("R", "L") else None
    used = home_dir or cfg_dir
    source = "config" if (used and used == cfg_dir) else ("inferred" if home_dir and not cfg_dir else "override")
    if not used and inferred["confidence"] in ("high", "medium"):
        used, source = inferred["home_dir_1h"], "inferred"
    hr = home_right(match, periods, used)
    df, gk_changed = fix_goalkeepers(df, hr)
    if gk_changed:   # keepers changed team -> rebuild the possession timeline with them
        poss = possession_frames(df, interval)
    df = team_frame(df, hr)
    warnings = []
    if inferred["home_dir_1h"] and inferred["confidence"] == "high" and inferred["home_dir_1h"] != used:
        warnings.append(f"Attack direction: the data says the home team attacks "
                        f"{'left -> right' if inferred['home_dir_1h'] == 'R' else 'right -> left'} in the 1st half "
                        f"(home_dir_1h = {inferred['home_dir_1h']}), but {used} was used ({source}). "
                        f"Check the video; if the data is right, every position metric here is mirrored.")

    report: dict[str, Any] = {
        "schema": "trackb_team_report_v1",
        "match_id": match["match_id"], "date": match.get("date"), "round": match.get("round"),
        "home": match["home"], "away": match["away"], "score": match.get("score"),
        "home_attacks_right_by_period": {str(k): v for k, v in hr.items()},
        "direction_check": {"used": used, "source": source, "inferred": inferred["home_dir_1h"],
                            "confidence": inferred["confidence"], "votes": inferred["votes"],
                            "goalkeeper_rows_relabelled": gk_changed},
        "warnings": warnings,
        "grid": list(GRID), "pitch": [PL, PW],
        "method": {
            "near_ball_m": NEAR_BALL_M, "min_outfield_players_for_shape": MIN_OUTFIELD,
            "possession": {"holder_m": HOLDER_M, "confirm_samples": CONFIRM_SAMPLES,
                           "loose_reset_s": LOOSE_RESET_S, "basis": "time, wide-camera samples"},
            "pressing": {"own_third_m": 35.0, "holder_m": HOLDER_M, "press_distance_m": 5.0,
                         "definition": "opponent-owned wide samples in the opponent's own third "
                                       "with an outfield presser within 5 m of the current holder",
                         "caveat": "Tracking proxy; not event-data PPDA."},
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
    for w in rep.get("warnings") or []:
        lines += [f"> **Warning:** {w}", ""]
    dc = rep.get("direction_check")
    if dc:
        lines += [f"Attack direction (home, 1st half): {dc['used']} ({dc['source']}; data says "
                  f"{dc['inferred'] or '?'}, {dc['confidence']} confidence, {len(dc['votes'])} votes).", ""]
    names = {"1": "1st half", "2": "2nd half", "3": "ET 1", "4": "ET 2", "all": "Full match"}
    for key in [k for k in ("1", "2", "3", "4", "all") if k in rep["periods"]]:
        b = rep["periods"][key]
        q = b["quality"]
        t = b["territory_home_view"]
        lines += [f"## {names[key]}", "",
                  f"Data: wide camera {q['wide_share']:.0%} · team known {q['known_team_share']:.0%} · "
                  f"ball visible {q['ball_visible_share']:.0%} · {q['ball_near_frames']} ball-near samples · "
                  f"possession owner known {q.get('possession_known_share', 0):.0%} of wide time", "",
                  f"| Metric | {h} | {a} |", "|---|---:|---:|"]
        def v(side: str, k: str, unit: str = "") -> str:
            x = b[side].get(k)
            return "—" if x is None else (f"{x:.0%}" if unit == "%" else f"{x} {unit}".strip())
        for label, k, u in [("Possession (time share)", "possession_share", "%"),
                            ("Ball-proximity share (diagnostic)", "ball_proximity_share", "%"),
                            ("Press share (opponent own third)", "press_share", "%"),
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
    ap.add_argument("--home-dir", choices=["R", "L"], help="override home_dir_1h from the config")
    a = ap.parse_args()
    match = load_match(a.config, a.match)
    meta_path = a.metadata or a.input.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    rep = build(a.input, match, meta, a.home_dir)
    for w in rep.get("warnings") or []:
        print("WARNING:", w)
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
