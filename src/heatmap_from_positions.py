"""Team hot-area maps from a tracking position CSV (Track B output).

Input follows DATA_CONTRACT.md -> tracking CSV. Required columns:
    period, team, x, y
x, y are metres on a 105 x 68 pitch in the CAMERA frame (x = 0 is the goal on
the left of the broadcast picture). Rows with role other than player /
goalkeeper are skipped when a `role` column exists.

Each team is flipped to attack left -> right using the per-period direction map
in the tracking metadata JSON (same basename as the CSV):
    python src/heatmap_from_positions.py tracking.csv
For old CSVs without metadata, use the legacy two-half fallback:
    python src/heatmap_from_positions.py tracking.csv --home-dir R

Memory stays flat on long matches: the CSV is read in chunks and only a small
histogram is kept, never the whole table.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PITCH_L, PITCH_W = 105.0, 68.0
BINS = (24, 16)


def draw_pitch(ax):
    ax.set_xlim(-2, PITCH_L + 2)
    ax.set_ylim(PITCH_W + 2, -2)          # y = 0 at the top, like the Tagger
    ax.set_aspect("equal")
    ax.axis("off")
    line = dict(color="white", lw=1.2, zorder=3)
    ax.plot([0, 0, PITCH_L, PITCH_L, 0], [0, PITCH_W, PITCH_W, 0, 0], **line)
    ax.plot([PITCH_L / 2, PITCH_L / 2], [0, PITCH_W], **line)
    ax.add_patch(plt.Circle((PITCH_L / 2, PITCH_W / 2), 9.15, fill=False, color="white", lw=1.2, zorder=3))
    for x0 in (0, PITCH_L - 16.5):
        ax.add_patch(plt.Rectangle((x0, 13.84), 16.5, 40.32, fill=False, color="white", lw=1.2, zorder=3))


def attacks_right(team: str, period: int, directions: dict[str, bool]) -> bool:
    key = str(int(period))
    if key not in directions:
        raise ValueError(
            f"No home attacking direction is recorded for period {period}. "
            "Add it to the tracking metadata; extra time must not be inferred."
        )
    home_right = directions[key]
    return home_right if team == "home" else not home_right


def accumulate(csv: Path, directions: dict[str, bool] | None, chunksize: int = 200_000):
    hists: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    cols = ["period", "team", "x", "y", "role"]
    header = pd.read_csv(csv, nrows=0).columns
    use = [c for c in cols if c in header]
    missing = {"team", "x", "y"} - set(use)
    if missing:
        raise SystemExit(f"CSV is missing columns {sorted(missing)}; found {list(header)}")
    dtypes = {"x": "float32", "y": "float32"}
    for chunk in pd.read_csv(csv, usecols=use, dtype=dtypes, chunksize=chunksize):
        if "role" in chunk:
            chunk = chunk[chunk["role"].isin(["player", "goalkeeper"])]
        chunk = chunk.dropna(subset=["x", "y", "team"])
        chunk = chunk[chunk["x"].between(0, PITCH_L) & chunk["y"].between(0, PITCH_W)]
        for team, g in chunk.groupby("team"):
            x, y = g["x"].to_numpy(), g["y"].to_numpy()
            if directions and team in ("home", "away") and "period" in g:
                flip = ~np.array([
                    attacks_right(team, int(p), directions) for p in g["period"]
                ])
                x = np.where(flip, PITCH_L - x, x)
                y = np.where(flip, PITCH_W - y, y)
            h, _, _ = np.histogram2d(x, y, bins=BINS, range=[[0, PITCH_L], [0, PITCH_W]])
            key = str(team)
            hists[key] = hists.get(key, 0) + h
            counts[key] = counts.get(key, 0) + len(x)
    return hists, counts


def save_heatmap(h: np.ndarray, title: str, out: Path):
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.imshow(np.sqrt(h.T), extent=[0, PITCH_L, PITCH_W, 0], cmap="inferno", alpha=0.85, zorder=1, aspect="auto")
    ax.set_facecolor("#15311f")
    fig.patch.set_facecolor("#15311f")
    draw_pitch(ax)
    ax.set_title(title, color="white", fontsize=14)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Hot-area maps from a tracking CSV.")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--meta", type=Path, help="tracking metadata JSON (defaults to <csv>.meta.json)")
    ap.add_argument("--home-dir", choices=["R", "L"], help="legacy fallback: home attacking direction in period 1")
    ap.add_argument("--out", default="data/heat", help="output path prefix")
    a = ap.parse_args()

    meta_path = a.meta or a.csv.with_suffix(".meta.json")
    directions = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = meta.get("home_attacks_right_by_period")
        if isinstance(raw, dict):
            directions = {}
            for period, value in raw.items():
                if isinstance(value, bool):
                    directions[str(period)] = value
                elif str(value).upper() in ("R", "L"):
                    directions[str(period)] = str(value).upper() == "R"
        elif meta.get("home_dir_1h") in ("R", "L"):
            right = meta["home_dir_1h"] == "R"
            directions = {"1": right, "2": not right}
    if not directions and a.home_dir:
        right = a.home_dir.upper() == "R"
        directions = {"1": right, "2": not right}
    if not directions and "period" in pd.read_csv(a.csv, nrows=0).columns:
        raise SystemExit(
            f"Tracking direction metadata not found in {meta_path}. "
            "Supply --meta or legacy --home-dir R/L."
        )

    hists, counts = accumulate(a.csv, directions)
    if not hists:
        print("No usable rows found.")
        return 1
    prefix = Path(a.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    note = "attacking →" if directions else "camera view"
    for team, h in hists.items():
        safe = "".join(ch for ch in team if ch.isalnum() or ch in "-_")
        save_heatmap(h, f"Hot areas — {team} ({counts[team]:,} positions, {note})",
                     prefix.with_name(prefix.name + f"_{safe}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
