"""Team hot-area maps from a tracking position CSV (Track B output).

Input follows DATA_CONTRACT.md -> tracking CSV. Required columns:
    period, team, x, y
x, y are metres on a 105 x 68 pitch in the CAMERA frame (x = 0 is the goal on
the left of the broadcast picture). Rows with role other than player /
goalkeeper are skipped when a `role` column exists.

Each team is flipped to attack left -> right. By default the per-period map is
read from the tracking metadata next to the CSV (<csv>.meta.json, key
home_attacks_right_by_period — run_match.py writes it). Otherwise periods 1 and 2
can use --home-dir R/L, and extra time requires an explicit direction map:
    python src/heatmap_from_positions.py tracking.csv
    python src/heatmap_from_positions.py tracking.csv --home-dir R
    python src/heatmap_from_positions.py tracking.csv --home-attacks-right-by-period '{"1": true, "2": false, "3": false, "4": true}'

Memory stays flat on long matches: the CSV is read in chunks and only a small
histogram is kept, never the whole table.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

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


def attacks_right(
    team: str,
    period: int,
    home_dir: str | Mapping[str, bool] | None,
) -> bool:
    """Return whether this team's attack direction is rightward on screen.

    A mapping is explicit: {period_number_as_string: home_attacks_right}.
    A legacy R/L string is retained for ordinary periods 1 and 2 only.
    Extra-time periods 3 and 4 are never inferred from regulation time.
    """
    if team not in {"home", "away"}:
        raise ValueError(f"Unknown team side: {team!r}")
    if period not in {1, 2, 3, 4}:
        raise ValueError(f"Unsupported period {period}; expected 1..4")

    if isinstance(home_dir, Mapping):
        key = str(period)
        if key not in home_dir:
            raise ValueError(
                f"Home attack direction for period {period} is missing; "
                "extra-time direction must not be inferred."
            )
        home_right = home_dir[key]
        if not isinstance(home_right, bool):
            raise ValueError(f"Direction for period {period} must be a boolean.")
    elif isinstance(home_dir, str):
        if period >= 3:
            raise ValueError(
                f"Direction for period {period} must be supplied explicitly; "
                "extra-time direction must not be inferred."
            )
        direction = home_dir.upper()
        if direction not in {"R", "L"}:
            raise ValueError("home_dir must be 'R' or 'L'.")
        home_right_1h = direction == "R"
        home_right = home_right_1h if period == 1 else not home_right_1h
    elif home_dir is None:
        raise ValueError(
            f"Direction for period {period} is unavailable; "
            "extra-time direction must not be inferred."
        )
    else:
        raise TypeError("home_dir must be 'R', 'L', a per-period mapping, or None.")

    return home_right if team == "home" else not home_right


def accumulate(
    csv: Path,
    home_dir: str | Mapping[str, bool] | None,
    chunksize: int = 200_000,
):
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
            if home_dir and team in ("home", "away") and "period" in g:
                flip = ~np.array([attacks_right(team, int(p), home_dir) for p in g["period"]])
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


def directions_from_meta(meta_path: Path) -> dict[str, bool] | None:
    """home_attacks_right_by_period from tracking metadata (bool or "R"/"L" values)."""
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = meta.get("home_attacks_right_by_period")
    if isinstance(raw, dict) and raw:
        out: dict[str, bool] = {}
        for period, value in raw.items():
            if isinstance(value, bool):
                out[str(period)] = value
            elif str(value).upper() in ("R", "L"):
                out[str(period)] = str(value).upper() == "R"
        return out or None
    if meta.get("home_dir_1h") in ("R", "L"):
        right = meta["home_dir_1h"] == "R"
        return {"1": right, "2": not right}
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Hot-area maps from a tracking CSV.")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--home-dir", choices=["R", "L"], help="which way home attacks on screen in the 1st half; infers period 2 only")
    ap.add_argument(
        "--home-attacks-right-by-period",
        help='JSON object mapping periods to whether home attacks right, e.g. \'{"1": true, "2": false, "3": false, "4": true}\' (required for extra time)',
    )
    ap.add_argument("--meta", type=Path, help="tracking metadata JSON (default: <csv>.meta.json)")
    ap.add_argument("--out", default="data/heat", help="output path prefix")
    a = ap.parse_args()

    if a.home_dir and a.home_attacks_right_by_period:
        ap.error("Use either --home-dir or --home-attacks-right-by-period, not both.")
    if a.home_attacks_right_by_period:
        try:
            direction_map = json.loads(a.home_attacks_right_by_period)
        except json.JSONDecodeError as exc:
            ap.error(f"Invalid direction JSON: {exc}")
        if not isinstance(direction_map, dict):
            ap.error("--home-attacks-right-by-period must be a JSON object.")
        home_directions = direction_map
    elif a.home_dir:
        home_directions = a.home_dir
    else:
        home_directions = directions_from_meta(a.meta or a.csv.with_suffix(".meta.json"))
        if home_directions is None and "period" in pd.read_csv(a.csv, nrows=0).columns:
            print("No direction in the tracking metadata and no --home-dir: drawing in camera view.")

    hists, counts = accumulate(a.csv, home_directions)
    if not hists:
        print("No usable rows found.")
        return 1
    prefix = Path(a.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    note = "attacking →" if home_directions else "camera view"
    for team, h in hists.items():
        safe = "".join(ch for ch in team if ch.isalnum() or ch in "-_")
        save_heatmap(h, f"Hot areas — {team} ({counts[team]:,} positions, {note})",
                     prefix.with_name(prefix.name + f"_{safe}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
