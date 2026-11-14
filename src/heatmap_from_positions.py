"""Build team hot-area maps from a per-frame position CSV.

Expects the CSV that the Track B export produces (ChatGPT's script), with at
least these columns:

    frame, team, x, y

where x,y are real pitch coordinates in metres (105 x 68). `team` is any label
(e.g. "home"/"away" or a colour). One row per player per frame.

Usage:
    python src/heatmap_from_positions.py data/positions.csv --out data/heat

Produces one PNG per team. No GPU needed. This is the downstream half of Track
B: ChatGPT exports positions, this turns them into the same hot-area view as
the Tagger.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PITCH_L, PITCH_W = 105.0, 68.0


def draw_pitch(ax):
    ax.set_xlim(-2, PITCH_L + 2)
    ax.set_ylim(-2, PITCH_W + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    line = dict(color="white", lw=1.2, zorder=3)
    ax.plot([0, 0, PITCH_L, PITCH_L, 0], [0, PITCH_W, PITCH_W, 0, 0], **line)
    ax.plot([PITCH_L / 2, PITCH_L / 2], [0, PITCH_W], **line)
    c = plt.Circle((PITCH_L / 2, PITCH_W / 2), 9.15, fill=False,
                   color="white", lw=1.2, zorder=3)
    ax.add_patch(c)
    for x0 in (0, PITCH_L - 16.5):
        ax.add_patch(plt.Rectangle((x0, 13.84), 16.5, 40.32, fill=False,
                                   color="white", lw=1.2, zorder=3))


def heatmap(df_team: pd.DataFrame, title: str, out_path: Path,
            bins=(24, 16)):
    h, xedges, yedges = np.histogram2d(
        df_team["x"], df_team["y"], bins=bins,
        range=[[0, PITCH_L], [0, PITCH_W]])
    h = h.T
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.imshow(np.sqrt(h), extent=[0, PITCH_L, 0, PITCH_W], origin="lower",
              cmap="inferno", alpha=0.85, zorder=1, aspect="auto")
    ax.set_facecolor("#15311f")
    fig.patch.set_facecolor("#15311f")
    draw_pitch(ax)
    ax.set_title(title, color="white", fontsize=14)
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", out_path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--out", default="data/heat", help="output path prefix")
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    need = {"team", "x", "y"}
    if not need.issubset(df.columns):
        print("CSV must have columns:", need, "— found:", list(df.columns))
        return 1
    df = df[(df.x.between(0, PITCH_L)) & (df.y.between(0, PITCH_W))]

    out_prefix = Path(a.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for team, g in df.groupby("team"):
        safe = "".join(ch for ch in str(team) if ch.isalnum() or ch in "-_")
        heatmap(g, f"Hot areas — {team}  ({len(g)} positions)",
                out_prefix.with_name(out_prefix.name + f"_{safe}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
