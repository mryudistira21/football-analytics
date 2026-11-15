"""Expected threat (xT) — same grid the Pitchside Tagger uses.

A 16 x 12 grid over a 105 x 68 m pitch (attacking left -> right). Each cell
holds the probability that having the ball there leads to a goal within the
next few actions. Fitted by value iteration on 230 StatsBomb open-data matches
(WC 2018 & 2022, Euro 2020 & 2024; ~432k passes and carries) — see
scripts/fit_xt.py to refit.

A completed pass or cross earns xT(end) - xT(start). Failed passes earn 0.

Also here: progressive-pass and into-the-box rules, shared with the Tagger.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
_GRID_PATH = Path(__file__).resolve().parent.parent / "models" / "xt_grid.json"
_D = json.loads(_GRID_PATH.read_text())
GRID: list[list[float]] = _D["grid"]
NX: int = _D["nx"]
NY: int = _D["ny"]


def xt_at(x: float, y: float) -> float:
    """xT value of the cell containing (x, y), metres."""
    i = min(NX - 1, max(0, int(x / PITCH_LENGTH * NX)))
    j = min(NY - 1, max(0, int(y / PITCH_WIDTH * NY)))
    return GRID[j][i]


def xt_added(x: float, y: float, ex: float, ey: float,
             completed: bool = True) -> float:
    """Threat added by moving the ball from (x, y) to (ex, ey)."""
    return xt_at(ex, ey) - xt_at(x, y) if completed else 0.0


def _goal_dist(x: float, y: float) -> float:
    return math.hypot(PITCH_LENGTH - x, PITCH_WIDTH / 2 - y)


def is_progressive(x: float, y: float, ex: float, ey: float) -> bool:
    """Completed pass that moves the ball >= 30 m closer to goal inside own
    half, >= 15 m when crossing halfway, >= 10 m inside the opponent half."""
    gain = _goal_dist(x, y) - _goal_dist(ex, ey)
    s_own, e_own = x < PITCH_LENGTH / 2, ex < PITCH_LENGTH / 2
    need = 30 if (s_own and e_own) else 15 if (s_own and not e_own) else 10
    return gain >= need


def in_box(x: float, y: float) -> bool:
    return x >= 88.5 and 13.84 <= y <= 54.16


def is_into_box(x: float, y: float, ex: float, ey: float) -> bool:
    return in_box(ex, ey) and not in_box(x, y)


if __name__ == "__main__":
    print("xT in front of goal (100, 34):", xt_at(100, 34))
    print("xT own half (30, 34):        ", xt_at(30, 34))
    print("pass 60,34 -> 95,34 adds:    ", round(xt_added(60, 34, 95, 34), 4))
    print("progressive 40->70? ", is_progressive(40, 34, 70, 34))
