"""Expected goals (xG) — single source of truth for the whole project.

Logistic regression fitted on 5,606 non-penalty shots from StatsBomb open data
(World Cup 2018 & 2022, Euro 2020 & 2024). Same coefficients as the Pitchside
Tagger web app, so tagged shots and pipeline-detected shots score identically.

Pitch coordinates: metres, 105 x 68. Attacking goal at x = 105, centre y = 34.

    logit = -1.406 - 0.1056*distance + 1.383*angle - 1.125*header + 0.525*free_kick
    penalty = 0.76 (fixed)

Broadcast-level only: no goalkeeper position, no defensive pressure. Treat a
single shot's value as rough; totals over many shots are what mean something.
"""
from __future__ import annotations
import math

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_WIDTH = 7.32
PENALTY_XG = 0.76

_B0 = -1.40636605
_B_DIST = -0.10564557
_B_ANGLE = 1.38289241
_B_HEAD = -1.12515185
_B_FK = 0.52487711


def shot_angle(x: float, y: float) -> float:
    """Angle (radians) the goal mouth subtends from shot location (x, y)."""
    dx = PITCH_LENGTH - x
    dy = abs(PITCH_WIDTH / 2 - y)
    a = math.atan2(GOAL_WIDTH * dx, dx * dx + dy * dy - (GOAL_WIDTH / 2) ** 2)
    return a + math.pi if a < 0 else a


def shot_distance(x: float, y: float) -> float:
    """Distance (metres) from (x, y) to the centre of the attacking goal."""
    return math.hypot(PITCH_LENGTH - x, PITCH_WIDTH / 2 - y)


def xg(x: float, y: float, *, header: bool = False,
       situation: str = "open") -> float:
    """Expected-goal value for a shot.

    situation: "open", "fk" (direct free kick) or "pen" (penalty).
    """
    if situation == "pen":
        return PENALTY_XG
    z = (_B0
         + _B_DIST * shot_distance(x, y)
         + _B_ANGLE * shot_angle(x, y)
         + _B_HEAD * (1 if header else 0)
         + _B_FK * (1 if situation == "fk" else 0))
    return 1.0 / (1.0 + math.exp(-z))


if __name__ == "__main__":
    # Self-check against the values quoted in the Tagger.
    print("central 11 m, foot:", round(xg(94, 34), 3), "(expect ~0.157)")
    print("central 11 m, head:", round(xg(94, 34, header=True), 3), "(expect ~0.057)")
    print("central  6 m, foot:", round(xg(99, 34), 3))
    print("penalty           :", round(xg(94, 34, situation="pen"), 3))
