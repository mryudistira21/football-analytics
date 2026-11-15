"""Fit the expected-threat (xT) grid from StatsBomb open data.

Writes models/xt_grid.json, which src/xt_model.py and the Pitchside Tagger use.

    python scripts/fit_xt.py                # default 4 tournaments, ~230 matches
    python scripts/fit_xt.py --workers 4    # fewer parallel downloads (less RAM/network)

Method (Karun Singh's xT): split the pitch into 16 x 12 cells. For each cell,
count shots, goals, and open-play "moves" (passes and carries) plus where
successful moves end. Then iterate
    xT = P(shoot) * P(goal | shot) + P(move) * sum(T(cell -> cell') * xT(cell'))
until it converges.

Open play only, defined precisely by StatsBomb's event-level `play_pattern`:
every event (pass, carry AND shot) is kept only when its play_pattern is one of
OPEN_PATTERNS below. This drops the WHOLE set-piece possession, not just the
delivery, so a carry or shot that follows a corner is excluded too. Penalties
are skipped as well.

Notes
- StatsBomb stores every event with the acting team attacking left -> right
  (towards x = 120), in both halves, so no per-team flip is needed. The fitted
  grid confirms it: threat peaks at one end only.
- Memory stays low: each match is reduced to small count arrays as soon as it
  downloads; raw events are never kept.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
COMPETITIONS = [(43, 106), (55, 282), (55, 43), (43, 3)]   # WC22, Euro24, Euro20, WC18
# Open play = these play_patterns. Everything else (From Corner / Free Kick /
# Throw In / Goal Kick, and the ambiguous "Other") is a set-piece possession
# and is excluded whole.
OPEN_PATTERNS = {"Regular Play", "From Counter", "From Keeper", "From Kick Off"}
NX, NY = 16, 12


def get_json(path: str):
    with urllib.request.urlopen(BASE + path, timeout=180) as r:
        return json.load(r)


def cell(x: float, y: float) -> int:
    """StatsBomb pitch is 120 x 80, acting team attacking towards x = 120."""
    i = min(NX - 1, max(0, int(x / 120 * NX)))
    j = min(NY - 1, max(0, int(y / 80 * NY)))
    return j * NX + i


def count_match(match_id: int):
    n = NX * NY
    shots, goals, moves = np.zeros(n), np.zeros(n), np.zeros(n)
    trans = np.zeros((n, n), dtype=np.float32)
    try:
        events = get_json(f"events/{match_id}.json")
    except Exception as exc:  # network hiccup: skip this match, don't crash the fit
        print(f"skip {match_id}: {exc}", file=sys.stderr)
        return None
    for e in events:
        loc = e.get("location")
        if not loc:
            continue
        if e.get("play_pattern", {}).get("name") not in OPEN_PATTERNS:
            continue                                    # whole set-piece possession excluded
        c = cell(*loc[:2])
        kind = e["type"]["name"]
        if kind == "Shot":
            if e["shot"]["type"]["name"] == "Penalty":
                continue
            shots[c] += 1
            goals[c] += e["shot"]["outcome"]["name"] == "Goal"
        elif kind == "Pass":
            p = e["pass"]
            moves[c] += 1
            if "outcome" not in p:                      # no outcome = completed
                trans[c, cell(*p["end_location"][:2])] += 1
        elif kind == "Carry":
            moves[c] += 1
            trans[c, cell(*e["carry"]["end_location"][:2])] += 1
    return shots, goals, moves, trans


def fit(workers: int = 12) -> dict:
    match_ids = []
    for comp, season in COMPETITIONS:
        match_ids += [m["match_id"] for m in get_json(f"matches/{comp}/{season}.json")]
    n = NX * NY
    S, G, M = np.zeros(n), np.zeros(n), np.zeros(n)
    T = np.zeros((n, n))
    used = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for res in ex.map(count_match, match_ids):
            if res is None:
                continue
            s, g, m, t = res
            S += s; G += g; M += m; T += t
            used += 1

    total = S + M
    p_shot = np.divide(S, total, out=np.zeros_like(S), where=total > 0)
    p_move = np.divide(M, total, out=np.zeros_like(M), where=total > 0)
    p_goal = np.divide(G, S, out=np.zeros_like(G), where=S > 0)
    T = np.divide(T, M[:, None], out=np.zeros_like(T), where=M[:, None] > 0)

    xt = np.zeros(n)
    for _ in range(100):
        new = p_shot * p_goal + p_move * (T @ xt)
        if np.abs(new - xt).max() < 1e-7:
            xt = new
            break
        xt = new
    grid = xt.reshape(NY, NX)
    grid = (grid + grid[::-1, :]) / 2          # mirror left/right wings to cut noise
    print(f"matches {used}  shots {int(S.sum())}  open-play moves {int(M.sum())}  "
          f"max xT {grid.max():.4f}", file=sys.stderr)
    return {
        "nx": NX, "ny": NY,
        "grid": [[round(float(v), 5) for v in row] for row in grid],
        "source": (f"StatsBomb open data: WC2018, WC2022, Euro2020, Euro2024; {used} matches; "
                   "open-play possessions only (play_pattern in " + ", ".join(sorted(OPEN_PATTERNS)) + ")"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit the xT grid.")
    ap.add_argument("--workers", type=int, default=12, help="parallel downloads")
    a = ap.parse_args()
    model = fit(a.workers)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "xt_grid.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(model, f)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
