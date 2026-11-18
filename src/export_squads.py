#!/usr/bin/env python3
"""Export the local club/player database (src/club_db.py SQLite) to a squads JSON
the Pitchside Tagger reads. Fully local — no cloud, no keys.

Usage:
  python src/export_squads.py                       # -> squads/clubs.json
  python src/export_squads.py --season 2026-27      # only that season's players
  python src/export_squads.py --out squads/clubs.json

Then in the Tagger: match Settings -> "Load squads JSON" -> pick this file ->
"From squads" on a lineup fills the players and sets the team name/colour.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.environ.get("FOOTBALL_DB", "data/football.db"))


def main():
    ap = argparse.ArgumentParser(description="Export clubs + players to a Tagger squads JSON.")
    ap.add_argument("--season", help="Only include players from this season (e.g. 2026-27).")
    ap.add_argument("--out", default="squads/clubs.json", help="Output path (default squads/clubs.json).")
    a = ap.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"Local DB not found at {DB_PATH}. Build it with src/club_db.py first.")
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    season = a.season
    if not season:  # default to the most recent season present
        row = con.execute("SELECT season FROM players ORDER BY season DESC LIMIT 1").fetchone()
        season = row["season"] if row else "2026-27"

    clubs = []
    for c in con.execute("SELECT id, name, short, color, league FROM clubs ORDER BY name"):
        players = [
            {"number": p["number"], "name": p["name"], "position": p["position"]}
            for p in con.execute(
                "SELECT number, name, position FROM players WHERE club_id=? AND season=? ORDER BY number",
                (c["id"], season),
            )
        ]
        clubs.append({
            "name": c["name"], "short": c["short"],
            "color": c["color"] or "#888888", "players": players,
        })

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"season": season, "clubs": clubs}, ensure_ascii=False, indent=2))
    total = sum(len(c["players"]) for c in clubs)
    print(f"Wrote {out} — {len(clubs)} clubs, {total} players (season {season}).")


if __name__ == "__main__":
    main()
