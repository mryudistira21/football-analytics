#!/usr/bin/env python3
"""Build hub/data.json for the Liga Nusantara Hub from squads/clubs.json plus the
scraped group / standings / scorers / fixtures snapshot (ileague.id, 2026-11-18)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
clubs = json.loads((ROOT / "squads" / "clubs.json").read_text())["clubs"]
by_short = {c["short"]: c for c in clubs}

# standings snapshot in EXACT ileague order per group (rank = position in the list,
# which already reflects the goal-difference tiebreak): (short, points)
STANDINGS = {
    "A": [("PKB",6),("PSK",3),("BAT",3),("PKT",3),("PKO",3),("PKN",0),("PJU",0)],
    "B": [("PSF",6),("PKA",4),("PBG",3),("BTL",2),("PIP",2),("GRS",2),("PBO",1),("PNG",0)],
    "C": [("PKP",6),("PSD",4),("PSU",3),("NGA",2),("PPI",2),("PKU",1),("DON",1),("PNB",1)],
}
GROUPS = {g: [s for s, _ in rows] for g, rows in STANDINGS.items()}
PTS = {s: p for rows in STANDINGS.values() for s, p in rows}
SCORERS = [
    {"name":"JOHAN YOGA","club":"PSF","goals":3},
    {"name":"FARID. F","club":"PSU","goals":2},
    {"name":"BETO","club":"PKP","goals":2},
    {"name":"A. SIREGAR","club":"BAT","goals":2},
    {"name":"DANYL TUKA","club":"PIP","goals":2},
]
# fixtures snapshot: (date, home short, home goals, away goals, away short, venue)
FIXTURES = [
    ("2026-11-17","PKN",0,2,"PKB","Stadion Galuh Ciamis"),
    ("2026-11-17","PKT",1,0,"PKO","Trisanja"),
    ("2026-11-17","PSK",3,0,"PJU","Kebogiro"),
    ("2026-11-17","PIP",2,2,"BTL","Moch. Soebroto"),
    ("2026-11-17","GRS",1,1,"PKA","Gelora Joko Samudro"),
    ("2026-11-18","PNG",1,2,"PSF","Stadion Ketonggo Ngawi"),
    ("2026-11-18","PBG",2,1,"PBO","Trisanja"),
    ("2026-11-18","PSU",0,1,"PKP","UNS"),
    ("2026-11-18","PNB",1,1,"PSD","Brawijaya"),
    ("2026-11-18","NGA",3,3,"DON","Gelora Joko Samudro"),
    ("2026-11-18","PKU",1,1,"PPI","Kudungga Sangatta"),
]

# attach group to each club
group_of = {s: g for g, ss in GROUPS.items() for s in ss}
for c in clubs:
    c["group"] = group_of.get(c["short"], "")

standings = {}
for g, rows in STANDINGS.items():
    standings[g] = [{"short": s, "name": by_short[s]["name"], "pts": p, "rank": i + 1}
                    for i, (s, p) in enumerate(rows)]

fixtures = [{"date": d, "home": h, "hg": hg, "ag": ag, "away": a, "venue": v,
             "round": "Pekan 2"} for (d, h, hg, ag, a, v) in FIXTURES]

scorers = [{"name": s["name"], "club": s["club"],
            "clubName": by_short[s["club"]]["name"], "goals": s["goals"]} for s in SCORERS]

out = ROOT / "hub" / "data.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "competition": "Liga Nusantara 2026-27",
    "updated": "2026-11-18",
    "source": "ileague.id",
    "note": "Snapshot. Standings show points only (ileague lists no W/D/L/GF/GA). Fixtures = Pekan 2 results. Kit colours are placeholders. Rebuild with scripts/build_hub_data.py.",
    "clubs": clubs,
    "standings": standings,
    "scorers": scorers,
    "fixtures": fixtures,
}, ensure_ascii=False, indent=2))
print(f"Wrote {out} — {len(clubs)} clubs, {sum(len(v) for v in standings.values())} standings rows, "
      f"{len(fixtures)} fixtures, {len(scorers)} scorers.")
