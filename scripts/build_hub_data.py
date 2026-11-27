#!/usr/bin/env python3
"""Build hub/data.json for the Liga Nusantara Hub.

Inputs
  data/clubs.json             squads (ileague club pages)
  data/groupb_matches.json    Group B results, line-ups, scorers (scripts/build_groupb_data.py)
  data/players_to_watch.json  1-3 players per club (scripts/players_to_watch.py)
  matches/persiba_2026-27.json + matches/analysis_status.json  -> the Analysis status tab
  + the ileague.id snapshot below (Groups A/C standings, top scorers, Pekan 5 fixtures; 27 Nov 2026)

    python scripts/build_hub_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
clubs = json.loads((DATA / "clubs.json").read_text())["clubs"]
by_short = {c["short"]: c for c in clubs}
groupb = json.loads((DATA / "groupb_matches.json").read_text(encoding="utf-8"))
ptw = json.loads((DATA / "players_to_watch.json").read_text(encoding="utf-8"))
config = json.loads((ROOT / "matches" / "persiba_2026-27.json").read_text(encoding="utf-8"))
ledger = json.loads((ROOT / "matches" / "analysis_status.json").read_text(encoding="utf-8"))["matches"]
UPDATED = "2026-11-27"

# ileague standings snapshot, exact order (rank = position). Group B is rebuilt from the
# match data (with W/D/L/GF/GA) and checked against this list.
STANDINGS_SNAPSHOT = {
    "A": [("PKB", 9), ("PSK", 7), ("BAT", 6), ("PKO", 6), ("PKT", 5), ("PKN", 1), ("PJU", 0)],
    "B": [("PSF", 12), ("PKA", 10), ("GRS", 6), ("PNG", 4), ("PBG", 4), ("PBO", 2), ("BTL", 2), ("PIP", 2)],
    "C": [("PKP", 10), ("PSD", 8), ("PKU", 5), ("PSU", 4), ("NGA", 4), ("PPI", 3), ("PNB", 3), ("DON", 2)],
}
SCORERS = [
    {"name": "FERRY CAHYO", "club": "BAT", "goals": 4},
    {"name": "JOHAN YOGA UTAMA", "club": "PSF", "goals": 4},
    {"name": "FARID F.", "club": "PSU", "goals": 3},
    {"name": "BETO", "club": "PKP", "goals": 3},
    {"name": "ALDON LALU", "club": "NGA", "goals": 3},
]
# Groups A/C: only the Pekan 2 results from the first snapshot are kept (B is complete).
OLD_AC_RESULTS = [
    ("2026-11-17", "PKN", 0, 2, "PKB", "Stadion Galuh Ciamis"), ("2026-11-17", "PKT", 1, 0, "PKO", "Trisanja"),
    ("2026-11-17", "PSK", 3, 0, "PJU", "Kebogiro"), ("2026-11-18", "PSU", 0, 1, "PKP", "UNS"),
    ("2026-11-18", "PNB", 1, 1, "PSD", "Brawijaya"), ("2026-11-18", "NGA", 3, 3, "DON", "Gelora Joko Samudro"),
    ("2026-11-18", "PKU", 1, 1, "PPI", "Kudungga Sangatta"),
]
PEKAN5 = [  # (date, time WIB or "", home, away, venue)
    ("2026-11-29", "", "PKN", "PSK", ""), ("2026-11-29", "", "PKT", "PJU", ""),
    ("2026-11-30", "15:00", "PNG", "PBO", "Stadion Ketonggo, Ngawi"), ("2026-11-30", "15:30", "GRS", "PIP", "Gelora Joko Samudro"),
    ("2026-11-30", "", "PKB", "BAT", ""), ("2026-11-30", "", "PPI", "DON", ""),
    ("2026-11-30", "15:30", "PBG", "BTL", "Stadion Trisanja, Tegal"), ("2026-11-30", "", "PNB", "PKU", ""),
    ("2026-11-30", "", "PSU", "NGA", ""), ("2026-11-30", "19:00", "PKA", "PSF", "Stadion Manahan, Solo"),
    ("2026-11-30", "", "PSD", "PKP", ""),
]

group_of = {s: g for g, rows in STANDINGS_SNAPSHOT.items() for s, _ in rows}
for c in clubs:
    c["group"] = group_of.get(c["short"], "")

# ---------------------------------------------------------------- standings
standings = {}
for g, rows in STANDINGS_SNAPSHOT.items():
    standings[g] = [{"short": s, "name": by_short[s]["name"], "pts": p, "rank": i + 1} for i, (s, p) in enumerate(rows)]
detail = {r["club"]: r for r in groupb["standings"]}
# ileague's order wins: PERSIBA and Persibo are level on points, GD and goals; ileague's
# tiebreak (not published) puts Persibo above PERSIBA.
assert sorted(detail) == sorted(s for s, _ in STANDINGS_SNAPSHOT["B"]), "Group B clubs differ from ileague"
for r in standings["B"]:
    d = detail[r["short"]]
    assert d["pts"] == r["pts"], r["short"]
    r.update({k: d[k] for k in ("p", "w", "d", "l", "gf", "ga", "gd", "gf_1h", "gf_2h", "ga_1h", "ga_2h")})

# ---------------------------------------------------------------- fixtures
def gname(short: str) -> str:
    return by_short[short]["name"]

fixtures = []
for d, h, hg, ag, a, v in OLD_AC_RESULTS:
    fixtures.append({"date": d, "home": h, "hg": hg, "ag": ag, "away": a, "venue": v, "round": "Pekan 2", "group": group_of[h]})
for m in groupb["matches"]:
    scorers = [f"{g['name'].title()} {g['minute']}'" + (" (p)" if g["note"] == "pen" else "")
               for g in m["goals"]]
    fixtures.append({"date": m["date"], "home": m["home"], "hg": m["score"][0], "ag": m["score"][1], "away": m["away"],
                     "venue": m["venue"], "round": f"Pekan {m['round']}", "group": "B",
                     "scorers_home": [s for s, g in zip(scorers, m["goals"]) if g["side"] == "h"],
                     "scorers_away": [s for s, g in zip(scorers, m["goals"]) if g["side"] == "a"]})
for d, t, h, a, v in PEKAN5:
    fixtures.append({"date": d, "time": t, "home": h, "hg": None, "ag": None, "away": a, "venue": v,
                     "round": "Pekan 5", "group": group_of[h]})
fixtures.sort(key=lambda f: (f["date"], f.get("time") or "", f["home"]))

scorers = [{"name": s["name"], "club": s["club"], "clubName": gname(s["club"]), "goals": s["goals"]} for s in SCORERS]
persiba_schedule = [{"round": f["round"], "date": f["date"], "home": f["home"], "away": f["away"]} for f in groupb["persiba_remaining"]]

# ---------------------------------------------------------------- analysis status
name_to_short = {v: k for k, v in groupb["teams"].items()}
name_to_short.update({"Gresik United FC": "GRS", "Persipa Pati": "PIP"})
cfg_by_key = {}
for m in config["matches"]:
    h, a = name_to_short.get(m["home"]), name_to_short.get(m["away"])
    if h and a:
        cfg_by_key[(m["date"], h, a)] = m
NEXT_OPP = "PBG"


def stage_of(m: dict | None) -> tuple[str, str]:
    if m is None:
        return "not_set_up", "Not set up"
    led = ledger.get(m["match_id"], {})
    if led.get("review") == "done":
        return "reviewed", "Analysed + reviewed"
    if led.get("track_b") == "done":
        return "analysed", "Analysed (Track B)"
    if led.get("track_b") == "queued":
        return "queued", f"Queued (Kaggle batch {led.get('batch', '?')})"
    if not m.get("video_url"):
        return "needs_video", "Needs video"
    if not all((m.get("periods") or {}).get(p) for p in ("1", "2")):
        return "needs_times", "Needs period times"
    return "ready", "Ready to run" + (f" (batch {led['batch']})" if led.get("batch") else "")


status = []
for f in fixtures:
    if f["group"] != "B":
        continue
    m = cfg_by_key.get((f["date"], f["home"], f["away"]))
    stage, label = stage_of(m)
    if m is None and f["hg"] is None:
        stage, label = "upcoming", "Not played yet"
    why = []
    if "BTL" in (f["home"], f["away"]):
        why.append("PERSIBA match")
    if NEXT_OPP in (f["home"], f["away"]) and f["hg"] is not None:
        why.append("Scouting: next opponent")
    status.append({"match_id": m["match_id"] if m else None, "date": f["date"], "round": f["round"],
                   "home": f["home"], "away": f["away"], "hg": f["hg"], "ag": f["ag"], "played": f["hg"] is not None,
                   "stage": stage, "label": label, "why": why,
                   "notes": ledger.get(m["match_id"], {}).get("notes", "") if m else ""})
# matches set up in the config but outside the fixture list (none today) are appended for completeness
known = {s["match_id"] for s in status if s["match_id"]}
for m in config["matches"]:
    if m["match_id"] not in known:
        stage, label = stage_of(m)
        status.append({"match_id": m["match_id"], "date": m["date"], "round": m.get("round", ""),
                       "home": name_to_short.get(m["home"], m["home"]), "away": name_to_short.get(m["away"], m["away"]),
                       "hg": (m.get("score") or [None, None])[0], "ag": (m.get("score") or [None, None])[1],
                       "played": bool(m.get("score")), "stage": stage, "label": label, "why": [], "notes": ""})

meta = {"competition": "Liga Nusantara 2026-27", "updated": UPDATED, "source": "ileague.id",
        "note": "Group B: full results, scorers and W/D/L from the ileague match pages (rounds 1-4). Groups A/C: "
                "points from ileague; only Pekan 2 results kept. Kit colours are placeholders. Rebuild with "
                "scripts/build_hub_data.py."}
players = [{"number": p["number"], "name": p["name"], "position": p.get("position", ""),
            "club": c["short"], "clubName": c["name"]} for c in clubs for p in c["players"]]


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


dump(DATA / "standings.json", {**meta, "standings": standings})
dump(DATA / "fixtures.json", {**meta, "fixtures": fixtures})
dump(DATA / "scorers.json", {**meta, "scorers": scorers})
dump(DATA / "players.json", {**meta, "players": players})
hub = ROOT / "hub" / "data.json"
dump(hub, {**meta, "clubs": clubs, "standings": standings, "scorers": scorers, "fixtures": fixtures,
           "players_to_watch": ptw["clubs"], "ptw_basis": ptw["basis"], "persiba_schedule": persiba_schedule,
           "status": status, "next_match": {"date": "2026-11-30", "time": "15:30 WIB", "home": "PBG", "away": "BTL",
                                            "venue": "Stadion Trisanja, Tegal", "round": "Pekan 5"}})
print(f"Wrote hub/data.json — {len(clubs)} clubs, {len(fixtures)} fixtures, {len(status)} status rows, "
      f"{sum(len(c['players']) for c in ptw['clubs'].values())} players to watch.")
for s in status:
    print(f"  {s['date']} {s['round']:8} {s['home']}-{s['away']} {'' if s['hg'] is None else str(s['hg'])+'-'+str(s['ag'])}"
          f"  -> {s['label']}  {', '.join(s['why'])}")
