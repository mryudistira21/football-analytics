#!/usr/bin/env python3
"""Sync the local club/player database (src/club_db.py SQLite) to Supabase, and
pull squads back out. Scope: clubs + players only — the shared online club DB
that feeds the Tagger's lineup chips.

No third-party dependencies (urllib + sqlite3 from the standard library).

Setup:
  1. Run supabase/schema.sql in the Supabase SQL editor.
  2. Export your keys (do NOT put the service key in the browser/Tagger):
       setx SUPABASE_URL "https://xxxx.supabase.co"        (Windows)
       setx SUPABASE_SERVICE_KEY "eyJ...service_role..."
     or export them in bash. The service_role key is needed to WRITE (push);
     reads (squad/list) work with either the service key or the anon key.

Usage:
  python src/club_supabase.py push                       # local SQLite -> Supabase
  python src/club_supabase.py list-clubs                 # what's online
  python src/club_supabase.py squad BTL --season 2026-27 # print "number name" lines
  python src/club_supabase.py squad "Persika" > persika.txt   # save for the Tagger
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DB_PATH = Path(os.environ.get("FOOTBALL_DB", "data/football.db"))


def _env():
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_ANON_KEY for reads).")
    return url, key


def sb(method: str, path: str, *, body=None, upsert_on: str | None = None):
    url, key = _env()
    full = f"{url}/rest/v1/{path}"
    if upsert_on:
        sep = "&" if "?" in full else "?"
        full = f"{full}{sep}on_conflict={upsert_on}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if method == "POST" and upsert_on:
        headers["Prefer"] = "resolution=merge-duplicates,return=representation"
    elif method == "POST":
        headers["Prefer"] = "return=representation"
    req = urllib.request.Request(full, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw.strip() else []
    except urllib.error.HTTPError as e:
        sys.exit(f"Supabase {method} {path} failed [{e.code}]: {e.read().decode()[:400]}")
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach Supabase: {e}")


def _local_con():
    if not DB_PATH.exists():
        sys.exit(f"Local DB not found at {DB_PATH}. Build it first with src/club_db.py.")
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def cmd_push(_a):
    con = _local_con()
    clubs = [dict(r) for r in con.execute("SELECT name, short, color, league FROM clubs")]
    if clubs:
        sb("POST", "clubs", body=clubs, upsert_on="name")
    print(f"Upserted {len(clubs)} clubs.")

    # Map club name/short -> Supabase id (ids differ from the local SQLite ids).
    remote = sb("GET", "clubs?select=id,name,short")
    by_name = {c["name"]: c["id"] for c in remote}
    by_short = {c["short"]: c["id"] for c in remote if c.get("short")}

    rows = con.execute(
        "SELECT c.name AS cname, c.short AS cshort, p.season, p.number, p.name, p.position "
        "FROM players p JOIN clubs c ON c.id = p.club_id"
    ).fetchall()
    players, missing = [], 0
    for r in rows:
        cid = by_short.get(r["cshort"]) or by_name.get(r["cname"])
        if cid is None:
            missing += 1
            continue
        players.append({
            "club_id": cid, "season": r["season"], "number": r["number"],
            "name": r["name"], "position": r["position"],
        })
    if players:
        sb("POST", "players", body=players, upsert_on="club_id,season,number")
    print(f"Upserted {len(players)} players." + (f" ({missing} skipped — club not matched)" if missing else ""))


def _find_club(key: str):
    q = urllib.parse.quote(key)
    hit = sb("GET", f"clubs?select=id,name,short&or=(short.ilike.{q},name.ilike.%{q}%)&limit=1")
    if not hit:
        sys.exit(f"No club in Supabase matches '{key}'. Run `push` first, or check the name.")
    return hit[0]


def cmd_squad(a):
    club = _find_club(a.club)
    flt = f"players?select=number,name,position&club_id=eq.{club['id']}&order=number"
    if a.season:
        flt += f"&season=eq.{urllib.parse.quote(a.season)}"
    players = sb("GET", flt)
    if not players:
        sys.exit(f"No players for {club['name']}" + (f" in {a.season}" if a.season else "") + ".")
    for p in players:
        print(f"{p['number']} {p['name']}")
    print(f"# {len(players)} players — {club['name']}", file=sys.stderr)


def cmd_list(_a):
    for c in sb("GET", "clubs?select=name,short,grp&order=name"):
        print(f"{c.get('short') or '—':>5}  {c['name']}" + (f"  [grp {c['grp']}]" if c.get("grp") else ""))


def main():
    ap = argparse.ArgumentParser(description="Sync clubs/players between local SQLite and Supabase.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("push", help="Upsert local clubs+players into Supabase").set_defaults(func=cmd_push)
    sub.add_parser("list-clubs", help="List clubs in Supabase").set_defaults(func=cmd_list)
    sq = sub.add_parser("squad", help="Print 'number name' lines for a club")
    sq.add_argument("club", help="Club short code or name")
    sq.add_argument("--season", help="Season, e.g. 2026-27")
    sq.set_defaults(func=cmd_squad)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
