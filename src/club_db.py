"""Local club & player database (SQLite). Stays on your PC, never pushed.

The database file lives at data/football.db by default (git-ignored), or
wherever FOOTBALL_DB points:   setx FOOTBALL_DB D:\\Video-Bola-Analis\\football.db

It feeds the Tagger: export a club's lineup to a text file, then in the
Tagger's match setup click "Load from file" under the lineup box.

Commands (run from the repo folder):
  python src/club_db.py add-club "PERSIBA Bantul" --short BTL --color "#d7263d" --league "Liga Nusantara"
  python src/club_db.py add-player BTL 10 "Player Name" --position AM
  python src/club_db.py import-players BTL players.csv        # columns: number,name[,position]
  python src/club_db.py clubs
  python src/club_db.py players BTL
  python src/club_db.py remove-player BTL 10
  python src/club_db.py lineup BTL                           # writes data/lineups/BTL.txt
  python src/club_db.py lineup BTL --numbers 1,4,5,6,8,10,7,11,9,2,3
  python src/club_db.py add-match BTL GRS --date 2026-11-13 --competition "Liga Nusantara" --video D:\\Video-Bola-Analis\\test20.mp4
  python src/club_db.py matches

Season defaults to 2026/27; change with --season or FOOTBALL_SEASON.
SQLite is a single small file and uses almost no memory.
"""
from __future__ import annotations
import argparse
import csv
import os
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("FOOTBALL_DB") or REPO / "data" / "football.db")
SEASON = os.environ.get("FOOTBALL_SEASON", "2026/27")

SCHEMA = """
CREATE TABLE IF NOT EXISTS clubs (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  short TEXT UNIQUE,
  color TEXT,
  league TEXT
);
CREATE TABLE IF NOT EXISTS players (
  id INTEGER PRIMARY KEY,
  club_id INTEGER NOT NULL REFERENCES clubs(id) ON DELETE CASCADE,
  season TEXT NOT NULL,
  number INTEGER NOT NULL,
  name TEXT NOT NULL,
  position TEXT,
  UNIQUE (club_id, season, number)
);
CREATE TABLE IF NOT EXISTS matches (
  id INTEGER PRIMARY KEY,
  date TEXT,
  competition TEXT,
  home_id INTEGER REFERENCES clubs(id),
  away_id INTEGER REFERENCES clubs(id),
  video_path TEXT,
  events_csv TEXT,
  notes TEXT
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def find_club(con: sqlite3.Connection, key: str) -> tuple[int, str, str | None]:
    row = con.execute(
        "SELECT id, name, short FROM clubs WHERE lower(short)=lower(?) OR lower(name)=lower(?)",
        (key, key)).fetchone()
    if not row:
        row = con.execute("SELECT id, name, short FROM clubs WHERE lower(name) LIKE lower(?)",
                          (f"%{key}%",)).fetchone()
    if not row:
        raise SystemExit(f"No club matches '{key}'. Add it with: add-club \"Name\" --short XXX")
    return row


def cmd_add_club(con, a):
    con.execute("INSERT INTO clubs (name, short, color, league) VALUES (?,?,?,?)",
                (a.name, a.short, a.color, a.league))
    con.commit()
    print(f"Added club {a.name}" + (f" ({a.short})" if a.short else ""))


def cmd_add_player(con, a):
    cid, cname, _ = find_club(con, a.club)
    con.execute("""INSERT INTO players (club_id, season, number, name, position) VALUES (?,?,?,?,?)
                   ON CONFLICT(club_id, season, number) DO UPDATE SET name=excluded.name, position=excluded.position""",
                (cid, a.season, a.number, a.name, a.position))
    con.commit()
    print(f"{cname} {a.season}: #{a.number} {a.name}")


def cmd_import(con, a):
    cid, cname, _ = find_club(con, a.club)
    n = skipped = 0
    with open(a.file, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip().isdigit():
                skipped += 1            # header line or blank
                continue
            num, name = int(row[0]), row[1].strip() if len(row) > 1 else ""
            if not name:
                skipped += 1
                continue
            pos = row[2].strip() if len(row) > 2 else None
            con.execute("""INSERT INTO players (club_id, season, number, name, position) VALUES (?,?,?,?,?)
                           ON CONFLICT(club_id, season, number) DO UPDATE SET name=excluded.name, position=excluded.position""",
                        (cid, a.season, num, name, pos))
            n += 1
    con.commit()
    print(f"{cname}: imported {n} players, skipped {skipped} lines")


def cmd_clubs(con, a):
    rows = con.execute("""SELECT c.short, c.name, c.league, c.color, count(p.id)
                          FROM clubs c LEFT JOIN players p ON p.club_id=c.id GROUP BY c.id ORDER BY c.name""").fetchall()
    if not rows:
        print("No clubs yet. Add one with add-club.")
    for short, name, league, color, n in rows:
        print(f"{short or '-':6} {name:30} {league or '':20} {color or '':8} {n} players")


def cmd_players(con, a):
    cid, cname, _ = find_club(con, a.club)
    rows = con.execute("SELECT number, name, position FROM players WHERE club_id=? AND season=? ORDER BY number",
                       (cid, a.season)).fetchall()
    print(f"{cname} — {a.season} ({len(rows)} players)")
    for num, name, pos in rows:
        print(f"  {num:>3}  {name:28} {pos or ''}")


def cmd_remove(con, a):
    cid, cname, _ = find_club(con, a.club)
    cur = con.execute("DELETE FROM players WHERE club_id=? AND season=? AND number=?", (cid, a.season, a.number))
    con.commit()
    print(f"Removed #{a.number} from {cname}" if cur.rowcount else f"No #{a.number} in {cname} {a.season}")


def cmd_lineup(con, a):
    cid, cname, short = find_club(con, a.club)
    rows = con.execute("SELECT number, name FROM players WHERE club_id=? AND season=? ORDER BY number",
                       (cid, a.season)).fetchall()
    if a.numbers:
        keep = {int(x) for x in a.numbers.split(",") if x.strip()}
        rows = [r for r in rows if r[0] in keep]
    if not rows:
        raise SystemExit("No players to export.")
    out = Path(a.out) if a.out else REPO / "data" / "lineups" / f"{short or cid}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(f"{n} {name}" for n, name in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} players to {out}\nIn the Tagger's match setup, click 'Load from file' under the lineup box.")


def cmd_add_match(con, a):
    hid = find_club(con, a.home)[0]
    aid = find_club(con, a.away)[0]
    con.execute("INSERT INTO matches (date, competition, home_id, away_id, video_path, events_csv, notes) VALUES (?,?,?,?,?,?,?)",
                (a.date, a.competition, hid, aid, a.video, a.csv, a.notes))
    con.commit()
    print("Match added.")


def cmd_matches(con, a):
    rows = con.execute("""SELECT m.id, m.date, h.name, w.name, m.competition, m.video_path, m.events_csv
                          FROM matches m JOIN clubs h ON h.id=m.home_id JOIN clubs w ON w.id=m.away_id
                          ORDER BY m.date""").fetchall()
    if not rows:
        print("No matches yet.")
    for mid, date, home, away, comp, video, ev in rows:
        print(f"[{mid}] {date or '?':10} {home} vs {away}  {comp or ''}")
        if video: print(f"      video:  {video}")
        if ev:    print(f"      events: {ev}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Local club & player database.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("add-club"); s.add_argument("name"); s.add_argument("--short"); s.add_argument("--color"); s.add_argument("--league"); s.set_defaults(fn=cmd_add_club)
    s = sub.add_parser("add-player"); s.add_argument("club"); s.add_argument("number", type=int); s.add_argument("name"); s.add_argument("--position"); s.add_argument("--season", default=SEASON); s.set_defaults(fn=cmd_add_player)
    s = sub.add_parser("import-players"); s.add_argument("club"); s.add_argument("file"); s.add_argument("--season", default=SEASON); s.set_defaults(fn=cmd_import)
    s = sub.add_parser("clubs"); s.set_defaults(fn=cmd_clubs)
    s = sub.add_parser("players"); s.add_argument("club"); s.add_argument("--season", default=SEASON); s.set_defaults(fn=cmd_players)
    s = sub.add_parser("remove-player"); s.add_argument("club"); s.add_argument("number", type=int); s.add_argument("--season", default=SEASON); s.set_defaults(fn=cmd_remove)
    s = sub.add_parser("lineup"); s.add_argument("club"); s.add_argument("--numbers"); s.add_argument("--out"); s.add_argument("--season", default=SEASON); s.set_defaults(fn=cmd_lineup)
    s = sub.add_parser("add-match"); s.add_argument("home"); s.add_argument("away"); s.add_argument("--date"); s.add_argument("--competition"); s.add_argument("--video"); s.add_argument("--csv"); s.add_argument("--notes"); s.set_defaults(fn=cmd_add_match)
    s = sub.add_parser("matches"); s.set_defaults(fn=cmd_matches)
    a = ap.parse_args()
    con = connect()
    try:
        a.fn(con, a)
    except sqlite3.IntegrityError as exc:
        print(f"Not saved: {exc}")
        return 1
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
