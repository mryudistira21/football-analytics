#!/usr/bin/env python3
"""Save everything about a match into one SQLite database file (no server, no install).

SQLite ships inside Python, so there is nothing to set up: the database is a single
file (default <root>/football.db). Open it with DB Browser for SQLite, Python/pandas,
Excel/Power BI (ODBC) — or push it to Postgres/Supabase later with the same tables.

What it stores per match (re-running replaces that match's rows; other matches stay):
  matches        config: teams, date, score, video, period times, direction
  tracking       every Track B position row (optional; ~1.3 M rows per match)
  moments        Track B per-track ball-near moments (from the moments manifest)
  candidates     Track B event candidates + your review decision (accepted/rejected/pending)
  track_map      Track B track id -> shirt number, from your review in the Tagger
  events         the final event list from the Tagger (manual + accepted candidates)
  team_metrics   the Track B team report, one row per period/side/metric
  team_heatmaps  21 x 14 position grids per period and side
  ingests        what was loaded, from which file, when

Sources it looks for, per match id:
  <root>/runs/<id>/                       (run_match.py output: tracking, moments, report, candidates)
  <id>_trackb_bundle.zip                  (from Kaggle/Colab) in <root>/runs/<id>/ or an --inbox folder
  <id>_events.csv, <id>_review.json       (Tagger downloads) in <root>/runs/<id>/tagger/ or an --inbox folder

    python scripts/save_to_db.py --config matches/persiba_2026-27.json --root D:/Video-Bola-Analis --all
    python scripts/save_to_db.py --config ... --root ... --match persiba_persika_20261121 --inbox %USERPROFILE%/Downloads
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
  match_id TEXT PRIMARY KEY, season TEXT, scope TEXT, round TEXT, date TEXT,
  home TEXT, away TEXT, home_score INTEGER, away_score INTEGER, video_url TEXT,
  periods_json TEXT, home_dir_1h TEXT, swap_teams INTEGER, updated_at TEXT);
CREATE TABLE IF NOT EXISTS tracking (
  match_id TEXT, frame INTEGER, video_seconds REAL, period INTEGER, track_id INTEGER,
  role TEXT, team TEXT, x REAL, y REAL, confidence REAL, camera_state TEXT);
CREATE INDEX IF NOT EXISTS ix_tracking_match ON tracking(match_id, period, frame);
CREATE INDEX IF NOT EXISTS ix_tracking_track ON tracking(match_id, track_id);
CREATE TABLE IF NOT EXISTS moments (
  match_id TEXT, moment TEXT, track_id INTEGER, team TEXT, start_s REAL, end_s REAL,
  duration_s REAL, label TEXT, calibration_error_m REAL, wide_share REAL,
  unknown_team_share REAL, ball_observation_share REAL, review_reasons TEXT,
  PRIMARY KEY (match_id, moment));
CREATE TABLE IF NOT EXISTS candidates (
  match_id TEXT, cand_id TEXT, period INTEGER, t REAL, t_end REAL, side TEXT, type TEXT,
  outcome TEXT, track_id TEXT, track_to TEXT, x REAL, y REAL, ex REAL, ey REAL,
  confidence TEXT, reason TEXT, review_status TEXT, event_id TEXT, priority INTEGER, linked TEXT,
  PRIMARY KEY (match_id, cand_id));
CREATE TABLE IF NOT EXISTS track_map (
  match_id TEXT, track_id TEXT, side TEXT, player TEXT, source TEXT, confidence REAL, votes INTEGER,
  PRIMARY KEY (match_id, track_id, source));
CREATE TABLE IF NOT EXISTS events (
  match_id TEXT, event_id TEXT, half INTEGER, match_clock TEXT, match_seconds REAL,
  video_seconds REAL, team TEXT, side TEXT, player TEXT, player_name TEXT, event TEXT,
  outcome TEXT, receiver TEXT, receiver_name TEXT, x REAL, y REAL, end_x REAL, end_y REAL,
  body_part TEXT, situation TEXT, assist_event_id TEXT, xg REAL, xt_added REAL,
  progressive INTEGER, into_box INTEGER, review TEXT, source TEXT, candidate_id TEXT,
  track_id TEXT, track_to TEXT, PRIMARY KEY (match_id, event_id));
CREATE TABLE IF NOT EXISTS team_metrics (
  match_id TEXT, period TEXT, side TEXT, metric TEXT, value REAL,
  PRIMARY KEY (match_id, period, side, metric));
CREATE TABLE IF NOT EXISTS team_heatmaps (
  match_id TEXT, period TEXT, side TEXT, grid_json TEXT, PRIMARY KEY (match_id, period, side));
CREATE TABLE IF NOT EXISTS team_physical (
  match_id TEXT, side TEXT, distance_km REAL, hi_distance_km REAL, sprint_distance_km REAL,
  top_speed_kmh REAL, touches INTEGER, tracked_player_seconds INTEGER, PRIMARY KEY (match_id, side));
CREATE TABLE IF NOT EXISTS player_physical (
  match_id TEXT, team TEXT, number TEXT, minutes REAL, distance_m REAL, hi_distance_m REAL,
  sprint_distance_m REAL, top_speed_kmh REAL, touches INTEGER, dist_per_min REAL, tracks INTEGER,
  PRIMARY KEY (match_id, team, number));
CREATE TABLE IF NOT EXISTS counter_press (
  match_id TEXT, period INTEGER, t REAL, side TEXT, pressers INTEGER, regained INTEGER, regain_s REAL);
CREATE TABLE IF NOT EXISTS ingests (
  match_id TEXT, part TEXT, source_file TEXT, rows INTEGER, ingested_at TEXT);
CREATE VIEW IF NOT EXISTS v_review_progress AS
  SELECT match_id, review_status, type, COUNT(*) AS n FROM candidates GROUP BY 1, 2, 3;
CREATE VIEW IF NOT EXISTS v_player_events AS
  SELECT e.match_id, m.date, e.side, e.team, e.player, e.player_name, e.event, e.outcome,
         COUNT(*) AS n, ROUND(SUM(COALESCE(e.xg, 0)), 3) AS xg, ROUND(SUM(COALESCE(e.xt_added, 0)), 4) AS xt
  FROM events e LEFT JOIN matches m USING (match_id) GROUP BY 1, 2, 3, 4, 5, 6, 7, 8;
CREATE VIEW IF NOT EXISTS v_track_numbers AS
  SELECT match_id, track_id, side, player, confidence, votes FROM track_map WHERE source='ocr' ORDER BY confidence DESC;
CREATE VIEW IF NOT EXISTS v_team_half AS
  SELECT t.match_id, m.date, m.home, m.away, t.period, t.side, t.metric, t.value
  FROM team_metrics t LEFT JOIN matches m USING (match_id);
CREATE VIEW IF NOT EXISTS v_team_physical AS
  SELECT p.match_id, m.date, CASE WHEN p.side='home' THEN m.home ELSE m.away END AS team,
         p.distance_km, p.hi_distance_km, p.sprint_distance_km, p.top_speed_kmh, p.touches
  FROM team_physical p LEFT JOIN matches m USING (match_id);
CREATE VIEW IF NOT EXISTS v_counter_press AS
  SELECT match_id, side, COUNT(*) AS losses, ROUND(AVG(regained), 3) AS regain_share,
         ROUND(AVG(pressers), 2) AS avg_pressers FROM counter_press GROUP BY 1, 2;
"""

EVENT_COLS = ["half", "match_clock", "match_seconds", "video_seconds", "team", "side", "player",
              "player_name", "event", "outcome", "receiver", "receiver_name", "x", "y", "end_x", "end_y",
              "body_part", "situation", "assist_event_id", "xg", "xt_added", "progressive", "into_box",
              "review", "source", "candidate_id", "track_id", "track_to"]
TRACK_COLS = ["frame", "video_seconds", "period", "track_id", "role", "team", "x", "y", "confidence", "camera_state"]
CAND_COLS = ["cand_id", "period", "t", "t_end", "side", "type", "outcome", "track_id", "track_to",
             "x", "y", "ex", "ey", "confidence", "reason"]


def num(v: Any) -> Any:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f.is_integer() and "." not in str(v) else f


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def batched(it: Iterable[tuple], n: int = 50_000) -> Iterator[list[tuple]]:
    buf: list[tuple] = []
    for row in it:
        buf.append(row)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


class Sources:
    """Find the files for one match in the run folder, an extracted bundle, or inbox folders."""

    def __init__(self, root: Path, mid: str, inboxes: list[Path]):
        self.mid, self.run = mid, root / "runs" / mid
        self.dirs = [self.run, self.run / "tagger", *inboxes]
        self.bundle = self._find(f"{mid}_trackb_bundle.zip")

    def _find(self, name: str) -> Path | None:
        hits = [d / name for d in self.dirs if (d / name).is_file()]
        return max(hits, key=lambda p: p.stat().st_mtime) if hits else None

    def file(self, rel: str) -> Path | None:
        p = self.run / rel
        return p if p.is_file() else None

    def bundle_member(self, name: str) -> tuple[zipfile.ZipFile, str] | None:
        if not self.bundle:
            return None
        z = zipfile.ZipFile(self.bundle)
        for n in z.namelist():
            if n == name or n.endswith("/" + name):
                return z, n
        z.close()
        return None

    def open_text(self, rel: str, bundle_name: str | None = None) -> tuple[io.TextIOBase, str] | None:
        p = self.file(rel)
        if p:
            return p.open(newline="", encoding="utf-8"), str(p)
        hit = self.bundle_member(bundle_name or Path(rel).name)
        if hit:
            z, n = hit
            return io.TextIOWrapper(z.open(n), encoding="utf-8", newline=""), f"{self.bundle}!{n}"
        return None


def log_ingest(con: sqlite3.Connection, mid: str, part: str, src: str, rows: int) -> None:
    con.execute("INSERT INTO ingests VALUES (?,?,?,?,?)", (mid, part, src, rows, now()))
    print(f"  {part:13s} {rows:>9,} rows  <- {src}")


def save_match(con: sqlite3.Connection, cfg: dict[str, Any], m: dict[str, Any]) -> None:
    sc = m.get("score") or [None, None]
    con.execute("INSERT OR REPLACE INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        m["match_id"], cfg.get("season"), m.get("scope"), m.get("round"), m.get("date"), m.get("home"),
        m.get("away"), sc[0], sc[1], m.get("video_url"), json.dumps(m.get("periods")),
        m.get("home_dir_1h"), int(bool(m.get("swap_teams"))), now()))


def save_tracking(con: sqlite3.Connection, src: Sources) -> None:
    opened = src.open_text(f"{src.mid}_tracking.csv")
    if not opened:
        return
    fh, name = opened
    con.execute("DELETE FROM tracking WHERE match_id=?", (src.mid,))
    n = 0
    with fh:
        rows = ((src.mid, *[num(r[c]) for c in TRACK_COLS]) for r in csv.DictReader(fh))
        for chunk in batched(rows):
            con.executemany(f"INSERT INTO tracking VALUES ({','.join('?' * 11)})", chunk)
            n += len(chunk)
    log_ingest(con, src.mid, "tracking", name, n)


def save_moments(con: sqlite3.Connection, src: Sources) -> None:
    man = next(iter(sorted((src.run / "moments").glob("*_moments.manifest.json"))), None) \
        if (src.run / "moments").exists() else None
    if man:
        data, name = json.loads(man.read_text(encoding="utf-8")), str(man)
    else:
        hit = None
        if src.bundle:
            z = zipfile.ZipFile(src.bundle)
            names = [n for n in z.namelist() if n.endswith("_moments.manifest.json")]
            if names:
                data, name = json.loads(z.read(names[0]).decode("utf-8")), f"{src.bundle}!{names[0]}"
                hit = True
            z.close()
        if not hit:
            return
    con.execute("DELETE FROM moments WHERE match_id=?", (src.mid,))
    rows = []
    for tr in data.get("tracks", []):
        for mo in tr.get("moments", []):
            rows.append((src.mid, mo.get("moment"), mo.get("track_id"), mo.get("team"), mo.get("start_s"),
                         mo.get("end_s"), mo.get("duration_s"), mo.get("label"), mo.get("calibration_error_m"),
                         mo.get("wide_share"), mo.get("unknown_team_share"), mo.get("ball_observation_share"),
                         ",".join(mo.get("review_reasons") or [])))
    con.executemany(f"INSERT OR REPLACE INTO moments VALUES ({','.join('?' * 13)})", rows)
    log_ingest(con, src.mid, "moments", name, len(rows))


def save_report(con: sqlite3.Connection, src: Sources) -> None:
    opened = src.open_text(f"report/{src.mid}_team_report.json", f"{src.mid}_team_report.json")
    if not opened:
        return
    fh, name = opened
    with fh:
        rep = json.load(fh)
    con.execute("DELETE FROM team_metrics WHERE match_id=?", (src.mid,))
    con.execute("DELETE FROM team_heatmaps WHERE match_id=?", (src.mid,))
    mrows, hrows = [], []
    for period, block in rep.get("periods", {}).items():
        for k, v in (block.get("quality") or {}).items():
            mrows.append((src.mid, period, "match", f"quality_{k}", v))
        for k, v in (block.get("territory_home_view") or {}).items():
            mrows.append((src.mid, period, "match", f"territory_{k}", v))
        for side in ("home", "away"):
            for k, v in (block.get(side) or {}).items():
                if k == "heatmap":
                    hrows.append((src.mid, period, side, json.dumps(v)))
                elif isinstance(v, (int, float)) or v is None:
                    mrows.append((src.mid, period, side, k, v))
    con.executemany("INSERT OR REPLACE INTO team_metrics VALUES (?,?,?,?,?)", mrows)
    con.executemany("INSERT OR REPLACE INTO team_heatmaps VALUES (?,?,?,?)", hrows)
    log_ingest(con, src.mid, "team_report", name, len(mrows))

    # physical running totals, per-player rows and counter-press turnovers (all embedded in the report)
    tt = (rep.get("physical") or {}).get("team_totals") or {}
    con.execute("DELETE FROM team_physical WHERE match_id=?", (src.mid,))
    con.executemany("INSERT OR REPLACE INTO team_physical VALUES (?,?,?,?,?,?,?,?)",
                    [(src.mid, side, d.get("distance_km"), d.get("hi_distance_km"), d.get("sprint_distance_km"),
                      d.get("top_speed_kmh"), d.get("touches"), d.get("tracked_player_seconds"))
                     for side, d in tt.items()])
    pcols = ["team", "number", "minutes", "distance_m", "hi_distance_m", "sprint_distance_m",
             "top_speed_kmh", "touches", "dist_per_min", "tracks"]
    con.execute("DELETE FROM player_physical WHERE match_id=?", (src.mid,))
    con.executemany(f"INSERT OR REPLACE INTO player_physical VALUES ({','.join('?' * (len(pcols) + 1))})",
                    [(src.mid, *[num(r.get(c)) if c not in ("team", "number") else r.get(c) for c in pcols])
                     for r in (rep.get("physical") or {}).get("by_player") or []])
    con.execute("DELETE FROM counter_press WHERE match_id=?", (src.mid,))
    con.executemany("INSERT INTO counter_press VALUES (?,?,?,?,?,?,?)",
                    [(src.mid, e.get("period"), e.get("t"), e.get("side"), e.get("pressers"),
                      int(bool(e.get("regained"))), e.get("regain_s"))
                     for e in rep.get("counter_press_events") or []])


def migrate(con: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created (no data loss)."""
    have = {r[1] for r in con.execute("PRAGMA table_info(candidates)")}
    for col, typ in (("priority", "INTEGER"), ("linked", "TEXT")):
        if col not in have:
            con.execute(f"ALTER TABLE candidates ADD COLUMN {col} {typ}")


def save_candidates(con: sqlite3.Connection, src: Sources, review: dict[str, Any] | None) -> None:
    opened = src.open_text(f"{src.mid}_candidates.csv")
    if not opened:
        return
    fh, name = opened
    status = {c["cand_id"]: (c.get("status"), c.get("event_id")) for c in (review or {}).get("candidates", [])}
    con.execute("DELETE FROM candidates WHERE match_id=?", (src.mid,))
    with fh:
        rows = []
        for r in csv.DictReader(fh):
            st, ev = status.get(r["cand_id"], ("pending", None))
            rows.append((src.mid, *[num(r.get(c)) if c not in ("cand_id", "track_id", "track_to") else r.get(c)
                                    for c in CAND_COLS], st, ev, num(r.get("priority")), r.get("linked") or None))
    cols = ["match_id", *CAND_COLS, "review_status", "event_id", "priority", "linked"]
    con.executemany(f"INSERT OR REPLACE INTO candidates ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", rows)
    log_ingest(con, src.mid, "candidates", name, len(rows))


def load_review(src: Sources) -> tuple[dict[str, Any] | None, str | None]:
    p = src._find(f"{src.mid}_review.json")
    if not p:
        return None, None
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("match_id") != src.mid:
        raise SystemExit(f"{p} belongs to {data.get('match_id')}, not {src.mid}")
    return data, str(p)


def save_track_map(con: sqlite3.Connection, src: Sources, review: dict[str, Any] | None, name: str | None) -> None:
    if not review:
        return
    con.execute("DELETE FROM track_map WHERE match_id=? AND source='review'", (src.mid,))
    rows = [(src.mid, str(tid), "home" if v.get("team") == 0 else "away", str(v.get("p", "")), "review", None, None)
            for tid, v in (review.get("track_map") or {}).items()]
    con.executemany("INSERT OR REPLACE INTO track_map VALUES (?,?,?,?,?,?,?)", rows)
    log_ingest(con, src.mid, "track_map", name or "", len(rows))


def save_track_numbers(con: sqlite3.Connection, src: Sources) -> None:
    """The automatic shirt-number OCR map (source='ocr')."""
    opened = src.open_text(f"{src.mid}_track_map.csv")
    if not opened:
        return
    fh, name = opened
    con.execute("DELETE FROM track_map WHERE match_id=? AND source='ocr'", (src.mid,))
    rows = []
    with fh:
        for r in csv.DictReader(fh):
            rows.append((src.mid, str(r["track_id"]), r.get("team"), str(r.get("number", "")), "ocr",
                         num(r.get("confidence")), num(r.get("votes"))))
    con.executemany("INSERT OR REPLACE INTO track_map VALUES (?,?,?,?,?,?,?)", rows)
    log_ingest(con, src.mid, "track_numbers", name, len(rows))


def save_events(con: sqlite3.Connection, src: Sources) -> None:
    p = src._find(f"{src.mid}_events.csv")
    if not p:
        return
    con.execute("DELETE FROM events WHERE match_id=?", (src.mid,))
    rows = []
    with p.open(newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("match_id") and r["match_id"] != src.mid:
                raise SystemExit(f"{p} contains match_id {r['match_id']}, expected {src.mid}")
            rows.append((src.mid, r.get("id"), *[num(r.get(c)) if c not in ("player", "receiver", "track_id",
                         "track_to", "candidate_id", "assist_event_id") else (r.get(c) or None) for c in EVENT_COLS]))
    con.executemany(f"INSERT OR REPLACE INTO events VALUES ({','.join('?' * 30)})", rows)
    log_ingest(con, src.mid, "events", str(p), len(rows))
    # keep a copy next to the run so the run folder is complete
    dst = src.run / "tagger" / p.name
    if p.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description="Save match data (Track B + Tagger review) into SQLite.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--db", type=Path, help="default: <root>/football.db")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--match", help="one match_id (comma-separated for several)")
    g.add_argument("--all", action="store_true", help="every match in the config that has any data")
    ap.add_argument("--inbox", type=Path, action="append", default=[],
                    help="extra folder to look for Tagger downloads and bundles (e.g. Downloads); repeatable")
    ap.add_argument("--no-tracking", action="store_true", help="skip the big tracking table")
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    by_id = {m["match_id"]: m for m in cfg["matches"]}
    ids = list(by_id) if a.all else [x.strip() for x in a.match.split(",") if x.strip()]
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"Not in the config: {missing}")
    db = a.db or a.root / "football.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    migrate(con)
    con.execute("PRAGMA journal_mode=WAL")
    saved = 0
    for mid in ids:
        src = Sources(a.root, mid, [p.expanduser() for p in a.inbox])
        has_any = src.run.exists() or src.bundle or src._find(f"{mid}_events.csv")
        if not has_any:
            if not a.all:
                print(f"{mid}: no data found yet (run folder, bundle or Tagger files).")
            continue
        print(f"{mid}:")
        with con:  # one transaction per match: all or nothing
            save_match(con, cfg, by_id[mid])
            review, review_name = load_review(src)
            if not a.no_tracking:
                save_tracking(con, src)
            save_moments(con, src)
            save_report(con, src)
            save_candidates(con, src, review)
            save_track_map(con, src, review, review_name)
            save_track_numbers(con, src)
            save_events(con, src)
        saved += 1
    con.close()
    print(f"\nSaved {saved} match(es) -> {db}")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
