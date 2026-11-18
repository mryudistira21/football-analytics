-- Pitchside / football-analytics — club & player database on Supabase
-- Scope: clubs + players + lineups only. Match events and Track B moments stay
-- on their current paths (artifact db / CSV). Mirrors src/club_db.py's SQLite schema.
--
-- How to use:
--   1. Supabase dashboard -> SQL Editor -> paste this whole file -> Run.
--   2. Settings -> API: copy the Project URL and the `anon` public key into the
--      Tagger (match Settings -> Supabase box). Copy the `service_role` key only
--      into your local env for club_supabase.py (never into the Tagger / browser).
--
-- Security model: Row-Level Security ON. Anyone with the anon key can READ
-- (the Tagger needs this to list squads). Only the service_role key can WRITE
-- (the sync script / scraper). The anon key is safe to embed in your private
-- artifact because it cannot insert, update or delete.

create table if not exists clubs (
  id         bigint generated always as identity primary key,
  name       text not null unique,
  short      text unique,
  color      text,
  league     text,
  grp        text,                       -- Liga Nusantara group: A / B / C
  ileague_id text,                       -- source id on ileague.id, for the scraper
  created_at timestamptz not null default now()
);

create table if not exists players (
  id        bigint generated always as identity primary key,
  club_id   bigint not null references clubs(id) on delete cascade,
  season    text not null,
  number    int  not null,
  name      text not null,
  position  text,
  unique (club_id, season, number)
);

create index if not exists players_club_season_idx on players (club_id, season);

-- Optional: named match lineups (a chosen XI), if you want to save them online.
create table if not exists lineups (
  id         bigint generated always as identity primary key,
  club_id    bigint not null references clubs(id) on delete cascade,
  season     text not null,
  label      text,                       -- e.g. "vs Persika (H) 21-11"
  numbers    int[] not null default '{}',
  created_at timestamptz not null default now()
);

-- ---------- Row-Level Security: public read, service-only write ----------
alter table clubs   enable row level security;
alter table players enable row level security;
alter table lineups enable row level security;

-- Readable by anyone holding the anon key (the Tagger). PostgREST exposes the
-- anon role as `anon` and signed-in users as `authenticated`.
create policy "read clubs"   on clubs   for select to anon, authenticated using (true);
create policy "read players" on players for select to anon, authenticated using (true);
create policy "read lineups" on lineups for select to anon, authenticated using (true);

-- No write policies for anon/authenticated => only the service_role key (which
-- bypasses RLS) can insert/update/delete. That is the sync script and scraper.
