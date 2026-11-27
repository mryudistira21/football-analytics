#!/usr/bin/env python3
"""Players to watch (1-3 per club) -> data/players_to_watch.json.

Picks are a judgement on top of the official numbers in data/groupb_matches.json
(goals, starts, minutes, captaincy, suspensions), so every pick carries the stats
it rests on. Reasons are given in English (Hub) and Bahasa Indonesia (social posts).
Tracking-based notes (distance, touches, per shirt number) are added once a club's
matches have been through Track B with shirt-number OCR.

Group B now; Groups A and C get added in the same shape.

    python scripts/players_to_watch.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "groupb_matches.json"
OUT = ROOT / "data" / "players_to_watch.json"

# club -> [(shirt number, role label, why (EN), why (ID))]
PICKS = {
    "BTL": [
        (86, "Right winger", "Scored PERSIBA's first goal of the season (9', from outside the box at Persipa). Back from a two-yellow suspension.",
         "Pencetak gol pertama PERSIBA musim ini (menit 9, tendangan dari luar kotak di kandang Persipa). Kembali setelah absen akumulasi kartu."),
        (14, "Central midfielder", "19 years old and already a starter: headed the 51' goal at Persipa, 3 starts + 1 sub appearance.",
         "Baru 19 tahun tapi sudah jadi starter: sundulannya di menit 51 ke gawang Persipa, 3 kali starter + 1 kali masuk dari bangku cadangan."),
        (87, "Centre-back", "The constant in a back line that has changed every game: most minutes in the squad (about 315').",
         "Satu-satunya yang konstan di lini belakang yang selalu berubah: menit bermain terbanyak di skuad (sekitar 315 menit)."),
    ],
    "PBG": [
        (18, "Left winger", "Persibangga's main threat: 3 of their 4 goals (27', 32', 32'), all before half-time, every minute played.",
         "Ancaman utama Persibangga: 3 dari 4 gol tim (menit 27, 32, 32), semuanya di babak pertama, main penuh di semua laga."),
        (87, "Defensive midfielder (captain)", "Captain and midfield screen; back from a two-yellow suspension (booked by 41' and 45+2' in earlier games).",
         "Kapten sekaligus gelandang bertahan; kembali dari akumulasi kartu (kartu kuning di menit 41 dan 45+2 pada laga sebelumnya)."),
        (7, "Attacking midfielder", "Played every minute (360'); captained the side at Gresik. The link between midfield and Usama.",
         "Main penuh 360 menit; jadi kapten saat di Gresik. Penghubung lini tengah dengan Usama."),
    ],
    "PSF": [
        (90, "Centre-forward", "4 goals in 4 (2 penalties), joint top scorer in the league. Unbeaten Putra Sang Fajar lean on him.",
         "4 gol dari 4 laga (2 penalti), top skor bersama di liga. Putra Sang Fajar yang belum terkalahkan bertumpu padanya."),
        (10, "Winger / impact sub", "2 goals, mostly off the bench: the 86' winner at Persinga and the only goal against Persibo.",
         "2 gol, kebanyakan dari bangku cadangan: gol kemenangan menit 86 di Ngawi dan satu-satunya gol melawan Persibo."),
        (21, "Attacking midfielder (captain)", "Captain, every minute played for the group leaders.",
         "Kapten, main penuh di setiap laga untuk pemuncak grup."),
    ],
    "PKA": [
        (21, "Right-back", "3 goals from right-back, including the counter-attack winner in Bantul (29') and a brace against Persipa.",
         "3 gol dari posisi bek kanan, termasuk gol kemenangan lewat serangan balik di Bantul (menit 29) dan dua gol ke gawang Persipa."),
        (91, "Central midfielder", "Goal against Persipa; 3 starts in midfield.",
         "Mencetak gol ke gawang Persipa; 3 kali starter di lini tengah."),
        (7, "Forward / impact sub", "Late threat off the bench: the 90+3' equaliser at Gresik.",
         "Ancaman di menit akhir dari bangku cadangan: gol penyeimbang menit 90+3 di Gresik."),
    ],
    "GRS": [
        (28, "Centre-forward", "2 goals after coming on at half-time against Persibangga (78' penalty, 90'). Gresik have scored all 4 goals after the break.",
         "2 gol setelah masuk di babak kedua melawan Persibangga (penalti menit 78 dan menit 90). Keempat gol Gresik lahir di babak kedua."),
        (22, "Centre-forward", "88' equaliser at Persibo; another Gresik goal that came late.",
         "Gol penyeimbang menit 88 di Bojonegoro; lagi-lagi gol telat dari Gresik."),
        (21, "Centre-back (captain)", "Captain of a defence that has let in 3 in 4.",
         "Kapten lini belakang yang hanya kebobolan 3 gol dari 4 laga."),
    ],
    "PNG": [
        (66, "Central midfielder", "2 goals (56', 67') from midfield; started every game.",
         "2 gol (menit 56 dan 67) dari lini tengah; selalu jadi starter."),
        (27, "Central midfielder", "Scored the winner in Bantul (29').",
         "Pencetak gol kemenangan di Bantul (menit 29)."),
        (19, "Centre-back (captain)", "Captain, about 338' played.",
         "Kapten, sekitar 338 menit bermain."),
    ],
    "PBO": [
        (7, "Centre-forward", "Scored at Persibangga; the starting striker.",
         "Mencetak gol di kandang Persibangga; striker utama."),
        (10, "Forward / impact sub", "79' goal against Gresik eight minutes after coming on.",
         "Gol menit 79 ke gawang Gresik, delapan menit setelah masuk."),
        (3, "Left-back (captain)", "Captain, every minute played (360').",
         "Kapten, main penuh 360 menit."),
    ],
    "PIP": [
        (16, "Left winger", "Both goals in the 2-2 with PERSIBA (15', 58').",
         "Dua gol ke gawang PERSIBA di laga 2-2 (menit 15 dan 58)."),
        (8, "Defensive midfielder", "Midfield anchor, about 337' played.",
         "Jangkar lini tengah, sekitar 337 menit bermain."),
    ],
}


def main() -> int:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    players = {(p["club"], p["no"]): p for p in d["players"]}
    out = {"schema": "players_to_watch_v1", "updated": d["updated"], "through_round": d["through_round"],
           "basis": "Official ileague data, rounds 1-4: goals, starts, approximate minutes, captaincy, suspensions.",
           "clubs": {}}
    for club, picks in PICKS.items():
        rows = []
        for no, role, why_en, why_id in picks:
            p = players.get((club, no))
            if p is None:
                raise SystemExit(f"{club} #{no} not found in {SRC.name}")
            rows.append({"no": no, "name": p["name"], "role": role, "why": why_en, "why_id": why_id,
                         "stats": {k: p[k] for k in ("apps", "starts", "sub_apps", "minutes", "goals", "pens", "captain")}})
        out["clubs"][club] = {"name": d["teams"][club], "group": "B", "players": rows}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {sum(len(c['players']) for c in out['clubs'].values())} players across {len(out['clubs'])} clubs")
    for club, c in out["clubs"].items():
        print(f"  {club}: " + "; ".join(f"#{p['no']} {p['name'].title()} ({p['stats']['goals']}g, {p['stats']['minutes']}')" for p in c["players"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
