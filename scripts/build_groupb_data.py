#!/usr/bin/env python3
"""Group B (Liga Nusantara 2026-27) official match data -> data/groupb_matches.json.

Source: the ileague.id match pages (result/detail/LIGA_NUSANTARA_2026-27/<date>/<HOME>/<AWAY>),
read 27 Nov 2026: starting XIs with positions and captain, substitutes used, the minute
markers next to each player, suspensions, coaches, venue, attendance.

The ileague page prints goals AND cards in the same list under the score, so goal
scorers were taken from that list only where its length equals the score, and
otherwise from match reports (source noted per goal):
  - Persipa 2-2 PERSIBA: Tribun Jogja (Elvateeh 9', Arsil 51' header; Tuka 15', 58')
  - PERSIBA 0-1 Persika: Jatengnews (Fabian 28-29', counter-attack)
  - Persika 1-0 Persinga: Radar Madiun (M. Syaiful Asri 88')
Every team's goal total below matches the official standings.

Minutes are approximate: a starter's last marker that equals a substitute's marker is
read as the substitution minute; markers are not labelled on the page.

    python scripts/build_groupb_data.py      # writes data/groupb_matches.json
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "groupb_matches.json"

TEAMS = {"BTL": "PERSIBA Bantul", "GRS": "Gresik United FC", "PBG": "Persibangga Purbalingga",
         "PBO": "Persibo Bojonegoro", "PKA": "Persika Karanganyar", "PNG": "Persinga Ngawi",
         "PIP": "Persipa Pati", "PSF": "Putra Sang Fajar FC"}

# Each player: "no|NAME|POS|markers" (C before markers = captain). Subs: "no|NAME|markers".
MATCHES = [
 dict(id="persiba_gresik_20261113", round=1, date="2026-11-13", home="BTL", away="GRS", hs=0, as_=0,
      venue="Stadion Sultan Agung, Bantul", att=2375, coach=("SUSANTO", "JOKO SUSILO"), goals=[],
      xi_h="""20|MOHAMMAD FEBRYANSYAH REFYAN SANUSI|GK|90;3|WIGANDA PRADIKA|LB|81;18|MUHAMMAD AULIA WIRAWAN|CB|25 33;87|AHMAD RIFAL|CB|90;66|CHOLIS BIMA SATRIA NATANEGARA|RB|71 81;8|AIDUN SASTRA UTAMI|CM|33;10|REYKI FARIZ RAMADHAN|AM|90;14|MUH ARSIL ANAS|CM|63;17|RISKY DWIYAN APRILYANTO|DM|90;7|ANTONI PUTRO NUGROHO|LWF|C 90;86|ELVATEEH VARESIA AKHADA|RWF|90""",
      sub_h="11|ALDITO DAFFA QUARISMA|81;24|MUH TIRTO WICAKSONO|33;29|RIAN RAMADAN|81;28|DHARMA JATI BINTARA PUTRA|63;16|FACHRIZAL AHNAF MAULANA|33",
      xi_a="""23|LALU MUHAMMAD RIZKI|GK|90;21|ABDUL RAHMAN|CB|C 90;33|JOKO SUPRIYANTO|CB|45;16|ADE M NUR SAIHITUA|RB|90;7|ALIF JAELANI|CM|90;13|M. SYAIFULLAH SALIM|AM|45;15|MUHAMAD FADILLA AKBAR|DM|90;24|MOHAMMAD JORDAN ZAMORANO|DM|49 90;9|DECO NOVERO PALANDI|CF|45;22|M. DZAKI ALAMSYAH|CF|58;99|OSCAR NOVANGGA|LWF|74""",
      sub_a="29|MUHAMMAD BAIHAQI RIFAI|45;8|M ALWI FURQON|45;17|MAULANA ILHAM JAYA KUSUMA|74;28|SHAHIH ELANG RISHANDY|45;18|ALDI TRI NUGROHO|58"),
 dict(id="persibo_persipa_20261113", round=1, date="2026-11-13", home="PBO", away="PIP", hs=0, as_=0,
      venue="Stadion Letjen H. Soedirman, Bojonegoro", att=1430, coach=("I PUTU GEDE SWI SANTOSO", "SALAHUDDIN"), goals=[],
      xi_h="""14|ALDO FERNANDI|GK|90;3|CRAH EKA ANGGER ISWANTO|LB|C 90;6|TEGAR ANDRIE SHEVANTON|CB|90;26|RIO PRATAMA VALENTINO|CB|90;13|MOCHAMAD ADITYA RANGGA SAPUTRA|RB|90;18|MOHAMAD SYAHRIL AFRIZAL|DM|87 90;27|ROMEO NIZAM ABISHA|AM|86;58|ADI PURNAMA S. SUAIB|DM|35 90 90;7|SEHABUDIN AHMAD|CF|77;11|RIFALDI BAWUO|LWF|61;77|NURDIAN SYAPUTRA CHANIAGO|RWF|61""",
      sub_h="29|BASAJUMP LATUCONSINA|61;25|AARON YEKTI|90;37|WIRAJA ARIA FADHIL AULIA|77;38|AZRIL DECO NORMADHANY|86;33|OCVIAN CHANIGIO|61",
      xi_a="""88|ARIFIN SETYADI|GK|90;17|MUHAMMAD IQBAL AL GHUZAT|LB|C 90;5|IMAM MAHMUDI|CB|90;44|MUHAMMAD RHAGA AL MISRI YULIANTO|CB|72 90;71|MUH. AL - QADRI|RB|90;8|AKHMAD RIYADI|DM|67;36|EDGARDA FAHAR BUNTORO|AM|11 90;47|HERSYA SCIFO|CM|61;10|FRITY ARMAN|CF|61;16|DANHYL POETRA MARSELINHO TUKA|LWF|67;99|MOCHAMMAD RISAL AMIN|RWF|87""",
      sub_a="9|QISCHIL GANDRUM MINNY|87;14|YOSUA RAFAEL ATALYA YOLTUWU|61;67|RIDUWAN JAMIL|67;86|YOHANES YAPAGAIMU|61 75;7|PANDI AHMAD LESTALUHU|67",
      notes=["Persibo #58 Adi Purnama S. Suaib sent off 90' (later banned for misconduct)"]),
 dict(id="persika_persinga_20261113", round=1, date="2026-11-13", home="PKA", away="PNG", hs=1, as_=0,
      venue="Stadion Manahan, Solo", att=2183, coach=("TAUFIQ", "DINDIN WAHYUDIN"),
      goals=[("h", 18, "M. SYAIFUL ASRI", "88", "", "Radar Madiun")],
      xi_h="""31|NURI AGUS WIBOWO|GK|90;38|MUSRIL|LB|41 90;17|MUHAMMAD RIO SAPUTRO|CB|C 90;19|IBNUL MUBARAK|CB|69;77|UMAR|RB|45;4|ABI DEFA MARENDRA|CM|66;6|MOCHAMAD RAGIL YUSUF SAPUTRA|DM|45;8|YOGIEK FERNANDA|AM|69;16|LIONEL ZAHRA ALFARIZKI|LWF|90;18|M. SYAIFUL ASRI|CF|88 90;23|WAWAN FEBRIANTO|RWF|90""",
      sub_h="25|YULFIKAR MANSUR BAYUAJI|69;91|FEBRIAN TRI YANTO|66;11|REYHAN FIRDAUS|45;28|MOHAMAD IQBAL|45;12|RONI SUGENG ARIYANTO|69",
      xi_a="""96|FAHMI IDRIS|GK|90;22|DADAN AKBAR NUGRAHA|LB|46 90;3|SYARFUL MUDAWAM|CB|58;15|BAGAS AGUNG UTOMO|CB|90;19|NANDA HELVRIT SAPUTRA|CB|68;25|JOHANES EDWARD SUBROTO|RB|45;11|DIMAS GALIH GUMILANG|AM|C 68;21|AKBAR SYAKIRA|CM|90;66|MUCH ILHAM NOER|CM|58 90;18|M.ALVIN DWI GUNA|CF|90;77|MUHAMAD FIRMAN JANUARI|LWF|58""",
      sub_a="13|IKBAL KHADAFI|58;14|BATHSYA DEWA APRILLIANO|45;4|RONALD AFFRIZAL ARDIANSYAH PUTRA SABATH|68;27|ADITYA SENTOSA|68;57|QOYUM NAZALUDIN MUHAMAD|58"),
 dict(id="psf_persibangga_20261114", round=1, date="2026-11-14", home="PSF", away="PBG", hs=2, as_=0,
      venue="Stadion Gelora Soepriadi, Blitar", att=711, coach=("AGUS SUGENG RIYANTO", "MOHAMAD IRFAN"),
      goals=[("h", 90, "JOHAN YOGA UTAMA", "47", "pen", "ileague"), ("h", 90, "JOHAN YOGA UTAMA", "74", "", "ileague")],
      xi_h="""86|WAHYU TRI NUGROHO|GK|90;30|OKY KHARISMA|LB|45 64;87|FATHUL IHSAN|CB|86;29|FAHRI AKBAR ROMADHON|RB|90;21|FIRMANSYAH|AM|90;22|FAJAR IHSANUDIN|DM|14 45;99|KEVIN AL HAFIZ|DM|56 90;11|MOCH. ALEX SANDRO|RWF|64;9|AHMAD ATHALLAH ARAIHAN|CF|90;90|JOHAN YOGA UTAMA|CF|47 74 90;7|RIDWAN ANSORI|RWF|77""",
      sub_h="12|FIGO FIRMANSYAH|77;27|ZIDANE ALZIZOU LUBIS|45 55;14|DIFO PRASETYA|64;10|KOMARODIN|64 83;75|RIO FERNANDO DWI ARDIANSYAH|86",
      xi_a="""76|HUSEIN AKBAR AL FAUZI|GK|90;16|AZHRIL SYAHDAN FABREGAS|LB|90;4|FATHUR YAN|CB|90;6|YUFRIZAL LU'AY ABIYU|CB|1 90;25|AHMAD RAMDIN TUASALAMONY|RB|90;7|ERLANGGA ADHYAKSA NUGRAHA|AM|58 90;30|ALBERT THEO WICAKSONO|CM|90;87|MUHAMMMAD ILHAMUL IRHAZ|DM|C 41 90;12|ARDYA MALIK PUTRA|RWF|52;18|USAMA AHMAD RAIHAN|LWF|90;79|ADITHYA JORY GURUH SETIAWAN|CF|77""",
      sub_a="17|AFIF TEGUH SAPUTRA|52;32|FARREL ARYA TRI SANDIKA SUPRATIKNO|77"),

 dict(id="persipa_persiba_20261117", round=2, date="2026-11-17", home="PIP", away="BTL", hs=2, as_=2, ht=(1, 1),
      venue="Stadion Moch. Soebroto, Magelang", att=361, coach=("SALAHUDDIN", "SUSANTO"),
      goals=[("a", 86, "ELVATEEH VARESIA AKHADA", "9", "shot from outside the box", "Tribun Jogja"),
             ("h", 16, "DANHYL POETRA MARSELINHO TUKA", "15", "", "Tribun Jogja"),
             ("a", 14, "MUH ARSIL ANAS", "51", "header", "Tribun Jogja"),
             ("h", 16, "DANHYL POETRA MARSELINHO TUKA", "58", "", "Tribun Jogja")],
      xi_h="""88|ARIFIN SETYADI|GK|90;5|IMAM MAHMUDI|CB|90;71|MUH. AL - QADRI|RB|90;17|MUHAMMAD IQBAL AL GHUZAT|LB|C 90;44|MUHAMMAD RHAGA AL MISRI YULIANTO|CB|90;8|AKHMAD RIYADI|DM|90;36|EDGARDA FAHAR BUNTORO|AM|75;47|HERSYA SCIFO|CM|82;16|DANHYL POETRA MARSELINHO TUKA|LWF|15 58 90;99|MOCHAMMAD RISAL AMIN|RWF|82;86|YOHANES YAPAGAIMU|CF|86""",
      sub_h="9|QISCHIL GANDRUM MINNY|86;30|GADA RAMADHAN|82;14|YOSUA RAFAEL ATALYA YOLTUWU|75;7|PANDI AHMAD LESTALUHU|82",
      xi_a="""20|MOHAMMAD FEBRYANSYAH REFYAN SANUSI|GK|90;87|AHMAD RIFAL|CB|90;66|CHOLIS BIMA SATRIA NATANEGARA|RB|72;16|FACHRIZAL AHNAF MAULANA|CB|90;33|JOSEPH ALEXANDER KURNIAWAN|CB|68;14|MUH ARSIL ANAS|CM|51 86;24|MUH TIRTO WICAKSONO|DM|90;17|RISKY DWIYAN APRILYANTO|DM|90;7|ANTONI PUTRO NUGROHO|LWF|C 45;86|ELVATEEH VARESIA AKHADA|RWF|9 32 90;77|SANDI ARTA SAMOSIR|RWF|90""",
      sub_a="12|LORENZO GOMES DE OLIVIERA|72;11|ALDITO DAFFA QUARISMA|86;10|REYKI FARIZ RAMADHAN|45 63;3|WIGANDA PRADIKA|68",
      cards=[("a", 86, "yellow", "32"), ("a", 10, "yellow", "63")]),
 dict(id="gresik_persika_20261117", round=2, date="2026-11-17", home="GRS", away="PKA", hs=1, as_=1,
      venue="Stadion Gelora Joko Samudro, Gresik", att=1583, coach=("JOKO SUSILO", "TAUFIQ"),
      goals=[("h", 17, "MAULANA ILHAM JAYA KUSUMA", "78", "", "ileague"), ("a", 7, "REXO DOLBY CALLEPSO", "90+3", "", "ileague")],
      xi_h="""23|LALU MUHAMMAD RIZKI|GK|90;21|ABDUL RAHMAN|CB|C 90;33|JOKO SUPRIYANTO|CB|90;18|ALDI TRI NUGROHO|LB|87;16|ADE M NUR SAIHITUA|RB|90;7|ALIF JAELANI|CM|74;13|M. SYAIFULLAH SALIM|AM|45;24|MOHAMMAD JORDAN ZAMORANO|DM|90;9|DECO NOVERO PALANDI|CF|45;28|SHAHIH ELANG RISHANDY|CF|90;99|OSCAR NOVANGGA|LWF|74""",
      sub_h="22|M. DZAKI ALAMSYAH|45;29|MUHAMMAD BAIHAQI RIFAI|87;8|M ALWI FURQON|45;17|MAULANA ILHAM JAYA KUSUMA|74 78;85|DIAN ARDIANSYAH|74",
      xi_a="""31|NURI AGUS WIBOWO|GK|90;38|MUSRIL|LB|90;17|MUHAMMAD RIO SAPUTRO|CB|C 45;25|YULFIKAR MANSUR BAYUAJI|RB|90;12|RONI SUGENG ARIYANTO|CM|90;28|MOHAMAD IQBAL|DM|90;91|FEBRIAN TRI YANTO|CM|57;11|REYHAN FIRDAUS|LWF|45;18|M. SYAIFUL ASRI|CF|90;16|LIONEL ZAHRA ALFARIZKI|LWF|79;23|WAWAN FEBRIANTO|RWF|79""",
      sub_a="3|DHANU SYAH PUTRA|45;52|FADHILAH TEGAR NUR AZIZ|79;7|REXO DOLBY CALLEPSO|57 90+3;77|UMAR|79;8|YOGIEK FERNANDA|45"),
 dict(id="persinga_psf_20261118", round=2, date="2026-11-18", home="PNG", away="PSF", hs=1, as_=2,
      venue="Stadion Ketonggo, Ngawi", att=682, coach=("DINDIN WAHYUDIN", "AGUS SUGENG RIYANTO"),
      goals=[("a", 90, "JOHAN YOGA UTAMA", "22", "", "ileague"), ("h", 66, "MUCH ILHAM NOER", "56", "", "ileague"),
             ("a", 10, "KOMARODIN", "86", "", "ileague")],
      xi_h="""96|FAHMI IDRIS|GK|90;20|ARFAMUZI|RB|42;15|BAGAS AGUNG UTOMO|CB|90;19|NANDA HELVRIT SAPUTRA|CB|90;17|SAMSUL ALAM|CB|42;8|DEJAN ALIKHA MUHAMMAD|CM|90;11|DIMAS GALIH GUMILANG|AM|C 66;66|MUCH ILHAM NOER|CM|56 90;7|REZZA MAULANA FAHRULLAH|CF|63;57|QOYUM NAZALUDIN MUHAMAD|RWF|58;77|MUHAMAD FIRMAN JANUARI|LWF|90""",
      sub_h="14|BATHSYA DEWA APRILLIANO|66;3|SYARFUL MUDAWAM|42;27|ADITYA SENTOSA|58;22|DADAN AKBAR NUGRAHA|42",
      xi_a="""20|DAVA ARYULA SIREGAR|GK|90;30|OKY KHARISMA|LB|66;87|FATHUL IHSAN|CB|90;15|MUHAMMAD SURYA MAULANA|RB|66;21|FIRMANSYAH|AM|C 90;22|FAJAR IHSANUDIN|DM|81;99|KEVIN AL HAFIZ|DM|90;11|MOCH. ALEX SANDRO|RWF|45;9|AHMAD ATHALLAH ARAIHAN|CF|90;90|JOHAN YOGA UTAMA|CF|22 90;7|RIDWAN ANSORI|RWF|52""",
      sub_a="14|DIFO PRASETYA|52;29|FAHRI AKBAR ROMADHON|45;76|RADITYA ILHAM PRAMANA|81;18|DONA SAPUTRA|66;10|KOMARODIN|66 86",
      cards=[("h", 7, "red", "63")], notes=["Persinga #7 Rezza Maulana Fahrullah sent off 63' (straight red)"]),
 dict(id="persibangga_persibo_20261118", round=2, date="2026-11-18", home="PBG", away="PBO", hs=2, as_=1, ht=(1, 0),
      venue="Stadion Trisanja, Tegal", att=277, coach=("MOHAMAD IRFAN", "I PUTU GEDE SWI SANTOSO"),
      goals=[("h", 18, "USAMA AHMAD RAIHAN", "32", "", "ileague"), ("a", 7, "SEHABUDIN AHMAD", "48", "", "ileague"),
             ("h", 32, "FARREL ARYA TRI SANDIKA SUPRATIKNO", "90+4", "", "ileague")],
      xi_h="""76|HUSEIN AKBAR AL FAUZI|GK|90;16|AZHRIL SYAHDAN FABREGAS|LB|76;4|FATHUR YAN|CB|90;6|YUFRIZAL LU'AY ABIYU|CB|90;25|AHMAD RAMDIN TUASALAMONY|RB|90;7|ERLANGGA ADHYAKSA NUGRAHA|AM|90;17|AFIF TEGUH SAPUTRA|CM|76;30|ALBERT THEO WICAKSONO|CM|90;87|MUHAMMMAD ILHAMUL IRHAZ|DM|C 90;18|USAMA AHMAD RAIHAN|LWF|32 90;79|ADITHYA JORY GURUH SETIAWAN|CF|79""",
      sub_h="44|RESTU SETYA PRATAMA|76;11|ALFANDI|76;32|FARREL ARYA TRI SANDIKA SUPRATIKNO|79 90+4",
      xi_a="""14|ALDO FERNANDI|GK|90;3|CRAH EKA ANGGER ISWANTO|LB|C 90;6|TEGAR ANDRIE SHEVANTON|CB|90;26|RIO PRATAMA VALENTINO|CB|90;13|MOCHAMAD ADITYA RANGGA SAPUTRA|RB|90;8|MARVIN NUGRAHA HILMANSYAH|DM|90;18|MOHAMAD SYAHRIL AFRIZAL|DM|55;33|OCVIAN CHANIGIO|AM|45;7|SEHABUDIN AHMAD|CF|48 68;27|ROMEO NIZAM ABISHA|LWF|68;77|NURDIAN SYAPUTRA CHANIAGO|RWF|87""",
      sub_a="29|BASAJUMP LATUCONSINA|45;25|AARON YEKTI|55;10|IQBAL TRI SAPUTRA|68;9|KAKA PUTRA PRAMANA|68;11|RIFALDI BAWUO|87"),

 dict(id="persiba_persika_20261121", round=3, date="2026-11-21", home="BTL", away="PKA", hs=0, as_=1, ht=(0, 1),
      venue="Stadion Sultan Agung, Bantul", att=1313, coach=("SUSANTO", "TAUFIQ"),
      goals=[("a", 21, "MUHAMMAD FABIAN ASLAM", "29", "counter-attack (Jatengnews: 28')", "ileague + Jatengnews")],
      xi_h="""20|MOHAMMAD FEBRYANSYAH REFYAN SANUSI|GK|90;16|FACHRIZAL AHNAF MAULANA|CB|C 42 90;33|JOSEPH ALEXANDER KURNIAWAN|CB|90;87|AHMAD RIFAL|CB|88 90;29|RIAN RAMADAN|RB|64 65;11|ALDITO DAFFA QUARISMA|CM|86;14|MUH ARSIL ANAS|CM|45;17|RISKY DWIYAN APRILYANTO|DM|90;73|MUHAMMAD IQBAL DALIMUNTHE|RWF|45;77|SANDI ARTA SAMOSIR|RWF|90;86|ELVATEEH VARESIA AKHADA|RWF|83 84""",
      sub_h="8|AIDUN SASTRA UTAMI|84;66|CHOLIS BIMA SATRIA NATANEGARA|65;12|LORENZO GOMES DE OLIVIERA|45;24|MUH TIRTO WICAKSONO|45;10|REYKI FARIZ RAMADHAN|86 90+1",
      xi_a="""31|NURI AGUS WIBOWO|GK|90;3|DHANU SYAH PUTRA|CB|90;5|IFAN IZDIHAR IYANANDA|CB|90;21|MUHAMMAD FABIAN ASLAM|RB|29 42 77;25|YULFIKAR MANSUR BAYUAJI|RB|90;52|FADHILAH TEGAR NUR AZIZ|RB|56;77|UMAR|RB|59 63;4|ABI DEFA MARENDRA|CM|C 63;8|YOGIEK FERNANDA|AM|90;28|MOHAMAD IQBAL|DM|90;91|FEBRIAN TRI YANTO|CM|77""",
      sub_a="38|MUSRIL|77;16|LIONEL ZAHRA ALFARIZKI|56;36|RELOSA RIVAN SATRIA HUTAMA|77 87;7|REXO DOLBY CALLEPSO|63;12|RONI SUGENG ARIYANTO|63",
      cards=[("h", 16, "yellow", "42"), ("h", 29, "yellow", "64"), ("h", 86, "yellow", "83"), ("h", 87, "yellow", "88"),
             ("h", 10, "yellow", "90+1"), ("a", 21, "yellow", "42"), ("a", 77, "yellow", "59"), ("a", 36, "yellow", "87")]),
 dict(id="persinga_persibangga_20261122", round=3, date="2026-11-22", home="PNG", away="PBG", hs=1, as_=1, ht=(0, 1),
      venue="Stadion Ketonggo, Ngawi", att=810, coach=("DINDIN WAHYUDIN", "MOHAMAD IRFAN"),
      goals=[("a", 18, "USAMA AHMAD RAIHAN", "27", "", "ileague + Radar Solo"), ("h", 66, "MUCH ILHAM NOER", "67", "", "ileague + Radar Solo")],
      xi_h="""96|FAHMI IDRIS|GK|90;13|IKBAL KHADAFI|LB|45;3|SYARFUL MUDAWAM|CB|90;19|NANDA HELVRIT SAPUTRA|CB|C 90;25|JOHANES EDWARD SUBROTO|RB|81;8|DEJAN ALIKHA MUHAMMAD|CM|90;27|ADITYA SENTOSA|CM|58;66|MUCH ILHAM NOER|CM|67 90;9|MOH.NASRUL ARIFIN|CF|68;57|QOYUM NAZALUDIN MUHAMAD|RWF|58;77|MUHAMAD FIRMAN JANUARI|LWF|90""",
      sub_h="17|SAMSUL ALAM|45;18|M.ALVIN DWI GUNA|58;10|DAMAR YUDHISTIRA PANGESTU|81;21|AKBAR SYAKIRA|68;99|MUHAMAD GAOSHIROWI|58",
      xi_a="""76|HUSEIN AKBAR AL FAUZI|GK|90;2|MUHAMMAD KHOIRUL AMIN|LB|45;16|AZHRIL SYAHDAN FABREGAS|LB|43;6|YUFRIZAL LU'AY ABIYU|CB|90;15|I MADE DWI MEIYANA PUTRA|CB|45;25|AHMAD RAMDIN TUASALAMONY|RB|90;7|ERLANGGA ADHYAKSA NUGRAHA|AM|90;30|ALBERT THEO WICAKSONO|CM|90;87|MUHAMMMAD ILHAMUL IRHAZ|DM|C 72;18|USAMA AHMAD RAIHAN|LWF|27 90;79|ADITHYA JORY GURUH SETIAWAN|CF|75""",
      sub_a="44|RESTU SETYA PRATAMA|45;4|FATHUR YAN|45;22|MUHAMMAD NAJA NAUFAL HILMI|72;11|ALFANDI|43;32|FARREL ARYA TRI SANDIKA SUPRATIKNO|75"),
 dict(id="persipa_psf_20261122", round=3, date="2026-11-22", home="PIP", away="PSF", hs=0, as_=1,
      venue="Stadion Moch. Soebroto, Magelang", att=147, coach=("SALAHUDDIN", "AGUS SUGENG RIYANTO"),
      goals=[("a", 90, "JOHAN YOGA UTAMA", "48", "pen", "ileague")],
      xi_h="""88|ARIFIN SETYADI|GK|90;17|MUHAMMAD IQBAL AL GHUZAT|LB|C 90;5|IMAM MAHMUDI|CB|90;44|MUHAMMAD RHAGA AL MISRI YULIANTO|CB|73;71|MUH. AL - QADRI|RB|90;8|AKHMAD RIYADI|DM|90;14|YOSUA RAFAEL ATALYA YOLTUWU|AM|90;47|HERSYA SCIFO|CM|56;10|FRITY ARMAN|CF|56;16|DANHYL POETRA MARSELINHO TUKA|LWF|86;99|MOCHAMMAD RISAL AMIN|RWF|56""",
      sub_h="36|EDGARDA FAHAR BUNTORO|56;24|ALFONS RIZAL SAMAN|86;19|MOCH REZA ANDREA ARSHAVIN|56;15|ILHAM WIBOWO YUSUF|73;52|SADDAM EMIRUDDIN GAFFAR|56",
      xi_a="""20|DAVA ARYULA SIREGAR|GK|90;14|DIFO PRASETYA|LB|90;75|RIO FERNANDO DWI ARDIANSYAH|CB|90;15|MUHAMMAD SURYA MAULANA|RB|90;51|AIRLANGGA MUTAMASIQDINA|RB|90;21|FIRMANSYAH|AM|C 90;99|KEVIN AL HAFIZ|DM|59;8|FREZY AL HUDAIFI|LWF|45;90|JOHAN YOGA UTAMA|CF|48 84;11|MOCH. ALEX SANDRO|RWF|79;7|RIDWAN ANSORI|RWF|59""",
      sub_a="27|ZIDANE ALZIZOU LUBIS|45;29|FAHRI AKBAR ROMADHON|59;76|RADITYA ILHAM PRAMANA|59;40|MUHAMMAD FAHRUL ROJDI|79;10|KOMARODIN|84"),
 dict(id="persibo_gresik_20261122", round=3, date="2026-11-22", home="PBO", away="GRS", hs=1, as_=1,
      venue="Stadion Letjen H. Soedirman, Bojonegoro", att=3360, coach=("I PUTU GEDE SWI SANTOSO", "JOKO SUSILO"),
      goals=[("h", 10, "IQBAL TRI SAPUTRA", "79", "", "ileague"), ("a", 22, "M. DZAKI ALAMSYAH", "88", "", "ileague")],
      xi_h="""14|ALDO FERNANDI|GK|90;3|CRAH EKA ANGGER ISWANTO|LB|C 90;6|TEGAR ANDRIE SHEVANTON|CB|90;26|RIO PRATAMA VALENTINO|CB|90;13|MOCHAMAD ADITYA RANGGA SAPUTRA|RB|90;8|MARVIN NUGRAHA HILMANSYAH|DM|90;18|MOHAMAD SYAHRIL AFRIZAL|DM|90;27|ROMEO NIZAM ABISHA|AM|90;7|SEHABUDIN AHMAD|CF|90;29|BASAJUMP LATUCONSINA|LWF|71;77|NURDIAN SYAPUTRA CHANIAGO|RWF|90""",
      sub_h="25|AARON YEKTI|90;10|IQBAL TRI SAPUTRA|71 79;9|KAKA PUTRA PRAMANA|90",
      xi_a="""77|SYAMIL BAHIJ IRAWAN|GK|90;18|ALDI TRI NUGROHO|LB|90;33|JOKO SUPRIYANTO|CB|C 90;16|ADE M NUR SAIHITUA|RB|90;7|ALIF JAELANI|CM|90;8|M ALWI FURQON|AM|80;13|M. SYAIFULLAH SALIM|AM|45;15|MUHAMAD FADILLA AKBAR|DM|70;24|MOHAMMAD JORDAN ZAMORANO|DM|90;9|DECO NOVERO PALANDI|CF|45;28|SHAHIH ELANG RISHANDY|CF|90""",
      sub_a="22|M. DZAKI ALAMSYAH|80 88;10|RAFI HERLIAN ADINATA|45;46|WAHYU AGONG DRAJAT MULYONO|70;99|OSCAR NOVANGGA|45;17|MAULANA ILHAM JAYA KUSUMA|90"),

 dict(id="gresik_persibangga_20261126", round=4, date="2026-11-26", home="GRS", away="PBG", hs=2, as_=1, ht=(0, 1),
      venue="Stadion Gelora Joko Samudro, Gresik", att=1209, coach=("JOKO SUSILO", "MOHAMAD IRFAN"),
      goals=[("a", 18, "USAMA AHMAD RAIHAN", "32", "", "ileague"), ("h", 28, "SHAHIH ELANG RISHANDY", "78", "pen", "ileague"),
             ("h", 28, "SHAHIH ELANG RISHANDY", "90", "", "ileague")],
      xi_h="""77|SYAMIL BAHIJ IRAWAN|GK|90;18|ALDI TRI NUGROHO|LB|90;21|ABDUL RAHMAN|CB|C 90;33|JOKO SUPRIYANTO|CB|90;16|ADE M NUR SAIHITUA|RB|82;7|ALIF JAELANI|CM|90;8|M ALWI FURQON|AM|73;13|M. SYAIFULLAH SALIM|AM|45;9|DECO NOVERO PALANDI|CF|45;22|M. DZAKI ALAMSYAH|CF|54;85|DIAN ARDIANSYAH|RWF|45""",
      sub_h="10|RAFI HERLIAN ADINATA|45;15|MUHAMAD FADILLA AKBAR|54;46|WAHYU AGONG DRAJAT MULYONO|73;99|OSCAR NOVANGGA|45;28|SHAHIH ELANG RISHANDY|45 90+5",
      xi_a="""76|HUSEIN AKBAR AL FAUZI|GK|90;25|AHMAD RAMDIN TUASALAMONY|RB|90;16|AZHRIL SYAHDAN FABREGAS|LB|90;4|FATHUR YAN|CB|59;15|I MADE DWI MEIYANA PUTRA|CB|90;7|ERLANGGA ADHYAKSA NUGRAHA|AM|C 90;17|AFIF TEGUH SAPUTRA|CM|90;30|ALBERT THEO WICAKSONO|CM|77;22|MUHAMMAD NAJA NAUFAL HILMI|DM|90;18|USAMA AHMAD RAIHAN|LWF|32 90;79|ADITHYA JORY GURUH SETIAWAN|CF|63""",
      sub_a="44|RESTU SETYA PRATAMA|59;11|ALFANDI|77;32|FARREL ARYA TRI SANDIKA SUPRATIKNO|63",
      suspended=[("a", 6, "YUFRIZAL LU'AY ABIYU", "2 yellow cards (matches 4 and 12)"),
                 ("a", 87, "MUHAMMMAD ILHAMUL IRHAZ", "2 yellow cards (matches 4 and 12)"),
                 ("h", 24, "MOHAMMAD JORDAN ZAMORANO", "2 yellow cards (matches 1 and 11)")]),
 dict(id="psf_persibo_20261126", round=4, date="2026-11-26", home="PSF", away="PBO", hs=1, as_=0,
      venue="Stadion Gelora Soepriadi, Blitar", att=733, coach=("AGUS SUGENG RIYANTO", "I PUTU GEDE SWI SANTOSO"),
      goals=[("h", 10, "KOMARODIN", "50", "", "ileague")],
      xi_h="""20|DAVA ARYULA SIREGAR|GK|90;14|DIFO PRASETYA|LB|45;87|FATHUL IHSAN|CB|90;15|MUHAMMAD SURYA MAULANA|RB|45;29|FAHRI AKBAR ROMADHON|RB|90;12|FIGO FIRMANSYAH|AM|88;21|FIRMANSYAH|AM|C 90;22|FAJAR IHSANUDIN|DM|66;9|AHMAD ATHALLAH ARAIHAN|CF|90;10|KOMARODIN|RWF|50 73;90|JOHAN YOGA UTAMA|CF|90""",
      sub_h="30|OKY KHARISMA|45;11|MOCH. ALEX SANDRO|45;76|RADITYA ILHAM PRAMANA|66;26|CHARLES GEORGE RUMAROPEN|88;18|DONA SAPUTRA|73",
      xi_a="""14|ALDO FERNANDI|GK|90;3|CRAH EKA ANGGER ISWANTO|LB|C 90;76|JOVAN JORDANTIAN SYAHWA MAULANA|RB|63;26|RIO PRATAMA VALENTINO|CB|90;6|TEGAR ANDRIE SHEVANTON|CB|90;8|MARVIN NUGRAHA HILMANSYAH|DM|90;18|MOHAMAD SYAHRIL AFRIZAL|DM|90;77|NURDIAN SYAPUTRA CHANIAGO|AM|63;11|RIFALDI BAWUO|LWF|56;25|AARON YEKTI|RWF|56;27|ROMEO NIZAM ABISHA|CF|80""",
      sub_a="34|KAFKA KHAYLA AZWA|80;13|MOCHAMAD ADITYA RANGGA SAPUTRA|63;7|SEHABUDIN AHMAD|56;10|IQBAL TRI SAPUTRA|56;33|OCVIAN CHANIGIO|63",
      suspended=[("h", 99, "KEVIN AL HAFIZ", "2 yellow cards"), ("h", 27, "ZIDANE ALZIZOU LUBIS", "2 yellow cards"),
                 ("a", 29, "BASAJUMP LATUCONSINA", "2 yellow cards"), ("a", 58, "ADI PURNAMA S. SUAIB", "disciplinary ban")]),
 dict(id="persiba_persinga_20261126", round=4, date="2026-11-26", home="BTL", away="PNG", hs=0, as_=1, ht=(0, 1),
      venue="Stadion Sultan Agung, Bantul", att=188, coach=("SUSANTO", "DINDIN WAHYUDIN"),
      goals=[("a", 27, "ADITYA SENTOSA", "29", "", "ileague")],
      xi_h="""81|YOGI HERMAWAN|GK|90;3|WIGANDA PRADIKA|LB|72;12|LORENZO GOMES DE OLIVIERA|LB|90;13|MARCELL JANUAR PUTRA|CB|45;33|JOSEPH ALEXANDER KURNIAWAN|CB|90;8|AIDUN SASTRA UTAMI|CM|60;11|ALDITO DAFFA QUARISMA|CM|90;24|MUH TIRTO WICAKSONO|DM|12;98|RAKHA ZULHILMI FASYIR|AM|90;7|ANTONI PUTRO NUGROHO|LWF|C 45;77|SANDI ARTA SAMOSIR|RWF|90""",
      sub_h="87|AHMAD RIFAL|45;6|MIFTHAHUL ULUMIL ULYA|12;14|MUH ARSIL ANAS|45;17|RISKY DWIYAN APRILYANTO|72;19|VENGKO ARMEDYA|60",
      xi_a="""34|MUHAMAD SURYANULLOH|GK|90;13|IKBAL KHADAFI|RB|87;19|NANDA HELVRIT SAPUTRA|CB|C 90;3|SYARFUL MUDAWAM|CB|90;27|ADITYA SENTOSA|CM|29 90;8|DEJAN ALIKHA MUHAMMAD|CM|90;66|MUCH ILHAM NOER|CM|45;16|ZAM ZAM SOVIA AROPA|CM|90;14|BATHSYA DEWA APRILLIANO|CF|90;9|MOH. NASRUL ARIFIN|CF|45;77|MUHAMAD FIRMAN JANUARI|LWF|79""",
      sub_a="21|AKBAR SYAKIRA|45;18|M.ALVIN DWI GUNA|45;22|DADAN AKBAR NUGRAHA|87;97|ZERY ALPINDO KEMBAREN|79",
      suspended=[("h", 10, "REYKI FARIZ RAMADHAN", "2 yellow cards (63' vs Persipa, 90+1' vs Persika)"),
                 ("h", 86, "ELVATEEH VARESIA AKHADA", "2 yellow cards (32' vs Persipa, 83' vs Persika)")]),
 dict(id="persika_persipa_20261126", round=4, date="2026-11-26", home="PKA", away="PIP", hs=3, as_=1,
      venue="Stadion Manahan, Solo", att=1962, coach=("TAUFIQ", "SALAHUDDIN"),
      goals=[("h", 21, "MUHAMMAD FABIAN ASLAM", "19", "", "ileague"), ("h", 21, "MUHAMMAD FABIAN ASLAM", "34", "", "ileague"),
             ("h", 91, "FEBRIAN TRI YANTO", "43", "", "ileague"), ("a", 15, "ILHAM WIBOWO YUSUF", "49", "", "ileague")],
      xi_h="""31|NURI AGUS WIBOWO|GK|90;38|MUSRIL|LB|90;5|IFAN IZDIHAR IYANANDA|CB|90;21|MUHAMMAD FABIAN ASLAM|RB|19 34 90;25|YULFIKAR MANSUR BAYUAJI|RB|90;4|ABI DEFA MARENDRA|CM|74;8|YOGIEK FERNANDA|AM|88;28|MOHAMAD IQBAL|DM|90;91|FEBRIAN TRI YANTO|CM|43 64;16|LIONEL ZAHRA ALFARIZKI|LWF|64;23|WAWAN FEBRIANTO|RWF|64""",
      sub_h="14|ARIF AGUNG MUSTHOFA|88;52|FADHILAH TEGAR NUR AZIZ|64;36|RELOSA RIVAN SATRIA HUTAMA|74;7|REXO DOLBY CALLEPSO|64;77|UMAR|64",
      xi_a="""1|ADHIEL TRISTANTO|GK|90;94|DIMAS SAPUTRA|LB|45;15|ILHAM WIBOWO YUSUF|CB|49 80;28|RIONALDI MEHRAN|CB|90;71|MUH. AL - QADRI|RB|90;8|AKHMAD RIYADI|DM|90;14|YOSUA RAFAEL ATALYA YOLTUWU|AM|90;36|EDGARDA FAHAR BUNTORO|AM|45;7|PANDI AHMAD LESTALUHU|CF|C 90;52|SADDAM EMIRUDDIN GAFFAR|CF|90;99|MOCHAMMAD RISAL AMIN|RWF|45""",
      sub_a="17|MUHAMMAD IQBAL AL GHUZAT|45;16|DANHYL POETRA MARSELINHO TUKA|45;44|MUHAMMAD RHAGA AL MISRI YULIANTO|80;86|YOHANES YAPAGAIMU|45"),
]

FIXTURES_LEFT = [  # Group B, from ileague / betexplorer (27 Nov 2026); times WIB
    (5, "2026-11-30", "PNG", "PBO"), (5, "2026-11-30", "GRS", "PIP"), (5, "2026-11-30", "PBG", "BTL"), (5, "2026-11-30", "PKA", "PSF"),
]
BTL_REMAINING = [  # PERSIBA's run-in (betexplorer team page, 27 Nov 2026)
    (5, "2026-11-30", "PBG", "BTL"), (6, "2026-12-04", "PBO", "BTL"), (7, "2026-12-08", "BTL", "PSF"),
    (8, "2026-12-12", "PKA", "BTL"), (9, "2026-12-16", "PNG", "BTL"), (10, "2026-12-20", "BTL", "PBO"),
]


def mins(tok: str) -> int:
    m = re.match(r"(\d+)(?:\+(\d+))?", tok)
    return int(m.group(1)) if m else 90


def parse_xi(block: str) -> list[dict]:
    out = []
    for item in [x for x in block.replace("\n", "").split(";") if x.strip()]:
        no, name, pos, marks = item.split("|")
        cap = marks.startswith("C ")
        marks = marks[2:] if cap else marks
        out.append({"no": int(no), "name": name.strip(), "pos": pos, "captain": cap, "markers": marks.split()})
    return out


def parse_subs(block: str) -> list[dict]:
    out = []
    for item in [x for x in block.split(";") if x.strip()]:
        no, name, marks = item.split("|")
        out.append({"no": int(no), "name": name.strip(), "markers": marks.split()})
    return out


def build() -> dict:
    matches, players = [], defaultdict(lambda: {"starts": 0, "sub_apps": 0, "minutes": 0, "goals": 0, "pens": 0,
                                                "captain": 0, "positions": defaultdict(int), "yellow": 0, "red": 0,
                                                "matches": []})
    for m in MATCHES:
        rec = {k: v for k, v in m.items() if not k.startswith(("xi_", "sub_"))}
        rec["score"] = [m["hs"], m["as_"]]
        rec.pop("hs"); rec.pop("as_")
        rec["home_name"], rec["away_name"] = TEAMS[m["home"]], TEAMS[m["away"]]
        rec["goals"] = [{"side": s, "no": no, "name": nm, "minute": mi, "note": nt, "source": src}
                        for s, no, nm, mi, nt, src in m.get("goals", [])]
        for side in ("h", "a"):
            club = m["home"] if side == "h" else m["away"]
            xi, subs = parse_xi(m[f"xi_{side}"]), parse_subs(m[f"sub_{side}"])
            goal_min = {(g["no"], mins(g["minute"])) for g in rec["goals"] if g["side"] == side}
            sub_marks = {mins(t) for s in subs for t in s["markers"]}
            xi_marks = {mins(t) for p in xi for t in p["markers"]}
            for p in xi:
                cand = sorted(mins(t) for t in p["markers"] if mins(t) in sub_marks and (p["no"], mins(t)) not in goal_min
                              and mins(t) < 90)
                p["off"] = cand[-1] if cand else 90
                p["minutes"] = p["off"]
            for s in subs:
                cand = sorted(mins(t) for t in s["markers"] if mins(t) in xi_marks and (s["no"], mins(t)) not in goal_min)
                on = cand[0] if cand else min(mins(t) for t in s["markers"])
                s["on"] = on
                s["minutes"] = max(1, 90 - on)
            rec[f"xi_{'home' if side == 'h' else 'away'}"] = xi
            rec[f"subs_{'home' if side == 'h' else 'away'}"] = subs
            for p, started in [(x, True) for x in xi] + [(x, False) for x in subs]:
                key = f"{club}|{p['no']}|{p['name']}"
                pl = players[key]
                pl["starts" if started else "sub_apps"] += 1
                pl["minutes"] += p["minutes"]
                if started:
                    pl["positions"][p["pos"]] += 1
                    pl["captain"] += int(p["captain"])
                pl["matches"].append(m["id"])
            for g in rec["goals"]:
                if g["side"] == side:
                    key = next((k for k in players if k.startswith(f"{club}|{g['no']}|")), None)
                    if key:
                        players[key]["goals"] += 1
                        players[key]["pens"] += int(g["note"] == "pen")
            for c in m.get("cards", []):
                if c[0] == side:
                    key = next((k for k in players if k.startswith(f"{club}|{c[1]}|")), None)
                    if key:
                        players[key]["yellow" if c[2] == "yellow" else "red"] += 1
        matches.append(rec)

    table = defaultdict(lambda: {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0,
                                 "gf_1h": 0, "gf_2h": 0, "ga_1h": 0, "ga_2h": 0})
    for r in matches:
        h, a = r["home"], r["away"]
        hs, as_ = r["score"]
        for club, gf, ga in ((h, hs, as_), (a, as_, hs)):
            t = table[club]
            t["p"] += 1; t["gf"] += gf; t["ga"] += ga
            t["w" if gf > ga else "d" if gf == ga else "l"] += 1
            t["pts"] += 3 if gf > ga else 1 if gf == ga else 0
        for g in r["goals"]:
            scorer = h if g["side"] == "h" else a
            conceder = a if g["side"] == "h" else h
            half = "1h" if mins(g["minute"]) <= 45 else "2h"
            table[scorer][f"gf_{half}"] += 1
            table[conceder][f"ga_{half}"] += 1
    standings = sorted(({"club": k, "name": TEAMS[k], **v, "gd": v["gf"] - v["ga"]} for k, v in table.items()),
                       key=lambda r: (-r["pts"], -r["gd"], -r["gf"]))
    plist = []
    for key, v in players.items():
        club, no, name = key.split("|", 2)
        plist.append({"club": club, "no": int(no), "name": name, **{k: v[k] for k in
                      ("starts", "sub_apps", "minutes", "goals", "pens", "captain", "yellow", "red")},
                      "positions": dict(v["positions"]), "apps": v["starts"] + v["sub_apps"]})
    plist.sort(key=lambda p: (p["club"], -p["minutes"]))
    return {"schema": "groupb_matches_v1", "competition": "Liga Nusantara 2026-27 — Group B",
            "updated": "2026-11-27", "through_round": 4, "source": "ileague.id match pages; scorers cross-checked "
            "with Tribun Jogja, Jatengnews, Radar Madiun, Radar Solo where the ileague list mixes goals and cards",
            "teams": TEAMS, "standings": standings, "matches": matches, "players": plist,
            "fixtures_next": [{"round": r, "date": d, "home": h, "away": a} for r, d, h, a in FIXTURES_LEFT],
            "persiba_remaining": [{"round": r, "date": d, "home": h, "away": a} for r, d, h, a in BTL_REMAINING],
            "minutes_note": "Minutes are approximate (substitution minutes inferred from unlabeled page markers)."}


def main() -> int:
    data = build()
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    goals = sum(len(m["goals"]) for m in data["matches"])
    print(f"wrote {OUT}: {len(data['matches'])} matches, {goals} goals, {len(data['players'])} player rows")
    for r in data["standings"]:
        print(f"  {r['club']} P{r['p']} {r['w']}-{r['d']}-{r['l']} {r['gf']}:{r['ga']} {r['pts']} pts  "
              f"(GF 1H/2H {r['gf_1h']}/{r['gf_2h']}, GA 1H/2H {r['ga_1h']}/{r['ga_2h']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
