import csv
import json

import pytest

from scripts.run_match import (
    TRACK_ID_BLOCK, chunk_done, chunk_name, combine, home_right_by_period,
    match_periods, parse_clock, plan_chunks, resolve_stride,
)


def test_parse_clock_formats():
    assert parse_clock("17:00") == 1020
    assert parse_clock("1:04:41") == 3881
    assert parse_clock(90) == 90.0
    with pytest.raises(ValueError):
        parse_clock("1:2:3:4")


def test_periods_require_halves_and_sort_by_source_time():
    m = {"match_id": "x", "periods": {"2": ["1:20:26", "2:07:27"], "1": ["17:00", "1:04:41"]}}
    ps = match_periods(m)
    assert [p["period"] for p in ps] == [1, 2]
    with pytest.raises(SystemExit):
        match_periods({"match_id": "x", "periods": {"1": ["0:10", "45:00"], "2": None}})
    with pytest.raises(SystemExit):  # overlapping halves
        match_periods({"match_id": "x", "periods": {"1": ["0:00", "50:00"], "2": ["49:00", "95:00"]}})


def test_extra_time_needs_its_own_direction():
    ps = [{"period": 1}, {"period": 2}, {"period": 3}, {"period": 4}]
    assert home_right_by_period({"home_dir_1h": "R"}, ps) is None
    d = home_right_by_period({"home_dir_1h": "R", "home_dir_et1": "L"}, ps)
    assert d == {"1": True, "2": False, "3": False, "4": True}
    assert home_right_by_period({"home_dir_1h": None}, ps[:2]) is None


def test_chunks_cover_each_half_without_slivers():
    ps = [{"period": 1, "start": 1020.0, "end": 3881.0}, {"period": 2, "start": 4826.0, "end": 7647.0}]
    ch = plan_chunks(ps, 600)
    for p in ps:
        mine = [c for c in ch if c["period"] == p["period"]]
        assert mine[0]["start"] == p["start"] and mine[-1]["end"] == p["end"]
        assert all(a["end"] == b["start"] for a, b in zip(mine, mine[1:]))
        assert all(c["end"] - c["start"] >= 30 for c in mine)
    assert [c["k"] for c in ch] == list(range(len(ch)))


def test_auto_stride_targets_ten_samples_per_second():
    s = {"frame_stride": "auto"}; resolve_stride(s, 29.97); assert s["frame_stride"] == 3
    s = {"frame_stride": "auto"}; resolve_stride(s, 59.94); assert s["frame_stride"] == 6
    s = {"frame_stride": 4}; resolve_stride(s, 60); assert s["frame_stride"] == 4


def _write_chunk(run, mid, c, rows, status="complete"):
    p = run / "chunks" / (chunk_name(mid, c) + ".csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = ["match_id", "frame", "video_seconds", "period", "track_id", "role", "team", "x", "y",
              "confidence", "camera_state"]
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    p.with_suffix(".meta.json").write_text(json.dumps({
        "status": status, "complete": status == "complete", "match_id": mid, "period": c["period"],
        "frame_stride": 3, "imgsz": 960, "fps": 30.0, "track_id_offset": c["k"] * TRACK_ID_BLOCK,
        "requested_start_seconds": c["start"], "requested_end_seconds": c["end"],
        "actual_end_seconds": c["end"] - 0.05, "processed_frames": len(rows),
    }))
    return p


def test_chunk_done_and_combine_with_team_swap(tmp_path):
    mid, settings = "m1", {"frame_stride": 3, "imgsz": 960}
    run = tmp_path / "runs" / mid
    c0 = {"k": 0, "period": 1, "start": 10.0, "end": 20.0}
    c1 = {"k": 1, "period": 2, "start": 100.0, "end": 110.0}
    row = lambda f, team: {"match_id": mid, "frame": f, "video_seconds": f / 30, "period": 1, "track_id": 1,
                           "role": "player", "team": team, "x": 50, "y": 30, "confidence": 0.9, "camera_state": "wide"}
    p0 = _write_chunk(run, mid, c0, [row(300, "home"), row(303, "away")])
    _write_chunk(run, mid, c1, [row(3000, "unknown")], status="partial_eof")
    assert chunk_done(p0, c0, mid, settings)
    assert not chunk_done(run / "chunks" / (chunk_name(mid, c1) + ".csv"), c1, mid, settings)
    assert not chunk_done(p0, c0, mid, {"frame_stride": 6, "imgsz": 960})  # settings changed -> redo

    _write_chunk(run, mid, c1, [row(3000, "unknown")])
    out = run / f"{mid}_tracking.csv"
    meta = combine(run, mid, [c0, c1], True, out, [{"period": 1, "start": 10.0, "end": 20.0}], None)
    teams = [r["team"] for r in csv.DictReader(out.open())]
    assert teams == ["away", "home", "unknown"]
    assert meta["rows"] == 3 and meta["teams_swapped_at_combine"] is True
