#!/usr/bin/env python3
"""How good are Track B's event candidates? Compare them with hand-tagged events.

Ground truth = the Tagger's `<match_id>_events.csv`, MANUAL events only (accepted
Track B candidates would grade Track B with its own answers). Tag a stretch of the
match by hand with the normal tagging panel — 10–15 minutes is enough — without
looking at the Review tab, export the events, then:

    python scripts/evaluate_candidates.py --candidates runs/<id>/<id>_candidates.csv \
        --events <id>_events.csv [--from 900 --to 1800] [--tolerance 2.0] [--output eval.json]

The window defaults to the span of the manual events (video seconds). A candidate
matches an event of the same type group and team whose time is within --tolerance
seconds (one-to-one, closest first). Reported per type:
  recall     share of hand-tagged events Track B proposed
  precision  share of Track B candidates that were real events
  ...also recall if only the default queue (priority 1-2) is reviewed, precision per
  priority, side errors (right time and type, wrong team) and pass-outcome agreement.
These numbers are what to tune build_event_candidates.py against.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

GROUP = {"pass": "pass", "cross": "pass", "shot": "shot", "tackle": "tackle", "interception": "interception",
         "recovery": "recovery", "clearance": "clearance"}
PASS_OUT = {"complete": "ok", "incomplete": "fail", "ok": "ok", "fail": "fail"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def fnum(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_truth(rows: list[dict[str, str]], include_accepted: bool) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        src = (r.get("source") or "manual").strip() or "manual"
        if src != "manual" and not include_accepted:
            continue
        t = fnum(r.get("video_seconds"))
        g = GROUP.get((r.get("event") or "").strip())
        if t is None or g is None:
            continue
        out.append({"t": t, "group": g, "type": r["event"].strip(), "side": (r.get("side") or "").strip(),
                    "outcome": PASS_OUT.get((r.get("outcome") or "").strip(), ""), "id": r.get("id", "")})
    return out


def load_cands(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        t = fnum(r.get("t"))
        g = GROUP.get((r.get("type") or "").strip())
        if t is None or g is None:
            continue
        out.append({"t": t, "group": g, "type": r["type"].strip(), "side": (r.get("side") or "").strip(),
                    "outcome": (r.get("outcome") or "").strip(), "priority": int(fnum(r.get("priority")) or 2),
                    "id": r.get("cand_id", "")})
    return out


def match_greedy(truth: list[dict[str, Any]], cands: list[dict[str, Any]], tol: float,
                 same_side: bool = True) -> list[tuple[int, int, float]]:
    pairs = []
    for i, e in enumerate(truth):
        for j, c in enumerate(cands):
            if c["group"] != e["group"] or (same_side and c["side"] != e["side"]):
                continue
            dt = abs(c["t"] - e["t"])
            if dt <= tol:
                pairs.append((dt, i, j))
    pairs.sort()
    used_t, used_c, out = set(), set(), []
    for dt, i, j in pairs:
        if i in used_t or j in used_c:
            continue
        used_t.add(i)
        used_c.add(j)
        out.append((i, j, dt))
    return out


def ratio(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


def evaluate(truth: list[dict[str, Any]], cands: list[dict[str, Any]], t0: float, t1: float,
             tol: float) -> dict[str, Any]:
    truth = [e for e in truth if t0 <= e["t"] <= t1]
    cands = [c for c in cands if t0 - tol <= c["t"] <= t1 + tol]
    m = match_greedy(truth, cands, tol)
    mt, mc = {i for i, _, _ in m}, {j for _, j, _ in m}
    # wrong team: right time and type group, other side
    rest_t = [i for i in range(len(truth)) if i not in mt]
    rest_c = [j for j in range(len(cands)) if j not in mc]
    loose = match_greedy([truth[i] for i in rest_t], [cands[j] for j in rest_c], tol, same_side=False)

    groups = sorted({e["group"] for e in truth} | {c["group"] for c in cands})
    per: dict[str, Any] = {}
    for g in groups + ["all"]:
        ti = [i for i, e in enumerate(truth) if g in ("all", e["group"])]
        cj = [j for j, c in enumerate(cands) if g in ("all", c["group"])]
        hit_t = [i for i in ti if i in mt]
        hit_c = [j for j in cj if j in mc]
        queue_hits = [i for i, j, _ in m if i in ti and cands[j]["priority"] <= 2]
        prio = {}
        for p in (1, 2, 3):
            cp = [j for j in cj if cands[j]["priority"] == p]
            prio[str(p)] = {"candidates": len(cp), "precision": ratio(sum(1 for j in cp if j in mc), len(cp))}
        row = {"tagged": len(ti), "candidates": len(cj), "matched": len(hit_t),
               "recall": ratio(len(hit_t), len(ti)), "precision": ratio(len(hit_c), len(cj)),
               "recall_default_queue": ratio(len(queue_hits), len(ti)), "by_priority": prio,
               "median_dt_s": None}
        dts = sorted(dt for i, _, dt in m if i in ti)
        if dts:
            row["median_dt_s"] = round(dts[len(dts) // 2], 2)
        if g in ("pass", "all"):
            pm = [(truth[i], cands[j]) for i, j, _ in m if truth[i]["group"] == "pass" and i in ti]
            agree = sum(1 for e, c in pm if e["outcome"] and e["outcome"] == c["outcome"])
            row["pass_outcome_agreement"] = ratio(agree, sum(1 for e, _ in pm if e["outcome"]))
        per[g] = row
    return {
        "window_s": [round(t0, 1), round(t1, 1)], "tolerance_s": tol,
        "tagged_events": len(truth), "candidates": len(cands), "matched": len(m),
        "wrong_team_matches": len(loose),
        "per_type": per,
        "missed": [truth[i] for i in range(len(truth)) if i not in mt][:200],
        "false_candidates": [cands[j] for j in range(len(cands)) if j not in mc][:200],
    }


def markdown(res: dict[str, Any]) -> str:
    pct = lambda v: "—" if v is None else f"{v:.0%}"
    lines = [f"Window {res['window_s'][0]:.0f}–{res['window_s'][1]:.0f} s (video) · tolerance ±{res['tolerance_s']} s · "
             f"{res['tagged_events']} hand-tagged events · {res['candidates']} candidates · "
             f"{res['wrong_team_matches']} right time/type but wrong team", "",
             "| Type | Tagged | Candidates | Recall | Recall (default queue) | Precision | P1 prec. | P2 prec. | P3 prec. |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g, r in res["per_type"].items():
        bp = r["by_priority"]
        lines.append(f"| {g} | {r['tagged']} | {r['candidates']} | {pct(r['recall'])} | {pct(r['recall_default_queue'])} | "
                     f"{pct(r['precision'])} | {pct(bp['1']['precision'])} | {pct(bp['2']['precision'])} | "
                     f"{pct(bp['3']['precision'])} |")
    po = res["per_type"].get("pass", {}).get("pass_outcome_agreement")
    if po is not None:
        lines += ["", f"Pass outcome (complete/incomplete) agrees on {po:.0%} of matched passes."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Precision/recall of Track B candidates against hand-tagged events.")
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True, help="Tagger <match_id>_events.csv")
    ap.add_argument("--from", dest="t0", type=float, help="window start, video seconds")
    ap.add_argument("--to", dest="t1", type=float, help="window end, video seconds")
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--include-accepted", action="store_true",
                    help="also count accepted Track B events as truth (biased; for a rough look only)")
    ap.add_argument("--output", type=Path, help="write the full result (incl. missed/false lists) as JSON")
    a = ap.parse_args()
    truth = load_truth(read_csv(a.events), a.include_accepted)
    if not truth:
        raise SystemExit("No manual events with video_seconds in that file (tag some by hand first).")
    t0 = a.t0 if a.t0 is not None else min(e["t"] for e in truth)
    t1 = a.t1 if a.t1 is not None else max(e["t"] for e in truth)
    res = evaluate(truth, load_cands(read_csv(a.candidates)), t0, t1, a.tolerance)
    print(markdown(res))
    if a.output:
        a.output.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("wrote", a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
