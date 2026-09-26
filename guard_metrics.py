#!/usr/bin/env python3
"""Metrics for the system-one-guard flow log.

Reads the JSONL the plugin writes and prints whether the flow is doing its job: latency per
hook, risk distribution, gate decisions, and the per-turn timeline. Read this instead of
guessing: a gate that never fires on your traffic is a threshold, not a safety property.

Usage:
  python3 guard_metrics.py [--log PATH] [--db PATH] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def default_log_path() -> str:
    if os.environ.get("JEV_LOG"):
        return os.environ["JEV_LOG"]
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    candidates = list((Path(home) / "plugin-data").glob("*jev-guard*/jev-flow.jsonl"))
    if candidates:
        return str(max(candidates, key=lambda p: p.stat().st_mtime))
    return str(Path(home) / "plugin-data" / "system-one-guard" / "jev-flow.jsonl")


def load(path: str) -> list[dict]:
    rows = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def noul(row: dict, key: str) -> float | None:
    block = (row.get("answers") or {}).get(key) or {}
    if not isinstance(block, dict):
        return None
    value = block.get("noul")
    return value if isinstance(value, (int, float)) else None


def tool_of(row: dict) -> str:
    """Tool name out of the logged state preview, so no extra field is needed."""
    head = row.get("state_head") or ""
    marker = '"tool":'
    if marker not in head:
        return "?"
    return head.split(marker, 1)[1].split('"')[1] if '"' in head.split(marker, 1)[1] else "?"


def risk_outcome(row: dict) -> str:
    """Which ask this call produced. Both thresholds return `block`; they only change how
    loudly the guard says so. `urgent` is the higher tier."""
    risk = noul(row, "risk")
    limits = row.get("thresholds") or {}
    if risk is None:
        return "?"
    if risk >= (limits.get("urgent_at") or limits.get("block_at") or 1):
        return "urgent"
    if risk >= (limits.get("approve_at") or 1):
        return "approve"
    return "pass"


def summarize(rows: list[dict]) -> dict:
    by_event: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_event[row.get("event") or "?"].append(row)

    out: dict = {"calls": len(rows), "events": {}, "turns": []}
    for event, group in sorted(by_event.items()):
        ms = [r["ms"] for r in group if isinstance(r.get("ms"), (int, float))]
        out["events"][event] = {
            "n": len(group),
            "failed": sum(1 for r in group if not r.get("ok")),
            "ms_p50": round(statistics.median(ms), 1) if ms else None,
            "ms_p95": round(sorted(ms)[int(len(ms) * 0.95) - 1], 1) if len(ms) > 1 else (ms[0] if ms else None),
            "ms_total_s": round(sum(ms) / 1000, 1) if ms else None,
        }
        if event == "pre_tool_call":
            out["risk"] = Counter(risk_outcome(r) for r in group)
            risks = [noul(r, "risk") for r in group]
            risks = [r for r in risks if r is not None]
            out["risk_mean"] = round(statistics.mean(risks), 3) if risks else None
            out["risk_max"] = max(risks) if risks else None
            by_tool: dict[str, list[float]] = defaultdict(list)
            for r in group:
                value = noul(r, "risk")
                if value is not None:
                    by_tool[tool_of(r)].append(value)
            out["risk_by_tool"] = {t: {"n": len(v), "mean": round(statistics.mean(v), 3),
                                       "max": round(max(v), 2)}
                                   for t, v in sorted(by_tool.items(), key=lambda kv: -len(kv[1]))}
        if event == "pre_verify":
            done = [noul(r, "done") for r in group]
            done = [d for d in done if d is not None]
            out["done_p"] = {"n": len(done),
                             "p50": round(statistics.median(done), 3) if done else None,
                             "min": round(min(done), 3) if done else None}
            out["stop_armed"] = sum(1 for r in group if (noul(r, "stop") or 0) >= 0.7)

    sessions: dict[str, dict] = {}
    for row in rows:
        sid = row.get("session") or "?"
        turn = sessions.setdefault(sid, {"gates": Counter(), "calls": 0, "ms": 0.0,
                                         "done": [], "first": row.get("ts")})
        turn["calls"] += 1
        turn["ms"] += row.get("ms") or 0
        if row.get("event") == "pre_tool_call":
            turn["gates"][risk_outcome(row)] += 1
        elif row.get("event") == "pre_verify":
            value = noul(row, "done")
            if value is not None:
                turn["done"].append(round(value, 2))
    out["turns"] = [dict(session=k, **v) for k, v in sessions.items()]
    return out


def db_blocks(log_rows: list[dict], db_path: str) -> dict | None:
    """Did a Jev block actually reach the conversation? A block returns before the Jev call
    logs, so it only exists in the session DB."""
    import sqlite3

    if not any(r.get("event") == "pre_tool_call" for r in log_rows):
        return None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    counts = {
        "risk_urgent": con.execute(
            "select count(*) from messages where content like '%as destructive or irreversible. Stopped.%'"
        ).fetchone()[0],
        "risk_block": con.execute(
            "select count(*) from messages where content like '%so it did not run.%'"
        ).fetchone()[0],
        "legacy_blocks": con.execute(
            "select count(*) from messages where content like '%rewrite as a safer, reversible step%'"
        ).fetchone()[0],
        "done_nudge": con.execute(
            "select count(*) from messages where content like '%Finish%before stopping%'"
        ).fetchone()[0],
        "human_gate": con.execute(
            "select count(*) from messages where content like '%may only run with your approval%'"
        ).fetchone()[0],
    }
    con.close()
    return counts


def report(path: str, db_path: str | None = None) -> int:
    rows = load(path)
    if not rows:
        print(f"no calls in {path}")
        return 1
    s = summarize(rows)
    print(f"log: {path}")
    print(f"calls: {s['calls']}   sessions: {len(s['turns'])}")
    print("\nlatency (ms, Jev call round-trip)")
    for event, e in s["events"].items():
        print(f"  {event:14} n={e['n']:4} fail={e['failed']}  p50={e['ms_p50']}  "
              f"p95={e['ms_p95']}  total={e['ms_total_s']}s")
    if "risk" in s:
        print(f"\ngates  {dict(s['risk'])}  mean risk={s['risk_mean']}  max={s['risk_max']}")
        for tool, stats in s["risk_by_tool"].items():
            print(f"  {tool:16} n={stats['n']:5} mean={stats['mean']} max={stats['max']}")
    if "done_p" in s:
        print(f"\ndone   p(complete) n={s['done_p']['n']} p50={s['done_p']['p50']} "
              f"min={s['done_p']['min']}  human-gate armed {s['stop_armed']}x")

    per_turn = sorted(s["turns"], key=lambda t: t["first"] or 0)
    print("\nturns (worst Jev time first)")
    for turn in sorted(per_turn, key=lambda t: -t["ms"])[:15]:
        gates = "+".join(f"{k}:{v}" for k, v in turn["gates"].items()) or "-"
        done = ",".join(str(d) for d in turn["done"]) or "-"
        stamp = time.strftime("%m-%d %H:%M", time.localtime(turn["first"])) if turn["first"] else "?"
        print(f"  {stamp}  calls={turn['calls']:3} jev={turn['ms'] / 1000:7.1f}s "
              f"gates={gates:18} p(complete)={done}")

    if db_path:
        blocks = db_blocks(rows, db_path)
        if blocks:
            print("\nenforcement seen in state.db")
            for key, count in blocks.items():
                print(f"  {key:14} {count}")
    return 0


def default_db_path() -> str:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return str(Path(home) / "state.db")


def self_test() -> int:
    rows = [
        {"event": "pre_tool_call", "ok": True, "ms": 800.0, "session": "a",
         "state_head": '{"tool": "terminal", "input": {"command": "rm -rf /tmp/x"}}',
         "answers": {"risk": {"noul": 0.8}}, "thresholds": {"approve_at": 0.7, "urgent_at": 0.97}},
        {"event": "pre_tool_call", "ok": True, "ms": 700.0, "session": "a",
         "state_head": '{"tool": "patch", "input": {"new_string": "x"}}',
         "answers": {"risk": {"noul": 0.1}}, "thresholds": {"approve_at": 0.7, "urgent_at": 0.97}},
        {"event": "pre_verify", "ok": True, "ms": 900.0, "session": "a",
         "answers": {"done": {"noul": 0.4}, "stop": {"noul": 0.95}}},
        {"event": "pre_verify", "ok": False, "ms": 100.0, "session": "a",
         "answers": {}, "error": "TimeoutError: The read operation timed out"},
    ]
    s = summarize(rows)
    assert s["risk"]["approve"] == 1 and s["risk"]["pass"] == 1
    assert s["risk_by_tool"]["terminal"]["max"] == 0.8
    assert s["done_p"]["p50"] == 0.4 and s["stop_armed"] == 1
    assert s["events"]["pre_verify"]["failed"] == 1
    assert s["turns"][0]["calls"] == 4 and s["turns"][0]["gates"]["approve"] == 1
    print("self-test OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default=default_log_path())
    parser.add_argument("--db", default=default_db_path(),
                        help="Hermes state.db for the enforcement count; '' to skip")
    parser.add_argument("--json", action="store_true", help="dump the summary as JSON")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.json:
        print(json.dumps(summarize(load(args.log)), default=str, indent=2))
        return 0
    return report(args.log, args.db if args.db and Path(args.db).exists() else None)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
