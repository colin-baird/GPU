#!/usr/bin/env python3
"""Summarize .claude/logs/tool_timing.jsonl.

Reports per-session agent-active time, broken down by command bucket, with
correct handling of:

  * User-input wait — gaps between Stop and UserPromptSubmit excluded.
  * Parallel tool calls — tool wait is the interval *union*, not the sum,
    so overlapping calls within a turn aren't double-counted.
  * Subagents — treated as siblings of the parent. Each agent (top-level or
    subagent) is its own session_id in the log; the analysis rolls a parent
    and its descendants into a single tree report. The parent's outer
    Agent-tool wait is excluded because it just wraps the child's work.

Subagent linkage: SubagentStart/SubagentStop events are logged with their
full payload meta (see logger). If they include an explicit parent
session_id we use it; otherwise we fall back to temporal containment:
a session whose first event falls inside another session's Agent tool
interval is assumed to be that Agent call's subagent.

Usage:
  python3 .claude/scripts/summarize_tool_timing.py            # all roots
  python3 .claude/scripts/summarize_tool_timing.py <session>  # one tree
  python3 .claude/scripts/summarize_tool_timing.py --flat     # no rollup
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = os.environ.get("CLAUDE_PROJECT_DIR") or str(Path(__file__).resolve().parents[2])
LOG_FILE = Path(PROJECT_DIR) / ".claude" / "logs" / "tool_timing.jsonl"

BASH_PATTERNS = [
    (re.compile(r"\brun_workload_benchmarks\.sh\b"), "benchmarks"),
    (re.compile(r"\bbench_compare\.py\b"),           "benchmarks"),
    (re.compile(r"\bctest\b"),                       "ctest"),
    (re.compile(r"\bcmake\s+--build\b"),             "cmake-build"),
    (re.compile(r"\bcmake\s+-B\b"),                  "cmake-config"),
    (re.compile(r"\b(rg|grep|find)\b"),              "search"),
    (re.compile(r"\bgit\b"),                         "git"),
    (re.compile(r"/sim/tests/test_"),                "sim-tests"),
]


def bucket(tool_name: str, tool_input: dict) -> str:
    if tool_name != "Bash":
        return tool_name
    cmd = (tool_input or {}).get("command", "").strip()
    if not cmd:
        return "Bash"
    for pat, label in BASH_PATTERNS:
        if pat.search(cmd):
            return label
    return cmd.split()[0]


def interval_union(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by the union of (start, end) intervals."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


class SessionData:
    def __init__(self, sid: str):
        self.sid = sid
        self.first_ts: float | None = None
        self.last_ts: float | None = None
        self.call_intervals: list[tuple[float, float, str, str]] = []
        self.agent_intervals: list[tuple[float, float]] = []
        self.user_wait: float = 0.0
        self.subagent_events: list[dict] = []
        self.declared_parent: str | None = None


def parse_log() -> dict[str, SessionData]:
    if not LOG_FILE.exists():
        return {}
    raw: dict[str, list[dict]] = defaultdict(list)
    with LOG_FILE.open() as f:
        for line in f:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            raw[ev.get("session", "")].append(ev)

    sessions: dict[str, SessionData] = {}
    for sid, events in raw.items():
        sd = SessionData(sid)
        events.sort(key=lambda e: e["ts"])
        pending_by_id: dict[str, tuple[float, str, dict]] = {}
        pending_by_tool: dict[str, list[tuple[float, dict, str | None]]] = defaultdict(list)
        stop_ts: float | None = None
        for ev in events:
            ts = ev["ts"]
            sd.first_ts = ts if sd.first_ts is None else min(sd.first_ts, ts)
            sd.last_ts = ts if sd.last_ts is None else max(sd.last_ts, ts)
            kind = ev.get("kind")
            tool = ev.get("tool", "")
            tuid = ev.get("tool_use_id")
            if kind == "stop":
                if stop_ts is None:
                    stop_ts = ts
                continue
            if kind == "submit":
                if stop_ts is not None:
                    sd.user_wait += ts - stop_ts
                    stop_ts = None
                continue
            if kind in ("subagent_start", "subagent_stop"):
                sd.subagent_events.append(ev)
                meta = ev.get("meta") or {}
                parent = (meta.get("parent_session_id")
                          or meta.get("parent_session")
                          or meta.get("parentSessionId"))
                if parent and not sd.declared_parent:
                    sd.declared_parent = parent
                continue
            stop_ts = None
            if kind == "start":
                inp = ev.get("input", {})
                if tuid:
                    pending_by_id[tuid] = (ts, tool, inp)
                else:
                    pending_by_tool[tool].append((ts, inp, tuid))
                continue
            if kind == "end":
                start_ts: float | None = None
                start_tool: str | None = None
                inp: dict = {}
                if tuid and tuid in pending_by_id:
                    start_ts, start_tool, inp = pending_by_id.pop(tuid)
                elif pending_by_tool.get(tool):
                    start_ts, inp, _ = pending_by_tool[tool].pop(0)
                    start_tool = tool
                if start_ts is None:
                    continue
                b = bucket(start_tool, inp)
                sd.call_intervals.append((start_ts, ts, b, start_tool))
                if start_tool in ("Agent", "Task"):
                    sd.agent_intervals.append((start_ts, ts))
        sessions[sid] = sd
    return sessions


def infer_parent(sd: SessionData, all_sessions: dict[str, SessionData]) -> str | None:
    if sd.declared_parent and sd.declared_parent in all_sessions:
        return sd.declared_parent
    if sd.first_ts is None:
        return None
    candidates: list[tuple[float, str]] = []
    for psid, psd in all_sessions.items():
        if psid == sd.sid:
            continue
        for a_s, a_e in psd.agent_intervals:
            if a_s <= sd.first_ts and (sd.last_ts is None or sd.last_ts <= a_e):
                candidates.append((a_e - a_s, psid))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def build_tree(sessions: dict[str, SessionData]) -> tuple[dict[str, list[str]], list[str]]:
    parent_of: dict[str, str | None] = {}
    for sid, sd in sessions.items():
        parent_of[sid] = infer_parent(sd, sessions)
    children_of: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    for sid, p in parent_of.items():
        if p is None:
            roots.append(sid)
        else:
            children_of[p].append(sid)
    return children_of, roots


def collect_descendants(root: str, children_of: dict[str, list[str]]) -> list[str]:
    out = [root]
    stack = list(children_of.get(root, []))
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(children_of.get(node, []))
    return out


def report_tree(root: str, sessions: dict[str, SessionData],
                children_of: dict[str, list[str]]) -> None:
    members = collect_descendants(root, children_of)
    members = [m for m in members if m in sessions]
    first = min(sessions[m].first_ts for m in members if sessions[m].first_ts is not None)
    last  = max(sessions[m].last_ts  for m in members if sessions[m].last_ts  is not None)
    wall = last - first

    user_wait = sessions[root].user_wait
    active = max(wall - user_wait, 0.0)

    intervals: list[tuple[float, float]] = []
    bucket_sum: dict[str, float] = defaultdict(float)
    bucket_count: dict[str, int] = defaultdict(int)
    for m in members:
        sd = sessions[m]
        has_kids = bool(children_of.get(m))
        for s, e, b, tool in sd.call_intervals:
            # Drop Agent/Task wrappers on any node that has children — they
            # are delegation, not work, and would double-count the child's
            # actual activity.
            if tool in ("Agent", "Task") and has_kids:
                continue
            intervals.append((s, e))
            bucket_sum[b] += e - s
            bucket_count[b] += 1
    tool_wait = interval_union(intervals)

    head = f"session {root[:8]}"
    if len(members) > 1:
        head += f"  (+{len(members)-1} subagent{'s' if len(members)>2 else ''})"
    print(head)
    print(f"  wall clock:            {wall:7.1f}s")
    print(f"  user-input wait:       {user_wait:7.1f}s  (excluded)")
    print(f"  agent-active time:     {active:7.1f}s")
    if active > 0:
        print(f"    tool wait (union):   {tool_wait:7.1f}s  ({100*tool_wait/active:.1f}%)")
        print(f"    gen + harness:       {active-tool_wait:7.1f}s  ({100*(active-tool_wait)/active:.1f}%)")
    if bucket_sum:
        print(f"  {'bucket':<26} {'n':>4} {'sum_s':>9} {'mean_s':>8}")
        for b, total in sorted(bucket_sum.items(), key=lambda x: -x[1]):
            n = bucket_count[b]
            print(f"  {b:<26} {n:>4} {total:>9.2f} {total/n:>8.2f}")
    print()


def report_flat(sd: SessionData) -> None:
    wall = (sd.last_ts - sd.first_ts) if (sd.first_ts and sd.last_ts) else 0.0
    active = max(wall - sd.user_wait, 0.0)
    intervals = [(s, e) for s, e, _, _ in sd.call_intervals]
    tool_wait = interval_union(intervals)
    bucket_sum: dict[str, float] = defaultdict(float)
    bucket_count: dict[str, int] = defaultdict(int)
    for s, e, b, _ in sd.call_intervals:
        bucket_sum[b] += e - s
        bucket_count[b] += 1
    print(f"session {sd.sid[:8]}")
    print(f"  wall clock:            {wall:7.1f}s")
    print(f"  user-input wait:       {sd.user_wait:7.1f}s  (excluded)")
    print(f"  agent-active time:     {active:7.1f}s")
    if active > 0:
        print(f"    tool wait (union):   {tool_wait:7.1f}s  ({100*tool_wait/active:.1f}%)")
        print(f"    gen + harness:       {active-tool_wait:7.1f}s  ({100*(active-tool_wait)/active:.1f}%)")
    if bucket_sum:
        print(f"  {'bucket':<26} {'n':>4} {'sum_s':>9} {'mean_s':>8}")
        for b, total in sorted(bucket_sum.items(), key=lambda x: -x[1]):
            n = bucket_count[b]
            print(f"  {b:<26} {n:>4} {total:>9.2f} {total/n:>8.2f}")
    print()


def main() -> int:
    args = sys.argv[1:]
    flat = "--flat" in args
    args = [a for a in args if a != "--flat"]
    target = args[0] if args else None

    sessions = parse_log()
    if not sessions:
        print(f"no log at {LOG_FILE}", file=sys.stderr)
        return 1

    if flat:
        sids = [target] if target else sorted(sessions, key=lambda s: sessions[s].first_ts or 0)
        for sid in sids:
            if sid in sessions:
                report_flat(sessions[sid])
        return 0

    children_of, roots = build_tree(sessions)
    if target:
        node = target
        seen = set()
        while node in sessions:
            if node in seen:
                break
            seen.add(node)
            parent = None
            for p, kids in children_of.items():
                if node in kids:
                    parent = p
                    break
            if parent is None:
                break
            node = parent
        roots_to_report = [node] if node in sessions else []
    else:
        roots_to_report = sorted(roots, key=lambda s: sessions[s].first_ts or 0)

    for r in roots_to_report:
        report_tree(r, sessions, children_of)
    return 0


if __name__ == "__main__":
    sys.exit(main())
