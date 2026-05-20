#!/usr/bin/env python3
"""Append a tool-timing event to .claude/logs/tool_timing.jsonl.

Reads the Claude Code hook payload from stdin and writes a single JSONL
line per hook firing:

  * PreToolUse        -> kind="start"  (includes full tool_input for later
                                        classification by the summarizer)
  * PostToolUse       -> kind="end"
  * Stop              -> kind="stop"
  * UserPromptSubmit  -> kind="submit"
  * SubagentStart     -> kind="subagent_start" (captures parent linkage
                                                via meta dump)
  * SubagentStop      -> kind="subagent_stop"

Exits 0 unconditionally and writes nothing to stdout/stderr so the hook
stays invisible to the model.
"""
import json
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = os.environ.get("CLAUDE_PROJECT_DIR") or str(Path(__file__).resolve().parents[2])
LOG_FILE = Path(PROJECT_DIR) / ".claude" / "logs" / "tool_timing.jsonl"


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    event = payload.get("hook_event_name", "")
    kind = {
        "PreToolUse": "start",
        "PostToolUse": "end",
        "Stop": "stop",
        "UserPromptSubmit": "submit",
        "SubagentStart": "subagent_start",
        "SubagentStop": "subagent_stop",
    }.get(event)
    if kind is None:
        return 0

    record = {
        "ts": time.time(),
        "session": payload.get("session_id", ""),
        "tool": payload.get("tool_name", ""),
        "kind": kind,
    }
    if "transcript_path" in payload:
        record["transcript"] = payload["transcript_path"]
    if "tool_use_id" in payload:
        record["tool_use_id"] = payload["tool_use_id"]
    if kind == "start":
        record["input"] = payload.get("tool_input", {})
    if kind in ("subagent_start", "subagent_stop"):
        # Dump unknown payload fields so subagent->parent linkage can be
        # discovered empirically. Inspect with:
        #   jq 'select(.kind | startswith("subagent"))' .claude/logs/tool_timing.jsonl
        known = {
            "hook_event_name", "session_id", "tool_name",
            "transcript_path", "cwd", "tool_input", "tool_response",
            "tool_use_id",
        }
        extra = {k: v for k, v in payload.items() if k not in known}
        if extra:
            record["meta"] = extra

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
