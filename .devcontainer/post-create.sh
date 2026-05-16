#!/usr/bin/env bash
# Container post-create: fix the statusline path, then build the simulator.
set -euo pipefail

settings=/home/dev/.claude/settings.json

# The container-local settings.json is seeded (cp -n) from the host's
# ~/.claude/settings.json, which was authored on macOS with an absolute
# /Users/<user>/.claude/... statusline path that does not exist in the
# container. Rewrite it to the in-container config path. Idempotent: on a
# rebuild the path is already correct and this is a no-op.
if [ -f "$settings" ] && jq -e . "$settings" >/dev/null 2>&1; then
  fixed=$(jq '.statusLine.command = "sh /home/dev/.claude/statusline-command.sh"' "$settings")
  # In-place truncate-and-write: settings.json is a single-file bind mount,
  # so a temp-file rename (sed -i, mv) onto it would fail. Shell '>' truncates
  # the existing inode instead.
  printf '%s\n' "$fixed" > "$settings"
fi

# Build the simulator.
rm -rf build
cmake -B build
cmake --build build -j"$(nproc)"
