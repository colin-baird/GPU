#!/usr/bin/env bash
#
# Validate timing model output against analytical reference statistics.
#
# Usage:
#   ./tests/references/validate.sh [--filter PATTERN] [--timeout SECS]
#
# Compares the simulator's --json output against reference JSON files
# in tests/references/isa/. Reports PASS, FAIL (mismatch), or SKIP
# (simulator timeout/error) for each test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/tools/validate.py" "$@"
