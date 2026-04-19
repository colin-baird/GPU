# Benchmark A/B Comparison Tool

## Context
When evaluating architectural changes to the FPGA GPU simulator, there's no way to quantify the performance delta across benchmarks. Currently, benchmarks produce human-readable text and there's no tooling to compare results between commits. This tool automates the build-run-compare cycle and stores history in SQLite for trend analysis.

## Design

### Part 1: Add `--json` output to benchmark harnesses

Each of the 6 benchmarks has a custom `print_summary()` that outputs text. None call `stats.report_json()`. We'll add a `--json` flag to each benchmark that outputs a single JSON object containing:
- All fields from `Stats::report_json()` (55+ metrics)
- Benchmark-specific derived metrics (e.g., `macs_per_cycle` for matmul)

**Files to modify:**
- `tests/matmul/matmul_bench.cpp`
- `tests/gemv/gemv_bench.cpp`
- `tests/fused_linear_activation/fused_linear_activation_bench.cpp`
- `tests/softmax_row/softmax_row_bench.cpp`
- `tests/embedding_gather/embedding_gather_bench.cpp`
- `tests/layernorm_lite/layernorm_lite_bench.cpp`

Changes per file:
1. Add `bool json_output = false;` to `Options` struct
2. Parse `--json` in `parse_options()`
3. After verification passes, if `json_output`: call `stats.report_json()` to a `std::ostringstream`, parse-inject the derived metrics before the closing `}`, print to stdout. Alternatively, modify `report_json` to accept extra key-value pairs — but that changes the shared API. Simpler: output a wrapper JSON that embeds the stats plus derived fields.

**Chosen approach:** Each benchmark outputs a JSON object where:
- Top-level has `"benchmark": "<name>"` and `"derived": { ... }` with benchmark-specific metrics
- `"stats": { ... }` contains the full `Stats::report_json()` output

This keeps the C++ changes minimal — just stringify stats JSON and wrap it.

Actually, even simpler: modify `Stats::report_json()` to NOT print the closing `}`, so callers can append fields. But that's fragile. 

**Simplest approach:** Output stats JSON to a stringstream, then in the benchmark, output a new JSON object that includes the stats fields plus derived fields by writing the stats JSON sans closing brace, appending derived fields, then closing. This is a small string manipulation.

### Part 2: Python comparison tool — `tools/bench_compare.py`

**Workflow:**
```
./tools/bench_compare.py --baseline HEAD~1
./tools/bench_compare.py --baseline v0.3.0
./tools/bench_compare.py --baseline abc123f --bench matmul --bench gemv
```

**Steps the tool performs:**
1. Resolve `--baseline` ref to a git SHA
2. Create a temporary git worktree for the baseline
3. Build the baseline in the worktree (`cmake --build`)
4. Run all (or selected) benchmarks with `--json` in the worktree, capture JSON
5. Run same benchmarks with `--json` against the current build dir
6. Parse both sets of JSON results
7. Compute deltas and percentage changes for all scalar metrics
8. Print a formatted comparison table, color-coded (green=improvement on cycles/stalls, red=regression)
9. Store both runs in SQLite with git SHA, timestamp, branch, benchmark name, and full JSON blob
10. Clean up worktree

**Key metrics to highlight in summary table (subset of 55+):**
- `total_cycles` (lower is better)
- `total_instructions_issued`
- IPC (higher is better)
- `cache_hits`, `cache_misses`, hit rate
- `scheduler_idle_cycles`
- Per-unit utilization
- Benchmark-specific derived metrics

**Flags:**
- `--baseline <git-ref>` — required, the commit to compare against
- `--build-dir <path>` — current build directory (default: `build`)
- `--bench <name>` — filter to specific benchmarks (repeatable)
- `--threshold <pct>` — only highlight changes above this % (default: 1.0)
- `--no-store` — skip writing to SQLite
- `--history <benchmark>` — show historical trend for a benchmark instead of comparing
- `--rebuild` — force rebuild of current tree too (default: assume current build is fresh)
- `--json` — output comparison as JSON instead of table

**SQLite schema:**
```sql
CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    git_sha TEXT NOT NULL,
    branch TEXT,
    timestamp TEXT NOT NULL,
    benchmark TEXT NOT NULL,
    metrics_json TEXT NOT NULL,  -- full JSON blob
    UNIQUE(git_sha, benchmark)
);
```

**Location:** `tools/bench_compare.py` (single file, stdlib only — sqlite3, json, subprocess, argparse)

DB stored at: `tools/.bench_history.db` (gitignored)

### Part 3: Update `.gitignore`

Add `tools/.bench_history.db` to `.gitignore`.

## File summary

| File | Action |
|------|--------|
| `tests/matmul/matmul_bench.cpp` | Add `--json` flag + JSON output |
| `tests/gemv/gemv_bench.cpp` | Add `--json` flag + JSON output |
| `tests/fused_linear_activation/fused_linear_activation_bench.cpp` | Add `--json` flag + JSON output |
| `tests/softmax_row/softmax_row_bench.cpp` | Add `--json` flag + JSON output |
| `tests/embedding_gather/embedding_gather_bench.cpp` | Add `--json` flag + JSON output |
| `tests/layernorm_lite/layernorm_lite_bench.cpp` | Add `--json` flag + JSON output |
| `tools/bench_compare.py` | New file — comparison tool |
| `.gitignore` | Add `tools/.bench_history.db` |

## Verification

1. Build the project: `cmake --build build`
2. Run a single benchmark with `--json` and verify valid JSON: `./build/tests/matmul/matmul_bench --json | python3 -m json.tool`
3. Run the comparison tool against the previous commit: `python3 tools/bench_compare.py --baseline HEAD~1`
4. Verify SQLite DB was created and populated: `sqlite3 tools/.bench_history.db "SELECT git_sha, benchmark, timestamp FROM runs"`
5. Run `--history` to verify trend output: `python3 tools/bench_compare.py --history matmul`
