#!/usr/bin/env python3
"""A/B benchmark comparison tool for the FPGA GPU simulator.

Compares benchmark results between a baseline git ref and the current build,
stores results in SQLite for trend analysis, and prints a formatted delta table.

Usage:
    python3 tools/bench_compare.py --baseline HEAD~1
    python3 tools/bench_compare.py --baseline v0.3.0 --bench matmul --bench gemv
    python3 tools/bench_compare.py --history matmul
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "tools" / ".bench_history.db"
DEFAULT_DRAMSIM3_INI = REPO_ROOT / "sim" / "configs" / "dram" / "DDR3_4Gb_x16_800.ini"

BENCHMARK_SPECS = [
    ("matmul", "tests/matmul/matmul_bench"),
    ("gemv", "tests/gemv/gemv_bench"),
    ("fused_linear_activation", "tests/fused_linear_activation/fused_linear_activation_bench"),
    ("softmax_row", "tests/softmax_row/softmax_row_bench"),
    ("embedding_gather", "tests/embedding_gather/embedding_gather_bench"),
    ("layernorm_lite", "tests/layernorm_lite/layernorm_lite_bench"),
]

# Metrics where higher is better (everything else: lower is better)
HIGHER_IS_BETTER = {
    "ipc",
    "cache_hits",
    "load_hits",
    "store_hits",
    "coalesced_requests",
    "macs_per_cycle",
    "outputs_per_cycle",
    "elements_per_cycle",
    "bytes_per_cycle",
}

# Key metrics shown in the summary table (in order)
SUMMARY_METRICS = [
    "total_cycles",
    "total_instructions_issued",
    "ipc",
    "scheduler_idle_cycles",
    "scheduler_frontend_stall_cycles",
    "scheduler_stall_backend_cycles",
    "cache_hits",
    "cache_misses",
    "coalesced_requests",
    "serialized_requests",
    "external_memory_reads",
    "external_memory_writes",
    "alu_busy_cycles",
    "mul_busy_cycles",
    "ldst_busy_cycles",
    "tlookup_busy_cycles",
    "fetch_skip_count",
    "branch_mispredictions",
    "writeback_conflicts",
]

# Derived metrics appended per-benchmark (shown if present)
DERIVED_METRICS = [
    "macs_per_cycle",
    "outputs_per_cycle",
    "elements_per_cycle",
    "bytes_per_cycle",
]


def run(cmd, **kwargs):
    """Run a subprocess command, returning stdout."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(str(c) for c in cmd)}", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise SystemExit(1)
    return result.stdout.strip()


def git_resolve_sha(ref):
    return run(["git", "-C", str(REPO_ROOT), "rev-parse", ref])


def git_current_sha():
    return run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])


def git_current_branch():
    try:
        return run(["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"])
    except SystemExit:
        return None


def git_short_sha(sha):
    return sha[:7]


def git_has_changes():
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def cmake_build(source_dir, build_dir):
    """Configure and build the project."""
    build_dir = Path(build_dir)
    if not (build_dir / "CMakeCache.txt").exists():
        print(f"  Configuring {build_dir} ...")
        run(["cmake", "-S", str(source_dir), "-B", str(build_dir),
             "-DCMAKE_BUILD_TYPE=Release"], cwd=str(source_dir))
    print(f"  Building {build_dir} ...")
    run(["cmake", "--build", str(build_dir), "-j"], cwd=str(source_dir))


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

def parse_text_output(name, text):
    """Parse legacy text output (pre --json) into a metrics dict with key fields."""
    import re
    metrics = {"benchmark": name}
    patterns = {
        "total_cycles": r"cycles:\s+(\d+)",
        "total_instructions_issued": r"issued instructions:\s+(\d+)",
        "fetch_skip_count": r"fetch skips:\s+(\d+)",
        "fetch_skip_backpressure": r"backpressure=(\d+)",
        "fetch_skip_all_full": r"all_full=(\d+)",
        "scheduler_idle_cycles": r"scheduler idle:\s+(\d+)",
        "scheduler_frontend_stall_cycles": r"frontend=(\d+)",
        "scheduler_stall_backend_cycles": r"backend=(\d+)",
        "cache_hits": r"cache hits/misses:\s+(\d+)",
        "cache_misses": r"cache hits/misses:\s+\d+/(\d+)",
        "coalesced_requests": r"coalesced/serialized memory ops:\s+(\d+)",
        "serialized_requests": r"coalesced/serialized memory ops:\s+\d+/(\d+)",
        "external_memory_reads": r"external reads/writes:\s+(\d+)",
        "external_memory_writes": r"external reads/writes:\s+\d+/(\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            metrics[key] = int(m.group(1))

    # Derived metrics from text
    float_patterns = {
        "macs_per_cycle": r"MACs/cycle:\s+([0-9.e+\-]+)",
        "outputs_per_cycle": r"outputs/cycle:\s+([0-9.e+\-]+)",
        "elements_per_cycle": r"elements/cycle:\s+([0-9.e+\-]+)",
        "bytes_per_cycle": r"copied bytes/cycle:\s+([0-9.e+\-]+)",
    }
    for key, pat in float_patterns.items():
        m = re.search(pat, text)
        if m:
            metrics[key] = float(m.group(1))

    # Compute IPC if we have cycles and instructions
    if "total_cycles" in metrics and "total_instructions_issued" in metrics and metrics["total_cycles"] > 0:
        metrics["ipc"] = metrics["total_instructions_issued"] / metrics["total_cycles"]

    return metrics if "total_cycles" in metrics else None


def run_benchmark(build_dir, name, rel_path, backend_args):
    """Run a single benchmark with --json and return parsed metrics dict, or None.
    Falls back to parsing text output if --json is not supported.
    backend_args is a list of CLI flags (e.g. memory-backend selection) passed
    to the benchmark binary."""
    exe = Path(build_dir) / rel_path
    if not exe.exists():
        print(f"  {name}: SKIP (executable not found)", file=sys.stderr)
        return None

    # Try --json first
    result = subprocess.run(
        [str(exe), "--json", *backend_args],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode == 0:
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            pass

    # Fallback: run without --json and parse text output
    result = subprocess.run(
        [str(exe), *backend_args],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  {name}: FAIL (exit code {result.returncode})", file=sys.stderr)
        if result.stderr:
            for line in result.stderr.strip().splitlines()[:5]:
                print(f"    {line}", file=sys.stderr)
        return None

    parsed = parse_text_output(name, result.stdout)
    if parsed is None:
        print(f"  {name}: FAIL (could not parse text output)", file=sys.stderr)
    else:
        print(f"  {name}: OK (text fallback)", file=sys.stderr)
    return parsed


def run_all_benchmarks(build_dir, selected, backend_args):
    """Run benchmarks and return {name: metrics_dict}."""
    results = {}
    for name, rel_path in BENCHMARK_SPECS:
        if selected and name not in selected:
            continue
        metrics = run_benchmark(build_dir, name, rel_path, backend_args)
        if metrics is not None:
            results[name] = metrics
    return results


def build_backend_args(args):
    """Return CLI args to pass to each benchmark for backend selection.
    Defaults to DRAMSim3 with the DE-10 Nano DDR3-800 .ini; --fixed-memory
    falls back to the FixedLatencyMemory backend."""
    if args.fixed_memory:
        return ["--memory-backend=fixed"]
    return [
        "--memory-backend=dramsim3",
        f"--dramsim3-config-path={DEFAULT_DRAMSIM3_INI}",
    ]


# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------

def init_db():
    db = sqlite3.connect(str(DB_PATH))
    db.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            git_sha TEXT NOT NULL,
            branch TEXT,
            dirty INTEGER NOT NULL DEFAULT 0,
            timestamp TEXT NOT NULL,
            benchmark TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            UNIQUE(git_sha, benchmark, dirty)
        )
    """)
    db.commit()
    return db


def store_run(db, git_sha, branch, dirty, benchmark, metrics):
    timestamp = datetime.now(timezone.utc).isoformat()
    db.execute(
        """INSERT OR REPLACE INTO runs (git_sha, branch, dirty, timestamp, benchmark, metrics_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (git_sha, branch, int(dirty), timestamp, benchmark, json.dumps(metrics)),
    )
    db.commit()


def load_run(db, git_sha, benchmark):
    row = db.execute(
        "SELECT metrics_json FROM runs WHERE git_sha = ? AND benchmark = ? ORDER BY id DESC LIMIT 1",
        (git_sha, benchmark),
    ).fetchone()
    if row:
        return json.loads(row[0])
    return None


def load_history(db, benchmark, limit=20):
    rows = db.execute(
        """SELECT git_sha, branch, dirty, timestamp, metrics_json
           FROM runs WHERE benchmark = ? ORDER BY id DESC LIMIT ?""",
        (benchmark, limit),
    ).fetchall()
    return rows


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def flatten_metrics(metrics):
    """Flatten a metrics dict into {key: numeric_value} for scalar fields only."""
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            flat[k] = v
    return flat


def compute_deltas(baseline, current):
    """Return list of (metric, baseline_val, current_val, delta, pct_change)."""
    b_flat = flatten_metrics(baseline)
    c_flat = flatten_metrics(current)
    all_keys = list(dict.fromkeys(
        [k for k in SUMMARY_METRICS if k in b_flat or k in c_flat] +
        [k for k in DERIVED_METRICS if k in b_flat or k in c_flat]
    ))
    rows = []
    for k in all_keys:
        bv = b_flat.get(k)
        cv = c_flat.get(k)
        if bv is None or cv is None:
            continue
        delta = cv - bv
        pct = (delta / bv * 100.0) if bv != 0 else 0.0
        rows.append((k, bv, cv, delta, pct))
    return rows


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"


def use_color():
    return sys.stdout.isatty()


def colorize(text, color):
    if use_color():
        return f"{color}{text}{RESET}"
    return text


def fmt_val(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def is_improvement(metric, delta):
    if metric in HIGHER_IS_BETTER:
        return delta > 0
    return delta < 0


def print_comparison_table(bench_name, deltas, threshold):
    """Print a formatted comparison table for one benchmark."""
    if not deltas:
        return

    # Column widths
    mw = max(len(d[0]) for d in deltas)
    bw = max(len(fmt_val(d[1])) for d in deltas)
    cw = max(len(fmt_val(d[2])) for d in deltas)

    header = f"  {'Metric':<{mw}}  {'Baseline':>{bw}}  {'Current':>{cw}}  {'Delta':>12}  {'%Change':>9}"
    sep = "  " + "-" * (mw + bw + cw + 30)

    print(f"\n{colorize(bench_name, BOLD)}")
    print(header)
    print(sep)

    for metric, bv, cv, delta, pct in deltas:
        delta_str = fmt_val(delta)
        if delta > 0:
            delta_str = "+" + delta_str
        pct_str = f"{pct:+.1f}%"

        improved = is_improvement(metric, delta)
        abs_pct = abs(pct)

        if abs_pct < threshold:
            indicator = " "
            color = DIM
        elif improved:
            indicator = colorize(" ▼", GREEN) if metric not in HIGHER_IS_BETTER else colorize(" ▲", GREEN)
            color = GREEN
        else:
            indicator = colorize(" ▲", RED) if metric not in HIGHER_IS_BETTER else colorize(" ▼", RED)
            color = RED

        line = f"  {metric:<{mw}}  {fmt_val(bv):>{bw}}  {fmt_val(cv):>{cw}}  {delta_str:>12}  {colorize(pct_str, color):>9}{indicator}"
        print(line)


def print_history_table(benchmark, rows):
    """Print a history table for a benchmark."""
    if not rows:
        print(f"No history for '{benchmark}'")
        return

    print(f"\n{colorize(f'History: {benchmark}', BOLD)}")
    print(f"  {'SHA':7}  {'Branch':15}  {'Date':19}  {'Cycles':>10}  {'IPC':>8}  {'Dirty':>5}")
    print("  " + "-" * 75)

    for git_sha, branch, dirty, timestamp, metrics_json in rows:
        m = json.loads(metrics_json)
        cycles = m.get("total_cycles", "?")
        ipc = m.get("ipc")
        ipc_str = f"{ipc:.4f}" if ipc is not None else "?"
        dirty_str = " *" if dirty else ""
        date_str = timestamp[:19] if timestamp else "?"
        branch_str = (branch or "?")[:15]
        print(f"  {git_sha[:7]}  {branch_str:15}  {date_str:19}  {str(cycles):>10}  {ipc_str:>8}{dirty_str}")


# ---------------------------------------------------------------------------
# Worktree management
# ---------------------------------------------------------------------------

def create_worktree(sha):
    """Create a temporary git worktree for the given SHA. Returns worktree path."""
    tmpdir = tempfile.mkdtemp(prefix="bench_baseline_")
    wt_path = os.path.join(tmpdir, "worktree")
    run(["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", wt_path, sha])
    return wt_path, tmpdir


def cleanup_worktree(wt_path, tmpdir):
    """Remove worktree and temp dir."""
    try:
        run(["git", "-C", str(REPO_ROOT), "worktree", "remove", "--force", wt_path])
    except SystemExit:
        pass
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main commands
# ---------------------------------------------------------------------------

def cmd_compare(args):
    baseline_sha = git_resolve_sha(args.baseline)
    current_sha = git_current_sha()
    current_branch = git_current_branch()
    dirty = git_has_changes()
    selected = set(args.bench) if args.bench else set()
    backend_args = build_backend_args(args)

    print(f"Baseline: {git_short_sha(baseline_sha)} ({args.baseline})")
    label = git_short_sha(current_sha)
    if dirty:
        label += " (dirty)"
    print(f"Current:  {label}" + (f" [{current_branch}]" if current_branch else ""))
    backend_label = "fixed" if args.fixed_memory else "dramsim3"
    print(f"Backend:  {backend_label}")

    # --- Baseline: build in worktree ---
    db = init_db() if not args.no_store else None

    # Check if we already have cached baseline results
    cached_baseline = {}
    if db:
        for name, _ in BENCHMARK_SPECS:
            if selected and name not in selected:
                continue
            cached = load_run(db, baseline_sha, name)
            if cached:
                cached_baseline[name] = cached

    need_baseline_build = False
    for name, _ in BENCHMARK_SPECS:
        if selected and name not in selected:
            continue
        if name not in cached_baseline:
            need_baseline_build = True
            break

    if need_baseline_build:
        print(f"\nBuilding baseline ({git_short_sha(baseline_sha)}) ...")
        wt_path, tmpdir = create_worktree(baseline_sha)
        try:
            wt_build = os.path.join(wt_path, "build")
            cmake_build(wt_path, wt_build)
            print(f"\nRunning baseline benchmarks ...")
            baseline_results = run_all_benchmarks(wt_build, selected, backend_args)
            if db:
                for name, metrics in baseline_results.items():
                    store_run(db, baseline_sha, None, False, name, metrics)
            # Merge with any cached results
            for name, metrics in cached_baseline.items():
                if name not in baseline_results:
                    baseline_results[name] = metrics
        finally:
            cleanup_worktree(wt_path, tmpdir)
    else:
        print(f"\nBaseline results loaded from cache.")
        baseline_results = cached_baseline

    # --- Current: use existing build or rebuild ---
    current_build = Path(args.build_dir).resolve()
    if args.rebuild or not (current_build / "CMakeCache.txt").exists():
        print(f"\nBuilding current ({label}) ...")
        cmake_build(str(REPO_ROOT), str(current_build))
    else:
        print(f"\nUsing existing build at {current_build}")

    print(f"\nRunning current benchmarks ...")
    current_results = run_all_benchmarks(str(current_build), selected, backend_args)

    if db:
        for name, metrics in current_results.items():
            store_run(db, current_sha, current_branch, dirty, name, metrics)

    # --- Compare ---
    if args.json_output:
        output = {}
        for name in sorted(set(baseline_results) & set(current_results)):
            output[name] = {
                "baseline": baseline_results[name],
                "current": current_results[name],
                "deltas": {
                    row[0]: {"baseline": row[1], "current": row[2], "delta": row[3], "pct": row[4]}
                    for row in compute_deltas(baseline_results[name], current_results[name])
                },
            }
        print(json.dumps(output, indent=2))
        return

    print(f"\n{'=' * 70}")
    print(f"  Comparison: {git_short_sha(baseline_sha)} -> {label}")
    print(f"{'=' * 70}")

    any_printed = False
    for name, _ in BENCHMARK_SPECS:
        if name not in baseline_results or name not in current_results:
            continue
        deltas = compute_deltas(baseline_results[name], current_results[name])
        print_comparison_table(name, deltas, args.threshold)
        any_printed = True

    if not any_printed:
        print("\nNo benchmarks to compare.")

    if db:
        db.close()


def cmd_history(args):
    db = init_db()
    rows = load_history(db, args.history, limit=args.limit)
    print_history_table(args.history, rows)
    db.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="A/B benchmark comparison tool for the FPGA GPU simulator.",
    )
    parser.add_argument(
        "--baseline", metavar="REF",
        help="Git ref to compare against (commit SHA, tag, branch, HEAD~N)",
    )
    parser.add_argument(
        "--build-dir", default="build",
        help="Current build directory (default: build)",
    )
    parser.add_argument(
        "--bench", action="append", default=[],
        help="Run only named benchmark (repeatable)",
    )
    parser.add_argument(
        "--threshold", type=float, default=1.0,
        help="Highlight changes above this %% (default: 1.0)",
    )
    parser.add_argument(
        "--no-store", action="store_true",
        help="Skip writing results to SQLite",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild of the current tree",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Output comparison as JSON",
    )
    parser.add_argument(
        "--history", metavar="BENCH",
        help="Show historical trend for a benchmark",
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Number of history entries to show (default: 20)",
    )
    parser.add_argument(
        "--fixed-memory", action="store_true",
        help="Run benchmarks against the FixedLatencyMemory backend instead of "
             "the DRAMSim3 default. Both baseline and current are run with the "
             "same backend selection.",
    )

    args = parser.parse_args()

    if args.history:
        cmd_history(args)
    elif args.baseline:
        cmd_compare(args)
    else:
        parser.error("Either --baseline <ref> or --history <bench> is required.")


if __name__ == "__main__":
    main()
