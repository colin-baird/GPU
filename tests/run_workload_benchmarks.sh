#!/usr/bin/env bash

set -euo pipefail

build_dir="build"
declare -a selected_benchmarks=()
declare -a benchmark_args=()

usage() {
    cat <<'EOF'
Usage: ./tests/run_workload_benchmarks.sh [--build-dir <dir>] [--bench <name>]... [-- <benchmark args>...]

Runs the canonical workload benchmark suite and emits structured RESULT lines
for the validation agent to summarize.

Options:
  --build-dir <dir>   Build directory containing benchmark executables. Default: build
  --bench <name>      Run only the named benchmark. May be repeated.
  --help              Show this help.

Any arguments after `--` are passed through to each benchmark executable.

Structured output:
  RESULT name=<bench> status=<pass|fail|missing> cycles=<N|na> issued_instructions=<N|na> ipc=<V|na>
  BEGIN_RAW name=<bench>
  ... raw benchmark stdout/stderr ...
  END_RAW name=<bench>
  SUMMARY total=<N> passed=<N> failed=<N>
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            build_dir="$2"
            shift 2
            ;;
        --bench)
            selected_benchmarks+=("$2")
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        --)
            shift
            benchmark_args=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

benchmark_specs=(
    "matmul:tests/matmul/matmul_bench"
    "gemv:tests/gemv/gemv_bench"
    "fused_linear_activation:tests/fused_linear_activation/fused_linear_activation_bench"
    "softmax_row:tests/softmax_row/softmax_row_bench"
    "embedding_gather:tests/embedding_gather/embedding_gather_bench"
    "layernorm_lite:tests/layernorm_lite/layernorm_lite_bench"
)

is_selected() {
    local name="$1"
    if [[ ${#selected_benchmarks[@]} -eq 0 ]]; then
        return 0
    fi

    local selected
    for selected in "${selected_benchmarks[@]}"; do
        if [[ "$selected" == "$name" ]]; then
            return 0
        fi
    done
    return 1
}

parse_metric() {
    local output="$1"
    local label="$2"
    printf '%s\n' "$output" | awk -F': ' -v key="$label" '$1 == "  " key { print $2; exit }'
}

print_result() {
    local name="$1"
    local status="$2"
    local cycles="$3"
    local issued="$4"
    local ipc="$5"
    echo "RESULT name=${name} status=${status} cycles=${cycles} issued_instructions=${issued} ipc=${ipc}"
}

total=0
passed=0
failed=0

for spec in "${benchmark_specs[@]}"; do
    name="${spec%%:*}"
    rel_path="${spec#*:}"

    if ! is_selected "$name"; then
        continue
    fi

    total=$((total + 1))
    exe_path="${build_dir%/}/${rel_path}"

    if [[ ! -x "$exe_path" ]]; then
        print_result "$name" "missing" "na" "na" "na"
        echo "BEGIN_RAW name=${name}"
        echo "missing benchmark executable: ${exe_path}"
        echo "END_RAW name=${name}"
        failed=$((failed + 1))
        continue
    fi

    if output="$("$exe_path" "${benchmark_args[@]}" 2>&1)"; then
        cycles="$(parse_metric "$output" "cycles")"
        issued="$(parse_metric "$output" "issued instructions")"
        if [[ -n "$cycles" && -n "$issued" ]]; then
            ipc="$(awk -v issued="$issued" -v cycles="$cycles" 'BEGIN { if (cycles > 0) printf "%.6f", issued / cycles; else print "na" }')"
        else
            cycles="na"
            issued="na"
            ipc="na"
        fi

        print_result "$name" "pass" "$cycles" "$issued" "$ipc"
        echo "BEGIN_RAW name=${name}"
        printf '%s\n' "$output"
        echo "END_RAW name=${name}"
        passed=$((passed + 1))
        continue
    fi

    print_result "$name" "fail" "na" "na" "na"
    echo "BEGIN_RAW name=${name}"
    printf '%s\n' "$output"
    echo "END_RAW name=${name}"
    failed=$((failed + 1))
done

echo "SUMMARY total=${total} passed=${passed} failed=${failed}"

if [[ $failed -ne 0 ]]; then
    exit 1
fi
