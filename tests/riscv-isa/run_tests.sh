#!/bin/bash
# Run compiled RISC-V ISA tests through the gpu_sim backend router.
#
# Usage:
#   ./run_tests.sh [options]
#
# Options:
#   --backend=<name>   Execution backend (default: perf_sim)
#   --sim=<path>       Path to gpu_sim binary (default: ../../sim/build/gpu_sim)
#   --verbose          Print full simulator output for each test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Defaults
BACKEND="perf_sim"
SIM="${SCRIPT_DIR}/../../build/runner/gpu_sim"
VERBOSE=0

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --backend=*) BACKEND="${arg#--backend=}" ;;
        --sim=*)     SIM="${arg#--sim=}" ;;
        --verbose)   VERBOSE=1 ;;
        --help)
            echo "Usage: $0 [--backend=<name>] [--sim=<path>] [--verbose]"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# Validate
if [ ! -x "$SIM" ]; then
    echo "Error: gpu_sim not found at $SIM"
    echo "Build the simulator first: cd sim/build && cmake .. && make"
    exit 1
fi

if [ ! -d "$BUILD_DIR" ] || [ -z "$(ls -A "$BUILD_DIR" 2>/dev/null)" ]; then
    echo "Error: no test ELFs found in $BUILD_DIR"
    echo "Build tests first: cd tests/riscv-isa && make"
    exit 1
fi

# Run tests
PASS=0
FAIL=0
ERRORS=""

for elf in "$BUILD_DIR"/rv32u*; do
    test_name="$(basename "$elf")"

    # Run through the backend router
    output=$("$SIM" --backend="$BACKEND" "$elf" --functional-only --num-warps=1 2>&1) || true
    exit_code=$?

    if [ $VERBOSE -eq 1 ]; then
        echo "--- $test_name ---"
        echo "$output"
        echo ""
    fi

    # Check for pass: exit code 0 AND gp (x3) == 1
    if [ $exit_code -ne 0 ]; then
        FAIL=$((FAIL + 1))
        ERRORS="${ERRORS}  FAIL  ${test_name} (exit code ${exit_code})\n"
        printf "  \033[31mFAIL\033[0m  %s (exit code %d)\n" "$test_name" "$exit_code"
        continue
    fi

    # Parse register output for x3 (gp). The functional-only output format is:
    #   x3 = 0x1 (1)
    gp_val=$(echo "$output" | grep -E '^\s+x3\s*=' | head -1 | \
             sed -E 's/.*0x([0-9a-fA-F]+).*/\1/')

    if [ -z "$gp_val" ]; then
        # x3 not in output -- might be 0 (only non-zero regs are printed)
        FAIL=$((FAIL + 1))
        ERRORS="${ERRORS}  FAIL  ${test_name} (gp not set -- test did not reach RVTEST_PASS)\n"
        printf "  \033[31mFAIL\033[0m  %s (gp not set)\n" "$test_name"
        continue
    fi

    if [ "$gp_val" = "1" ]; then
        PASS=$((PASS + 1))
        printf "  \033[32mPASS\033[0m  %s\n" "$test_name"
    else
        # Decode failing test number: gp = (test_num << 1) | 1
        gp_dec=$((16#$gp_val))
        test_num=$(( (gp_dec) >> 1 ))
        FAIL=$((FAIL + 1))
        ERRORS="${ERRORS}  FAIL  ${test_name} (test case ${test_num})\n"
        printf "  \033[31mFAIL\033[0m  %s (test case %d)\n" "$test_name" "$test_num"
    fi
done

# Summary
TOTAL=$((PASS + FAIL))
echo ""
echo "================================"
printf "Results: %d/%d passed" "$PASS" "$TOTAL"
if [ $FAIL -gt 0 ]; then
    printf " (\033[31m%d failed\033[0m)" "$FAIL"
fi
echo ""
echo "Backend: $BACKEND"
echo "================================"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "Failures:"
    printf "$ERRORS"
    exit 1
fi

exit 0
