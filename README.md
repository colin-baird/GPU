# FPGA GPU Accelerator

This repository contains a from-scratch GPU design aimed at FPGA implementation, plus a C++ performance simulator and runner used to execute and benchmark kernels today. The target machine is a single streaming multiprocessor (SM) that executes 32-thread SIMT warps using RV32IM plus custom instructions for quantized LLM inference.

The architecture is loosely inspired by an NVIDIA SM, but constrained for FPGA feasibility: integer-only arithmetic, a single SM, no shared memory, and no divergence handling.

## Current Scope

- Performance simulator with functional and timing execution modes
- Backend runner that loads ELF or raw binaries and dispatches them through the simulator
- RV32IM decode/execute support plus custom `VDOT8` and `TLOOKUP` instructions
- Unit tests for simulator subsystems
- RISC-V ISA compliance flow
- Workload-style benchmarks for `matmul`, `gemv`, `fused_linear_activation`, `softmax_row`, `embedding_gather`, and `layernorm_lite`

## Architecture At A Glance

- **ISA:** RV32IM + `VDOT8` + `TLOOKUP`
- **Pipeline:** Fetch -> Decode -> Issue -> Operand Collect -> Dispatch -> Execute -> Writeback
- **Warps:** 32 threads per warp, with 4-8 resident warps parameterized in config
- **Execution units:** ALU, pipelined multiply/`VDOT8`, iterative divide, LD/ST, `TLOOKUP`
- **Memory system:** direct-mapped L1 data cache, write-through/write-allocate policy, MSHRs, and address coalescing. External DRAM is modeled by one of two swappable backends — a fixed-latency stub (default for unit tests) and a DRAMSim3 model of the DE-10 Nano DDR3-800 target (default for benchmarks). See `sim/configs/dram/DDR3_4Gb_x16_800.ini` for the canonical config.
- **Host/program model:** runner loads programs and optional data/lookup-table images, then executes via the `perf_sim` backend

The architectural source of truth is [`resources/gpu_architectural_spec.md`](resources/gpu_architectural_spec.md).

## Repository Layout

```text
sim/                        Performance simulator library
  include/gpu_sim/            Public headers
  src/                        Functional model, timing model, decoder, stats, loader
  tests/                      Catch2 unit tests

runner/                     Backend abstraction and gpu_sim executable
  include/runner/             Runner headers
  src/                        Main entry point and backend implementations

tests/                      Benchmarks and ISA test flows
  riscv-isa/                  RISC-V ISA compliance build/run scripts
  matmul/                     Matmul kernel + benchmark harness
  gemv/                       GEMV kernel + benchmark harness
  fused_linear_activation/    Fused linear + activation benchmark
  softmax_row/                Softmax-row benchmark
  embedding_gather/           Embedding gather benchmark
  layernorm_lite/             Lightweight layernorm benchmark

resources/                  Design and simulator documentation
```

## Prerequisites

Required:

- CMake 3.16+
- A C++17 compiler

Optional, depending on what you want to run:

- `riscv64-unknown-elf-gcc`, `riscv32-unknown-elf-gcc`, or `riscv64-elf-gcc` for workload benchmark kernel builds
- `riscv64-unknown-elf-gcc` for the RISC-V ISA compliance flow in `tests/riscv-isa/`

If no supported RISC-V cross-compiler is installed, the benchmark targets are skipped at CMake configure time.

## Build

Build everything from the repository root:

```bash
cmake -B build
cmake --build build -j8
```

This produces:

- `build/runner/gpu_sim` for program execution
- `build/sim/test_*` for simulator unit tests
- `build/tests/.../*_bench` for workload benchmarks when a RISC-V cross-compiler is available

## Running Programs

The main entry point is `build/runner/gpu_sim`.

```bash
# Run an ELF with the default backend
./build/runner/gpu_sim program.elf

# Functional-only execution
./build/runner/gpu_sim program.elf --functional-only

# Override configuration and warp count
./build/runner/gpu_sim program.elf --config=config.json --num-warps=4

# Load a lookup table image
./build/runner/gpu_sim program.elf --lookup-table=table.bin

# Load a data blob at a specific address
./build/runner/gpu_sim program.elf --data=input.bin@0x2000

# Emit human-readable or structured traces
./build/runner/gpu_sim program.elf --trace
./build/runner/gpu_sim program.elf --trace-file=trace.json

# Emit stats as JSON
./build/runner/gpu_sim program.elf --json
```

Run `./build/runner/gpu_sim --help` for the full CLI.

## Testing

### Simulator Unit Tests

From the repository root:

```bash
ctest --test-dir build --output-on-failure
```

Or run only the simulator tests:

```bash
ctest --test-dir build/sim --output-on-failure
```

### RISC-V ISA Compliance Tests

The ISA flow builds test ELFs under `tests/riscv-isa/build` and executes them through the runner in functional-only mode.

```bash
cd tests/riscv-isa
make
./run_tests.sh
```

Useful options:

```bash
./run_tests.sh --backend=perf_sim
./run_tests.sh --sim=../../build/runner/gpu_sim
./run_tests.sh --verbose
```

## Workload Benchmarks

The canonical benchmark entry point is:

```bash
bash ./tests/run_workload_benchmarks.sh --build-dir build
```

This script runs the current benchmark suite and emits structured lines in the form:

```text
RESULT name=<bench> status=<pass|fail|missing> cycles=<N|na> issued_instructions=<N|na> ipc=<V|na>
SUMMARY total=<N> passed=<N> failed=<N>
```

Run a subset or pass options through to each benchmark executable:

```bash
bash ./tests/run_workload_benchmarks.sh --build-dir build --bench gemv --bench matmul
bash ./tests/run_workload_benchmarks.sh --build-dir build -- --num-warps=8
```

The script defaults to the DRAMSim3 backend (DE-10 Nano DDR3-800 at `sim/configs/dram/DDR3_4Gb_x16_800.ini`); pass `--fixed-memory` to fall back to the fixed-latency stub for ad-hoc runs:

```bash
bash ./tests/run_workload_benchmarks.sh --build-dir build --fixed-memory
```

For A/B comparisons against a baseline git ref:

```bash
python3 tools/bench_compare.py --baseline HEAD~1
```

`bench_compare.py` builds the baseline in a temporary worktree, runs both sides under DRAMSim3 by default, stores results in `tools/.bench_history.db`, and prints a delta table. Pass `--fixed-memory` to compare under the fixed-latency stub instead.

## Configuration And Tracing

The simulator configuration lives in `SimConfig` and can be supplied from JSON plus CLI overrides. Important runtime controls include:

- `--num-warps=<N>`
- `--functional-only`
- `--max-cycles=<N>`
- `--trace` / `--trace-text`
- `--trace-file=<path>`
- `--json`
- `--arg0=<N>` through `--arg3=<N>`
- `--start-pc=<N>`
- `--memory-backend=<fixed|dramsim3>` (default `fixed` for the runner; benchmarks default to `dramsim3`)
- `--dramsim3-config-path=<file.ini>` (required when `--memory-backend=dramsim3`)

Structured trace output written via `--trace-file` is suitable for Chrome trace viewers and Perfetto. For a complete reference — track layout, slice coalescing, instant-event schema, counter track catalog, and the `Stats` performance-counter fields — see [`resources/trace_and_perf_counters.md`](resources/trace_and_perf_counters.md).

## Documentation

- [Architectural specification](resources/gpu_architectural_spec.md)
- [Performance simulator file reference](resources/perf_sim_arch.md)
- [Trace generation and performance counters](resources/trace_and_perf_counters.md)
- [C++ coding standard](resources/cpp_coding_standard.md)
- [RISC-V ISA reference card](resources/riscv_card.md)
- [Untested changes log](UNTESTED.md)
- [Project workflow instructions](AGENTS.md)

## Target Workload

The design is aimed at quantized LLM inference: `INT8 x INT8 -> INT32` accumulation for matrix-heavy kernels, with lookup-table-assisted nonlinear functions such as softmax, GELU, SiLU, and layer norm. The intended demo range is roughly 100M-300M parameter models.
