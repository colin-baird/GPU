# Onboarding Guide

This guide is for a new human contributor to the FPGA GPU Accelerator project. Read it top-to-bottom once, then use the linked references as needed.

---

## What This Project Is

A from-scratch GPU simulator and architectural design targeting FPGA synthesis. The goal is a single **streaming multiprocessor (SM)** — similar in concept to an NVIDIA SM — that executes SIMT workloads using the **RISC-V RV32IM ISA** plus two custom instructions, ultimately running quantized LLM inference (INT8 weights × INT8 activations).

There is no RTL (Verilog) yet. The current deliverable is a **cycle-accurate C++ performance simulator** that models the full pipeline: Fetch → Decode → Issue → Operand Collect → Dispatch → Execute → Writeback.

---

## Orientation: Key Concepts

### SIMT Execution

Each **warp** is 32 threads that execute the same instruction in lockstep. The SM holds 4–8 resident warps simultaneously. While one warp stalls (waiting for memory, divide, etc.), the scheduler switches to another warp. This hides latency.

The ISA is vanilla **RV32IM** (integer + multiply/divide) plus:

- **`VDOT8 rd, rs1, rs2`** — packed INT8 dot-product accumulate; four signed INT8×INT8 products summed into `rd`. Shares the multiply pipeline.
- **`TLOOKUP rd, rs1, imm12`** — reads a preloaded lookup table BRAM for fast nonlinear approximation (softmax, GELU, etc.). Has its own execution unit. Warp latency is 64 cycles (2 cycles/lane × 32 lanes, serialized).

### Register File

Each of the 32 threads in a warp has its own copy of the 32 RISC-V registers. Register `r0` is always zero. Kernel arguments arrive in `r1`–`r4` at startup.

---

## Repository Layout

```
AGENTS.md                  Workflow rules and agent definitions — READ THIS
README.md                  Quick-start reference
UNTESTED.md                Log of changes lacking targeted tests

sim/                       C++ performance simulator (library)
  include/gpu_sim/           Public headers — start here when reading code
  src/functional/            Functional model (ISA execution, ALU, memory)
  src/timing/                Timing model (pipeline stages, execution units)
  src/                       Decoder, stats, config, ELF loader
  tests/                     Catch2 unit tests

runner/                    Entry point and backend abstraction
  src/main.cpp               CLI entry point
  src/backends/              perf_sim backend (wraps functional + timing models)

tests/                     Workload benchmarks and ISA compliance
  riscv-isa/                 Official RISC-V ISA compliance test flow
  matmul/ gemv/ softmax_row/ fused_linear_activation/
  embedding_gather/ layernorm_lite/
  run_workload_benchmarks.sh   Canonical benchmark runner

resources/                 All design and simulator documentation
  gpu_architectural_spec.md   THE source of truth for all architecture decisions
  perf_sim_arch.md            Every simulator file documented (start here for the codebase map)
  cpp_coding_standard.md      Naming, formatting, ownership conventions
  riscv_card.md               RISC-V ISA reference card
```

---

## The Two Most Important Documents

1. **`/resources/gpu_architectural_spec.md`** — the complete architectural specification. Before touching any pipeline behavior, ISA semantics, memory system, or execution unit, read the relevant section. If the spec and the code disagree, the spec wins (or you file an issue). Never make architectural assumptions without consulting it.

2. **`/resources/perf_sim_arch.md`** — a file-by-file reference for every source file in the simulator. Use this as a map when navigating the codebase: it describes each file's purpose, public interface, and key methods.

---

## How the Simulator Is Structured

The simulator has two layers that run together:

| Layer | What it does |
|-------|-------------|
| **Functional model** (`sim/src/functional/`) | Executes instructions correctly — computes results, updates register state, performs memory operations. Think of it as the ISA oracle. |
| **Timing model** (`sim/src/timing/`) | Models pipeline timing — when instructions issue, when results are available, how stalls propagate. It calls into the functional model at issue time to get a `TraceEvent` with pre-computed results, then moves that data through the pipeline. |

The two models communicate through **`TraceEvent`** structs (see `include/gpu_sim/trace_event.h`). The functional model fills a `TraceEvent` with per-lane results; the timing model uses those results when the writeback stage fires.

### Pipeline stages (in order)

1. **FetchStage** — round-robin across warps, fills each warp's instruction buffer. Uses a static branch predictor (backward branches predicted taken, forward not taken).
2. **DecodeStage** — decodes raw instruction words into `DecodedInstruction` structs.
3. **WarpScheduler** — picks an eligible warp each cycle (buffer non-empty, scoreboard clear, operand collector free, target unit ready). Calls `FunctionalModel::execute()` here.
4. **OperandCollector** — models the 1–2 cycle operand read delay (2 cycles for `VDOT8` which needs 3 source registers).
5. **Dispatch** — routes the instruction to the appropriate execution unit.
6. **Execution units** — ALU (1 cycle), MultiplyUnit (pipelined, configurable depth), DivideUnit (iterative), LdStUnit (cache + MSHR), TLookupUnit (64-cycle warp latency).
7. **WritebackArbiter** — arbitrates between units completing in the same cycle and writes results back to the register file.

### Scoreboard

The scoreboard (`include/gpu_sim/timing/scoreboard.h`) tracks in-flight destination registers. An instruction cannot issue if any of its source registers (or `rd` for `VDOT8`) has a pending write. The scoreboard is double-buffered: `set_pending` at issue, `clear_pending` at writeback, with `seed_next`/`commit` called once per cycle.

### Memory system

- **L1 data cache** — direct-mapped, write-through, write-allocate. Parameterizable size and line size.
- **MSHRs** — allow multiple outstanding cache misses (non-blocking cache).
- **Address coalescing** — LD/ST instructions from a single warp may access multiple addresses; the coalescing unit merges requests to the same cache line.
- **`TLOOKUP`** bypasses the data cache entirely.

---

## Build

Requires: CMake 3.16+, a C++17 compiler. Cross-compiler (`riscv32-unknown-elf-gcc` or similar) is needed only for benchmark kernels.

```bash
cmake -B build
cmake --build build -j8
```

Outputs:
- `build/runner/gpu_sim` — the simulator executable
- `build/sim/test_*` — unit test binaries

---

## Running the Simulator

```bash
# Basic run (timing simulation)
./build/runner/gpu_sim program.elf

# Functional-only (no timing model)
./build/runner/gpu_sim program.elf --functional-only

# Override warp count
./build/runner/gpu_sim program.elf --num-warps=8

# Load lookup table and data
./build/runner/gpu_sim program.elf --lookup-table=table.bin --data=input.bin@0x2000

# Emit stats as JSON
./build/runner/gpu_sim program.elf --json

# Emit human-readable trace to stderr
./build/runner/gpu_sim program.elf --trace

# Emit structured Chrome/Perfetto trace
./build/runner/gpu_sim program.elf --trace-file=trace.json
```

Kernel arguments are passed as `--arg0=N` through `--arg3=N` and arrive in registers `r1`–`r4`.

---

## Testing

### Unit tests

```bash
ctest --test-dir build --output-on-failure
```

Tests live in `sim/tests/`. Each `test_*.cpp` covers a specific subsystem (cache, decoder, scoreboard, branch predictor, timing components, etc.).

### ISA compliance

```bash
cd tests/riscv-isa
make
./run_tests.sh
```

### Workload benchmarks

```bash
bash ./tests/run_workload_benchmarks.sh --build-dir build
```

The script emits structured `RESULT` and `SUMMARY` lines. Run a subset with `--bench matmul --bench gemv`. Pass extra flags with `-- --num-warps=8`.

---

## The Agent Workflow (How Changes Are Made)

The project uses a structured change workflow documented in `AGENTS.md`:

1. **Implement** — write code + update `gpu_architectural_spec.md` in the same commit.
2. **Validate** — run regressions (all must pass) then run benchmarks.
3. **Test authoring** — write targeted Catch2 tests for the changed subsystem.
4. Commit implementation + spec update + tests together.

If you skip step 2 or leave regressions broken, do not proceed. The spec update and implementation must always be in the same commit.

Deferred tests are logged in `UNTESTED.md` until they are written or the change is reverted.

---

## Coding Conventions

Read `/resources/cpp_coding_standard.md` before writing any C++. Key points:

- `snake_case` for everything (files, types, variables, methods).
- Headers go in `include/gpu_sim/` (or `include/runner/`); implementation in `src/`.
- No raw owning pointers — use `unique_ptr` or value types.
- Double-buffered state: `evaluate()` writes to `next_`, `commit()` flips to `current_`.
- No exceptions in hot paths; use return values for errors at system boundaries.

When you add a new source file to the simulator, also add a description entry to `resources/perf_sim_arch.md`.

---

## Where to Start

1. Read `AGENTS.md` (5 minutes) — understand the workflow.
2. Read the ISA section of `gpu_architectural_spec.md` (§1–§3) to understand what the machine executes.
3. Read `resources/perf_sim_arch.md` to get a map of the codebase.
4. Build and run the unit tests to confirm your environment works.
5. Run the workload benchmarks to see the simulator in action.
6. Browse `sim/include/gpu_sim/` — the headers are the public contract; read them before the `.cpp` files.

---

## Reference Map

| Question | Where to look |
|----------|--------------|
| What does instruction X do? | `resources/gpu_architectural_spec.md` §2 |
| How does the pipeline work? | `resources/gpu_architectural_spec.md` §4 |
| What does file `foo.cpp` do? | `resources/perf_sim_arch.md` |
| How is the memory system designed? | `resources/gpu_architectural_spec.md` §5 |
| What are the C++ conventions? | `resources/cpp_coding_standard.md` |
| What is RISC-V RV32IM? | `resources/riscv_card.md` |
| What changes lack test coverage? | `UNTESTED.md` |
| What is the change workflow? | `AGENTS.md` | 