# FPGA GPU Accelerator Project

## What This Is

A from-scratch GPU design in Verilog targeting FPGA synthesis and simulation. The goal is a single streaming multiprocessor (SM) that executes SIMT warps of 32 threads using the RISC-V RV32IM ISA with custom extensions, ultimately running quantized LLM inference (INT8×INT8 → INT32 accumulate).

The design is loosely modeled on NVIDIA SM architecture but simplified for FPGA feasibility: integer-only arithmetic, no shared memory, no divergence handling, single SM.

## Architecture at a Glance

- **ISA:** RV32IM + two custom instructions (VDOT8 for packed INT8 dot-product accumulate, TLOOKUP for fast nonlinear function approximation)
- **Pipeline:** Fetch → Decode → Issue (warp scheduler) → Operand Collect → Dispatch → Execute → Writeback
- **Warps:** 32 threads per warp, 4–8 resident warps per SM (parameterizable)
- **Execution units:** ALU, pipelined multiply/VDOT8, iterative divide, LD/ST, TLOOKUP — each with independent dispatch controllers
- **Memory:** Direct-mapped L1 data cache (write-through, write-allocate), MSHRs for non-blocking misses, address coalescing, external memory via Avalon-MM behind a swappable bus wrapper
- **Host interface:** CSR register block for kernel configuration, DMA engine for loading instruction and lookup table BRAMs before launch

## Key References

- **Architectural specification:** [/resources/gpu_architectural_spec.md](/resources/gpu_architectural_spec.md) — the fully specified architectural spec covering ISA, pipeline, memory system, host interface, register file, scoreboard, and all design decisions.
- **RISC-V ISA reference:** [/resources/riscv_card.md](/resources/riscv_card.md) — reference card for the RISC-V instruction set. The base ISA for this project is RV32IM (integer base + multiply/divide extension).
- **Performance Sim Documentation:** [/resources/perf_sim_arch.md](/resources/perf_sim_arch.md) — a single reference sheet covering every file in the simulator
- **C++ Coding Standard:** [/resources/cpp_coding_standard.md](/resources/cpp_coding_standard.md) — naming, formatting, ownership, error handling, and structural conventions for all C++ code
- **Untested Changes Log:** [/UNTESTED.md](/UNTESTED.md) — tracker for changes that passed regression but lack targeted test coverage

## Project Structure

```
/sim/                          # Performance simulator library (C++17)
  src/                           # Functional model, timing model, decoder, etc.
  include/gpu_sim/               # Public headers
  tests/                         # Catch2 unit tests
/runner/                       # Backend router and executable
  src/                           # Entry point, backend abstraction, backends
  include/runner/                # Runner headers
/tests/                        # Backend-agnostic test suites
  riscv-isa/                     # Official RISC-V ISA compliance tests
/resources/
  gpu_architectural_spec.md      # Full architectural spec (the source of truth)
  riscv_card.md                  # RISC-V ISA reference card
  perf_sim_arch.md               # Documentation for the simulator and runner
  cpp_coding_standard.md         # C++ style and conventions for the simulator
```

Build from the project root: `cmake -B build && cmake --build build -j8`

## Design Principles

1. **Parameterize everything** — warp count, pipeline depths, cache size, execution unit widths are all top-level parameters.
2. **Decouple with FIFOs/buffers** — no single slow stage blocks the pipeline.
3. **Uniform interfaces** — all execution units use valid-in/valid-out with warp tags.
4. **Hide latency via warp switching** — multiple resident warps cover memory and long-latency operation delays.
5. **FPGA-friendly** — DSP slices for multiply, BRAMs for storage, no associative matching or multi-ported structures.

## Target Workload

Quantized LLM inference: INT8 weights × INT8 activations with INT32 accumulation for matrix operations, ROM lookup tables with software interpolation for nonlinear functions (softmax, GELU, SiLU, layer norm). Realistic demo targets are 100M–300M parameter models.

---

## Agent Definitions

Agent prompts live in `.claude/agents/`. Each agent has a focused responsibility and strict boundaries on what it may and may not do.

| Agent | File | Responsibility |
|-------|------|---------------|
| **Implementation** | [`.claude/agents/implement.md`](.claude/agents/implement.md) | Writes code and updates the architectural spec. Does not test or benchmark. |
| **Validation** | [`.claude/agents/validate.md`](.claude/agents/validate.md) | Builds, runs regression suite, runs benchmarks. Reports structured results. Does not modify code. |
| **Test Authoring** | [`.claude/agents/test-author.md`](.claude/agents/test-author.md) | Writes targeted Catch2 tests adversarially against the spec. Does not modify implementation code. |

## Orchestration Model

The main conversation agent (the orchestrator) owns the workflow state machine and all user-facing decisions. Sub-agents are dispatched sequentially — each phase must complete before the next begins.

### Architectural Change Workflow

When the user suggests an architectural change:

```
User suggests change
        │
        ▼
┌─────────────────┐
│  Implementation  │  ← implement.md agent
│  Agent           │  Writes code + spec update
└────────┬────────┘
         │ build must succeed
         ▼
┌─────────────────┐
│  Validation      │  ← validate.md agent
│  Agent           │  Runs regression suite
└────────┬────────┘
         │ HARD GATE: all regressions must pass
         │ If failures → fix or revert, do not proceed
         ▼
┌─────────────────┐
│  Validation      │  ← validate.md agent (benchmark phase)
│  Agent           │  Runs benchmarks, reports numbers
└────────┬────────┘
         │
         ▼
    ┌────────────┐
    │ Major perf  │──── Yes ──→ Test Authoring Agent writes targeted tests
    │ win?        │             → Orchestrator commits impl + spec + tests together
    └────┬───────┘
         │ No / Unclear
         ▼
    Orchestrator consults user with regression + benchmark data
         │
    ┌────┴────┐
    │ Keep?   │──── Yes ──→ Test Authoring Agent writes tests
    │         │             → Orchestrator commits everything together
    └────┬────┘
         │ No / No response within session
         ▼
    Revert or stash the change
```

### Decision criteria

- **"Major performance win"**: measurable IPC improvement on a representative workload, or reduction in a critical-path latency. The orchestrator evaluates this from the validation agent's benchmark report.
- **Regression hard gate**: if any regression test fails, the workflow stops. The implementation agent fixes the issue or the change is reverted. Benchmarking does not proceed with broken regressions.
- **Session timeout**: if the user is consulted and does not respond within the session, the default disposition is to shelve (revert or stash).

### Untested change logging

If a change is kept but targeted tests are deferred (e.g., user decides to keep it but skip testing for now), the orchestrator logs it in [`/UNTESTED.md`](/UNTESTED.md). The entry is removed when targeted tests are committed or the change is reverted.

## General Instructions

These apply to all agents and the orchestrator:

- When updating files in the performance simulator, update the relative entries in `perf_sim_arch.md`.
- Don't make assumptions about any interfaces — consult the documentation and implementation files.
- When creating a new documentation artifact, use Markdown, place it in `./resources`, and add a pointer to the Key References section of `AGENTS.md`.
- Strictly adhere to the architectural spec. Ask before making assumptions or deviations.
- Any architectural change must be reflected in an update to `gpu_architectural_spec.md`. The spec update and implementation must be committed together in the same git commit.
- Strictly adhere to the coding standard documentation (`/resources/cpp_coding_standard.md`).
- When creating new sources of build artifacts, update the global `.gitignore` to exclude them from tracking.
- After a session with major changes, make a new git commit with a message describing the changes.
