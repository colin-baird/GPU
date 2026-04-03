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
- **Onboarding Guide:** [/resources/onboarding.md](/resources/onboarding.md) — introduction to the project, codebase map, build/test/run instructions, and workflow overview for new contributors

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
    │ win?        │             → If adversarial tests pass: commit impl + spec + tests
    └────┬───────┘             → If adversarial tests fail: see Adversarial Test Failure loop
         │ No / Unclear
         ▼
    Orchestrator consults user with regression + benchmark data
         │
    ┌────┴────┐
    │ Keep?   │──── Yes ──→ Test Authoring Agent writes tests
    │         │             → If adversarial tests pass: commit everything together
    └────┬────┘             → If adversarial tests fail: see Adversarial Test Failure loop
         │ No / No response within session
         ▼
    Revert or stash the change
```

### Adversarial Test Failure Loop

When the test-author agent's adversarial tests expose a bug in the implementation:

```
Adversarial test finds bug
         │
         ▼
   ┌─────────────────────────────────────────────────┐
   │ Orchestrator classifies the bug                  │
   │                                                  │
   │  Major / architectural? ──────────────────────── │──→ Escalate to user immediately
   │  (wrong spec, design flaw, wrong abstraction)    │    with test + failure summary
   │                                                  │
   │  Localized bug? (off-by-one, wrong signal,       │
   │  missed edge case, encoding error, etc.)  ───────│──→ Enter fix loop (below)
   └─────────────────────────────────────────────────┘

Fix loop (max 2 iterations):

  [Loop counter: 1 of 2]
         │
         ▼
  Implementation Agent  ← bug-fix only, no refactoring or feature changes
         │ build must succeed
         ▼
  Validation Agent      ← re-run full regression suite + new adversarial tests
         │
    ┌────┴──────────────┐
    │ All tests pass?    │──── Yes ──→ Commit impl + spec + tests together
    └────┬──────────────┘
         │ No — still failing
         ▼
  [Loop counter: 2 of 2] — repeat fix loop once more
         │
    ┌────┴──────────────┐
    │ All tests pass?    │──── Yes ──→ Commit
    └────┬──────────────┘
         │ No — still failing after 2 attempts
         ▼
  Escalate to user: present both failing tests, both fix attempts,
  and a diagnosis. Do not attempt a third fix autonomously.
```

**Bug classification criteria** (orchestrator decides before entering fix loop):
- **Escalate immediately** if the fix would require changing the architectural spec, redesigning an interface, changing pipeline semantics, or if the root cause is ambiguous after reading the code.
- **Enter fix loop** if the bug is self-contained: wrong constant, missed case in a switch, incorrect bit-field extraction, logic error in a single function.

### Decision criteria

- **"Major performance win"**: measurable IPC improvement on a representative workload, or reduction in a critical-path latency. The orchestrator evaluates this from the validation agent's benchmark report.
- **Regression hard gate**: if any regression test fails, the workflow stops. The implementation agent fixes the issue or the change is reverted. Benchmarking does not proceed with broken regressions.
- **Session timeout**: if the user is consulted and does not respond within the session, the default disposition is to shelve (revert or stash).
- **Fix loop scope**: the implementation agent in a fix loop may only modify the specific code path identified as buggy. It must not refactor surrounding code, add new features, or expand scope beyond the failing test.

### Untested change logging

If a change is kept but targeted tests are deferred (e.g., user decides to keep it but skip testing for now), the orchestrator logs it in [`/UNTESTED.md`](/UNTESTED.md). The entry is removed when targeted tests are committed or the change is reverted.

### Orchestrator Responsibilities

The orchestrator is the main conversation agent. It owns the workflow state machine, all user-facing decisions, git commits, and final documentation completeness. Sub-agents implement, test, and validate — the orchestrator integrates and ships.

**Workflow and decisions:**
- Dispatch sub-agents in the correct sequence and enforce all hard gates.
- Classify bugs (architectural vs. localized) before entering the fix loop.
- Consult the user at every decision point that sub-agents cannot resolve autonomously.
- Never proceed past a failed regression gate or past the fix loop limit without user input.

**Documentation verification before every commit:**
Before committing any changeset, the orchestrator must verify that all documentation is consistent with the code. This is a hard requirement — do not skip it. Check each of the following:

| Document | Update required when |
|----------|----------------------|
| `resources/gpu_architectural_spec.md` | Any architectural change: new behavior, changed semantics, modified interfaces, new instructions |
| `resources/perf_sim_arch.md` | Any source file added, removed, renamed, or with changed responsibilities |
| `AGENTS.md` — Key References | A new documentation artifact is created |
| `AGENTS.md` — Project Structure map | Top-level directories or major layout changes |
| `AGENTS.md` — Agent Definitions / Workflow | Agent prompts or orchestration rules change |
| `README.md` | New top-level features, new CLI flags, new build targets, changed repo layout |
| `resources/onboarding.md` | New concepts, new top-level directories, workflow changes, new tools or entry points |
| `UNTESTED.md` | A change is kept but targeted tests are deferred; remove the entry when tests are committed or the change reverted |

If a sub-agent's output report indicates that no spec update was needed, verify this is correct before accepting it — the sub-agent may have missed a documentation obligation.

**Git commit ownership:**
- The orchestrator creates all commits. Sub-agents do not commit.
- Commits must bundle the implementation, spec updates, and tests together in one atomic commit.
- Do not commit documentation separately from the code change that motivated it.
- After a session with major changes, ensure a commit with a clear message describing the changes exists.

**AGENTS.md ownership:**
The orchestrator owns `AGENTS.md`. Update it when:
- An agent's responsibilities or constraints change.
- A new workflow phase or decision rule is added.
- A new documentation artifact is added to the project (add it to Key References).
- The project directory structure changes (update the Project Structure map).

---

## Documentation Sync

**Documentation must always reflect the current state of the codebase. A stale document is a bug.**

This rule applies in every context — multi-agent workflow, direct implementation, and any ad-hoc change regardless of scope. There are no exceptions for "minor" changes. If code changes, the relevant documents change in the same commit.

### Document registry

| Document | What it describes | Owner |
|----------|------------------|-------|
| `resources/gpu_architectural_spec.md` | Authoritative architecture: ISA, pipeline, memory, interfaces, all design decisions | Implementation agent (writes); orchestrator (verifies) |
| `resources/perf_sim_arch.md` | Every source file in the simulator and runner: purpose, key types, relationships | Implementation agent (writes); orchestrator (verifies) |
| `resources/onboarding.md` | Repository layout, key concepts, build/test/run instructions for new contributors | Orchestrator |
| `README.md` | Quick-start reference: features, layout, build, run, test, benchmark instructions | Orchestrator |
| `AGENTS.md` | Project context, agent definitions, orchestration workflow, documentation rules | Orchestrator |
| `UNTESTED.md` | Changes that passed regression but lack targeted test coverage | Orchestrator |

### Trigger table — what to update and when

| Change type | Required doc updates |
|-------------|---------------------|
| New or changed architectural behavior (pipeline, ISA, memory, interfaces) | `gpu_architectural_spec.md` (in-place edit, not a changelog entry) |
| Source file added, removed, or renamed | `perf_sim_arch.md` |
| Source file with significantly changed responsibilities | `perf_sim_arch.md` |
| New top-level directory or major restructuring of the repo | `AGENTS.md` Project Structure, `README.md` Repository Layout, `resources/onboarding.md` Repository Layout |
| New feature or capability visible to users (new CLI flags, new execution mode, new benchmark) | `README.md`, `resources/onboarding.md` if it affects workflow |
| New documentation artifact created | `AGENTS.md` Key References |
| Agent prompt or workflow rule changed | `AGENTS.md` |
| Change kept without targeted tests | Add entry to `UNTESTED.md` |
| Targeted tests committed for a logged change | Remove entry from `UNTESTED.md` |

### Direct implementation (no multi-agent workflow)

When making changes directly (outside the multi-agent workflow) — bug fixes, config changes, doc corrections, tooling changes, refactors — the same documentation obligations apply. The orchestrator is solely responsible for all updates since there is no implementation sub-agent.

**Before starting any direct change:**
- Read the relevant spec sections and source files. Never modify code based on assumptions.
- Identify which documents will need updating based on the trigger table above.

**Before committing a direct change:**
- Verify the build: `cmake -B build && cmake --build build -j8`.
- Walk the trigger table and confirm every applicable document has been updated.
- For macro structural changes (new directories, new major components, removed subsystems), explicitly check `README.md` and `resources/onboarding.md` — these are the most commonly forgotten.
- Do not commit with a mental note to update docs later. Docs ship with the code.

---

## General Instructions

These apply to all agents and the orchestrator:

- Don't make assumptions about any interfaces — consult the documentation and implementation files.
- When creating a new documentation artifact, use Markdown, place it in `./resources`, and add a pointer to the Key References section of `AGENTS.md`.
- Strictly adhere to the architectural spec. Ask before making assumptions or deviations.
- Strictly adhere to the coding standard documentation (`/resources/cpp_coding_standard.md`).
- When creating new sources of build artifacts, update the global `.gitignore` to exclude them from tracking.
- Use `bash ./tests/run_workload_benchmarks.sh --build-dir build` as the canonical workload benchmark entry point. Benchmark automation and validation should consume its `RESULT` and `SUMMARY` lines rather than scraping individual benchmark binaries directly.
