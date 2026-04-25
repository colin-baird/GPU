# FPGA GPU Accelerator Project

> **Note on this file:** `CLAUDE.md` is the single source of truth; `/AGENTS.md` at the repository root is a symlink to this file. Edit this file; the other view updates automatically. Any agent prompt or document that references `/AGENTS.md` is reading this exact content.

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
- **Trace & Performance Counters:** [/resources/trace_and_perf_counters.md](/resources/trace_and_perf_counters.md) — operator-facing reference for `--trace`, `--trace-file`, the Chrome/Perfetto track layout and event schema, and the `Stats` counter catalog
- **C++ Coding Standard:** [/resources/cpp_coding_standard.md](/resources/cpp_coding_standard.md) — naming, formatting, ownership, error handling, and structural conventions for all C++ code
- **Untested Changes Log:** [/UNTESTED.md](/UNTESTED.md) — tracker for changes that passed regression but lack targeted test coverage
- **Onboarding Guide:** [/resources/onboarding.md](/resources/onboarding.md) — introduction to the project, codebase map, build/test/run instructions, and workflow overview for new contributors

## Project Structure

```
/sim/                          # Performance simulator library (C++17)
  src/                           # Functional model, timing model, decoder, etc.
  include/gpu_sim/               # Public headers
  tests/                         # Catch2 unit tests
  configs/dram/                  # External-DRAM .ini files for the DRAMSim3 backend
/runner/                       # Backend router and executable
  src/                           # Entry point, backend abstraction, backends
  include/runner/                # Runner headers
/tests/                        # Backend-agnostic test suites
  riscv-isa/                     # Official RISC-V ISA compliance tests
/tools/                        # Developer tooling
  bench_compare.py               # A/B benchmark comparison and history tracking
/resources/
  gpu_architectural_spec.md      # Full architectural spec (the source of truth)
  riscv_card.md                  # RISC-V ISA reference card
  perf_sim_arch.md               # Documentation for the simulator and runner
  trace_and_perf_counters.md     # Trace generation and performance-counter reference
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
| **Validation** | [`.claude/agents/validate.md`](.claude/agents/validate.md) | Builds, runs regression suite, runs benchmarks, runs A/B benchmark comparisons. Reports structured results. Does not modify code. |
| **Test Authoring** | [`.claude/agents/test-author.md`](.claude/agents/test-author.md) | Writes targeted Catch2 tests adversarially against the spec. Does not modify implementation code. |

## Orchestration Model

**Default: single-agent.** Handle user requests directly in the main conversation. Do not dispatch sub-agents unless the user explicitly asks for the multi-agent workflow (e.g., "use the multi-agent workflow", "run this through the orchestrator", "use implement/validate/test-author").

When the user invokes the multi-agent workflow, follow the `multi-agent-workflow` skill (`.claude/skills/multi-agent-workflow/SKILL.md`). It defines the implementation → validation → test-author sequence, the regression hard gate, the adversarial fix loop, decision criteria, and commit integration rules.

The rules below (documentation verification, commit ownership, doc sync) apply in **both** single-agent and multi-agent modes.

### Architectural Change Workflow

See `.claude/skills/multi-agent-workflow/SKILL.md` for the full state machine, adversarial fix loop, decision criteria, and orchestrator dispatch rules. Invoke it only when the user explicitly asks for the multi-agent workflow.

### Orchestrator Responsibilities (always apply)

**Documentation verification before every commit:**
Before committing any changeset, the orchestrator must verify that all documentation is consistent with the code. This is a hard requirement — do not skip it. Check each of the following:

| Document | Update required when |
|----------|----------------------|
| `resources/gpu_architectural_spec.md` | Any architectural change: new behavior, changed semantics, modified interfaces, new instructions |
| `resources/perf_sim_arch.md` | Any source file added, removed, renamed, or with changed responsibilities |
| `resources/trace_and_perf_counters.md` | Any change to logging, tracing, or performance counters — new/changed trace events, counter tracks, `WarpTraceState` / `WarpRestReason` values, CLI trace flags, `Stats` fields, report formats, or Perfetto schema |
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
| `resources/trace_and_perf_counters.md` | Operator/tooling reference for `--trace`, `--trace-file`, the Chrome/Perfetto schema, and the `Stats` counter catalog | Implementation agent (writes); orchestrator (verifies) |
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
| New/changed trace event, counter track, `WarpTraceState`/`WarpRestReason`, CLI trace flag, or `Stats` field | `trace_and_perf_counters.md` |
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
- Use `python3 tools/bench_compare.py --baseline <git-ref>` for A/B benchmark comparisons when evaluating architectural changes. The tool builds the baseline in a temporary worktree, runs all benchmarks, computes deltas, and stores results in a local SQLite database for trend analysis. All six benchmark binaries support `--json` for structured output. Key flags: `--bench <name>` to filter, `--threshold <pct>` to control highlighting, `--history <bench>` to view trends.
