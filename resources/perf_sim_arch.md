# GPU Simulator -- File Reference

This document describes every file in the simulator, what it does, and where to find the interfaces it exposes. For the full architectural specification, see [`/resources/gpu_architectural_spec.md`](/resources/gpu_architectural_spec.md). For the operator-facing reference on trace generation (`--trace`, `--trace-file`) and performance counters (`Stats`, `--json`), see [`/resources/trace_and_perf_counters.md`](/resources/trace_and_perf_counters.md) — this file describes the types and wiring; that file documents user-visible formats, the Perfetto track schema, and the full counter catalog.

---

## Build System

| File | Purpose |
|------|---------|
| `/CMakeLists.txt` | Top-level build. Includes `sim/` and `runner/` as subdirectories. C++17. When tests are enabled, registers Python-based tooling checks: `timing_naming_lint` and `signal_diagram_ast_snapshot` (the latter passes the active build's `compile_commands.json` through `GPU_SIGNAL_COMPILE_DB`). |
| `.devcontainer/Dockerfile` | Ubuntu 24.04 development image. Installs build tools, the RISC-V cross compiler, Graphviz (`dot` for signal-diagram SVG generation), and a version-matched clang/libclang/Python binding stack for AST extraction. Exports `LIBCLANG_LIBRARY_PATH` and `CLANG_RESOURCE_DIR` so libclang can find its library and builtin compiler headers. |
| `.devcontainer/devcontainer.json` / `.devcontainer/post-create.sh` | VS Code/Codex container wiring, Claude state mounts, and post-create build bootstrap (`cmake -B build && cmake --build build`). |
| `sim/CMakeLists.txt` | Builds `gpu_sim_lib` (static library) and optionally test targets. Can also be built standalone. Vendors DRAMSim3 via `FetchContent` (option `GPU_SIM_USE_DRAMSIM3`, default ON) and links it `PUBLIC` into `gpu_sim_lib`. |
| `runner/CMakeLists.txt` | Builds `runner_lib` (backend abstraction) and `gpu_sim` executable. Links against `gpu_sim_lib`. |
| `sim/tests/CMakeLists.txt` | Registers each `test_*.cpp` as a Catch2 test executable linked against `gpu_sim_lib`. |

| Runtime config | Purpose |
|----------------|---------|
| `sim/configs/dram/DDR3_4Gb_x16_800.ini` | DRAMSim3 timing/structure/system parameters for the DE-10 Nano DDR3-800 target. Loaded by the `dramsim3` memory backend at startup; ignored by the default `fixed` backend. |

Build: `cmake -B build && cmake --build build -j8`

---

## Core / Foundation

### `include/gpu_sim/types.h`

Central type definitions used everywhere. No implementation file.

- **Type aliases**: `WarpId`, `LaneId`, `RegIndex`, `Address`, `Word` (all `uint32_t`); `WarpData` (`std::array<Word, 32>`)
- **Enums**: `ExecUnit` (ALU, MULTIPLY, DIVIDE, LDST, TLOOKUP, SYSTEM, NONE), `InstructionType` (17 variants), `AluOp`, `MulDivOp`, `MemOp`, `BranchOp`
- **Constants**: `WARP_SIZE = 32`, `MAX_WARPS = 8`, `NUM_REGS = 32`

### `include/gpu_sim/config.h` -- `src/config.cpp`

Simulator configuration. All parameterizable values live in `SimConfig`.

- **Struct `SimConfig`**: `num_warps`, `instruction_mem_size_bytes`, `instruction_buffer_depth`, `multiply_pipeline_stages`, `num_ldst_units`, `addr_gen_fifo_depth`, `l1_cache_size_bytes`, `cache_line_size_bytes`, `num_mshrs`, `write_buffer_depth`, `lookup_table_entries`, `external_memory_latency_cycles`, `external_memory_size_bytes`, `memory_backend` (`"fixed"` | `"dramsim3"`), `dramsim3_config_path`, `dramsim3_output_dir`, `fpga_clock_mhz`, `dram_clock_mhz`, `dramsim3_request_fifo_depth`, `dramsim3_bytes_per_burst`, `kernel_args[6]` (loaded into x1..x6 of every lane at launch), `start_pc`, `trace_enabled`, `functional_only`
- **`validate()`**: Checks constraints (warp count in [1,8], cache size power-of-2, `memory_backend` is one of `{"fixed","dramsim3"}`, clocks > 0, `cache_line_size_bytes` is a multiple of `dramsim3_bytes_per_burst`, and — when `memory_backend == "dramsim3"` — `dramsim3_request_fifo_depth >= (num_mshrs + write_buffer_depth) * (cache_line_size_bytes / dramsim3_bytes_per_burst)`. The minimum is also the recommended value: it sizes the FIFO to exactly the cache's worst-case in-flight chunk count, with `num_mshrs * chunks_per_line` slots reserved for reads and `write_buffer_depth * chunks_per_line` slots for writes. Sizing larger is wasteful; sizing smaller is rejected). Throws on failure.
- **`from_json(path)`**: Minimal hand-rolled JSON parser (key-value pairs). Supports string, bool, integer, and double fields.
- **`apply_cli_overrides(argc, argv)`**: Parses `--key=value` CLI arguments plus boolean aliases such as `--trace-text` for the human-readable stderr trace.

### `include/gpu_sim/isa.h`

RISC-V ISA constants and instruction field extraction. Header-only, no `.cpp`.

- **Opcode constants**: `OP_LUI`, `OP_AUIPC`, `OP_JAL`, `OP_JALR`, `OP_BRANCH`, `OP_LOAD`, `OP_STORE`, `OP_ALU_I`, `OP_ALU_R`, `OP_SYSTEM`, `OP_VDOT8`, `OP_TLOOKUP`
- **funct3/funct7 constants**: For all ALU, M-extension, load, store, branch, and system operations.
- **CSR addresses**: `CSR_WARP_ID` (0xC00), `CSR_LANE_ID` (0xC01), `CSR_NUM_WARPS` (0xC02)
- **Inline field extractors**: `opcode()`, `rd()`, `funct3()`, `rs1()`, `rs2()`, `funct7()`
- **Inline immediate decoders** (sign-extended): `imm_i()`, `imm_s()`, `imm_b()`, `imm_u()`, `imm_j()`

### `include/gpu_sim/trace_event.h`

The interface contract between the functional and timing models. Header-only.

- **Struct `DecodedInstruction`**: `type`, `alu_op`, `muldiv_op`, `mem_op`, `branch_op`, `target_unit`, `rd`, `rs1`, `rs2`, `imm`, `has_rd`, `reads_rd`, `num_src_regs`, `csr_addr`, `raw`
- **Struct `TraceEvent`**: `warp_id`, `pc`, `decoded`, per-lane arrays (`results[32]`, `mem_addresses[32]`, `store_data[32]`, `mem_size[32]`), branch info (`is_branch`, `branch_taken`, `branch_target`), control flow flags (`is_ecall`, `is_ebreak`, `panic_cause`), TLOOKUP/CSR fields, `sequence_number`

### `include/gpu_sim/decoder.h` -- `src/decoder.cpp`

Instruction decoder shared by functional and timing models.

- **`Decoder::decode(uint32_t instruction)`** (static): Returns `DecodedInstruction`. Large switch on opcode bits [6:0], sub-switched on funct3/funct7. Sets `target_unit`, `num_src_regs`, `has_rd`, `reads_rd` (true only for VDOT8). Only `CSRRS rd, csr, x0` is accepted for the identity CSRs; `FENCE` is treated as unsupported and decodes to `INVALID`.

### `include/gpu_sim/elf_loader.h` -- `src/elf_loader.cpp`

Binary loading utilities. Provides both model-independent parsing and convenience wrappers that load directly into a `FunctionalModel`.

- **Struct `ProgramSegment`**: `vaddr`, `data` (byte vector). One loadable segment from an ELF or raw binary.
- **Struct `ProgramImage`**: `entry_pc`, `segments`. Model-independent representation of a parsed program binary, consumed by backends.
- **`load_program_image(path)`** -> `ProgramImage`. Parses ELF32 or raw binary without loading into any model. ELF loader reads `PT_LOAD` segments; raw binary produces a single segment at address 0.
- **`load_image_into_model(FunctionalModel&, image)`**: Copies `ProgramImage` segments into the model's memories. Segments with `vaddr` below `instruction_mem_size` go to instruction memory, others to data memory.
- **`load_program(FunctionalModel&, path)`** -> `uint32_t` entry PC. Convenience wrapper: calls `load_program_image()` then `load_image_into_model()`.
- **`load_lookup_table(FunctionalModel&, path)`**: Reads binary file into `LookupTable`.
- **`load_data(FunctionalModel&, path, base_addr)`**: Reads raw data into `FlatMemory`.

### `include/gpu_sim/stats.h` -- `src/stats.cpp`

Statistics collection and reporting.

- **Struct `Stats`**: All counters default to 0. Grouped into Global, Per-warp, Pipeline, Per-unit (`UnitStats`), Memory system, and Writeback sections.
- **`report(ostream, num_warps)`**: Human-readable text summary.
- **`report_json(ostream, num_warps)`**: Machine-parseable JSON.
- For the full field catalog, report formats, and the relationship between Stats counters and structured-trace counter tracks, see [`/resources/trace_and_perf_counters.md`](/resources/trace_and_perf_counters.md) §5.

### `include/gpu_sim/timing/branch_predictor.h` -- `src/timing/branch_predictor.cpp`

Branch metadata interface used by speculative fetch steering and timing diagnostics.

- **Struct `BranchPrediction`**: `is_control_flow`, `predicted_taken`, `predicted_target`
- **Class `BranchPredictor`** (abstract): `predict(pc, raw_instruction)` -> `BranchPrediction`, `update(pc, decoded, prediction, actual_taken, actual_target)`
- **Class `StaticDirectionalBranchPredictor`**: Predicts backward conditional branches taken, forward conditional branches not taken, direct `JAL` taken, and `JALR` not taken because the target is not available at fetch time.
- **Important:** fetch uses this prediction to choose the warp's speculative next PC. Execute-stage resolution validates the prediction and triggers frontend recovery only on a mismatch.

### `include/gpu_sim/timing/timing_trace.h` -- `src/timing/timing_trace.cpp`

Structured timing-trace types and Chrome trace writer.

- **Struct `TimingTraceOptions`**: Structured trace configuration (`output_path`). This is separate from `SimConfig` so trace-file emission remains a backend/CLI concern.
- **Enums `WarpTraceState` / `WarpRestReason`**: Canonical per-cycle warp classification for structured traces and tests.
- **Structs `WarpTraceSnapshot` / `CycleTraceSnapshot`**: Post-commit per-cycle summaries used for trace emission and test assertions.
- **Class `ChromeTraceWriter`**: Dependency-free Chrome trace JSON emitter supporting metadata, complete events, instant events, and counters.
- Full semantics of every state, rest reason, slice, instant event, and counter are documented in [`/resources/trace_and_perf_counters.md`](/resources/trace_and_perf_counters.md) §3.

---

## Runner (lives in `/runner/`, separate from `sim/`)

### `runner/include/runner/backend.h` -- `runner/src/backend.cpp`

Backend routing system. Decouples program loading from execution so multiple backends can consume the same `ProgramImage`.

- **Scope:** This backend interface lives only under `runner/include/runner/`; the simulator library does not expose duplicate backend declarations under `sim/include/gpu_sim/`.

- **Class `Backend`** (abstract): `run(image, config, argc, argv)` -> exit code, `name()` -> string.
- **`create_backend(name)`** -> `unique_ptr<Backend>`. Factory function; returns `nullptr` for unknown names. Currently supports `"perf_sim"`.

### `runner/include/runner/backends/perf_sim_backend.h` -- `runner/src/backends/perf_sim_backend.cpp`

Performance simulator backend. Wraps `FunctionalModel` and `TimingModel`.

- **Class `PerfSimBackend`**: Implements `Backend`. Loads `ProgramImage` into a `FunctionalModel`, handles `--lookup-table`, `--data`, `--json`, `--max-cycles`, `--trace`, `--trace-text`, and `--trace-file` CLI options, runs functional-only or timing simulation, reports stats and register state, and can emit structured Chrome trace JSON for Perfetto.

### `runner/src/main.cpp`

Entry point. Parses `--backend=<name>` (default: `perf_sim`), loads config, calls `load_program_image()`, creates and runs the selected backend. `--trace` / `--trace-text` enable the legacy text trace; `--trace-file=<path>` enables structured timing traces. See `--help` for usage.

---

## Functional Model

### `include/gpu_sim/functional/register_file.h`

Per-warp, per-lane, per-register storage. Header-only.

- **Class `FunctionalRegisterFile`**: Flat array `regs[MAX_WARPS][WARP_SIZE][NUM_REGS]`.
- **`read(warp, lane, reg)`**: Returns 0 for r0.
- **`write(warp, lane, reg, value)`**: Silently ignores writes to r0.
- **`init_warp(warp, kernel_args[6])`**: Sets r1-r6 from kernel arguments, zeroes rest.
- **`reset()`**: Zeroes everything.

### `include/gpu_sim/functional/memory.h`

Three memory classes, all header-only.

- **Class `FlatMemory`**: Byte-addressable external memory. `read8/16/32()`, `write8/16/32()`. Little-endian. Bounds-checked.
- **Class `InstructionMemory`**: Word-indexed instruction store. `read(pc)` (indexes by pc/4), `write(index, instruction)`. Bounds-checked.
- **Class `LookupTable`**: Direct-indexed table. `read(index)` (returns 0 on OOB), `write(index, value)`.

### `include/gpu_sim/functional/alu.h` -- `src/functional/alu.cpp`

Pure compute functions, no state.

- **`execute_alu(AluOp, a, b)`**: ADD, SUB, XOR, OR, AND, SLL, SRL, SRA, SLT, SLTU.
- **`execute_mul(MulDivOp, a, b)`**: MUL, MULH, MULHSU, MULHU. Uses `int64_t` intermediate for high-half products.
- **`execute_div(MulDivOp, a, b)`**: DIV, DIVU, REM, REMU. Handles div-by-zero (returns -1 or dividend) and `INT32_MIN / -1` overflow.
- **`execute_vdot8(rs1, rs2, rd_accum)`**: Unpacks 4 signed bytes from rs1 and rs2, computes dot product, accumulates into rd.
- **`evaluate_branch(BranchOp, a, b)`**: BEQ, BNE, BLT, BGE, BLTU, BGEU.

### `include/gpu_sim/functional/functional_model.h` -- `src/functional/functional_model.cpp`

Top-level functional execution engine.

- **`FunctionalModel(const SimConfig&)`**: Allocates memory regions, calls `init_kernel()`.
- **`execute(warp_id, pc)`** -> `TraceEvent`: Fetches instruction, decodes, executes across all 32 lanes. Reads source registers, computes results, writes destination registers, performs memory operations. Handles ECALL (marks warp inactive), EBREAK (sets panic state), CSR reads (warp_id, lane_id, num_warps). Returns fully populated `TraceEvent`.
- **`latch_panic(warp_id, pc, cause)`**: Lets the timing model publish host-visible panic diagnostics when decode intercepts EBREAK before functional execution.
- **`init_kernel(config)`**: Marks warps active per `num_warps`, calls `register_file().init_warp()` for each.
- **Accessors**: `register_file()`, `memory()`, `instruction_memory()`, `lookup_table()`, `is_warp_active()`, `set_warp_active()`, `is_panicked()`, `panic_warp()`, `panic_cause()`, `panic_pc()`.

---

## Timing Model

### `include/gpu_sim/timing/reg.h`

State primitives for the timing model. Header-only. Make the
double-buffer / combinational discipline structural so a same-cycle-visible
mutation cannot be written by accident.

- **`Reg<T>`**: a clock-edge register. `current()` committed read;
  `set_next()` / `next_mut()` stage the new value; `commit()` latches
  `next -> current`; `seed()` re-establishes `next_ = current_`.
  `current_mut()` is a committed read-write reserved for the redirect-flush
  pattern (a backward control signal invalidating committed state
  mid-`evaluate()`); ordinary updates stage via `set_next()` / `next_mut()`.
- **`RegFifo<T>`**: a commit-disciplined FIFO. `stage_push()` / `stage_pop()`
  stage intents; `commit()` applies pop-then-push; `seed()` is a no-op;
  single enqueue-port claim via `claim_port()` / `port_claimed()`.
- **`Wire<T>`**: a combinational backward signal. `drive()` asserts,
  `value()` reads, `reset()` de-asserts. No committed twin, no `commit()`.
- **`RegBase`**: non-template base so heterogeneous registers share a list.
- **`RegisteredStage`**: mixin holding the register list; `register_state()`
  (called once in the owner's constructor body) enrolls each primitive;
  `seed_all()` / `commit_all()` / `reset_all()` drive them uniformly.

See `resources/timing_discipline.md` § State primitives.

### `include/gpu_sim/timing/pipeline_stage.h`

Abstract base for pipeline stages. Header-only.

- **`evaluate()`**: Reads current state, computes next state.
- **`commit()`**: Flips double-buffered state (next -> current).
- **`reset()`**: Returns to initial state.

Pipelined stages additionally provide `seed_next()` — copies each carry-forward field `current_* -> next_*` at the top of the tick (Phase 10B.0.5 / 10D.0). Stages that expose a back-pressure signal add either a `current_*()` accessor (REGISTERED — reads only the stage's own committed state) or a `next_*()` accessor (COMBINATIONAL — a transient asserted this cycle and read backward by the upstream stage). See `resources/timing_discipline.md`.

### `include/gpu_sim/timing/execution_unit.h`

Abstract base for execution units plus the writeback data structures. Header-only.

- **Struct `WritebackEntry`**: `valid`, `warp_id`, `dest_reg`, `values[WARP_SIZE]`, `source_unit`, `pc`, `raw_instruction`, `issue_cycle`.
- **Struct `RedirectRequest`**: `valid`, `warp_id`, `target_pc`. The combinational-backward branch-redirect signal (Phase 10E) — produced by `ALUUnit`, consumed by `FetchStage` / `DecodeStage`.
- **Issue-scoreboard constants/helpers**: `kUnitIterationLatency[]` (per-unit structural-hazard latency: `DIVIDE`=32, `TLOOKUP`=17, fully-pipelined `ALU`/`MULTIPLY`/`LDST`=0), the default-offset table `kIssueToWritebackOffset[]`, and `compute_issue_to_writeback_offset(unit, multiply_pipeline_stages, is_vdot8)` / `compute_writeback_bitmap_len(multiply_pipeline_stages)`. `WarpScheduler` uses the runtime helper so the binding writeback bitmap tracks `SimConfig::multiply_pipeline_stages`; the two-argument helper and `kWritebackBitmapLen` are default-config conveniences.
- **Class `ExecutionUnit`**: A separate hierarchy from `PipelineStage` (units have a different lifecycle — they produce results consumed by `WritebackArbiter` rather than participating in the unified evaluate/commit fan-in), but it shares the same convention: pure-virtual `seed_next()` (Phase 10B.0.5 — `current_* -> next_*` for carry-forward state; an empty body for the 1-cycle ALU), `evaluate()`, `commit()`, `reset()`, plus the unit-specific `current_busy()`, `current_has_result()`, `consume_result()`, and `get_type()`. Discipline: `WarpScheduler` no longer polls `current_busy()` for issue gating (Phase 10B.0 — it predicts availability scheduler-side); `current_busy()` survives for the panic-drain query (`pipeline_drained` / `execution_units_drained`) and unit tests. `current_has_result()` is the REGISTERED unit→arbiter edge (the arbiter reads committed result-buffer state); `consume_result()` is a pure read that mutates nothing (Phase 10B.3).
- **Class `QueuedWritebackSource`**: a small FIFO-backed `ExecutionUnit` retained for tests and generic writeback-source scenarios; empty `seed_next()`.

### `include/gpu_sim/timing/warp_state.h`

Per-warp timing state. Header-only.

- **Struct `WarpState`**: `pc`, `active`, `instr_buffer` (InstructionBuffer). `reset(start_pc)` sets active and clears the buffer. Phase 5: the prior `branch_in_flight` plain bool was removed and replaced by the REGISTERED `BranchShadowTracker` (see below).

### `include/gpu_sim/timing/branch_shadow_tracker.h`

Per-warp branch-shadow ("branch_in_flight") bit, double-buffered in the
exact `Scoreboard` shape. Header-only.

- **Class `BranchShadowTracker`**: Derives `RegisteredStage`; holds the
  whole per-warp bit array as a single `Reg<std::array<bool, MAX_WARPS>>`
  member (`shadow_`), registered via `register_state()` in the
  constructor body. `seed_next()` / `commit()` / `reset()` delegate to the
  `RegisteredStage` `seed_all()` / `commit_all()` / `reset_all()` loops.
- **`current_in_flight(warp)`**: Reads `shadow_.current()`. Used by
  `WarpScheduler::evaluate()` to gate issue of further instructions for a
  warp whose branch is still in flight.
- **Event-shaped writers** (all mutate `shadow_.next_mut()`):
  `note_branch_issued(w)` — called by `WarpScheduler::evaluate()` when
  it issues a BRANCH/JAL/JALR. `note_resolved_correctly(w)` — called by
  `ALUUnit::evaluate()` (branch resolution moved into the ALU in Phase
  10A) when the branch resolves with prediction == actual; clears
  immediately because no flush is needed.
  `note_redirect_applied(w)` — called by `FetchStage::apply_redirect()`
  when a mispredict-redirect lands at the top of fetch's `evaluate()`
  (Phase 10E); the deferred clear means the scheduler keeps observing
  `current()=true` through the cycle the redirect propagates from
  opcoll to fetch.
- **`seed_next()`**: Seeds the staged array from the committed one at top
  of cycle.
- **`commit()`**: Latches the staged array into the committed one at end
  of cycle.
- **Tick-order**: `branch_tracker_.seed_next()` runs alongside
  `scoreboard_.seed_next()` near the top of `TimingModel::tick()`;
  `branch_tracker_.commit()` runs after `scoreboard_.commit()` near the
  end.

### `include/gpu_sim/timing/instruction_buffer.h`

Per-warp instruction FIFO. Header-only.

- **Struct `BufferEntry`**: `decoded` (DecodedInstruction), `warp_id`, `pc`.
- **Class `InstructionBuffer`**: Deque-backed FIFO with fixed `max_depth`. Methods: `is_full()`, `is_empty()`, `size()`, `capacity()`, `push()`, `pop()`, `front()`, `flush()`, `reset()`.

### `include/gpu_sim/timing/scoreboard.h`

Double-buffered register dependency tracking. Header-only.

- **Struct `ScoreboardBits`**: POD wrapping the whole `bool
  pending[MAX_WARPS][NUM_REGS]` array, so it can be held in one register.
- **Class `Scoreboard`**: Derives `RegisteredStage`; holds the whole bit
  array as a single `Reg<ScoreboardBits>` member (`bits_`), registered via
  `register_state()` in the constructor body. `seed_next()` / `commit()` /
  `reset()` delegate to the `RegisteredStage` `seed_all()` / `commit_all()`
  / `reset_all()` loops.
- **`current_pending(warp, reg)`**: Reads `bits_.current()`. r0 always
  returns false.
- **`set_pending(warp, reg)`**: Mutates `bits_.next_mut()`. Called at issue time.
- **`clear_pending(warp, reg)`**: Mutates `bits_.next_mut()`. Called at writeback.
- **`seed_next()`**: Seeds the staged array from the committed one at start of cycle.
- **`commit()`**: Latches the staged array into the committed one at end of cycle.

### `include/gpu_sim/timing/fetch_stage.h` -- `src/timing/fetch_stage.cpp`

Instruction fetch with round-robin warp selection.

- **`FetchStage(num_warps, warps_ptr, imem_ref, predictor_ref, stats_ref)`**: derives `PipelineStage` and `RegisteredStage`; `register_state(&rr_pointer_, &output_)` in the constructor body.
- **Registered state** (`reg.h`): `rr_pointer_` is a `Reg<uint32_t>` round-robin warp pointer; `output_` is a `Reg<std::optional<FetchOutput>>` fetch→decode output slot. `FetchStage` has **no** `seed_next()` and is not seeded by `TimingModel::tick()` — faithful because `evaluate()` stages both registers on every code path each cycle.
- **`evaluate()`**: First applies any combinational-backward redirect: reads `alu_->next_redirect()` (or the test override) at the top of `evaluate()` and, if valid, calls private `apply_redirect(warp_id, target_pc)` — sets warp PC, flushes the warp's instruction buffer, invalidates any in-flight fetch for that warp (clearing the matching committed/staged `output_` slots; the committed-slot clear via `Reg::current_mut()` is the same-cycle redirect-flush effect), clears `branch_tracker_` for the warp via `note_redirect_applied()`, and marks the warp to be skipped this cycle (Phase 10E — moved here from `commit()`). Then scans forward from `rr_pointer_.current()` through all warps to find the first eligible warp (active and buffer not full after accounting for decode's pending entry and the in-flight fetch register). Stalls (produces no output) when `decode->current_busy()` is true and the previous output is still held — backpressure carries the held output forward into the staged `output_` slot so `commit()` retains it; increments `fetch_skip_backpressure` and `fetch_skip_count`. Reads instruction from `InstructionMemory`, records branch-prediction metadata, advances the warp PC. The RR pointer is staged to `(rr_pointer_.current() + 1) % num_warps`.
- **`set_decode(decode_ptr)`**: Wires fetch to the decode stage so `evaluate()` can query `decode->current_busy()` and `decode->current_pending_warp()` directly. Called once during `TimingModel` construction.
- **`set_alu(alu_ptr)`** / **`set_branch_tracker(tracker_ptr)`**: Phase 10A/10E wiring. Called once during `TimingModel` construction so `evaluate()` can read the ALU's combinational-backward `next_redirect()` and clear the branch tracker when applying it. (Replaced the former `set_opcoll(...)` — branch resolution moved from `OperandCollector` to `ALUUnit` in Phase 10A.)
- **`commit()`**: delegates to `RegisteredStage::commit_all()`, latching the `rr_pointer_` and `output_` registers. The redirect-apply was moved out of `commit()` into `evaluate()` in Phase 10E.
- **`reset()`**: delegates to `RegisteredStage::reset_all()` (clears both slots of each register), then clears the test-only override fields.
- **Test-only hooks**: `set_decode_pending_warp_override(...)`, `set_decode_ready_override(...)`, `set_redirect_request_override(...)` / `clear_redirect_request_override()` — for unit tests driving `FetchStage` in isolation.
- **Outputs**: `output()` (staged slot, `output_.next_mut()`), `current_output()` (committed slot, `output_.current()`). Struct `FetchOutput`: `raw_instruction`, `warp_id`, `pc`, `prediction`.

### `include/gpu_sim/timing/decode_stage.h` -- `src/timing/decode_stage.cpp`

Instruction decode with EBREAK detection.

- **`DecodeStage(warps_ptr, fetch_ref)`**: derives `PipelineStage` and `RegisteredStage`; `register_state(&ebreak_request_, &pending_)` in the constructor body.
- **Registered state** (`reg.h`): `pending_` is a `Reg<PendingDecode>` (the carry-forward pending decode result); `ebreak_request_` is a `Reg<EBreakRequest>` (the Phase 6 REGISTERED ebreak side-channel). Both are seeded.
- **`seed_next()`**: Phase 10D.0 — delegates to `RegisteredStage::seed_all()`, seeding `pending_` (the carry-forward staged slot from the committed slot) at the top of the tick, making `current_busy()` genuinely committed state and the back-to-front sweep safe. `seed_all()` also seeds `ebreak_request_`; byte-identical because `evaluate()` unconditionally restages it before any conditional write.
- **`current_busy()`**: `const` accessor reading `pending_.current().valid` (committed state). Consumed by `FetchStage::evaluate()` as the REGISTERED back-pressure gate.
- **`evaluate()`**: Restages `ebreak_request_` to an empty `EBreakRequest{}`. Then applies any combinational-backward redirect — reads `alu_->next_redirect()` (or the test override) and, if valid and the staged `pending_` warp matches, drops the staged entry (Phase 10E — targets the staged slot, since the committed slot would be overwritten by `commit()`). Then, if no prior decode is still pending (`pending_.next().valid` is false — an intra-stage read of the seeded carry-forward slot), reads `fetch_.current_output()` and decodes via `Decoder::decode()`. If EBREAK, stages `ebreak_request_` with `{valid=true, warp_id, pc}` (Phase 6 REGISTERED side-channel). Otherwise stages one decoded instruction in the `pending_` staged slot.
- **`commit()`**: delegates to `RegisteredStage::commit_all()`, latching the Phase 6 ebreak register and the `pending_` register, then pushes the staged instruction to the target warp's instruction buffer once space is available (a committed-state mutation via `Reg::current_mut()` that belongs in `commit()`); otherwise the staged decode stays pending.
- **`reset()`**: delegates to `RegisteredStage::reset_all()` (clears both slots of each register), then clears the test-only override field.
- **`set_alu(alu_ptr)`**: Phase 10A/10E wiring. Called once during `TimingModel` construction so `evaluate()` can read the ALU's combinational-backward `next_redirect()`. (Replaced the former `set_opcoll(...)`.)
- **`set_redirect_request_override(...)`** / **`clear_redirect_request_override()`**: test-only hooks.
- **Snapshot helpers**: `current_pending_warp()`, `pending_entry()`.
- **`current_ebreak_request()`**: Phase 6 REGISTERED EBREAK side-channel. Returns `EBreakRequest{valid, warp_id, pc}`. `TimingModel::tick()` observes this at the top of the *next* tick (one cycle after `decode.commit()` latches it) to call `panic_->trigger`. Replaces the prior plain-bool `ebreak_detected()`/`ebreak_warp()`/`ebreak_pc()` accessors that were mutated and read in the same evaluate phase (in violation of the cross-stage signaling discipline).
The staged `BufferEntry` also carries the fetch-time `BranchPrediction`.

### `include/gpu_sim/timing/warp_scheduler.h` -- `src/timing/warp_scheduler.cpp`

Issue stage with loose round-robin scheduling and a scheduler-side issue scoreboard. Phase 4 (reg.h migration): derives `RegisteredStage` and holds every cross-cycle register as a `Reg<T>` (`rr_pointer_`, `unit_busy_`, `writeback_bitmap_`, `bitmap_head_`, `opcoll_cooldown_cycles_`, `output_`, `diagnostics_`), enrolled via `register_state(...)` in the constructor body. `seed_next()` delegates to `seed_all()`; `commit()` delegates to `commit_all()` under the writeback-stall self-gate; `reset()` calls `reset_all()` and re-sizes/re-fills the `writeback_bitmap_` vector to the configured length. `ldst_issued_total_` stays plain (annotated `// sim-instrumentation`: monotonic accumulator, not a clocked hardware register); `ldst_iteration_latency_` stays plain (annotated `// config`: set once in `set_dependencies`).

- **Enum `SchedulerIssueOutcome`**: `INACTIVE`, `BUFFER_EMPTY`, `BRANCH_SHADOW`, `SCOREBOARD`, `OPCOLL_BUSY`, `UNIT_BUSY_ALU`, `UNIT_BUSY_MULTIPLY`, `UNIT_BUSY_DIVIDE`, `UNIT_BUSY_TLOOKUP`, `UNIT_BUSY_LDST`, `READY_NOT_SELECTED`, `ISSUED`.
- **`WarpScheduler(num_warps, warps_ptr, func_model_ref, stats_ref, multiply_pipeline_stages)`**: constructor body sizes both slots of `writeback_bitmap_` to `compute_writeback_bitmap_len(multiply_pipeline_stages)` and calls `register_state(&rr_pointer_, &unit_busy_, &writeback_bitmap_, &bitmap_head_, &opcoll_cooldown_cycles_, &output_, &diagnostics_)`.
- **Issue scoreboard (Phase 10B.0)**: the scheduler predicts unit and writeback-port availability from its own *issue history* rather than polling downstream `current_busy()` signals. Fields (all `Reg<T>` since Phase 4 — `evaluate()` mutates the staged slot via `next_mut()` and reads it back via `next()`):
  - **`unit_busy_`** (`Reg<std::array<uint32_t, kExecUnitCount>>`) — per-unit structural-hazard countdowns. Armed at issue to the unit's iteration latency, decremented once per non-frozen cycle at the top of `evaluate()`; issue to that unit is blocked while `> 0`. Non-zero only for the non-pipelined `DIVIDE`/`TLOOKUP` (compile-time `kUnitIterationLatency`) and `LDST` (runtime addr-gen latency `ldst_iteration_latency_`, captured from the wired `LdStUnit` in `set_dependencies` — the human-approved deviation from the plan's literal `kUnitIterationLatency[LDST]=0`, guarding the single addr-gen slot `LdStUnit::accept()` clobbers).
  - **`writeback_bitmap_` (`Reg<std::vector<std::optional<ExecUnit>>>`) + `bitmap_head_` (`Reg<uint32_t>`)** — the binding fixed-latency writeback schedule. A circular vector sized by `compute_writeback_bitmap_len(multiply_pipeline_stages)`; a fixed-latency op that writes back reserves the slot at `bitmap_head_ + issue_to_writeback_offset(...)`; the issue gate refuses a colliding reservation. `bitmap_head_` advances one slot per non-frozen cycle. Fixed-latency reservations only; the arbiter does not consult it.
  - **`ldst_issued_total_`** — monotonic count of LDST issues; the event-driven FIFO-occupancy gate computes in-transit population as `ldst_issued_total_ - ldst_->current_fifo_total_pushes()` plus the committed FIFO depth.
  - **`opcoll_cooldown_cycles_`** (`Reg<uint32_t>`) — interim operand-collector cooldown (2 for VDOT8, 1 otherwise; decremented each cycle).
- **`seed_next()`**: Phase 4 — delegates to `seed_all()`. Called from `TimingModel::tick()`'s seed phase alongside `opcoll_->seed_next()` / the execution units' `seed_next()`. Byte-identical to the pre-Phase-4 behavior: on a non-stalled cycle the prior tick's `commit_all()` already left `next == current` for every register, so re-seeding is a no-op; on a writeback-stall cycle `evaluate()` early-returns (no mutation) and `commit_all()` is gated (no flip), so the next tick's `seed_all()` simply repeats the same re-establishment.
- **`evaluate()`**: First checks `wb_arbiter_->next_writeback_stall()` — on a stalled cycle it **early-returns**, issuing nothing and advancing no issue bookkeeping (the bitmap head, countdowns, cooldown, LDST counters all freeze). Otherwise runs the top-of-evaluate bookkeeping on the staged slots (clear the elapsed bitmap entry via `writeback_bitmap_.next_mut()`, advance `bitmap_head_.next_mut()` by 1, decrement non-zero `unit_busy_.next_mut()` entries, decrement non-zero `opcoll_cooldown_cycles_.next_mut()`), then scans warps from `rr_pointer_.next()`: for each, checks (1) buffer not empty, (2) no branch in flight via `branch_tracker_->current_in_flight(w)`, (3) scoreboard clear, (4) the combined Phase-10B.0 issue gate (`unit_busy_.next()` countdown, the `writeback_bitmap_.next()` conflict, the LDST FIFO-occupancy gate, the `opcoll_cooldown_cycles_.next()` cooldown). First eligible warp wins. Calls `func_model_.execute()`, pops the warp's instruction buffer, performs issue-side bookkeeping (arms countdowns via `unit_busy_.next_mut()`, reserves a bitmap slot via `writeback_bitmap_.next_mut()`, advances `ldst_issued_total_`, sets `opcoll_cooldown_cycles_.set_next(...)`), sets the scoreboard pending bit, and writes `branch_tracker_->note_branch_issued(w)` on a BRANCH/JAL/JALR. Stall counters: `warp_stall_branch_shadow`, `warp_stall_scoreboard`, and `warp_stall_unit_busy` (the latter incremented for every issue-gate stall — a roll-up; the finer per-reason counters `scheduler_unit_busy_stall_cycles[]` / `scheduler_writeback_contention_stall_cycles[]` / `scheduler_ldst_fifo_full_stall_cycles` break it down). Records one committed `SchedulerIssueOutcome` per warp via `diagnostics_.next_mut()`.
- **`commit()`**: gated on `next_writeback_stall()` — on a stalled cycle it holds (keeps the already-issued `output_.current()`). Otherwise calls `commit_all()`, flipping every registered Reg.
- **`set_dependencies(scoreboard, branch_tracker, ldst)`**: wires the scoreboard and branch tracker (issue hazards) and the `LdStUnit` (read only via its REGISTERED FIFO-occupancy accessors for the LDST gate). All other unit availability is scheduler-side bookkeeping — no execution-unit pointers are wired (Phase 10B.3 removed the interim writeback-result-buffer gate that needed them).
- **`set_writeback_arbiter(arbiter)`**: wires the `WritebackArbiter` so `evaluate()` can read `next_writeback_stall()`.
- **`test_set_unit_busy(...)` / `test_reserve_writeback_slot(...)`**: test hooks that drive the issue scoreboard directly for `WarpScheduler` unit tests. Phase 4: both hooks write through BOTH `current_mut()` and `next_mut()` of the wrapped `Reg`, so the armed value is visible to `evaluate()` regardless of whether the test calls `scheduler->seed_next()` first (the existing tests do not).
- **`current_diagnostics()`**: committed per-warp `SchedulerIssueOutcome` array (`diagnostics_.current()`).
- **Outputs**: `current_output()` (committed REGISTERED slot, pulled by `OperandCollector::evaluate()`; returns `output_.current()`). Struct `IssueOutput`: `decoded`, `trace`, `warp_id`, `pc`, `prediction`.

### `include/gpu_sim/timing/operand_collector.h` -- `src/timing/operand_collector.cpp`

Models operand read timing (no actual data movement -- values are in TraceEvent). Phase 4 (reg.h migration): derives `RegisteredStage` and holds every cross-cycle field as a `Reg<T>` (`busy_`, `cycles_remaining_`, `instr_`, `output_`), enrolled via `register_state(...)` in the constructor body. `busy_`, `cycles_remaining_`, and `instr_` are genuine multi-cycle carry-forward (a VDOT8 stays in opcoll for two cycles); `output_` is the REGISTERED opcoll→unit slot. `busy_this_cycle_` stays plain (`// scratch`). Phase 10A removed all branch-resolution / redirect machinery from `OperandCollector` (it moved to `ALUUnit`).

- **`OperandCollector(stats_ref)`**: constructor body calls `register_state(&busy_, &cycles_remaining_, &instr_, &output_)`.
- **`seed_next()`**: Phase 4 — delegates to `seed_all()`. Pre-Phase-4 the hand-rolled `seed_next()` copied `busy_` / `cycles_remaining_` / `instr_` but NOT `output_`; `seed_all()` now seeds `output_` too, which is byte-identical because `evaluate()` unconditionally re-stages `output_` at its top (`output_.set_next(std::nullopt);`) before any conditional read.
- **`current_busy()`**: `const` accessor reading `busy_.current()`. Phase 10B.0 — the `WarpScheduler` no longer reads this for issue gating (it predicts opcoll availability scheduler-side via `opcoll_cooldown_cycles_`); retained for the panic-drain query and post-commit observers.
- **`accept(IssueOutput)`**: Writes only the staged slots via `set_next()`. Sets `cycles_remaining_` to 1 (2-operand) or 2 (VDOT8/3-operand).
- **`evaluate()`**: Phase 10B.2 pull model — pulls `scheduler_->current_output()` (the REGISTERED scheduler→opcoll edge) and, if present, latches it via its own `accept()`. Decrements `cycles_remaining_.next_mut()`; when 0, produces a `DispatchInput` into `output_.set_next(...)` and clears `busy_.set_next(false)`.
- **`commit()`**: Phase 10B.3 — gated on `wb_arbiter_->next_writeback_stall()`; on a stalled cycle it holds (skips the flip). Otherwise applies the relocated `operand_collector_busy_cycles` `Stats` increment (10B.0.5 — counted in `commit()` so a re-evaluated stalled cycle is not double-counted) and calls `commit_all()` to flip every registered Reg.
- **`set_scheduler(scheduler)`** / **`set_writeback_arbiter(arbiter)`**: Phase 10B.1/10B.2 back-pointers. The opcoll pulls the scheduler's committed output in `evaluate()` and reads the arbiter's writeback stall in `commit()`. Null in isolated unit tests.
- **`current_output()`**: the committed REGISTERED opcoll→unit slot (`output_.current()`), pulled by each execution unit's `evaluate()`.
- **Snapshot helpers**: `busy()`, `current_cycles_remaining()`, `resident_warp()`, `current_instruction()` -- all read committed (`*.current()`) state, consumed by `build_cycle_snapshot()` after the tick's full commit phase.
- **Outputs**: Struct `DispatchInput`: `decoded`, `trace`, `warp_id`, `pc`, `prediction`.

### `include/gpu_sim/timing/alu_unit.h` -- `src/timing/alu_unit.cpp`

1-cycle ALU execution unit, and the owner of branch resolution (Phase 10A). Phase 3 (reg.h migration): derives `RegisteredStage` and holds its execution slot as `Reg<WritebackEntry> result_buffer_`, `Reg<bool> has_pending_`, `Reg<DispatchInput> pending_input_`, enrolled via `register_state(...)` in the constructor. The ALU has 1-cycle latency, so its execution slot spans no cycle boundary — it is category-1 boundary I/O recomputed each tick, so `seed_next()` is an **empty body** (the registered `Reg`s are committed and reset but never seeded; `evaluate()` re-stages them every cycle, so skipping the seed is byte-identical). `accept()` and `evaluate()` write only the staged slot via `set_next()` / `next_mut()`; `commit()` calls `commit_all()` at the cycle boundary, gated on the writeback stall. `current_has_result()` reads `result_buffer_.current()` for the writeback arbiter; `consume_result()` is a pure read. `branch_resolved_` is a deliberate non-register (annotated `// sim-instrumentation: deliberate non-register, see timing_discipline.md`): it fires its side-effect exactly once across a multi-cycle stall — an auto-seeded `Reg` would break the intent. `next_redirect_` stays a plain transient this phase (becomes a `Wire<T>` in a later migration phase); `pending_cycle_` and `processed_this_cycle_` are scratch (`// scratch`).

- **`ALUUnit(stats_ref)`**: constructor body calls `register_state(&result_buffer_, &has_pending_, &pending_input_)`.
- **`seed_next()`**: empty body (no multi-cycle carry-forward state — see above; deliberately does **not** call `seed_all()`).
- **`current_busy()`**: `const` accessor returning `result_buffer_.current().valid || has_pending_.current()`. Retained for the panic-drain query and unit tests (the scheduler no longer polls it).
- **`evaluate()`**: Phase 10B.1 pull model — reads `opcoll_->current_output()`, self-selects if `target_unit == ALU`, and latches via its own `accept()`. Produces a result into `result_buffer_.next_mut()` in 1 cycle — but `valid` is set only for writeback ops (`has_rd && rd != 0`); a non-writeback ALU op (a conditional branch, an x0-target op) reserves no writeback-bitmap slot and must occupy no writeback port. Resets the combinational-backward `next_redirect_` at its top. Runs **branch resolution** (Phase 10A): scores the prediction, updates the branch predictor, and on misprediction asserts `next_redirect_` / on correct prediction clears the branch tracker — all gated by the `branch_resolved_` category-4 control bit so the side-effects fire exactly once even across a multi-cycle writeback stall.
- **`commit()`**: gated on `wb_arbiter_->next_writeback_stall()`; on a stalled cycle holds. Otherwise applies the relocated `alu_stats` `Stats` increments (10B.0.5), calls `commit_all()`, and clears `branch_resolved_` (the resolve-stage register advanced).
- **`next_redirect()`**: the combinational-backward branch-redirect transient, read by `FetchStage`/`DecodeStage` the same cycle (Phase 10E).
- **`set_operand_collector(...)`** / **`set_writeback_arbiter(...)`** / **`set_sim_cycle(...)`** / **`set_branch_tracker(...)`** / **`set_branch_predictor(...)`**: post-construction wiring, nullptr-tolerant for isolated unit tests.
- **`branch_mispredicted(input)`**: public static — shared misprediction check, reused by the `branch_redirect` trace site.
- **Snapshot helpers**: `busy()`, `active_warp()`, `pending_input()`, `result_entry()`.

### `include/gpu_sim/timing/multiply_unit.h` -- `src/timing/multiply_unit.cpp`

Pipelined multiply/VDOT8 unit with configurable depth. Phase 3 (reg.h migration): derives `RegisteredStage` and holds the pipeline and result buffer as `Reg<std::deque<PipelineEntry>> pipeline_` and `Reg<WritebackEntry> result_buffer_`, enrolled via `register_state(...)`. `seed_next()` delegates to `seed_all()`, which seeds both Regs from their committed values — for `pipeline_` this is genuine multi-cycle carry-forward (the pre-Phase-3 hand-rolled `seed_next()` did the same deque copy); for `result_buffer_` the seed is overwritten unconditionally at the top of `evaluate()` (`result_buffer_.set_next(WritebackEntry{});`), so seeding it is byte-identical (the pre-Phase-3 header comment "or holds it" was stale — there is no held branch). `commit()` delegates to `commit_all()`, gated on the writeback stall.

- **`MultiplyUnit(pipeline_stages, stats_ref)`**: constructor body calls `register_state(&pipeline_, &result_buffer_)`.
- **`seed_next()`**: delegates to `seed_all()`.
- **`current_busy()`**: `const` accessor reading committed state (`pipeline_.current().empty()` / `result_buffer_.current().valid`).
- **`evaluate()`**: Phase 10B.1 pull model — reads `opcoll_->current_output()`, self-selects if `target_unit == MULTIPLY`, latches via `accept()`. Decrements pipeline entries toward completion. Re-stages `result_buffer_` fresh every cycle; if the head is ready it is popped into the staged result buffer. (A load-preempted prior result is held by the stalled (gated) commit, not by a hold-branch.)
- **`commit()`**: gated on the writeback stall; applies the relocated `mul_stats` increments and calls `commit_all()`.
- **Snapshot helpers**: `pipeline_occupancy()`, `pipeline_snapshot()`, `active_warps()`, `result_entry()`.

### `include/gpu_sim/timing/divide_unit.h` -- `src/timing/divide_unit.cpp`

Iterative divide unit, 32-cycle latency. Phase 3 (reg.h migration): derives `RegisteredStage` and holds its carry-forward state as `Reg<bool> busy_`, `Reg<uint32_t> cycles_remaining_`, `Reg<WritebackEntry> pending_result_`, `Reg<WritebackEntry> result_buffer_`, enrolled via `register_state(...)`. `seed_next()` delegates to `seed_all()`; for `busy_` / `cycles_remaining_` / `pending_result_` this is genuine carry-forward, and for `result_buffer_` the seed is byte-identical because `evaluate()` unconditionally restages it (`result_buffer_.set_next(WritebackEntry{});`) at the top. `commit()` delegates to `commit_all()`, gated on the writeback stall.

- **`DivideUnit(stats_ref)`**. Constant `DIVIDE_LATENCY = 32`. Constructor body calls `register_state(&busy_, &cycles_remaining_, &pending_result_, &result_buffer_)`.
- **`seed_next()` / `current_busy()`**: `seed_all()` delegation; committed-state busy accessor reads `busy_.current() || result_buffer_.current().valid`.
- **`evaluate()`**: pull-model `accept()` (self-selects `target_unit == DIVIDE`); starts/advances the countdown via `cycles_remaining_.next_mut()--` and reads back the mutated value via `cycles_remaining_.next()`. Busy until complete.
- **`commit()`**: gated on the writeback stall; applies `div_stats` increments and calls `commit_all()`.
- **Snapshot helpers**: `busy()`, `current_cycles_remaining()`, `pending_entry()`, `result_entry()`, `active_warp()`.

### `include/gpu_sim/timing/tlookup_unit.h` -- `src/timing/tlookup_unit.cpp`

Pipelined dual-port BRAM table lookup, 17-cycle latency (2 lanes/cycle, ceil(32/2)+1 = 17 cycles). Same shape as `DivideUnit` (Phase 3): derives `RegisteredStage`; the same four `Reg<T>` fields (`busy_`, `cycles_remaining_`, `pending_result_`, `result_buffer_`) are enrolled via `register_state(...)`; `seed_next()` delegates to `seed_all()` (byte-identical because `evaluate()` unconditionally restages `result_buffer_`); `commit()` delegates to `commit_all()`, gated on the writeback stall.

- **`TLookupUnit(stats_ref)`**. Constant `TLOOKUP_LATENCY = 17`. Constructor body calls `register_state(&busy_, &cycles_remaining_, &pending_result_, &result_buffer_)`.
- Same interface pattern as DivideUnit (pull-model `evaluate()`, writeback-stall-gated `commit()`, `seed_next()`), plus snapshot helpers `busy()`, `current_cycles_remaining()`, `pending_entry()`, `result_entry()`, `active_warp()`.

### `include/gpu_sim/timing/ldst_unit.h` -- `src/timing/ldst_unit.cpp`

Address generation unit with output FIFO. Phase 3 (reg.h migration): derives `RegisteredStage` and holds its iterative carry-forward state as `Reg<bool> busy_`, `Reg<uint32_t> cycles_remaining_`, `Reg<AddrGenFIFOEntry> pending_entry_`, enrolled via `register_state(...)`. `seed_next()` / `commit()` / `reset()` delegate to `seed_all()` / `commit_all()` / `reset_all()`; `commit_all()` is gated on the writeback stall as before. The address-generation FIFO is intentionally NOT a `RegFifo<T>`: it is a plain `std::deque<AddrGenFIFOEntry> addr_gen_fifo_` with an `std::optional<AddrGenFIFOEntry> next_push_` staging slot, because its push (here, gated by the writeback stall) and its pop (driven by `CoalescingUnit::commit`, ungated) live in different stages with different commit-time gating — a discipline the simple `RegFifo` (pop-then-push inside one owning stage's commit) cannot model without changing observable behavior. Phase 3 verified the asymmetry empirically: a `RegFifo`-wrapped migration produced cycle-count deltas across five of six workload benchmarks; reverting `addr_gen_fifo_` to the hand-rolled pattern restored byte-identicality. `fifo_total_pushes_` stays a plain `uint32_t` annotated `// sim-instrumentation` — a monotonic accumulator — and is advanced inside `commit()` in lockstep with the application of `next_push_`.

- **Struct `AddrGenFIFOEntry`**: `valid`, `warp_id`, `dest_reg`, `is_load`, `is_store`, `trace`, `issue_cycle`.
- **`LdStUnit(num_ldst_units, fifo_depth, stats_ref)`**: constructor body calls `register_state(&busy_, &cycles_remaining_, &pending_entry_)`.
- **`seed_next()`**: delegates to `seed_all()`.
- **`current_busy()`**: `const` accessor returning `busy_.current()`.
- **`evaluate()`**: Phase 10B.1 pull model — reads `opcoll_->current_output()`, self-selects if `target_unit == LDST`, and latches via `accept()` (begins address generation, latency `ceil(32 / num_ldst_units)` cycles). When address gen completes, stages the entry via `next_push_ = pending_entry_.next()` (applied at commit). If the FIFO is full at start of cycle, the completed entry is held at 0 cycles remaining. The eligibility check uses the start-of-cycle (committed) FIFO size and does not account for the consumer's same-cycle staged pop — a one-cycle bubble in the FIFO-full case, parity with `fetch_stage.cpp`'s `will_be_full` check.
- **`commit()`**: gated on the writeback stall; applies the relocated `ldst_stats` increments, applies `next_push_` to `addr_gen_fifo_` and advances `fifo_total_pushes_`, then calls `commit_all()` to latch every `Reg`.
- **`current_fifo_empty()`**, **`current_fifo_front()`**, **`pop_front()`**, **`current_fifo_size()`**, **`current_fifo_total_pushes()`**, **`current_fifo_capacity()`**, **`current_addr_gen_latency()`**, **`current_fifo_entries()`**: accessors over the committed `addr_gen_fifo_` deque or the static config. `pop_front()` is called from `CoalescingUnit::commit()` (ungated) — the cross-stage pop path documented above. The push-count / capacity / addr-gen-latency accessors feed the `WarpScheduler`'s LDST issue gate (Phase 10B.0).
- **`current_has_result()`**: Always false — LD/ST results flow through the cache/MSHR fill path and the gather buffer, not through the execution-unit result buffer.
- **Snapshot helpers**: `busy()`, `current_cycles_remaining()`, `pending_entry()`, `current_fifo_entries()`, `active_warp()`.

### `include/gpu_sim/timing/writeback_arbiter.h` -- `src/timing/writeback_arbiter.cpp`

Fixed-priority writeback arbitration among execution units and the load gather buffers (Phase 10B.3).

- **`WritebackArbiter(scoreboard_ref, stats_ref)`**
- **`add_source(ExecutionUnit*)`**: Registers ALU, MUL, DIV, TLOOKUP, and the `LoadGatherBufferFile`. Sources are classified at `evaluate()` time by `get_type()`: the gather buffer (`ExecUnit::LDST`) is the variable-latency source; the four units are fixed-latency.
- **`evaluate()`**: Sequenced **first** in the evaluate sweep. Reads each source's committed `current_has_result()`. **Fixed-priority arbitration** — the variable-latency load gather buffer wins the writeback port over a fixed-latency unit. If a load and a fixed-latency unit both have a result, the load is consumed and `next_writeback_stall()` is asserted (the fixed unit is preempted); `fixed_writeback_preempted_cycles` is incremented. The winner's result is consumed via `consume_result()` (a pure read) and the scoreboard pending bit cleared. An `assert(count_fixed_with_result() <= 1)` is the live check that the scheduler's runtime issue→writeback offsets are exact — the binding writeback bitmap guarantees at most one fixed-latency unit presents a result per cycle, so fixed-vs-fixed contention cannot occur. (Round-robin and the `writeback_conflicts` counter were removed.)
- **`next_writeback_stall()`**: the COMBINATIONAL-backward writeback stall. A single-slot transient, reset at the top of every `evaluate()`. Read same-cycle by the five execution units and `OperandCollector` (in their `commit()`) and by `WarpScheduler` (in `evaluate()`) — they self-gate.
- **`commit()`**: flips the committed-entry slot (`committed_ = pending_commit_`).
- **`current_committed_entry()`**: the writeback that happened this cycle (for trace/stats).
- **Snapshot helpers**: `current_busy()`, `ready_source_count()`.

`include/gpu_sim/timing/execution_unit.h` also defines **`QueuedWritebackSource`**, a tiny FIFO-backed `ExecutionUnit` implementation retained for tests and generic writeback-source scenarios. The LD/ST writeback source in production wiring is the `LoadGatherBufferFile` (see below).

### `include/gpu_sim/timing/cache.h` -- `src/timing/cache.cpp`

Direct-mapped L1 data cache with MSHRs and write buffer.

This timing model intentionally tracks cache residency, misses, backpressure, and writeback-source timing without storing full cache-line payloads. Load values are replayed from the functional-model trace; the cache exists here to model performance behavior, not data correctness.

- **Enum `CacheStallReason`**: `NONE`, `MSHR_FULL`, `WRITE_BUFFER_FULL`, `LINE_PINNED`.
- **`L1Cache(cache_size, line_size, num_mshrs, write_buffer_depth, max_outstanding_writes, mem_if_ref, gather_file_ref, stats_ref)`** — `max_outstanding_writes` is the global cap on enqueued-but-unacked write-throughs (asserted `>= 1`).
- **`process_load(addr, warp_id, lane_mask, results, issue_cycle, pc, raw_instruction)`**: Direct synchronous API for tests. Returns false immediately (fill-conflict retry) when a fill installed into the same set this cycle (`fill_installed_set_`). Hit -> calls `gather_file_.try_write()` for the lanes selected by `lane_mask`. Phase 7: arbitration is owned by `LoadGatherBufferFile` (single REGISTERED port-claim flag); `try_write()` returns false when the port was already claimed earlier in the same tick by an in-cycle FILL or secondary drain — caller must retry next cycle. Miss -> allocates an MSHR recording `lane_mask`, submits the read, and records a miss-allocation trace event. Returns false if MSHR full (stall). Does not produce a writeback directly. Phase M3: production coalescing path no longer calls this directly — see `set_next_load_cmd` and the cmd-stall machinery below.
- **`process_store(line_addr, warp_id, issue_cycle, pc, raw_instruction)`**: Direct synchronous API for tests. Returns false immediately (fill-conflict retry) when a fill installed into the same set this cycle. Hit -> updates cache, pushes to write buffer. Miss -> allocates MSHR (write-allocate) with trace metadata. Returns false if MSHR or write buffer full. Phase M3: production coalescing uses `set_next_store_cmd` instead.
- **Registered tag array & fill-conflict retry**: `tags_` is `Reg<std::vector<CacheTag>>` (Phase 5a: the whole tag vector is one register). Lookups (`process_load`/`process_store` hit and pin checks, `any_pinned_tag`, `pinned_line_count`, `is_pinned`) read `tags_.current()`; `complete_fill` installs into `tags_.next_mut()`; `drain_secondary_chain_head` reads `tags_.current()` (like every other tag reader) and pin-clears `tags_.next_mut()` (the write side only — a secondary of the just-filled line is gated out of the fill cycle by the registered MSHR head-detection scan regardless). `complete_fill` records a successful install's set in the COMBINATIONAL same-tick scratch `fill_installed_set_`; a load or store command to that set the same cycle is rejected (fill-conflict retry, `fill_conflict_retry_cycles` bumped) and re-staged by coalescing's valid/ready handshake. The retry covers loads **and** stores — a load miss is not side-effect-free (it allocates/chains an MSHR). Fills always win and are never blocked by a command.
- **M3-refactor — valid/ready command handshake**: `set_next_load_cmd(addr, warp_id, lane_mask, results, issue_cycle, pc, raw_instruction)` and `set_next_store_cmd(line_addr, warp_id, issue_cycle, pc, raw_instruction)` stage a cmd in `load_cmd_.next_mut()` / `store_cmd_.next_mut()` (Phase 5a: `Reg<LoadCommand>` / `Reg<StoreCommand>`). `commit()` flips next → current; `evaluate()` consumes the committed slot after handle_responses (FILL) and drain_secondary_chain_head (secondary). The cache is a **memoryless cmd consumer** — `evaluate()` always clears the committed cmd slot after the attempt (via `current_mut().valid = false`), success or not, and `commit()` re-clears the staged slot (via `set_next(LoadCommand{})` / `set_next(StoreCommand{})`) after `commit_all()` flips current = next. The staged-slot re-clear is what makes the cmd path safe under the cache's no-auto-seed policy: an in-cycle set_next from coalescing is the only path that can re-populate the next slot. The consumer-side ready signal `next_cmd_ready()` is the COMBINATIONAL-backward handshake: it is asserted iff this cycle's `evaluate()` actually processed a cmd from the slot. Coalescing reads `next_cmd_ready()` same-cycle (cache.evaluate runs first) to decide whether to advance or re-stage; producer-side retry lives entirely in the coalescing unit's `(current_entry_, serial_index_, processing_, cmd_in_flight_)` state. An assert in `cache.evaluate` enforces the throughput invariant (at most one cmd processed per cycle). `next_cmd_stall_reason()` returns the structural resource-exhaustion reason (MSHR_FULL / WRITE_BUFFER_FULL / LINE_PINNED) for trace classification.
- **Write-ack line pinning**: a queued write-through pins its cache set until the external write ack returns. `is_pinned(set)` = chain pin (`tags_.current()[set].pinned`) OR write-ack pin (`outstanding_writes_.current()[set] > 0`); both are REGISTERED (Phase 5a: `Reg<std::vector<CacheTag>>` and `Reg<std::vector<uint32_t>>` plus a `Reg<uint32_t>` total). `queue_write_through(line_addr)` is the single, **fallible** write-buffer enqueue wrapper — the write buffer has one enqueue port per cycle. The first caller claims the port (`write_buffer_.claim_port()`), stages the push (`write_buffer_.stage_push(line_addr)`, applied at commit), bumps `outstanding_writes_.next_mut()[set]` and `outstanding_writes_total_.next_mut()`, and returns true; a later caller this cycle finds `write_buffer_.port_claimed()` true and returns false without staging anything. The three enqueue sites (store-miss fill, store-secondary drain, store hit — FILL > secondary > HIT by tick order) call it after confirming admission (write-buffer depth AND `outstanding_writes_at_cap()`), and perform any tag install / MSHR free / pin clear only on a true return; a port-busy false routes into the existing backpressure path (`complete_fill` defers, `drain_secondary_chain_head` leaves the secondary, `process_store` retries — no `stalled_`/`stall_reason_`, quantified by `write_buffer_port_conflict_cycles`). A cap refusal routes into the same write-buffer-full backpressure path (stall reason `WRITE_BUFFER_FULL`, attributed to `write_throttle_stall_cycles`). The three pin-enforcement sites (`process_load`/`process_store` miss pin-check, `complete_fill` deferred-fill guard) use `is_pinned`; a stall is attributed to `line_pin_stall_cycles` when the chain pin is the cause, else `write_ack_pin_stall_cycles` (precedence chain > write-ack).
- **`handle_responses()`**: First, unconditionally consumes at most one **write ack** from `mem_if_`'s separate write-ack channel — decrementing `outstanding_writes_.next_mut()[set]` and `outstanding_writes_total_.next_mut()` — *before* the deferred-fill early return, so a fill deferred on a write-ack pin always makes progress (the deadlock fix). Then processes at most one read fill per cycle: for load fills, deposits lane values into the owning warp's gather buffer via `gather_file_.try_write(... FILL)`; for store fills, queues a write-through. Read fills are buffered internally (`pending_fill_`, `Reg<PendingCacheFill>`) if a store fill is blocked by the write buffer or a conflicting fill is deferred by a pin.
- **Registered write-buffer FIFO**: `write_buffer_` is `RegFifo<uint32_t>` (Phase 5a) — a single-enqueue-port FIFO mutated **only** by `commit()`. Evaluate-phase reads (`write_buffer_size()` -> `current_size()`, `current_empty()`, `current_front()`, depth checks) read the committed deque. `commit()` (driven by `RegFifo<T>::commit()` invoked from `commit_all()`) applies the staged ops — pop first, then push — and clears the staging slots plus the port-claim flag. It is therefore not fall-through: a line enqueued at cycle T is submittable no earlier than T+1.
- **`drain_write_buffer()`**: Reads the committed front of the write buffer, stages it to memory via `mem_if_.set_next_write_request(...)` (gated on `!mem_if_.next_request_stall()`), and **stages** the pop (`write_buffer_.stage_pop()`) instead of popping directly — the entry leaves `write_buffer_` at `commit()`. The COMBINATIONAL backward stall is the architectural backpressure path: `DRAMSim3Memory` raises `next_request_stall()` when its write-region FIFO is at capacity, and the entry stays at the buffer's head until the stall clears (`FixedLatencyMemory::next_request_stall()` is hardwired false — unbounded queue). Silently popping while stalled would lose the write (the timing-only model has no recovery).
- **`evaluate()`**: Resets the COMBINATIONAL stall scratch (`stalled_` / `stall_reason_` / `next_cmd_ready_`) to NONE/false and the fill-conflict scratch `fill_installed_set_` to `-1`, then **in-place seeds** every registered field from its committed state (`tags_.set_next(tags_.current())`, `outstanding_writes_.set_next(outstanding_writes_.current())`, `outstanding_writes_total_.set_next(outstanding_writes_total_.current())`, `pending_fill_.set_next(pending_fill_.current())`, the four `last_*_event_` slots — full struct seed plus an explicit `.valid = false` clear), and seeds the registered MSHR file via `mshrs_.seed_next()`. The cache is intentionally NOT enrolled in `TimingModel::tick()`'s seed phase and has no `seed_next()` method: an unconditional auto-seed of `load_cmd_` / `store_cmd_` would re-latch a consumed cmd on the cycle after evaluate() invalidates it, breaking the memoryless-consumer contract; in-place seeding preserves byte-identical behavior with the pre-migration code. After seeding, `handle_responses()` (FILL) and `drain_secondary_chain_head()` (secondary) run, then the HIT slot processes `load_cmd_.current()` / `store_cmd_.current()` (mid-cycle invalidation via `current_mut().valid = false`). Phase 7: the prior `gather_extract_port_used_` cache-side scratch flag is removed; FILL > secondary > HIT priority is encoded by tick ordering — `cache_->evaluate()` runs at the top of the non-panic tick (FILL then secondary), and `coalescing_->evaluate()` runs later in the tick (HIT via the cmd path). The shared port itself is arbitrated by `LoadGatherBufferFile::try_write` reading its REGISTERED `next_port_claimed_` flag.
- **`commit()`**: Phase 9 boundary discipline. `commit_all()` (mixed in from `RegisteredStage`) flips every enrolled `Reg`'s next → current (`tags_`, `pending_fill_`, `outstanding_writes_`, `outstanding_writes_total_`, the four `last_*_event_` slots, `load_cmd_`, `store_cmd_`) and applies the `write_buffer_` `RegFifo` (pop-then-push plus staging/port-claim clear) in a single sweep — so post-commit observers read consistent committed values via the `current_*` accessors. The MSHR file is committed separately (`mshrs_.commit()`) — Phase 5b will fold it into the `RegisteredStage` list. After `commit_all()`, `load_cmd_` and `store_cmd_` are re-cleared via `set_next(LoadCommand{})` / `set_next(StoreCommand{})` to match the pre-migration `next_*_cmd_ = LoadCommand{}` tail-of-commit clear (without it, a commit() with no intervening `set_next_*_cmd` would re-latch a stale staged value). The COMBINATIONAL same-tick scratch (`stalled_`, `stall_reason_`, `fill_installed_set_`, `next_cmd_ready_`) is not pipeline state; `fill_installed_set_` is additionally cleared here to bound its lifetime to one tick. See inventory rows 9, 10 in `resources/timing_discipline.md`.
- **Same-line MSHR merging**: Each `MSHREntry` carries `next_in_chain` and `is_secondary`. On a miss, the cache scans MSHRs for a same-line entry; if found, the new MSHR is linked as a secondary and does not submit an external read. A primary fill installs the line, sets the tag's `pinned` bit when a chain follows, and subsequent cycles drain the chain head one per cycle via `drain_secondary_chain_head()` (loads extract into the owning warp's gather buffer; stores push to the write buffer). The pin clears when the last secondary retires. Pin conflicts (different-tag miss into a pinned set) stall with `LINE_PINNED`.
- **`is_idle()`**: True only when there are no live MSHRs, no queued write-through entries, no pending fill, no in-flight command (checks `load_cmd_.current().valid` AND `load_cmd_.next().valid` — the staged-slot intra-class self-read catches a cmd submitted by coalescing this cycle but not yet committed), and `outstanding_writes_total_.current() == 0` (a write that left the write buffer but is not yet acked keeps the cache non-idle).
- **Snapshot helpers**: `next_stalled()` and `next_stall_reason()` are COMBINATIONAL same-tick reads (single slot, no current/next pair) — coalescing observes mid-tick after `cache.evaluate()` has run as a same-cycle backpressure handshake. `current_pending_fill()`, `current_last_miss_event()`, `current_last_fill_event()`, `current_last_drain_event()`, `current_last_pin_stall_event()` are REGISTERED reads of each Reg's `current()` slot (post-commit). `active_mshr_count()`, `write_buffer_size()` (delegates to `write_buffer_.current_size()`), `mshrs()` read internal helper state.
- **Indexing**: `set = (addr / line_size) % num_sets`, `tag = addr / line_size / num_sets`.

### `include/gpu_sim/timing/load_gather_buffer.h` -- `src/timing/load_gather_buffer.cpp`

Per-resident-warp load gather buffers — the sole assembly point for load data before writeback. See §5.2.1 of the architectural spec.

- **Struct `LoadGatherBuffer`**: One per resident warp. Fields: `busy`, `dest_reg`, `values[32]`, `slot_valid[32]`, `filled_count`, `pc`, `issue_cycle`, `raw_instruction`.
- **Class `LoadGatherBufferFile : public ExecutionUnit, public RegisteredStage`**: Allocates `num_warps` buffers indexed by `warp_id` and registers with the writeback arbiter. Phase 10D: the fill presentation is genuinely **double-buffered** — the committed buffers are the state read by `consume_result()` / `buffer()` / `current_busy()`; the staged buffers accumulate this cycle's `try_write()` fills and the staged claim/release. This makes the cache → gather → writeback-arbiter forward edge a REGISTERED edge whose result is independent of the arbiter's sweep position. The gather buffer is NOT frozen by the writeback stall — it `seed_next()`s and `commit()`s every cycle. Phase 5b (reg.h migration): the file derives `RegisteredStage` and holds every registered field as a `Reg<T>` — `buffers_` is `Reg<std::vector<LoadGatherBuffer>>`, `claim_request_` is `Reg<GatherClaimRequest>`, `has_result_` is `Reg<bool>`, `rr_pointer_` is `Reg<uint32_t>`. The two non-register slots — `next_port_claimed_` (combinational, intra-tick port arbitration; Phase 7 → `Wire<bool>`) and `next_release_` (an evaluate-staging slot with no committed twin, applied at commit) — stay plain and are annotated as such.
  - **`seed_next()`**: Phase 10D + 5b — seeds `buffers_`, `rr_pointer_`, and `has_result_` (the carry-forward Regs) so per-cycle accumulating `try_write()` calls extend committed state. Deliberately does NOT seed `claim_request_`: auto-seed would re-latch a consumed claim into the staged slot on the cycle after `evaluate()` cleared the committed copy, breaking the memoryless-consumer contract. Same opt-out rationale as L1Cache's `load_cmd_` / `store_cmd_` in Phase 5a.
  - **`current_busy(warp_id)`**: per-warp committed-state query used by the coalescing unit to gate claiming a load — reads `buffers_.current()[warp_id].busy`. (The base-class `current_busy()` override is a constant `false` — the file is never a scheduler dispatch target.)
  - **`claim(warp_id, dest_reg, pc, issue_cycle, raw_instruction)`**: Phase M2 REGISTERED. Stages a `GatherClaimRequest` in `claim_request_.next_mut()`; `commit()` flips next → current; `evaluate()` (top of each tick, before `cache.evaluate`) consumes `claim_request_.current()` and applies metadata + `busy`. Single-slot — production guarantees at most one claim per cycle.
  - **`evaluate()`**: applies any pending claim from `claim_request_.current()`. The claim metadata is written into BOTH the committed buffer (`buffers_.current_mut()` — the documented Reg escape hatch, mirroring L1Cache's `load_cmd_.current_mut().valid = false`) and the staged buffer (`buffers_.next_mut()`); the committed write makes the busy flag visible to coalescing's same-cycle `current_busy()` gate, the staged write carries the claim across the commit-phase flip. After applying, clears `claim_request_.current_mut().valid = false`. Runs at the top of every tick (before `cache.evaluate`).
  - **`try_write(warp_id, lane_mask, values, source)`** where `source` is `HIT` or `FILL`: arbitrates the single gather-extract write port via the shared `next_port_claimed_` flag (first writer in the tick wins; encodes FILL > secondary > HIT priority by tick order). The write lands in `buffers_.next_mut()` and becomes visible after the next `commit()`.
  - **`current_has_result()`** (Phase M4 / 10B.3 / 5b): returns `has_result_.current()`, recomputed at `commit()` from the about-to-be-committed buffer state (`buffers_.next()`) via `has_result_.set_next(any_full)` and flipped by `commit_all()`. The REGISTERED unit→arbiter edge.
  - **`consume_result()`** (Phase 10D): a **pure read** of `buffers_.current()` — it stages a `GatherReleaseRequest` into `next_release_` and mutates no committed state; `commit()` applies the release (mutates `buffers_.next_mut()` to clear `busy`/`slot_valid`/`filled_count`, advances `rr_pointer_` via `set_next`). This fixed a Forbidden-Pattern violation where the arbiter synchronously mutated the gather buffer's committed state through `consume_result()`.
  - **`commit()`** (Phase 5b): clears `next_port_claimed_`; applies the staged release into `buffers_.next_mut()` and `rr_pointer_.next_mut()`; recomputes the staged `has_result_` from `buffers_.next()`; calls `commit_all()` to flip every enrolled Reg (`buffers_`, `rr_pointer_`, `claim_request_`, `has_result_`) in one sweep; then re-clears `claim_request_` via `set_next(GatherClaimRequest{})` so a subsequent `commit()` with no intervening `claim()` does not re-latch a stale value (matches the pre-Phase-5b `next_claim_request_ = GatherClaimRequest{}` tail-of-commit clear; same memoryless-consumer pattern as L1Cache's `load_cmd_` / `store_cmd_`).
  - **`reset()`** (Phase 5b): `reset_all()` value-initializes every Reg; the `buffers_` inner vector is then re-sized back to `num_warps_` via the `set_next` + `commit` + `set_next` pattern (same Phase 5a precedent used for L1Cache's `tags_` and `outstanding_writes_`); plain scratch (`next_port_claimed_`, `next_release_`) cleared by hand.

### `include/gpu_sim/timing/mshr.h` -- `src/timing/mshr.cpp`

Miss Status Holding Registers.

- **Struct `MSHREntry`**: `valid`, `cache_line_addr`, `is_store`, `warp_id`, `dest_reg`, `pc`, `raw_instruction`, `issue_cycle`, per-lane arrays (`mem_addresses`, `store_data`, `mem_size`, `results`), `lane_mask` (for load misses: lanes of the owning warp waiting on this fill; the cache deposits these lanes into the warp's gather buffer on fill), plus same-line merging fields `next_in_chain` (index or `INVALID_MSHR`) and `is_secondary`.
- **Class `MSHRFile : public RegisteredStage`**: REGISTERED (double-buffered) MSHR file. Phase 5b (reg.h migration): the entire `std::vector<MSHREntry>` is wrapped as one `Reg<std::vector<MSHREntry>> entries_` (whole-vector wrapping, identical to L1Cache's `Reg<std::vector<CacheTag>> tags_`); `MSHRFile` derives `RegisteredStage` for the `seed_all()` / `commit_all()` / `reset_all()` mixin only — it is an internal helper buffer (lint's `INTERNAL_HELPER_CLASSES`), NOT a pipeline stage, and is not enrolled in `TimingModel::tick()`'s seed phase. `allocate(entry)` scans `entries_.current()` for a free slot and writes the entry into `entries_.next_mut()`, returning the index (or -1). `free(index)` clears `entries_.next_mut()[index].valid`. Readers `has_free()`, `has_active()`, `find_chain_tail(line_addr)` scan `entries_.current()`. `current_at(index)` (committed read) returns `entries_.current()[index]`; `next_at(index)` (mutable next-state write handle, e.g. a chain-link write) returns `entries_.next_mut()[index]`. `seed_next()` delegates to `seed_all()` (copies current→next); `commit()` delegates to `commit_all()` (flips next→current); both driven by `L1Cache` (`seed_next()` from `evaluate()`'s in-place seeding block, `commit()` from `L1Cache::commit()`). `reset()` delegates to `reset_all()` and re-sizes `entries_` back to `num_entries_` (Reg's default value-initialization leaves the inner vector empty, same Phase 5a pattern as cache `tags_` / `outstanding_writes_`). Consequence: a slot freed this cycle is not reusable until next cycle, and a chain tail freed this cycle stays visible to a same-cycle `find_chain_tail`. The lint treats `MSHRFile` as an internal helper buffer (out of scope for the cross-stage accessor convention; its `next_at` is an intra-module registered-pair write handle).

### `include/gpu_sim/timing/coalescing_unit.h` -- `src/timing/coalescing_unit.cpp`

All-or-nothing address coalescing.

- **`CoalescingUnit(ldst_ref, cache_ref, gather_file_ref, line_size, stats_ref)`**
- **`evaluate()`**: Reads `cache_.next_stalled()` as a COMBINATIONAL backward same-tick edge (cache's stall is combinationally driven from registered tag/write-buffer/pending_fill state; `cache.evaluate()` runs earlier in the back-to-front sweep); bails this cycle if the cache reports stalled. Reads `ldst_.current_fifo_front()` (REGISTERED). For loads, stalls (without popping) if the target warp's gather buffer is busy (`gather_file_.current_busy(warp)`), incrementing `stats_.gather_buffer_stall_cycles`; once free, claims it. Checks if all 32 thread addresses fall in one cache line — if yes, stages one coalesced command; if no, 32 serialized single-lane commands, one per cycle. Commands are submitted to the cache via the M3-refactor `set_next_load_cmd` / `set_next_store_cmd` REGISTERED path, with `next_cmd_ready()` driving advance-vs-re-stage. The FIFO pop is staged in `next_pop_` here at evaluate; `CoalescingUnit::commit()` (ungated — coalescing is not part of the writeback-stall freeze) applies it via `ldst_.pop_front()`. `LdStUnit::commit()` (gated) applies its own staged push (`next_push_`) — the asymmetric pop/push gating is preserved (see the `LdStUnit` entry; Phase 3 verified empirically that the simple `RegFifo` pop-then-push-in-one-commit alternative produced cycle-count deltas). Never produces a writeback — the gather buffer emits it once all 32 slots are valid.
- **`is_idle()`**: Reports whether the unit is currently holding a warp entry mid-coalescing/serialization.
- **Snapshot helpers**: `active_warp()`, `current_entry()`, `is_coalesced()`, `serial_index()`.

### `include/gpu_sim/timing/memory_interface.h` -- `src/timing/memory_interface.cpp`

External memory abstraction. The cache talks to the abstract
`ExternalMemoryInterface` and is agnostic to the underlying timing model.

Two concrete backends derive from this interface: `FixedLatencyMemory`
(below) and `DRAMSim3Memory` (`dramsim3_memory.{h,cpp}`).

- **Struct `MemoryRequest`**: `line_addr`, `mshr_id`, `is_write`, `cycles_remaining`.
- **Struct `MemoryResponse`**: `line_addr`, `mshr_id`, `is_write`.
- **Abstract `ExternalMemoryInterface`**: Pure-virtual surface
  (`evaluate`, `commit`, `reset`, `set_next_read_request`,
  `set_next_write_request`, `next_request_stall`, `submit_read`,
  `submit_write`, `current_has_response`, `get_response`,
  `current_has_write_ack`, `get_write_ack`, `write_ack_count`, `is_idle`,
  `in_flight_count`, `response_count`). Concrete backends derive from
  this. `set_next_*_request` are the production REGISTERED stagers used
  by the cache; `submit_read` / `submit_write` are retained as the
  test-direct path (push straight into `in_flight_` for backend-isolation
  tests in `test_dramsim3_memory` and `test_timing_components`).
  `next_has_response()` is preserved as an alias for
  `current_has_response()` for compatibility.
  - **Write-ack channel**: read fills are delivered on
    `current_has_response`/`get_response`; write completions on the
    **separate** `current_has_write_ack`/`get_write_ack` channel. The
    cache drains the write-ack channel unconditionally, one ack per
    cycle, so a deferred read fill never blocks a write ack — the
    deadlock-free property the write-ack pin (`cache.*`) depends on.
- **`FixedLatencyMemory(latency, stats_ref)`**: Default backend. Every
  request completes after exactly `latency` cycles. Used by all unit tests
  and by the simulator unless a different backend is selected. Phase 6
  (reg.h migration): derives `RegisteredStage`; the REGISTERED request
  slots are `Reg<PendingMemoryRequest> read_request_` and `write_request_`,
  enrolled via `register_state(...)`. `commit()` calls `commit_all()` then
  explicitly clears the staged slot via `set_next(PendingMemoryRequest{})`
  — equivalent to the pre-migration `next_*_request_ = PendingMemoryRequest{}`
  at the tail of commit(). The backend deliberately opts out of the seed
  phase (no `seed_next()`, not in `TimingModel::tick()`'s seed list):
  evaluate consumes the slot via `current_mut().valid = false`
  (memoryless-consumer pattern, mirroring `L1Cache::load_cmd_`/`store_cmd_`
  from Phase 5a). Auto-seeding next from current would re-latch the
  consumed request.
  - `set_next_read_request(line_addr, mshr_id)`,
    `set_next_write_request(line_addr)`: REGISTERED forward; stage by
    writing `read_request_.next_mut()` / `write_request_.next_mut()`,
    increment stats. `commit_all()` flips next → current.
  - `next_request_stall()`: COMBINATIONAL backward; always `false`
    (the in_flight_ deque is unbounded for FixedLatencyMemory).
  - `submit_read(line_addr, mshr_id)`, `submit_write(line_addr)`:
    Test-direct path; pushes straight into `in_flight_` with
    `latency` countdown. Production cache uses the REGISTERED stagers.
  - `evaluate()`: Drains any valid `read_request_.current()` /
    `write_request_.current()` into `in_flight_` with `cycles_remaining =
    latency` and invalidates the committed slot via `current_mut()`, then
    decrements all in-flight countdowns. Routes completed requests to the
    right channel — reads to `responses_`, writes to the separate
    `write_acks_` deque. On read completion, accumulates `latency_` into
    `Stats::external_read_latency_total` / `external_read_latency_count`
    so the average can be compared against the DRAMSim3 backend's measured
    per-request latency.
  - `current_has_response()` / `next_has_response()` (alias),
    `get_response()`: read-fill consumption interface.
  - `current_has_write_ack()`, `get_write_ack()`, `write_ack_count()`:
    write-ack consumption interface (separate `write_acks_` deque).
  - `is_idle()`: True only when no requests are in flight, no requests
    are staged in `read_request_` / `write_request_` current/next slots,
    and neither the response queue nor the write-ack queue holds anything.
  - Snapshot helpers: `in_flight_count()`, `response_count()`.
  - Plain members (annotated): `in_flight_`, `responses_`, `write_acks_`
    are internal scheduling queues mutated only by the backend's own
    evaluate (not REGISTERED double-buffered pairs); `latency_` / `stats_`
    are config / back-pointers.

### `include/gpu_sim/timing/dramsim3_memory.h` -- `src/timing/dramsim3_memory.cpp`

DRAMSim3-backed external memory model. Selected when
`SimConfig::memory_backend == "dramsim3"`. Implements the abstract
`ExternalMemoryInterface` so the cache call sites are unchanged.

Phase 6 (reg.h migration): derives `RegisteredStage`. The REGISTERED
request slots `Reg<PendingMemoryRequest> read_request_` and
`write_request_` are enrolled via `register_state(...)`; `commit()` calls
`commit_all()` then explicitly clears the staged slot via
`set_next(PendingMemoryRequest{})`. Opts out of the seed phase (same
memoryless-consumer rationale as `FixedLatencyMemory` and
`L1Cache::load_cmd_`/`store_cmd_`): `evaluate()` consumes the committed
slot and invalidates it in-place via the documented `current_mut()`
escape hatch. The DRAM-scheduling internals (`request_fifo_`,
`write_chunks_in_fifo_`, `responses_`, `write_acks_`,
`pending_write_acks_`, `read_assembly_`, `read_chunk_to_mshr_`) are
plain internal scheduling state — NOT REGISTERED double-buffered pairs —
and stay plain. Sim-instrumentation counters (`fabric_cycle_`,
`dram_ticks_`, `phase_`, `max_response_queue_`, `max_write_ack_queue_`)
are annotated `// sim-instrumentation`; config (const after construction)
fields are annotated `// config`.

`request_fifo_` is the Phase-6 analogue of Phase 3's
`addr_gen_fifo_` `RegFifo` exception (commit `bed4043`): the producer
(`submit_read` / `submit_write`, invoked at the top of `evaluate()` to
drain `current_*_request_`, and called directly by tests) and the
consumer (the `while (phase_ >= 1.0)` DRAM-tick loop later in the same
`evaluate()`) co-occur within a single `evaluate()`. `RegFifo`'s
commit-time pop-then-push would defer the staged push until after
`evaluate()` returns, denying the consumer this cycle's just-submitted
chunks (a real cycle-of-latency behavior change). The plain
`std::deque<PendingChunk>` + explicit `write_chunks_in_fifo_` occupancy
counter encodes the in-evaluate producer→consumer pattern with no
behavior change.

- **`DRAMSim3Memory(SimConfig, Stats&)`**: Loads the DRAMSim3 `.ini` at
  `cfg.dramsim3_config_path`, sets `chunks_per_line = cache_line_size_bytes
  / dramsim3_bytes_per_burst`, sizes the per-MSHR read-assembly array to
  `cfg.num_mshrs`, pre-creates `cfg.dramsim3_output_dir` so DRAMSim3 doesn't
  emit its "WARNING: Output directory ... not exists!" line on stdout, and
  binds the DRAMSim3 read callback to its per-MSHR reassembly routine; the
  write callback is a no-op (writes are acked synthetically — see below).
- **Asynchronous fabric/DRAM clocks**: a phase accumulator advances DRAMSim3
  by `dram_clock_mhz / fpga_clock_mhz` ticks per fabric `evaluate()`,
  handling non-integer clock ratios without long-run drift.
- **Bounded request FIFO with reserved regions** (`dramsim3_request_fifo_depth`): logically split into a `num_mshrs * chunks_per_line` read region and a `write_buffer_depth * chunks_per_line` write region. `submit_read` only fails if the entire FIFO is full (architecturally impossible — at most `num_mshrs` reads can be in flight, and the read region holds exactly that many). `submit_write` returns `false` once the write region is full; `next_request_stall()` exposes the same condition as a COMBINATIONAL backward stall, and `L1Cache::drain_write_buffer` reads it before staging a write via `set_next_write_request`. `assert`s at both push sites convert any violation into an immediate failure.
- **Two bounded completion queues**: read fills land on `responses_` (capacity `read_response_queue_capacity_ = num_mshrs`), populated by `on_read_complete`; write acks land on the separate `write_acks_` deque (capacity `write_ack_queue_capacity_ = max_outstanding_writes + chunks_per_line`), populated by the synthetic write-ack release in `evaluate()`. Each push site asserts against its bound. The `worst-case cache traffic` Catch2 case in `test_dramsim3_memory.cpp` drives peak production and verifies both bounds hold (snapshots via `max_observed_response_queue()` / `max_observed_write_ack_queue()`).
- **Synthetic write acks**: write chunks are still issued to DRAMSim3 (bank/bus contention), but DRAMSim3's write callback is a no-op — it fires at submit+1 (a posted write, not durability), is address-only, and folds same-line writes. `submit_write` instead enqueues one `PendingWriteAck{line_addr, release_dram_tick}` per call; `evaluate()` promotes entries whose `release_dram_tick` has elapsed into `write_acks_`. Exactly one ack per `submit_write` — 1:1 with the cache's outstanding-write counter, no same-line folding. The release latency is `SimConfig::dramsim3_write_commit_latency_tck` DRAM cycles (default 30). `pending_write_acks_` is sorted by release tick, so `evaluate()` front-drains it.
- **Per-request latency tracking**: each `ReadAssembly` records its `submit_cycle` (sampled from a fabric-clock counter incremented in `evaluate()`); on the final chunk callback, the (`fabric_cycle_ - submit_cycle`) delta is accumulated into `Stats::external_read_latency_total` / `external_read_latency_count`. The averaged value is used to calibrate the `FixedLatencyMemory` default against representative DDR3 behavior.
- **Chunked transfers**: `submit_read(line_addr, mshr_id)` allocates the
  per-MSHR `ReadAssembly` slot and pushes one chunk per
  `dramsim3_bytes_per_burst`. The read callback decrements the slot's
  `chunks_remaining`; when it reaches zero, a single `MemoryResponse` for
  the full line is queued.
- **Write reassembly**: writes are tracked by line address. Multiple
  in-flight `submit_write` calls to the same line share one slot and emit
  one combined response, matching the cache's current semantics (write
  responses are discarded by `L1Cache`).
- **`evaluate()`**: drains at most one chunk from the request FIFO per DRAM
  tick (subject to `WillAcceptTransaction`), then `ClockTick()`s DRAMSim3.
- **`reset()`**: clears the FIFOs and assembly state and rebuilds the
  underlying `dramsim3::MemorySystem`.
- **`is_idle()`**: true only when both FIFOs are empty and no read or
  write assembly slot is active.
- Snapshot helpers: `in_flight_count()`, `response_count()`,
  `request_fifo_size()`, `chunks_per_line()`, `dram_ticks()`,
  `max_observed_response_queue()`, `response_queue_capacity()`.

### `include/gpu_sim/timing/panic_controller.h` -- `src/timing/panic_controller.cpp`

EBREAK state machine.

- **`PanicController(num_warps, warps_ptr, func_model_ref)`**
- **`trigger(warp_id, pc)`**: Activates the decode-detected panic sequence. Called from `TimingModel::tick()` at the top of the cycle after observing the REGISTERED `decode_->current_ebreak_request()` (Phase 6: one cycle after decode latched the request).
- **`evaluate()`**: After trigger, the first controller tick reads `r31` lane 0 and latches panic diagnostics into `FunctionalModel`. Later cycles drain execution units and writeback (bounded by `MAX_DRAIN_CYCLES = 32`) by querying the wired `drained_query_` callable; once drained (or the cap is hit), marks all warps inactive and sets `done_`. In-flight memory requests are abandoned for architectural purposes and are not part of the drain criterion.
- **`set_drained_query(std::function<bool()>)`**: Phase 6 wiring — `TimingModel` registers a callable bound to `execution_units_drained()`. `evaluate()` invokes it inside the controller's own evaluate (no longer a pre-evaluate setter that latched live state from another stage). nullptr-tolerant: when unset (e.g. tests that never wire it), the controller treats units as drained immediately so the state machine progresses.
- **`is_active()`**, **`is_done()`**: State queries.
- **Snapshot helpers**: `step()`, `panic_warp()`, `panic_pc()`, `panic_cause()`.

### `include/gpu_sim/timing/timing_model.h` -- `src/timing/timing_model.cpp`

Top-level cycle stepper wiring everything together.

- **`TimingModel(config, func_model_ref, stats_ref, trace_options)`**: Constructs and wires all sub-components. Selects the external-memory backend by branching on `config.memory_backend`: `"dramsim3"` constructs `DRAMSim3Memory`, anything else (default `"fixed"`) constructs `FixedLatencyMemory`. Wiring relevant to the issue/execute path: `scheduler_->set_dependencies(&scoreboard_, &branch_tracker_, ldst_.get())` and `scheduler_->set_writeback_arbiter(wb_arbiter_.get())`; each execution unit and the operand collector get their `set_operand_collector` / `set_scheduler` / `set_writeback_arbiter` / `set_sim_cycle` back-pointers; the ALU gets `set_branch_tracker` / `set_branch_predictor` (it owns branch resolution, Phase 10A); fetch and decode get `set_alu(alu_.get())` so their `evaluate()` can read the ALU's combinational-backward `next_redirect()` (Phase 10E — replaced the former `set_opcoll` wiring). The Phase 6 panic drained-query callable is wired via `panic_->set_drained_query([this](){ return execution_units_drained(); })`. When `trace_options.output_path` is non-empty, opens a `ChromeTraceWriter` and registers warp/hardware/counter tracks.
- **`tick()`** -> `bool` (continue?): One cycle of simulation. See the "Tick discipline" section of `resources/timing_discipline.md` for the full contract. Steps:
  1. **Seed phase**: `seed_next()` for `scoreboard_`, `branch_tracker_`, `opcoll_`, the five units, `decode_`, and `gather_file_` — copies each stage's carry-forward state `current_* -> next_*` so `evaluate()` is a pure function of committed state.
  2. **Top-of-cycle ebreak observation** (Phase 6): read `decode_->current_ebreak_request()`; if valid and panic not already active, call `panic_->trigger(...)` and arm `pending_panic_flush_`.
  3. **Evaluate phase (back-to-front sweep, Phase 10D)**: `wb_arbiter_` (first — asserts the combinational-backward writeback stall) -> `gather_file_` -> `{cache_.evaluate -> mem_if_.evaluate -> cache_.drain_write_buffer}` (the M5 ordering triple) -> `coalescing_` -> the five units (each pulls `opcoll_->current_output()`; the ALU resolves branches and asserts `next_redirect()`) -> `opcoll_` (pulls `scheduler_->current_output()`) -> `fetch_` -> `decode_` (both read the ALU's `next_redirect()` and apply the redirect) -> `scheduler_` (early-returns on a writeback stall). The tick-level dispatch/accept glue is gone — the issue/execute path is a bare `evaluate()` sweep over the REGISTERED pull-model edges. Two ordering carve-outs: `fetch_` evaluates before `scheduler_` (fetch must read committed `instr_buffer` occupancy, not the scheduler's same-cycle pop), and the units evaluate before the frontend (so the ALU's redirect is asserted before fetch/decode read it).
  4. **Commit phase**: every stage flips `next_* -> current_*`; the five units, `opcoll_`, and `scheduler_` self-gate on `wb_arbiter_->next_writeback_stall()` and hold on a stalled cycle. `scoreboard_.commit()` / `branch_tracker_.commit()` run unconditionally.
  5. **Panic-flush cascade**: if `pending_panic_flush_` is armed, `flush()` on `scheduler_` / `opcoll_` / `gather_file_` / `wb_arbiter_` and `branch_tracker_.reset()`, at the commit-phase boundary.
  6. Termination check: `all_warps_done() && pipeline_drained()`. When panic is active, the controller's `is_done()` arm exits.
- **`run(max_cycles)`**: Calls `tick()` in a loop.
- **SYSTEM (ECALL) retirement**: SYSTEM has no execution unit; `tick()` reads `opcoll_->current_output()` after the opcoll evaluates and, for an ECALL, marks the warp inactive (idempotent — safe to re-run on a stalled cycle).
- **`pipeline_drained()`**: Includes operand collector, execution units, coalescer, cache, memory interface, and buffered writeback state. Used for normal DONE detection to ensure all stores reach memory.
- **`execution_units_drained()`**: Narrower check covering only operand collector, execution units, LD/ST FIFO, and writeback. Used for panic drain where in-flight memory requests are abandoned.
- **Panic behavior detail:** once panic is active, cache/memory submodels may continue advancing internal timing state, but the timing model discards any would-be committed writebacks from those paths.
- **`build_cycle_snapshot()`**: Classifies each warp after commit into one `WarpTraceState` plus optional `WarpRestReason`, preferring explicit ownership (`fetch`, `decode_pending`, `operand_collect`, `execute_*`, `addr_gen`, `ldst_fifo`, `coalescing`, `memory_wait`, `writeback_wait`, `panic_drain`) and falling back to committed scheduler diagnostics for active rest reasons.
- **`record_cycle_trace()` / `emit_cycle_events()`**: Coalesce adjacent identical warp/hardware states into complete trace slices, emit per-cycle counters (`active_warps`, unit occupancy, `ldst_fifo_depth`, `active_mshrs`, `secondary_mshrs`, `pinned_lines`, `write_buffer_depth`), and publish instant events (`issue`, `branch_redirect`, `writeback`, `panic_trigger`, `cache_miss_alloc` (with `merged_secondary`), `memory_response_complete` (with `chain_length_at_fill`, `deferred`), `secondary_drain`, `line_pin_stall`).
- **`last_cycle_snapshot()`**: Returns the most recent committed `CycleTraceSnapshot` for tests.
- **`trace_cycle()`**: Prints per-cycle pipeline state to stderr when `--trace` enabled.

### Structured Trace Workflow

See [`/resources/trace_and_perf_counters.md`](/resources/trace_and_perf_counters.md) for the complete operator-facing reference (CLI flags, Perfetto track layout, slice/counter/instant-event schema, viewer workflow). Helper SQL queries live in [`/resources/perfetto_trace_queries.sql`](/resources/perfetto_trace_queries.sql).

---

## Tests

All tests use Catch2 v2.13.10 (single-header at `tests/vendor/catch.hpp`). Run from the build directory: `./tests/test_<name>`.

| File | Cases | Focus |
|------|-------|-------|
| `test_reg.cpp` | 13 | State primitives (`reg.h`): `Reg<T>` staged-vs-committed isolation, seed idempotence, stalled-cycle re-run, `next_mut` in-place mutation, reset; `RegFifo<T>` staged push/pop ordering, port claim; `Wire<T>` drive/reset; `RegisteredStage` uniform seed/commit/reset and gated-commit freeze. |
| `test_decoder.cpp` | 27 | Every RV32IM + VDOT8 + TLOOKUP + ECALL/EBREAK/CSR encoding, including unsupported CSR/FENCE forms. |
| `test_alu.cpp` | 30 | All ALU ops, MUL/DIV edge cases (overflow, div-by-zero), VDOT8 byte patterns, branch conditions. |
| `test_functional.cpp` | 16 | End-to-end functional model: ADDI chains, x0 discard, load/store, branches, ECALL/EBREAK, CSR, VDOT8, TLOOKUP, kernel args, multi-warp independence. |
| `test_scoreboard.cpp` | 9 | r0 never pending, set/clear, double-buffer isolation, same-cycle set+clear, multi-warp independence, seed_next, all-registers, reset. |
| `test_cache.cpp` | 27 | Load miss-then-hit, transient MSHR retry, store hit write buffer, store miss write-allocate, store-fill write-buffer backpressure, replayed multiple fills, direct-mapped eviction, write buffer full, reset, fill-conflict retry (store/load racing a same-set fill, evicting-fill miss, orphaned-secondary regression, store-hit-vs-evicting-fill, fill never blocked, no spurious retry), write-ack pinning (store hit pins set, pin releases the cycle after the ack, conflicting fill deferred, deadlock regression, N outstanding writes, chain→write-ack handoff, outstanding-write cap), registered write buffer (not fall-through, single enqueue port FILL>HIT, termination with the registered MSHR file). |
| `test_coalescing.cpp` | 3 | Contiguous addresses coalesce (1 request), scattered addresses serialize (32), boundary case (1 lane different). |
| `test_warp_scheduler.cpp` | 14 | Issues from buffer, skips empty, scoreboard (RAW) stall, VDOT8 rd-as-source hazard, intra-warp WAW destination-register hazard stall (pathological MUL-then-ALU-to-same-rd that would otherwise writeback out of order), unit-busy stall, writeback-bitmap contention including non-default multiply pipeline depth, RR fairness, committed scheduler diagnostics, scoreboard-pending-on-issue, idle pointer advance, scoreboard-stalled warp re-eligibility when the hazard clears, fully-pipelined ALU back-to-back issue. |
| `test_branch.cpp` | 6 | Forward-taken mispredict recovery, non-taken fall-through, backward-loop prediction accuracy, shadow-instruction-no-commit after mispredict, taken-vs-straight-line penalty comparison, fetch decode-FIFO-full backpressure. |
| `test_panic.cpp` | 5 | EBREAK halts simulation, multi-warp EBREAK, state machine step-by-step progression, panic writeback freeze, reset. |
| `test_integration.cpp` | 23 | Full timing model end-to-end: ADD chain, independent ADDIs, RAW chain, load-use stall, store-then-load, write-through completion drain, branch loop, JAL, multi-warp CSR, multi-warp ECALL, memory coalescing, LUI+ADDI, MUL, MUL-latency-vs-ALU, VDOT8, TLOOKUP, EBREAK, stats collection, x0 discard, max-cycles limit, trace snapshot classification, trace-file smoke coverage. |
| `test_timing_components.cpp` | 26 | Fetch skips full-buffer warps, fetch-decode backpressure stall, fetch PC steering from the static predictor, static branch predictor decisions, operand collection latency, ALU/MUL/DIV/TLOOKUP timing (busy signal, cycles_remaining countdown, ready_out lifecycle, back-to-back dispatch, stats tracking, reset, writeback metadata, accessor lifecycle), LD/ST FIFO backpressure, memory-interface ordering, writeback arbitration, simultaneous queued memory writebacks. |
| `test_instruction_latency.cpp` | 5 | Per-pipeline issue->writeback latency measured end-to-end through the `TimingModel` (`committed_cycle - WritebackEntry.issue_cycle`): ALU/MULTIPLY/TLOOKUP/DIVIDE each match their expected execute latency and the scheduler's issue→writeback offset helper, including a non-default multiply pipeline depth; the front pipeline is decomposed so the scheduler->opcoll issue edge (1 cycle) and the operand-collection cycles (`Stats::operand_collector_busy_cycles` — 1 for scalar ops, 2 for VDOT8) are confirmed directly rather than assumed, with VDOT8 landing exactly one cycle after an equivalent MUL; the load pipeline is variable and exceeds the external-memory latency; writeback contention — a load preempting a fixed-latency unit — extends the preempted op's latency by exactly one cycle per arbiter stall. |
| `test_config.cpp` | 8 | Default values for memory-backend knobs, default request FIFO depth equals (num_mshrs + write_buffer_depth) * chunks_per_line worst case, JSON parse round-trip for all DRAMSim3 fields, validation rejects unknown `memory_backend`, validation rejects non-positive clocks, validation rejects line size not a multiple of burst, validation rejects undersized DRAMSim3 request FIFO, CLI override parses string/double/uint fields. |
| `test_dramsim3_memory.cpp` | 10 | DE-10 Nano `.ini` loads, single-read chunk reassembly emits exactly one `MemoryResponse` after at least 16 DRAM ticks, eight reads with distinct mshr_ids each return once, sequential reads drain in fewer cycles than thrashing-strided reads, fabric/DRAM phase accumulator hits ~2667 DRAM ticks per 1000 fabric cycles at 150/400 MHz, `is_idle()` and `reset()` clear all state (write completions arrive on the write-ack channel) and the backend remains usable afterward, worst-case cache traffic (peak read+write production with bool-respecting drain and the outstanding-write cap modeled) drops no requests and keeps both the read-response and write-ack queues within capacity, write-region saturation propagates from DRAMSim3 through the cache, write-ack pinning survives a store→load across a conflicting access (the pin holds, the conflicting store stalls `LINE_PINNED`, and the system terminates), same-line writes each get their own synthetic write ack (1:1, no folding — the write-ack folding deadlock regression). |

**Totals**: 187 direct Catch2 cases.

Python tooling checks registered with top-level CTest:

| Test | Focus |
|------|-------|
| `timing_naming_lint` | Enforces cross-stage accessor naming and the libclang cross-module `next_*` read rule. |
| `signal_diagram_ast_snapshot` | Runs `tests/test_signal_diagram.py`; fails if the AST extractor reports completeness errors, misses the expected module set or edge floor, or drifts from the Markdown cross-check. |

---

## Data Flow Summary

```
                        Functional Model
                       (execute at issue)
                              |
                         TraceEvent
                              |
Fetch --> Decode --> Scheduler --> OpColl --> Dispatch --> Exec Units
  |                    |                         |         |
  |              (scoreboard                     |    ALU (1 cyc)
  |               set pending)                   |    MUL (N cyc pipelined)
  |                                              |    DIV (32 cyc)
  |                                              |    TLOOKUP (17 cyc)
  |                                              |    LD/ST -> addr FIFO
  |                                              |               |
  |                                              |        Coalescing Unit
  |                                              |               |
  |                                              |          L1 Cache
  |                                              |         /         \
  |                                              |    hit(wb)    miss(MSHR)
  |                                              |                   |
  |                                              |           Ext Memory IF
  |                                              |                   |
  |                                              |             MSHR Fill(wb)
  |                                              |                   |
  +--- branch redirect <--- Writeback Arbiter <--+-------------------+
                             (scoreboard clear)
```

---

## Validation Strategy

The repository no longer maintains a dedicated manifest-driven performance-alignment gate or a separate analytical timing oracle.

Validation relies on:

- focused Catch2 component tests in `sim/tests/`
- end-to-end timing coverage in `test_integration.cpp`
- targeted timing-path checks in `test_branch.cpp`, `test_cache.cpp`, and `test_timing_components.cpp`
- optional ISA compliance runs from `tests/riscv-isa/`
- optional workload-level benchmarking through `matmul_bench`, `gemv_bench`, etc.
- targeted spec-edge synthetic kernels under `tests/synthetic/` (`rr_tiebreak_bench`, `line_boundary_load_bench`, `mshr_same_line_race_bench`, `jalr_storm_bench`, `divide_chain_bench`, `panic_drain_test_bench`) — small RV32IM kernels that probe individual spec ambiguities (RR tie-break, coalescer line-boundary, MSHR same-line merging, JALR mispredict, iterative DIV, EBREAK panic drain); each bench self-asserts an analytical expectation derived from the spec.

The expected workflow for architectural changes is to add or update targeted Catch2 coverage near the affected component, then run the normal regression suite with:

```
ctest --test-dir build --output-on-failure
```
