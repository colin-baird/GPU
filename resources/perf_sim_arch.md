# GPU Simulator -- File Reference

This document describes every file in the simulator, what it does, and where to find the interfaces it exposes. For the full architectural specification, see [`/resources/gpu_architectural_spec.md`](/resources/gpu_architectural_spec.md). For the operator-facing reference on trace generation (`--trace`, `--trace-file`) and performance counters (`Stats`, `--json`), see [`/resources/trace_and_perf_counters.md`](/resources/trace_and_perf_counters.md) — this file describes the types and wiring; that file documents user-visible formats, the Perfetto track schema, and the full counter catalog.

---

## Build System

| File | Purpose |
|------|---------|
| `/CMakeLists.txt` | Top-level build. Includes `sim/` and `runner/` as subdirectories. C++17. |
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

- **Struct `SimConfig`**: `num_warps`, `instruction_mem_size_bytes`, `instruction_buffer_depth`, `multiply_pipeline_stages`, `num_ldst_units`, `addr_gen_fifo_depth`, `l1_cache_size_bytes`, `cache_line_size_bytes`, `num_mshrs`, `write_buffer_depth`, `lookup_table_entries`, `external_memory_latency_cycles`, `external_memory_size_bytes`, `memory_backend` (`"fixed"` | `"dramsim3"`), `dramsim3_config_path`, `dramsim3_output_dir`, `fpga_clock_mhz`, `dram_clock_mhz`, `dramsim3_request_fifo_depth`, `dramsim3_bytes_per_burst`, `kernel_args[4]`, `start_pc`, `trace_enabled`, `functional_only`
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
- **`init_warp(warp, kernel_args[4])`**: Sets r1-r4 from kernel arguments, zeroes rest.
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

### `include/gpu_sim/timing/pipeline_stage.h`

Abstract base for pipeline stages. Header-only.

- **`evaluate()`**: Reads current state, computes next state.
- **`commit()`**: Flips double-buffered state (next -> current).
- **`reset()`**: Returns to initial state.

### `include/gpu_sim/timing/execution_unit.h`

Abstract base for execution units plus the writeback data structure. Header-only.

- **Struct `WritebackEntry`**: `valid`, `warp_id`, `dest_reg`, `values[WARP_SIZE]`, `source_unit`, `pc`, `raw_instruction`, `issue_cycle`.
- **Class `ExecutionUnit`**: Extends pipeline stage interface with `is_ready()`, `has_result()`, `consume_result()`, `get_type()`.

### `include/gpu_sim/timing/warp_state.h`

Per-warp timing state. Header-only.

- **Struct `WarpState`**: `pc`, `active`, `branch_in_flight`, `instr_buffer` (InstructionBuffer). `reset(start_pc)` sets active, clears `branch_in_flight`, and clears buffer. The `branch_in_flight` flag is set when the scheduler issues a branch/JAL/JALR and cleared when the operand collector dispatches it; while set, the scheduler stalls the warp to prevent issuing shadow instructions.

### `include/gpu_sim/timing/instruction_buffer.h`

Per-warp instruction FIFO. Header-only.

- **Struct `BufferEntry`**: `decoded` (DecodedInstruction), `warp_id`, `pc`.
- **Class `InstructionBuffer`**: Deque-backed FIFO with fixed `max_depth`. Methods: `is_full()`, `is_empty()`, `size()`, `capacity()`, `push()`, `pop()`, `front()`, `flush()`, `reset()`.

### `include/gpu_sim/timing/scoreboard.h`

Double-buffered register dependency tracking. Header-only.

- **Class `Scoreboard`**: Two arrays `current_[MAX_WARPS][NUM_REGS]` and `next_[...]`.
- **`is_pending(warp, reg)`**: Reads `current_`. r0 always returns false.
- **`set_pending(warp, reg)`**: Writes to `next_`. Called at issue time.
- **`clear_pending(warp, reg)`**: Writes to `next_`. Called at writeback.
- **`seed_next()`**: Copies current -> next at start of cycle.
- **`commit()`**: Copies next -> current at end of cycle.

### `include/gpu_sim/timing/fetch_stage.h` -- `src/timing/fetch_stage.cpp`

Instruction fetch with round-robin warp selection.

- **`FetchStage(num_warps, warps_ptr, imem_ref, predictor_ref, stats_ref)`**
- **`evaluate()`**: Scans forward from `rr_pointer` through all warps to find the first eligible warp (active and buffer not full after accounting for decode's pending entry). Stalls (produces no output) if the previous output has not been consumed by decode (backpressure); increments `fetch_skip_count` during backpressure stalls. Reads instruction from `InstructionMemory`, records branch-prediction metadata, and advances the warp PC to the predicted target when `predicted_taken` is set, otherwise `pc + 4`. The RR pointer always advances to `(original + 1) % num_warps` regardless of which warp was fetched.
- **`set_decode_pending_warp(optional<uint32_t>)`**: Called by the timing model before `evaluate()` to communicate which warp (if any) has a pending decode entry. Fetch treats that warp's buffer as full when it has only one slot remaining, preventing a fetch that would cause decode backpressure on the next cycle.
- **`consume_output()`**: Called by decode stage to signal it has consumed `current_output()`. Enables fetch to produce the next instruction.
- **`redirect_warp(warp_id, target_pc)`**: Sets warp PC, flushes instruction buffer, invalidates any pending fetch for that warp, resets consumption flag. Used for branch mispredict recovery.
- **Outputs**: `output()` (next), `current_output()` (committed). Struct `FetchOutput`: `raw_instruction`, `warp_id`, `pc`, `prediction`.

### `include/gpu_sim/timing/decode_stage.h` -- `src/timing/decode_stage.cpp`

Instruction decode with EBREAK detection.

- **`DecodeStage(warps_ptr, fetch_ref)`**
- **`evaluate()`**: Reads `fetch_.current_output()`. Decodes via `Decoder::decode()`. If EBREAK, signals panic and returns. Otherwise stages one decoded instruction for commit. If a prior decode is still pending because the target warp buffer is full, evaluate holds that instruction and does not decode the fetch output — EBREAK detection is deferred until the pending instruction clears.
- **`commit()`**: Pushes the staged instruction to the target warp's instruction buffer once space is available; otherwise the staged decode remains pending.
- **`has_pending()`**: Reports whether decode is holding an uncommitted instruction.
- **Snapshot helpers**: `pending_warp()`, `pending_entry()`.
- **`invalidate_warp(warp_id)`**: Clears any pending decode for that warp (branch redirect).
- **`ebreak_detected()`**, **`ebreak_warp()`**, **`ebreak_pc()`**: Panic detection interface.
The staged `BufferEntry` also carries the fetch-time `BranchPrediction`.

### `include/gpu_sim/timing/warp_scheduler.h` -- `src/timing/warp_scheduler.cpp`

Issue stage with round-robin scheduling and 4-way eligibility check.

- **Enum `SchedulerIssueOutcome`**: `INACTIVE`, `BUFFER_EMPTY`, `BRANCH_SHADOW`, `SCOREBOARD`, `OPCOLL_BUSY`, `UNIT_BUSY_ALU`, `UNIT_BUSY_MULTIPLY`, `UNIT_BUSY_DIVIDE`, `UNIT_BUSY_TLOOKUP`, `UNIT_BUSY_LDST`, `READY_NOT_SELECTED`, `ISSUED`.
- **`WarpScheduler(num_warps, warps_ptr, scoreboard_ref, func_model_ref, stats_ref)`**
- **`evaluate()`**: Scans warps starting from `rr_pointer`. For each, checks: (1) buffer not empty, (2) no branch in flight for this warp, (3) scoreboard clear for source registers, (4) operand collector free, (5) target execution unit ready. First eligible warp wins. Calls `func_model_.execute(warp_id, pc)` to produce TraceEvent, forwards the buffered `BranchPrediction`, sets scoreboard pending for `rd`, and sets `branch_in_flight` on the warp if the issued instruction is a branch/JAL/JALR. Pointer advances unconditionally. Stall counters: branch shadow stalls increment `warp_stall_branch_shadow`, scoreboard stalls increment `warp_stall_scoreboard`, operand collector busy and target unit busy both increment `warp_stall_unit_busy`. Also records one committed `SchedulerIssueOutcome` per warp each cycle for trace attribution.
- **`set_opcoll_free(bool)`**, **`set_unit_ready_fn(fn)`**: External readiness inputs.
- **`current_diagnostics()`**: Returns the committed per-warp `SchedulerIssueOutcome` array.
- **Outputs**: `output()` (next), `current_output()` (committed). Struct `IssueOutput`: `decoded`, `trace`, `warp_id`, `pc`, `prediction`.

### `include/gpu_sim/timing/operand_collector.h` -- `src/timing/operand_collector.cpp`

Models operand read timing (no actual data movement -- values are in TraceEvent).

- **`OperandCollector(stats_ref)`**
- **`accept(IssueOutput)`**: Sets `cycles_remaining` to 1 (2-operand) or 2 (VDOT8/3-operand).
- **`evaluate()`**: Decrements counter. When 0, produces output.
- **`is_free()`**: True when not busy.
- **Snapshot helpers**: `busy()`, `cycles_remaining()`, `resident_warp()`, `current_instruction()`.
- **Outputs**: Struct `DispatchInput`: `decoded`, `trace`, `warp_id`, `pc`, `prediction`.

### `include/gpu_sim/timing/alu_unit.h` -- `src/timing/alu_unit.cpp`

1-cycle ALU execution unit.

- **`ALUUnit(stats_ref)`**
- **`accept(DispatchInput, cycle)`**: Stores pending input.
- **`evaluate()`**: Produces result in `result_buffer_` in 1 cycle.
- **`is_ready()`**: True when result buffer is empty and no pending input.
- **`has_result()`** / **`consume_result()`**: Writeback interface.
- **Snapshot helpers**: `busy()`, `active_warp()`, `pending_input()`, `result_entry()`.

### `include/gpu_sim/timing/multiply_unit.h` -- `src/timing/multiply_unit.cpp`

Pipelined multiply/VDOT8 unit with configurable depth.

- **`MultiplyUnit(pipeline_stages, stats_ref)`**
- **`accept()`**: Pushes entry into pipeline shift register with `pipeline_stages` cycles remaining.
- **`evaluate()`**: Decrements pipeline entries toward completion. If the head is ready but `result_buffer_` is occupied, the ready head is held at 0 until writeback consumes the buffer, then it resumes cleanly instead of underflowing.
- **`is_ready()`**: True when result buffer is empty (can accept into pipeline every cycle).
- **Snapshot helpers**: `pipeline_occupancy()`, `pipeline_snapshot()`, `result_entry()`.

### `include/gpu_sim/timing/divide_unit.h` -- `src/timing/divide_unit.cpp`

Iterative divide unit, 32-cycle latency.

- **`DivideUnit(stats_ref)`**. Constant `DIVIDE_LATENCY = 32`.
- **`accept()`**: Starts countdown. Busy until complete.
- **`is_ready()`**: True when not busy and result buffer empty.
- **Snapshot helpers**: `busy()`, `cycles_remaining()`, `pending_entry()`, `result_entry()`.

### `include/gpu_sim/timing/tlookup_unit.h` -- `src/timing/tlookup_unit.cpp`

Pipelined dual-port BRAM table lookup, 17-cycle latency (2 lanes/cycle, ceil(32/2)+1 = 17 cycles).

- **`TLookupUnit(stats_ref)`**. Constant `TLOOKUP_LATENCY = 17`.
- Same interface pattern as DivideUnit, plus snapshot helpers `busy()`, `cycles_remaining()`, `pending_entry()`, `result_entry()`.

### `include/gpu_sim/timing/ldst_unit.h` -- `src/timing/ldst_unit.cpp`

Address generation unit with output FIFO.

- **Struct `AddrGenFIFOEntry`**: `valid`, `warp_id`, `dest_reg`, `is_load`, `is_store`, `trace`, `issue_cycle`.
- **`LdStUnit(num_ldst_units, fifo_depth, stats_ref)`**
- **`accept()`**: Begins address generation. Latency = `ceil(32 / num_ldst_units)` cycles.
- **`evaluate()`**: When address gen completes, pushes entry to FIFO. If the FIFO is full, the completed entry is held at 0 cycles remaining until space opens instead of underflowing and disappearing.
- **`fifo_empty()`**, **`fifo_front()`**, **`fifo_pop()`**: Interface consumed by `CoalescingUnit`.
- **`has_result()`**: Always false -- LD/ST results flow through cache/MSHR fill path, not through the execution unit result buffer.
- **Snapshot helpers**: `busy()`, `cycles_remaining()`, `pending_entry()`, `fifo_entries()`.

### `include/gpu_sim/timing/writeback_arbiter.h` -- `src/timing/writeback_arbiter.cpp`

Round-robin writeback arbitration among execution units and queued memory-result sources.

- **`WritebackArbiter(scoreboard_ref, stats_ref)`**
- **`add_source(ExecutionUnit*)`**: Registers ALU, MUL, DIV, TLOOKUP, and queued LD/ST hit/fill writeback sources.
- **`evaluate()`**: Round-robin scans all registered sources. First with valid result wins. Calls `scoreboard_.clear_pending(warp, reg)`. Counts conflicts when multiple sources ready.
- **`committed_entry()`**: The writeback that happened this cycle (for trace/stats).
- **`has_pending_work()`**: Reports queued writeback work so DONE/PANIC waits for the writeback path to drain.
- **Snapshot helpers**: `ready_source_count()`.

`include/gpu_sim/timing/execution_unit.h` also defines **`QueuedWritebackSource`**, a tiny FIFO-backed `ExecutionUnit` implementation retained for tests and generic writeback-source scenarios. The LD/ST writeback source in production wiring is the `LoadGatherBufferFile` (see below).

### `include/gpu_sim/timing/cache.h` -- `src/timing/cache.cpp`

Direct-mapped L1 data cache with MSHRs and write buffer.

This timing model intentionally tracks cache residency, misses, backpressure, and writeback-source timing without storing full cache-line payloads. Load values are replayed from the functional-model trace; the cache exists here to model performance behavior, not data correctness.

- **Enum `CacheStallReason`**: `NONE`, `MSHR_FULL`, `WRITE_BUFFER_FULL`, `LINE_PINNED`.
- **`L1Cache(cache_size, line_size, num_mshrs, write_buffer_depth, mem_if_ref, gather_file_ref, stats_ref)`**
- **`process_load(addr, warp_id, lane_mask, results, issue_cycle, pc, raw_instruction)`**: Hit -> attempts to write the lanes selected by `lane_mask` into the owning warp's gather buffer; returns false when the gather-buffer write port was already used this cycle (caller must retry). Miss -> allocates an MSHR recording `lane_mask`, submits the read, and records a miss-allocation trace event. Returns false if MSHR full (stall). Does not produce a writeback directly.
- **`process_store(line_addr, warp_id, issue_cycle, pc, raw_instruction)`**: Hit -> updates cache, pushes to write buffer. Miss -> allocates MSHR (write-allocate) with trace metadata. Returns false if MSHR or write buffer full.
- **`handle_responses()`**: Processes at most one readable cache-line fill per cycle. For load fills, deposits the lane values into the owning warp's gather buffer via the FILL write-port path (FILL wins a same-cycle HIT). For store fills, pushes the line into the write buffer. Responses are buffered internally if a store fill is blocked by the write buffer.
- **`drain_write_buffer()`**: Submits the front of the write buffer to external memory, popping only on success. The bool return from `mem_if_.submit_write` is the architectural backpressure path from the memory backend: `DRAMSim3Memory` returns `false` when its bounded write region is full, in which case the entry stays at the buffer's head and is retried next cycle. Silently popping on failure would lose the write (the timing-only model has no recovery).
- **`evaluate()`**: Clears one-cycle cache backpressure, resets the per-cycle gather-extract-port flag, runs `handle_responses()` so FILL writes to the gather buffer occur before any HIT writes later in the cycle, then runs `drain_secondary_chain_head()` (priority FILL > secondary > HIT).
- **Same-line MSHR merging**: Each `MSHREntry` carries `next_in_chain` and `is_secondary`. On a miss, the cache scans MSHRs for a same-line entry; if found, the new MSHR is linked as a secondary and does not submit an external read. A primary fill installs the line, sets the tag's `pinned` bit when a chain follows, and subsequent cycles drain the chain head one per cycle via `drain_secondary_chain_head()` (loads extract into the owning warp's gather buffer; stores push to the write buffer). The pin clears when the last secondary retires. Pin conflicts (different-tag miss into a pinned set) stall with `LINE_PINNED`.
- **`is_idle()`**: True only when there are no live MSHRs, queued write-through entries, or pending fills.
- **Snapshot helpers**: `stall_reason()`, `active_mshr_count()`, `write_buffer_size()`, `pending_fill()`, `mshrs()`, `last_miss_event()`, `last_fill_event()`.
- **Indexing**: `set = (addr / line_size) % num_sets`, `tag = addr / line_size / num_sets`.

### `include/gpu_sim/timing/load_gather_buffer.h` -- `src/timing/load_gather_buffer.cpp`

Per-resident-warp load gather buffers — the sole assembly point for load data before writeback. See §5.2.1 of the architectural spec.

- **Struct `LoadGatherBuffer`**: One per resident warp. Fields: `busy`, `dest_reg`, `values[32]`, `slot_valid[32]`, `filled_count`, `pc`, `issue_cycle`, `raw_instruction`, plus a `port_used_this_cycle` scratch bit cleared every `commit()`.
- **Class `LoadGatherBufferFile : public ExecutionUnit`**: Allocates `num_warps` buffers indexed by `warp_id` and registers with the writeback arbiter.
  - **`is_busy(warp_id)`**: Query used by the coalescing unit to gate claiming a load.
  - **`claim(warp_id, dest_reg, pc, issue_cycle, raw_instruction)`**: Marks the buffer busy and stamps metadata.
  - **`try_write(warp_id, lane_mask, values, source)`** where `source` is `HIT` or `FILL`: Writes the selected lanes into the buffer and increments `filled_count`. Returns false iff the buffer's single write port was already used this cycle; a HIT losing to a same-cycle FILL increments `stats_.gather_buffer_port_conflict_cycles`.
  - **`has_result()` / `consume_result()`**: Raises and consumes a full 32-lane writeback when any buffer's `filled_count` reaches 32, using round-robin selection across buffers. Consuming releases the buffer.

### `include/gpu_sim/timing/mshr.h` -- `src/timing/mshr.cpp`

Miss Status Holding Registers.

- **Struct `MSHREntry`**: `valid`, `cache_line_addr`, `is_store`, `warp_id`, `dest_reg`, `pc`, `raw_instruction`, `issue_cycle`, per-lane arrays (`mem_addresses`, `store_data`, `mem_size`, `results`), `lane_mask` (for load misses: lanes of the owning warp waiting on this fill; the cache deposits these lanes into the warp's gather buffer on fill), plus same-line merging fields `next_in_chain` (index or `INVALID_MSHR`) and `is_secondary`.
- **Class `MSHRFile`**: Vector of entries. `allocate(entry)` -> index or -1. `free(index)`. `has_free()`. `has_active()`. `at(index)`. `find_chain_tail(line_addr)` -> index of the chain tail for `line_addr` (entry with `next_in_chain == INVALID_MSHR`) or -1 if no MSHR currently holds this line.

### `include/gpu_sim/timing/coalescing_unit.h` -- `src/timing/coalescing_unit.cpp`

All-or-nothing address coalescing.

- **`CoalescingUnit(ldst_ref, cache_ref, gather_file_ref, line_size, stats_ref)`**
- **`evaluate()`**: Pulls entry from LD/ST FIFO. For loads, stalls (without popping) if the target warp's gather buffer is busy, incrementing `stats_.gather_buffer_stall_cycles`; once the buffer is free, claims it before issuing cache transactions. Checks if all 32 thread addresses fall in the same cache line. If yes: one cache request with `lane_mask = 0xFFFFFFFF`. If no: 32 serialized single-lane requests with one-hot `lane_mask`, one per cycle. Stalls if cache cannot accept. Never produces a writeback — the gather buffer emits it once all 32 slots are valid.
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
  (`evaluate`, `commit`, `reset`, `submit_read`, `submit_write`,
  `has_response`, `get_response`, `is_idle`, `in_flight_count`,
  `response_count`). Concrete backends derive from this.
- **`FixedLatencyMemory(latency, stats_ref)`**: Default backend. Every
  request completes after exactly `latency` cycles. Used by all unit tests
  and by the simulator unless a different backend is selected.
  - `submit_read(line_addr, mshr_id)`, `submit_write(line_addr)`: Enqueue
    request with `latency` countdown.
  - `evaluate()`: Decrements all in-flight countdowns. Moves completed
    requests to response queue.
  - `has_response()`, `get_response()`: Response consumption interface.
  - `is_idle()`: True only when no requests are in flight and no responses
    are queued.
  - Snapshot helpers: `in_flight_count()`, `response_count()`.

### `include/gpu_sim/timing/dramsim3_memory.h` -- `src/timing/dramsim3_memory.cpp`

DRAMSim3-backed external memory model. Selected when
`SimConfig::memory_backend == "dramsim3"`. Implements the abstract
`ExternalMemoryInterface` so the cache call sites are unchanged.

- **`DRAMSim3Memory(SimConfig, Stats&)`**: Loads the DRAMSim3 `.ini` at
  `cfg.dramsim3_config_path`, sets `chunks_per_line = cache_line_size_bytes
  / dramsim3_bytes_per_burst`, sizes the per-MSHR read-assembly array to
  `cfg.num_mshrs`, pre-creates `cfg.dramsim3_output_dir` so DRAMSim3 doesn't
  emit its "WARNING: Output directory ... not exists!" line on stdout, and
  binds the DRAMSim3 read/write callbacks to its own reassembly routines.
- **Asynchronous fabric/DRAM clocks**: a phase accumulator advances DRAMSim3
  by `dram_clock_mhz / fpga_clock_mhz` ticks per fabric `evaluate()`,
  handling non-integer clock ratios without long-run drift.
- **Bounded request FIFO with reserved regions** (`dramsim3_request_fifo_depth`): logically split into a `num_mshrs * chunks_per_line` read region and a `write_buffer_depth * chunks_per_line` write region. `submit_read` only fails if the entire FIFO is full (architecturally impossible — at most `num_mshrs` reads can be in flight, and the read region holds exactly that many). `submit_write` returns `false` once the write region is full; the cache (`L1Cache::drain_write_buffer`) consumes this as backpressure and retries next cycle. `assert`s at both push sites convert any violation into an immediate failure.
- **Bounded response queue** (capacity = `num_mshrs + write_buffer_depth + chunks_per_line`): asserted at both push sites. The `worst-case cache traffic` Catch2 case in `test_dramsim3_memory.cpp` drives peak production and verifies the bound holds (snapshot via `max_observed_response_queue()`).
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
- **`trigger(warp_id, pc)`**: Activates the decode-detected panic sequence.
- **`evaluate()`**: After decode triggers panic, the first controller tick reads `r31` lane 0 and latches panic diagnostics into `FunctionalModel`. Later cycles drain execution units and writeback (bounded by `MAX_DRAIN_CYCLES = 32`) before marking all warps inactive and setting `done_`. In-flight memory requests are abandoned for architectural purposes and are not part of the drain criterion.
- **`set_units_drained(bool)`**: Called by timing model each cycle with `execution_units_drained()`.
- **`is_active()`**, **`is_done()`**: State queries.
- **Snapshot helpers**: `step()`, `panic_warp()`, `panic_pc()`, `panic_cause()`.

### `include/gpu_sim/timing/timing_model.h` -- `src/timing/timing_model.cpp`

Top-level cycle stepper wiring everything together.

- **`TimingModel(config, func_model_ref, stats_ref, trace_options)`**: Constructs and wires all sub-components. Selects the external-memory backend by branching on `config.memory_backend`: `"dramsim3"` constructs `DRAMSim3Memory`, anything else (default `"fixed"`) constructs `FixedLatencyMemory`. Sets up scheduler's unit readiness callback. When `trace_options.output_path` is non-empty, opens a `ChromeTraceWriter` and registers warp/hardware/counter tracks.
- **`tick()`** -> `bool` (continue?): One cycle of simulation. Forward-order evaluation:
  1. `scoreboard_.seed_next()` and `cache_.evaluate()`
  2. `fetch_` -> `decode_` -> EBREAK check -> `scheduler_` -> `opcoll_` -> dispatch -> execute (all units) -> `coalescing_` -> `mem_if_` -> MSHR fill -> write buffer drain -> `wb_arbiter_`
  3. All `.commit()`
  4. Termination check: `all_warps_done() && pipeline_drained()`
- **`run(max_cycles)`**: Calls `tick()` in a loop.
- **`dispatch_to_unit(DispatchInput)`**: Routes to appropriate execution unit. ECALL (via SYSTEM case) marks warp inactive. CSR reads are routed to ALU by the decoder (`target_unit = ExecUnit::ALU`), so they match the ALU case directly.
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
| `test_decoder.cpp` | 27 | Every RV32IM + VDOT8 + TLOOKUP + ECALL/EBREAK/CSR encoding, including unsupported CSR/FENCE forms. |
| `test_alu.cpp` | 30 | All ALU ops, MUL/DIV edge cases (overflow, div-by-zero), VDOT8 byte patterns, branch conditions. |
| `test_functional.cpp` | 16 | End-to-end functional model: ADDI chains, x0 discard, load/store, branches, ECALL/EBREAK, CSR, VDOT8, TLOOKUP, kernel args, multi-warp independence. |
| `test_scoreboard.cpp` | 9 | r0 never pending, set/clear, double-buffer isolation, same-cycle set+clear, multi-warp independence, seed_next, all-registers, reset. |
| `test_cache.cpp` | 10 | Load miss-then-hit, transient MSHR retry, store hit write buffer, store miss write-allocate, store-fill write-buffer backpressure, replayed multiple fills, direct-mapped eviction, write buffer full, reset. |
| `test_coalescing.cpp` | 3 | Contiguous addresses coalesce (1 request), scattered addresses serialize (32), boundary case (1 lane different). |
| `test_warp_scheduler.cpp` | 10 | Issues from buffer, skips empty, scoreboard stall, VDOT8 rd-as-source hazard, opcoll-busy stall, unit-busy stall, RR fairness, committed scheduler diagnostics, scoreboard-pending-on-issue, idle pointer advance. |
| `test_branch.cpp` | 6 | Forward-taken mispredict recovery, non-taken fall-through, backward-loop prediction accuracy, shadow-instruction-no-commit after mispredict, taken-vs-straight-line penalty comparison, fetch decode-FIFO-full backpressure. |
| `test_panic.cpp` | 5 | EBREAK halts simulation, multi-warp EBREAK, state machine step-by-step progression, panic writeback freeze, reset. |
| `test_integration.cpp` | 23 | Full timing model end-to-end: ADD chain, independent ADDIs, RAW chain, load-use stall, store-then-load, write-through completion drain, branch loop, JAL, multi-warp CSR, multi-warp ECALL, memory coalescing, LUI+ADDI, MUL, MUL-latency-vs-ALU, VDOT8, TLOOKUP, EBREAK, stats collection, x0 discard, max-cycles limit, trace snapshot classification, trace-file smoke coverage. |
| `test_timing_components.cpp` | 26 | Fetch skips full-buffer warps, fetch-decode backpressure stall, fetch PC steering from the static predictor, static branch predictor decisions, operand collection latency, ALU/MUL/DIV/TLOOKUP timing (busy signal, cycles_remaining countdown, is_ready lifecycle, back-to-back dispatch, stats tracking, reset, writeback metadata, accessor lifecycle), LD/ST FIFO backpressure, memory-interface ordering, writeback arbitration, simultaneous queued memory writebacks. |
| `test_config.cpp` | 8 | Default values for memory-backend knobs, default request FIFO depth equals (num_mshrs + write_buffer_depth) * chunks_per_line worst case, JSON parse round-trip for all DRAMSim3 fields, validation rejects unknown `memory_backend`, validation rejects non-positive clocks, validation rejects line size not a multiple of burst, validation rejects undersized DRAMSim3 request FIFO, CLI override parses string/double/uint fields. |
| `test_dramsim3_memory.cpp` | 8 | DE-10 Nano `.ini` loads, single-read chunk reassembly emits exactly one `MemoryResponse` after at least 16 DRAM ticks, eight reads with distinct mshr_ids each return once, sequential reads drain in fewer cycles than thrashing-strided reads, fabric/DRAM phase accumulator hits ~2667 DRAM ticks per 1000 fabric cycles at 150/400 MHz, `is_idle()` and `reset()` clear all state and the backend remains usable afterward, worst-case cache traffic (peak read+write production with bool-respecting drain) drops no requests and keeps the response queue within capacity, write-region saturation propagates from DRAMSim3 through the cache: `submit_write` rejection causes `drain_write_buffer` to leave entries in place, `write_buffer_` fills, `process_store` stalls with `WRITE_BUFFER_FULL`, `write_buffer_stall_cycles` increments, and no writes are silently lost. |

**Totals**: 181 direct Catch2 cases.

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
- optional workload-level benchmarking through `matmul_bench`

The expected workflow for architectural changes is to add or update targeted Catch2 coverage near the affected component, then run the normal regression suite with:

```
ctest --test-dir build --output-on-failure
```
