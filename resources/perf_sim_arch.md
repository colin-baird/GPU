# GPU Simulator -- File Reference

This document describes every file in the simulator, what it does, and where to find the interfaces it exposes. For the full architectural specification, see [`/resources/gpu_architectural_spec.md`](/resources/gpu_architectural_spec.md).

---

## Build System

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Top-level build. Builds `gpu_sim_lib` (static library from all src/), `gpu_sim` (executable), and optionally test targets. C++17. |
| `tests/CMakeLists.txt` | Registers each `test_*.cpp` as a Catch2 test executable linked against `gpu_sim_lib`. |

Build: `cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j8`

---

## Core / Foundation

### `include/gpu_sim/types.h`

Central type definitions used everywhere. No implementation file.

- **Type aliases**: `WarpId`, `LaneId`, `RegIndex`, `Address`, `Word` (all `uint32_t`); `WarpData` (`std::array<Word, 32>`)
- **Enums**: `ExecUnit` (ALU, MULTIPLY, DIVIDE, LDST, TLOOKUP, SYSTEM, NONE), `InstructionType` (17 variants), `AluOp`, `MulDivOp`, `MemOp`, `BranchOp`
- **Constants**: `WARP_SIZE = 32`, `MAX_WARPS = 8`, `NUM_REGS = 32`

### `include/gpu_sim/config.h` -- `src/config.cpp`

Simulator configuration. All parameterizable values live in `SimConfig`.

- **Struct `SimConfig`**: `num_warps`, `instruction_mem_size_bytes`, `instruction_buffer_depth`, `multiply_pipeline_stages`, `num_ldst_units`, `addr_gen_fifo_depth`, `l1_cache_size_bytes`, `cache_line_size_bytes`, `num_mshrs`, `write_buffer_depth`, `lookup_table_entries`, `external_memory_latency_cycles`, `external_memory_size_bytes`, `kernel_args[4]`, `start_pc`, `trace_enabled`, `functional_only`
- **`validate()`**: Checks constraints (warp count in [1,8], cache size power-of-2, etc.). Throws on failure.
- **`from_json(path)`**: Minimal hand-rolled JSON parser (key-value pairs).
- **`apply_cli_overrides(argc, argv)`**: Parses `--key=value` CLI arguments.

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

- **`Decoder::decode(uint32_t instruction)`** (static): Returns `DecodedInstruction`. Large switch on opcode bits [6:0], sub-switched on funct3/funct7. Sets `target_unit`, `num_src_regs`, `has_rd`, `reads_rd` (true only for VDOT8).

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

- **Struct `Stats`**: All counters default to 0. Categories:
  - Global: `total_cycles`, `total_instructions_issued`
  - Per-warp: `warp_instructions[8]`, `warp_cycles_active[8]`, `warp_stall_scoreboard[8]`, `warp_stall_buffer_empty[8]`, `warp_stall_unit_busy[8]`
  - Pipeline: `fetch_skip_count`, `scheduler_idle_cycles`, `operand_collector_busy_cycles`, `branch_flushes`
  - Per-unit (`UnitStats`): `busy_cycles`, `instructions` -- for ALU, MUL, DIV, LD/ST, TLOOKUP
  - Memory: `cache_hits/misses`, `load_hits/misses`, `store_hits/misses`, `mshr_stall_cycles`, `write_buffer_stall_cycles`, `coalesced_requests`, `serialized_requests`, `external_memory_reads/writes`, `total_load_latency`, `total_loads_completed`
  - Writeback: `writeback_conflicts`
- **`report(ostream, num_warps)`**: Human-readable text summary.
- **`report_json(ostream, num_warps)`**: Machine-parseable JSON.

### `include/gpu_sim/backend.h` -- `src/backend.cpp`

Backend routing system. Decouples program loading from execution so multiple backends can consume the same `ProgramImage`.

- **Class `Backend`** (abstract): `run(image, config, argc, argv)` -> exit code, `name()` -> string.
- **`create_backend(name)`** -> `unique_ptr<Backend>`. Factory function; returns `nullptr` for unknown names. Currently supports `"perf_sim"`.

### `include/gpu_sim/backends/perf_sim_backend.h` -- `src/backends/perf_sim_backend.cpp`

Performance simulator backend. Wraps `FunctionalModel` and `TimingModel`.

- **Class `PerfSimBackend`**: Implements `Backend`. Loads `ProgramImage` into a `FunctionalModel`, handles `--lookup-table`, `--data`, `--json`, `--max-cycles` CLI options, runs functional-only or timing simulation, reports stats and register state.

### `src/main.cpp`

Entry point. Parses `--backend=<name>` (default: `perf_sim`), loads config, calls `load_program_image()`, creates and runs the selected backend. See `--help` for usage.

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

- **Struct `WritebackEntry`**: `valid`, `warp_id`, `dest_reg`, `values[WARP_SIZE]`, `source_unit`, `issue_cycle`.
- **Class `ExecutionUnit`**: Extends pipeline stage interface with `is_ready()`, `has_result()`, `consume_result()`, `get_type()`.

### `include/gpu_sim/timing/warp_state.h`

Per-warp timing state. Header-only.

- **Struct `WarpState`**: `pc`, `active`, `instr_buffer` (InstructionBuffer). `reset(start_pc)` sets active and clears buffer.

### `include/gpu_sim/timing/instruction_buffer.h`

Per-warp instruction FIFO. Header-only.

- **Struct `BufferEntry`**: `decoded` (DecodedInstruction), `warp_id`, `pc`.
- **Class `InstructionBuffer`**: Deque-backed FIFO with fixed `max_depth`. Methods: `is_full()`, `is_empty()`, `size()`, `push()`, `pop()`, `front()`, `flush()`, `reset()`.

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

- **`FetchStage(num_warps, warps_ptr, imem_ref, stats_ref)`**
- **`evaluate()`**: Selects warp via `rr_pointer`, skips if inactive or buffer full. Reads instruction from `InstructionMemory`, increments warp PC by 4. Pointer advances unconditionally.
- **`redirect_warp(warp_id, target_pc)`**: Sets warp PC, flushes instruction buffer, invalidates any pending fetch for that warp.
- **Outputs**: `output()` (next), `current_output()` (committed). Struct `FetchOutput`: `raw_instruction`, `warp_id`, `pc`.

### `include/gpu_sim/timing/decode_stage.h` -- `src/timing/decode_stage.cpp`

Instruction decode with EBREAK detection.

- **`DecodeStage(warps_ptr, fetch_ref)`**
- **`evaluate()`**: Reads `fetch_.current_output()`. Decodes via `Decoder::decode()`. If EBREAK, signals panic and returns. Otherwise stages decoded instruction for commit.
- **`commit()`**: Pushes staged instruction to target warp's instruction buffer.
- **`invalidate_warp(warp_id)`**: Clears any pending decode for that warp (branch redirect).
- **`ebreak_detected()`**, **`ebreak_warp()`**, **`ebreak_pc()`**: Panic detection interface.

### `include/gpu_sim/timing/warp_scheduler.h` -- `src/timing/warp_scheduler.cpp`

Issue stage with round-robin scheduling and 4-way eligibility check.

- **`WarpScheduler(num_warps, warps_ptr, scoreboard_ref, func_model_ref, stats_ref)`**
- **`evaluate()`**: Scans warps starting from `rr_pointer`. For each, checks: (1) buffer not empty, (2) scoreboard clear for source registers, (3) operand collector free, (4) target execution unit ready. First eligible warp wins. Calls `func_model_.execute(warp_id, pc)` to produce TraceEvent. Sets scoreboard pending for `rd`. Pointer advances unconditionally.
- **`set_opcoll_free(bool)`**, **`set_unit_ready_fn(fn)`**: External readiness inputs.
- **Outputs**: `output()` (next), `current_output()` (committed). Struct `IssueOutput`: `decoded`, `trace`, `warp_id`, `pc`.

### `include/gpu_sim/timing/operand_collector.h` -- `src/timing/operand_collector.cpp`

Models operand read timing (no actual data movement -- values are in TraceEvent).

- **`OperandCollector(stats_ref)`**
- **`accept(IssueOutput)`**: Sets `cycles_remaining` to 1 (2-operand) or 2 (VDOT8/3-operand).
- **`evaluate()`**: Decrements counter. When 0, produces output.
- **`is_free()`**: True when not busy.
- **Outputs**: Struct `DispatchInput`: `decoded`, `trace`, `warp_id`, `pc`.

### `include/gpu_sim/timing/alu_unit.h` -- `src/timing/alu_unit.cpp`

1-cycle ALU execution unit.

- **`ALUUnit(stats_ref)`**
- **`accept(DispatchInput, cycle)`**: Stores pending input.
- **`evaluate()`**: Produces result in `result_buffer_` in 1 cycle.
- **`is_ready()`**: True when result buffer is empty and no pending input.
- **`has_result()`** / **`consume_result()`**: Writeback interface.

### `include/gpu_sim/timing/multiply_unit.h` -- `src/timing/multiply_unit.cpp`

Pipelined multiply/VDOT8 unit with configurable depth.

- **`MultiplyUnit(pipeline_stages, stats_ref)`**
- **`accept()`**: Pushes entry into pipeline shift register with `pipeline_stages` cycles remaining.
- **`evaluate()`**: Decrements all pipeline entries. Head with 0 remaining moves to `result_buffer_` (structural hazard if occupied -- pipeline stalls).
- **`is_ready()`**: True when result buffer is empty (can accept into pipeline every cycle).

### `include/gpu_sim/timing/divide_unit.h` -- `src/timing/divide_unit.cpp`

Iterative divide unit, 32-cycle latency.

- **`DivideUnit(stats_ref)`**. Constant `DIVIDE_LATENCY = 32`.
- **`accept()`**: Starts countdown. Busy until complete.
- **`is_ready()`**: True when not busy and result buffer empty.

### `include/gpu_sim/timing/tlookup_unit.h` -- `src/timing/tlookup_unit.cpp`

Serial table lookup, 64-cycle latency (2 cycles/lane x 32 lanes).

- **`TLookupUnit(stats_ref)`**. Constant `TLOOKUP_LATENCY = 64`.
- Same interface pattern as DivideUnit.

### `include/gpu_sim/timing/ldst_unit.h` -- `src/timing/ldst_unit.cpp`

Address generation unit with output FIFO.

- **Struct `AddrGenFIFOEntry`**: `valid`, `warp_id`, `dest_reg`, `is_load`, `is_store`, `trace`, `issue_cycle`.
- **`LdStUnit(num_ldst_units, fifo_depth, stats_ref)`**
- **`accept()`**: Begins address generation. Latency = `ceil(32 / num_ldst_units)` cycles.
- **`evaluate()`**: When address gen completes, pushes entry to FIFO (stalls if FIFO full).
- **`fifo_empty()`**, **`fifo_front()`**, **`fifo_pop()`**: Interface consumed by `CoalescingUnit`.
- **`has_result()`**: Always false -- LD/ST results flow through cache/MSHR fill path, not through the execution unit result buffer.

### `include/gpu_sim/timing/writeback_arbiter.h` -- `src/timing/writeback_arbiter.cpp`

Round-robin writeback arbitration among execution units and MSHR fills.

- **`WritebackArbiter(scoreboard_ref, stats_ref)`**
- **`add_source(ExecutionUnit*)`**: Registers ALU, MUL, DIV, TLOOKUP as writeback sources.
- **`submit_fill(WritebackEntry)`**: Accepts cache hit or MSHR fill results into a dedicated fill buffer.
- **`evaluate()`**: Round-robin scans sources + fill buffer. First with valid result wins. Calls `scoreboard_.clear_pending(warp, reg)`. Counts conflicts when multiple sources ready.
- **`committed_entry()`**: The writeback that happened this cycle (for trace/stats).

### `include/gpu_sim/timing/cache.h` -- `src/timing/cache.cpp`

Direct-mapped L1 data cache with MSHRs and write buffer.

- **`L1Cache(cache_size, line_size, num_mshrs, write_buffer_depth, mem_if_ref, stats_ref)`**
- **`process_load(addr, warp_id, dest_reg, results, issue_cycle, wb_out)`**: Hit -> fills `wb_out` immediately. Miss -> allocates MSHR, submits read to external memory. Returns false if MSHR full (stall).
- **`process_store(line_addr)`**: Hit -> updates cache, pushes to write buffer. Miss -> allocates MSHR (write-allocate). Returns false if MSHR or write buffer full.
- **`handle_responses(wb_out, wb_valid)`**: Processes external memory responses. Installs cache lines. For load misses: produces writeback entry. For store misses: installs line and pushes to write buffer.
- **`drain_write_buffer()`**: Submits one write-buffer entry per cycle to external memory.
- **Indexing**: `set = (addr / line_size) % num_sets`, `tag = addr / line_size / num_sets`.

### `include/gpu_sim/timing/mshr.h` -- `src/timing/mshr.cpp`

Miss Status Holding Registers.

- **Struct `MSHREntry`**: `valid`, `cache_line_addr`, `is_store`, `warp_id`, `dest_reg`, `issue_cycle`, per-lane arrays (`mem_addresses`, `store_data`, `mem_size`, `results`).
- **Class `MSHRFile`**: Vector of entries. `allocate(entry)` -> index or -1. `free(index)`. `has_free()`. `at(index)`.

### `include/gpu_sim/timing/coalescing_unit.h` -- `src/timing/coalescing_unit.cpp`

All-or-nothing address coalescing.

- **`CoalescingUnit(ldst_ref, cache_ref, line_size, stats_ref)`**
- **`evaluate(wb_out, wb_valid)`**: Pulls entry from LD/ST FIFO. Checks if all 32 thread addresses fall in the same cache line. If yes: 1 cache request. If no: serializes to 32 individual requests (1 per cycle). Stalls if cache cannot accept.

### `include/gpu_sim/timing/memory_interface.h` -- `src/timing/memory_interface.cpp`

Fixed-latency external memory model.

- **Struct `MemoryRequest`**: `line_addr`, `mshr_id`, `is_write`, `cycles_remaining`.
- **Struct `MemoryResponse`**: `line_addr`, `mshr_id`, `is_write`.
- **`ExternalMemoryInterface(latency, stats_ref)`**
- **`submit_read(line_addr, mshr_id)`**, **`submit_write(line_addr)`**: Enqueue request with `latency` countdown.
- **`evaluate()`**: Decrements all in-flight countdowns. Moves completed requests to response queue.
- **`has_response()`**, **`get_response()`**: Response consumption interface.

### `include/gpu_sim/timing/panic_controller.h` -- `src/timing/panic_controller.cpp`

EBREAK state machine.

- **`PanicController(num_warps, warps_ptr, func_model_ref)`**
- **`trigger(warp_id, pc)`**: Activates the 5-step panic sequence.
- **`evaluate()`**: Steps 0-2: diagnostic latching (1 cycle each). Step 3: drain in-flight instructions (waits for `units_drained_` or `MAX_DRAIN_CYCLES = 64`). Step 4: marks all warps inactive, sets `done_`.
- **`set_units_drained(bool)`**: Called by timing model each cycle with `all_units_idle()`.
- **`is_active()`**, **`is_done()`**: State queries.

### `include/gpu_sim/timing/timing_model.h` -- `src/timing/timing_model.cpp`

Top-level cycle stepper wiring everything together.

- **`TimingModel(config, func_model_ref, stats_ref)`**: Constructs and wires all sub-components. Sets up scheduler's unit readiness callback.
- **`tick()`** -> `bool` (continue?): One cycle of simulation. Forward-order evaluation:
  1. `scoreboard_.seed_next()`
  2. `fetch_` -> `decode_` -> EBREAK check -> `scheduler_` -> `opcoll_` -> dispatch -> execute (all units) -> `coalescing_` -> `mem_if_` -> MSHR fill -> write buffer drain -> `wb_arbiter_`
  3. All `.commit()`
  4. Termination check: `all_warps_done() && all_units_idle()`
- **`run(max_cycles)`**: Calls `tick()` in a loop.
- **`dispatch_to_unit(DispatchInput)`**: Routes to appropriate execution unit. ECALL marks warp inactive. CSR reads routed through ALU.
- **`trace_cycle()`**: Prints per-cycle pipeline state to stderr when `--trace` enabled.

---

## Tests

All tests use Catch2 v2.13.10 (single-header at `tests/vendor/catch.hpp`). Run from the build directory: `./tests/test_<name>`.

| File | Cases | Focus |
|------|-------|-------|
| `test_decoder.cpp` | 24 | Every RV32IM + VDOT8 + TLOOKUP + ECALL/EBREAK/CSR encoding. |
| `test_alu.cpp` | 30 | All ALU ops, MUL/DIV edge cases (overflow, div-by-zero), VDOT8 byte patterns, branch conditions. |
| `test_functional.cpp` | 16 | End-to-end functional model: ADDI chains, x0 discard, load/store, branches, ECALL/EBREAK, CSR, VDOT8, TLOOKUP, kernel args, multi-warp independence. |
| `test_scoreboard.cpp` | 9 | r0 never pending, set/clear, double-buffer isolation, same-cycle set+clear, multi-warp independence, seed_next, all-registers, reset. |
| `test_cache.cpp` | 7 | Load miss-then-hit, store hit write buffer, store miss write-allocate, MSHR stall, direct-mapped eviction, write buffer full, reset. |
| `test_coalescing.cpp` | 3 | Contiguous addresses coalesce (1 request), scattered addresses serialize (32), boundary case (1 lane different). |
| `test_warp_scheduler.cpp` | 8 | Issues from buffer, skips empty, scoreboard stall, opcoll-busy stall, unit-busy stall, RR fairness, scoreboard-pending-on-issue, idle pointer advance. |
| `test_branch.cpp` | 4 | Taken branch redirect+flush, not-taken no penalty, loop iteration counting, taken vs straight-line cycle comparison. |
| `test_panic.cpp` | 4 | EBREAK halts simulation, multi-warp EBREAK, state machine step-by-step progression, reset. |
| `test_integration.cpp` | 19 | Full timing model end-to-end: ADD chain, independent ADDIs, RAW chain, load-use stall, store-then-load, branch loop, JAL, multi-warp CSR, multi-warp ECALL, memory coalescing, LUI+ADDI, MUL, MUL-latency-vs-ALU, VDOT8, TLOOKUP, EBREAK, stats collection, x0 discard, max-cycles limit. |

**Totals**: 124 test cases, 833 assertions.

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
  |                                              |    TLOOKUP (64 cyc)
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
