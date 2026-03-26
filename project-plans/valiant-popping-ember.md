# Cycle-Accurate GPU Simulator Plan

## Context

The project has a fully specified GPU architecture (`resources/gpu_architectural_spec.md`) for a single streaming multiprocessor (SM) executing SIMT warps of 32 threads using RV32IM + custom extensions (VDOT8, TLOOKUP), targeting quantized LLM inference. No implementation code exists yet. This plan specifies a C++ cycle-accurate simulator with decoupled functional/timing models, parameterizable configuration, and modular components for architectural experimentation.

---

## 1. Simulator Architecture: Decoupled Functional/Timing Model

**Functional Model**: Executes instructions immediately against flat memory. For every instruction a warp executes, produces a `TraceEvent` containing decoded fields, all 32 threads' computed results, memory addresses, branch outcomes, and register metadata. Never stalls.

**Timing Model**: Consumes `TraceEvent` objects and models exact cycle-by-cycle behavior of every pipeline stage, execution unit, cache, and memory interface. Determines when each instruction enters/exits each stage, when stalls occur, when cache misses resolve. Uses pre-computed results from functional model -- never re-executes arithmetic.

**Interaction Protocol (synchronous pull)**: The timing model drives simulation. When the warp scheduler **issues** an instruction, it calls `functional_model.execute(warp_id, pc)`. The functional model executes across all 32 threads, returns a `TraceEvent`. The timing model moves that event through pipeline stages cycle-by-cycle.

**Why issue-time, not decode-time**: Instructions fetched after a branch may be on the wrong path. If the functional model executed at decode, flushed wrong-path instructions would corrupt functional state. By calling at issue-time, only instructions that pass the scheduler (and thus are on the correct path) get functionally executed. Branch flushes simply discard instruction buffer entries that were never functionally executed.

**Why this design**: The functional model can be tested independently (run programs, check results). The timing model can be tested with synthetic events. Both evolve independently.

---

## 2. Directory Structure

```
sim/
  CMakeLists.txt                         # C++17, no external deps except test framework
  include/gpu_sim/
    config.h                             # SimConfig struct, parameter loading
    types.h                              # WarpId, LaneId, RegIndex, Address, Word typedefs
    trace_event.h                        # TraceEvent struct (interface contract)
    isa.h                                # Opcode enums, instruction format definitions
    decoder.h                            # Decoder class
    functional/
      functional_model.h                 # FunctionalModel class
      register_file.h                    # Per-warp/lane/reg storage
      memory.h                           # FlatMemory (byte-addressable, immediate)
      alu.h                              # Pure functions: ALU/MUL/DIV/VDOT8 compute
    timing/
      timing_model.h                     # Top-level cycle stepper
      pipeline_stage.h                   # PipelineStage abstract base
      fetch_stage.h
      decode_stage.h
      warp_scheduler.h                   # Issue stage
      operand_collector.h
      dispatch_controller.h              # Base + per-unit variants
      execution_unit.h                   # ExecutionUnit abstract base
      alu_unit.h                         # 1-cycle
      multiply_unit.h                    # Pipelined, configurable depth
      divide_unit.h                      # Iterative, ~32 cycles
      ldst_unit.h                        # Address gen + FIFO
      tlookup_unit.h                     # 2 cycles/lane serial drain
      writeback_arbiter.h               # Round-robin
      scoreboard.h                       # Double-buffered pending bits
      cache.h                            # L1 direct-mapped
      mshr.h                             # MSHRFile
      write_buffer.h
      coalescing_unit.h                  # All-or-nothing
      memory_interface.h                 # External memory latency model
      warp_state.h                       # Per-warp PC, active flag, instruction buffer
      instruction_buffer.h               # Per-warp FIFO
      panic_controller.h                 # EBREAK sequence
    stats.h                              # Statistics collection + reporting
    elf_loader.h                         # ELF/raw binary loader
  src/
    main.cpp                             # Entry point, arg parsing, run loop
    config.cpp                           # JSON + CLI config parsing
    decoder.cpp
    elf_loader.cpp
    stats.cpp
    functional/
      functional_model.cpp
      register_file.cpp
      memory.cpp
      alu.cpp
    timing/
      timing_model.cpp                   # Top-level tick() loop
      fetch_stage.cpp
      decode_stage.cpp
      warp_scheduler.cpp
      operand_collector.cpp
      dispatch_controller.cpp
      alu_unit.cpp
      multiply_unit.cpp
      divide_unit.cpp
      ldst_unit.cpp
      tlookup_unit.cpp
      writeback_arbiter.cpp
      scoreboard.cpp
      cache.cpp
      mshr.cpp
      write_buffer.cpp
      coalescing_unit.cpp
      memory_interface.cpp
      warp_state.cpp
      instruction_buffer.cpp
      panic_controller.cpp
  tests/
    CMakeLists.txt
    test_decoder.cpp
    test_functional.cpp
    test_alu.cpp
    test_scoreboard.cpp
    test_cache.cpp
    test_coalescing.cpp
    test_pipeline.cpp
    test_warp_scheduler.cpp
    test_branch.cpp
    test_panic.cpp
    test_integration.cpp
    test_programs/                        # Pre-assembled RISC-V test binaries
```

---

## 3. Configuration System

### SimConfig struct (`config.h`)

```cpp
struct SimConfig {
    // Warp configuration
    uint32_t num_warps = 4;                          // 1-8
    static constexpr uint32_t WARP_SIZE = 32;        // Fixed

    // Instruction memory
    uint32_t instruction_mem_size_bytes = 8192;      // 8 KB = 2048 instructions

    // Pipeline
    uint32_t instruction_buffer_depth = 2;           // Per-warp FIFO depth
    uint32_t multiply_pipeline_stages = 3;           // MUL/VDOT8 latency

    // Load/Store
    uint32_t num_ldst_units = 8;                     // 8-16 address gen units
    uint32_t addr_gen_fifo_depth = 4;                // 4-8 entries

    // Cache
    uint32_t l1_cache_size_bytes = 4096;             // 4-8 KB
    uint32_t cache_line_size_bytes = 128;            // Fixed
    uint32_t num_mshrs = 4;
    uint32_t write_buffer_depth = 4;

    // Lookup table
    uint32_t lookup_table_entries = 1024;            // 1024 x 32-bit

    // External memory
    uint32_t external_memory_latency_cycles = 100;   // Round-trip
    uint32_t external_memory_size_bytes = 64 * 1024 * 1024;  // 64 MB

    // Kernel arguments
    uint32_t kernel_args[4] = {0, 0, 0, 0};         // ARG0-ARG3 -> r1-r4
    uint32_t start_pc = 0;
};
```

**Loading**: JSON config file with CLI overrides (`--num-warps=8 --l1-cache-size=8192`). Unspecified fields use defaults. Use nlohmann/json (single-header) or minimal hand-rolled parser.

**Propagation**: Passed by const ref to `FunctionalModel` and `TimingModel` constructors. Each sub-component receives only its relevant parameters via explicit constructor injection.

**Validation**: `SimConfig::validate()` checks constraints (num_warps in [1,8], cache size power-of-2, etc.) at startup.

---

## 4. Core Data Structures

### DecodedInstruction

```cpp
struct DecodedInstruction {
    InstructionType type;       // ALU, MUL, DIV, LOAD, STORE, BRANCH, JAL, JALR,
                                // LUI, AUIPC, VDOT8, TLOOKUP, ECALL, EBREAK, CSR
    AluOp alu_op;               // ADD, SUB, XOR, OR, AND, SLL, SRL, SRA, SLT, SLTU
    MulDivOp muldiv_op;         // MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
    MemOp mem_op;               // LB, LH, LW, LBU, LHU, SB, SH, SW
    BranchOp branch_op;         // BEQ, BNE, BLT, BGE, BLTU, BGEU
    ExecUnit target_unit;       // ALU, MULTIPLY, DIVIDE, LDST, TLOOKUP, SYSTEM
    uint8_t rd, rs1, rs2;
    int32_t imm;
    bool has_rd;                // Writes to rd?
    bool reads_rd;              // True only for VDOT8 (3-operand)
    uint8_t num_src_regs;       // 2 for most, 3 for VDOT8
    uint16_t csr_addr;
    uint32_t raw;
};
```

### TraceEvent (functional-timing interface contract)

```cpp
struct TraceEvent {
    uint32_t warp_id;
    uint32_t pc;
    DecodedInstruction decoded;

    // Per-thread results (32 lanes)
    std::array<uint32_t, 32> results;           // Computed rd values
    std::array<uint32_t, 32> mem_addresses;     // Effective addresses (LD/ST only)
    std::array<uint32_t, 32> store_data;        // Store data per thread
    std::array<uint8_t, 32> mem_size;           // Access size: 1, 2, or 4 bytes
    bool is_load, is_store;

    // Branch info
    bool is_branch;
    bool branch_taken;                          // Lane 0 outcome (no divergence)
    uint32_t branch_target;

    // Control flow
    bool is_ecall, is_ebreak;
    uint32_t panic_cause;                       // r31 lane 0 if EBREAK

    // TLOOKUP
    bool is_tlookup;
    std::array<uint32_t, 32> tlookup_indices;

    uint64_t sequence_number;                   // Global sequence for debugging
};
```

### WritebackEntry

```cpp
struct WritebackEntry {
    bool valid = false;
    uint32_t warp_id;
    uint8_t dest_reg;
    std::array<uint32_t, 32> values;            // One per lane
    ExecUnitType source_unit;
};
```

---

## 5. Instruction Decoder

Shared between functional and timing models. Input: 32-bit instruction word. Output: `DecodedInstruction`.

### Decoding table (opcode bits [6:0])

| Opcode    | Format | Category                                                    |
|-----------|--------|-------------------------------------------------------------|
| `0110011` | R      | ALU-R (funct7=0x00/0x20) or M-ext (funct7=0x01)           |
| `0010011` | I      | ALU-I (ADDI, XORI, ORI, ANDI, SLLI, SRLI, SRAI, SLTI...) |
| `0000011` | I      | Loads (LB, LH, LW, LBU, LHU by funct3)                    |
| `0100011` | S      | Stores (SB, SH, SW by funct3)                              |
| `1100011` | B      | Branches (BEQ, BNE, BLT, BGE, BLTU, BGEU by funct3)      |
| `1101111` | J      | JAL                                                         |
| `1100111` | I      | JALR                                                        |
| `0110111` | U      | LUI                                                         |
| `0010111` | U      | AUIPC                                                       |
| `1110011` | I      | SYSTEM: ECALL (imm=0), EBREAK (imm=1), CSRRS (funct3!=0)  |
| `0001011` | R      | **VDOT8** (custom-0, funct7=0x00, funct3=0x0)              |
| `0101011` | I      | **TLOOKUP** (custom-1)                                      |

### M-extension routing (opcode `0110011`, funct7=`0x01`)

| funct3 | Instruction | Unit    |
|--------|-------------|---------|
| 0x0    | MUL         | MULTIPLY |
| 0x1    | MULH        | MULTIPLY |
| 0x2    | MULHSU      | MULTIPLY |
| 0x3    | MULHU       | MULTIPLY |
| 0x4    | DIV         | DIVIDE   |
| 0x5    | DIVU        | DIVIDE   |
| 0x6    | REM         | DIVIDE   |
| 0x7    | REMU        | DIVIDE   |

### VDOT8 specifics
- `reads_rd = true`, `num_src_regs = 3`, `target_unit = MULTIPLY`
- Scoreboard must check rs1, rs2, AND rd at issue

### TLOOKUP specifics
- `num_src_regs = 1` (only rs1 needs scoreboard check), `target_unit = TLOOKUP`
- rd is destination only

---

## 6. Functional Model

### State
- `FunctionalRegisterFile`: Array `[MAX_WARPS][WARP_SIZE][32]` of uint32_t. r0 always returns 0, writes to r0 discarded.
- `FlatMemory`: `std::vector<uint8_t>` of `external_memory_size_bytes`. Supports read8/16/32 and write8/16/32 at arbitrary byte addresses.
- `LookupTable`: `std::vector<uint32_t>` of `lookup_table_entries`. Direct index access.
- `InstructionMemory`: `std::vector<uint32_t>` of `instruction_mem_size_bytes / 4`.

### `execute(warp_id, pc) -> TraceEvent`

1. Fetch instruction word from `InstructionMemory[pc / 4]`
2. Decode via shared `Decoder::decode()`
3. For each of 32 lanes:
   - Read source registers from `register_file[warp_id][lane]`
   - Compute result based on instruction type
   - **Loads**: addr = reg[rs1] + imm, read from FlatMemory
   - **Stores**: addr = reg[rs1] + imm, write reg[rs2] to FlatMemory
   - **Branches**: evaluate condition on lane 0 (no divergence). Record taken/not-taken + target
   - **CSR reads**: 0xC00=warp_id, 0xC01=lane (varies per lane), 0xC02=num_warps
   - **VDOT8**: `rd += sum(sign_extend(rs1_byte[i] * rs2_byte[i]) for i in 0..3)`
   - **TLOOKUP**: `lookup_table[reg[rs1] + sign_extend(imm)]`
   - Write result to register file (if has_rd, rd != 0)
4. For ECALL: mark warp inactive in functional state
5. For EBREAK: set panic flag, capture r31 lane 0 as cause
6. Package all data into TraceEvent, return it

**The functional model does NOT maintain PCs**. The timing model passes PC explicitly and is the single source of truth for PC management.

---

## 7. Timing Model

### Cycle Evaluation Strategy: Forward-Order with Double-Buffered State

Each component has `evaluate()` (reads current state, computes next state) and `commit()` (current = next). This ensures 1-cycle propagation delays match the real hardware.

**Within `TimingModel::tick()`:**

```
1. fetch_stage.evaluate()           // Read instruction memory, write to fetch-decode latch
2. decode_stage.evaluate()          // Decode instruction, push to per-warp instruction buffer
3. warp_scheduler.evaluate()        // Select eligible warp, issue instruction
                                    //   -> calls functional_model.execute() here
                                    //   -> sets scoreboard pending in next state
4. operand_collector.evaluate()     // Model 1-2 cycle operand read timing
5. dispatch_controllers[*].evaluate()  // Drain threads to execution units
6. execution_units[*].evaluate()    // Advance pipelines
7. coalescing_unit.evaluate()       // Process address batches
8. cache.evaluate()                 // Hit/miss, MSHR allocation
9. memory_interface.evaluate()      // Advance in-flight requests, deliver responses
10. writeback_arbiter.evaluate()    // Select one unit, write result
                                    //   -> clears scoreboard pending in next state
11. ALL.commit()                    // Flip all double-buffered state
```

The scoreboard uses double-buffering: writes go to `next`, reads come from `current`. `commit()` copies `next -> current`. This ensures a scoreboard clear in cycle N is visible to the scheduler in cycle N+1 (1-cycle minimum RAW gap).

### 7.1 Fetch Stage

**State**: `rr_pointer` (round-robin warp index), per-warp PCs (via WarpState), per-warp active flags.

**Each cycle**:
1. Target warp = `rr_pointer`
2. If warp active AND instruction buffer not full: read `instruction_memory[pc/4]`, place `{raw_word, warp_id, pc}` into fetch-decode latch, increment warp PC by 4
3. If buffer full or warp inactive: skip (no BRAM read)
4. `rr_pointer = (rr_pointer + 1) % num_warps` unconditionally

**Branch redirect interface**: When execute resolves a taken branch/jump for warp W:
- Set `warp_state[W].pc = target`
- Flush `warp_state[W].instruction_buffer`
- Invalidate any in-flight instruction for warp W in fetch-decode latch or decode stage

### 7.2 Decode Stage

**Each cycle**:
1. Read fetch-decode latch. If invalid, do nothing.
2. Decode raw word via `Decoder::decode()`
3. **Panic check**: If EBREAK, trigger panic controller instead of buffering
4. Push `{DecodedInstruction, warp_id, pc}` into `warp_state[warp_id].instruction_buffer`

Note: The functional model is NOT called here. Only the timing-side decode happens.

### 7.3 Warp Scheduler (Issue Stage)

**State**: `rr_pointer` (loose round-robin base, always advances by 1 per cycle).

**Each cycle**:
1. Scan from `rr_pointer` through all warps
2. For each warp W, check eligibility:
   - `buffer_not_empty`: instruction buffer has entries
   - `scoreboard_clear`: For head instruction's source regs -- 2-operand: `!pending(W, rs1) && !pending(W, rs2)`. VDOT8: also `!pending(W, rd)`. r0 never pending.
   - `operand_collector_free`: not currently occupied
   - `target_unit_ready`: dispatch controller for target unit is not busy
3. First eligible warp wins. Pop from buffer. **Call `functional_model.execute(warp_id, pc)`** to get TraceEvent. Place `{instruction, trace_event}` into scheduler-to-operand-collector latch.
4. If `has_rd && rd != 0`: set `scoreboard_next.set_pending(warp_id, rd)`
5. `rr_pointer = (rr_pointer + 1) % num_warps` unconditionally

### 7.4 Operand Collector

**State**: `busy`, `cycles_remaining`, `current_instruction + trace_event`

**Each cycle**:
1. If not busy and scheduler latch has valid instruction: accept. Set `cycles_remaining = 1` (2-operand) or `2` (VDOT8/3-operand). Set busy.
2. If busy: decrement `cycles_remaining`
3. If `cycles_remaining == 0`: hand off to target dispatch controller. Set busy = false.

The operand collector only models timing. Actual operand values are in the TraceEvent.

### 7.5 Dispatch Controllers (per execution unit)

Each has: `busy` flag, `warp_tag`, instruction+trace_event, `threads_dispatched`/`threads_completed` counters, single-entry `result_buffer` (WritebackEntry).

**ALU Dispatch**: All 32 threads in 1 cycle. Result buffer valid after 1 cycle. Busy for 1 cycle.

**Multiply/VDOT8 Dispatch**: Dispatches all 32 threads into pipelined shift register. Busy for 1 cycle (pipeline accepts every cycle). Pipeline internally tracks for `multiply_pipeline_stages` cycles. Result buffer filled when pipeline output ready. Structural hazard if result buffer still occupied (pipeline stalls).

**Divide Dispatch**: All 32 threads processed simultaneously via iterative algorithm. Busy for ~32 cycles (counter-based). Result buffer filled on completion.

**LD/ST Dispatch**: With `num_ldst_units` AGUs and 32 threads: `ceil(32 / num_ldst_units)` cycles for address generation. After all addresses generated, push batch to address gen FIFO. If FIFO full, stall. Dispatch controller becomes free once batch enters FIFO. Load results return via MSHR fill writeback path (not through this dispatch controller's result buffer).

**TLOOKUP Dispatch**: 32 threads x 2 cycles/lane = 64 cycles serial drain. Busy for all 64 cycles. Result buffer filled after completion.

**`target_unit_ready` signal**: Each dispatch controller reports whether it can accept new work. ALU/MUL: ready unless result buffer occupied. DIV: not ready while iterating. LD/ST: not ready while processing addresses or FIFO full. TLOOKUP: not ready while draining.

### 7.6 Writeback Arbiter

**State**: `rr_pointer` among writeback sources (ALU, MUL, DIV, TLOOKUP, MSHR-fill).

**Each cycle**:
1. Scan from `rr_pointer` for first source with valid result buffer
2. If found: consume result buffer, set `scoreboard_next.clear_pending(warp_id, dest_reg)`, advance `rr_pointer`
3. No forwarding -- pending bit cleared means the value is committed; next cycle the scheduler can see it

**MSHR fill as writeback source**: When external memory returns data for a load miss, the MSHR fill logic extracts per-thread values and places them in a dedicated writeback buffer that participates in the arbiter.

### 7.7 Scoreboard

**State**: 2D bit array `pending[MAX_WARPS][32]`. Two copies: `current` (read by scheduler) and `next` (written by issue/writeback).

- `set_pending(warp, reg)`: called at issue, modifies `next`
- `clear_pending(warp, reg)`: called at writeback commit, modifies `next`
- `is_pending(warp, reg)`: reads `current`. r0 always returns false.
- `commit()`: `current = next`, then reset `next = current` to seed next cycle

### 7.8 Memory System Timing

**L1 Data Cache** (direct-mapped):
- `num_sets = l1_cache_size_bytes / cache_line_size_bytes`
- Set index = `(addr / 128) % num_sets`, tag = `addr / (128 * num_sets)`
- Processes one request per cycle from coalescing unit

**Cache operations**:
- **Load hit**: Extract per-thread values, place in MSHR-fill writeback buffer
- **Load miss**: Allocate MSHR (stall if none free), send read to external memory. When response arrives: install line, extract values, writeback, clear scoreboard, free MSHR.
- **Store hit**: Update line in L1, push full line to write buffer (write-through)
- **Store miss** (write-allocate): Allocate MSHR, fetch line, install, apply store data, push to write buffer, free MSHR. No register writeback.

**MSHRs**: Each entry tracks `{valid, cache_line_addr, is_store, warp_id, dest_reg, thread_mask[32], byte_offsets[32], store_data[32x32], store_byte_en[32x4]}`. No duplicate detection/merging.

**Write Buffer**: FIFO, `write_buffer_depth` entries. Full -> stall coalescing unit. Reads prioritized over drain.

**Coalescing Unit**:
- Pulls one entry from address gen FIFO, processes one warp at a time
- All-or-nothing: if all 32 addresses in same 128B line -> 1 request, else 32 serialized requests
- Stalls if cache cannot accept (MSHR full, write buffer full)

**External Memory Interface**:
- Queue of in-flight requests with countdown timers
- `external_memory_latency_cycles` round-trip per request
- Accepts 1 new request per cycle (reads prioritized over write buffer drains)
- Responses delivered to originating MSHR

### 7.9 Branch Handling

All control flow (branches taken, JAL, JALR) resolved in execute stage:
1. Execute stage checks `TraceEvent.branch_taken`
2. If taken: signal fetch stage to redirect warp PC, flush instruction buffer, invalidate in-flight instructions for that warp
3. Penalty: empty buffer -> wait for round-robin fetch slot (up to `num_warps` cycles) + fetch (1 cycle) + decode (1 cycle) before new instruction available
4. Not-taken: no penalty (PC already incremented at fetch)

### 7.10 Panic Controller (EBREAK)

State machine triggered by decode stage detecting EBREAK:

| Step   | Cycles | Action                                                              |
|--------|--------|---------------------------------------------------------------------|
| 0      | 1      | Assert panic_pending, inhibit scheduler                             |
| 1      | 1      | Latch r31 from panicking warp lane 0 (from functional model state) |
| 2      | 1      | Latch PANIC_WARP, PANIC_LANE=0, PANIC_PC, PANIC_CAUSE              |
| 3..N   | ~32    | Drain in-flight instructions (wait for all units to complete)       |
| N+1    | 1      | Mark all warps inactive, set STATUS.PANIC + STATUS.DONE             |

---

## 8. Abstract Interfaces for Modularity

```cpp
class PipelineStage {
public:
    virtual ~PipelineStage() = default;
    virtual void evaluate() = 0;    // Process one cycle
    virtual void commit() = 0;      // Flip double-buffered state
    virtual void reset() = 0;
};

class ExecutionUnit {
public:
    virtual ~ExecutionUnit() = default;
    virtual void evaluate() = 0;
    virtual void commit() = 0;
    virtual void reset() = 0;
    virtual bool is_ready() const = 0;
    virtual bool has_result() const = 0;
    virtual WritebackEntry consume_result() = 0;
    virtual ExecUnitType get_type() const = 0;
};

class MemoryInterface {
public:
    virtual ~MemoryInterface() = default;
    virtual bool submit_read(uint32_t line_addr, uint32_t mshr_id) = 0;
    virtual bool submit_write(uint32_t line_addr, const CacheLine& data) = 0;
    virtual void evaluate() = 0;
    virtual void commit() = 0;
    virtual bool has_read_response() const = 0;
    virtual MemoryResponse get_read_response() = 0;
};

class CacheModel {
public:
    virtual ~CacheModel() = default;
    virtual CacheResult lookup(uint32_t address) = 0;
    virtual void install_line(uint32_t line_addr, const CacheLine& data) = 0;
    virtual void update_line(uint32_t line_addr, uint32_t offset,
                             uint32_t data, uint8_t byte_en) = 0;
    virtual void reset() = 0;
};
```

These enable swapping implementations (e.g., set-associative cache, variable-latency memory) without touching other components.

---

## 9. Binary Loading

### Supported formats

**Raw binary**: Read entire file into `instruction_memory` starting at address 0. Interpreted as 32-bit little-endian instruction words.

**ELF32 (RV32)**: Parse ELF32 headers (minimal hand-rolled parser using `<elf.h>` struct definitions):
1. Verify `EI_CLASS=ELFCLASS32`, `e_machine=EM_RISCV`, little-endian
2. Load `PT_LOAD` segments: instruction-range addresses -> instruction_memory, data-range -> flat_memory
3. Set `start_pc` from `e_entry` (unless overridden by config)

**Lookup table**: Loaded separately via `--lookup-table=file.bin` or from designated ELF section.

### Initialization sequence

For each warp 0..`num_warps-1`, for each lane 0..31:
- PC = `start_pc`
- `reg[0] = 0` (hardwired), `reg[1..4] = kernel_args[0..3]`, `reg[5..31] = 0`
- Mark warp active

Clear: scoreboard, instruction buffers, operand collector, dispatch controllers, writeback buffers, cache (all valid bits = 0), MSHRs, write buffer.

---

## 10. Statistics

### Collected metrics

| Category | Metrics |
|----------|---------|
| Global | total_cycles, total_instructions_issued, IPC |
| Per-warp | instructions_issued, cycles_active, stall_cycles_scoreboard, stall_cycles_buffer_empty, stall_cycles_unit_busy |
| Pipeline | fetch_skip_count, scheduler_idle_cycles, operand_collector_busy_cycles, branch_flushes |
| Per-unit | busy_cycles, instructions, utilization |
| Memory | cache_hits, cache_misses, hit_rate, load_hits, load_misses, store_hits, store_misses, mshr_stall_cycles, write_buffer_stall_cycles, coalesced_requests, serialized_requests, external_memory_reads/writes, average_memory_latency |
| Writeback | writeback_conflicts (multiple valid, only one selected) |

### Output

- `Stats::report(ostream&)`: human-readable summary at simulation end
- `Stats::to_json(ostream&)`: machine-parseable JSON
- **Per-cycle trace** (`--trace` flag): one line per cycle showing full pipeline state, invaluable for debugging

---

## 11. Implementation Sequencing

### Phase 1 -- Foundation
1. `types.h`, `isa.h`, `config.h/cpp`
2. `decoder.h/cpp` + `test_decoder.cpp` -- full RV32IM + VDOT8 + TLOOKUP decoder with tests
3. `functional/alu.h/cpp` + `test_alu.cpp` -- pure compute functions
4. `functional/register_file.h/cpp`, `functional/memory.h/cpp`
5. `functional/functional_model.h/cpp` + `test_functional.cpp`
6. `elf_loader.h/cpp`
7. `main.cpp` (functional-only mode) -- load binary, run to completion, print register state

### Phase 2 -- Timing Skeleton
1. `scoreboard.h/cpp` + `test_scoreboard.cpp`
2. `instruction_buffer.h/cpp`, `warp_state.h/cpp`, `trace_event.h`
3. `fetch_stage.h/cpp`, `decode_stage.h/cpp`
4. `warp_scheduler.h/cpp` + `test_warp_scheduler.cpp`
5. `operand_collector.h/cpp`
6. `writeback_arbiter.h/cpp`
7. `timing_model.h/cpp` -- top-level tick loop

### Phase 3 -- Execution Units
1. `dispatch_controller.h/cpp` -- base + per-unit variants
2. `alu_unit.h/cpp` -- test full pipeline flow end-to-end
3. `multiply_unit.h/cpp` -- pipelined shift register
4. `divide_unit.h/cpp` -- iterative counter
5. `tlookup_unit.h/cpp` -- 2-cycle-per-lane serial drain
6. `ldst_unit.h/cpp` -- address generation + FIFO
7. `test_pipeline.cpp`

### Phase 4 -- Memory System
1. `cache.h/cpp` + `test_cache.cpp` -- direct-mapped L1
2. `mshr.h/cpp`, `write_buffer.h/cpp`
3. `coalescing_unit.h/cpp` + `test_coalescing.cpp`
4. `memory_interface.h/cpp` -- fixed-latency model
5. Wire: LD/ST -> coalescing -> cache -> memory interface -> MSHR fill -> writeback

### Phase 5 -- Control Flow and Completion
1. Branch handling (flush + redirect)
2. `panic_controller.h/cpp` + `test_panic.cpp`
3. ECALL warp completion
4. `stats.h/cpp`
5. `test_integration.cpp` -- full integration tests
6. Per-cycle trace output

### Phase 6 -- Validation
1. Micro-benchmarks with hand-calculated expected cycle counts
2. Matmul kernel correctness + performance validation
3. Simulator performance optimization

---

## 12. Verification Plan

### Unit test targets
- **Decoder**: Every RV32IM instruction encoding + VDOT8 + TLOOKUP + ECALL/EBREAK/CSRRS
- **ALU**: All operations including edge cases (overflow, shift-by-0/31, DIV/0->-1, REM/0->dividend, INT32_MIN/-1)
- **VDOT8**: Known INT8 byte patterns with accumulator values
- **Scoreboard**: Set/clear/is_pending, double-buffer semantics, r0 never pending
- **Cache**: Direct-mapped hit/miss, MSHR allocation/stall, write-through, write-allocate, write buffer full stall
- **Coalescing**: 32 addresses in 1 line -> 1 request; spanning 2+ lines -> 32 requests; boundary edge cases
- **Scheduler**: RR fairness, scoreboard stall, buffer empty skip, unit busy skip, starvation prevention

### Integration test programs
1. **ADD chain**: 10 sequential ADDs, verify result + cycle count
2. **Load-use stall**: Load into r5, immediately use r5. Verify stall-on-use + cycle count
3. **Branch loop**: Loop 10 times, verify iteration count + branch penalty cycles
4. **Multi-warp**: 4 warps each computing different results, all ECALL to complete
5. **Memory coalescing**: Contiguous addresses (coalesce) vs scattered (serialize), compare cycle counts
6. **VDOT8 matmul**: 4x4 matrix multiply, verify result matrix
7. **TLOOKUP interpolation**: Execute spec's interpolation sequence, verify result

### Cycle-count validation
For each micro-benchmark, hand-calculate expected cycle counts from the architectural spec and compare:
- Fetch RR period = `num_warps` cycles
- Operand collector: 1 cycle (2-operand), 2 cycles (VDOT8)
- ALU: 1 cycle dispatch-to-result
- MUL: `multiply_pipeline_stages` cycles
- DIV: ~32 cycles
- TLOOKUP: 64 cycles (32 lanes x 2)
- Cache hit: issue to writeback commit (pipeline depth + arbiter wait)
- Cache miss: above + `external_memory_latency_cycles`

---

## Critical Reference Files
- `/Users/colinbaird/Projects/GPU/resources/gpu_architectural_spec.md` -- authoritative source for all architectural behavior
- `/Users/colinbaird/Projects/GPU/resources/RISCV_CARD.md` -- RV32IM instruction encodings for decoder implementation
- `/Users/colinbaird/Projects/GPU/CLAUDE.md` -- project context and design principles
