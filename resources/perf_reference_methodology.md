# Performance Reference Methodology

This document defines the analytical methodology for deriving reference performance statistics from the [architectural specification](/resources/gpu_architectural_spec.md). These reference values serve as ground truth for validating the timing model implementation.

All derivations use the **default configuration** with **1 warp** (`--num-warps=1`).

## 1. Default Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| `num_warps` | 1 | Test convention |
| `instruction_buffer_depth` | 2 | Spec SS 4.2 |
| `multiply_pipeline_stages` | 3 | Spec SS 4.6 |
| `num_ldst_units` | 8 | Spec SS 5.1 |
| `l1_cache_size_bytes` | 4096 | Spec SS 5.3 |
| `cache_line_size_bytes` | 128 | Spec SS 5.3 |
| `num_mshrs` | 4 | Spec SS 5.3.1 |
| `write_buffer_depth` | 4 | Spec SS 5.3.2 |
| `external_memory_latency_cycles` | 100 | Config default |

## 2. Pipeline Model (1 Warp)

With 1 warp, the pipeline simplifies significantly. The round-robin fetch pointer always targets warp 0 (the only active warp), so fetch occurs every cycle if the instruction buffer is not full.

### 2.1 Pipeline Stages and Latencies

The pipeline is: **Fetch -> Decode -> Issue -> Operand Collect -> Dispatch -> Execute -> Writeback**

All stages use double-buffered state (evaluate/commit). Within each cycle, stages evaluate in forward order. The commit at end-of-cycle makes computed state visible to the next cycle.

**Stage interactions per cycle (forward evaluation order):**

1. **Fetch:** reads instruction from BRAM at current PC, produces output
2. **Decode:** reads previous cycle's fetch output, decodes, pushes to instruction buffer
3. **Scheduler:** reads instruction buffer, checks eligibility, issues if possible
4. **Operand Collector:** accepts issued instruction (same cycle), reads operands
5. **Dispatch:** routes completed operand collection to target execution unit
6. **Execution Units:** process dispatched work (ALU, MUL, DIV, LDST, TLOOKUP)
7. **Memory System:** coalescing, cache, external memory interface
8. **Writeback Arbiter:** selects one unit with valid result, writes to register file, clears scoreboard
9. **Commit all:** double-buffer flip makes all state changes visible to next cycle

### 2.2 Pipeline Startup (1 Warp)

The pipeline has a 2-cycle fill latency before the first instruction can issue:

```
Cycle 1: Fetch I0 -> next output
         Decode: nothing (no fetch output in current state)
         Scheduler: buffer empty -> idle
         Commit: fetch output = I0

Cycle 2: Fetch I1 -> next output
         Decode: reads I0 from fetch current, decodes -> push to buffer
         Scheduler: buffer empty (decode hasn't committed yet) -> idle
         Commit: fetch output = I1, buffer = [I0]

Cycle 3: Fetch I2 -> next output
         Decode: reads I1, decodes -> push to buffer
         Scheduler: buffer has I0, eligible -> ISSUES I0
         OpColl: accepts I0 (1-cycle for 2-operand)
         OpColl evaluate: completes, produces dispatch output
         Dispatch: I0 -> ALU (if ALU instruction)
         ALU: evaluates, result ready
         WB Arbiter: picks up ALU result, writes back, clears scoreboard
         Commit: buffer = [I1], scoreboard clear for I0.rd
```

**Key insight:** For 1-cycle operand collection and ALU execution, the entire path from issue to scoreboard clear completes within the *same cycle* as issue. This means a dependent instruction can issue in the very next cycle.

### 2.3 Steady-State Throughput

For independent instructions with 1 warp: **1 instruction issued per cycle** after the 2-cycle pipeline fill.

The instruction buffer depth of 2 allows fetch to stay 1 instruction ahead of issue. With 1 warp, fetch fills the buffer every cycle (no round-robin delay), and the scheduler drains it every cycle.

## 3. Instruction Category Timing

### 3.1 Scoreboard Stall Model

The **scoreboard stall** is the number of extra idle cycles between issuing instruction A (which writes `rd`) and issuing instruction B (which reads `rd`). Without the dependency, B would issue the cycle after A.

| Category | Issue-to-scoreboard-clear | RAW stall cycles | Spec reference |
|----------|--------------------------|------------------|----------------|
| ALU (R/I), LUI, AUIPC | 0 (same cycle as issue) | 0 | SS 4.6: 1 cycle |
| JAL, JALR (link write) | 0 (ALU path) | 0 | SS 4.6: ALU |
| BRANCH | N/A (no rd) | N/A | SS 4.6: ALU |
| MUL, MULH, MULHSU, MULHU | 2 cycles after issue | 2 | SS 4.6: parameterizable (3 stages) |
| DIV, DIVU, REM, REMU | 31 cycles after issue | 31 | SS 4.6: ~32 cycles |
| LOAD (cache hit) | 3 cycles after issue | 3 | SS 5.1 + SS 5.2 + SS 5.3 |
| LOAD (cache miss) | 3 + mem_latency cycles | 3 + 100 = 103 | + SS 5.6: external memory |
| STORE | N/A (no rd) | N/A | - |
| FENCE | N/A (no rd, pipeline NOP) | N/A | - |
| ECALL | N/A (terminates warp) | N/A | - |
| VDOT8 | 3 cycles after issue | 3 | SS 4.4: 2-cycle opcoll + 3-stage MUL |
| TLOOKUP | 63 cycles after issue | 63 | SS 4.6: 2 cycles/lane x 32 |

**Derivation for ALU (0-cycle stall):**
Per spec SS 4.6, ALU latency is 1 cycle. Per SS 4.7, the writeback arbiter writes results and clears the scoreboard in 1 cycle. With the forward evaluation order, ALU result, writeback, and scoreboard clear all happen within the same cycle as dispatch. Since dispatch occurs in the same cycle as issue (for 1-cycle opcoll), the scoreboard is cleared by end of the issue cycle. The dependent instruction sees a clear scoreboard on the *next* cycle.

**Derivation for MUL (2-cycle stall):**
MUL has 3 pipeline stages (spec SS 4.6). Issue cycle N: opcoll completes same cycle (1-cycle), dispatches to MUL pipeline. MUL evaluates: 3 stages means the result is ready 2 cycles after dispatch (stages count down: 3->2 in N, 2->1 in N+1, 1->0 in N+2). WB arbiter writes back at end of cycle N+2. Dependent instruction eligible at cycle N+3. Stall = (N+3) - (N+1) = 2 extra cycles.

**Derivation for LOAD (cache hit, 3-cycle stall):**
Load issues at cycle N, opcoll same cycle, dispatches to LDST unit. LDST address generation takes ceil(32/8) = 4 cycles (8 units for 32 threads). With the countdown starting same cycle: 4->3 (N), 3->2 (N+1), 2->1 (N+2), 1->0 (N+3). At cycle N+3, addresses pushed to FIFO. Coalescing unit processes same cycle (ISA tests: all 32 threads access same address -> coalesced). Cache hit: result produced same cycle. WB arbiter writes back at end of N+3. Dependent instruction eligible at N+4. Stall = (N+4) - (N+1) = 3 extra cycles.

**Derivation for LOAD (cache miss):**
Same as hit through cycle N+3, but cache misses. MSHR allocated, external memory request submitted. Response arrives at cycle N+3+100 = N+103. WB arbiter writes back at end of N+103. Dependent instruction eligible at N+104. Stall = 103 extra cycles.

### 3.2 Branch Penalty

Per spec SS 4.2, when a branch is taken:
1. The warp's instruction buffer is flushed
2. The warp's PC is updated to the branch target
3. Fetching resumes at the new PC on the next round-robin slot

With 1 warp, the penalty after a taken branch dispatches at cycle N:
- Cycle N+1: Fetch targets new PC. Decode sees invalidated output from N -> no decode output.
- Cycle N+2: Decode processes the fetched target instruction -> pushes to buffer.
- Cycle N+3: Scheduler can issue the target instruction.

**Branch penalty = 2 idle cycles** between the branch issue and the next issue.

Not-taken branches incur **0 penalty** -- the sequential instructions already in the buffer are correct.

**JAL and JALR** are unconditional jumps that always redirect the PC. They incur the same 2-cycle penalty as taken branches.

### 3.3 FENCE Handling

FENCE is decoded with `target_unit = SYSTEM` and `has_rd = false`. It issues and passes through operand collection (1 cycle) but does not dispatch to any execution unit and does not write any register. It effectively consumes 1 issue slot as a pipeline NOP.

### 3.4 ECALL Handling

ECALL issues and dispatches to the SYSTEM path, which marks the warp as inactive. No register write occurs. The warp terminates after ECALL completes dispatch.

## 4. Memory System Model

### 4.1 Cache Behavior

The L1 data cache is **direct-mapped**, 4096 bytes with 128-byte lines = **32 sets**.

- `set_index = (byte_address / 128) % 32`
- `tag = byte_address / 128 / 32`

The cache starts **cold** (all lines invalid). The first access to any cache line is a **compulsory miss** incurring 100-cycle external memory latency. Subsequent accesses to the same line (if not evicted) are hits.

### 4.2 Coalescing in ISA Tests

In ISA tests running with 1 warp, all 32 threads execute the same instruction with the same register values (no per-thread divergence). Therefore:
- All 32 thread addresses are **identical** for any load/store
- All accesses **coalesce** to a single cache line request
- `coalesced_requests` increments by 1 per load/store operation
- `serialized_requests` remains 0

### 4.3 ISA Test Data Layout

The linker script places `.data` at address `0x2000`. Test data (e.g., `tdat` in load/store tests) starts at this address. Address `0x2000` maps to:
- `line_address = 0x2000 / 128 = 64`
- `set_index = 64 % 32 = 0`

All test data for a single ISA test typically fits within a single 128-byte cache line (test data is only a few words). The first access misses, all subsequent accesses to the same line hit.

### 4.4 Store Timing

Stores have no destination register, so no scoreboard interaction. The timing path is:
- Issue -> opcoll (1 cycle) -> LDST address gen (4 cycles) -> coalescing -> cache
- On store hit: write to L1 + push to write buffer. No stall.
- On store miss: MSHR allocated -> external memory fetch (100 cycles) -> install line -> write data -> push to write buffer. The coalescing unit stalls during the external memory fetch, but since stores don't block the scoreboard, subsequent non-store instructions can continue if they don't need the LDST unit.

## 5. Total Cycle Count Formula

For a test with 1 warp, the total cycle count is:

```
total_cycles = pipeline_fill + instruction_cycles + stall_cycles + branch_penalty_cycles + memory_latency_cycles + drain_cycles
```

Where:
- **pipeline_fill** = 2 (fetch + decode before first issue)
- **instruction_cycles** = number of dynamic instructions (1 per cycle for independent instructions)
- **stall_cycles** = sum of RAW dependency stalls (per the table in SS 3.1)
- **branch_penalty_cycles** = 2 x number_of_taken_branches (includes JAL, JALR)
- **memory_latency_cycles** = sum of cache miss penalties (100 per cold miss, 0 for hits)
- **drain_cycles** = cycles for last instruction to complete through the pipeline after issue (0 for ALU ending with ECALL, variable for others)

Note: the ECALL that terminates the warp is the last instruction. After ECALL dispatches, the pipeline needs additional cycles to finish draining (writeback of any in-flight results, plus the tick that detects `all_warps_done && all_units_idle`).

## 6. Worked Example: `simple.S`

**Program:** `fence; li gp, 1; ecall; unimp`

Only 3 instructions execute (FENCE, LI, ECALL). The `unimp` is never reached.

**Instruction analysis:**
| # | Instruction | Type | Unit | has_rd | Depends on |
|---|-------------|------|------|--------|------------|
| 0 | `fence` | FENCE | SYSTEM | no | - |
| 1 | `li gp, 1` (= `addi x3, x0, 1`) | ALU_I | ALU | yes (x3) | x0 (always ready) |
| 2 | `ecall` | ECALL | SYSTEM | no | - |

**No RAW dependencies** (x3 is written by LI but not read before ECALL). **No taken branches.** **No memory accesses.**

**Cycle trace:**
```
Cycle 1: Fetch fence      | Decode: -         | Sched: idle (empty)
Cycle 2: Fetch li         | Decode: fence     | Sched: idle (empty)
Cycle 3: Fetch ecall      | Decode: li -> buf | Sched: ISSUE fence  (opcoll 1c, SYSTEM dispatch = NOP)
Cycle 4: Fetch unimp      | Decode: ecall     | Sched: ISSUE li     (opcoll 1c, ALU 1c, WB clears x3)
Cycle 5: -                | -                 | Sched: ISSUE ecall  (warp inactive)
         all_warps_done && all_units_idle -> terminate
```

**Expected statistics:**
- `total_cycles` = **5**
- `total_instructions_issued` = **3** (fence + li + ecall)
- `branch_flushes` = **0**
- `alu_instructions` = **1** (li)
- `scheduler_idle_cycles` = **2** (cycles 1 and 2)
- `cache_hits` = **0**, `cache_misses` = **0**
- `warp_stall_buffer_empty` = **2** (cycles 1-2, buffer empty)

## 7. Worked Example: `add.S` (first 3 test cases)

The `add.S` test contains 38 test cases using `TEST_RR_OP`, bypass tests, and zero-register tests. Here we trace the first 3 cases to illustrate the methodology.

Each `TEST_RR_OP(testnum, add, result, val1, val2)` expands to:
```assembly
test_N:
    li  gp, N          # addi x3, x0, N    (ALU, writes x3)
    li  x11, val1      # lui+addi or addi   (ALU, writes x11)
    li  x12, val2      # lui+addi or addi   (ALU, writes x12)
    add x14, x11, x12  # ALU, reads x11+x12, writes x14
    li  x7, result     # lui+addi or addi   (ALU, writes x7)
    bne x14, x7, fail  # BRANCH, reads x14+x7, not taken (test passes)
```

For small immediates, `li` is a single `addi`. For large immediates, `li` expands to `lui + addi` (2 instructions).

**All ALU operations with 0-cycle RAW stall.** The `add x14, x11, x12` reads x11 and x12, which were just written by the previous `li` instructions -- but since ALU clears the scoreboard same-cycle, no stall occurs.

**Branch `bne` is not taken** (test passes), so 0-cycle penalty.

**Per test case (small immediates):** 6 instructions, 0 stalls, 0 branch penalty.

The final `TEST_PASSFAIL` macro:
```assembly
    bne x0, gp, pass   # BRANCH, taken (gp != 0) -> 2-cycle penalty
fail:
    RVTEST_FAIL         # not reached
pass:
    RVTEST_PASS         # fence; li gp, 1; ecall
```

This adds 1 taken branch (2-cycle penalty) + 3 instructions (fence, li, ecall).

## 8. Statistics Fields

The reference JSON files include the following fields (matching `report_json()` output):

| Field | Derivation |
|-------|-----------|
| `total_cycles` | Formula from SS 5 |
| `total_instructions_issued` | Dynamic instruction count |
| `branch_flushes` | Count of taken branches (including JAL, JALR) |
| `alu_instructions` | ALU-type instructions (R, I, LUI, AUIPC, branches, JAL, JALR) |
| `mul_instructions` | MUL/MULH/MULHSU/MULHU instructions |
| `div_instructions` | DIV/DIVU/REM/REMU instructions |
| `ldst_instructions` | LOAD + STORE instructions |
| `cache_hits` | Warm cache line accesses |
| `cache_misses` | Cold cache line accesses (first touch) |
| `load_hits` / `load_misses` | Load-specific cache stats |
| `store_hits` / `store_misses` | Store-specific cache stats |
| `coalesced_requests` | Number of load/store ops (all coalesce in ISA tests) |
| `serialized_requests` | 0 for ISA tests (all addresses identical) |
| `external_memory_reads` | Equal to load_misses (one per cold miss) |
| `external_memory_writes` | Depends on store-through behavior |
