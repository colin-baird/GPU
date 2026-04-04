# GPU Accelerator — Architectural Specification

**Status:** Complete — all architectural questions resolved
**Revision:** 1.0
**Target:** Simulation + FPGA implementation
**Goal:** Quantized LLM inference (INT8×INT8 → INT32 accumulate)

---

## 1. Overview

A single streaming multiprocessor (SM) executing SIMT warps of 32 threads, using the **RV32IM** instruction set with a custom packed INT8 dot-product accumulate extension (`VDOT8`). The design is integer-only, targeting quantized neural network inference workloads. The architecture is modeled loosely on NVIDIA SM concepts but simplified for FPGA feasibility.

**Pipeline:** Fetch → Decode → Issue (Warp Scheduler) → Operand Collect → Dispatch → Execute → Writeback

---

## 2. ISA

### 2.1 Base ISA

- **RISC-V RV32IM** (32-bit integer base + multiply/divide extension)
- **Divide-by-zero** follows standard RV32M behavior: `DIV` by zero returns `−1`, `REM` by zero returns the dividend. No exception or panic is triggered. Software that needs to trap on zero divisors should check explicitly before dividing.
- **Unsupported instruction note:** `FENCE` is not supported in Phase 1. Software must not rely on it for ordering or synchronization in this machine model.

### 2.2 Custom Extension: Packed INT8 Dot-Product Accumulate

A single custom instruction, `VDOT8`, is added using the RISC-V `custom-0` opcode space (opcode `0001011`).

**Encoding:** R-type format — `VDOT8 rd, rs1, rs2`

```
 31        25  24   20  19   15  14  12  11    7  6       0
┌───────────┬────────┬────────┬───────┬────────┬──────────┐
│  funct7   │  rs2   │  rs1   │funct3 │   rd   │  opcode  │
│  0000000  │ 5 bits │ 5 bits │  000  │ 5 bits │ 0001011  │
└───────────┴────────┴────────┴───────┴────────┴──────────┘
```

- `funct7 = 0000000`, `funct3 = 000`. All other funct7/funct3 combinations within custom-0 are reserved for future use.

**Semantics:**
```
rd ← rd + sign_extend(rs1[7:0]   × rs2[7:0])
        + sign_extend(rs1[15:8]  × rs2[15:8])
        + sign_extend(rs1[23:16] × rs2[23:16])
        + sign_extend(rs1[31:24] × rs2[31:24])
```

- Four **signed** INT8 × INT8 multiplies producing INT16 intermediates, summed, and accumulated into the 32-bit value in `rd`.
- `rd` is both a source and destination (3-operand instruction): it is read for the current accumulator value and written with the updated result.
- **Signed operands only.** Unsigned and mixed-sign variants are not included in Phase 1. Encoding space is available in funct7/funct3 for future variants if needed.
- **Accumulate-only.** To compute a non-accumulated dot product, software zeros `rd` before issuing `VDOT8`.
- **Overflow: wrapping.** The 32-bit accumulation wraps on overflow using standard two's complement arithmetic, consistent with RISC-V `MUL`/`ADD` behavior. No saturation or trapping. Software is responsible for managing accumulator range (e.g., periodically shifting or splitting long dot products).

**Overflow headroom analysis:** Each signed INT8 × INT8 product has magnitude at most 16,384 (−128 × −128). A single VDOT8 adds at most 4 × 16,384 = 65,536 to the accumulator. INT32 can hold ±2,147,483,647, so approximately 32,768 VDOT8 instructions can execute before overflow risk — corresponding to a dot product of length 131,072. Typical inference dot products (hidden dimensions of 256–4096) are well within range.

**Hardware mapping:**
- `VDOT8` shares the existing **multiply pipeline** and DSP slices. No dedicated functional unit.
- The multiply pipeline's local dispatch controller recognizes `VDOT8` (via opcode + funct7 + funct3) and routes operands to packed INT8 multiply logic within the same DSP resources.
- Pipeline latency for `VDOT8` is the same parameterizable `STAGES` depth as scalar `MUL`.
- DSP slice packing details (how many INT8 multiplies fit per physical DSP slice) are documented in the FPGA Implementation Notes, as this varies by FPGA family.

**Operand collection impact:**
- `VDOT8` requires 3 register reads (rs1, rs2, rd), which takes 2 cycles in the operand collection stage (see §4.4).

### 2.3 Custom Extension: Table Lookup

A custom instruction, `TLOOKUP`, provides fast access to a dedicated lookup table BRAM for nonlinear function approximation, bypassing the data cache path entirely.

**Encoding:** I-type format in the RISC-V `custom-1` opcode space (opcode `0101011`) — `TLOOKUP rd, rs1, imm12`

**Semantics:**
```
table_addr ← rs1 + sign_extend(imm12)
rd ← lookup_table_bram[table_addr]
```

- `rs1` provides the table index (typically computed from the input value).
- `imm12` (12-bit signed immediate) provides the **base offset** selecting which function's sub-table to access (e.g., softmax entries start at offset 0, GELU at offset 256, SiLU at offset 512).
- The sum `rs1 + imm12` forms the BRAM read address. The 12-bit immediate gives a 4096-entry address space (matching a 4 KB default table with 32-bit entries, or larger with parameterization).
- `rd` receives the 32-bit table entry.
- This is a standard **2-operand instruction** (rs1 + immediate → rd). 1 cycle in the operand collector.

**Hardware mapping:**
- `TLOOKUP` has a **dedicated functional unit** with its own local dispatch controller, independent from ALU, multiply, and LD/ST paths.
- The TLOOKUP unit contains a read port into the lookup table BRAM. It does not use the data cache or coalescing unit.
- **Latency:** Each BRAM read takes 2 cycles (cycle 1: address presented; cycle 2: data out), but reads are **pipelined** — a new read can be issued every cycle. The BRAM is **true dual-port**, so both ports are used for simultaneous reads, giving 2 lanes per cycle throughput. Total warp latency: **17 cycles** (ceil(32 lanes / 2 ports) = 16 issue cycles + 1 pipeline drain cycle for the last pair's results). The TLOOKUP unit asserts busy for the full duration.
- The TLOOKUP unit asserts a **busy signal** back to the global scheduler, like all other functional units.

**Lookup table BRAM:**
- **Size:** Parameterizable (default: 4 KB = 1024 × 32-bit entries).
- **Contents:** Host-loaded via DMA before kernel launch, alongside instruction BRAM loading. Different functions occupy different address ranges within the same BRAM.
- **Read-only at runtime.** A second BRAM port (write) is used only during host loading; it is idle during execution.
- **Typical partitioning (default 1024 entries):**

| Function   | Base offset | Entries | Description                        |
|------------|-------------|---------|------------------------------------|
| exp (softmax) | 0        | 256     | Exponential approximation          |
| GELU       | 256         | 256     | Gaussian error linear unit         |
| SiLU       | 512         | 256     | Sigmoid linear unit                |
| rsqrt      | 768         | 256     | Reciprocal square root (layer norm)|

**Software interpolation pattern:**
```
# Compute interpolated GELU(x):
# Assume r1 = table index (integer part), r2 = fractional part (fixed-point)
TLOOKUP  r3, r1, 256       # r3 = table[256 + index]     (GELU base offset)
ADDI     t0, r1, 1         # t0 = index + 1
TLOOKUP  r4, t0, 256       # r4 = table[256 + index + 1]
SUB      r5, r4, r3        # r5 = table[index+1] - table[index]
MUL      r5, r5, r2        # r5 = diff * frac
SRA      r5, r5, FRAC_BITS # fixed-point shift
ADD      r3, r3, r5        # r3 = interpolated result
```

---

## 3. Warp and Thread Configuration

| Parameter              | Value                  | Notes                                |
|------------------------|------------------------|--------------------------------------|
| Warp size              | 32 threads             | Fixed                                |
| Resident warps per SM  | 4–8                    | Parameterizable; start with 4        |
| Registers per thread   | 32 × 32-bit            | Matches RV32I register file          |
| Register file total    | 16–32 KB               | Scales with warp count               |

---

## 4. Pipeline Organization

### 4.1 Instruction Memory

- **Preloaded BRAM**, read-only at runtime. The program is loaded by the host interface before kernel launch.
- **Single read port.** Only one fetch per cycle is needed (round-robin across warps).
- **Size:** Parameterizable (default 8 KB = 2048 RV32 instructions). Each instruction is 4 bytes.
- All warps execute the **same program** but each warp has its own independent program counter, so warps may be at different points in the program.
- A second BRAM port (write) is used only during program loading from the host; it is idle during execution.

### 4.2 Frontend: Fetch and Decode

Fetch and decode are **two separate pipelined stages**. In steady state, the pipeline delivers one decoded instruction per cycle (fetch of warp N overlaps with decode of warp N−1).

**Fetch stage:**
- A **round-robin pointer** cycles through warps: W0, W1, W2, W3, W0, ...
- Each cycle, the fetch unit starts at the current pointer position and **scans forward** through warps (in round-robin order) to find the first eligible warp — one that is active and whose instruction buffer will not be full after the decode stage commits any pending entry. Specifically, if the decode stage has a pending instruction targeting warp W and W's buffer has only one free slot, the fetch unit treats W's buffer as full. If an eligible warp is found, the fetch unit reads the instruction BRAM at that warp's PC. If no eligible warp is found in the scan, no fetch occurs that cycle.
- The round-robin pointer always advances to `(original_position + 1) % num_warps` regardless of which warp (if any) was fetched. This ensures fairness: the scan starting point rotates uniformly even when some warps are skipped.
- **Static branch prediction:** after each successful fetch, the frontend predicts the warp's next PC using a fixed policy:
  - backward conditional branches: **taken**
  - forward conditional branches: **not taken**
  - `JAL`: **taken** to `PC + imm`
  - `JALR`: **predicted as fall-through to `PC + 4`** (register-indirect target is unknown at fetch time; always mispredicted, resolved at execute with full refill penalty)
- Each warp maintains its own **program counter (PC)**, updated by the fetch unit after each successful fetch to either the predicted target or `PC + 4`.

**Decode stage:**
- Takes the raw instruction word from the fetch stage output register and decodes it (opcode, register indices, immediate, functional unit target, operand count).
- The decoded instruction is placed into the **per-warp instruction buffer** of the warp it belongs to.
- **Panic detection:** if the decoded instruction is EBREAK, the decode stage triggers the panic sequence (§4.8) instead of placing the instruction into the buffer.

**Per-warp instruction buffer:**
- **Depth:** Parameterizable (default: 3 entries). Implemented as a small FIFO per warp.
- With 4 warps and strict round-robin, each warp receives a new decoded instruction every 4 cycles (assuming no skips). A depth of 3 provides more tolerance for fetch stalls and branch shadow periods, keeping the warp scheduler fed even when the frontend experiences transient disruptions.
- The warp scheduler (§4.3) consumes from the head of this FIFO.

**Decode backpressure:** If the decode stage holds a decoded instruction that cannot be pushed to its target warp's instruction buffer (buffer full), the fetch stage is **stalled** — no new instruction is fetched and no warp PC is updated until the decode stage commits the pending instruction. This prevents instruction loss from unconsumed fetch outputs being overwritten. During a decode stall, the fetch round-robin pointer still advances per the standard rule. Backpressure stalls count toward `fetch_skip_count` statistics.

**EBREAK and decode backpressure:** When the decode stage has a pending instruction stalled on a full warp buffer, EBREAK detection is deferred until the pending instruction commits and frees the decode stage. The stalled instruction takes priority because it was fetched first and must not be lost. The EBREAK instruction remains in the fetch output register and will be decoded once the stall clears.

**Branch handling:**
- Branches, `JAL`, and `JALR` are resolved in the execute stage.
- If the predicted next PC matches the actual next PC, execution continues with no frontend recovery penalty.
- On a misprediction:
  1. The warp's instruction buffer is **flushed** (all entries invalidated).
  2. The warp's PC is updated to the **actual next PC** (branch target for taken control flow, `PC + 4` for fall-through).
  3. Fetching for that warp resumes at the new PC on the next round-robin slot.
- The mispredict penalty is the refill latency: the warp has an empty buffer until fetch and decode deliver new instructions (minimum 2 cycles for the first instruction, plus round-robin wait). Other warps continue executing during this time, hiding the penalty.

**Steady-state timing (4 warps):**
```
Cycle:   1     2     3     4     5     6     7     8     9
Fetch:   W0    W1    W2    W3    W0    W1    W2    W3    W0
Decode:  --    W0    W1    W2    W3    W0    W1    W2    W3
Buffer:        W0←   W1←   W2←   W3←   W0←   W1←   W2←   W3←
```

### 4.3 Warp Scheduler (Issue Stage)

- **Policy:** Loose round-robin (scan-from-pointer).
- A round-robin pointer increments by 1 each cycle. The scheduler scans from the pointer position through the warp vector to find the **first eligible** warp. The issued warp may differ from the pointer position — this is the "loose" aspect.
- **Eligibility:** `buffer_not_empty AND no_branch_in_flight AND scoreboard_clear AND operand_collector_free AND target_unit_ready`
- **Branch shadow stall:** When the scheduler issues a branch, JAL, or JALR instruction, it sets a per-warp `branch_in_flight` flag. While set, the warp is ineligible for scheduling — this prevents shadow instructions (fetched speculatively after an unresolved branch) from being issued and executing in the functional model. The flag is cleared when the operand collector dispatches the branch instruction and the branch is resolved (whether correctly predicted or mispredicted). With the current 1-cycle operand collector for branches, the flag is typically set and cleared within the same cycle; it serves as a safety net against issuing bad-path instructions if the operand collector latency changes.
- **Scoreboard check at issue includes all source operands.** For standard 2-operand instructions, rs1 and rs2 must not be pending. For 3-operand instructions like `VDOT8`, rs1, rs2, and rd must all not be pending.
- If no warp is eligible, no instruction issues that cycle.
- The RR pointer always advances to `(original_position + 1) % num_warps` regardless of which warp (if any) actually issued. This ensures fairness: the scan starting point rotates uniformly. Both fetch and scheduler use the same pointer-advance rule.
- **Implementation:** Combinational priority scan over 4–8 eligible bits with rotating base. Trivial at this width.

### 4.4 Operand Collection Stage

A pipeline stage between issue and functional unit dispatch. It reads source operands from the register file and assembles a complete operand set before handing off to the target dispatch controller.

- **Single slot:** the operand collector holds one warp's instruction at a time. While occupied, the warp scheduler cannot issue another instruction (the `operand_collector_free` signal in the eligibility check gates this).
- **Variable latency by operand count:**
  - **2-operand instructions** (ALU, MUL, DIV, loads, stores, branches): reads rs1 and rs2 simultaneously using the register file's two read ports. Completes in **1 cycle**.
  - **3-operand instructions** (currently `VDOT8`): reads rs1 and rs2 in cycle 1, reads rd (accumulator) in cycle 2. Completes in **2 cycles**. The rule is operand-count-driven: any future 3-operand instruction would also take 2 cycles.
- After collection completes, the full operand set (warp tag + decoded instruction + operand values) is handed to the target functional unit's dispatch controller, and the operand collector becomes free.
- **Stores** read rs1 (base address) and rs2 (store data) — 2 register reads, 1 cycle. The offset is an immediate from the instruction word, not a register read.
- **0- and 1-operand instructions** (ECALL, EBREAK, CSR reads, LUI, AUIPC): complete operand collection in **1 cycle**. The 2-cycle path applies only to 3-operand instructions; all others use the 1-cycle path regardless of actual operand count.

### 4.5 Functional Unit Dispatch

- Each functional unit group (ALU, multiply/VDOT8, divide, LD/ST, TLOOKUP) has its **own local dispatch controller**.
- The operand collector hands off (warp tag + decoded instruction + operand values) to the target unit's dispatch controller.
- The local dispatch controller drains 32 threads through its execution units over multiple cycles and asserts a **busy signal** back to the global scheduler.
- This enables overlapped execution: e.g., one warp draining through ALU while another warp's lookup dispatches to TLOOKUP simultaneously.

### 4.6 Execution Units

All units use a **valid-in / valid-out interface**. Warp tag and destination register ID travel alongside data through the pipeline.

| Unit            | Behavior                                                        | Latency                     |
|-----------------|-----------------------------------------------------------------|-----------------------------|
| ALU             | Add, subtract, shift, bitwise logic                             | 1 cycle                     |
| Multiply/VDOT8  | Pipelined; accepts new op every cycle; VDOT8 uses same DSP slices with packed INT8 routing | Parameterizable (`STAGES`, default: 3) |
| Divide          | Iterative (radix-2); busy until complete; follows stock RV32M div-by-zero behavior (DIV/0 → −1, REM/0 → dividend) | 32 cycles (all operand values, including div-by-zero) |
| LD/ST           | Address generation only; decoupled from cache via FIFO          | Variable (see §5)           |
| TLOOKUP         | Pipelined dual-port BRAM read; bypasses data cache              | 17 cycles (2 lanes/cycle pipelined) |

### 4.7 Writeback

**Parallel bank writes:** The register file has 32 independent banks (one per thread lane), each with its own write port. Since all threads in a warp map to distinct banks, an entire warp's results (32 × 32-bit values) can be written to the register file **in a single cycle** — all 32 banks are written simultaneously.

**Per-unit writeback buffer:** Each execution unit (ALU, multiply/VDOT8, divide, TLOOKUP, MSHR fill path) has a **single-entry buffer** that holds one complete warp result (32 values + warp tag + destination register ID). The dispatch controller collects results as threads exit the execution unit; once all 32 threads have completed, the buffer is marked valid.

**Writeback arbiter:**
- **Round-robin** among execution units with a valid writeback buffer.
- Each cycle, the arbiter selects one unit. That unit's 32 results are written to all 32 register file banks simultaneously (1 cycle).
- Units not selected hold their results in their buffer until the next arbitration cycle.
- The round-robin pointer advances past the selected unit, ensuring fairness.
- **Conflict frequency is low:** conflicts only occur when two or more units finish their last thread lane in the same cycle. With different pipeline depths (1-cycle ALU, multi-cycle multiply, ~32-cycle divide, variable MSHR fill), simultaneous completions are infrequent.

**Scoreboard interaction:** The scoreboard uses **double-buffered state**. When the writeback arbiter selects a unit, the scoreboard's pending flag for the destination register is cleared in the "next" buffer. At the end of the cycle, the next buffer is committed to the current buffer. This means scoreboard clears are visible to the scheduler at the **next cycle boundary**, not in the same cycle as the writeback. The clear is triggered by the arbiter's selection, not by the unit's buffer becoming valid.

**No forwarding:** There is no bypass path from writeback to the operand collector. The scoreboard prevents a warp from issuing an instruction that reads a pending register. Once the write commits and the scoreboard clears (visible next cycle), the warp becomes eligible on the next scheduler scan. There is a minimum **1-cycle gap** between a write committing and the dependent warp reading the result (scoreboard clear → next scheduler cycle → operand collect reads committed value). This gap is acceptable; warp-switching hides it.

### 4.8 Panic Mechanism

EBREAK is a **panic instruction** that immediately halts the entire SM and captures diagnostic state for host inspection. Unlike ECALL (which marks only the executing warp inactive), EBREAK is a fatal, whole-SM halt.

#### 4.8.1 Panic Sequence

When the decode stage identifies an EBREAK instruction, the SM performs the following sequence atomically from the perspective of the host:

1. **Assert panic-pending.** The warp scheduler is immediately inhibited — no new instructions are issued to any warp.
2. **Read cause register.** The operand collector is commandeered for a single-cycle read of `r31` (lane 0) from the panicking warp's register file bank. Register `r31` holds the software-defined cause code (see §4.8.3).
3. **Latch diagnostics.** The panic diagnostic registers (§6.1) are written:
   - `PANIC_WARP` ← warp ID of the panicking warp
   - `PANIC_LANE` ← 0 (EBREAK is detected at decode, before per-lane execution; since all threads in a SIMT warp execute the same instruction, lane 0 is reported)
   - `PANIC_PC` ← program counter of the EBREAK instruction
   - `PANIC_CAUSE` ← value read from `r31`
4. **Assert global panic wire.** The single-bit `panic` signal propagates to the SM top-level controller.
5. **Drain in-flight instructions.** Instructions already dispatched to execution units are allowed to complete and write back. Their results are architecturally irrelevant (the SM is halting), but allowing drain avoids the complexity of mid-pipeline squash logic.
6. **Mark all warps inactive.** After drain completes, all warps (not just the panicking warp) are marked inactive.
7. **Signal host.** `STATUS.PANIC` (bit 2) and `STATUS.DONE` (bit 1) are both set. The SM is now idle.

**Timing:** Steps 1–4 take 2 cycles (1 cycle for decode + panic-pending assertion, 1 cycle for r31 read + latch + panic assertion). Step 5 takes a variable number of cycles, bounded by `MAX_DRAIN_CYCLES = 32`. The drain covers **execution units and writeback only** — in-flight external memory requests (MSHR fills, write buffer drains) are abandoned for architectural purposes and are not part of the drain-completion criterion. Cache, coalescer, and memory submodels may continue advancing internal timing state for observability, but they must not produce any new architecturally committed effects after panic is active. Steps 6–7 occur once drain completes or the drain timeout is reached. The host sees `STATUS.DONE` and `STATUS.PANIC` set simultaneously.

**Priority:** EBREAK is detected at the decode stage, which processes one instruction per cycle. With single-issue decode, two warps cannot have EBREAK detected in the same cycle. If future hardware panic sources (§4.8.2) can assert simultaneously from multiple warps, the **lowest-numbered warp** wins the diagnostic latch via a fixed-priority mux.

#### 4.8.2 Panic Signal Microarchitecture

A single-bit global `panic` wire connects all potential panic sources to the SM top-level controller. In Phase 1, the only source is EBREAK at the decode stage; the wire is designed for extensibility.

| Source   | Detection point | Signals provided to panic latch                                                    |
|----------|-----------------|-------------------------------------------------------------------------------------|
| EBREAK   | Decode stage    | warp_id, lane_id (hardwired 0), PC, r31 value (read from register file)            |

**Extensibility:** Future hardware panic sources (e.g., illegal instruction detection, MSHR timeout watchdogs) connect to the same global `panic` wire with their own cause codes in the reserved 0x01–0xFF range. Adding a new source requires only a new input to the priority mux and a defined cause code — no changes to the panic sequencing logic.

#### 4.8.3 Panic Cause Codes

Cause codes occupy the full 32-bit `PANIC_CAUSE` register. Values 0x0000_0000 through 0x0000_00FF are reserved for hardware-defined causes (none in Phase 1; reserved for future use). Values 0x0000_0100 and above are available for software use.

| Code                       | Source   | Meaning                                               |
|----------------------------|----------|-------------------------------------------------------|
| 0x0000_0000                | Software | Generic / unspecified panic (software did not set r31) |
| 0x0000_0001–0x0000_00FF    | —        | Reserved for future hardware-defined causes            |
| ≥ 0x0000_0100              | Software | Application-defined                                   |

To trigger a panic with a meaningful cause code, software writes the cause value to register `r31` before executing EBREAK:

```
LI   r31, 0x100     # application-defined cause: e.g., "assertion failed"
EBREAK               # panic — SM halts, PANIC_CAUSE latched from r31
```

If software does not write `r31` before EBREAK, `PANIC_CAUSE` will contain whatever value `r31` held at the time (which is 0 if registers were not otherwise modified since launch, per §6.3).

**Register `r31` convention:** `r31` (aliased `t6` in the standard RISC-V ABI) is designated as the panic cause register by software convention only. The hardware reads `r31` of the panicking thread's lane unconditionally on any EBREAK — no ISA encoding change is needed.

---

## 5. Memory System

### 5.1 Load/Store Address Generation

- 8 LD/ST units (parameterizable) compute effective addresses (base + offset) for a warp's 32 threads over multiple cycles.
- Results are written into a **decoupling FIFO** (4 entries, parameterizable). Each entry holds: 32 addresses, warp ID, destination register info, operation type (load/store).
- The address generation side fills an entry and moves on; the cache side consumes entries asynchronously.

### 5.2 Coalescing Unit

- Sits between the address generation FIFO and the L1 cache.
- Pulls one warp's 32-address batch from the FIFO.
- **Blocking per-warp:** the coalescing unit completes all transactions for one warp before pulling the next FIFO entry. The FIFO absorbs rate mismatches.
- **Phase 1 strategy (all-or-nothing):** checks if all 32 thread addresses fall within a single 128-byte cache line. If yes → single cache line request. If no → falls back to 32 serialized individual requests.
- **Future optimization (Phase 2+):** proper cache-line grouping — sort addresses by cache line, issue one request per unique line touched. This requires comparator/sorting logic but dramatically reduces requests for strided access patterns.
- Badly uncoalesced accesses naturally slow down, matching real GPU performance characteristics.
- **Serialized load writeback:** When a load is serialized into 32 individual cache requests, a **single writeback** is produced for the entire warp. The first lane's cache interaction is designated as the writeback source: a cache hit on the first lane produces an immediate writeback carrying all 32 lanes' results; a cache miss on the first lane produces a writeback when the corresponding MSHR fill completes. Subsequent lanes' cache interactions install cache lines and update hit/miss statistics but suppress duplicate writebacks. This ensures one scoreboard clear and one register file write per serialized load instruction.
- **Writeback suppression mechanism:** Each MSHR entry carries a `suppress_writeback` flag. The coalescing unit sets this flag to `false` for the first lane and `true` for all subsequent lanes. On a cache hit, the flag controls whether a writeback entry is generated. On a cache miss, the flag is stored in the allocated MSHR entry and checked when the fill completes: fills with `suppress_writeback = true` install the cache line but skip writeback generation and scoreboard clear. This ensures that even when fills complete out of order (e.g., the first lane's MSHR fills last), exactly one writeback is produced and the scoreboard remains pending until the designated first-lane fill arrives.

### 5.3 L1 Data Cache

| Parameter         | Value                              |
|-------------------|------------------------------------|
| Size              | 4 KB (parameterizable)             |
| Associativity     | Direct-mapped                      |
| Line size         | 128 bytes                          |
| Write policy      | Write-through + write-allocate     |
| Allocation policy | Allocate on any miss (read or write) |
| MSHRs             | Parameterizable (default: 4)       |

**Read hit:** Cache line is present. The cache extracts the requested bytes for each thread (using the thread mask and per-thread byte offsets within the line) and routes values to the register file via writeback.

**Read miss:** A free MSHR is allocated (see §5.3.1). The request is sent to external memory via the memory interface (§5.6). The requesting warp's destination register is marked pending in the scoreboard; the warp continues executing non-dependent instructions (stall-on-use).

**Write hit:** The cache line is updated in L1 **and** the full cache line is pushed into the write buffer for draining to external memory (write-through).

**Write miss (write-allocate):** A free MSHR is allocated. The cache line is fetched from external memory, installed in L1, then the store data is written into the newly installed line. The updated full cache line is pushed to the write buffer for write-through to external memory. This ensures that a subsequent load to the same address will hit in L1 and see the correct data, eliminating the read-after-write hazard across cache misses.

#### 5.3.1 MSHRs (Miss Status Holding Registers)

Each MSHR entry tracks one outstanding cache line fetch from external memory (for both load misses and store misses due to write-allocate).

**MSHR entry contents:**

| Field             | Width                | Description                                      |
|-------------------|----------------------|--------------------------------------------------|
| `valid`           | 1                    | Entry is active                                  |
| `cache_line_addr` | Address bits         | Cache line address of the miss                   |
| `is_store`        | 1                    | 0 = load miss, 1 = store miss (write-allocate)   |
| `warp_id`         | log2(max warps)      | Requesting warp                                  |
| `dest_reg`        | 5                    | Destination register index (loads only)           |
| `thread_mask`     | 32                   | Which threads in the warp need data from this line |
| `byte_offsets`    | 32 × offset bits     | Per-thread byte offset within the cache line     |
| `store_data`      | 32 × 32 bits         | Per-thread store data (store misses only)         |
| `store_byte_en`   | 32 × 4 bits          | Per-thread byte enables (store misses only)       |

**Allocation:** On any cache miss (read or write), the cache allocates the first free MSHR. If no MSHR is free, the coalescing unit **stalls** (backpressure) until one becomes available.

**No duplicate detection or merging:** If two warps (or two serialized requests from the same warp) miss on the same cache line, each gets its own MSHR and generates its own external memory request. This trades redundant bandwidth for simpler hardware (no CAM lookup across MSHRs).

**Load miss fill path:** When external memory returns a cache line for a load miss:
1. The MSHR's `cache_line_addr` is used to install the line in L1.
2. The cache then extracts per-thread values using the MSHR's `thread_mask` and `byte_offsets`.
3. Extracted values are routed to the register file (warp_id, dest_reg) via the writeback path.
4. The scoreboard clears the pending flag for the destination register.
5. The MSHR entry is freed.

**Store miss fill path (write-allocate):** When external memory returns a cache line for a store miss:
1. The line is installed in L1.
2. The store data (`store_data`, `store_byte_en`) is written into the cache line at the appropriate byte positions.
3. The full updated cache line is pushed to the write buffer for write-through to external memory. **If the write buffer is full, the fill stalls until space is available.** While stalled, no other MSHR fills (including load-miss fills) can be processed — this creates potential cascading stalls under heavy store-miss traffic with a full write buffer.
4. The MSHR entry is freed.
5. No register writeback or scoreboard update is needed (stores don't write to the register file).

#### 5.3.2 Write Buffer

A FIFO buffer between the L1 cache and the external memory interface, absorbing write-through traffic so stores don't stall the pipeline waiting for external memory.

| Parameter | Value                          |
|-----------|--------------------------------|
| Depth     | Parameterizable (default: 4)   |
| Entry     | Cache line address + full cache line data (128 bytes) |
| Drain     | FIFO order (oldest first)      |

- On a store (hit or after write-allocate fill completes), the full updated cache line is pushed to the back of the write buffer. If the buffer is full, the store (and the coalescing unit) **stalls** until an entry drains.
- The write buffer drains to external memory via the same memory interface (§5.6) as cache miss reads. Within a cycle, cache miss read submissions are processed before write buffer drain submissions. The external memory interface processes requests in FIFO order with no read/write priority distinction.
- **No write coalescing:** each store generates its own buffer entry. Multiple stores to the same cache line are not merged. (Phase 2 optimization candidate.)

### 5.4 Memory Ordering

- **Stall-on-use:** a warp continues executing after issuing a load; it stalls only when a subsequent instruction reads the load's destination register before data arrives.
- The scoreboard marks load destinations as pending on issue and clears them when the MSHR fill completes and data is written to the register file.
- No store buffer or store-to-load forwarding. Stores update L1 (if hit) and write through to external memory. Within a single warp, program order ensures correctness.
- **Read-after-write correctness:** the write-allocate policy ensures that stores always install a cache line in L1 (fetching it first on a miss). A subsequent load to the same address will hit in L1 and see the updated value. No RAW hazard exists across cache misses.

### 5.5 Instruction Memory

See §4.1. Instruction memory is a preloaded BRAM, separate from the data memory path. No instruction cache or miss handling.

### 5.6 External Memory Interface

The SM connects to external memory (DDR3/DDR4 on the FPGA board) to serve L1 data cache misses and write-through traffic.

**Bus protocol:** Avalon Memory-Mapped for Phase 1, targeting Intel/Altera FPGA memory controller IP. The interface is isolated behind a **bus wrapper module** (`mem_bus_wrapper`) so the protocol can be swapped (e.g., to AXI4 for Xilinx) by replacing only the wrapper, with no changes to SM-internal logic.

**Interface signals (SM side of wrapper):**

| Signal              | Direction | Width          | Description                                    |
|---------------------|-----------|----------------|------------------------------------------------|
| `mem_req_valid`     | Out       | 1              | SM requests a memory transaction               |
| `mem_req_ready`     | In        | 1              | External memory accepts the request            |
| `mem_req_addr`      | Out       | 32             | Byte address                                   |
| `mem_req_write`     | Out       | 1              | 0 = read, 1 = write                            |
| `mem_req_wdata`     | Out       | Cache line width | Write data (one cache line)                   |
| `mem_resp_valid`    | In        | 1              | Read data available                            |
| `mem_resp_data`     | In        | Cache line width | Read data (one cache line)                    |

- Transactions are **cache-line granularity** (128 bytes). The wrapper handles burst conversion to match the external memory controller's native burst size.
- Valid/ready handshake: the SM asserts `req_valid` with address and data; it holds until `req_ready` is asserted. Responses return asynchronously via `resp_valid`.
- The wrapper module translates this simple interface to the target bus protocol (Avalon burst transactions, AXI4 AR/AW/R/W/B channels, etc.).

**Simulation defaults:** External memory latency is parameterizable (default: **100 cycles**). External memory size is parameterizable (default: **64 MB**). These defaults model a representative DDR3/DDR4 access pattern for FPGA prototyping.

---

## 6. Host Interface and Kernel Launch

The SM is controlled by an external host CPU (soft-core on FPGA or external processor) via a **memory-mapped control/status register (CSR) block** accessible over a simple bus (Avalon-MM slave or AXI4-Lite slave, matching the platform).

### 6.1 Control/Status Register Map

| Offset | Name              | R/W | Description                                                       |
|--------|-------------------|-----|-------------------------------------------------------------------|
| 0x00   | `CTRL`            | R/W | Bit 0: START (write 1 to launch). Bit 1: RESET (write 1 to reset SM, including all panic state). |
| 0x04   | `STATUS`          | R   | Bit 0: BUSY (1 while kernel is running). Bit 1: DONE (1 when all warps inactive). Bit 2: PANIC (1 if SM halted due to a panic; when set, DONE is always also set). |
| 0x08   | `NUM_WARPS`       | R/W | Number of active warps (1–N, where N is the parameterized max).   |
| 0x0C   | `START_PC`        | R/W | Starting program counter for all warps.                           |
| 0x10   | `DMA_SRC_ADDR`    | R/W | External memory source address for DMA transfer.                  |
| 0x14   | `DMA_LENGTH`      | R/W | Number of bytes to DMA.                                           |
| 0x18   | `DMA_CTRL`        | R/W | Bit 0: DMA_START. Bit 1: DMA_BUSY (read-only). Bit 2: DMA_DONE (read-only). Bits 4–5: DMA_TARGET (00 = instruction BRAM, 01 = lookup table BRAM). |
| 0x20–0x2C | `ARG0`–`ARG3` | R/W | Kernel argument values. Preloaded into registers r1–r4 of every warp's every thread before launch. |
| 0x30   | `PANIC_WARP`      | R   | Warp ID of the panicking warp (0 to NUM_WARPS−1). Undefined when `STATUS.PANIC` is not set. |
| 0x34   | `PANIC_LANE`      | R   | Lane ID of the panicking thread (0–31). Undefined when `STATUS.PANIC` is not set. |
| 0x38   | `PANIC_PC`        | R   | Program counter of the instruction that triggered the panic. Undefined when `STATUS.PANIC` is not set. |
| 0x3C   | `PANIC_CAUSE`     | R   | Cause code: software-defined via r31, or hardware-defined (see §4.8.3). Undefined when `STATUS.PANIC` is not set. |

### 6.2 Kernel Launch Sequence

The host follows this sequence to launch a kernel:

1. **Load program:** Write `DMA_SRC_ADDR`, `DMA_LENGTH`, and set `DMA_TARGET = 00` (instruction BRAM) in `DMA_CTRL`. Set `DMA_START`. Poll `DMA_DONE` until complete.
2. **Load lookup tables:** Write `DMA_SRC_ADDR`, `DMA_LENGTH`, and set `DMA_TARGET = 01` (lookup table BRAM) in `DMA_CTRL`. Set `DMA_START`. Poll `DMA_DONE` until complete.
3. **Configure kernel:** Write `NUM_WARPS`, `START_PC`, and `ARG0`–`ARG3`.
4. **Launch:** Write 1 to `CTRL.START`. The SM initializes all active warps (sets PCs, preloads argument registers, clears scoreboards) and begins fetch/decode/execute.
5. **Wait for completion:** Poll `STATUS`. When `DONE` is set:
   - If `PANIC` is clear: normal completion. Proceed to step 6.
   - If `PANIC` is set: kernel panicked. Read `PANIC_WARP`, `PANIC_LANE`, `PANIC_PC`, and `PANIC_CAUSE` for diagnostics. Handle error (log, reset SM via `CTRL.RESET`, retry, etc.).
6. **Read results:** Results are in external memory (written by store instructions during execution). The host reads them via the normal memory interface.

### 6.3 Kernel Arguments

- Registers **r1–r4** of every thread in every active warp are preloaded with `ARG0`–`ARG3` before launch.
- These typically hold base addresses for input/output buffers, matrix dimensions, or other scalar parameters.
- Register **r0** remains hardwired to 0 per RV32I convention.
- Registers r5–r31 are initialized to 0.

### 6.4 Thread Identity

Threads discover their position via **read-only CSRs** using the standard RISC-V `CSRRS` instruction (Zicsr extension):

| CSR Address | Name         | Description                                         |
|-------------|--------------|-----------------------------------------------------|
| 0xC00       | `warp_id`    | Warp index (0 to NUM_WARPS−1) for the executing warp |
| 0xC01       | `lane_id`    | Thread lane index (0–31) within the warp            |
| 0xC02       | `num_warps`  | Total number of active warps                        |

A thread computes its global thread ID as: `global_id = warp_id × 32 + lane_id`.

**Supported encoding restriction:** Phase 1 supports only `CSRRS rd, csr, x0` for these identity CSRs. `CSRRW`, `CSRRC`, and any `CSRRS` with nonzero `rs1` are unsupported encodings.

**Pipeline routing:** CSR read instructions are routed through the **ALU** execution unit with **1-cycle latency**, consistent with their simple register-read semantics. No dedicated CSR unit exists in the pipeline.

### 6.5 Warp Completion and Panic

- A warp signals **normal completion** by executing the **ECALL** instruction (standard RV32I, opcode `1110011`). ECALL flows through the full pipeline: it is decoded and buffered normally, issued by the warp scheduler (which triggers functional execution), passes through operand collection (1 cycle, no source operands), and reaches the dispatch stage. At dispatch, the hardware marks the warp **inactive**. ECALL has no destination register and produces no writeback. An inactive warp is excluded from fetch round-robin and warp scheduler eligibility. When **all active warps** are inactive, the SM sets `STATUS.DONE` and halts.
- A warp signals a **panic** by executing the **EBREAK** instruction (standard RV32I, opcode `1110011`). On EBREAK, the hardware halts the **entire SM** — all warps, not just the executing warp — and latches diagnostic state into the panic registers (§6.1). See §4.8 for the full panic sequence. When a panic occurs, `STATUS.DONE` and `STATUS.PANIC` are both set.
- ECALL and EBREAK are the only supported environment calls. No trap handler or exception vector.

### 6.6 Reset Behavior

Writing 1 to `CTRL.RESET` resets the entire SM to its initial state:

- All warps are marked inactive; PCs, instruction buffers, and scoreboards are cleared.
- `STATUS` is zeroed (BUSY, DONE, PANIC all cleared).
- `PANIC_WARP`, `PANIC_LANE`, `PANIC_PC`, `PANIC_CAUSE` are zeroed.
- Pipeline state (operand collector, dispatch controllers, writeback buffers) is flushed.
- The L1 cache, MSHRs, and write buffer are invalidated/cleared.
- Instruction BRAM and lookup table BRAM contents are **not** cleared (they persist across resets; the host reloads them only if the program or tables change).

The SM is ready for a new kernel launch after reset completes.

### 6.7 DMA Engine

A simple DMA engine transfers data from external memory into on-chip BRAMs (instruction BRAM or lookup table BRAM) before kernel launch. It is controlled by the host via the DMA CSRs (§6.1).

**Architecture:** A state machine with no internal FIFO. It reads fixed-size bursts from external memory and writes sequentially to the target BRAM.

**Operation:**
1. Host writes `DMA_SRC_ADDR` (external memory source), `DMA_LENGTH` (byte count), and `DMA_TARGET` (00 = instruction BRAM, 01 = lookup table BRAM) into the CSRs.
2. Host sets `DMA_START` in `DMA_CTRL`.
3. The DMA state machine issues burst read requests to external memory via the memory interface (§5.6), starting at `DMA_SRC_ADDR`.
4. As each burst returns, the data is written sequentially to the target BRAM, starting at BRAM address 0.
5. The state machine continues until `DMA_LENGTH` bytes have been transferred, then asserts `DMA_DONE`.

**Parameters:**

| Parameter         | Value                              |
|-------------------|------------------------------------|
| Burst size        | 32 bytes (fixed)                   |
| FIFO              | None (direct state machine)        |
| Target BRAMs      | Instruction BRAM, lookup table BRAM |
| Arbitration       | DMA has exclusive memory access before kernel launch (no contention with cache) |

- The DMA engine operates **only before kernel launch** and is idle during execution. There is no arbitration with the L1 cache for the external memory interface because the SM is not executing when DMA is active.
- The 32-byte burst size keeps address logic simple (5-bit aligned offsets). Multiple bursts are issued sequentially until the full transfer is complete.
- The DMA engine writes to the target BRAM's write port, which is otherwise idle at runtime (instruction BRAM and lookup table BRAM are read-only during execution).

---

## 7. Register File

- **32 banks**, one per thread lane in a warp. Thread lane N in every warp maps to bank N.
- Each bank provides **2 read ports + 1 write port** (via BRAM replication — see FPGA Implementation Notes).
- Each bank stores all warps' copies of its thread lane's registers. Bank address = {warp_id, register_index}.
- **Single-cycle operand read:** both source operands A and B for all 32 thread lanes are read simultaneously (one read per bank × 2 read ports × 32 banks) in the operand collection stage.
- **Single-cycle warp writeback:** all 32 banks are written simultaneously (one write per bank), committing an entire warp's results in 1 cycle. Since threads within a warp map to distinct banks, there are no write port conflicts.
- The scoreboard tracks pending writes per (warp, register) pair.
- **No forwarding/bypass.** Reads always see committed register file state. The scoreboard ensures no read occurs while a write is pending.

---

## 8. Scoreboard

- Tracks in-flight writes to the register file via a **pending bit** per (warp, register) pair.
- **Set pending:** on instruction issue, the destination register is marked pending for the issuing warp.
- **Clear pending:** when the writeback arbiter selects a unit and the write commits to the register file (not when the unit's buffer becomes valid — only on actual commit).
- **Issue-time check covers all source operands:** for 2-operand instructions, rs1 and rs2 must not be pending. For 3-operand instructions (`VDOT8`), rs1, rs2, and rd must all not be pending. A warp is ineligible if any required source register is pending.
- **Load destinations:** marked pending on load issue; cleared when the MSHR fill completes and data is written to the register file via the writeback arbiter.
- **Minimum read-after-write latency:** 1 cycle after scoreboard clear (write commits → scoreboard clears → next scheduler cycle the warp becomes eligible → operand collect reads committed value). Warp-switching hides this gap.
- **Panic interaction:** when a panic is triggered, the scoreboard state is abandoned (not explicitly cleared). The `CTRL.RESET` path handles full cleanup before the next kernel launch.

---

## 9. LLM Inference Support

### 9.1 Core Compute

- INT8 × INT8 multiply with INT32 accumulate — the primary matmul kernel.
- Weights and activations are quantized to INT8 via post-training quantization (GPTQ, AWQ, absmax, zero-point).
- Inter-layer rescaling is integer multiply-and-shift.

### 9.2 Nonlinear Function Approximation

- Implemented via the **TLOOKUP** custom instruction (§2.3), which reads from a dedicated lookup table BRAM bypassing the data cache.
- Table BRAM: parameterizable size (default 4 KB = 1024 × 32-bit entries), host-loaded before kernel launch.
- Default partitioning: 256 entries each for exp (softmax), GELU, SiLU, and reciprocal square root (layer norm).
- **Linear interpolation** between adjacent table entries is performed in software using standard ALU and MUL instructions (see §2.3 for the instruction sequence).
- Covers: softmax (exp), GELU, SiLU, layer normalization (reciprocal square root approximation).

### 9.3 Practical Constraints

- Memory bandwidth is the dominant bottleneck for inference. A 1B-parameter INT8 model requires ~1 GB of weight storage.
- Realistic FPGA demo targets: 100M–300M parameter models.

---

## 10. Phase 1 Exclusions

The following features are explicitly deferred:

| Feature                          | Reason for deferral                                                   |
|----------------------------------|-----------------------------------------------------------------------|
| Shared memory / scratchpad       | Requires banked SRAM, bank conflict handling, barrier hardware        |
| Floating point                   | Integer-only is sufficient for quantized inference                    |
| Special function units (SFU)     | Replaced by lookup tables                                            |
| Divergence handling / reconvergence stack | Adds significant control complexity                         |
| Multiple SMs / inter-SM fabric   | Single SM first; multi-SM is a scaling exercise                       |
| Hardware-triggered panic sources | Phase 1 panic is software-only (EBREAK); cause codes 0x01–0xFF reserved for future hardware faults (e.g., illegal instruction, watchdog timeout) |

---

## 11. Design Principles

1. **Parameterize everything.** Warp count, pipeline depths, cache size, execution unit widths — retarget by changing parameters, not rewriting RTL.
2. **Decouple stages with FIFOs/buffers.** Fetch ↔ decode, decode ↔ schedule, address generation ↔ cache — no single slow stage blocks the whole pipeline.
3. **Uniform execution unit interfaces.** Valid-in / valid-out with warp tags. The scheduler is agnostic to pipeline depths.
4. **Hide latency via warp switching.** Multiple resident warps cover memory, multiply, and divide latency.
5. **FPGA-friendly structures.** DSP slices for multiply, BRAMs for storage, avoid associative matching and multi-ported structures.
6. **Fail loud.** EBREAK panics the entire SM with full diagnostic capture. No silent failure modes — software bugs produce actionable halt state rather than corrupt output.

---

## Appendix A: Resolved Questions

All architectural questions have been resolved. This appendix records the resolution of each item for traceability.

- ~~Custom ISA extensions~~ → Resolved: VDOT8 signed INT8×4 dot-product accumulate using custom-0 opcode (§2.2).
- ~~Operand read for 3-operand instructions~~ → Resolved: operand collection stage with variable latency (§4.4).
- ~~Instruction buffer depth~~ → Resolved: parameterizable, default 3 entries per warp (§4.2).
- ~~Instruction memory architecture~~ → Resolved: preloaded BRAM, no cache (§4.1).
- ~~External memory interface~~ → Resolved: Avalon-MM behind modular wrapper (§5.6).
- ~~Thread block / kernel launch~~ → Resolved: CSR-based host interface with DMA program loading (§6).
- ~~Program loading interface~~ → Resolved: DMA from external memory (§6.1, §6.2, §6.7).
- ~~Interrupt and exception handling~~ → Resolved: no exceptions; ECALL marks warp inactive, EBREAK triggers whole-SM panic with diagnostic capture, host polls STATUS.DONE and STATUS.PANIC (§6.5).
- ~~Coalescing granularity~~ → Resolved: all-or-nothing for Phase 1; proper cache-line grouping deferred to Phase 2 (§5.2).
- ~~MSHR design details~~ → Resolved: parameterizable count (default 4), stall on exhaustion, no duplicate detection/merging, separate fill paths for load and store misses (§5.3.1).
- ~~Writeback arbiter policy~~ → Resolved: round-robin among units with valid output; single-cycle parallel write to all 32 banks (§4.7).
- ~~Register file forwarding~~ → Resolved: no forwarding; scoreboard prevents reads of pending registers; 1-cycle gap after clear is acceptable (§4.7, §8).
- ~~Lookup-table instruction~~ → Resolved: TLOOKUP custom instruction in custom-1 opcode space, I-type encoding, dedicated functional unit, pipelined dual-port BRAM with 17-cycle warp latency, host-loaded BRAM (§2.3).
- ~~VDOT8 overflow behavior~~ → Resolved: wrapping (2's complement), consistent with RV32I convention; software manages range (§2.2).
- ~~VDOT8 encoding details~~ → Resolved: R-type in custom-0, funct7=0000000, funct3=000 (§2.2).
- ~~DSP slice packing for VDOT8~~ → Moved to FPGA Implementation Notes (physical mapping detail, not architecture).
- ~~DMA engine details~~ → Resolved: simple state machine, 32-byte fixed bursts, no FIFO, exclusive memory access before kernel launch (§6.7).
- ~~Read-after-write across cache misses~~ → Resolved: changed to write-allocate policy; store misses fetch the line via MSHR, update L1, then write through. RAW hazard eliminated (§5.3, §5.4).
- ~~EBREAK / panic behavior~~ → Resolved: EBREAK triggers whole-SM panic with diagnostic latch (warp, lane, PC, cause). Software convention: r31 holds cause code. Divide-by-zero follows stock RV32M (no panic). Panic wire extensible for future hardware sources (§4.8).

---

## Appendix B: Performance Validation Contract

This appendix defines which timing behaviors are part of the architectural contract for validation purposes.

### B.1 Exact Contracted Behavior

The following are architecturally contracted and may be used as exact expectations in validation:

- fetch/decode/issue ordering and visibility across cycle boundaries
- static branch-prediction policy and mispredict recovery semantics
- scoreboard set/clear timing
- execution-unit latency parameters
- duplicate-miss behavior for same-line outstanding misses
- write-through drain before completion
- writeback arbitration conflicts delaying scoreboard clear
- DONE/PANIC waiting for full pipeline drain

### B.2 Contracted Validation Surfaces

The following simulator outputs are part of the validation surface:

- `total_cycles`
- `total_instructions_issued`
- `branch_flushes`
- per-unit instruction counts
- cache hit/miss and external-memory traffic counters
- `writeback_conflicts`
- per-warp stall counters when a scenario cites them
- committed `CycleTraceSnapshot` state classifications
- final architected register state
- panic diagnostics (`warp`, `pc`, `cause`)

### B.3 Informational Counters

The following counters are useful diagnostics but are not frozen architectural metrics by default:

- `branch_predictions`
- `branch_mispredictions`
- `fetch_skip_count`

They may be checked by tests when explicitly called out, but they are not the core performance contract.
