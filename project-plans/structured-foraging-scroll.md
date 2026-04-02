# Audit Fix Plan: Spec Ambiguity Resolution, Divergence Fixes, and Doc Scrub

## Context

An audit of the GPU simulator found spec ambiguities, spec-model divergences, and stale documentation. This plan resolves all findings:
- Ambiguities: document decisions in the spec, implement to match
- Divergences: fix implementation to match spec
- Stale docs: scrub comments and documentation

## Changes Overview

| # | Type | Priority | Description | Files |
|---|------|----------|-------------|-------|
| 1 | Divergence fix | HIGH | Decode-to-fetch backpressure (instruction loss bug) | fetch_stage.h/cpp, decode_stage.cpp, timing_model.cpp, test_timing_components.cpp |
| 2 | Divergence fix | HIGH | Serialized loads: single writeback instead of 32 | coalescing_unit.h/cpp, mshr.h, cache.cpp, test_timing_components.cpp |
| 3 | Dead code | LOW | Remove unreachable CSR dispatch in SYSTEM case | timing_model.cpp |
| 4 | Stale comment | LOW | Fix cache store-hit comment | cache.cpp |
| 5 | Stale comment | LOW | Fix coalescing writeback comment (folded into #2) | coalescing_unit.cpp |
| 6 | Spec update | HIGH | Document all ambiguity resolutions | gpu_architectural_spec.md |
| 7 | Doc update | MED | Update perf_sim_arch.md descriptions | perf_sim_arch.md |
| 8 | Cleanup | LOW | Remove empty test_programs/ directory | sim/tests/test_programs/ |

## Execution Order

1. **Spec first** (Change 6) — establish ground truth before changing code
2. **Implementation** (Changes 1, 2, 3, 4, 5) — make code match spec
3. **Build + test** after each implementation change
4. **Doc update** (Change 7) — update perf_sim_arch.md to match new state
5. **Cleanup** (Change 8)

---

## Change 1: Decode-to-Fetch Backpressure

### Problem
When decode's `pending_` buffer is stuck (target warp's instruction buffer full), fetch continues producing instructions and updating PCs. The `commit()` method unconditionally overwrites `current_output_` with `next_output_`, losing the unconsumed instruction. The warp whose instruction was lost has its PC advanced past that instruction — a silent correctness bug.

### Fix

**`sim/include/gpu_sim/timing/fetch_stage.h`** — Add field and method:
```cpp
// Private field:
bool output_consumed_ = true;  // Starts true so first cycle works

// Public method:
void consume_output() { output_consumed_ = true; }
```

**`sim/src/timing/fetch_stage.cpp`** — Three changes:

1. `evaluate()`: Gate production on consumption. After `next_output_ = std::nullopt;`, add:
```cpp
// Backpressure: don't produce if decode hasn't consumed previous output
if (current_output_.has_value() && !output_consumed_) {
    rr_pointer_ = (rr_pointer_ + 1) % num_warps_;  // RR always advances per spec
    return;
}
```
Note: do NOT increment `fetch_skip_count` here — this is decode backpressure, not "no eligible warp."

2. `commit()`: Only overwrite `current_output_` when there's new output or old was consumed:
```cpp
void FetchStage::commit() {
    if (next_output_.has_value()) {
        current_output_ = next_output_;
        output_consumed_ = false;
    } else if (output_consumed_) {
        current_output_ = std::nullopt;
    }
    // else: retain current_output_ for decode to consume
}
```

3. `redirect_warp()`: Reset consumption flag when clearing current_output_ for redirected warp:
```cpp
if (current_output_ && current_output_->warp_id == warp_id) {
    current_output_ = std::nullopt;
    output_consumed_ = true;  // ADD THIS LINE
}
```

4. `reset()`: Add `output_consumed_ = true;` to reset.

**`sim/src/timing/decode_stage.cpp`** — Signal consumption:

In `evaluate()`, after latching into `pending_` (after line 33):
```cpp
fetch_.consume_output();
```

Also in the EBREAK path (after line 21, before return):
```cpp
fetch_.consume_output();
```

**`sim/tests/test_timing_components.cpp`**:
- Update test at ~line 182 ("Fetch skips warp with full buffer") if it asserts fetch produces while decode is blocked
- Add new test: "Fetch stalls when decode has unconsumed output" — set up decode with pending_ stuck on a full buffer, verify fetch does NOT produce or update PCs for other warps

---

## Change 2: Serialized Load Single Writeback

### Problem
Serialized loads call `cache_.process_load()` 32 times (one per lane). Each cache hit produces a `WritebackEntry` with the full 32-lane results. Each MSHR fill also produces one. Result: up to 32 redundant writebacks for one instruction, wasting writeback bandwidth and inflating `writeback_conflicts`.

### Fix
Designate the **first lane** (serial_index_ == 0) as the writeback source. All subsequent lanes perform cache lookups (for timing/allocation) but suppress writebacks.

**`sim/include/gpu_sim/timing/mshr.h`** — Add field to MSHREntry:
```cpp
bool suppress_writeback = false;
```

**`sim/include/gpu_sim/timing/coalescing_unit.h`** — Add tracking field:
```cpp
bool wb_already_produced_ = false;  // Private field
```

**`sim/src/timing/coalescing_unit.cpp`** — Three changes:

1. When starting a new serialized entry (where `processing_` becomes true, ~line 36):
```cpp
wb_already_produced_ = false;
```

2. Serialized load hit path (~lines 75-80): Gate writeback propagation:
```cpp
if (accepted && wb_out.valid) {
    if (!wb_already_produced_) {
        wb_valid = true;
        wb_already_produced_ = true;
    } else {
        wb_out.valid = false;  // Suppress redundant writeback
    }
}
```

3. Serialized load miss path: Before MSHR allocation in cache_.process_load(), we can't directly set the MSHR field. Instead, add a `suppress_writeback` parameter to `process_load()` OR set it post-allocation. Simplest: pass a flag to `process_load()`.

**Alternative (simpler)**: Don't modify `process_load()` signature. Instead, in the coalescing unit, after `process_load()` returns true for a miss (wb_out.valid is false, meaning miss), check `wb_already_produced_`. If true, mark the MSHR. This requires the cache to expose the last-allocated MSHR index, or the coalescing unit to tell the cache to suppress.

**Cleanest approach**: Add `bool suppress_writeback = false` parameter to `L1Cache::process_load()`. The cache passes it through to the MSHR entry on miss allocation.

**`sim/src/timing/cache.cpp`** — Two changes:

1. `process_load()` signature: add `bool suppress_writeback = false` parameter.
   On miss path, set `entry.suppress_writeback = suppress_writeback;` before allocating.

2. `complete_fill()`: After installing line in L1 cache, check:
```cpp
if (!mshr.is_store && !mshr.suppress_writeback) {
    // Produce writeback (existing code)
} else if (!mshr.is_store) {
    // Suppressed: free MSHR, install line, but no writeback
}
```

**`sim/include/gpu_sim/timing/cache.h`** — Update `process_load()` signature to include the new parameter.

**`sim/src/timing/coalescing_unit.cpp`** — Update the serialized `process_load()` call:
```cpp
accepted = cache_.process_load(
    addr, current_entry_.warp_id, current_entry_.dest_reg,
    current_entry_.trace.results, current_entry_.issue_cycle,
    current_entry_.trace.pc, current_entry_.trace.decoded.raw,
    wb_out, wb_already_produced_);  // suppress if first wb already done
```
After the call, if accepted and it was a miss (wb_out.valid is false) and !wb_already_produced_:
```cpp
wb_already_produced_ = true;  // First miss will produce the writeback on fill
```

3. Fix comment at lines 73-74:
```cpp
// For serialized loads, pass the full result array. Only the first
// lane's cache interaction produces a writeback; subsequent lanes
// suppress writebacks via the suppress_writeback flag.
```

**`sim/tests/test_timing_components.cpp`** — Add test:
"Serialized load produces exactly one writeback" — set up a scattered-address load that serializes, run enough cycles for all lanes to complete, verify writeback_conflicts count and that only 1 writeback entry was produced.

---

## Change 3: Remove Dead CSR Dispatch Code

**`sim/src/timing/timing_model.cpp`** lines 240-242:
Remove the unreachable CSR check inside the SYSTEM case:
```cpp
// REMOVE:
if (input.decoded.type == InstructionType::CSR) {
    alu_->accept(input, cycle_);
}
```
CSR instructions have `target_unit = ExecUnit::ALU` (set by decoder at `decoder.cpp:206`), so they match the ALU case at line 221, never reaching the SYSTEM case.

---

## Change 4: Fix Cache Store-Hit Comment

**`sim/src/timing/cache.cpp`** line 89:
Change:
```cpp
// Store hit: update cache line and push to write buffer
```
To:
```cpp
// Store hit: write-through to write buffer (timing model tracks tags only, not data)
```

---

## Change 6: Spec Ambiguity Resolutions

**`resources/gpu_architectural_spec.md`** — All additions below:

### a) §4.2 — Decode backpressure (insert after line 162, before "Branch handling")

Add new paragraph:
> **Decode backpressure:** If the decode stage holds a decoded instruction that cannot be pushed to its target warp's instruction buffer (buffer full), the fetch stage is **stalled** — no new instruction is fetched and no warp PC is updated until the decode stage commits the pending instruction. This prevents instruction loss from unconsumed fetch outputs being overwritten. During a decode stall, the fetch round-robin pointer still advances per the standard rule.

### b) §4.4 — 0/1-operand instruction latency (append after line 200)

Add:
> **0- and 1-operand instructions** (ECALL, EBREAK, CSR reads, LUI, AUIPC): complete operand collection in **1 cycle**. The 2-cycle path applies only to 3-operand instructions; all others use the 1-cycle path regardless of actual operand count.

### c) §4.8.1 — Panic priority clarification (replace line 260)

Replace the current priority paragraph with:
> **Priority:** EBREAK is detected at the decode stage, which processes one instruction per cycle. With single-issue decode, two warps cannot have EBREAK detected in the same cycle. If future hardware panic sources (§4.8.2) can assert simultaneously from multiple warps, the **lowest-numbered warp** wins the diagnostic latch via a fixed-priority mux.

### d) §5.2 — Serialized load writeback behavior (append after line 310)

Add:
> **Serialized load writeback:** When a load is serialized into 32 individual cache requests, a **single writeback** is produced for the entire warp. The first lane's cache interaction is designated as the writeback source: a cache hit on the first lane produces an immediate writeback carrying all 32 lanes' results; a cache miss on the first lane produces a writeback when the corresponding MSHR fill completes. Subsequent lanes' cache interactions install cache lines and update hit/miss statistics but suppress duplicate writebacks. This ensures one scoreboard clear and one register file write per serialized load instruction.

### e) §2.3 — TLOOKUP total latency clarification (modify line 87)

Change:
> **Latency:** 2 cycles per thread lane (cycle 1: BRAM address presented; cycle 2: data out). The dispatch controller drains 32 threads through the BRAM read port over multiple cycles.

To:
> **Latency:** 2 cycles per thread lane (cycle 1: BRAM address presented; cycle 2: data out), with lanes drained serially through the single BRAM read port. Total warp latency: **64 cycles** (2 cycles/lane x 32 lanes). The TLOOKUP unit asserts busy for the full drain duration.

### f) §6.4 — CSR pipeline routing (append after line 467)

Add:
> **Pipeline routing:** CSR read instructions are routed through the **ALU** execution unit with **1-cycle latency**, consistent with their simple register-read semantics. No dedicated CSR unit exists in the pipeline.

### g) §6.5 — ECALL pipeline path (expand line 471)

Replace the current ECALL sentence with:
> A warp signals **normal completion** by executing the **ECALL** instruction (standard RV32I, opcode `1110011`). ECALL flows through the full pipeline: it is decoded and buffered normally, issued by the warp scheduler (which triggers functional execution), passes through operand collection (1 cycle, no source operands), and reaches the dispatch stage. At dispatch, the hardware marks the warp **inactive**. ECALL has no destination register and produces no writeback. An inactive warp is excluded from fetch round-robin and warp scheduler eligibility. When **all active warps** are inactive, the SM sets `STATUS.DONE` and halts.

### h) §5.6 — External memory defaults (append after line 412)

Add:
> **Simulation defaults:** External memory latency is parameterizable (default: **100 cycles**). External memory size is parameterizable (default: **64 MB**). These defaults model a representative DDR3/DDR4 access pattern for FPGA prototyping.

---

## Change 7: Update perf_sim_arch.md

- Verify test count after new tests are added (currently 148; will increase by ~2)
- Update the test descriptions for `test_timing_components` to mention the new backpressure and writeback suppression tests
- Verify all function/method descriptions still match implementation after Changes 1-4

---

## Change 8: Cleanup

- Remove empty directory: `rmdir sim/tests/test_programs/`
- Note to user: run `cmake --build build --target clean` or `rm -rf build` to remove stale `test_alignment` artifacts

---

## Verification

After all changes:
1. `cmake --build build` — must compile cleanly
2. `ctest --test-dir build --output-on-failure` — all tests pass (including new ones)
3. Manual review: grep for "analytical_model", "reference_flow", "audit_matrix", "manifest" — no hits in active code
4. Verify no stale comments remain in modified files
