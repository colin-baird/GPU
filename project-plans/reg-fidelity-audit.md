# `Reg<T>` Fidelity Audit — Cross-Stage `next()` Read Checklist

## Context

The `reg.h` migration (plan: [`project-plans/shiny-meandering-pine.md`](shiny-meandering-pine.md)) wrapped every state-holding member of the timing model in `Reg<T>` / `RegFifo<T>` / `Wire<T>`. The migration was designed **byte-identical**: every read site was faithfully translated to observe the exact value it observed before the wrap — committed reads use `current()`, staged / mid-evaluate reads use `next()` / `next_mut()`. No behavior was changed and no fidelity bug was repaired or introduced.

That is the prerequisite for this audit. With the migration landed, every same-cycle staged read in the timing model is now visible as a `.next()` call site, and a `grep` is enough to enumerate them. The audit's job is to look at each `.next()` read and decide:

- **Legitimate intra-stage staged read.** The producer reads back its own staged value (typically a value it wrote earlier in the same `evaluate()`, e.g. `cycles_remaining_.next_mut()--` then `if (cycles_remaining_.next() == 0)`). Hardware would observe the post-write value because the read and write live on the same logical "wire" inside the same combinational region.
- **Genuine same-cycle hazard — fidelity bug.** A staged read where hardware would observe the pre-stage (committed) value because the read site and the write site are on opposite sides of a clock edge. The fix is to change `.next()` to `.current()`. By the migration's byte-identical contract, every such site is a pre-existing fidelity bug now made visible — not regression.

`next_mut()` writes (staging side) and bare `next()` reads on the **producer's own** field are evidence; cross-class `next()` reads are already lint-enforced as forbidden (`tools/lint_timing_naming.py` cross-module-read layer). What this audit chases are the in-class reads in observer queries (`is_idle()`, debug snapshots) and any in-evaluate reads where the staged value is the wrong choice on the hardware side of the clock edge.

The byte-identical contract means: a `.next()` → `.current()` swap will, by definition, change at least one benchmark cycle if the read site is exercised. That delta is the bug being fixed; the per-bug benchmark gate moves with it.

## Methodology

```
grep -rn "\.next(" sim/src/timing/ sim/include/gpu_sim/timing/
grep -rn "\.next_mut(" sim/src/timing/ sim/include/gpu_sim/timing/
```

Classify each site as one of:

1. **Intra-stage staged write** (`.next_mut() = ...`, `.set_next(...)`) — producer staging a value; not an audit candidate.
2. **Intra-stage staged read after the producer's own write THIS evaluate** (`if (foo_.next() == 0)` after `foo_.next_mut()--` in the same `evaluate()` body) — legitimate by design; not a candidate.
3. **Cross-class read of another stage's `Reg::next()`** — forbidden, already enforced by `tools/lint_timing_naming.py` cross-module-read layer.
4. **Observer query outside `evaluate()`** — `is_idle()`, snapshot accessors, panic-drain queries reading `.next().valid`. **Audit candidate** — should they be `.current()` reads?
5. **Anything else** — report and let the operator decide.

The sections below enumerate every `.next()`-read site in the tree (writes via `.next_mut() = ...` are mentioned only when they pair with an audit-relevant read). Within each phase, the table identifies the file + line, the surrounding context, and the classification. Audit candidates are marked **AUDIT**; obvious-legitimate sites are marked **OK** with the reason.

## Per-stage `next()`-read inventory

### Phase 1 — `Scoreboard` + `BranchShadowTracker`

| File:line | Context | Classification |
|-----------|---------|----------------|
| `sim/include/gpu_sim/timing/scoreboard.h:35` | `bits_.next_mut().pending[warp][reg] = true;` (`set_busy`) | Write only — type (1) staged write. |
| `sim/include/gpu_sim/timing/scoreboard.h:40` | `bits_.next_mut().pending[warp][reg] = false;` (`clear_busy`) | Write only — type (1). |
| `sim/include/gpu_sim/timing/branch_shadow_tracker.h:65-67` | `shadow_.next_mut()[w] = true/false;` (`note_branch_issued`, `note_resolved_correctly`, `note_redirect_applied`) | Write only — type (1). Called from `OperandCollector::resolve_branch` and `FetchStage::apply_redirect`; commit happens at end of cycle. |

No `.next()` reads. **0 audit candidates.**

### Phase 2 — `FetchStage` + `DecodeStage`

| File:line | Context | Classification |
|-----------|---------|----------------|
| `sim/include/gpu_sim/timing/fetch_stage.h:33` | `std::optional<FetchOutput>& output() { return output_.next_mut(); }` — test hook | Write handle — type (1). |
| `sim/src/timing/fetch_stage.cpp:155-156` | `if (output_.next() && output_.next()->warp_id == warp_id) { output_.next_mut() = std::nullopt; }` in `apply_redirect()` | Type (2). Intra-stage redirect-flush of the staged slot, called from the top of `evaluate()`. Paired with `output_.current_mut() = std::nullopt;` at line 152-154 so both slots are cleared on redirect — the documented redirect-flush idiom (`reg.h` `current_mut()` rationale (1)). |
| `sim/src/timing/decode_stage.cpp:53` | `if (pending_.next().valid) return;` at the top of the input-pull guard in `evaluate()` | **AUDIT.** Reads `pending_.next()` to decide whether decode is occupied this cycle. At this point in evaluate, `seed_all()` has already copied `pending_.current()` → `pending_.next()`, so the read is equivalent to `pending_.current()` *unless* something earlier in this same evaluate has mutated `pending_.next()` — and at line 53, nothing has. In hardware, the "occupied this cycle" gate would read the clocked register, i.e. `pending_.current()`. Likely a `.next()` → `.current()` swap candidate; verify by walking the evaluate sequence. |
| `sim/src/timing/decode_stage.cpp:112-113` | `if (pending_.next().valid && pending_.next().target_warp == warp_id) { pending_.next_mut().valid = false; }` in `clear_pending_on_redirect()` | Type (2). The redirect-flush clears both slots; the staged-slot read sees the same value as the committed-slot read because no mutation has occurred yet this cycle other than `seed_all()`. Mirror of `FetchStage::apply_redirect`. |
| `sim/include/gpu_sim/timing/decode_stage.h:102` (comment) | Documents the staged-slot read at `decode_stage.cpp:53`. | Comment only. |

**2 audit candidates** (decode_stage.cpp:53 is the substantive one; decode_stage.cpp:112-113 is a flush mirror — re-examine after deciding line 53).

### Phase 3 — Execution units (ALU / Divide / Multiply / TLookup / LdSt)

The execution units share a near-identical structure: `accept()` stages an arriving instruction into `pending_*`, then `evaluate()` reads the staged values back. Every `.next()` read inside an execution unit's `evaluate()` is type (2) — the read is documented as "the input written THIS cycle by accept() earlier in the tick" (see `alu_unit.cpp:51-58`).

| File:line | Context | Classification |
|-----------|---------|----------------|
| `sim/src/timing/alu_unit.cpp:51` | `processed_this_cycle_ = has_pending_.next();` | Type (2). `accept()` wrote `has_pending_.set_next(true)` earlier in the same tick; the read is the documented same-cycle pickup. |
| `sim/src/timing/alu_unit.cpp:57-58` | `if (has_pending_.next()) { const DispatchInput& in = pending_input_.next(); ... }` | Type (2). Same pattern. The comment at L52-56 is explicit: "Read has_pending_.next() here: this is the input written THIS cycle by accept()". |
| `sim/src/timing/divide_unit.cpp:40-44` | `busy_this_cycle_ = busy_.next(); if (busy_.next()) { if (cycles_remaining_.next() > 0) { cycles_remaining_.next_mut()--; } if (cycles_remaining_.next() == 0) { result_buffer_.set_next(pending_result_.next()); } }` | Type (2). Iterative-unit countdown: `seed_all()` copied current_ → next_, evaluate decrements in place and reads the decremented value. Hardware semantics: the counter is a register, the decrement is combinational, the result reads the combinational output the same cycle. |
| `sim/src/timing/tlookup_unit.cpp:40-44` | Same shape as divide. | Type (2). |
| `sim/src/timing/ldst_unit.cpp:44-49` | `busy_this_cycle_ = busy_.next(); if (busy_.next()) { if (cycles_remaining_.next() > 0) { cycles_remaining_.next_mut()--; } if (cycles_remaining_.next() == 0) { ... } }` | Type (2). Same iterative-unit pattern as divide. |
| `sim/src/timing/ldst_unit.cpp:58` | `next_push_ = pending_entry_.next();` inside the countdown-finished branch | Type (2). Reads the staged pending entry (which `accept()` populated this tick or which was seeded from current) to compose the FIFO push intent. |
| `sim/src/timing/ldst_unit.cpp:59` | `pending_entry_.next_mut().valid = false;` | Type (1) write. |
| `sim/src/timing/multiply_unit.cpp:8, 35` (comments) | Document that `pipeline_.next()` is read as the post-accept staged deque. | Comment only. The actual reads are via `pipeline_.next_mut()` (writes / pop, not bare `.next()`). |

**0 audit candidates** in Phase 3. The "input written THIS cycle by accept()" pattern is the hardware-faithful encoding of "the producer (warp scheduler / opcoll / coalescing) drove its `current_output()` this tick; we are the consumer downstream in the back-to-front sweep, so by the time we run we have already seen the committed value through `accept()`." A `.next()` → `.current()` swap here would re-introduce the +1-cycle lag that the pull model removed.

### Phase 4 — `WarpScheduler` + `OperandCollector`

| File:line | Context | Classification |
|-----------|---------|----------------|
| `sim/src/timing/operand_collector.cpp:32` | `if (scheduler_ != nullptr && !busy_.next()) { ... accept(*issued); }` — top of `evaluate()`, gate on whether opcoll has room | Type (2). At this point in evaluate, `seed_all()` has copied `busy_.current()` → `busy_.next()` and `accept()` has not yet run; the comment at L22-31 documents the gate as "exactly 'opcoll free'" with `!next_busy_`. A hardware register clocks at the cycle edge — same-tick `accept()` writing `busy_.set_next(true)` then this gate reading `busy_.next()` is the same evaluate, so the read sees the seeded value (effectively `current()` here). **Borderline:** at this exact line the read is byte-identical to `busy_.current()` because no mutation has run yet, but flipping to `.current()` would be a semantic-equivalent cleanup. Lint-driven preference: prefer `.current()` when the read happens before any same-evaluate write. |
| `sim/src/timing/operand_collector.cpp:46-47` | `busy_this_cycle_ = busy_.next(); if (!busy_.next()) return;` | Type (2). At this line, accept() may have run earlier in the same evaluate (L32-37) and set `busy_.set_next(true)`; reading `.next()` here observes that same-cycle stage. Hardware: the new instruction arrived combinationally from opcoll, so the busy gate seeing it this cycle is correct. Keep as `.next()`. |
| `sim/src/timing/operand_collector.cpp:51` | `if (cycles_remaining_.next() == 0)` after the in-place `cycles_remaining_.next_mut()--` at L49 | Type (2). Read-after-write within evaluate; the post-decrement check is the documented countdown idiom. |
| `sim/src/timing/operand_collector.cpp:53-57` | `out.decoded = instr_.next().decoded; out.trace = instr_.next().trace; ...` — copying the staged instr fields when ready | Type (2). `accept()` wrote `instr_.set_next(issue)` earlier this evaluate (when fresh); for in-flight, `seed_all()` copied current → next. Either way, `.next()` is the live in-evaluate value. |
| `sim/src/timing/warp_scheduler.cpp:86` | `return (bitmap_head_.next() + offset) % writeback_bitmap_.next().size();` — `bitmap_slot()` helper | Type (2). Helper called from `evaluate()` *after* `bitmap_head_.next_mut() = (bitmap_head_.next() + 1) % ...` at L117; the helper must observe the post-advance value, so reading `.next()` is mandatory. (The helper is also called from the `test_reserve_writeback_slot` test hook before any evaluate, where the staged slot equals the committed slot anyway — see comment at L83-85.) |
| `sim/src/timing/warp_scheduler.cpp:116-117` | `bitmap_next[bitmap_head_.next()] = std::nullopt; bitmap_head_.next_mut() = (bitmap_head_.next() + 1) % bitmap_next.size();` | Type (2). Reads `bitmap_head_.next()` once to address the wrap-slot to clear, then again as the source for the increment. Hardware-faithful: the seeded value is the "cycle N" head; the post-advance is the "cycle N+1" head — the entire transition is one combinational region. |
| `sim/src/timing/warp_scheduler.cpp:121` | `if (opcoll_cooldown_cycles_.next() > 0) { --opcoll_cooldown_cycles_.next_mut(); }` | Type (2). Countdown decrement. |
| `sim/src/timing/warp_scheduler.cpp:133` | `uint32_t w = (rr_pointer_.next() + i) % num_warps_;` — round-robin scan | Type (2). At this point in evaluate, the issue scan reads the staged head pointer; this is before the post-issue rotation at L319. |
| `sim/src/timing/warp_scheduler.cpp:181` | `if (unit_busy_.next()[exec_unit_index(ExecUnit::LDST)] > 0)` — issue eligibility check | Type (2). Reads after the per-unit decrement loop at L118-120 (post-decrement value is what gates this cycle's issue). |
| `sim/src/timing/warp_scheduler.cpp:211` | `if (unit_busy_.next()[exec_unit_index(target)] > 0)` — same as L181 | Type (2). Same. |
| `sim/src/timing/warp_scheduler.cpp:224` | `if (writes_back && writeback_bitmap_.next()[bitmap_slot(offset)])` — writeback-slot reservation check | Type (2). Reads the bitmap after the head-advance at L116; `bitmap_slot(offset)` returns the post-advance position. |
| `sim/src/timing/warp_scheduler.cpp:237` | `if (opcoll_cooldown_cycles_.next() > 0)` — issue gate | Type (2). Reads after the L121-122 decrement. |
| `sim/src/timing/warp_scheduler.cpp:319` | `rr_pointer_.next_mut() = (rr_pointer_.next() + 1) % num_warps_;` — post-issue rotation | Type (1) write reading its own seeded value. |
| `sim/include/gpu_sim/timing/warp_scheduler.h:117, 121` | `unit_busy_.next_mut()[...] = cycles; writeback_bitmap_.next_mut()[...] = unit;` — `test_reserve_writeback_slot` and `test_set_unit_busy` hooks | Type (1) writes. |

**1 borderline audit candidate**: `operand_collector.cpp:32` (cleanup, not a fidelity bug).

### Phase 5 — `L1Cache` + `MSHRFile` + `LoadGatherBufferFile`

| File:line | Context | Classification |
|-----------|---------|----------------|
| `sim/include/gpu_sim/timing/mshr.h:71` | `MSHREntry& next_at(uint32_t index) { return entries_.next_mut()[index]; }` | Type (1) write handle. |
| `sim/src/timing/cache.cpp:758-759` | `&& !load_cmd_.current().valid && !load_cmd_.next().valid && !store_cmd_.current().valid && !store_cmd_.next().valid;` — in `L1Cache::is_idle()` | **AUDIT.** Observer query reading both slots of memoryless-consumer Regs. `load_cmd_` / `store_cmd_` are NOT seeded by `seed_all()` (memoryless-consumer opt-out, documented at `reg.h` `current_mut()` rationale (3)). The `.next().valid` read catches a cmd staged by coalescing this same cycle but not yet committed; the `.current().valid` read catches a cmd consumed in a prior evaluate but not yet cleared. Both reads are necessary for the panic-drain idle check. **Decision needed:** confirm that `is_idle()` is only ever called between ticks (after commit_all() flips and before the next tick's seed) — if so, `.next().valid` is always equal to `.current().valid` and the redundancy is harmless. If called mid-evaluate (e.g. from a snapshot inside `tick()`), the dual-read is load-bearing. The comment at L753-755 documents the read as intentional — leaving for the audit to confirm the call-site discipline. |
| `sim/src/timing/load_gather_buffer.cpp:179` | `for (const auto& buf : buffers_.next())` — inside `commit()`, recomputing `has_result_` from the about-to-be-committed buffer state | **AUDIT (low-risk).** Read in `commit()`, after evaluate has finished mutating `buffers_.next_mut()`. The comment at L174-179 explains: the staged buffer is the source-of-truth for the new `has_result_` value, latched in via `has_result_.set_next(any_buffer_filled)`. Hardware-faithful (the new has_result is a combinational function of the new buffer state, both clocked at the same edge). **No fidelity bug** — but the audit task should confirm `commit()` runs after all evaluate-phase fills have landed. |
| `sim/src/timing/load_gather_buffer.cpp:20, 115, 132, 165, 174, 195, 232` (comments) | Documentation of staged-buffer mutations. | Comment only. |

**2 audit candidates** — `cache.cpp:758-759` (substantive) and `load_gather_buffer.cpp:179` (low-risk verification).

### Phase 6 — `MemoryInterface` + `DRAMSim3Memory`

| File:line | Context | Classification |
|-----------|---------|----------------|
| `sim/include/gpu_sim/timing/memory_interface.h:120-121` | `&& !read_request_.current().valid && !read_request_.next().valid && !write_request_.current().valid && !write_request_.next().valid;` — in `FixedLatencyMemory::is_idle()` | **AUDIT.** Observer query. `read_request_` / `write_request_` are also memoryless-consumer Regs (memory plan M5). Same question as `L1Cache::is_idle()`: when does `is_idle()` run relative to the tick boundary? If always between ticks, `.next().valid == .current().valid`; if mid-evaluate, the dual-read is necessary. |
| `sim/src/timing/dramsim3_memory.cpp:235-236` | `if (read_request_.current().valid || read_request_.next().valid) return false; if (write_request_.current().valid || write_request_.next().valid) return false;` — in `DRAMSim3Memory::is_idle()` | **AUDIT.** Same pattern as `FixedLatencyMemory::is_idle()`. |
| `sim/src/timing/memory_interface.cpp:48` | `auto& rr = read_request_.next_mut();` — in `set_next_read_request()` | Type (1) write handle. |
| `sim/src/timing/memory_interface.cpp:57` | `auto& wr = write_request_.next_mut();` — in `set_next_write_request()` | Type (1) write handle. |
| `sim/src/timing/dramsim3_memory.cpp:138, 145` | Same write-handle pattern as `memory_interface.cpp:48, 57`. | Type (1) writes. |

**2 audit candidates** — both `is_idle()` implementations, with the same investigation as `L1Cache::is_idle()`.

### Phase 7 — Combinational wires (`Wire<T>`)

`Wire<T>` exposes `value()` / `drive()` / `reset()` — there is no `next()` accessor by design. The Phase 7 migration (alu_unit `next_redirect_`, writeback_arbiter `writeback_stall_`, cache `stalled_` / `stall_reason_` / `next_cmd_ready_`, load_gather_buffer `next_port_claimed_`) eliminated the `.next()` cross-stage read for backward signals entirely.

**0 audit candidates** in Phase 7.

## Summary of audit candidates

| Phase | File:line | Kind | Severity |
|-------|-----------|------|----------|
| 2 | `sim/src/timing/decode_stage.cpp:53` | Top-of-evaluate guard read | High — likely fidelity-relevant |
| 2 | `sim/src/timing/decode_stage.cpp:112-113` | Redirect-flush mirror of L53 | Re-examine after L53 |
| 4 | `sim/src/timing/operand_collector.cpp:32` | Top-of-evaluate gate | Low — likely equivalent to `.current()` cleanup |
| 5 | `sim/src/timing/cache.cpp:758-759` | `is_idle()` observer | Medium — depends on call-site discipline |
| 5 | `sim/src/timing/load_gather_buffer.cpp:179` | `commit()`-phase recompute | Low — likely correct as-is |
| 6 | `sim/include/gpu_sim/timing/memory_interface.h:120-121` | `is_idle()` observer | Medium — same as Phase 5 cache `is_idle()` |
| 6 | `sim/src/timing/dramsim3_memory.cpp:235-236` | `is_idle()` observer | Medium — same as above |

**Total: 7 sites to audit** (4 substantive + 3 low-risk verifications).

## Known pre-existing fidelity issues

These were observed during the migration and explicitly verified as **not introduced by it**. They belong to the audit follow-up, not to the migration phases.

- **`test_cache` aborts on a `-DGPU_SIM_USE_DRAMSIM3=OFF` build at `sim/src/timing/load_gather_buffer.cpp` — assertion `"deferred claim landing on a busy gather buffer"`.** Verified present at the migration baseline (commit `da13605`), so the regression existed before any Phase 1+ change. The abort is a real hazard in the gather-buffer claim-arbitration discipline (the claim slot lands on a buffer that is still busy from a prior cycle's claim), surfaced only on the fixed-latency-memory backend. Track separately from the audit; the migration is byte-identical to the same baseline that hits the assertion.

## Future enforcement: rule (3) annotations

The migration plan called for a third lint rule — every plain (non-`Reg` / non-`RegFifo` / non-`Wire`) member of a timing-model class must carry one of the documented annotations: `// config`, `// sim-instrumentation`, `// scratch`, or `// staging`. The intent is to make every member's *declared kind* visible at the declaration site, so the next contributor cannot accidentally add a plain `bool busy_;` and have it become a same-cycle-visible state hazard.

**This rule is NOT yet enforced by `tools/lint_timing_naming.py`.** Implementing it would require both:

1. A ~30-field annotation sweep across the headers that currently mix plain members in with `Reg<>` members without the annotation tags:
   - `sim/include/gpu_sim/timing/coalescing_unit.h` (`processing_`, `is_coalesced_`, `serial_index_`, `next_pop_`, `cmd_in_flight_` — and `current_entry_` which is plain by design)
   - `sim/include/gpu_sim/timing/fetch_stage.h` (the warp-PC array, RR back-pointers)
   - `sim/include/gpu_sim/timing/decode_stage.h` (per-warp state)
   - `sim/include/gpu_sim/timing/panic_controller.h` (one-shot state-machine fields)
   - `sim/include/gpu_sim/timing/timing_model.h` (orchestrator)
   - `sim/include/gpu_sim/timing/ldst_unit.h` (`addr_gen_fifo_`, `next_push_`, `fifo_total_pushes_`, `busy_this_cycle_`, `accepted_this_cycle_`)
   - `sim/include/gpu_sim/timing/dramsim3_memory.h` (the documented sim-instrumentation counters `fabric_cycle_`, `dram_ticks_`, `phase_`, the peak observers, and the internal scheduling queues)
   - `sim/include/gpu_sim/timing/load_gather_buffer.h` (`next_release_` and any other plain staging fields)
2. A migration of `CoalescingUnit` to `RegisteredStage` discipline. The plan listed phases 1–7 plus a final lint phase but did NOT include a phase for `CoalescingUnit` — its current members (`processing_`, `is_coalesced_`, `serial_index_`, `next_pop_`, `current_entry_`, `cmd_in_flight_`) are all plain. Some are genuine same-cycle scratch (and would carry `// scratch`); `current_entry_` is the documented "processing entry" — examining whether it should be a `Reg` is part of the rule (3) sweep.

The lint extension itself is straightforward (loop fields, skip `Reg<` / `RegFifo<` / `Wire<` types, require one of the four annotations on the previous or same line) — it is gated on the sweep being done first, because turning the rule on against the current tree would generate ~30 false-positive findings that swamp the legitimate ones.

## How to use this checklist

For each AUDIT row in the per-stage tables and the summary:

1. **Inspect the read site in context.** Open the surrounding `evaluate()` / `commit()` / `is_idle()` body and follow the seed → evaluate-mutate → commit timeline.
2. **Decide whether a hardware register would observe the pre- or post-stage value.** The decision rule: a register clocks at the cycle edge. If the read site is logically "after" the producer's write within the same combinational region (same `evaluate()`, same `commit()`), the staged value is the right read. If the read site is logically "before" the producer's write (top-of-evaluate gate, observer query between ticks), the committed value is the right read.
3. **If pre-write hardware semantics:** change `.next()` to `.current()`. Run benchmarks — the cycle delta on each affected benchmark IS the bug being fixed. Land the swap with the delta as the new gate.
4. **If post-write hardware semantics:** document the rationale in a code comment at the read site, citing the producer's write site by file:line and naming the in-evaluate / in-commit ordering it depends on. The audit row is then closed; the lint already permits the read (intra-class `next()` reads are allowed).
5. **If unclear:** trace the first divergent cycle with `--trace` / Perfetto on the smallest benchmark that exercises the path; the structured trace shows which value the consumer should have observed at the cycle boundary. See `resources/trace_and_perf_counters.md`.

The audit is finished when every AUDIT row is either (a) swapped to `.current()` with the benchmark delta landed, or (b) annotated in code with the rationale and removed from this checklist.
