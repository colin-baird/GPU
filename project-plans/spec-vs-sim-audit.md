# Spec-vs-Simulator Audit Findings

**Date:** 2026-04-28
**Mode:** Read-only inspection. No fixes applied.
**Scope:** Performance simulator (`sim/`) validated against `/workspace/resources/gpu_architectural_spec.md` Phase 1, organized by architectural component. Each finding cites the spec section, the source location, and a one-line direction (the spec-and-sim-side direction is suggested, not implemented).

Severity legend:
- **Critical** — observable wrong behavior or risk of incorrect committed effects.
- **Moderate** — discipline / observability / contract gap that can produce wrong results in plausible configurations or hides bugs.
- **Minor** — comment / documentation / naming / metric inconsistency, or stricter-than-spec behavior worth noting.

---

## Executive Summary

The simulator is broadly faithful to the spec. The audit surfaced **4 critical**, **~22 moderate**, and **~17 minor** items. The most consequential clusters are:

1. **EBREAK / panic boundaries.** Shadow-path EBREAK can panic the SM (Frontend C1); the trigger cycle still runs one full evaluate sweep before the flush cascade (TimingModel #1); `FunctionalModel` has a duplicate panic-latch path (Functional M3, Panic #3); INVALID-instruction latches hardware cause `0x2` outside any spec definition (Functional M2). Cumulatively these mean panic timing is +1 cycle off and a wrong-path EBREAK can wrongly halt the SM.
2. **Cache store path is timing-tag-only without an explicit declaration.** `process_store` never populates the spec-mandated MSHR fields (`store_data`, `store_byte_en`, `byte_offsets`, `lane_mask`) and the secondary-store drain skips the merge step (Cache #1, #2). Functional correctness rides on the trace-replay model, but the spec text describes hardware that the simulator does not model — this is a documentation gap as much as a code gap.
3. **DRAMSim3 silent-failure paths.** `submit_read` falls through to `return false` after a stripped-in-release `assert`; `on_read_complete` / `on_write_complete` silently discard unknown callbacks (Memory #2, #3, #6). The cache cannot recover from any of these, so an architectural-invariant violation manifests as an MSHR leak hang in release builds.
4. **Counter-semantics drift.** `branch_predictions` overcounts JAL (Operand-Collector F5); `writeback_conflicts` is "cycles" while the spec's intent is arguably "delayed writebacks" (Writeback F1); gather-buffer cross-warp queueing is invisible in stats (Writeback F2). These don't break behavior but distort validation surfaces in Appendix B.2 / B.3.
5. **FENCE accepted as NOP**, contradicting §2.1 explicit Phase-1 exclusion (Functional C1).

A `TimingModel::reset()` covering CTRL.RESET semantics is also not implemented (Panic #4).

---

## 1. Frontend (Fetch · Decode · Instruction Buffer · Branch Predictor)

### Critical

**FE-C1. Shadow-path EBREAK can panic the SM.**
- **Location:** `sim/src/timing/decode_stage.cpp:22-27,44,52-55,72-76`; observed at `sim/src/timing/timing_model.cpp:401-407`.
- **Spec:** §4.2 (speculative fetch past unresolved branch must be flushed); §4.8 (panic latched on the *committed* program path).
- **Observation:** A wrong-path EBREAK fetched between branch issue and the redirect-apply commit is decoded into `next_ebreak_request_`, latched into `current_ebreak_request_` by `decode_.commit()`, and panicked unconditionally at the top of the next tick. `apply_redirect_invalidate` only clears `pending_`; it never inspects/clears the ebreak request slots.
- **Direction:** In `DecodeStage::commit`, when a redirect is valid for the EBREAK's warp, also clear `next_ebreak_request_` (and/or `current_ebreak_request_`); or gate the timing_model.cpp:401 trigger on `branch_in_flight==false`.

### Moderate

**FE-M1. Warp PC mutated inside `evaluate()` instead of via a registered shadow.**
- **Location:** `sim/src/timing/fetch_stage.cpp:75-76`.
- **Spec:** `resources/timing_discipline.md` — committed state mutated only in `commit()`.
- **Observation:** `warps_[w].pc = ...` writes during `evaluate()`. No current consumer reads mid-evaluate, so no observable bug today; the rule is broken and is a latent footgun.
- **Direction:** Stage `next_pc` per-warp (or in `next_output_`) and apply in `commit()`.

**FE-M2. `apply_redirect` invalidation of `next_output_` is dead.**
- **Location:** `sim/src/timing/fetch_stage.cpp:129-131`.
- **Observation:** `apply_redirect` runs after `current_output_ = next_output_`, so the subsequent `next_output_` invalidate has no effect — `next_output_` is recomputed next cycle. Comment claim is misleading; the spec's "in-flight fetch register" is `current_output_`.
- **Direction:** Drop the dead invalidate; tighten the comment.

### Minor

**FE-m1.** `BRANCH` predicted_target is set even when `predicted_taken=false` (`branch_predictor.cpp:14-15`). Consumers ignore it; trace events still carry it.
**FE-m2.** `BRANCH` with `imm == 0` predicted not-taken (treated as forward); a self-loop branch would be predicted not-taken. Mostly cosmetic / degenerate ISA case.

### Verified-correct topics
RR pointer always advances `(orig+1) % num_warps`; in-flight gating formula matches spec; static prediction policy (backward taken / forward not-taken / JAL taken / JALR fall-through); decode-backpressure path retains held output and increments `fetch_skip_count` + `fetch_skip_backpressure`; EBREAK deferral when decode is stalled; mispredict-redirect plumbing (REGISTERED `RedirectRequest` opcoll → fetch.commit + decode.commit at N+1); per-warp instruction-buffer depth respects `instruction_buffer_depth`; Phase 5 boundary discipline for `ready_to_consume_fetch()`.

---

## 2. Warp Scheduler · Scoreboard · Branch Shadow Tracker

### Moderate

**SCH-M1. `query_unit_ready(SYSTEM)` returns hard-coded `true`; no override hook for SYSTEM.**
- **Location:** `sim/src/timing/warp_scheduler.cpp:63`.
- **Spec:** §4.3 — uniform eligibility; §4.5 — local dispatch controllers per unit.
- **Observation:** ECALL/EBREAK ride SYSTEM and bypass any backpressure. Benign today (they don't queue), but inconsistent with the uniform-eligibility rule and invisible to test overrides.
- **Direction:** Either declare SYSTEM always-ready in spec/comment, or wire a real ready signal.

**SCH-M2. Silent "all ready" default if `set_consumers` is forgotten.**
- **Location:** `sim/include/gpu_sim/timing/warp_scheduler.h:84-95`, `sim/src/timing/warp_scheduler.cpp:40-66`.
- **Observation:** Production wires consumers in `TimingModel`, but missing wiring would silently produce a scheduler that never stalls — a hard-to-find correctness bug masquerading as a perf regression.
- **Direction:** Make `set_consumers` a constructor parameter, or assert "wired-or-overridden" in evaluate.

### Minor

**SCH-m1.** Branch-shadow stall under buffer-empty attributes to `buffer_empty` (eligibility checked first), so post-redirect drain cycles are recorded as buffer-empty rather than branch-shadow. Matches spec eligibility order; flag for operators interpreting `branch_shadow_cycles ≈ flush penalty`.
**SCH-m2.** `reads_rd` semantics depend on `has_rd` being true; future ISA additions could introduce `reads_rd && !has_rd`. Add an assert.

### Verified-correct topics
5-condition eligibility check matches spec §4.3 in order; scoreboard set on issue / clear on writeback **commit**; `seed_next` / `commit` placement; loose RR scan with unconditional pointer advance; per-warp stall counters indexed correctly; `note_branch_issued` for BRANCH/JAL/JALR; `is_in_flight` reads `current_`; r0 short-circuit in scoreboard; ECALL deactivation occurs at dispatch (not scheduler); `SchedulerIssueOutcome` recorded once per warp per cycle.

---

## 3. Operand Collector · Branch Resolution · Redirect Signal

### Moderate

**OC-M1. `branch_predictions` counter overcounts JAL.**
- **Location:** `sim/src/timing/timing_model.cpp:461-462`.
- **Spec:** Appendix B.3 — informational, but naming implies speculative.
- **Observation:** `if (out.trace.is_branch) stats_.branch_predictions++;`. JAL is unconditional; including it skews mispredict rate downward.
- **Direction:** Rename to `branch_resolves`, or gate on `decoded.type == BRANCH || JALR` to align with the speculation semantics.

**OC-M2. `current_redirect_request_` retains stale `valid` between consumer reads.**
- **Location:** `sim/src/timing/operand_collector.cpp:54-55`.
- **Observation:** `commit()` does `current_ = next_; next_.valid=false;` — `current_.valid` stays true until next `commit()` overwrites with the fresh (invalid) `next_`. Consumers fire correctly (commit-time only), but debug observers between commits see a stale "valid" pulse.
- **Direction:** Have consumers consume-and-clear, or clear `current_` after fetch+decode have read it.

**OC-M3. `next_output_` cleared every evaluate; relies on dispatch always accepting same tick.**
- **Location:** `sim/src/timing/operand_collector.cpp:16`.
- **Observation:** Today safe because `dispatch_to_unit` runs immediately after `opcoll_->evaluate()` and the scheduler already gated on unit readiness. A future change making dispatch conditional would silently drop the output.
- **Direction:** Document the assumption or add a defensive check.

### Minor

**OC-m1.** `decode_stage` redirect-override discards `target_pc` (`decode_stage.h:64-70`); decode never reads it, but lacks symmetry with fetch.

### Verified-correct topics
1-cycle vs 2-cycle latency by `num_src_regs`; 0/1-operand path uses 1 cycle; double-buffer discipline (`accept`→`next_`, `evaluate`→ decrement, `commit`→ flip); `ready_out()` reads `current_busy_`; tick ordering for redirect signal (fetch+decode commit before opcoll commit); single-fire redirect (`next_redirect_request_.valid = false` reset per commit); branch-tracker clear paths (deferred mispredict via `note_redirect_applied`, immediate correct-prediction via `note_resolved_correctly`); JALR mispredict identification.

---

## 4. Execution Units (ALU · Multiply / VDOT8 · Divide · TLOOKUP)

### Minor

**EU-m1.** Multiply / Divide / TLookup all decrement `cycles_remaining` on the same tick `accept` runs. Effective end-to-end latency from accept to writeback-visible matches spec, but the same-tick decrement convention is unstated in code and surprising to readers; particularly STAGES=1 multiply collapses to 1-tick observable like ALU.
- **Direction:** Add a one-line comment in each `accept()` clarifying the convention.

**EU-m2.** No separate `multiply_stall_cycles` (head-blocked-by-result-buffer) or `divide_busy_cycles` distinct from `busy_cycles`. Spec doesn't mandate; observability gap only.

**EU-m3.** TLookup is single-occupancy (busy 17 cycles), but spec §2.3 phrasing "pipelined" is ambiguous. Behavior matches "asserts busy for the full duration" but the wording reads as cross-warp pipelining.
- **Direction:** Tighten spec §2.3 wording (no code change).

### Verified-correct topics
ALU 1-cycle; pipelined multiply at `pipeline_stages` depth with back-to-back accept and head-held-on-buffer-occupied; DIV iterative 32 cycles, single-occupancy; TLOOKUP 17 cycles; VDOT8 routes to MULTIPLY (no separate path); Phase-1 next/current discipline uniform across all four units; `ready_out()` reads `current_*`, `has_result()` reads `next_*` (COMBINATIONAL edge to writeback); WritebackEntry metadata populated; per-unit Stats counters incremented; reset semantics; STAGES=1 edge case correct.

---

## 5. LD/ST · Coalescing · Load Gather Buffer

### Moderate

**LS-M1. `LdStUnit::accept()` overwrites in-flight without backpressure assertion.**
- **Location:** `sim/src/timing/ldst_unit.cpp:8-22`.
- **Observation:** No `assert(!current_busy_ && !next_busy_)`. If accept is ever called while busy, the in-flight LD/ST silently drops. Other units typically assert.
- **Direction:** Add a defensive assert.

**LS-M2. `cycles_remaining` decrement in same tick as accept; `num_ldst_units = 0` is undefended.**
- **Location:** `sim/src/timing/ldst_unit.cpp:11-12,27-32`.
- **Observation:** With `N=32` (or higher), ceil collapses to 1 → push to FIFO same tick as accept. With `N=0`, divide-by-zero in `(WARP_SIZE + N - 1) / N`.
- **Direction:** Either defer same-tick decrement or document; add `Config::validate()` enforcing `num_ldst_units >= 1`.

### Minor

**LS-m1.** Comments in `cache.h:175-181`, `load_gather_buffer.h:73-79`, and `timing_discipline.md` row 11 describe `next_port_claimed_` as a REGISTERED next/current pair, but Phase 7 cleanup made it a single-slot COMBINATIONAL flag with end-of-cycle reset. Code is correct; docs lie.
- **Direction:** Update comments + `timing_discipline.md` row 11 to "COMBINATIONAL same-tick scratch with end-of-cycle reset."

**LS-m2.** No parameterized test for non-power-of-2 `num_ldst_units` (verifying `ceil(32/N)` cycles).

**LS-m3.** `CoalescingUnit::reset()` clears `processing_` / `serial_index_` but not `current_entry_` / `is_coalesced_`. Safe today (guarded by `processing_`), but stale `current_entry_.trace` lingers.

**LS-m4.** `gather_buffer_port_conflict_cycles` counted only on HIT loss; FILL+secondary same-cycle collisions (secondary loses) are invisible in stats. Cache retries the secondary next tick (correct), so this is observability-only.

### Verified-correct topics
Tick ordering `cache_.evaluate() → ldst_.evaluate() → coalescing_.evaluate()`; AddrGenFIFO single-buffered live deque with COMBINATIONAL pop edge; `gather_buffer_stall_cycles` increments without popping FIFO when buffer busy; per-warp serialization of loads; all-or-nothing coalescing with one-hot serialized lane masks; stores walk lanes in place / no gather-buffer claim / no writeback; port arbitration via single shared `next_port_claimed_`; `consume_result()` round-robin among gather buffers and full release on commit.

---

## 6. L1 Cache · MSHR · Write Buffer · Same-Line Merging · Pinning

### Critical

**CA-C1. `process_store` never captures per-lane store_data / store_byte_en / byte_offsets.**
- **Location:** `sim/src/timing/cache.cpp:116-192`; `MSHREntry` fields exist but unwritten (`mshr.h:23-25`).
- **Spec:** §5.3.1 MSHR field table mandates these fields; §5.3.1 store-miss fill / secondary wake explicitly merges store data into the resident line.
- **Observation:** Timing model is tag-only by design. Functional correctness rides on trace replay, but the spec text describes hardware the simulator does not model. The omission is undeclared.
- **Direction:** Either annotate the omission explicitly in `cache.h` / `mshr.h` referencing §5.3.1, or wire `process_store` to populate the spec-mandated fields.

### Moderate

**CA-M1. `lane_mask` of secondary stores left at default 0.**
- **Location:** `sim/src/timing/cache.cpp:162-170`.
- **Spec:** §5.3.1 — store `lane_mask` reflects "lanes whose data must be merged".
- **Direction:** Either populate (1u<<lane serialized / 0xFFFFFFFF coalesced) or document the omission.

**CA-M2. HIT-port-loss does not surface via `is_stalled()`.**
- **Location:** `sim/src/timing/cache.cpp:35-52`.
- **Observation:** `process_load` returns false on HIT port conflict, but `stalled_` is reserved for MSHR_FULL / LINE_PINNED / WRITE_BUFFER_FULL. Coalescer correctly retries via its own `processing_` flag, so behavior is correct; observability is asymmetric.
- **Direction:** Document the two-channel signaling design; no code change needed.

### Minor

**CA-m1.** `pinned_line_addr` field name is ambiguous — it's the line *index* (post `addr/line_size`), not a byte address.
**CA-m2.** Secondary chain identification is O(n²) per cycle at default 4 MSHRs (fine); revisit at MSHRs ≥ 16 (already noted as Phase-2 candidate in spec §5.3.1).

### Verified-correct topics
Direct-mapped indexing; allocation-order = program-order via single-port-per-cycle and serial coalescing; same-line merging via `find_chain_tail`; primary/secondary linkage; pin set on primary fill iff chain non-empty; pin clear on last secondary retire; allocation stall reasons set `stalled_=true` with explicit `stall_reason_`; FILL > secondary > HIT priority by tick order + single shared port flag; write buffer FIFO drain pops only on `submit_write` true; write-buffer-full cascading stall on store-fill; store secondary write-buffer stall preserves MSHR + pin; `is_idle()` requires no pending_fill / no MSHRs / empty write buffer; Phase 9 boundary (REGISTERED next/current for `pending_fill_`/`last_*_event_`, COMBINATIONAL for `stalled_`/`stall_reason_`); cross-warp same-line merging works (`find_chain_tail` scans by line addr only); RAW correctness for load-after-store on same line (chained secondary drains after primary frees).

---

## 7. External Memory Interface (FixedLatencyMemory · DRAMSim3Memory)

### Moderate

**MEM-M1. `submit_read` falls through to `return false` after a stripped-in-NDEBUG assert.**
- **Location:** `sim/src/timing/dramsim3_memory.cpp:66-72`.
- **Spec:** §5.6 lines 484-486 — `submit_read` must never return false in legitimate cache traffic; cache call sites do not check the bool.
- **Observation:** In release builds the assert vanishes; a violated invariant becomes a silent false return → MSHR leaked → infinite hang.
- **Direction:** Replace post-assert `return false` with `throw std::logic_error` (or another always-on check).

**MEM-M2. Same risk for `submit_read` MSHR-id-reuse path.**
- **Location:** `sim/src/timing/dramsim3_memory.cpp:73-78`.
- **Direction:** Throw on duplicate active MSHR rather than returning false.

**MEM-M3. `on_read_complete` / `on_write_complete` silently ignore unknown addresses.**
- **Location:** `sim/src/timing/dramsim3_memory.cpp:181-184,209-211,186-188,216-218`.
- **Observation:** A spurious or doubled DRAMSim3 callback is a real bug — silent return masks it.
- **Direction:** Convert to assert or add `spurious_dramsim3_callbacks` counter.

**MEM-M4. `dram_clock_mhz` not sanity-checked against fpga_clock_mhz.**
- **Location:** `sim/src/config.cpp:58-69`.
- **Observation:** Config rejects non-positive but accepts `dram_clock_mhz = 1e-9`; phase accumulator never reaches 1.0 → silent hang.
- **Direction:** Bound the ratio (e.g. `[0.01, 100]`).

### Minor

**MEM-m1.** Phase accumulator uses `double`; long-run drift after ~2^53 fabric cycles. Practically irrelevant (~6 weeks at 150 MHz).
**MEM-m2.** No per-write latency stat (`external_write_latency_*`); reads only — symmetric across both backends.
**MEM-m3.** `Config::validate()` does not enforce `dramsim3_request_fifo_depth == minimum`; accepts larger, which is wasteful but correct (§5.6 line 486 calls oversizing wasteful, not invalid).
**MEM-m4.** `is_idle()` and `in_flight_count()` semantics vary slightly between backends; counts of submitted-but-not-yet-issued reads include the assembly slot. Verified consistent.

### Verified-correct topics
FixedLatencyMemory exact-latency completion + FIFO order + read latency stats + `is_idle()` + always-true submits. DRAMSim3 bounded read region (= `num_mshrs * chunks_per_line`), bounded write region (= `write_buffer_depth * chunks_per_line`) with assert-guarded submit. Response queue capacity = `num_mshrs + write_buffer_depth + chunks_per_line` asserted at both push sites. Per-MSHR ReadAssembly with `chunks_remaining` and per-request latency tracking. Per-line WriteAssembly folds same-line writes. Per-DRAM-tick at most one chunk drained subject to `WillAcceptTransaction`, strict FIFO. `reset()` clears all state and rebuilds `dramsim3::MemorySystem`. Output-dir pre-create runs before `GetMemorySystem` in both ctor and reset. `chunks_per_line` validated. `chunks_per_line=1` edge case correct. Config rejects unknown backend, non-positive clocks, undersized FIFO, line-size not a multiple of burst.

---

## 8. Writeback Arbiter · Scoreboard Clear · Register File

### Moderate

**WB-M1. `writeback_conflicts` semantics ambiguous: cycles vs. delayed-writebacks.**
- **Location:** `sim/src/timing/writeback_arbiter.cpp:34-36`.
- **Observation:** Code increments by 1 per cycle when `valid_count > 1`; counter doc says "cycles in which more than one source was ready". A 3-way conflict adds 1, not 2. If the metric intent is "delayed writebacks queued behind arbitration," the code under-reports backlog by `(valid_count - 2)` per conflict cycle.
- **Direction:** Decide spec intent and align spec/doc/code.

**WB-M2. Cross-warp gather-buffer queueing invisible in stats.**
- **Location:** `sim/src/timing/load_gather_buffer.cpp:91-126`.
- **Observation:** `LoadGatherBufferFile` is registered as one arbiter source but has its own internal RR over per-warp buffers. Two warps' loads completing the same cycle: one wins the arbiter, the other queues — but the arbiter sees a single "ready" so `writeback_conflicts` doesn't increment for the second warp.
- **Direction:** Either expose per-warp gather buffers as separate arbiter sources, or add a dedicated `gather_buffer_writeback_queue_cycles` counter.

**WB-M3. `has_pending_work()` reads live `next_*`, not committed state.**
- **Location:** `sim/src/timing/writeback_arbiter.cpp:55-67`.
- **Observation:** Drain checks (`pipeline_drained`, panic drain) call `has_pending_work()` which observes a committed-this-cycle entry as still pending until next-tick commit, possibly delaying panic drain by one tick.
- **Direction:** Read `committed_` + source `has_result()`, or document as intentional.

### Minor

**WB-m1.** Same-cycle set+clear of `next_[warp][reg]` from issue and writeback can in principle race, but spec §4.7's 1-cycle gap rule prevents the specific dependent-issue case from manifesting. Document the order invariant in `scoreboard.h`.

### Verified-correct topics
Round-robin scan with `(rr_pointer + i) % N`; pointer advance past winner only on selection (different from fetch's unconditional advance — by design); scoreboard `clear_pending` writes `next_` and respects `dest_reg != 0`; WritebackEntry metadata fields populated and used by trace; single-cycle 32-lane gather-buffer writeback with full release; ECALL/EBREAK/STORE/FENCE/INVALID/BRANCH suppressed from register write at `functional_model.cpp:230-235`; SYSTEM-typed instructions don't enqueue writeback bandwidth; Phase 1 boundary discipline (units' `has_result()` reads `next_*` for COMBINATIONAL edge); register file r0 read=0 / r0 write discarded / `init_warp` zeros all + sets r1-r4; no-forwarding rule (operand collector reads committed register file).

---

## 9. Panic Mechanism (EBREAK · PanicController · Drain · Diagnostics)

### Critical

**PA-C1. Panic flush cascade armed at top-of-tick is overwritten by the same tick's normal evaluate sweep.**
- **Location:** Arming at `sim/src/timing/timing_model.cpp:401-407` vs. forward sweep at `:437-525`.
- **Spec:** §4.8.1 step 1 — "the warp scheduler is immediately inhibited."
- **Observation:** When a fresh ebreak is observed, `panic_->trigger()` happens after the panic-active early-exit check. The cycle still runs fetch / decode / scheduler / opcoll / dispatch / units / cache+mem / wb_arbiter, then commits, then runs the flush cascade. The flush is at-end-of-tick `reset()`, so staged `next_*` is wiped — net architectural effect is benign because flush == reset wipes next before commit flips, but one cycle of "normal" issue+writeback executes and updates the scoreboard / register file.
- **Direction:** Reorder so the trigger-cycle takes the panic branch immediately (arm before the `is_active()` check, or run flush before evaluate and skip the evaluate phase).

### Moderate

**PA-M1. `FunctionalModel::execute()` latches panic for `INVALID` with hardware cause `0x2`.**
- **Location:** `sim/src/functional/functional_model.cpp:220-224`.
- **Spec:** §4.8.2 — only EBREAK in Phase 1; §4.8.3 — causes `0x01–0xFF` reserved, none defined.
- **Observation:** Defines a hardware cause not in spec; bypasses `PanicController` so drain/scheduler-flush/all-warps-inactive don't engage; only `FunctionalModel.panicked_` is set.
- **Direction:** Either remove the INVALID hardware-panic path until Phase 2, or route it through `panic_->trigger()` and document cause `0x2` in §4.8.3.

**PA-M2. No `TimingModel::reset()` for CTRL.RESET semantics.**
- **Location:** `sim/src/timing/timing_model.cpp` (none); `sim/src/timing/panic_controller.cpp` `reset()` exists but not called from any host-visible path.
- **Spec:** §6.1 / §6.6 — CTRL.RESET clears all panic state including PANIC_WARP/LANE/PC/CAUSE.
- **Observation:** After a panic, the only way to clear is to construct a fresh TimingModel.
- **Direction:** Add `TimingModel::reset()` propagating to all sub-components; wire to a CSR-block reset path when CSR control is implemented.

**PA-M3. `panic_->evaluate()` reads `execution_units_drained()` from committed-state accessors before this cycle's commits.**
- **Location:** `sim/src/timing/timing_model.cpp:353` followed by units' `commit()` at `:366-374`.
- **Observation:** Panic-cycle drain check sees previous-cycle drainedness — adds one cycle to every panic completion. Within the 32-cycle bound, correctness-acceptable.
- **Direction:** Move `panic_->evaluate()` to post-commit, or document the deliberate +1 cycle.

**PA-M4. Functional EBREAK execute path latches panic in parallel with timing decode path.**
- **Location:** `sim/src/functional/functional_model.cpp:209-215`.
- **Observation:** Decode short-circuits EBREAK so the functional execute branch is never reached in the timing pipeline; it only fires for direct functional-model unit tests. Two panic-latch paths can desync PANIC_PC/CAUSE.
- **Direction:** Document or no-op the functional EBREAK branch.

### Minor

**PA-m1.** `trigger()` comment numbering vs. spec cycle enumeration is misaligned ("step 1 = cycle 2"); behavior matches spec, comment is confusing.
**PA-m2.** `PANIC_LANE` host-visible accessor missing from `FunctionalModel`; lane is hardwired 0 but never returned.
**PA-m3.** `PanicController::trigger()` does not check `is_active()` itself; relies on caller-side guard. Add early-return for defensive depth.
**PA-m4.** Scheduler / opcoll / gather / wb_arbiter `flush()` cascade does NOT include `scoreboard_.reset()` — matches spec §6 (line 620): scoreboard state is intentionally abandoned, with CTRL.RESET responsible for cleanup. Verified-correct given §6, but tied to PA-M2 since CTRL.RESET path is missing.

### Verified-correct topics
`MAX_DRAIN_CYCLES = 32`; `nullptr`-tolerant drained_query_; mark-all-warps-inactive at end of drain; flush cascade at commit phase covers scheduler/opcoll/gather_file/wb_arbiter + branch_tracker; `discard_writeback_results()` consumes alu/mul/div/tlookup/gather so cache/mem advancement produces no committed effects; PANIC_LANE = 0 by virtue of `latch_panic` signature; single-issue priority moot; cache/coalescer/mem_if intentionally left advancing after panic per spec §4.8.1 line 266.

---

## 10. Functional Model · Decoder · ISA

### Critical

**FN-C1. FENCE decoded as valid NOP.**
- **Location:** `sim/src/decoder.cpp:209-214` (comment even says "NOP per RV32I base spec").
- **Spec:** §2.1 — "FENCE is not supported in Phase 1."
- **Direction:** Decoder should map `OP_FENCE` → `InstructionType::INVALID`.

### Moderate

**FN-M1. INVALID instructions latch hardware panic cause `0x2` not defined in spec.** (Same finding as PA-M1.)

**FN-M2. ECALL / EBREAK semantics in `FunctionalModel::execute` do not match the timing-model boundary.**
- **Location:** `sim/src/functional/functional_model.cpp:202-215`.
- **Spec:** §6.5 ECALL marked inactive at *dispatch*; §4.8 EBREAK detected at *decode* and panicked after r31 read.
- **Observation:** Functional model is the oracle, latching at "execute" is acceptable, but a divergence can arise if timing-model issue/dispatch order disagrees on the boundary.
- **Direction:** Document the contract or move side-effects to a method invoked at the modelled boundary.

### Minor

**FN-m1.** Unknown CSR addresses → INVALID is stricter than spec §6.4 (which lists three CSRs without explicitly forbidding others). Consider documenting in the spec.
**FN-m2.** TLOOKUP decoder accepts all funct3 values; spec §2.3 doesn't restrict — verified-correct, flag for spec tightening if intent was funct3=0 only.

### Verified-correct topics
All RV32I + M-extension opcode/funct3/funct7 combinations correctly decoded except FENCE (FN-C1). VDOT8 only for funct7=0 ∧ funct3=0; other custom-0 → INVALID. ECALL/EBREAK funct12 distinction. CSRRS-only with rs1=x0 enforcement (CSRRW/CSRRC/non-zero-rs1 → INVALID). All `num_src_regs` / `has_rd` / `reads_rd` / `target_unit` mappings. Imm extraction (I/S/B/U/J). ALU semantics (ADD/SUB wrap, SRA arithmetic, SLL/SRL shamt mask, SLT/SLTU). MUL low/high with int64 widening. DIV/0 → -1, REM/0 → dividend, INT32_MIN/-1 → INT32_MIN/0 saturation. VDOT8 byte unpack + signed multiply + wraparound accumulate. TLOOKUP rs1+sext(imm12) → table_addr. JALR target masks bit 0. Branch evaluate signed/unsigned. LB/LH sign-extend, LBU/LHU zero-extend. Kernel arg preload (r1-r4, r5-r31=0). CSR per-lane semantics. Trace event population.

---

## 11. TimingModel Orchestration · Drain · Cycle Ordering

### Critical

**TM-C1.** Same as **PA-C1** — one cycle of normal pipeline work executes between ebreak-latch observation and the flush cascade.

### Moderate

**TM-M1. wb_arbiter not driven during panic; scoreboard busy bits leak.**
- **Location:** Panic loop at `sim/src/timing/timing_model.cpp:347-382`.
- **Observation:** `discard_writeback_results()` consumes results from each unit but `wb_arbiter_->evaluate()` is not called, so scoreboard `clear_pending` never fires for in-flight ops. Mostly cosmetic (panic ends in all-warps-inactive), but `pipeline_drained()` is not consulted in panic exit anyway.
- **Direction:** Either run the arbiter during panic with flush-style suppression, or document scoreboard abandonment (§6 line 620).

**TM-M2. Cache evaluate is split across the tick (FILL/secondary at top, HIT mid-sweep).**
- **Location:** `sim/src/timing/timing_model.cpp:409` (top), `:494` (coalescer triggers HIT).
- **Observation:** This is by design — encodes FILL > secondary > HIT priority via tick order. Auditors should mentally treat `cache_.evaluate` as a partial pass.
- **Direction:** Call out as documented invariant; no code change.

### Minor

**TM-m1.** `dispatch_to_unit` SYSTEM/ECALL deactivates the warp synchronously inside the evaluate sweep, not at commit — slight discipline blur, no current consumer broken.
**TM-m2.** `record_cycle_trace` runs after commit using `current_*` accessors — verified correct; final-cycle trace is emitted before termination check.

### Verified-correct topics
`scoreboard_.seed_next()` and `branch_tracker_.seed_next()` at top of every non-panic tick; `branch_tracker_.commit()` after `scoreboard_.commit()` and after every writer; `gather_file_->commit()` placed after `wb_arbiter_->commit()`; CSR routing through `target_unit=ALU`; cache evaluate before coalescing; FixedLatencyMemory ticks once per tick; pending_panic_flush re-arm guard; branch resolution actual-target reads functional `TraceEvent` and predicted comes from `out.prediction`; `pipeline_drained()` includes mem_if while `execution_units_drained()` does not (correct per §4.8 vs §6.5); termination check (`all_warps_done() && pipeline_drained()`) at end of tick.

---

## Cross-cutting observations

1. **Phase-7 / Phase-9 documentation drift.** Several REGISTERED-vs-COMBINATIONAL classifications in `timing_discipline.md` and per-file comments lag the actual code (e.g., LS-m1, the `next_port_claimed_` description). A single sweep through the discipline doc against the current sources would close most of these.
2. **Defensive `return false` paths in DRAMSim3 are silent in release builds.** Pattern: `assert(...)` followed by `if (...) return false;`. The cache cannot recover from these paths, so they should throw rather than return.
3. **CTRL.RESET path is unimplemented end-to-end.** Tied to no `TimingModel::reset()` and no CSR-block plumbing. Several "scoreboard abandoned" / "panic state cleared on reset" rules in spec rely on this path existing.
4. **Counter semantics need a doc pass.** `branch_predictions`, `writeback_conflicts`, gather-buffer queueing, secondary-loses-to-FILL — each is either over-counted, under-counted, or invisible; they affect the validation surface in Appendix B.

---

## Triage list (priority order)

| # | Item | Severity | Section |
|---|------|----------|---------|
| 1 | Shadow-path EBREAK can panic the SM | Critical | FE-C1 |
| 2 | Panic-trigger cycle still runs full evaluate sweep | Critical | PA-C1 / TM-C1 |
| 3 | FENCE accepted as valid NOP | Critical | FN-C1 |
| 4 | `process_store` omits per-lane MSHR fields | Critical | CA-C1 |
| 5 | DRAMSim3 silent `return false` after stripped assert | Moderate | MEM-M1 / MEM-M2 |
| 6 | DRAMSim3 silently discards unknown callbacks | Moderate | MEM-M3 |
| 7 | INVALID instructions latch undocumented cause `0x2` | Moderate | PA-M1 / FN-M1 |
| 8 | No `TimingModel::reset()` (CTRL.RESET unimplemented) | Moderate | PA-M2 |
| 9 | Functional EBREAK execute duplicates panic-latch | Moderate | PA-M4 |
| 10 | `branch_predictions` overcounts JAL | Moderate | OC-M1 |
| 11 | `writeback_conflicts` cycle-vs-delayed semantics | Moderate | WB-M1 |
| 12 | Cross-warp gather-buffer queueing invisible | Moderate | WB-M2 |
| 13 | `wb_arbiter` not driven during panic | Moderate | TM-M1 |
| 14 | `panic_->evaluate()` reads pre-commit drain state | Moderate | PA-M3 |
| 15 | Scheduler `set_consumers` silent default | Moderate | SCH-M2 |
| 16 | Scheduler SYSTEM unconditionally ready | Moderate | SCH-M1 |
| 17 | LdStUnit `accept` overwrites without assert | Moderate | LS-M1 |
| 18 | LdStUnit `num_ldst_units` not validated > 0 | Moderate | LS-M2 |
| 19 | DRAM clock sanity bounds missing | Moderate | MEM-M4 |
| 20 | `process_store` lane_mask never set | Moderate | CA-M1 |
| 21 | Warp PC mutated in `evaluate()` | Moderate | FE-M1 |
| 22 | `next_redirect_request_.valid` stale across cycles | Moderate | OC-M2 |
| 23 | `next_output_` cleared every evaluate (latent) | Moderate | OC-M3 |
| 24 | `has_pending_work()` reads live next_* | Moderate | WB-M3 |
| 25 | Functional ECALL/EBREAK boundary mismatch | Moderate | FN-M2 |
| 26 | Cache HIT-port-loss not reflected in `is_stalled` | Moderate | CA-M2 |
| 27 | Phase-7 / Phase-9 doc drift on `next_port_claimed_` | Minor | LS-m1 |
| 28+ | Remaining minor items (15+) | Minor | (see per-section) |

---

## Files referenced

- `sim/include/gpu_sim/timing/{fetch_stage,decode_stage,instruction_buffer,branch_predictor,branch_shadow_tracker,warp_scheduler,scoreboard,operand_collector,alu_unit,multiply_unit,divide_unit,tlookup_unit,ldst_unit,coalescing_unit,load_gather_buffer,cache,mshr,memory_interface,dramsim3_memory,writeback_arbiter,panic_controller,timing_model,execution_unit,warp_state,pipeline_stage}.h`
- `sim/src/timing/{...}.cpp` (matching the headers above)
- `sim/include/gpu_sim/{decoder,isa,config,types,trace_event,stats}.h`
- `sim/src/{decoder,config,stats}.cpp`
- `sim/include/gpu_sim/functional/{functional_model,register_file,memory,alu}.h`
- `sim/src/functional/{functional_model,alu}.cpp`
- `resources/gpu_architectural_spec.md`
- `resources/perf_sim_arch.md`
- `resources/timing_discipline.md`
- `resources/trace_and_perf_counters.md`

---

# Pass 2 — Supplemental Findings

A second adversarial pass was run with each agent given the pass-1 findings list and instructed to find *only new* issues. This appendix captures those new findings. Numbering uses an `-N`/`-n` suffix (capital `N` for moderate or higher; lowercase `n` for minor) and continues the same severity scheme.

**Headline:** Pass 2 surfaced **3 additional critical / high**, **~25 moderate**, and **~20 minor** items. Several reinforce pass-1 themes (panic boundary discipline, scoreboard-on-flush, observability) and several are genuinely new: a head-of-line cache-fill blocker, DRAMSim3 FIFO-order violation, an asymmetric branch-shadow source-of-truth, and `--max-cycles` cutoff bypassing drain.

## Pass-2 Top-Severity Items

| # | Item | Severity | Section |
|---|------|----------|---------|
| P2-1 | EBREAK trigger cycle does not flush fetch / decode / instr_buffer / warp PC | High | FE-C2 |
| P2-2 | Cache deferred fill HOL-blocks unrelated fills (pin-defer + write-buffer-full) | High | CA-C2 |
| P2-3 | DRAMSim3 read responses violate FIFO ordering across MSHRs | High | MEM-M5 |
| P2-4 | `--max-cycles` cutoff bypasses panic and pipeline-drained termination | High | TM-N1 |
| P2-5 | Asymmetric source-of-truth: branch-shadow set on `decoded.type`, cleared on `trace.is_branch` | Moderate | SCH-M3 |
| P2-6 | Scoreboard not cleared on panic flush (BST cleared, scoreboard isn't) | Moderate | SCB-M2 / OC-N6 / WB-N1 |
| P2-7 | `cache_/coalescing_` not in `pending_panic_flush_` cascade; spec wants MSHR/write-buffer abandoned | Moderate | PA-N6 |
| P2-8 | Reentrant DRAMSim3 callbacks mutate `responses_` mid-`ClockTick` | Moderate | MEM-M6 |
| P2-9 | DRAMSim3 MSHR-id reuse vs. stale chunk-to-mshr map entries | Moderate | MEM-M7 |
| P2-10 | Speculative wrong-path fetch can throw on out-of-range PC | Moderate | FE-m8 |

---

## 1 (Pass 2). Frontend

**FE-C2. EBREAK trigger cycle does not flush fetch / decode / instr_buffer / warp PC.**
- **Location:** `sim/src/timing/timing_model.cpp:401-407,535-541`; `sim/src/timing/fetch_stage.cpp:75-76`.
- **Spec:** §4.1 / §4.8 — panic flush should clean wrong-path state.
- **Observation:** When `decode_->current_ebreak_request()` becomes valid, `panic_->trigger()` runs only after the panic-active early-exit check, so the full tick body still runs. fetch.evaluate fetches another instruction for the EBREAK warp and mutates `warps_[w].pc`. The end-of-tick `pending_panic_flush_` cascade flushes scheduler / opcoll / gather_file / wb_arbiter but does not flush fetch's `current_output_`, decode's `pending_`, the per-warp `instr_buffer`, or roll back the warp PC. The warp arrives at panic drain with PC two instructions past the EBREAK and a polluted instruction buffer.
- **Direction:** Add `fetch_->flush()`, `decode_->flush()`, per-warp `instr_buffer.flush()`, and snapshot the EBREAK warp's PC at trigger time into the `pending_panic_flush_` cascade.

**FE-m8. Speculative wrong-path fetch can throw on out-of-range PC.**
- **Location:** `sim/include/gpu_sim/functional/memory.h:72-78` called from `sim/src/timing/fetch_stage.cpp:70`.
- **Observation:** `InstructionMemory::read` throws `std::out_of_range`. After a mispredicted JAL/BRANCH, fetch.evaluate at cycle N speculatively reads at the wrong-path PC; if past the loaded program, it kills the simulator. Real hardware would just see garbage that decodes as INVALID and gets flushed.
- **Direction:** Return 0 / harmless encoded INVALID on out-of-range, or guard the fetch read.

**FE-m3. `fetch_skip_all_full` covers the "all warps inactive" case too.**
- **Location:** `sim/src/timing/fetch_stage.cpp:82-85`. Counter inflates without indicating a buffer-full condition.
- **Direction:** Split into `fetch_skip_no_active` vs `fetch_skip_all_full`, or rename.

**FE-m4. `warp_cycles_active[w]` keeps incrementing during panic-active ticks.**
- **Location:** `sim/src/timing/timing_model.cpp:341-345` runs before `panic_->is_active()` early-return.
- **Observation:** Warps remain `active=true` during drain, so the counter climbs despite no forward progress.
- **Direction:** Skip the increment when panic is active, or rename to `warp_cycles_resident`.

---

## 2 (Pass 2). Scheduler · Scoreboard · Branch Shadow

**SCH-M3. Asymmetric source-of-truth between scheduler issue and mispredict-clear.**
- **Location:** `sim/src/timing/warp_scheduler.cpp:163-169` (set, gates on `decoded.type`); clear sites at `sim/src/timing/timing_model.cpp:461` and operand_collector resolve_branch (gate on `out.trace.is_branch`); `sim/src/functional/functional_model.cpp:128/137/145` only sets `is_branch=true` when lane == 0.
- **Observation:** If lane 0 is ever inactive (predication, masked warp, future divergence), scheduler still asserts `note_branch_issued` but `resolve_branch` / `apply_redirect` never fire. Bit set with no clear → permanent stall for that warp. Currently masked because Phase 1 has no predication, but a hidden time bomb.
- **Direction:** Drive both ends from the same predicate — gate `note_branch_issued` on `out.trace.is_branch` or vice versa.

**SCB-M2. Asymmetric panic-flush: `branch_tracker_.reset()` IS in cascade, `scoreboard_.reset()` is NOT.**
- **Location:** `sim/src/timing/timing_model.cpp:535-542`; `sim/include/gpu_sim/timing/scoreboard.h:12-15`.
- **Spec:** §6 line 620 says scoreboard intentionally abandoned, with CTRL.RESET cleaning up. But CTRL.RESET path is missing (PA-M2) and tracker/scoreboard treatment is inconsistent.
- **Observation:** After panic flush, scoreboard pending bits set by issued-but-not-yet-written-back instructions remain set. If the simulator is reused without full reconstruction, those stale bits block future issues. (Also captured as OC-N6 and WB-N1 from other agents.)
- **Direction:** Either add `scoreboard_.reset()` to the cascade, or implement `TimingModel::reset()` (PA-M2) and make panic→reset the documented recovery sequence.

**SCH-m3. `INACTIVE` diagnostic clobbers ECALL-just-deactivated warp's outcome.**
- **Location:** `sim/src/timing/warp_scheduler.cpp:70` (`fill(INACTIVE)`).
- **Observation:** Tools cannot distinguish "never activated" from "deactivated this cycle by ECALL".
- **Direction:** Add a transient `JUST_RETIRED` outcome.

**SCH-m4. `current_diagnostics()` is a pre-commit slot for orchestrator's mid-tick consumers.**
- **Location:** `sim/include/gpu_sim/timing/warp_scheduler.h:100-102`.
- **Observation:** Any consumer reading mid-evaluate sees previous-cycle diagnostics paired with this-cycle issued instruction. Production callers run post-commit and are fine; document the contract.

**SCH-m5. `warp_stall_unit_busy[w]` conflates opcoll-busy and unit-busy.**
- **Location:** `sim/src/timing/warp_scheduler.cpp:118,124`.
- **Direction:** Split into `warp_stall_opcoll_busy` vs `warp_stall_unit_busy`.

**BST-m1. `branch_tracker_.reset()` skips `seed_next()` invariant; works only because reset zeros both halves.**
- **Location:** `sim/include/gpu_sim/timing/branch_shadow_tracker.h:51-54` and `sim/src/timing/timing_model.cpp:540`.
- **Direction:** Document the invariant on `reset()` or add an assert.

---

## 3 (Pass 2). Operand Collector · Branch Resolution

**OC-N1. `branch_predictor_->update()` called for JAL/JALR.**
- **Location:** `sim/src/timing/timing_model.cpp:461-464`. Static predictor stub ignores args; any future stateful predictor (2-bit counter, BTB) will be polluted by JAL/JALR samples that always have `branch_taken=true`.
- **Direction:** Gate update on `decoded.type == BRANCH`.

**OC-N2. `actual_target` for not-taken branches passed as `pc+4`, not `branch_target`.**
- **Location:** `sim/src/timing/timing_model.cpp:466-468`. Future BTB-style predictor cannot learn the static target on not-taken samples.
- **Direction:** Pass `out.trace.branch_target` unconditionally as `actual_target`; use `actual_taken` for direction.

**OC-N3. Two branches resolving same tick lose one redirect (currently unreachable, fragile).**
- **Location:** `sim/src/timing/operand_collector.cpp:58-76`. `resolve_branch` overwrites `next_redirect_request_` without checking `valid`.
- **Direction:** Add `assert(!next_redirect_request_.valid)` to lock the invariant.

**OC-N4. `num_src_regs == 3` fast-path silent on out-of-range values.**
- **Location:** `sim/src/timing/operand_collector.cpp:12`.
- **Direction:** Switch on `num_src_regs` explicitly with assert.

**OC-N5 / EU2-n1. VDOT8 silently merged with MUL stats.**
- **Location:** `sim/src/decoder.cpp:251-255`; `sim/src/timing/multiply_unit.cpp:19,27`. No `vdot_stats` counter; entries don't carry `InstructionType`.
- **Direction:** Split into `vdot_stats`, or stamp `InstructionType` on the pipeline entry.

**OC-N6 / WB-N1. Scoreboard never cleared on panic flush.** (See SCB-M2.)

**OC-N7. `set_redirect_request_override(valid=false, ...)` silently masks real opcoll redirects.**
- **Location:** `read_redirect_request` in `operand_collector.h:130-136`.
- **Observation:** Override takes precedence regardless of its `valid` field; a test that forgets to clear it kills production redirects.
- **Direction:** Have `read_redirect_request` fall through when override has `valid==false`.

**OC-N8. `branch_predictor->update()` runs before `resolve_branch()`.**
- **Location:** `sim/src/timing/timing_model.cpp:461-481`. Footgun for any stateful predictor; today benign because static predictor is a no-op.
- **Direction:** Move `update()` after `resolve_branch()` and gate on `decoded.type == BRANCH`.

---

## 4 (Pass 2). Execution Units

**EU2-n2. `MultiplyUnit::reset()` doesn't zero `current_/next_result_buffer_.values`; only `valid` flag.**
- **Location:** `sim/src/timing/multiply_unit.cpp:62-67`. Snapshot reads after reset see stale 32-lane data, gated only by the `valid` bit.
- **Direction:** Zero the entry bodies in reset, or document the contract.

**EU2-n3. `consume_result()` non-idempotent and unguarded.**
- **Location:** All four units. No precondition assert that `next_*.valid == true` on entry; defensive double-drain would silently return stale fields.
- **Direction:** Assert preconditions or return `std::optional<WritebackEntry>`.

**EU2-n4. Loser-of-arbitration ordering hazard with `next_*` reads after `evaluate()`.**
- **Location:** `sim/src/timing/writeback_arbiter.cpp:24-48` plus units' `has_result()` reading `next_result_buffer_`.
- **Observation:** After arbiter clears the winner's `next_*.valid`, the unit's `current_result_buffer_` is still valid until commit. Trace consumers using `result_entry()` (next_*) vs `pending_entry()` (current_*) see inconsistent slot occupancy.
- **Direction:** Document or unify the trace-vs-arbiter snapshot path.

**EU2-n5. Inconsistent `busy_cycles` accounting across units.**
- **Location:** ALU bumps on produce, MUL on every cycle pipeline non-empty, DIV/TLOOKUP on every busy tick. "Occupancy %" reports are incomparable across units.
- **Direction:** Define `busy_cycles` semantics in spec, or add a separate `produce_cycles` counter for ALU symmetry.

**EU2-n6. `TLOOKUP_LATENCY` hardcoded constant 17, not parameterizable.**
- **Location:** `sim/include/gpu_sim/timing/tlookup_unit.h:41`. Spec §2.3 ties latency to architectural parameters (warp width / port count).
- **Direction:** Parameterize via `SimConfig`, or assert against `WARP_SIZE` / port count.

**EU2-n7. `DivideUnit::evaluate()` lacks underflow guard if `next_cycles_remaining_ == 0` at entry.**
- **Location:** `sim/src/timing/divide_unit.cpp:24-31`. `reset()` doesn't enforce mutual exclusion of `next_busy_=false && next_cycles_remaining_==0`.
- **Direction:** Guard `if (next_cycles_remaining_ > 0)` defensively before decrement.

**EU2-n8. Div-by-zero unconditional 32-cycle latency unasserted.**
- **Location:** `sim/src/timing/divide_unit.cpp:5-18`. Matches spec §4.5 line 225 but no test/assert verifies the invariant.
- **Direction:** Cite spec or add a regression test.

---

## 5 (Pass 2). LD/ST · Coalescing · Gather Buffer

**LS-M3. `is_stalled()` snapshot ordering across MSHR-exhaustion stalls.**
- **Location:** `sim/src/timing/coalescing_unit.cpp:20`; `sim/src/timing/cache.cpp:69-74,150-155`.
- **Observation:** `cache_.evaluate()` only sets `stalled_` from FILL/secondary; MSHR-exhaustion stall is set later by `process_load`/`process_store`. Coalescer's bail check sees stale `is_stalled()` next tick.
- **Direction:** Latch `accepted=false` in coalescer's own one-shot stall flag, or move cache stall set to registered side.

**LS-M4. `process_store` has no `lane_mask` parameter.** (Cross-check with CA-M1.)
- **Location:** `sim/src/timing/coalescing_unit.cpp:77-81,102-105`. Bypasses any future write-mask plumbing; can't model partially-disabled warp stores.
- **Direction:** Add `lane_mask` to `process_store` even if Phase 1 wires it to all-ones / one-hot.

**LS-M5. Coalescer's `is_coalesced` check ignores predicated-off lanes.**
- **Location:** `sim/src/timing/coalescing_unit.cpp:38-45`. Compares all 32 lane addresses unconditionally; garbage addresses in dead lanes force serialization.
- **Direction:** AND with `trace.active_mask` before comparing, or skip masked lanes.

**LS-m5. No `addr_gen_fifo_full_cycles` counter.**
- **Location:** `sim/src/timing/ldst_unit.cpp:32-41`. Backpressure stall invisible to perf reports.

**LS-m6. `LdStUnit::reset()` leaves `pending_entry_` payload stale.**
- **Location:** `sim/src/timing/ldst_unit.cpp:53-61`. Trace dumps after panic flush may leak previous warp PC.

**LS-m7. `gather_file_->flush()` races same-tick `wb_arbiter_` `committed_entry()` trace state.**
- **Location:** `sim/src/timing/load_gather_buffer.cpp:87-89` invoked at `sim/src/timing/timing_model.cpp:538`, after `wb_arbiter_->commit()` at 514.
- **Observation:** Spec §4.8 says drain writeback first, then mark inactive; the current order flushes the gather file even if the arbiter still needs the buffer state next tick.

**LS-m8. `consume_result()` RR pointer reset to 0 on flush.**
- **Location:** `sim/src/timing/load_gather_buffer.cpp:79-89`. After non-panic full-system reset, fairness across warps is lost on the first cycle.

---

## 6 (Pass 2). L1 Cache · MSHR · Write Buffer · Pinning

**CA-C2. Deferred fill in `pending_fill_` head-of-line-blocks unrelated fills.**
- **Location:** `sim/src/timing/cache.cpp:265-291` (`handle_responses`) + `complete_fill` lines 207-218 (pin-defer) and 228-233 (write-buffer-full defer).
- **Spec:** §5.3 fill-port cadence (one per cycle); §5.3.1 explicitly authorizes blocking only for write-buffer-full store-fill case.
- **Observation:** Pin-defer also parks the response in `pending_fill_`; `handle_responses` then early-returns whenever `current_pending_fill_.valid`, blocking unrelated fills (different sets) until the pin clears. Spec scope of HOL blocking was narrower.
- **Direction:** Move deferred response into a side-buffer keyed by target line so `handle_responses` can skip past it; or document this HOL behavior in §5.3.1 as an additional cascading stall.

**CA-C3. `find_chain_tail` non-defensive against malformed chains.**
- **Location:** `sim/src/timing/mshr.cpp:39-49`. Returns -1 if no entry has `next_in_chain == INVALID`.
- **Observation:** -1 is then interpreted as "no existing line" (cache.cpp:91/170), so the cache silently allocates a primary and submits a duplicate external read, orphaning the existing chain.
- **Direction:** Assert that if any valid entry matches `line_addr`, function must return a non-negative tail.

**CA-C4. Pin-defer path bumps `line_pin_stall_cycles` without setting `stalled_` — conflates two events.**
- **Location:** `sim/src/timing/cache.cpp:207-218`. Coalescer-stalled-on-pin (sets `stalled_`) and fill-defer (does not) share the same counter.
- **Direction:** Introduce `fill_pin_defer_cycles`, or document that `line_pin_stall_cycles` aggregates both events.

**CA-m3. `process_load` / `process_store` accept `lane_mask == 0` without validation.**
- **Location:** `sim/src/timing/cache.cpp:28-114`. Zero mask wastes a port-claim cycle but does no useful work.
- **Direction:** Assert `lane_mask != 0` at entry.

**CA-m4. `is_idle()` does not include `mem_if_->is_idle()`.**
- **Location:** `sim/src/timing/cache.cpp:436-440`. Verified safe — `pipeline_drained()` checks both — but the cache header could mislead a future caller.
- **Direction:** Comment that callers wanting full memory-system quiescence must additionally check `mem_if_->is_idle()`.

---

## 7 (Pass 2). External Memory Interface

**MEM-M5. DRAMSim3 read responses violate FIFO ordering across MSHRs.**
- **Location:** `sim/src/timing/dramsim3_memory.cpp:181-207`.
- **Spec:** §5.6 lines 477, 436 — "FIFO-ordered".
- **Observation:** A read response is pushed to `responses_` only when the *last* chunk of its MSHR returns. DRAMSim3's bank/refresh/locality scheduler reorders chunk completions across MSHRs freely; reads submitted A,B can complete to the cache as B,A. Cache routes by `mshr_id`, so it tolerates this — but the spec contract is not actually held. FixedLatency preserves submit order; DRAMSim3 doesn't.
- **Direction:** Tighten spec wording to "responses are unordered, routed by mshr_id," OR add a per-submit sequence number and stall-emit until the head completes.

**MEM-M6. Reentrant DRAMSim3 callbacks mutate `responses_` mid-`ClockTick()`.**
- **Location:** `sim/src/timing/dramsim3_memory.cpp:135` calls `mem_->ClockTick()`; callbacks fire synchronously inside the tick and mutate `read_assembly_`/`write_assembly_`/maps/`responses_`.
- **Observation:** Capacity assertion checks each push individually but not jointly; an iterator over `read_assembly_` around `ClockTick` would be invalidated.
- **Direction:** Document the reentrancy contract and prove every container access is reentrancy-safe, or stage completions into a temporary list and drain after `ClockTick` returns.

**MEM-M7. MSHR-id reuse vs. stale chunk-to-mshr map entries.**
- **Location:** `sim/src/timing/dramsim3_memory.cpp:56-92`; `read_chunk_to_mshr_` populated at line 129.
- **Observation:** Cache reuses MSHR ids freely. `read_chunk_to_mshr_` keys on byte address; if a duplicate or post-completion DRAMSim3 callback arrives for the previous transaction's chunk, it decrements the *new* MSHR's `chunks_remaining` and double-completes it. Safety relies on DRAMSim3 never delivering duplicates — undocumented.
- **Direction:** Tag chunks with a monotonically increasing transaction id and key the reverse map on it.

**MEM-m5. `dramsim3_output_dir` `create_directories` errors swallowed silently.**
- **Location:** `sim/src/timing/dramsim3_memory.cpp:46-47`. On failure DRAMSim3 falls back to cwd and emits a warning to stdout — the very pollution this pre-creation was meant to suppress.

**MEM-m6. `reset()`'s DRAMSim3 rebuild not wrapped in try/catch.**
- **Location:** `sim/src/timing/dramsim3_memory.cpp:166-179`. If the `.ini` was deleted between construction and reset, DRAMSim3 aborts fatally.

**MEM-m7. `FixedLatencyMemory::reset()` doesn't zero latency stats.**
- **Location:** `sim/src/timing/memory_interface.cpp:58-61`. Multi-kernel-launch driver mixes latency samples from dead kernel into new average. Same in DRAMSim3.

**MEM-m8. Same-line back-to-back writes — first response artificially deferred until second completes.**
- **Location:** `sim/src/timing/dramsim3_memory.cpp:104-105,209-228`. Spec §5.6 line 489 says "folded into a single response," implying merging not delaying.
- **Direction:** Spec clarification, or per-submit completion tracking.

**MEM-m9. `WillAcceptTransaction` head-of-line stall hides per-request latency growth.**
- **Direction:** Add `external_read_queue_wait_total` counter sampled when chunk leaves `request_fifo_`.

**MEM-m10. Response queue capacity bound's off-by-one allows transient overshoot.**
- **Location:** `sim/src/timing/dramsim3_memory.cpp:196,221`. Audit assertion direction vs. the worst-case test's tolerance together.

---

## 8 (Pass 2). Writeback · Scoreboard · Register File

**WB-N1.** Same finding as **SCB-M2** / **OC-N6** — scoreboard not reset on panic flush.

**WB-N2. Functional write at issue precedes scoreboard set in `next_`.**
- **Location:** `sim/src/timing/warp_scheduler.cpp:150-157`; `sim/src/functional/functional_model.cpp:230-234`.
- **Observation:** Scheduler invokes `func_model_.execute(...)` which synchronously writes `reg_file_[w][lane][rd] = result`, BEFORE `scoreboard_.set_pending(...)` at line 156. Functional state is "value at issue or later," not "pre-issue." Any consumer that reads `reg_file_` for the *pre-issue* value would race.
- **Direction:** Confirm no consumer reads pre-issue regfile state, or document the model decoupling explicitly.

**WB-N3. `committed_entry()` cleared by panic flush before `record_cycle_trace` runs.**
- **Location:** `sim/src/timing/timing_model.cpp:535-542,547`. `flush()` calls `reset()` which sets `committed_ = nullopt`. Trace path that reads it after flush sees the legitimate writeback erased.
- **Direction:** Move `committed_` capture before the panic-flush block, or have `flush()` not clear `committed_`.

**WB-N4. `add_source(nullptr)` silent corruption.**
- **Location:** `sim/src/timing/writeback_arbiter.cpp:8-10`. No null check.

Note: gather-buffer source reports `ExecUnit::LDST` (`load_gather_buffer.cpp:110`); cannot distinguish FILL completion from HIT completion in trace.

---

## 9 (Pass 2). Panic Mechanism

**PA-N1. `discard_writeback_results()` drains only one buffer per cycle; mul pipeline can't drain in 32 cycles.**
- **Location:** `sim/src/timing/timing_model.cpp:299-311`. `MultiplyUnit` advances `current_pipeline_` deque by ONE stage per `evaluate()`; with `pipeline_stages` near `MAX_DRAIN_CYCLES=32`, a moderately-loaded mul plus LDST FIFO entries can hit the timeout.
- **Direction:** Emit `Stats::panic_drain_timeout` and `panic_in_flight_abandoned` counters; document per-cycle drain throughput in spec.

**PA-N2. Drain timeout silently abandons in-flight ops with no counter / no log.**
- **Location:** `sim/src/timing/panic_controller.cpp:38`. No distinction between clean drain and forced timeout. `Stats` has no panic-related fields.
- **Direction:** Add `Stats::panic_drain_cycles` and `panic_drain_timed_out`; log on timeout.

**PA-N3. LDST FIFO included in `execution_units_drained()` though spec scopes drain to "execution units and writeback only".**
- **Location:** `sim/src/timing/timing_model.cpp:289-297`. LDST FIFO entries can only retire by being consumed by `coalescing_->evaluate()` → cache; if cache stalls on MSHR_FULL/WRITE_BUFFER_FULL, panic drain hits 32-cycle timeout — caused by the very memory subsystem the spec says shouldn't gate drain.
- **Direction:** Drop `ldst_->fifo_empty()` from `execution_units_drained()`, or treat FIFO entries as abandoned the same way MSHRs are.

**PA-N4. `panic_->evaluate()` reads stale (last-cycle committed) drained-query.**
- **Location:** `sim/src/timing/timing_model.cpp:353-364`. Panic.evaluate runs at top of tick before unit evaluates; one extra panic cycle than necessary.
- **Direction:** Move `panic_->evaluate()` after unit evaluates and `discard_writeback_results`, or split it.

**PA-N5. Cache/coalescer/mem_if continue producing committed effects + stat counter increments during panic.**
- **Location:** Cache/coalescer/mem_if calls in `sim/src/timing/timing_model.cpp:348,360-362`.
- **Spec:** §4.8.1: "must not produce any new architecturally committed effects after panic is active."
- **Observation:** `cache_->drain_write_buffer()` continues issuing external writes; counters (`coalesced_requests`, `cache_hits`, `external_memory_writes`) increment as if architecturally retired.
- **Direction:** Gate counter updates and external submissions on `!panic_active`, OR clarify spec wording (Phase-1-acceptable per spec line 266 observability allowance).

**PA-N6. `pending_panic_flush_` cascade does not reset cache or coalescer.**
- **Location:** `sim/src/timing/timing_model.cpp:535-542`. In-flight pre-panic LDST entries continue draining during panic, contradicting "abandon MSHR fills / write buffer drains".
- **Direction:** Reset/abandon `coalescing_`'s active entry and cache MSHR queue at `pending_panic_flush_` time.

**PA-N7. No `wb_arbiter_->commit()` on panic ticks; `committed_entry()` stays stuck on last pre-panic value.** (Overlaps WB-N3.)
- **Direction:** Call `wb_arbiter_->commit()` (with `pending_commit_=nullopt`) in panic commit phase.

**PA-N8. Functional `panicked_` flag has no public clear/reset method.**
- **Location:** `sim/src/functional/functional_model.cpp:36-39,18-21`. Only `reset_warp_state()` clears it as a side-effect.
- **Direction:** Add `FunctionalModel::clear_panic()`; have `PanicController::reset()` call it.

**PA-N9. Panic-trigger does not check `warps_[panic_warp_].active`.**
- **Location:** `sim/src/timing/timing_model.cpp:401-407`. PANIC_WARP could point at an inactive warp on a contrived race.
- **Direction:** Assert `warps_[ebreak_req.warp_id].active` at trigger time, or reject the request.

**PA-N10. `EBreakRequest` silently dropped if `panic_->is_active()`.**
- **Location:** `sim/src/timing/timing_model.cpp:403`. No log/counter.
- **Direction:** Assert the guard never fires, or count drops.

**PA-N11. Operand-collector "commandeer" semantics from spec §4.8.1 step 2 not modeled.**
- **Location:** `sim/src/timing/panic_controller.cpp:27` reads functional regfile directly, bypassing `OperandCollector`.
- **Direction:** Either route the r31 read through opcoll on the trigger cycle, or update spec wording.

**PA-N12. `STATUS.DONE` and `STATUS.PANIC` host-visible bits not exposed.**
- **Location:** `sim/include/gpu_sim/timing/panic_controller.h:18-19`. Only `is_active()` / `is_done()` exist.
- **Direction:** Add `TimingModel::status_panic()` and `status_done()` accessors that mirror spec §6.1, or document that the simulator skips CSR modeling.

**PA-N13. Single-issue assumption in panic-trigger guard not enforced.**
- **Location:** `sim/src/timing/decode_stage.cpp:9-38`. Decode produces ≤1 EBreakRequest per cycle by construction; future multi-issue decode would break panic flow with no defined ordering.
- **Direction:** Add a comment / static assertion documenting the Phase-1 invariant.

---

## 10 (Pass 2). Functional Model · Decoder · ISA

**FN-C2. `branch_target` populated unconditionally for not-taken BRANCH.**
- **Location:** `sim/src/functional/functional_model.cpp:147`. JSON arg emission at `timing_model.cpp:961-963` writes the target unconditionally.
- **Direction:** Set only when `branch_taken == true`, or document.

**FN-C3. Validation error message lies about `MAX_WARPS`.**
- **Location:** `sim/src/config.cpp:14-16` says "must be in [1, 32]" though `MAX_WARPS == 8`.
- **Direction:** Format using actual `MAX_WARPS` constant.

**FN-M3. `init_kernel` doesn't reinit register banks of warps deactivated since last launch.**
- **Location:** `sim/src/functional/functional_model.cpp:17-32`. Inactive warps retain stale values from prior kernel; `reset()` doesn't reinit `memory_`/`instr_mem_`/`lookup_table_` either.
- **Spec:** §6.3 — "Registers r5–r31 are initialized to 0… r1–r4 of every thread in every active warp are preloaded."
- **Direction:** Zero register banks of inactive warps in `init_kernel`, or clarify in spec/code whether kernel relaunch should clear data memory.

**FN-M4. `panic_warp/cause/pc` accessors return defaults (all 0) when no panic occurred.**
- **Location:** `sim/include/gpu_sim/functional/functional_model.h:31-34`.
- **Spec:** §6.1 — undefined when STATUS.PANIC clear; cannot distinguish "no panic" from "warp 0 panicked at PC 0 cause 0" (legitimate per spec line 297).
- **Direction:** Gate on `is_panicked()`, return optional, or document.

**FN-m3. ECALL/EBREAK in functional execute loops 32 lanes despite only lane 0 acting.**
- **Location:** `sim/src/functional/functional_model.cpp:202-215,220-224`. No double-latch guard on `latch_panic`.
- **Direction:** Short-circuit out of the lane loop, or guard `latch_panic` to be no-op if already panicked.

**FN-m4. `rs2_val`/`rd_val` read every iteration regardless of decoder's `num_src_regs`.**
- **Location:** `sim/src/functional/functional_model.cpp:104-105`. Stale-register reads possible for I-type/LUI/AUIPC; results unused, harmless today.
- **Direction:** Read only operands the decoder declares.

---

## 11 (Pass 2). TimingModel Orchestration

**TM-N1. `--max-cycles` cutoff bypasses panic and pipeline-drained termination.**
- **Location:** `sim/src/timing/timing_model.cpp:556-564` (`run()`). Loop simply breaks at limit; no flush of in-flight ops, no panic trigger, no warning. Stats can be partial.
- **Direction:** Set a non-zero exit code path or `Stats::truncated_run` flag, and consider triggering panic to flush gracefully.

**TM-N2. ECALL deactivation coincident with armed panic produces ambiguous trace state.**
- **Location:** `sim/src/timing/timing_model.cpp:266-270`. ECALL retires the warp before flush cascade runs at commit; snapshot shows warp `RETIRED` while panic is also active.
- **Direction:** When `pending_panic_flush_` is set, defer ECALL deactivation through the flush path.

**TM-N3. `pipeline_drained()` doesn't directly check `gather_file_` busy state.**
- **Location:** `sim/src/timing/timing_model.cpp:276-287`. Relies on `cache_->is_idle()` covering it; if a gather buffer has `busy=true` with `filled_count<WARP_SIZE`, simulator could declare completion with a partially-filled load.
- **Direction:** Add `gather_file_->is_idle()` to the drained predicates, or audit `cache_->is_idle()` to confirm coverage.

**TM-N4. Possible `accept()`/`evaluate()` race writing `current_busy_`.**
- **Location:** `sim/src/timing/timing_model.cpp:249-274` vs `:484-488`. `dispatch_to_unit` calls `unit->accept(input, cycle_)` before units' own `evaluate()`; if any unit's `accept()` writes `current_*` rather than `next_*`, snapshot at line 577-580 could see this cycle's accept as already committed.
- **Direction:** Verify each `accept()` writes only `next_*`; document any exception.

**TM-N5. `set_drained_query` lambda captures `[this]`; destructor order fragile.**
- **Location:** `sim/src/timing/timing_model.cpp:201`. `panic_` destructs before `wb_arbiter_` etc.; if the destructor invokes the callable, it touches partially-destructed state.
- **Direction:** Document the lifetime invariant or have `PanicController`'s destructor null the callback.

**TM-N6. Cycle counter `cycle_` is 1-based after first `tick()`; `finalize_trace` `end_cycle = cycle_ + 1` extends one cycle past the last simulated tick.**
- **Location:** `sim/src/timing/timing_model.cpp:338-339,567-568`. Internally consistent; may surprise users expecting 0-based reports.
- **Direction:** Audit `total_cycles` semantics.

---

## Pass 2 — Cross-cutting observations

1. **Panic is the largest cluster of unresolved issues.** Pass 1 found one critical (PA-C1 / TM-C1). Pass 2 added FE-C2 (front-end not flushed), PA-N6 (cache/coalescer not flushed), PA-N3 (LDST FIFO drain confusion), PA-N1 (drain throughput), PA-N5 (committed effects during drain), WB-N3/PA-N7 (committed_entry stale), and PA-N9-N13. Together the panic boundary is structurally underspecified in code; a focused refactor pass is warranted.
2. **Scoreboard-and-flush asymmetry.** SCB-M2 / OC-N6 / WB-N1 all converge on the same fix: either reset the scoreboard alongside `branch_tracker_` in the panic-flush cascade, or implement `TimingModel::reset()` (PA-M2) and have panic→reset be the documented recovery sequence.
3. **DRAMSim3 contract gaps.** Pass 1 found three silent-failure paths (MEM-M1/M2/M3); Pass 2 added FIFO-ordering violation (MEM-M5), reentrancy (MEM-M6), and MSHR-id reuse (MEM-M7). The DRAMSim3 wrapper would benefit from a contract sweep.
4. **Predictor-update site is wrong on multiple axes.** OC-N1 (counts JAL/JALR), OC-N2 (uses `pc+4` for actual_target), OC-N8 (runs before resolve). All three are silent today because the static predictor is a no-op, but together they make the `update()` plumbing unsafe to swap in a stateful predictor.
5. **Stats / observability gaps proliferate.** Pass 2 surfaced: VDOT8 vs MUL split (EU2-n1 / OC-N5), addr-gen FIFO backpressure (LS-m5), panic drain counters (PA-N1/N2), pin-defer vs coalescer-pin separation (CA-C4), DRAMSim3 queue-wait (MEM-m9). Combined with pass-1's metric ambiguities, a single counter-catalog review is overdue.

---

## Combined triage (Pass 1 + Pass 2 critical / high)

| # | Item | Section |
|---|------|---------|
| 1 | Shadow-path EBREAK can panic the SM | FE-C1 |
| 2 | Panic-trigger cycle still runs full evaluate sweep | PA-C1 / TM-C1 |
| 3 | EBREAK trigger cycle does not flush fetch / decode / instr_buffer / warp PC | **FE-C2 (P2)** |
| 4 | FENCE accepted as valid NOP | FN-C1 |
| 5 | `process_store` omits per-lane MSHR fields | CA-C1 |
| 6 | Cache deferred fill HOL-blocks unrelated fills | **CA-C2 (P2)** |
| 7 | DRAMSim3 read responses violate FIFO ordering | **MEM-M5 (P2)** |
| 8 | `--max-cycles` cutoff bypasses drain | **TM-N1 (P2)** |
| 9 | DRAMSim3 silent return-false after stripped assert | MEM-M1 / MEM-M2 |
| 10 | DRAMSim3 silent unknown-callback discard | MEM-M3 |
| 11 | DRAMSim3 reentrant callback during ClockTick | **MEM-M6 (P2)** |
| 12 | DRAMSim3 MSHR-id reuse vs stale chunk-to-mshr map | **MEM-M7 (P2)** |
| 13 | Asymmetric branch-shadow source-of-truth | **SCH-M3 (P2)** |
| 14 | Scoreboard not reset on panic flush | **SCB-M2 / OC-N6 / WB-N1 (P2)** |
| 15 | INVALID instructions latch undocumented cause `0x2` | PA-M1 / FN-M1 |
| 16 | No `TimingModel::reset()` (CTRL.RESET unimplemented) | PA-M2 |
| 17 | Cache/coalescer not in panic flush cascade | **PA-N6 (P2)** |
| 18 | Speculative wrong-path fetch can throw on out-of-range PC | **FE-m8 (P2)** |
| 19 | LDST FIFO included in execution_units_drained against spec | **PA-N3 (P2)** |
| 20 | Cache/mem committed effects during panic | **PA-N5 (P2)** |

