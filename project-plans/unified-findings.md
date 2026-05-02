# Unified Findings — Timing Bug Hunt + Spec-vs-Sim Audit

> Consolidated from `timing-bug-hunt-findings.md` (20 numbered findings F-01..F-20 from
> the operational Phase 0–9 bug hunt) and `spec-vs-sim-audit.md` (Pass 1 + Pass 2,
> ~85 letter-coded items from a read-only spec-vs-impl review).
> All findings preserved; duplicates merged with both source codes cited.

## How to use this file

- Items are organized into priority tiers **P0 (critical)** → **P5 (observability gap)**.
- Within each tier, items are grouped loosely by subsystem so related work clusters.
- Each item carries: source code(s), severity from the original doc, file:line citation,
  short evidence summary, and a one-line direction.
- To resolve: strike through (or delete) the item and note the commit. Items that need
  user judgment ("rename A vs change impl B?") are flagged **DECISION**.

## Priority tiers

- **P0 — Critical:** observable wrong behavior, silent failure, spec compliance violation.
- **P1 — High:** definite known bugs/contract violations that don't crash but mislead;
  structural debt with high regression risk (7b5f713-class footguns).
- **P2 — Moderate:** discipline gaps, counter inconsistencies, probable bugs in
  plausible configurations, fragile invariants.
- **P3 — Minor:** doc drift, naming, defensive-depth, naming inconsistencies.
- **P4 — Spec ambiguity:** spec wording vs impl mismatch, no code bug; needs spec author.
- **P5 — Observability gap:** instrumentation/CLI improvement; no bug.

---

## P0 — Critical

### P0-1. Panic trigger cycle still executes one full evaluate sweep before flush
- **Sources:** PA-C1 / TM-C1 (audit pass 1)
- **Severity:** Critical
- **Location:** Arming at `sim/src/timing/timing_model.cpp:401-407` vs. forward sweep
  at `:437-525`; flush at `:535-541`.
- **Spec:** §4.8.1 step 1 — "the warp scheduler is immediately inhibited."
- **Observation:** When a fresh ebreak is observed, `panic_->trigger()` happens after
  the panic-active early-exit check. The cycle still runs fetch / decode / scheduler /
  opcoll / dispatch / units / cache+mem / wb_arbiter, then commits, then runs the flush
  cascade. One cycle of "normal" issue+writeback updates the scoreboard / register file.
- **Direction:** Reorder so the trigger-cycle takes the panic branch immediately
  (arm before the `is_active()` check, or run flush before evaluate and skip evaluate).

### P0-2. Shadow-path EBREAK can panic the SM
- **Sources:** FE-C1 (audit pass 1)
- **Severity:** Critical
- **Location:** `sim/src/timing/decode_stage.cpp:22-27,44,52-55,72-76`; observed at
  `sim/src/timing/timing_model.cpp:401-407`.
- **Spec:** §4.2 (speculative fetch past unresolved branch must be flushed); §4.8
  (panic latched on the *committed* program path).
- **Observation:** A wrong-path EBREAK fetched between branch issue and the redirect-apply
  commit is decoded into `next_ebreak_request_`, latched into `current_ebreak_request_`
  by `decode_.commit()`, and panicked unconditionally at the top of the next tick.
  `apply_redirect_invalidate` only clears `pending_`; it never inspects/clears the
  ebreak request slots.
- **Direction:** In `DecodeStage::commit`, when a redirect is valid for the EBREAK's warp,
  also clear `next_ebreak_request_` (and/or `current_ebreak_request_`); or gate the
  `timing_model.cpp:401` trigger on `branch_in_flight==false`.

### P0-3. EBREAK trigger cycle does not flush fetch / decode / instr_buffer / warp PC
- **Sources:** FE-C2 (audit pass 2)
- **Severity:** High (P2 critical/high band)
- **Location:** `sim/src/timing/timing_model.cpp:401-407,535-541`;
  `sim/src/timing/fetch_stage.cpp:75-76`.
- **Spec:** §4.1 / §4.8 — panic flush should clean wrong-path state.
- **Observation:** `panic_->trigger()` runs only after the panic-active early-exit check,
  so the full tick body still runs. fetch.evaluate fetches another instruction for the
  EBREAK warp and mutates `warps_[w].pc`. The end-of-tick `pending_panic_flush_`
  cascade flushes scheduler / opcoll / gather_file / wb_arbiter but does NOT flush
  fetch's `current_output_`, decode's `pending_`, the per-warp `instr_buffer`, or roll
  back the warp PC. The warp arrives at panic drain with PC two instructions past the
  EBREAK and a polluted instruction buffer.
- **Direction:** Add `fetch_->flush()`, `decode_->flush()`, per-warp `instr_buffer.flush()`,
  and snapshot the EBREAK warp's PC at trigger time into the `pending_panic_flush_` cascade.

### P0-4. FENCE decoded as valid NOP — spec violation
- **Sources:** FN-C1 (audit pass 1)
- **Severity:** Critical
- **Location:** `sim/src/decoder.cpp:209-214` (comment even says "NOP per RV32I base spec").
- **Spec:** §2.1 — "FENCE is not supported in Phase 1."
- **Direction:** Decoder should map `OP_FENCE` → `InstructionType::INVALID`.

### P0-5. `process_store` never captures per-lane store_data / store_byte_en / byte_offsets
- **Sources:** CA-C1 (audit pass 1)
- **Severity:** Critical
- **Location:** `sim/src/timing/cache.cpp:116-192`; `MSHREntry` fields exist but
  unwritten (`mshr.h:23-25`).
- **Spec:** §5.3.1 MSHR field table mandates these fields; §5.3.1 store-miss fill /
  secondary wake explicitly merges store data into the resident line.
- **Observation:** Timing model is tag-only by design. Functional correctness rides on
  trace replay, but the spec text describes hardware the simulator does not model. The
  omission is undeclared.
- **Direction:** Either annotate the omission explicitly in `cache.h` / `mshr.h`
  referencing §5.3.1, or wire `process_store` to populate the spec-mandated fields.

### P0-6. Cache deferred fill in `pending_fill_` HOL-blocks unrelated fills
- **Sources:** CA-C2 (audit pass 2)
- **Severity:** High
- **Location:** `sim/src/timing/cache.cpp:265-291` (`handle_responses`) +
  `complete_fill` lines 207-218 (pin-defer) and 228-233 (write-buffer-full defer).
- **Spec:** §5.3 fill-port cadence (one per cycle); §5.3.1 explicitly authorizes
  blocking only for write-buffer-full store-fill case.
- **Observation:** Pin-defer also parks the response in `pending_fill_`; `handle_responses`
  early-returns whenever `current_pending_fill_.valid`, blocking unrelated fills (different
  sets) until the pin clears. Spec scope of HOL blocking was narrower.
- **Direction:** Move deferred response into a side-buffer keyed by target line so
  `handle_responses` can skip past it; or document this HOL behavior in §5.3.1 as an
  additional cascading stall.

### P0-7. `find_chain_tail` non-defensive against malformed chains
- **Sources:** CA-C3 (audit pass 2)
- **Severity:** Critical (silent corruption path)
- **Location:** `sim/src/timing/mshr.cpp:39-49`.
- **Observation:** Returns -1 if no entry has `next_in_chain == INVALID`. -1 is then
  interpreted as "no existing line" (`cache.cpp:91/170`), so the cache silently allocates
  a primary and submits a duplicate external read, orphaning the existing chain.
- **Direction:** Assert that if any valid entry matches `line_addr`, function must return
  a non-negative tail.

### P0-8. DRAMSim3 read responses violate FIFO ordering across MSHRs
- **Sources:** MEM-M5 (audit pass 2)
- **Severity:** High
- **Location:** `sim/src/timing/dramsim3_memory.cpp:181-207`.
- **Spec:** §5.6 lines 477, 436 — "FIFO-ordered".
- **Observation:** A read response is pushed to `responses_` only when the *last* chunk
  of its MSHR returns. DRAMSim3's bank/refresh/locality scheduler reorders chunk
  completions across MSHRs freely. Cache routes by `mshr_id`, so it tolerates this — but
  the spec contract is not actually held. FixedLatency preserves submit order;
  DRAMSim3 doesn't.
- **Direction:** Tighten spec wording to "responses are unordered, routed by mshr_id,"
  OR add a per-submit sequence number and stall-emit until the head completes.
  **DECISION** (spec vs impl).

### P0-9. `--max-cycles` cutoff bypasses panic and pipeline-drained termination
- **Sources:** TM-N1 (audit pass 2)
- **Severity:** High
- **Location:** `sim/src/timing/timing_model.cpp:556-564` (`run()`).
- **Observation:** Loop simply breaks at limit; no flush of in-flight ops, no panic
  trigger, no warning. Stats can be partial.
- **Direction:** Set a non-zero exit code path or `Stats::truncated_run` flag, and
  consider triggering panic to flush gracefully.

### P0-10. DRAMSim3 silent `return false` after stripped-in-NDEBUG assert
- **Sources:** MEM-M1 / MEM-M2 (audit pass 1) + F-15 (findings)
- **Severity:** Moderate (audit) / Suspicious (findings) — promoted to P0 because
  release-build silent infinite hang.
- **Location:** `sim/src/timing/dramsim3_memory.cpp:66-78`; cache caller at
  `sim/src/timing/cache.cpp:110,188` (caller ignores return).
- **Spec:** §5.6 lines 484-486 — `submit_read` must never return false in legitimate
  cache traffic; cache call sites do not check the bool.
- **Observation:** In release builds the assert vanishes; a violated invariant becomes a
  silent false return → MSHR leaked → infinite hang. Architectural invariant
  (`num_mshrs * chunks_per_line ≤ in_flight capacity`) prevents this in practice, but
  defense-in-depth is debug-only.
- **Direction:** Replace post-assert `return false` with `throw std::logic_error`
  (or another always-on check). Same fix for the MSHR-id-reuse path.

### P0-11. DRAMSim3 silently discards unknown read/write callbacks
- **Sources:** MEM-M3 (audit pass 1)
- **Severity:** Moderate — promoted because masks real bugs
- **Location:** `sim/src/timing/dramsim3_memory.cpp:181-184,209-211,186-188,216-218`.
- **Observation:** A spurious or doubled DRAMSim3 callback is a real bug — silent return
  masks it.
- **Direction:** Convert to assert or add `spurious_dramsim3_callbacks` counter.

### P0-12. Reentrant DRAMSim3 callbacks mutate `responses_` mid-`ClockTick()`
- **Sources:** MEM-M6 (audit pass 2)
- **Severity:** Moderate — promoted because reentrancy bugs are silent
- **Location:** `sim/src/timing/dramsim3_memory.cpp:135` calls `mem_->ClockTick()`;
  callbacks fire synchronously inside the tick and mutate `read_assembly_` /
  `write_assembly_` / maps / `responses_`.
- **Observation:** Capacity assertion checks each push individually but not jointly; an
  iterator over `read_assembly_` around `ClockTick` would be invalidated.
- **Direction:** Document the reentrancy contract and prove every container access is
  reentrancy-safe, or stage completions into a temporary list and drain after
  `ClockTick` returns.

### P0-13. DRAMSim3 MSHR-id reuse vs. stale chunk-to-mshr map entries
- **Sources:** MEM-M7 (audit pass 2)
- **Severity:** Moderate — promoted because double-completion path
- **Location:** `sim/src/timing/dramsim3_memory.cpp:56-92`; `read_chunk_to_mshr_`
  populated at line 129.
- **Observation:** Cache reuses MSHR ids freely. `read_chunk_to_mshr_` keys on byte
  address; if a duplicate or post-completion DRAMSim3 callback arrives for the previous
  transaction's chunk, it decrements the *new* MSHR's `chunks_remaining` and double-
  completes it. Safety relies on DRAMSim3 never delivering duplicates — undocumented.
- **Direction:** Tag chunks with a monotonically increasing transaction id and key the
  reverse map on it.

---

## P1 — High / Definite Bugs

### P1-1. `total_loads_completed` and `total_load_latency` declared, documented, never incremented
- **Sources:** F-02 (findings)
- **Severity:** Definite
- **Location:** `sim/include/gpu_sim/stats.h:62-63` (declarations);
  `sim/src/stats.cpp:146-147` (JSON emit), `sim/src/stats.cpp:86-89` (text report).
  No increment site exists in `sim/`, `runner/`, or `tests/`. Doc reference at
  `resources/trace_and_perf_counters.md:418`.
- **Observation:** All 42 (config × bench) Phase-1 tuples report 0 for both fields,
  despite e.g. matmul/fixed-default executing 32768 loads. Text report at
  `stats.cpp:86-89` divides one by the other for "average load latency"; with both at 0
  the report prints nothing useful.
- **Direction:** Either wire the increment sites (per load completion in cache /
  gather buffer) or remove the counters and the doc reference.

### P1-2. `writeback_conflicts` measures backlog cycles, not simultaneous-completion events
- **Sources:** F-05 (findings) + WB-M1 (audit pass 1)
- **Severity:** Definite
- **Location:** `sim/src/timing/writeback_arbiter.cpp:24-36`;
  `sim/src/timing/{alu,multiply,divide,tlookup,ldst}_unit.cpp::has_result` (sticky valid).
- **Spec:** §4.7 — "conflicts only occur when two or more units finish their last thread
  lane in the same cycle."
- **Observation:** `has_result()` returns the unit's `next_result_buffer_.valid` —
  sticky until consumed. After the arbiter selects a winner and calls `consume_result()`,
  only that unit's slot clears. Other units' valid bits persist into next cycle. So
  `valid_count > 1` is true for every cycle that two or more units are *waiting on the
  arbiter*, not just the cycle they completed. Counter is therefore "cycles with
  multiple sources holding ready results" — backlog occupancy, not transition events.
  layernorm_lite/fixed-default reports 27.9% (spec implies single-digit %).
  Additionally, a 3-way conflict adds 1, not 2 — under-reports backlog by `(valid_count-2)`
  per cycle (audit observation).
- **Direction:** **DECISION.** Recommended: rename current counter to
  `writeback_backlog_cycles` and add a transition-event counter
  `writeback_simultaneous_completions` that increments only when a unit's `has_result()`
  flips false→true AND another unit also produced this cycle. Aligns code, doc, spec.

### P1-3. WritebackArbiter `pending_commit_` retains stale state; `has_pending_work()` reads live next_*
- **Sources:** F-11 (findings) + WB-M3 (audit pass 1)
- **Severity:** Definite (minor; +1 cycle drain detection delay; latent panic-flush dependency)
- **Location:** `sim/src/timing/writeback_arbiter.cpp:19,42,51-53,55-67`;
  `sim/src/timing/timing_model.cpp:286,347-381,539`.
- **Observation:** `pending_commit_` is set in evaluate, latched via commit
  (`pending_commit_` → `committed_`), but `pending_commit_` itself is cleared only at
  the **next** evaluate, NOT in commit. End-of-tick `pipeline_drained()` reads
  `wb_arbiter_->has_pending_work()` which returns true because `pending_commit_` has a
  value, even though work was already committed. Drain detection delayed 1 cycle.
  **Worse:** during panic, `wb_arbiter.evaluate` does not run, so `pending_commit_`
  from before panic is never cleared until the panic-flush cascade calls
  `wb_arbiter_->flush()→reset()` at line 539. The dependency on the panic flush for state
  hygiene is implicit and undocumented; if the flush is ever moved or removed,
  `pending_commit_` permanently reports `has_pending_work=true` during panic, blocking
  drain (only saved by MAX_DRAIN_CYCLES timeout).
- **Direction:** Clear `pending_commit_` at the end of `commit()`, or read `committed_` +
  source `has_result()` in `has_pending_work()`. Either fix removes the implicit
  panic-flush dependency.

### P1-4. `serialized_requests` increments per-warp-decision, not per-lane (counter ↔ doc/spec mismatch)
- **Sources:** F-18 (findings)
- **Severity:** Definite
- **Location:** `sim/src/timing/coalescing_unit.cpp:46-51` (one increment per coalesced/
  serialized decision); `resources/trace_and_perf_counters.md:415` (doc claim).
- **Spec:** §5.2 ("falls back to 32 serialized individual requests") implies per-lane.
- **Observation:** Doc says "totals per lane, not per warp"; impl increments once per
  warp's straddle decision. Per-lane requests *do* reach the cache (confirmed by 33
  cache lookups for one straddling warp) — counter just doesn't track them.
- **Direction:** **DECISION.** Either rename to `serialized_decisions` (match impl), OR
  change impl to `stats_.serialized_requests += WARP_SIZE` (match doc/spec). Affects
  calibration of `coalesced/serialized` ratio.

### P1-5. `EXECUTE_DIV` trace slice systematically 1 cycle short of `div_busy_cycles`
- **Sources:** F-20 (findings)
- **Severity:** Definite
- **Location:** `sim/src/timing/divide_unit.cpp:20-32` (counter increment site);
  `sim/include/gpu_sim/timing/divide_unit.h:30-32` (`pending_entry()` reading
  `current_busy_`); `sim/src/timing/timing_model.cpp:670-673` (trace state set from
  `pending_entry()`).
- **Observation:** divide_chain shows 18 DIVs with `div_busy_cycles=576` (=18×32) but
  every `execute_div` trace event has `dur=31`. Two observability surfaces disagree by
  exactly 1 cycle per DIV. Likely same pattern for `execute_mul` and `execute_tlookup`
  (not verified). Operator-facing impact: anyone deriving per-DIV latency from trace
  slice widths under-counts by 1 cycle.
- **Direction:** **DECISION.** Either (a) defer `current_busy_=false` commit by one
  cycle so the trace covers 32 cycles, or (b) accept the convention that the 32nd cycle
  is "writeback-ready" and decrement `busy_cycles` to match.

### P1-6. INVALID instructions latch hardware panic cause `0x2` not in spec
- **Sources:** PA-M1 / FN-M1 (audit pass 1, cross-referenced)
- **Severity:** Moderate — promoted because spec compliance and bypasses panic flow
- **Location:** `sim/src/functional/functional_model.cpp:220-224`.
- **Spec:** §4.8.2 — only EBREAK in Phase 1; §4.8.3 — causes `0x01–0xFF` reserved,
  none defined.
- **Observation:** Defines a hardware cause not in spec; bypasses `PanicController` so
  drain/scheduler-flush/all-warps-inactive don't engage; only `FunctionalModel.panicked_`
  is set.
- **Direction:** Either remove the INVALID hardware-panic path until Phase 2, or route
  it through `panic_->trigger()` and document cause `0x2` in §4.8.3.

### P1-7. No `TimingModel::reset()` for CTRL.RESET semantics
- **Sources:** PA-M2 (audit pass 1)
- **Severity:** Moderate — promoted because blocks panic recovery and ties to many
  panic-flush items
- **Location:** `sim/src/timing/timing_model.cpp` (none); `sim/src/timing/panic_controller.cpp`
  has `reset()` but it is not called from any host-visible path.
- **Spec:** §6.1 / §6.6 — CTRL.RESET clears all panic state including
  PANIC_WARP/LANE/PC/CAUSE.
- **Observation:** After a panic, the only way to clear is to construct a fresh TimingModel.
- **Direction:** Add `TimingModel::reset()` propagating to all sub-components; wire to a
  CSR-block reset path when CSR control is implemented.

### P1-8. Scoreboard not reset on panic flush (asymmetric vs branch_tracker)
- **Sources:** SCB-M2 / OC-N6 / WB-N1 (audit pass 2, three-way cross-referenced)
- **Severity:** Moderate — promoted because converges with PA-M2 / panic-flush cluster
- **Location:** `sim/src/timing/timing_model.cpp:535-542`;
  `sim/include/gpu_sim/timing/scoreboard.h:12-15`.
- **Spec:** §6 line 620 says scoreboard intentionally abandoned, with CTRL.RESET
  cleaning up. But CTRL.RESET path is missing (P1-7); tracker/scoreboard treatment is
  inconsistent.
- **Observation:** After panic flush, scoreboard pending bits set by issued-but-not-
  yet-written-back instructions remain set. If the simulator is reused without full
  reconstruction, those stale bits block future issues.
- **Direction:** Either add `scoreboard_.reset()` to the cascade, or implement
  `TimingModel::reset()` (P1-7) and make panic→reset the documented recovery sequence.

### P1-9. `pending_panic_flush_` cascade does not reset cache or coalescer
- **Sources:** PA-N6 (audit pass 2)
- **Severity:** Moderate — promoted with panic cluster
- **Location:** `sim/src/timing/timing_model.cpp:535-542`.
- **Observation:** In-flight pre-panic LDST entries continue draining during panic,
  contradicting "abandon MSHR fills / write buffer drains".
- **Direction:** Reset/abandon `coalescing_`'s active entry and cache MSHR queue at
  `pending_panic_flush_` time.

### P1-10. LDST FIFO included in `execution_units_drained()` against spec scope
- **Sources:** PA-N3 (audit pass 2)
- **Severity:** Moderate — promoted with panic cluster
- **Location:** `sim/src/timing/timing_model.cpp:289-297`.
- **Spec:** §4.8.1 scopes drain to "execution units and writeback only".
- **Observation:** LDST FIFO entries can only retire by being consumed by
  `coalescing_->evaluate()` → cache; if cache stalls on MSHR_FULL/WRITE_BUFFER_FULL,
  panic drain hits 32-cycle timeout — caused by the very memory subsystem the spec
  says shouldn't gate drain.
- **Direction:** Drop `ldst_->fifo_empty()` from `execution_units_drained()`, or treat
  FIFO entries as abandoned the same way MSHRs are.

### P1-11. Cache/coalescer/mem_if continue committed effects + counter increments during panic
- **Sources:** PA-N5 (audit pass 2)
- **Severity:** Moderate — promoted because spec says no committed effects post-panic
- **Location:** Cache/coalescer/mem_if calls in `sim/src/timing/timing_model.cpp:348,360-362`.
- **Spec:** §4.8.1: "must not produce any new architecturally committed effects after
  panic is active."
- **Observation:** `cache_->drain_write_buffer()` continues issuing external writes;
  counters (`coalesced_requests`, `cache_hits`, `external_memory_writes`) increment as
  if architecturally retired.
- **Direction:** Gate counter updates and external submissions on `!panic_active`, OR
  clarify spec wording (Phase-1-acceptable per spec line 266 observability allowance).
  **DECISION** (spec vs impl).

### P1-12. Asymmetric source-of-truth between scheduler issue and mispredict-clear (predication time bomb)
- **Sources:** SCH-M3 (audit pass 2)
- **Severity:** Moderate — promoted as latent footgun for Phase-2 predication
- **Location:** `sim/src/timing/warp_scheduler.cpp:163-169` (set, gates on `decoded.type`);
  clear sites at `sim/src/timing/timing_model.cpp:461` and operand_collector
  `resolve_branch` (gate on `out.trace.is_branch`); `sim/src/functional/functional_model.cpp:128/137/145`
  only sets `is_branch=true` when lane == 0.
- **Observation:** If lane 0 is ever inactive (predication, masked warp, future
  divergence), scheduler still asserts `note_branch_issued` but `resolve_branch` /
  `apply_redirect` never fire. Bit set with no clear → permanent stall for that warp.
  Currently masked because Phase 1 has no predication; hidden time bomb.
- **Direction:** Drive both ends from the same predicate — gate `note_branch_issued` on
  `out.trace.is_branch` or vice versa.

### P1-13. `instr_buffer` is the only first-class warp-state structure not under next_/current_ discipline
- **Sources:** F-16 (findings) — "the most structurally significant finding from re-audit"
- **Severity:** Probable — design-debt; structural risk for 7b5f713-class regressions
- **Location:** `sim/src/timing/warp_scheduler.cpp:154`;
  `sim/src/timing/decode_stage.cpp:59`; `sim/src/timing/fetch_stage.cpp:54-66,124`;
  `sim/include/gpu_sim/timing/warp_state.h`.
- **Observation:** `WarpState::instr_buffer` (per-warp circular buffer of decoded
  instructions) is mutated directly mid-tick by 4 different stages with NO next_/current_
  pair. Saved from 7b5f713-class regression today only by strict tick order:
  fetch.evaluate (read pre-pop) → scheduler.evaluate (pop) → decode.commit (push) →
  fetch.commit::apply_redirect (flush). If any ordering changes OR a new mid-tick reader
  observes post-pop / post-push state, the 7b5f713 bug class returns. The 7b5f713 fix
  added `inflight_to_w` accounting in fetch's eligibility gate to compensate.
- **Direction:** Double-buffer `instr_buffer` (next_buffer / current_buffer with commit()
  flip), aligning it with the rest of the timing model's REGISTERED state.

### P1-14. `DecodeStage::pending_` is single-buffered; READY/STALL stability claim is brittle
- **Sources:** F-08 (findings)
- **Severity:** Suspicious / Probable design fragility
- **Location:** `sim/include/gpu_sim/timing/decode_stage.h:42-45,54`;
  `sim/src/timing/decode_stage.cpp:35-37,60`; tick-order at
  `sim/src/timing/timing_model.cpp:437-438`.
- **Observation:** `ready_to_consume_fetch()` and `pending_warp()` both read
  `pending_.valid`, mutated mid-tick by `DecodeStage::evaluate()`. Discipline doc claim
  that READY/STALL accessor values are "stable across the entire evaluate phase
  regardless of where queried" is technically false here — value depends on whether the
  read happens before or after decode.evaluate. Currently safe ONLY because
  `FetchStage::evaluate` (the sole mid-tick caller) runs before `decode.evaluate` per
  tick order. Inserting any new mid-tick reader of `ready_to_consume_fetch()` after
  `decode.evaluate` would silently observe inconsistent state — exactly the 0383f04
  bug template.
- **Direction:** Either (a) double-buffer `pending_` (REGISTERED current_/next_ pair),
  or (b) tighten the row 1 inventory comment to "stable only when read before
  decode.evaluate within the same tick."

### P1-15. `LoadGatherBuffer::busy` is unclassified cross-stage edge between WBArbiter and Coalescing
- **Sources:** F-13 (findings)
- **Severity:** Suspicious / Probable design fragility
- **Location:** `sim/src/timing/load_gather_buffer.cpp:115-118` (consume_result write);
  `sim/src/timing/coalescing_unit.cpp:28` (is_busy read);
  `sim/src/timing/timing_model.cpp:494,500` (tick order).
- **Observation:** `WritebackArbiter::evaluate` calls `consume_result()` on
  LoadGatherBuffer sources, which directly mutates `buffers_[idx].busy=false`.
  `CoalescingUnit::evaluate` reads `gather_file_.is_busy(warp_id)`. Coalescing.evaluate
  (line 494) runs BEFORE wb_arbiter.evaluate (line 500), so coalescing observes
  pre-consume `busy=true` this cycle, post-consume `busy=false` next cycle. Functionally
  correct due to tick order, but unclassified — neither REGISTERED, COMBINATIONAL, nor
  READY/STALL. Discipline-doc inventory has no row covering it.
- **Direction:** Add an explicit inventory row (Row 16) for this edge, classify formally
  (likely COMBINATIONAL with tick-order constraint).

### P1-16. Speculative wrong-path fetch can throw on out-of-range PC
- **Sources:** FE-m8 (audit pass 2)
- **Severity:** Minor in audit — promoted because can crash sim
- **Location:** `sim/include/gpu_sim/functional/memory.h:72-78` called from
  `sim/src/timing/fetch_stage.cpp:70`.
- **Observation:** `InstructionMemory::read` throws `std::out_of_range`. After a
  mispredicted JAL/BRANCH, fetch.evaluate at cycle N speculatively reads at the
  wrong-path PC; if past the loaded program, kills the simulator. Real hardware would
  see garbage decoding to INVALID and get flushed.
- **Direction:** Return 0 / harmless encoded INVALID on out-of-range, or guard the
  fetch read.

---

## P2 — Moderate / Discipline & Probable Bugs

### Frontend

- **F-03 — Scheduler stall partition leaves "all-warps-inactive" bucket uncounted.**
  41/42 Phase-1 tuples have `scheduler_idle_cycles ≠
  scheduler_frontend_stall_cycles + scheduler_stall_backend_cycles`. The third case
  `(!any_buffer_empty && !any_active)` (program tail / pipeline drain) increments only
  `scheduler_idle_cycles`. `sim/src/timing/warp_scheduler.cpp:139-144`.
  **DECISION:** add `scheduler_idle_drain_cycles` OR widen
  `scheduler_stall_backend_cycles` to cover `!any_buffer_empty` regardless of any_active.

- **FE-M1 — Warp PC mutated inside `evaluate()` instead of via REGISTERED shadow.**
  `sim/src/timing/fetch_stage.cpp:75-76`. No current consumer reads mid-evaluate; latent
  footgun. Direction: stage `next_pc` per-warp and apply in `commit()`.

- **FE-M2 — `apply_redirect` invalidation of `next_output_` is dead.**
  `sim/src/timing/fetch_stage.cpp:129-131`. Comment misleading; spec's "in-flight fetch
  register" is `current_output_`. Direction: drop the dead invalidate; tighten comment.

### Scheduler / Scoreboard / Branch Shadow

- **SCH-M1 — `query_unit_ready(SYSTEM)` returns hard-coded `true`.**
  `sim/src/timing/warp_scheduler.cpp:63`. ECALL/EBREAK ride SYSTEM and bypass
  backpressure. Benign today; inconsistent with uniform-eligibility rule.
  Direction: declare always-ready in spec/comment, or wire a real ready signal.

- **SCH-M2 — Silent "all ready" default if `set_consumers` is forgotten.**
  `sim/include/gpu_sim/timing/warp_scheduler.h:84-95`,
  `sim/src/timing/warp_scheduler.cpp:40-66`. Direction: make `set_consumers` a
  constructor parameter, or assert "wired-or-overridden" in evaluate.

### Operand Collector / Branch Resolution

- **OC-M1 — `branch_predictions` counter overcounts JAL.**
  `sim/src/timing/timing_model.cpp:461-462`. Direction: rename to `branch_resolves`,
  or gate on `decoded.type == BRANCH || JALR`.

- **OC-M2 — `current_redirect_request_` retains stale `valid` between consumer reads.**
  `sim/src/timing/operand_collector.cpp:54-55`. Consumers fire correctly at commit-time;
  debug observers between commits see stale "valid" pulse. Direction: consumers consume-
  and-clear, or clear `current_` after fetch+decode have read it.

- **OC-M3 — `next_output_` cleared every evaluate; relies on dispatch always accepting same tick.**
  `sim/src/timing/operand_collector.cpp:16`. Direction: document the assumption or add
  a defensive check.

- **OC-N1 — `branch_predictor_->update()` called for JAL/JALR.**
  `sim/src/timing/timing_model.cpp:461-464`. Static stub ignores args; future stateful
  predictor (2-bit, BTB) polluted by JAL/JALR samples. Direction: gate on `decoded.type
  == BRANCH`.

- **OC-N2 — `actual_target` for not-taken branches passed as `pc+4`, not `branch_target`.**
  `sim/src/timing/timing_model.cpp:466-468`. Future BTB cannot learn static target on
  not-taken samples. Direction: pass `out.trace.branch_target` unconditionally as
  `actual_target`; use `actual_taken` for direction.

- **OC-N3 — Two branches resolving same tick lose one redirect.**
  `sim/src/timing/operand_collector.cpp:58-76`. `resolve_branch` overwrites
  `next_redirect_request_` without checking `valid`. Currently unreachable, fragile.
  Direction: add `assert(!next_redirect_request_.valid)`.

- **OC-N4 — `num_src_regs == 3` fast-path silent on out-of-range values.**
  `sim/src/timing/operand_collector.cpp:12`. Direction: switch on `num_src_regs`
  explicitly with assert.

- **OC-N5 / EU2-n1 — VDOT8 silently merged with MUL stats.**
  `sim/src/decoder.cpp:251-255`; `sim/src/timing/multiply_unit.cpp:19,27`. No
  `vdot_stats`; entries don't carry `InstructionType`. Direction: split into
  `vdot_stats`, or stamp `InstructionType` on the pipeline entry.

- **OC-N7 — `set_redirect_request_override(valid=false, ...)` silently masks real opcoll redirects.**
  `operand_collector.h:130-136`. Direction: have `read_redirect_request` fall through
  when override has `valid==false`.

- **OC-N8 — `branch_predictor->update()` runs before `resolve_branch()`.**
  `sim/src/timing/timing_model.cpp:461-481`. Footgun for any stateful predictor.
  Direction: move `update()` after `resolve_branch()` and gate on `BRANCH`.

### Execution Units

- **F-09 — `ALUUnit::current_has_pending_` is structurally dead state.**
  `sim/include/gpu_sim/timing/alu_unit.h:14-16`; `sim/src/timing/alu_unit.cpp:10-13,33,43`.
  After commit, `current_has_pending_` is structurally always false. Vestigial copy-paste
  from a multi-cycle unit pattern. Direction: remove for clarity.

- **F-10 — `MultiplyUnit::next_pipeline_` has no explicit capacity assertion.**
  `sim/src/timing/multiply_unit.cpp:5-20`. Bound is structural (validated by trace), not
  asserted. Direction: add `assert(next_pipeline_.size() < pipeline_stages_)` in `accept()`
  to catch future scheduler bugs (e.g. test override issuing despite ready_out=false).

- **EU2-n2 — `MultiplyUnit::reset()` doesn't zero `current_/next_result_buffer_.values`.**
  `sim/src/timing/multiply_unit.cpp:62-67`. Snapshot reads after reset see stale 32-lane
  data, gated only by valid bit. Direction: zero entry bodies, or document.

- **EU2-n3 — `consume_result()` non-idempotent and unguarded.**
  All four units. No precondition assert that `next_*.valid == true` on entry; defensive
  double-drain returns stale fields silently. Direction: assert preconditions or return
  `std::optional<WritebackEntry>`.

- **EU2-n4 — Loser-of-arbitration ordering hazard with `next_*` reads after `evaluate()`.**
  `sim/src/timing/writeback_arbiter.cpp:24-48` plus units' `has_result()`. After arbiter
  clears the winner's `next_*.valid`, the unit's `current_result_buffer_` is still valid
  until commit. Trace consumers using `result_entry()` (next_*) vs `pending_entry()`
  (current_*) see inconsistent slot occupancy. Direction: document or unify.

- **EU2-n5 — Inconsistent `busy_cycles` accounting across units.**
  ALU bumps on produce, MUL on every cycle pipeline non-empty, DIV/TLOOKUP on every busy
  tick. "Occupancy %" reports incomparable. Direction: define semantics in spec, or add
  `produce_cycles` for ALU symmetry.

- **EU2-n6 — `TLOOKUP_LATENCY` hardcoded constant 17.**
  `sim/include/gpu_sim/timing/tlookup_unit.h:41`. Spec §2.3 ties latency to
  architectural parameters. Direction: parameterize via `SimConfig`, or assert against
  `WARP_SIZE` / port count.

- **EU2-n7 — `DivideUnit::evaluate()` lacks underflow guard.**
  `sim/src/timing/divide_unit.cpp:24-31`. `reset()` doesn't enforce mutual exclusion of
  `next_busy_=false && next_cycles_remaining_==0`. Direction: guard
  `if (next_cycles_remaining_ > 0)` before decrement.

- **EU2-n8 — Div-by-zero unconditional 32-cycle latency unasserted.**
  `sim/src/timing/divide_unit.cpp:5-18`. Matches spec §4.5 line 225 but no test/assert
  verifies the invariant.

### LD/ST · Coalescing · Gather Buffer

- **LS-M1 — `LdStUnit::accept()` overwrites in-flight without backpressure assertion.**
  `sim/src/timing/ldst_unit.cpp:8-22`. No `assert(!current_busy_ && !next_busy_)`.
  Other units typically assert. Direction: add defensive assert.

- **LS-M2 — `cycles_remaining` decrement in same tick as accept; `num_ldst_units = 0` undefended.**
  `sim/src/timing/ldst_unit.cpp:11-12,27-32`. With N=32, ceil collapses to 1 → push to
  FIFO same tick as accept. With N=0, divide-by-zero. Direction: defer same-tick
  decrement or document; add `Config::validate()` enforcing `num_ldst_units >= 1`.

- **LS-M3 — `is_stalled()` snapshot ordering across MSHR-exhaustion stalls.**
  `sim/src/timing/coalescing_unit.cpp:20`; `sim/src/timing/cache.cpp:69-74,150-155`.
  Cache's `evaluate()` only sets `stalled_` from FILL/secondary; MSHR-exhaustion stall
  is set later by `process_load`/`process_store`. Coalescer's bail check sees stale
  `is_stalled()` next tick. Direction: latch `accepted=false` in coalescer's own
  one-shot stall flag, or move cache stall set to registered side.

- **LS-M4 — `process_store` has no `lane_mask` parameter.** (Cross-check with CA-M1.)
  `sim/src/timing/coalescing_unit.cpp:77-81,102-105`. Bypasses any future write-mask
  plumbing. Direction: add `lane_mask` to `process_store` even if Phase 1 wires to
  all-ones / one-hot.

- **LS-M5 — Coalescer's `is_coalesced` check ignores predicated-off lanes.**
  `sim/src/timing/coalescing_unit.cpp:38-45`. Compares all 32 lane addresses
  unconditionally; garbage addresses in dead lanes force serialization. Direction:
  AND with `trace.active_mask` before comparing.

### L1 Cache · MSHR · Pinning

- **CA-M1 — `lane_mask` of secondary stores left at default 0.**
  `sim/src/timing/cache.cpp:162-170`. Spec §5.3.1 says lane_mask reflects "lanes whose
  data must be merged". Direction: populate (1u<<lane serialized / 0xFFFFFFFF coalesced)
  or document the omission.

- **CA-M2 — HIT-port-loss does not surface via `is_stalled()`.**
  `sim/src/timing/cache.cpp:35-52`. Behavior correct (coalescer retries via own
  `processing_` flag); observability asymmetric. Direction: document the two-channel
  signaling design.

- **CA-C4 — Pin-defer path bumps `line_pin_stall_cycles` without setting `stalled_`.**
  `sim/src/timing/cache.cpp:207-218`. Coalescer-stalled-on-pin and fill-defer share the
  same counter. Direction: introduce `fill_pin_defer_cycles`, or document the
  aggregation.

### External Memory

- **MEM-M4 — `dram_clock_mhz` not sanity-checked against `fpga_clock_mhz`.**
  `sim/src/config.cpp:58-69`. Accepts `dram_clock_mhz = 1e-9`; phase accumulator never
  reaches 1.0 → silent hang. Direction: bound the ratio (e.g. `[0.01, 100]`).

### Writeback / Scoreboard / Register File

- **WB-M2 — Cross-warp gather-buffer queueing invisible in stats.**
  `sim/src/timing/load_gather_buffer.cpp:91-126`. `LoadGatherBufferFile` is one arbiter
  source but has internal RR over per-warp buffers. Two warps' loads completing same
  cycle: arbiter sees one "ready" so `writeback_conflicts` doesn't increment for the
  second warp. Direction: expose per-warp gather buffers as separate arbiter sources, or
  add `gather_buffer_writeback_queue_cycles`.

- **WB-N2 — Functional write at issue precedes scoreboard set in `next_`.**
  `sim/src/timing/warp_scheduler.cpp:150-157`;
  `sim/src/functional/functional_model.cpp:230-234`. Scheduler invokes
  `func_model_.execute(...)` which writes `reg_file_[w][lane][rd] = result` BEFORE
  `scoreboard_.set_pending(...)`. Functional state is "value at issue or later," not
  "pre-issue." Direction: confirm no consumer reads pre-issue regfile state, or
  document.

- **WB-N3 — `committed_entry()` cleared by panic flush before `record_cycle_trace` runs.**
  `sim/src/timing/timing_model.cpp:535-542,547`. `flush()` calls `reset()` which sets
  `committed_ = nullopt`. Trace path that reads it after flush sees the legitimate
  writeback erased. Direction: move `committed_` capture before the panic-flush block,
  or have `flush()` not clear `committed_`.

- **WB-N4 — `add_source(nullptr)` silent corruption.**
  `sim/src/timing/writeback_arbiter.cpp:8-10`. No null check.

### Panic Mechanism

- **PA-M3 — `panic_->evaluate()` reads `execution_units_drained()` from committed-state accessors before this cycle's commits.**
  `sim/src/timing/timing_model.cpp:353` followed by units' `commit()` at `:366-374`.
  Adds one cycle to every panic completion. Within 32-cycle bound, correctness-acceptable.
  Direction: move `panic_->evaluate()` to post-commit, or document the +1.

- **PA-M4 — Functional EBREAK execute path latches panic in parallel with timing decode path.**
  `sim/src/functional/functional_model.cpp:209-215`. Decode short-circuits EBREAK so the
  functional execute branch is never reached in the timing pipeline; only fires for
  direct functional-model unit tests. Two panic-latch paths can desync PANIC_PC/CAUSE.
  Direction: document or no-op the functional EBREAK branch.

- **F-14 — `PanicController::set_drained_query` callable claim is documentation-inaccurate.**
  `sim/include/gpu_sim/timing/panic_controller.h:31-34` claims callable "is expected to
  read only committed (current_*) state." Actual callable reads several `next_*`
  surfaces (`alu_->has_result()`, `mul_->has_result()`, etc.). Functionally OK because
  call site at `timing_model.cpp:353` runs BEFORE units evaluate, but documented contract
  is wrong. Direction: fix the doc comment, OR read committed state.

- **PA-N1 — `discard_writeback_results()` drains only one buffer per cycle; mul pipeline can't drain in 32 cycles.**
  `sim/src/timing/timing_model.cpp:299-311`. With `pipeline_stages` near
  MAX_DRAIN_CYCLES=32, a moderately-loaded mul plus LDST FIFO entries can hit timeout.
  Direction: emit `Stats::panic_drain_timeout` and `panic_in_flight_abandoned` counters.

- **PA-N2 — Drain timeout silently abandons in-flight ops with no counter / no log.**
  `sim/src/timing/panic_controller.cpp:38`. No distinction between clean drain and
  forced timeout. Direction: add `Stats::panic_drain_cycles` and `panic_drain_timed_out`;
  log on timeout.

- **PA-N4 — `panic_->evaluate()` reads stale (last-cycle committed) drained-query.**
  `sim/src/timing/timing_model.cpp:353-364`. Panic.evaluate runs at top of tick before
  unit evaluates; one extra panic cycle than necessary. Direction: move
  `panic_->evaluate()` after unit evaluates and `discard_writeback_results`, or split.

- **PA-N7 — No `wb_arbiter_->commit()` on panic ticks; `committed_entry()` stuck.** (Overlaps WB-N3.)
  Direction: call `wb_arbiter_->commit()` (with `pending_commit_=nullopt`) in panic
  commit phase.

- **PA-N8 — Functional `panicked_` flag has no public clear/reset method.**
  `sim/src/functional/functional_model.cpp:36-39,18-21`. Only `reset_warp_state()`
  clears it as side-effect. Direction: add `FunctionalModel::clear_panic()`; have
  `PanicController::reset()` call it.

- **PA-N9 — Panic-trigger does not check `warps_[panic_warp_].active`.**
  `sim/src/timing/timing_model.cpp:401-407`. PANIC_WARP could point at an inactive warp
  on a contrived race. Direction: assert `warps_[ebreak_req.warp_id].active`, or reject.

- **PA-N10 — `EBreakRequest` silently dropped if `panic_->is_active()`.**
  `sim/src/timing/timing_model.cpp:403`. No log/counter. Direction: assert the guard
  never fires, or count drops.

- **PA-N11 — Operand-collector "commandeer" semantics from spec §4.8.1 step 2 not modeled.**
  `sim/src/timing/panic_controller.cpp:27` reads functional regfile directly, bypassing
  `OperandCollector`. Direction: route the r31 read through opcoll on the trigger cycle,
  or update spec wording. **DECISION** (spec vs impl).

- **PA-N12 — `STATUS.DONE` and `STATUS.PANIC` host-visible bits not exposed.**
  `sim/include/gpu_sim/timing/panic_controller.h:18-19`. Only `is_active()` /
  `is_done()` exist. Direction: add `TimingModel::status_panic()` and `status_done()`
  accessors per spec §6.1, or document that the simulator skips CSR modeling.

### Functional Model · Decoder · ISA

- **FN-M2 — ECALL / EBREAK semantics in `FunctionalModel::execute` do not match the timing-model boundary.**
  `sim/src/functional/functional_model.cpp:202-215`. Spec §6.5 ECALL marked inactive at
  *dispatch*; §4.8 EBREAK detected at *decode* and panicked after r31 read. Functional
  model is the oracle, latching at "execute" is acceptable, but a divergence can arise
  if timing-model issue/dispatch order disagrees. Direction: document the contract or
  move side-effects to a method invoked at the modelled boundary.

- **FN-C2 — `branch_target` populated unconditionally for not-taken BRANCH.**
  (Audit "C" suffix is misleading; this is a trace JSON artifact, not Critical.)
  `sim/src/functional/functional_model.cpp:147`; JSON arg emission at
  `timing_model.cpp:961-963`. Direction: set only when `branch_taken == true`, or document.

- **FN-C3 — Validation error message lies about MAX_WARPS.**
  `sim/src/config.cpp:14-16` says "must be in [1, 32]" though `MAX_WARPS == 8`.
  Direction: format using actual `MAX_WARPS` constant.

- **FN-M3 — `init_kernel` doesn't reinit register banks of warps deactivated since last launch.**
  `sim/src/functional/functional_model.cpp:17-32`. Inactive warps retain stale values
  from prior kernel; `reset()` doesn't reinit `memory_` / `instr_mem_` / `lookup_table_`
  either. Spec §6.3 — "Registers r5–r31 are initialized to 0… r1–r4 of every thread in
  every active warp are preloaded." Direction: zero register banks of inactive warps in
  `init_kernel`, or clarify in spec.

- **FN-M4 — `panic_warp/cause/pc` accessors return defaults (all 0) when no panic occurred.**
  `sim/include/gpu_sim/functional/functional_model.h:31-34`. Spec §6.1 — undefined when
  STATUS.PANIC clear; cannot distinguish "no panic" from "warp 0 panicked at PC 0
  cause 0" (legitimate per spec line 297). Direction: gate on `is_panicked()`, return
  optional, or document.

### TimingModel Orchestration

- **TM-M1 — wb_arbiter not driven during panic; scoreboard busy bits leak.**
  `sim/src/timing/timing_model.cpp:347-382`. `discard_writeback_results()` consumes
  results but `wb_arbiter_->evaluate()` is not called, so scoreboard `clear_pending`
  never fires for in-flight ops. Mostly cosmetic (panic ends in all-warps-inactive).
  Direction: run arbiter during panic with flush-style suppression, or document.

- **TM-M2 — Cache evaluate is split across the tick (FILL/secondary at top, HIT mid-sweep).**
  `sim/src/timing/timing_model.cpp:409` (top), `:494` (coalescer triggers HIT). By
  design — encodes FILL > secondary > HIT priority. Direction: call out as documented
  invariant; no code change.

- **TM-N2 — ECALL deactivation coincident with armed panic produces ambiguous trace state.**
  `sim/src/timing/timing_model.cpp:266-270`. Snapshot shows warp `RETIRED` while panic
  is also active. Direction: when `pending_panic_flush_` is set, defer ECALL
  deactivation through the flush path.

- **TM-N3 — `pipeline_drained()` doesn't directly check `gather_file_` busy state.**
  `sim/src/timing/timing_model.cpp:276-287`. Relies on `cache_->is_idle()`; if a gather
  buffer has `busy=true` with `filled_count<WARP_SIZE`, simulator could declare
  completion with a partially-filled load. Direction: add `gather_file_->is_idle()` to
  drained predicates, or audit `cache_->is_idle()` to confirm coverage.

- **TM-N4 — Possible `accept()` / `evaluate()` race writing `current_busy_`.**
  `sim/src/timing/timing_model.cpp:249-274` vs `:484-488`. `dispatch_to_unit` calls
  `unit->accept(input, cycle_)` before units' own `evaluate()`; if any unit's `accept()`
  writes `current_*` rather than `next_*`, snapshot at line 577-580 could see this
  cycle's accept as already committed. Direction: verify each `accept()` writes only
  `next_*`; document any exception.

- **TM-N5 — `set_drained_query` lambda captures `[this]`; destructor order fragile.**
  `sim/src/timing/timing_model.cpp:201`. `panic_` destructs before `wb_arbiter_` etc.;
  if destructor invokes the callable, touches partially-destructed state. Direction:
  document the lifetime invariant or have `PanicController`'s destructor null the callback.

- **TM-N6 — Cycle counter `cycle_` is 1-based after first `tick()`; `finalize_trace` `end_cycle = cycle_ + 1`.**
  `sim/src/timing/timing_model.cpp:338-339,567-568`. Internally consistent; may surprise
  users. Direction: audit `total_cycles` semantics.

---

## P3 — Minor (doc drift, naming, defensive depth)

### Stale `current_port_claimed_` doc references (3-way merged)
- **Sources:** F-07 (findings, original) + F-12 (findings, supersedes F-07) + LS-m1 (audit)
- **Severity:** Definite (documentation drift)
- **Location:**
  - `sim/include/gpu_sim/timing/cache.h:175-176`
  - `resources/timing_discipline.md:184` (row 11)
  - `sim/include/gpu_sim/timing/load_gather_buffer.h:73-79`
- **Observation:** Three locations reference a "REGISTERED `next_port_claimed_` /
  `current_port_claimed_` pair" that no longer exists. Consolidation pass (commit
  54e6542) made it COMBINATIONAL same-cycle scratch (single slot, first-writer-wins,
  cleared at gather_file.commit). Behavior correct; doc is the bug.
- **Direction:** Update all three locations to "COMBINATIONAL same-tick scratch with
  end-of-cycle reset."

### Other minors

- **FE-m1.** `BRANCH` predicted_target set even when `predicted_taken=false`
  (`branch_predictor.cpp:14-15`). Consumers ignore it; trace events still carry it.
- **FE-m2.** `BRANCH` with `imm == 0` predicted not-taken (treated as forward); a
  self-loop branch would be predicted not-taken. Mostly cosmetic.
- **FE-m3.** `fetch_skip_all_full` covers the "all warps inactive" case too.
  `sim/src/timing/fetch_stage.cpp:82-85`. Direction: split into `fetch_skip_no_active`
  vs `fetch_skip_all_full`, or rename.
- **FE-m4.** `warp_cycles_active[w]` keeps incrementing during panic-active ticks.
  `sim/src/timing/timing_model.cpp:341-345` runs before `panic_->is_active()`
  early-return. Direction: skip during panic, or rename to `warp_cycles_resident`.
- **SCH-m1.** Branch-shadow stall under buffer-empty attributes to `buffer_empty`
  (eligibility checked first); post-redirect drain cycles recorded as buffer-empty
  rather than branch-shadow. Matches spec eligibility order.
- **SCH-m2.** `reads_rd` semantics depend on `has_rd` being true; future ISA additions
  could introduce `reads_rd && !has_rd`. Add an assert.
- **SCH-m3.** `INACTIVE` diagnostic clobbers ECALL-just-deactivated warp's outcome.
  `sim/src/timing/warp_scheduler.cpp:70`. Direction: add transient `JUST_RETIRED`.
- **SCH-m4.** `current_diagnostics()` is a pre-commit slot for orchestrator's mid-tick
  consumers. `sim/include/gpu_sim/timing/warp_scheduler.h:100-102`. Direction: document
  the contract.
- **SCH-m5.** `warp_stall_unit_busy[w]` conflates opcoll-busy and unit-busy.
  `sim/src/timing/warp_scheduler.cpp:118,124`. Direction: split into
  `warp_stall_opcoll_busy` vs `warp_stall_unit_busy`.
- **BST-m1.** `branch_tracker_.reset()` skips `seed_next()` invariant; works only
  because reset zeros both halves.
  `sim/include/gpu_sim/timing/branch_shadow_tracker.h:51-54` and
  `sim/src/timing/timing_model.cpp:540`. Direction: document or assert.
- **OC-m1.** `decode_stage` redirect-override discards `target_pc` (`decode_stage.h:64-70`);
  decode never reads it, but lacks symmetry with fetch.
- **EU-m1.** Multiply / Divide / TLookup decrement `cycles_remaining` on the same tick
  `accept` runs. Effective end-to-end matches spec, but the same-tick decrement is
  unstated in code. Direction: add a one-line comment in each `accept()`.
- **EU-m2.** No separate `multiply_stall_cycles` (head-blocked-by-result-buffer) or
  `divide_busy_cycles` distinct from `busy_cycles`. Spec doesn't mandate.
- **EU-m3.** TLookup is single-occupancy (busy 17 cycles), but spec §2.3 phrasing
  "pipelined" is ambiguous. Behavior matches "asserts busy for the full duration" but
  wording reads as cross-warp pipelining. Direction: tighten spec §2.3 wording.
- **LS-m2.** No parameterized test for non-power-of-2 `num_ldst_units` (verifying
  `ceil(32/N)` cycles).
- **LS-m3.** `CoalescingUnit::reset()` clears `processing_` / `serial_index_` but not
  `current_entry_` / `is_coalesced_`. Safe today (guarded by `processing_`), but stale
  `current_entry_.trace` lingers.
- **LS-m4.** `gather_buffer_port_conflict_cycles` counted only on HIT loss; FILL +
  secondary same-cycle collisions invisible. Cache retries the secondary next tick;
  observability-only.
- **LS-m5.** No `addr_gen_fifo_full_cycles` counter.
  `sim/src/timing/ldst_unit.cpp:32-41`. Backpressure stall invisible to perf reports.
- **LS-m6.** `LdStUnit::reset()` leaves `pending_entry_` payload stale.
  `sim/src/timing/ldst_unit.cpp:53-61`. Trace dumps after panic flush may leak previous
  warp PC.
- **LS-m7.** `gather_file_->flush()` races same-tick `wb_arbiter_` `committed_entry()`
  trace state. `sim/src/timing/load_gather_buffer.cpp:87-89` invoked at
  `sim/src/timing/timing_model.cpp:538`, after `wb_arbiter_->commit()` at 514.
- **LS-m8.** `consume_result()` RR pointer reset to 0 on flush.
  `sim/src/timing/load_gather_buffer.cpp:79-89`. After non-panic full-system reset,
  fairness across warps is lost on the first cycle.
- **CA-m1.** `pinned_line_addr` field name is ambiguous — it's the line *index*
  (post `addr/line_size`), not a byte address.
- **CA-m2.** Secondary chain identification is O(n²) per cycle at default 4 MSHRs (fine);
  revisit at MSHRs ≥ 16 (already noted as Phase-2 candidate in spec §5.3.1).
- **CA-m3.** `process_load` / `process_store` accept `lane_mask == 0` without validation.
  `sim/src/timing/cache.cpp:28-114`. Direction: assert `lane_mask != 0` at entry.
- **CA-m4.** `is_idle()` does not include `mem_if_->is_idle()`.
  `sim/src/timing/cache.cpp:436-440`. Verified safe — `pipeline_drained()` checks both —
  but the cache header could mislead a future caller. Direction: comment.
- **MEM-m1.** Phase accumulator uses `double`; long-run drift after ~2^53 fabric cycles.
  Practically irrelevant (~6 weeks at 150 MHz).
- **MEM-m2.** No per-write latency stat (`external_write_latency_*`); reads only.
- **MEM-m3.** `Config::validate()` does not enforce `dramsim3_request_fifo_depth ==
  minimum`; accepts larger, which is wasteful but correct.
- **MEM-m4.** `is_idle()` and `in_flight_count()` semantics vary slightly between
  backends; counts of submitted-but-not-yet-issued reads include the assembly slot.
  Verified consistent.
- **MEM-m5.** `dramsim3_output_dir` `create_directories` errors swallowed silently.
  `sim/src/timing/dramsim3_memory.cpp:46-47`. On failure DRAMSim3 falls back to cwd and
  emits a warning to stdout — the very pollution this pre-creation was meant to suppress.
- **MEM-m6.** `reset()`'s DRAMSim3 rebuild not wrapped in try/catch.
  `sim/src/timing/dramsim3_memory.cpp:166-179`. If the `.ini` was deleted between
  construction and reset, DRAMSim3 aborts fatally.
- **MEM-m7.** `FixedLatencyMemory::reset()` doesn't zero latency stats. Same in DRAMSim3.
- **MEM-m8.** Same-line back-to-back writes — first response artificially deferred until
  second completes. Spec §5.6 line 489 says "folded into a single response," implying
  merging not delaying.
- **MEM-m9.** `WillAcceptTransaction` head-of-line stall hides per-request latency growth.
  Direction: add `external_read_queue_wait_total` counter.
- **MEM-m10.** Response queue capacity bound's off-by-one allows transient overshoot.
  `sim/src/timing/dramsim3_memory.cpp:196,221`.
- **WB-m1.** Same-cycle set+clear of `next_[warp][reg]` from issue and writeback can
  in principle race, but spec §4.7's 1-cycle gap rule prevents the specific
  dependent-issue case from manifesting. Direction: document the order invariant in
  `scoreboard.h`.
- **PA-m1.** `trigger()` comment numbering vs. spec cycle enumeration is misaligned
  ("step 1 = cycle 2"); behavior matches spec, comment is confusing.
- **PA-m2.** `PANIC_LANE` host-visible accessor missing from `FunctionalModel`; lane is
  hardwired 0 but never returned.
- **PA-m3.** `PanicController::trigger()` does not check `is_active()` itself; relies on
  caller-side guard. Add early-return for defensive depth.
- **PA-m4.** Scheduler / opcoll / gather / wb_arbiter `flush()` cascade does NOT include
  `scoreboard_.reset()` — matches spec §6 (line 620). Tied to P1-7 / P1-8.
- **PA-N13.** Single-issue assumption in panic-trigger guard not enforced.
  `sim/src/timing/decode_stage.cpp:9-38`. Direction: add comment / static assertion
  documenting the Phase-1 invariant.
- **TM-m1.** `dispatch_to_unit` SYSTEM/ECALL deactivates the warp synchronously inside
  the evaluate sweep, not at commit — slight discipline blur, no current consumer broken.
- **TM-m2.** `record_cycle_trace` runs after commit using `current_*` accessors —
  verified correct; final-cycle trace is emitted before termination check.
- **FN-m1.** Unknown CSR addresses → INVALID is stricter than spec §6.4 (which lists
  three CSRs without explicitly forbidding others). Direction: document in spec.
- **FN-m2.** TLOOKUP decoder accepts all funct3 values; spec §2.3 doesn't restrict —
  verified-correct, flag for spec tightening if intent was funct3=0 only.
- **FN-m3.** ECALL/EBREAK in functional execute loops 32 lanes despite only lane 0 acting.
  `sim/src/functional/functional_model.cpp:202-215,220-224`. No double-latch guard on
  `latch_panic`. Direction: short-circuit out of the lane loop, or guard `latch_panic`.
- **FN-m4.** `rs2_val` / `rd_val` read every iteration regardless of decoder's
  `num_src_regs`. `sim/src/functional/functional_model.cpp:104-105`. Stale-register reads
  possible for I-type/LUI/AUIPC; results unused, harmless today.
- (note) Gather-buffer source reports `ExecUnit::LDST` (`load_gather_buffer.cpp:110`);
  cannot distinguish FILL completion from HIT completion in trace.

---

## P4 — Spec ambiguities (clarification needed; no code bug)

### F-04 / F-06. embedding_gather IPC stagnation under warp scaling
- **Severity:** Spec-Ambiguity (F-04 reclassified by F-06)
- **Evidence:** IPC across warp counts: 0.033 / 0.030 / 0.030. Total cycles scale
  linearly; `mshr_stall_cycles` scales linearly (4855 → 11303 → 23049). Trace audit
  confirmed `active_mshrs.value: avg=3.89, max=4` — pool saturated.
- **Spec section:** §4.3 (latency hiding via warps); §5.3.1 (MSHR file capacity).
- **Direction:** §4.3 should note that warp-count latency hiding is bounded by MSHR
  pool capacity for memory-bound workloads.

### F-17. Same-warp consecutive scheduler issues are common (legitimate per §4.3 RR)
- **Severity:** Spec-Ambiguity
- **Evidence:** Trace shows same-warp consecutive issues at 1-cycle granularity in all
  6 benchmarks; ranging 7% (fused_linear_activation) to 88% (matmul) of issue events.
  Happens when 7 of 8 warps are backend-stalled and round-robin lands on same warp on
  consecutive cycles.
- **Spec section:** §4.3 ("loose round-robin").
- **Direction:** Spec should explicitly allow same-warp consecutive issue when others
  are ineligible, or document the actual scan order more precisely.

### F-19. Spec §4.2 says JALR "always mispredicted"; impl conditionally mispredicts
- **Severity:** Spec-Ambiguity
- **Quoted text:** §4.2: "JALR: predicted as fall-through to PC+4… **always mispredicted**."
- **Implementation pinning:** `sim/src/timing/branch_predictor.cpp:24-26` leaves
  `predicted_taken = false`; `sim/src/timing/timing_model.cpp:322-335`
  `branch_mispredicted()` computes `predicted_next_pc = pc + 4`. If a JALR's actual
  target equals `pc + 4`, no mispredict counted, no flush.
- **Direction:** **DECISION.** Either replace "always mispredicted" with "mispredicted
  whenever the register-indirect target ≠ PC+4," OR change the impl to unconditionally
  treat JALR as mispredicted (`if (type == JALR) return true;` short-circuit).

---

## P5 — Observability gaps (no bug; instrumentation/CLI)

### F-01. Bench CLI does not expose MSHR-count or buffer-depth knobs
- **Severity:** Observability-Gap
- **Location:** `tests/matmul/matmul_bench.cpp:69-72` (and analogous in 5 other bench
  binaries) lists CLI options as `--num-warps`, `--memory-latency`, `--max-cycles`,
  `--memory-backend`, `--dramsim3-config-path`, `--json`. Nothing exposes
  `SimConfig::num_mshrs` or instruction-buffer depth. Standalone `runner/src/main.cpp`
  accepts `--config=<file.json>` for SimConfig overrides but requires explicit kernel
  ELFs and is capped at `--num-warps≤8`.
- **Direction:** Add `--num-mshrs=<N>` and `--inst-buffer-depth=<N>` to the bench-binary
  common options (or factor a shared option-parser helper).

---

## Closed observations / negative results (kept for audit trail)

- **AT_REST always has populated rest_reason** (Phase 3 trace audit). The hypothetical
  Phase-8 counter `rest_reason_unset_cycles` proposed in the plan is unnecessary;
  current `WarpRestReason` enum is exhaustive in practice.
- **branch_redirect events always preceded by an issue** for the same warp earlier in
  the trace (Phase 3 trace audit). The redirect-tracking path is well-formed.
- **F-04 reclassified to F-06 — not a bug** (memory-bound saturation explains
  embedding_gather IPC stagnation).
- **F-07 superseded by F-12, then merged with LS-m1** into the P3 doc-drift item above.

---

## Phase 6 — Synthetic kernel positive verifications (no findings)

| Kernel | Spec target | Result |
|--------|-------------|--------|
| `rr_tiebreak` | §4.2/§4.3 RR tie-break | PASS — per-warp instruction count imbalance = 0 at warps ∈ {2,4,8} |
| `line_boundary_load` | §5.2 boundary detection | PASS (counter behavior; surfaced F-18 / P1-4) |
| `mshr_same_line_race` | §5.3.1 same-line merging + chain order | PASS — 1 primary + 3 secondaries, allocations strictly ordered |
| `jalr_storm` | §4.2 static-mispredict + redirect | PASS (counter tally; surfaced F-19 / P4) |
| `divide_chain` | §4.6 32-cyc DIV + RV32M div-by-zero | PASS counter; surfaced F-20 / P1-5 trace slice mismatch |
| `panic_drain_test` | §4.8.1 drain bound + "in-flight" | PASS — panic span = 32, `panic_cause = 0x101`, DIV completes during drain |

Confirmed by these kernels:
- §4.2 mispredict-recovery: per-warp penalty ≈ 1 redirect-register cycle + 2 refill
  cycles, fully hidden at N≥4 warps.
- §5.2 single-line coalescing: aligned-base load → 1 coalesced cache request;
  aligned+64 base → 32 per-lane lookups.
- §5.3.1 chain ordering by allocation time: primary first, secondaries in arrival order.
- §4.6 single-occupancy DIV: 18 DIVs across 2 warps fully serialize, busy_cycles = exact.
- §4.7 scoreboard 1-cycle release gap: divide_chain implies release works.

---

## Cross-cutting themes

1. **Panic boundary discipline is the largest unresolved cluster.** P0-1, P0-2, P0-3
   plus P1-7, P1-8, P1-9, P1-10, P1-11 plus PA-M3, PA-M4, PA-N1..PA-N13. A focused
   refactor pass is warranted.
2. **Scoreboard-and-flush asymmetry.** P1-8 (= SCB-M2 / OC-N6 / WB-N1) all converge on
   the same fix: either reset the scoreboard alongside `branch_tracker_` in the panic
   cascade, or implement `TimingModel::reset()` (P1-7) and document panic→reset.
3. **DRAMSim3 contract gaps.** P0-10..P0-13 (silent failures, FIFO-order violation,
   reentrancy, MSHR-id reuse). The DRAMSim3 wrapper would benefit from a contract sweep.
4. **Predictor-update site is wrong on multiple axes.** OC-N1 (counts JAL/JALR), OC-N2
   (uses `pc+4` for actual_target), OC-N8 (runs before resolve). All silent today
   (static stub) but make `update()` plumbing unsafe to swap in stateful predictor.
5. **Counter-semantics drift.** P1-1, P1-2, P1-4, P1-5 plus OC-M1, WB-M2, EU2-n5,
   CA-C4 all distort the validation surface in Appendix B. A single counter-catalog
   review is overdue.
6. **Phase-7 / Phase-9 documentation drift.** Several REGISTERED-vs-COMBINATIONAL
   classifications in `timing_discipline.md` and per-file comments lag actual code.
   A single sweep through the discipline doc against current sources would close most.
7. **CTRL.RESET path is unimplemented end-to-end.** Tied to P1-7 and no CSR-block
   plumbing. Several "scoreboard abandoned" / "panic state cleared on reset" rules in
   spec rely on this path existing.

---

## Files referenced (consolidated)

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

## Provenance / source documents

- `project-plans/timing-bug-hunt-findings.md` — operational ledger from Phase 0–9 of the
  timing-bug-hunt plan (`project-plans/lively-conjuring-neumann.md`). Contains 20
  numbered F-NN findings, audit row verdicts, Phase 6 synthetic-kernel results, and a
  detailed session log. Session-log content is preserved here only as the cross-cutting
  themes section; the per-day commentary is left in the original ledger if you want it.
- `project-plans/spec-vs-sim-audit.md` — read-only spec-vs-impl audit run in two passes
  (date 2026-04-28). Pass 1 surfaced 4 critical / ~22 moderate / ~17 minor; Pass 2
  added 3 critical / ~25 moderate / ~20 minor. Original triage tables remain in the
  source file; this consolidation reproduces the items themselves under unified priority
  tiers.
