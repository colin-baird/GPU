# Phase 10 — Memory Subsystem Pipeline Discipline

## Context

The simulator's cross-stage signaling refactor (the broader Phase 10) eliminates COMBINATIONAL forward data edges throughout the timing model: every forward data edge becomes REGISTERED, only backward control may be combinational, and the evaluate sweep reverses to back-to-front. This plan covers the **memory subsystem half of that refactor** as an independently executable workstream. The issue/execute path, branch resolution, sweep reversal, and combinational-backward redirect are owned by `/workspace/project-plans/phase-10-pipeline-discipline.md`.

The memory subsystem audit found three forward-data violations that need conversion (LdSt→Coalescing FIFO, Coalescing→Cache process_load/store, Coalescing→Gather claim), one writeback-arbiter source (Gather→WritebackArbiter result-ready predicate) that aligns with the issue-execute plan's 10B.1 pattern, and one cache↔mem_if boundary that stays as a documented trusted-subsystem carve-out. Cache internals (tags, MSHRs, write_buffer) and the gather-buffer write-port arbitration are correct as-is and remain direct mutations — both are documented internal carve-outs (per discipline doc rows 10 and 11).

The cache↔mem_if carve-out documentation (Phase M5 below) is **load-bearing for the broader Phase 10D sweep reversal** in the issue-execute plan; landing memory first satisfies that prerequisite automatically.

## Sequencing

This plan executes **first and end-to-end**, before the issue-execute plan starts. The issue-execute plan picks up from a baseline where:

- All memory forward-data edges (LdSt FIFO, Coalescing.claim, Coalescing→Cache command, Gather→Arbiter result-ready) are REGISTERED.
- The cache↔mem_if pair is formally documented as an ordering unit (Phase M5).
- Memory unit tests and end-to-end tests are calibrated to the post-memory cycle counts.
- Issue/execute path remains COMBINATIONAL forward (unchanged from pre-Phase-10).

Intermediate state during this plan: the pipeline has a "fast frontend (1 cycle scheduler→units), slow memory tail (4-5 cycles ldst→coalescing→cache→gather→wb_arbiter)" shape. Functionally correct but lopsided. This persists until the issue-execute plan converts the frontend.

Shared files with the issue-execute plan (each plan touches different lines; no merge conflict expected):

- `/workspace/sim/src/timing/writeback_arbiter.cpp` — this plan flips the gather source's `next_has_result()` read to `current_has_result()` in Phase M4. Issue-execute plan later does the same for ALU/MUL/DIV/TLOOKUP. Source-iteration loop is touched by both, on different source-pointer entries.
- `/workspace/sim/src/timing/timing_model.cpp` — this plan updates the gather/ldst/cache sites in `pipeline_drained()`, `execution_units_drained()`, `discard_writeback_results()`, and `build_cycle_snapshot()`. Issue-execute plan updates the ALU/MUL/DIV/TLOOKUP sites in the same functions later.
- `/workspace/resources/timing_discipline.md` Phasing section — this plan adds the Phase M-series rows; issue-execute plan extends with Phase 10A-G rows later.

## Phase plan

The memory refactor is sequenced so each phase builds successfully and runs the full regression suite. Every phase has a per-benchmark cycle delta captured by `python3 tools/bench_compare.py --baseline <phase-start-ref>`.

### Phase M1 — LdSt addr_gen FIFO → Coalescing becomes REGISTERED

Convert the FIFO from a live `next_addr_gen_fifo_` read by coalescing same-tick into a single-FIFO design where each side defers its mutation to its own `commit()`. No double-buffer mirror — the FIFO is one field, mutated only at commit phase, and reads during evaluate see the start-of-cycle state. Producer (`LdStUnit`) owns the push intent and applies it at `LdStUnit::commit`; consumer (`CoalescingUnit`) owns the pop intent and applies it at `CoalescingUnit::commit`. This matches an existing precedent in the codebase: `DecodeStage::commit()` pushes into `warps_[w].instr_buffer` for the scheduler to read next cycle (`decode_stage.cpp:62-67`).

Files:
- `/workspace/sim/include/gpu_sim/timing/ldst_unit.h`, `/workspace/sim/src/timing/ldst_unit.cpp` — keep the FIFO as a single `std::deque<AddrGenFIFOEntry> addr_gen_fifo_` (drop the `next_` prefix on the field; it's no longer cross-stage-mutable mid-evaluate). Add `std::optional<AddrGenFIFOEntry> next_push_` slot. `evaluate()` writes `next_push_` instead of pushing directly. `commit()` applies the push: `if (next_push_) addr_gen_fifo_.push_back(*next_push_); next_push_.reset();`. Replace `next_fifo_empty()` / `next_fifo_front()` / `next_fifo_entries()` accessors with `current_fifo_empty()` / `current_fifo_front()` / `current_fifo_entries()` (returning the stable single-field state — semantically REGISTERED). Add a public `pop_front()` method called by `CoalescingUnit::commit()`.
- `/workspace/sim/src/timing/coalescing_unit.cpp`, `.h` — `evaluate()` reads `ldst_.current_fifo_empty()` / `current_fifo_front()` and sets a private `next_pop_ = true` flag instead of calling `ldst_.fifo_pop()` mid-evaluate. `commit()` applies the pop: `if (next_pop_ && !ldst_.current_fifo_empty()) { ldst_.pop_front(); } next_pop_ = false;`. The `current_fifo_empty()` defensive check at commit should never fire in correct code (front is stable from evaluate to commit; ldst's commit only adds to back, never removes from front) — keep it as belt-and-suspenders.
- `/workspace/sim/src/timing/timing_model.cpp` — update `build_cycle_snapshot()` (line 583, 689) to read `ldst_->current_fifo_entries()`.
- Also expose `current_fifo_size() const { return addr_gen_fifo_.size(); }` (or equivalent) — the pipeline plan's Phase 10B.0 reads this from the scheduler for LDST FIFO accounting in the issue gate. Cheap to add now alongside the other accessors; saves a back-edit at 10B.0 land time.
- Add a one-line comment at both `ldst_unit.cpp` (in `evaluate()` near the push-eligibility check) and `coalescing_unit.cpp` (in `commit()` near the pop) noting the structural parity with the frontend `instr_buffer`: producer decides push from registered fullness, consumer decides pop, end-of-cycle state matches a hypothetical mid-evaluate model. Reference `fetch_stage.cpp` `will_be_full` check.

Atomicity: **single atomic commit.** Producer-side push deferral, consumer-side pop deferral, and accessor renames must land together.

Behavior to call out (not a regression — matches existing frontend pattern): when the FIFO is full at start of cycle N and coalescing pops this cycle, `LdStUnit::evaluate()` checks `addr_gen_fifo_.size()` against capacity (full) and skips the push. Even though coalescing.commit will free a slot at end of N, ldst missed its window and retries on cycle N+1. This is a one-cycle bubble identical to the bubble in `fetch_stage.cpp` (where fetch's `will_be_full` check doesn't account for scheduler's same-cycle pop of `instr_buffer`). Accepting M1's bubble is parity with that pattern; eliminating it would require the consumer to expose pop intent as a backward-combinational signal the producer reads during its own evaluate — out of scope for Phase 10.

Commit-phase ordering: `LdStUnit::commit()` runs before `CoalescingUnit::commit()` in current `TimingModel::tick()` order (`timing_model.cpp:512-513`), but the order is irrelevant for correctness since the push touches the back and the pop touches the front of the deque (different ends). End-of-cycle state is invariant under either commit order.

Verification: regression green; cycle counts regress by ~1 cycle for every load/store; capture per-benchmark delta with `bench_compare.py`.

### Phase M2 — Coalescing.claim → LoadGatherBufferFile becomes REGISTERED

Convert the synchronous `claim()` mutation into a REGISTERED claim-request slot. Gather's evaluate applies the claim from `current_claim_request_` on the next cycle.

Files:
- `/workspace/sim/include/gpu_sim/timing/load_gather_buffer.h`, `/workspace/sim/src/timing/load_gather_buffer.cpp` — add `ClaimRequest` struct (warp_id, dest_reg, pc, issue_cycle, raw_instruction, valid bool) with `current_claim_request_` and `next_claim_request_` slots. `claim()` writes only `next_claim_request_`. `LoadGatherBufferFile::evaluate()` consumes `current_claim_request_` and applies the buffer mutation (sets buf.busy, buf.dest_reg, etc.). `commit()` flips next→current.
- `/workspace/sim/src/timing/coalescing_unit.cpp` — no API change; `gather_file_.claim(...)` still works the same way externally.

Atomicity: single commit.

Verification: regression green; cycle counts regress by ~1 cycle for every load (claim now takes one cycle to apply); capture per-benchmark delta.

### Phase M3 — Coalescing → Cache process_load/store becomes REGISTERED with combinational backward stall

Convert the synchronous `cache_.process_load/process_store` calls into a REGISTERED forward cmd path plus a COMBINATIONAL backward stall signal. Coalescing writes `next_load_cmd_`/`next_store_cmd_` at cycle N; cache reads `current_load_cmd_`/`current_store_cmd_` at cycle N+1 and processes. The backward stall signal `cache.next_cmd_stall()` is computed at end of cache.evaluate at cycle N from cache + mem_if's just-finished state, and coalescing reads it same-cycle (combinational backward) before deciding whether to submit. When cache asserts not-stalled at cycle N, the cmd it receives at cycle N+1 is guaranteed acceptable — no response slot, no retry, no rejection. Throughput is 1 cmd/cycle when not stalled.

This requires the back-to-front sweep order to put the cache↔mem_if ordering unit **before** coalescing — see the pipeline plan's Phase 10D for the corrected sweep order.

Files:
- `/workspace/sim/include/gpu_sim/timing/cache.h`, `/workspace/sim/src/timing/cache.cpp` — add `LoadCommand`, `StoreCommand` structs. Add `current_load_cmd_`/`next_load_cmd_` and `current_store_cmd_`/`next_store_cmd_` slots. Add setter methods `set_next_load_cmd(...)`, `set_next_store_cmd(...)` (called from coalescing). Move the body of `process_load`/`process_store` into `L1Cache::evaluate()` — after `handle_responses` and `drain_secondary_chain_head`, read `current_load_cmd_` / `current_store_cmd_` and dispatch to the existing logic. The old bool return goes away; cmd is guaranteed acceptable when received. Add `bool next_cmd_stall() const` accessor that returns true when any of these conditions hold from end-of-cycle state: MSHR free count is below the cmd-acceptance threshold, write buffer is at capacity, target line is pinned with conflicting tag (conservatively: any pinned tag), or a FILL/secondary will land at N+1 and steal the gather port (function of `pending_fill_` and `mem_if.responses_` / pending in-flight at end of N).
- `/workspace/sim/src/timing/coalescing_unit.cpp`, `.h` — replace direct `cache_.process_load(...)` / `process_store(...)` calls with `cache_.set_next_load_cmd(...)` / `set_next_store_cmd(...)`. Gate submission with `if (cache_.next_cmd_stall()) hold; else submit`. Hold semantics: keep the buffered cmd in coalescing's `current_entry_` and `processing_` state; do not pop the LdSt FIFO again until the cmd is submitted. No `pending_cmd_` flag, no rejection-retry — the stall signal subsumes both.

Atomicity: single commit. Cache cmd-slot addition, cache evaluate-time cmd processing, cache stall accessor, and coalescing's submission/hold logic must flip together.

Cycle behavior:
- Cycle N (back-to-front sweep): cache.evaluate runs first (in the memory ordering unit), processes any current cmd from N-1, computes `next_cmd_stall` from end-of-cycle state. Coalescing.evaluate runs later in the sweep, reads `cache.next_cmd_stall()`. If clear, writes `next_load_cmd_`. If set, holds.
- Cycle N+1: cache.evaluate reads `current_load_cmd_`, processes. Guaranteed acceptable.

Throughput: 1 cmd/cycle when not stalled. Stall asserts under real cache pressure (MSHRs near full, write-buffer full, pin conflict, predicted gather-port conflict). HIT cmds wait for free cycles instead of being rejected and retried — same effective throughput as today's port-arbitration retry, just expressed as upstream stall instead of downstream rejection.

Verification: regression green; per-benchmark cycle regression depends on workload (memory-bound workloads see more stall cycles, ALU-bound see fewer); capture deltas. The DRAMSim3 backend must also be tested (`-DGPU_SIM_USE_DRAMSIM3=ON`) — the stall signal must account for DRAMSim3's bounded write-region request FIFO too.

### Phase M4 — Gather → WritebackArbiter result-ready becomes REGISTERED

Coordinated with the issue-execute plan's Phase 10B.1 pattern.

Files:
- `/workspace/sim/include/gpu_sim/timing/load_gather_buffer.h`, `/workspace/sim/src/timing/load_gather_buffer.cpp` — add `bool current_has_result_` flag. Update `try_write` so when `filled_count` reaches `WARP_SIZE`, the resulting full-buffer state is reflected in `next_has_result_` (a corresponding next-side mirror). `commit()` flips `current_has_result_ = next_has_result_`. Rename `next_has_result()` → `current_has_result()` returning `current_has_result_`. `consume_result()` continues to release the buffer (writes `next_busy=false`, etc.); arbiter sees `current_has_result_=false` next cycle.
- `/workspace/sim/src/timing/writeback_arbiter.cpp` — for the gather source, read `current_has_result()` instead of `next_has_result()`. The arbiter is still pre-10B.0 (round-robin priority) at this point in the timeline; M4 does not restructure the arbiter, only flips the gather accessor.
- `/workspace/sim/src/timing/timing_model.cpp` — update memory-source reads in `pipeline_drained()`, `execution_units_drained()`, `discard_writeback_results()`, `build_cycle_snapshot()` to use `current_has_result()` for the gather source.

Atomicity: single commit (producer flip, arbiter flip, drain/snapshot flips).

Verification: regression green; cycle counts may shift slightly (gather writeback arbitration delayed by one cycle); capture delta.

**Forward-compatibility note for the pipeline plan's 10B.0:** M4 establishes the gather buffer's writeback contract as a REGISTERED `current_has_result()` predicate plus `consume_result()` that releases the buffer. This contract is preserved verbatim under 10B.0's arbiter restructure — the arbiter migrates from round-robin priority to bitmap-driven (fixed-latency claim takes the slot) plus opportunistic gather consumption (consume from gather when the bitmap slot for this cycle is unclaimed). The gather source itself does **not** participate in the writeback bitmap (loads are variable-latency; bitmap reservations only exist for fixed-latency ops). Under 10B.0:

- Cycle where bitmap is claimed by ALU/MUL/DIV/TLOOKUP: arbiter consumes from that unit; gather waits even if `current_has_result()` is true. This case increments `load_writeback_stall_cycles` (10B.0 counter, not introduced here).
- Cycle where bitmap is unclaimed: arbiter checks `gather_->current_has_result()`; if true, consumes from gather (round-robin within gather's per-warp buffers via existing `consume_result` logic).

No additional gather-side changes are required at 10B.0 land time — M4's accessor surface is sufficient. If the round-robin gather-buffer selection inside `consume_result()` ever needs to become fairness-aware (e.g., prefer the warp whose load has been waiting longest), that's a separate optimization beyond Phase 10's scope.

### Phase M5 — Cache ↔ mem_if becomes REGISTERED forward + combinational backward stall

The earlier draft of this phase framed the cache↔mem_if boundary as a "trusted-subsystem carve-out" and proposed leaving it as same-cycle submit/decrement. That framing doesn't survive Principle 6 — it's the same anti-pattern (forward edge with no backward stall wire, paired with a forward-direction acceptance bool) that M3 just fixed for the coalescing↔cache boundary. DRAMSim3's bounded write-region request FIFO is exactly the kind of state a combinational backward stall is designed to expose.

Convert the boundary in the same shape as M3:

- `mem_if::next_request_stall()` — combinational backward stall, computed from registered queue depth (and DRAMSim3's write-region FIFO state for that backend). Cache reads it before deciding to submit.
- `mem_if::next_read_request_` / `next_write_request_` slots for REGISTERED forward request submission. Cache writes when stall is clear; mem_if reads `current_*_request_` next cycle, admits to `in_flight_`, and decrements that cycle.
- `mem_if::next_has_response()` → rename to `current_has_response()`. The current name is misleading: `cache.handle_responses` runs at the top of cache.evaluate, before mem_if.evaluate produces this cycle's responses, so the read is already against committed state from end of last cycle. Naming fix only — no semantic change.
- The `submit_read` / `submit_write` bool returns go away. With the stall signal gating submission, acceptance is guaranteed.

Latency math is preserved end-to-end: today's submit-then-decrement-same-cycle becomes write-next-then-admit-and-decrement-next-cycle. Cache's externally-observable response timing is unchanged.

Files:
- `/workspace/sim/include/gpu_sim/timing/memory_interface.h` — add `next_read_request_` / `next_write_request_` REGISTERED slots, `set_next_read_request(...)` / `set_next_write_request(...)` setters, `next_request_stall()` combinational backward accessor. Rename `next_has_response()` → `current_has_response()`. Drop `bool` returns from submit setters.
- `/workspace/sim/src/timing/memory_interface.cpp`, `dramsim3_memory.cpp` — implement `next_request_stall()` for both backends. `FixedLatencyMemory` stalls when `in_flight_` is at some configured threshold (or never, if unbounded). `DRAMSim3Memory` stalls when the write-region FIFO is at capacity (the existing `submit_write` rejection condition becomes a stall query). At top of `mem_if.evaluate`, drain `current_read_request_` and `current_write_request_` into `in_flight_`.
- `/workspace/sim/src/timing/cache.cpp` — replace `mem_if_.submit_read(line_addr, mshr_id)` calls (lines 110, 188) with `mem_if_.set_next_read_request(line_addr, mshr_id)`, gated on `!mem_if_.next_request_stall()` (combinational backward read). Same for `submit_write` at line 386. Cache's MSHR allocation logic must check stall before allocating; if stalled, defer the allocation (or hold pin/MSHR via the existing `pending_fill_` carrier shape — extend if needed). `handle_responses` reads `current_has_response()` (the rename) instead of `next_has_response()`.
- `/workspace/sim/src/timing/timing_model.cpp` — sweep ordering: cache.evaluate before mem_if.evaluate stays the same (forward sweep order within the memory ordering unit; back-to-front of the outer pipeline puts the whole unit before coalescing per 10D). The combinational backward stall is read by cache (consumer) from mem_if (producer); mem_if computes stall from its own registered state, so it's stable across the whole evaluate phase.
- `/workspace/resources/timing_discipline.md` — rewrite row 15 to reflect REGISTERED forward + combinational backward stall classification; Phase M5 entry in the Phasing section captures the cycle delta.

Atomicity: single commit. Cache submit-call rewrites, mem_if slot/stall additions, accessor renames, both backends' stall implementations, and tick comment updates land together.

Throughput: 1 request/cycle when not stalled — same as today's effective throughput for FixedLatencyMemory, and now correctly modeled for DRAMSim3 (the backend-specific FIFO-full backpressure becomes part of the discipline rather than a special-case bool return).

Verification: regression green; capture per-benchmark cycle delta with `bench_compare.py`. The delta should be small in most cases (memory-bound workloads with deep MSHR pressure may see a modest shift from the rephrased submit-accept timing). Build with `-DGPU_SIM_USE_DRAMSIM3=ON` and verify `test_dramsim3_memory.cpp` passes — particularly the write-buffer-full saturation loop, which exercises the new stall signal directly.

### Phase M6 — Test cycle-count recalibration

Test recalibration after Phases M1-M4 land. Memory-specific tests get precise recalibration; cross-cutting tests get defensive bumps to survive the post-memory state until the issue-execute plan does final precise tuning in its Phase 10G.

**Memory-specific tests (precise recalibration to new values):**

- `test_cache.cpp`, `test_cache_mshr_merging.cpp`, `test_load_gather_buffer.cpp`, `test_dramsim3_memory.cpp`, `test_coalescing.cpp`. Each hard-coded cycle assertion gets re-derived from the post-refactor binary; the new value lands with a brief comment noting what changed (e.g., "+2 cycles per cache request from M3 REGISTERED command/response").
- Migrate `next_has_result` / `next_fifo_empty` / `next_fifo_front` reads in these test files to `current_*` equivalents (~13 occurrences in `test_load_gather_buffer.cpp`, ~7 in `test_cache.cpp`).
- Run with `-DGPU_SIM_USE_DRAMSIM3=ON` and verify `test_dramsim3_memory.cpp` passes — particularly the write-buffer-full saturation loop, which is sensitive to the cache↔mem_if ordering unit.

**Cross-cutting tests (defensive bumps; final tuning in issue-execute plan's 10G):**

- `test_integration.cpp` — bounds like `cycles < 200`, `cycles < 1000` may trip after M3's +2-cycle handshake on memory-bound workloads. Bump tight upper bounds to survive; do not tighten lower bounds. Add a comment that the issue-execute plan's 10G will replace these with precise post-Phase-10 values.
- `test_panic.cpp` — `REQUIRE(timing.cycle_count() < 1000)` checks. Bump to a generous ceiling (e.g., `< 2000`) if any benchmark trips it.
- Workload benchmarks (`bash ./tests/run_workload_benchmarks.sh --build-dir build`) — audit each benchmark's `max_cycles` budget; bump generously where memory-bound workloads now exceed the prior budget. Issue-execute plan's 10G will reset these to precise post-Phase-10 budgets.

The intent of the defensive bumps is to keep CI green throughout the issue-execute plan's work without forcing the issue-execute plan to update memory cycle counts twice. Final precise calibration of cross-cutting tests is owned by the issue-execute plan's 10G.

## Memory subsystem audit (M1-M19)

Disposition column: **Convert** = becomes REGISTERED in this plan; **Reframe** = not really a cross-stage edge under closer inspection (internal subsystem mutation); **Carve-out** = stays COMBINATIONAL backward or backward-only by deliberate design with strengthened call-site comment in M5; **Compliant** = already correct.

| ID | Producer → Consumer | Signal | Today | Disposition | Notes |
|----|--------------------|--------|-------|-------------|-------|
| M1 | `LdStUnit::next_addr_gen_fifo_` → `CoalescingUnit::evaluate` | FIFO entry payload | COMBINATIONAL forward | **Convert** (Phase M1) | Mechanical double-buffering of the deque; `fifo_pop()` semantics shift to write `next_` only |
| M2 | `CoalescingUnit::process_load/store call` → `L1Cache` | Load/store request, accept bool | COMBINATIONAL forward + back-pressure return | **Convert** (Phase M3) | REGISTERED command/response slots; +2 cycles per first-attempt request, +0 for retries |
| M3 | `CoalescingUnit::claim()` → `LoadGatherBufferFile` | Gather buffer claim metadata | COMBINATIONAL forward | **Convert** (Phase M2) | REGISTERED claim-request slot; gather.evaluate applies on next cycle |
| M4 | `L1Cache::next_stalled()` → `CoalescingUnit::evaluate` | Cache stall back-pressure | COMBINATIONAL backward | **Carve-out** (Phase M5) | Back-pressure may stay COMBINATIONAL; signal is combinationally driven from registered tag/MSHR/write-buffer state |
| M5 | `LoadGatherBufferFile::current_busy(w)` → `CoalescingUnit::evaluate` | Gather busy back-pressure | REGISTERED back-pressure | **Compliant** | Reads committed busy bit |
| M6/M7/M8 | `L1Cache::try_write(...)` → `LoadGatherBufferFile` (FILL, secondary, HIT paths) | Lane values + port arbitration | COMBINATIONAL with first-writer-wins port flag | **Reframe** + (M18 part) **Convert** (Phase M4) | Gather is the cache's result_buffer for LDST; data write is internal cache writeback (analogous to `alu->next_result_buffer_`). The "result valid" predicate (M18) becomes REGISTERED in Phase M4. The intra-cycle `next_port_claimed_` arbitration (M16) stays Phase-7 internal. |
| M9 | `L1Cache::next_last_miss_event_` → `record_cycle_trace` | Trace event | REGISTERED | **Compliant** | Phase 9 |
| M10 | `L1Cache::next_last_fill/drain_event_` → `record_cycle_trace` | Trace event | REGISTERED | **Compliant** | Phase 9 |
| M11 | `L1Cache::current_pending_fill_` carrier | Multi-cycle deferred fill | REGISTERED | **Compliant** | Phase 9 |
| M12 | `L1Cache::set_next_read_request(...)` → `mem_if.current_read_request_` (cycle N+1 admit to `in_flight_`) | Read request | REGISTERED forward via REGISTERED slot | **Convert** (Phase M5) | Cache miss path stages via setter; mem_if commit flips next→current; mem_if.evaluate drains to in_flight |
| M13 | `mem_if.current_has_response()` / `get_response()` → `L1Cache::handle_responses` | Memory response | REGISTERED (read combinationally one cycle late) | **Compliant** (Phase M5) | Renamed from `next_has_response`; cache.handle_responses runs at top of cache.evaluate (before mem_if.evaluate produces this cycle's responses), so the read sees committed end-of-previous-cycle state |
| M14 | `L1Cache::set_next_write_request(...)` → `mem_if.current_write_request_`; `next_request_stall()` gates the stage | Write request + back-pressure | REGISTERED forward + COMBINATIONAL backward stall | **Convert** (Phase M5) | drain_write_buffer pops only when `!next_request_stall()`; replaces the old bool-return retry loop |
| M15 | `L1Cache` internal `tags_`/`mshrs_`/`write_buffer_` | Cache hardware state | Direct mutation | **Compliant by design** | No cross-stage observers; tests assert synchronously; row 10 carve-out |
| M16 | `LoadGatherBufferFile::next_port_claimed_` | Intra-cycle write-port arbitration | COMBINATIONAL | **Compliant** | Phase 7; internal to gather; first-writer-wins, cleared at commit |
| M17 | `mem_if.next_request_stall()` (replaces the prior `submit_read/write` bool-return) | Back-pressure | COMBINATIONAL backward stall | **Compliant** (Phase M5) | drain_write_buffer reads `next_request_stall()` before staging; the synchronous submit_read/write APIs (now test-direct only) retain their bool returns for backend-isolation tests |
| M18 | `LoadGatherBufferFile::next_has_result()` → `WritebackArbiter::evaluate` | Result-ready predicate | COMBINATIONAL forward | **Convert** (Phase M4) | Add `current_has_result_` flag latched at commit when `filled_count == WARP_SIZE` |
| M19 | `LoadGatherBufferFile::consume_result()` → `WritebackArbiter` | WritebackEntry payload | REGISTERED | **Compliant** | Same protocol as other units |

Net: 6 forward-data violations get converted (M1, M2, M3, M12, M14, M18, plus the reframing of M6/M7/M8 around M18); the cache↔mem_if read-response read (M13) is a name fix only; 9 edges are already compliant or are pre-existing carve-outs.

## Critical files

Source:
- `/workspace/sim/include/gpu_sim/timing/ldst_unit.h`, `/workspace/sim/src/timing/ldst_unit.cpp` (Phase M1)
- `/workspace/sim/include/gpu_sim/timing/coalescing_unit.h`, `/workspace/sim/src/timing/coalescing_unit.cpp` (Phases M1, M2, M3)
- `/workspace/sim/include/gpu_sim/timing/load_gather_buffer.h`, `/workspace/sim/src/timing/load_gather_buffer.cpp` (Phases M2, M4)
- `/workspace/sim/include/gpu_sim/timing/cache.h`, `/workspace/sim/src/timing/cache.cpp` (Phases M3, M5)
- `/workspace/sim/src/timing/writeback_arbiter.cpp` (Phase M4 — coordinate with issue-execute plan)
- `/workspace/sim/src/timing/timing_model.cpp` (Phases M1, M3, M4, M5 — coordinate with issue-execute plan)
- `/workspace/sim/include/gpu_sim/timing/memory_interface.h`, `/workspace/sim/src/timing/memory_interface.cpp` (Phase M5 — comments only)
- `/workspace/sim/src/timing/dramsim3_memory.cpp` (Phase M5 — comments only)

Tests:
- `/workspace/sim/tests/test_cache.cpp` (Phase M6)
- `/workspace/sim/tests/test_cache_mshr_merging.cpp` (Phase M6)
- `/workspace/sim/tests/test_load_gather_buffer.cpp` (Phase M6)
- `/workspace/sim/tests/test_dramsim3_memory.cpp` (Phase M6)
- `/workspace/sim/tests/test_coalescing.cpp` (Phase M6)

Documentation:
- `/workspace/resources/timing_discipline.md` (Phase M5 — row 15 carve-out formalization, Phasing section addition)
- `/workspace/resources/perf_sim_arch.md` (Phase M5 — note ordering-unit constraint on cache↔mem_if)

## Reused functions and patterns

- `Scoreboard` (`/workspace/sim/include/gpu_sim/timing/scoreboard.h`) — canonical REGISTERED reference. The new double-buffered FIFO (Phase M1), claim slot (Phase M2), command/response slots (Phase M3), and `current_has_result_` flag (Phase M4) all follow this shape verbatim.
- The existing `current_pending_fill_` REGISTERED carrier in `L1Cache` (cache.cpp:265-291, 405) is the precedent for a multi-cycle REGISTERED slot inside cache; the new command-response slots in Phase M3 reuse this idiom.
- `python3 tools/bench_compare.py --baseline <git-ref>` — A/B benchmark snapshot at every phase boundary that changes cycle counts (M1, M2, M3, M4).
- `bash ./tests/run_workload_benchmarks.sh --build-dir build` — canonical workload-benchmark entry point.

## Verification

Per phase:

1. `cmake --build build -j8` succeeds.
2. `cd build && ctest --output-on-failure` passes (memory tests in particular: `test_cache`, `test_load_gather_buffer`, `test_cache_mshr_merging`, `test_dramsim3_memory`, `test_coalescing`).
3. For phases that intentionally shift cycle counts (M1, M2, M3, M4, M5): capture per-benchmark delta from `bench_compare.py`, record in commit message and in the `timing_discipline.md` Phase 10 summary.
5. Build with `-DGPU_SIM_USE_DRAMSIM3=ON` and rerun `test_dramsim3_memory.cpp` after Phases M3 and M5 — the cache↔mem_if boundary is most sensitive to changes here.
6. `python3 tools/render_signal_diagram.py --validate` passes after Phase M4 (AST and markdown extractors agree on every memory edge).

End-to-end:

- Run the workload benchmarks (`bash ./tests/run_workload_benchmarks.sh --build-dir build`) at end of Phase M6 and confirm all six benchmarks complete within their `max_cycles` budget.
- Generate the signal diagram and visually inspect the memory cluster: `LdStUnit→CoalescingUnit`, `CoalescingUnit→L1Cache`, and `CoalescingUnit→LoadGatherBufferFile` should all render with the REGISTERED line style; `L1Cache↔ExternalMemoryInterface` and `L1Cache→CoalescingUnit` (stall) should render as the documented carve-outs.
