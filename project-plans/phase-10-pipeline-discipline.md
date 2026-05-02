# Phase 10 — Synchronous Pipeline Discipline (issue / execute / branch / sweep)

## Context

The simulator's cross-stage signaling discipline (Phases 1-9 in `/workspace/resources/timing_discipline.md`) shipped two flavors of forward data edge: **REGISTERED** (consumer reads producer's `current_*` next cycle, +1 cycle of pipeline depth) and **COMBINATIONAL** (consumer reads producer's `next_*` same tick, zero handoff). The brief that prompted this plan (independently verified against the code in the prior conversation) identified that **COMBINATIONAL forward is being used pervasively in the issue→execute→writeback path** — `WarpScheduler::output() → OperandCollector::accept() → opcoll.evaluate → dispatch_to_unit → unit.accept → unit.evaluate → wb_arbiter.evaluate` collapses what should be 4-5 cycles into a single tick. To slow branch resolution to a sane delay, Phase 5 grafted a REGISTERED `RedirectRequest` slot onto `OperandCollector`; the slot exists only because the surrounding combinational forward edges resolve the branch too early.

The fix is the standard synchronous-pipeline discipline:

1. **Forward data edges are REGISTERED — always.** Each stage takes exactly one cycle.
2. **Backward control edges (stalls, branch redirect) are COMBINATIONAL backward.** Producer asserts mid-cycle; consumer (earlier in the pipeline) reads same cycle.
3. **The evaluate sweep reverses to back-to-front** so backward producers run before their consumers.
4. **Branch redirect becomes a combinational-backward wire from `ALUUnit`** read same-cycle by `FetchStage`/`DecodeStage`/`WarpScheduler`. The latched redirect slot on `OperandCollector` is deleted.

This is a deliberate fidelity gain; per-benchmark cycle counts will regress and the regressions are documented as expected.

**Scope:** this plan covers the issue/execute path, branch resolution, the evaluate sweep reversal, the combinational-backward redirect conversion, and the discipline-wide tooling and documentation tightening. The memory subsystem (LdSt FIFO, Coalescing→Cache, Coalescing.claim→Gather, Gather→WritebackArbiter, cache↔mem_if carve-out) is covered separately by `/workspace/project-plans/phase-10-memory-discipline.md`.

## Sequencing

This plan executes **after the memory plan has fully landed.** The starting baseline is the post-memory state:

- All memory forward-data edges already REGISTERED.
- Cache↔mem_if pair already documented as an ordering unit (memory plan Phase M5), so this plan's Phase 10D sweep reversal can proceed without further memory work.
- Memory unit tests already calibrated to post-memory cycle counts. Cross-cutting tests (`test_integration`, `test_panic`, workload benchmarks) carry defensive `max_cycles` bumps from memory plan's M6; this plan's Phase 10G replaces those defensive bumps with precise post-Phase-10 values.
- Issue/execute path still COMBINATIONAL forward at start (this plan converts it).

Cycle-byte-identical claims in this plan are relative to **the post-memory baseline**, not the original pre-Phase-10 baseline. Use `bench_compare.py --baseline <post-memory-ref>` for the relevant comparisons.

Shared files with the memory plan (memory plan's edits already landed; this plan touches different lines):

- `/workspace/sim/src/timing/writeback_arbiter.cpp` — memory plan flipped the gather source to `current_has_result()`. This plan flips ALU/MUL/DIV/TLOOKUP sources the same way in Phase 10B.1.
- `/workspace/sim/src/timing/timing_model.cpp` — memory plan updated gather/ldst/cache sites in `pipeline_drained()`, `execution_units_drained()`, `discard_writeback_results()`, `build_cycle_snapshot()`. This plan updates the ALU/MUL/DIV/TLOOKUP sites in the same functions.
- `/workspace/resources/timing_discipline.md` Phasing section — memory plan added Phase M-series rows; this plan extends with Phase 10A-G rows.

## Phase plan

The refactor is sequenced so each phase builds successfully and runs the full regression suite. Phases 10A and 10D are byte-identical to the post-memory baseline; every other phase has a deliberate per-benchmark cycle delta captured by `python3 tools/bench_compare.py --baseline <phase-start-ref>`.

### Phase 10A — Branch resolution moves to ALUUnit (byte-identical, interim REGISTERED redirect)

Move branch resolution from inline-in-`TimingModel::tick()` (lines 459-484) into `ALUUnit::evaluate()`. Keep the redirect REGISTERED on ALU for now; only the **owner** of the latched slot changes, not its cycle behavior.

**Interim discipline violation, deliberate:** under Principle 6 the redirect is a backward control signal and should be combinational backward, not REGISTERED. Phase 10A keeps it REGISTERED purely as a staging step so the relocation to ALU is byte-identical and reviewable in isolation. The conversion to combinational backward (the discipline-correct form) happens in Phase 10E once the back-to-front sweep is in place. Do **not** treat the REGISTERED-on-ALU shape as a steady-state design.

Files:
- `/workspace/sim/include/gpu_sim/timing/alu_unit.h`, `/workspace/sim/src/timing/alu_unit.cpp` — add `RedirectRequest current_redirect_request_/next_redirect_request_`, `set_branch_tracker(BranchShadowTracker*)`, `current_redirect_request()`, `current_redirect_request_or_override(...)`, `set_redirect_request_override(...)`. Inside `ALUUnit::evaluate()` after the existing result-buffer write: if `next_pending_input_.trace.is_branch`, run the `branch_mispredicted` check (move from `TimingModel`), call `branch_predictor_->update(...)`, and on mispredict write `next_redirect_request_`; on correct prediction call `branch_tracker_->note_resolved_correctly(w)`. The existing `ALUUnit::accept` already carries `DispatchInput.trace` and `.prediction` into `next_pending_input_` (alu_unit.cpp:5-13), so no payload plumbing is needed.
- `/workspace/sim/include/gpu_sim/timing/operand_collector.h`, `/workspace/sim/src/timing/operand_collector.cpp` — delete `RedirectRequest`, `set_branch_tracker`, `resolve_branch`, `current_redirect_request_or_override`, both `current_/next_redirect_request_` fields, and `branch_tracker_` pointer. Hoist `RedirectRequest` to `/workspace/sim/include/gpu_sim/timing/execution_unit.h`.
- `/workspace/sim/src/timing/timing_model.cpp` — delete the inline branch-resolution block (lines 459-484); delete `TimingModel::branch_mispredicted` (lines 324-337) and the declaration; replace `opcoll_->set_branch_tracker(...)` and the `set_opcoll(...)` calls on fetch/decode with `alu_->set_branch_tracker(...)`, `fetch_->set_alu(alu_.get())`, `decode_->set_alu(alu_.get())`. Pass `branch_predictor_` reference into ALU via `alu_->set_branch_predictor(...)`.
- `/workspace/sim/src/timing/fetch_stage.cpp`, `.h` — change `opcoll_` field/setter to `alu_` (typed `ALUUnit*`); `FetchStage::commit()` reads `alu_->current_redirect_request_or_override(redirect_override_)` instead of opcoll's. Override semantics on `redirect_override_` unchanged.
- `/workspace/sim/src/timing/decode_stage.cpp`, `.h` — same shape as fetch.
- `/workspace/sim/tests/test_branch.cpp`, `test_timing_components.cpp` — wire `alu->set_branch_tracker(...)` and `alu->set_redirect_request_override(...)` instead of opcoll equivalents. Keep `fetch.set_redirect_request_override` and `decode.set_redirect_request_override` as test-only late-stage overrides.

Atomicity: **single atomic commit.** Both sides write `next_redirect_request_` simultaneously; splitting "add ALU side" from "remove opcoll side" would fire the tracker twice.

Verification: full regression suite green; `bash ./tests/run_workload_benchmarks.sh --build-dir build` byte-identical to the pre-10A baseline.

### Phase 10B — Issue/execute forward data edges become REGISTERED

Convert the `scheduler → opcoll → dispatch → units → wb_arbiter` path from COMBINATIONAL forward to REGISTERED. **Phase 10B.0 lands first** — it introduces the scheduler-side issue scoreboard and writeback-slot bitmap that replace the existing busy-polling discipline. Without 10B.0, the subsequent REGISTERED-forward conversions (10B.1–10B.3) would create a multi-cycle blind spot in the scheduler's issue gate where it over-issues into busy units. After 10B.0 lands, the substep order is **downstream-first** so each substep's cycle delta is local and interpretable. **Every substep is atomic per consumer** — producer-side accessor flip, every consumer's read flip, plus `pipeline_drained()`, `execution_units_drained()`, `discard_writeback_results()`, and `build_cycle_snapshot()` all in one commit.

#### 10B.0 — Scheduler issue scoreboard + writeback slot bitmap

Replace the scheduler's availability polling (`unit->current_busy()`, `opcoll->current_busy()`) with scheduler-side bookkeeping that predicts unit availability and writeback contention from issue history and parameterized latencies. This is the canonical READY/STALL pattern in real GPU schedulers — the scheduler doesn't poll its consumers; it knows their pipeline depths and tracks its own issues.

Two pieces of state, both owned by the scheduler:

**(A) Per-unit issue scoreboard.** For each fixed-latency unit (`ALU`, `MULTIPLY`, `DIVIDE`, `TLOOKUP`) and the variable-latency `LDST`:

```cpp
struct UnitIssueState {
    uint32_t cooldown_cycles = 0;       // blocking units count down
    uint32_t in_flight = 0;             // pipelined units track depth
    uint32_t max_in_flight = 1;         // 1 for blocking; pipeline_stages_ for pipelined
};
```

Per unit-type configuration:
- `ALU`: `max_in_flight = 1` (single result_buffer slot — back-pressure on result-buffer occupancy is the gate; bitmap reservation prevents overlap implicitly).
- `MULTIPLY`: `max_in_flight = pipeline_stages_`, no cooldown (pipelined).
- `DIVIDE`: `max_in_flight = 1`, `cooldown_cycles = DIVIDE_LATENCY` on issue (blocking, sequential).
- `TLOOKUP`: `max_in_flight = 1`, `cooldown_cycles = TLOOKUP_LATENCY` on issue (blocking, sequential).
- `LDST`: separate FIFO accounting (see below).

Each cycle: decrement all `cooldown_cycles`; decrement `in_flight` when the corresponding unit's writeback completes (signaled by the bitmap routing this cycle's claim to that unit, or by observing the arbiter's consume).

**(B) Writeback slot bitmap.** A circular array of `std::optional<ExecUnit>` with length `kWritebackBitmapLen`, tracking which unit's result will arrive at the arbiter on each upcoming cycle.

The offset table evolves across 10B.0–10B.3 as REGISTERED edges replace today's combinational forward shape — each subsequent substep adds 1 to every fixed-latency entry. **At 10B.0 land time the values reflect the still-combinational forward path** (current code reality); subsequent substeps update the constants in lockstep with the edge conversions.

```cpp
// 10B.0 land state (matches today's combinational forward path):
constexpr uint32_t kIssueToWritebackOffset[ExecUnit::COUNT] = {
    /* ALU      */ 0,                                   // accept→evaluate→arbiter all same tick today
    /* MULTIPLY */ kMulPipelineStages - 1,              // pipeline_stages cycles, first decrement same tick
    /* DIVIDE   */ kDivideLatency - 1,                  // LATENCY cycles, first decrement same tick
    /* TLOOKUP  */ kTlookupLatency - 1,
    /* LDST     */ 0,                                   // variable-latency: never reserves a slot
    /* SYSTEM   */ 0,                                   // no writeback
};
// Each of 10B.1, 10B.2, 10B.3 adds 1 to every fixed-latency entry as that substep's edge converts.
// End-of-10B values: ALU=3, MUL=kMulPipelineStages+2, DIV=kDivideLatency+2, TLOOKUP=kTlookupLatency+2.
constexpr uint32_t kWritebackBitmapLen =
    std::max({kIssueToWritebackOffset[ALU],
              kIssueToWritebackOffset[MULTIPLY] + 1,    // +1 for VDOT8's extra opcoll cycle
              kIssueToWritebackOffset[DIVIDE],
              kIssueToWritebackOffset[TLOOKUP]});
```

VDOT8 instructions go to MUL but spend 2 cycles in opcoll instead of 1 (currently — until opcoll becomes always-1-cycle); compute the per-issue offset at issue time using the decoded instruction's `num_src_regs == 3` flag and add 1 to MUL's table value.

**Instructions without a writeback (BRANCH, store, ECALL/EBREAK) skip the bitmap reservation.** Detect via `decoded.has_rd && decoded.rd != 0`; if false, no slot is claimed. The unit's other state (cooldown, in_flight) still applies — these instructions still occupy unit cycles even if they don't writeback.

The bitmap is a circular buffer with `head_` index advancing one slot per cycle:

```cpp
class WarpScheduler {
    std::array<std::optional<ExecUnit>, kWritebackBitmapLen> writeback_bitmap_{};
    uint32_t bitmap_head_ = 0;

    void evaluate() {
        // Advance: consume bitmap[head] (the cycle that just elapsed) by clearing it.
        // The arbiter has already routed it.
        writeback_bitmap_[bitmap_head_] = std::nullopt;
        bitmap_head_ = (bitmap_head_ + 1) % kWritebackBitmapLen;
        // ...
    }

    std::optional<ExecUnit> current_writeback_claim() const {
        return writeback_bitmap_[bitmap_head_];
    }
};
```

**Combined issue gate.** The scheduler's `evaluate()` walks each round-robin warp candidate and issues to the first one that passes:

```cpp
ExecUnit target = decoded.target_unit;

if (scoreboard hazard on rs1/rs2/rd) skip;
if (branch_tracker_->current_in_flight(w)) skip;

if (target == LDST) {
    // FIFO accounting: ldst_in_flight_ = scheduler-side count of issues in transit.
    if (ldst_->current_fifo_size() + ldst_in_flight_ + 1 > LDST_FIFO_DEPTH) skip;
} else {
    if (unit_state_[target].cooldown_cycles > 0) skip;
    if (unit_state_[target].in_flight >= unit_state_[target].max_in_flight) skip;
    uint32_t offset = compute_issue_to_writeback_offset(target, is_vdot8);
    if (writeback_bitmap_[(bitmap_head_ + offset) % kWritebackBitmapLen]) skip;
}

// Interim opcoll cooldown — drops when opcoll becomes always-1-cycle.
if (opcoll_cooldown_cycles_ > 0) skip;

// Issue.
if (target == LDST) {
    ldst_in_flight_++;
} else {
    unit_state_[target].in_flight++;
    if (target == DIVIDE)  unit_state_[target].cooldown_cycles = DIVIDE_LATENCY;
    if (target == TLOOKUP) unit_state_[target].cooldown_cycles = TLOOKUP_LATENCY;
    writeback_bitmap_[(bitmap_head_ + offset) % kWritebackBitmapLen] = target;
}
opcoll_cooldown_cycles_ = (is_vdot8 ? 2 : 1);
```

Failure on any gate stalls *this* warp but the loop continues to the next round-robin warp. Other warps targeting non-blocked units still issue. This is the head-of-line-blocking-fix property — it falls out for free from per-unit gates.

**Arbiter restructure.** `WritebackArbiter::evaluate()` becomes deterministic for fixed-latency, opportunistic for loads:

```cpp
void WritebackArbiter::evaluate() {
    auto claim = scheduler_->current_writeback_claim();
    if (claim) {
        ExecutionUnit* src = unit_for(*claim);
        // Fixed-latency invariant: src must have a result this cycle.
        // Debug assert in non-release builds.
        assert(src->current_has_result());
        WritebackEntry entry = src->consume_result();
        scoreboard_->clear_pending(entry.warp_id, entry.dest_reg);
        // Decrement scheduler's in-flight count for *claim
        scheduler_->note_writeback_consumed(*claim);
        // ... emit writeback ...
    } else if (gather_file_->current_has_result()) {
        WritebackEntry entry = gather_file_->consume_result();
        scoreboard_->clear_pending(entry.warp_id, entry.dest_reg);
        // ... emit writeback ...
    }
    // else: no writeback this cycle (counter).
    if (claim && gather_file_->current_has_result()) {
        stats_.load_writeback_stall_cycles++;
    }
}
```

The arbiter's source-iteration loop and round-robin pointer go away. Priority is encoded in the bitmap (first-issue-wins, naturally).

**LDST in-flight decrement.** The scheduler's `ldst_in_flight_` counter increments on issue. Decrement when the in-transit issue actually arrives at LDST.accept — the scheduler can compute this deterministically because the LDST pipeline depth from issue to FIFO entry is fixed (`opcoll_latency + 1` cycles). Track per-cycle arrivals in a small `std::array<uint32_t, kLdstInFlightDepth>` ring buffer parallel to the bitmap; on each cycle, decrement `ldst_in_flight_` by the arrivals scheduled for this cycle.

**Stat counters added:**
- `load_writeback_stall_cycles` — load result ready but bitmap slot is claimed by a fixed-latency op.
- `scheduler_unit_cooldown_stall_cycles[unit]` — issue blocked by `cooldown_cycles > 0`.
- `scheduler_unit_pipeline_full_stall_cycles[unit]` — issue blocked by `in_flight >= max_in_flight`.
- `scheduler_writeback_contention_stall_cycles[unit]` — issue blocked by bitmap conflict.
- `scheduler_ldst_fifo_full_stall_cycles` — LDST issue blocked by FIFO accounting.

These replace the existing `warp_stall_unit_busy[w]` family with finer-grained reasons. Tooling/trace-counter docs (`resources/trace_and_perf_counters.md`) update in 10F.

**Files:**
- `/workspace/sim/include/gpu_sim/timing/execution_unit.h` — add `kIssueToWritebackOffset[]` table, `kWritebackBitmapLen` constant, helper `compute_issue_to_writeback_offset(ExecUnit, bool is_vdot8)`.
- `/workspace/sim/include/gpu_sim/timing/warp_scheduler.h`, `/workspace/sim/src/timing/warp_scheduler.cpp` — add `unit_state_[]`, `writeback_bitmap_`, `bitmap_head_`, `ldst_in_flight_`, `opcoll_cooldown_cycles_`. Add `current_writeback_claim()` accessor and `note_writeback_consumed(ExecUnit)` mutator. Rewrite `evaluate()` issue gate per the sketch. Delete `set_dependencies(...)`'s opcoll/unit-pointer plumbing for busy reads (keep scoreboard, branch_tracker, plus lightweight pointers for arbiter back-channel).
- `/workspace/sim/src/timing/writeback_arbiter.cpp`, `.h` — restructure `evaluate()` per the sketch. Source list shrinks to a typed array of pointers (one per unit type) so `unit_for(ExecUnit)` is a constant-time lookup. Round-robin pointer field deleted.
- `/workspace/sim/include/gpu_sim/timing/operand_collector.h`, `.cpp` — `current_busy()` accessor stays for the panic-drain query (`execution_units_drained()`) but is no longer read by the scheduler. Note this in the header.
- Each unit's `current_busy()` accessor stays for the same panic-drain query; same note.
- `/workspace/sim/include/gpu_sim/stats.h` — add the four new stall counters and `load_writeback_stall_cycles`.
- `/workspace/sim/src/timing/timing_model.cpp` — wire `scheduler_->note_writeback_consumed` into the arbiter's per-source consume path (or move the call into the arbiter's evaluate, with the scheduler pointer wired in `set_dependencies`).
- Tests: rewrite `set_unit_ready_override` / `set_opcoll_ready_override` test hooks (`test_warp_scheduler.cpp`, `test_timing_components.cpp`) to instead manipulate the scheduler's `unit_state_[]` directly through a test-only accessor (`scheduler.test_set_unit_cooldown(...)`, `test_set_unit_in_flight(...)`).

**Atomicity:** single atomic commit. The issue gate, the bitmap reservation at issue time, the bitmap consumption in the arbiter, the in-flight increment/decrement, and the test-hook migration all land together. Splitting would leave the simulator in a state where the scheduler over-issues or under-issues for a window of commits.

**Cycle behavior:** small per-benchmark cycle delta from replacing reactive arbitration with proactive reservation. Today's behavior — multiple units finishing on the same cycle, arbiter picks one and the others stretch their effective latency — gets replaced by the scheduler refusing to issue when a slot is already claimed. Net throughput should be similar (the bottleneck is still 1 writeback/cycle), but cycle-by-cycle issue patterns differ. Capture per-benchmark deltas with `bench_compare.py --baseline <pre-10B.0-ref>`. Substantial regressions indicate the bookkeeping is over-conservative relative to today's reactive arbitration; investigate before proceeding.

**Note for 10B.1–10B.3:** polling is gone after 10B.0 — the scheduler no longer reads `unit->current_busy()` or `opcoll->current_busy()` for issue gating. The remaining substeps focus on flipping forward-edge accessors and bumping the bitmap offset table; they do **not** need to introduce or modify backward stall signals. Each substep is a single atomic commit and bumps every fixed-latency entry in `kIssueToWritebackOffset[]` by 1 to reflect the new edge's added cycle of pipeline depth.

#### 10B.1 — Units → WritebackArbiter (one commit per unit: alu, mul, div, tlookup)

For each of `ALUUnit`, `MultiplyUnit`, `DivideUnit`, `TLookupUnit` (the gather-buffer source is owned by the memory plan):

- Rename `next_has_result()` → `current_has_result()`, returning the committed slot. The existing `current_result_buffer_` field is sufficient — derive the predicate from `current_result_buffer_.valid`.
- `consume_result()` continues to invalidate `next_result_buffer_`; under registered semantics the arbiter consumes from the committed slot and the unit clears for the next cycle's commit (one extra cycle of result-occupancy).
- `WritebackArbiter::evaluate`'s assert (the bitmap-driven consume from 10B.0) flips to read `current_has_result()`. Same change in `pipeline_drained()` (timing_model.cpp:278-289), `execution_units_drained()` (lines 291-299), `discard_writeback_results()` (lines 301-313), and `build_cycle_snapshot()` (lines 568+).
- **Bump `kIssueToWritebackOffset[]` for the converted unit by 1.** ALU goes 0→1, MUL goes `pipeline_stages-1`→`pipeline_stages`, DIV goes `LATENCY-1`→`LATENCY`, TLOOKUP similarly. Bitmap length recomputes; may grow by 1 if the affected unit was the longest.

Per-unit atomic commit: rename + consume_result semantics + arbiter assert flip + drain/snapshot flips + offset table bump for that unit, all together.

Cycle delta: +1 cycle per writeback for the converted unit (result waits one extra cycle in the unit's `current_result_buffer_` before the arbiter sees it). Capture with `bench_compare.py`.

#### 10B.2 — OperandCollector output → dispatch → unit accept (one commit)

- `OperandCollector::output()` returns `current_output_` (rename or move the existing `current_output()` accessor accordingly). `OperandCollector::evaluate` continues to write `next_output_`.
- `TimingModel::tick()` reads `opcoll_->current_output()` for `dispatch_to_unit`. Dispatch happens one cycle after opcoll produces.
- The unit's `accept(DispatchInput)` is called from `dispatch_to_unit` with the committed payload. The within-unit accept→evaluate path (alu_unit.cpp:10-37, divide_unit.cpp:5-32, etc.) where evaluate reads `next_pending_*` written by accept this same tick stays — that's an internal-stage combinational path, not cross-stage.
- **Bump every fixed-latency entry in `kIssueToWritebackOffset[]` by 1.** Bitmap length recomputes accordingly.

The scheduler's bookkeeping naturally accommodates the deeper pipeline because the bitmap offset table grew. No issue-gate logic changes are needed in 10B.2.

Cycle delta: +1 cycle per instruction (everything spends one extra cycle traversing the opcoll→unit register).

#### 10B.3 — WarpScheduler output → OperandCollector accept (one commit)

- `WarpScheduler::output()` returns `current_output_`; `OperandCollector::accept(scheduler->current_output())` reads from end-of-last-cycle, executing one cycle after issue.
- The top-of-tick `if (scheduler_->output()) opcoll_->accept(...)` block in `TimingModel::tick()` updates to use `current_output()`.
- **Bump every fixed-latency entry in `kIssueToWritebackOffset[]` by 1.** Bitmap length recomputes.
- The `opcoll_cooldown_cycles_` countdown in the scheduler (introduced in 10B.0) is unchanged — it already accounts for opcoll's per-cycle holding behavior.

Cycle delta: +1 cycle per instruction.

After 10B: every cross-module read in the issue/execute path is `current_*`; the scheduler's bitmap-and-bookkeeping handles all issue gating. End-of-10B `kIssueToWritebackOffset[]` values: ALU=3, MUL=`pipeline_stages+2`, DIV=`DIVIDE_LATENCY+2`, TLOOKUP=`TLOOKUP_LATENCY+2`. Capture cumulative per-benchmark deltas with `python3 tools/bench_compare.py --baseline <pre-10B.0-ref>`.

### Phase 10D — Reverse evaluate sweep to back-to-front

The cache↔mem_if ordering unit is already documented (memory plan's Phase M5). Update `TimingModel::tick()` (timing_model.cpp:411-502) to evaluate stages in reverse pipeline order:

```
wb_arbiter
{cache.evaluate → mem_if.evaluate → cache.drain_write_buffer}  [memory ordering unit]
coalescing
execution units (alu, mul, div, tlookup, ldst)
opcoll
scheduler
decode
fetch
```

The cache/mem_if/drain_write_buffer triple stays in its current relative order — naive reversal breaks the same-cycle "submit-then-decrement" interaction (`submit_read` appends to `mem_if.in_flight_`, `mem_if.evaluate` decrements it the same cycle, which is the existing memory-latency arithmetic). Treat as one ordering unit per the memory plan's M5 carve-out.

The memory ordering unit runs **before coalescing** in the back-to-front sweep so that combinational-backward signals from cache flow correctly to coalescing within the same cycle. The memory plan's Phase M3 design depends on this ordering: coalescing reads `cache.next_cmd_stall()` mid-evaluate after cache has finished computing its end-of-cycle state. The same ordering supports the existing `cache.next_stalled()` back-pressure (M4 in the memory audit), so this is the canonical position regardless of M3.

Under REGISTERED forward edges (this plan's Phase 10B and the memory plan's M1-M3 conversions, both already landed by the time 10D runs), every other stage reads only committed state. The reversal places every combinational-backward producer ahead of its consumer:
- `cache → coalescing` (cache stall signals — `next_stalled` and `next_cmd_stall`): cache runs first ✓
- `ALU → fetch/decode/scheduler` (Phase 10E redirect): ALU runs first ✓ (execution units run before opcoll/scheduler/decode/fetch)

Verification: byte-identical to end of Phase 10B.3. If `bench_compare.py` reports any delta, a hidden order dependency exists — investigate before proceeding.

### Phase 10E — Branch redirect becomes combinational backward

Delete the latched `RedirectRequest` slot on `ALUUnit`. Branch resolution still happens in `ALUUnit::evaluate()`; instead of writing `next_redirect_request_`, ALU asserts a **transient** `next_redirect()` accessor (single-slot, reset at top of `ALUUnit::evaluate`).

- `/workspace/sim/src/timing/alu_unit.cpp` — drop `current_redirect_request_`/`next_redirect_request_` and `commit()`'s flip; keep the override slot for tests but read it directly into the transient signal.
- `/workspace/sim/src/timing/fetch_stage.cpp` — move the redirect-apply logic from `commit()` into `evaluate()`. Fetch reads `alu_->next_redirect()` at the top of evaluate; if asserted, sets `warps_[w].pc = target`, calls `warps_[w].instr_buffer.flush()`, sets `next_output_ = std::nullopt` for that warp, calls `branch_tracker_->note_redirect_applied(w)`. Skip fetching this cycle for the redirected warp.
- `/workspace/sim/src/timing/decode_stage.cpp` — same: move pending-invalidate from `commit` into `evaluate`. Read `alu_->next_redirect()`; if asserted and `pending_.target_warp == warp_id`, invalidate `pending_`.
- `/workspace/sim/src/timing/warp_scheduler.cpp` — the existing `branch_tracker_.current_in_flight(w)` gate is sufficient. **Do not** add an extra `alu_->next_redirect()` issue gate to the scheduler — `current_in_flight` already covers this case and a second gate is redundant.

Mispredict shadow under back-to-front sweep with combinational redirect: ALU resolves at cycle N (writes transient), fetch.evaluate same cycle N applies flush (sets new PC, flushes buffer), fetch.evaluate next cycle N+1 fetches new PC, decode N+2, scheduler issues N+3. Approximately one cycle shorter than the same scenario at the end of Phase 10C (where redirect was REGISTERED via ALU). Capture deltas via `bench_compare.py`.

Single atomic commit: producer slot deletion + all three consumer migrations + test-redirect updates land together.

### Phase 10F — Tooling and documentation

Atomically tighten the discipline surface so the COMBINATIONAL forward flavor is no longer expressible, and update every documentation artifact to reflect the new architectural model. **Includes test-file accessor migration** so the lint tightening doesn't fail on tests that still read `next_*` accessors that the production code has deprecated.

**Tooling:**

- **`/workspace/tools/lint_timing_naming.py`** — add `BACK_PRESSURE_CARVEOUTS` allowlist for the small set of legitimate `next_*` accessors (e.g., `L1Cache::next_stalled`, `L1Cache::next_stall_reason`, `LoadGatherBufferFile::next_port_claimed`, plus the new `ExternalMemoryInterface::next_request_stall` and `L1Cache::next_cmd_stall` from M3/M5). Emit a violation for any `next_*` cross-module read outside this list. Field-name `next_*` declarations on the producer remain allowed (internal staging). The lint must distinguish "this module declares `next_foo_`" (allowed) from "this module reads `other->next_foo()`" (disallowed).
- **`/workspace/tools/diagram_extract_ast.py`** — strip `EDGE_CLASSIFICATION_OVERRIDES` entries for forward-data COMBINATIONAL edges that are now REGISTERED; keep entries for true carve-outs (cache stall, gather port arbitration, mem_if stall). Remove obsolete `next_*` entries from `KNOWN_ACCESSOR_NAMES`/`KNOWN_ACCESSOR_RE`.
- **`/workspace/tools/diagram_extract_md.py`** — update `ROW_OVERRIDES` so rows 5, 7, 12, 15 reflect REGISTERED forward classifications.
- **`/workspace/tests/test_signal_diagram.py`** — flip 3-5 entries in the edge floor from COMBINATIONAL to REGISTERED; recompute the floor count from post-refactor extractor output.

**Architectural specification (`/workspace/resources/gpu_architectural_spec.md`):** the bookkeeping change in 10B.0 is an architectural change — the scheduler's interface to the rest of the SM has shifted from polling-based to bookkeeping-based. The spec must reflect this:

- **Warp scheduler section** — describe the issue scoreboard (per-unit cooldown / in-flight tracking with parameterized blocking latency) and writeback-slot bitmap (length, per-unit reservation offsets, opportunistic load writeback in unclaimed slots). Replace any "scheduler polls unit-ready signals" language with "scheduler tracks unit availability from issue history." Note the LDST FIFO accounting on the scheduler side. Note the interim opcoll cooldown and the planned migration to always-1-cycle opcoll.
- **Writeback arbiter section** — replace the priority-arbitration model with bitmap-driven routing for fixed-latency sources and opportunistic gather writeback for loads. Document the new contention model: fixed-latency ops never collide (bookkeeping prevents it); load writeback stalls when a fixed-latency op claims the slot.
- **Branch resolution section** — update to reflect that branches resolve in `ALUUnit::evaluate()` (not in a privileged inline TimingModel block) and that misprediction redirect is a combinational-backward signal from ALU to fetch/decode/scheduler (post-10E). Pre-10E redirect was REGISTERED via OperandCollector; that path is gone.
- **Memory subsystem section** — describe the REGISTERED forward + combinational backward stall pattern at coalescing↔cache (M3) and cache↔mem_if (M5) boundaries. Reference the gather-buffer port arbitration (already documented).
- **Pipeline diagram** — update the cycle-by-cycle issue-to-writeback latency table for each unit type. End-of-Phase-10 values: ALU=3, MUL=`pipeline_stages+2`, DIV=`DIVIDE_LATENCY+2`, TLOOKUP=`TLOOKUP_LATENCY+2` (cycle offsets from issue to arbiter consume).
- **Design principles section** (if it exists, or add one) — synchronous pipeline discipline as documented in `cpp_coding_standard.md` and `timing_discipline.md`.

**Coding standard (`/workspace/resources/cpp_coding_standard.md`):** § Cross-stage signaling discipline: rewrite to define REGISTERED as the only forward-data flavor and document COMBINATIONAL backward as the only same-cycle classification, restricted to back-pressure / control. Update the prefix table and postfix examples. Cross-reference Principle 6 in `CLAUDE.md`.

**Timing discipline doc (`/workspace/resources/timing_discipline.md`):** rewrite the COMBINATIONAL section into "COMBINATIONAL backward control"; rewrite per-boundary inventory rows 5, 7, 12, 15 (and any others affected by 10B's REGISTERED conversions and M-series memory conversions) to reflect new classifications. Add a Phase 10A-G section to the Phasing reference summarizing each subphase's commit boundary and cycle delta. The memory plan's M1-M6 phases get their own subsection. Add a per-stage "scheduler scoreboard" entry describing the bookkeeping that replaces the polling paths.

**Performance simulator architecture (`/workspace/resources/perf_sim_arch.md`):**
- Update `WarpScheduler` description with the scoreboard fields (`unit_state_[]`, `writeback_bitmap_`, `bitmap_head_`, `ldst_in_flight_`, `opcoll_cooldown_cycles_`) and the bitmap-driven issue gate.
- Update `WritebackArbiter` description to reflect bitmap-driven routing (no more round-robin source iteration).
- Update `ALUUnit` description to mention branch resolution responsibility (post-10A).
- Update `OperandCollector` description to drop redirect machinery (deleted in 10A).
- Update `L1Cache` description to mention the new `next_cmd_stall()` (M3) and the cmd/response slot model.
- Update `ExternalMemoryInterface` description to mention `next_request_stall()` and the REGISTERED request slots (M5).
- Update `CoalescingUnit` description to reflect REGISTERED FIFO read (M1) and REGISTERED command submission (M3).
- Update `LoadGatherBufferFile` description to mention REGISTERED claim (M2) and `current_has_result` (M4).

**Trace and performance counters (`/workspace/resources/trace_and_perf_counters.md`):** add the new `Stats` fields:
- `load_writeback_stall_cycles` (10B.0)
- `scheduler_unit_cooldown_stall_cycles[ExecUnit]` (10B.0; per-unit array)
- `scheduler_unit_pipeline_full_stall_cycles[ExecUnit]` (10B.0; per-unit array)
- `scheduler_writeback_contention_stall_cycles[ExecUnit]` (10B.0; per-unit array)
- `scheduler_ldst_fifo_full_stall_cycles` (10B.0)

The existing `warp_stall_unit_busy[w]` counter is deprecated; replaced by the finer-grained per-reason counters above. Decide whether to remove it or keep it as a roll-up. Document the decision in the change.

**Onboarding (`/workspace/resources/onboarding.md`):** brief update to point new contributors at the canonical synchronous-pipeline discipline (Principle 6 in `CLAUDE.md`, full rules in `timing_discipline.md`). Mention the scheduler bookkeeping pattern as the answer to "how does back-pressure work?"

**`AGENTS.md` Key References:** no new entries (`/AGENTS.md` is a symlink to `CLAUDE.md`, so the Principle 6 update there is already in place from the prior commit).

**Test file accessor migration:** scatter across `test_timing_components.cpp`, `test_warp_scheduler.cpp`, `test_branch.cpp` of `next_has_result` / `next_busy` / `next_output` reads on the four units that flip to `current_*`. Memory-related test accessor migrations (`test_load_gather_buffer.cpp`, `test_cache.cpp`) belong to the memory plan.

### Phase 10G — Test cycle-count recalibration and benchmark snapshot

Final precise calibration of every cycle-asserting test and benchmark, replacing the defensive `max_cycles` bumps that the memory plan's M6 left in place for cross-cutting tests.

- Recalibrate every hard-coded cycle assertion across `test_panic.cpp`, `test_timing_components.cpp`, `test_branch.cpp`, `test_warp_scheduler.cpp`, `test_integration.cpp`. Each assertion gets re-derived from the post-refactor binary; existing magic numbers (`REQUIRE(wb.issue_cycle == 12)`, `REQUIRE(timing.cycle_count() < 50)`) are rewritten with the new value plus a brief comment of what changed.
- Replace memory plan M6's defensive `max_cycles` bumps in `test_integration`, `test_panic`, and the workload benchmarks with precise post-Phase-10 budgets. Strip the "issue-execute plan's 10G will replace this" comments memory plan added.
- Run `python3 tools/bench_compare.py --baseline <pre-phase-10-ref>` (the original pre-memory baseline) and capture the cumulative per-benchmark delta from end-of-Phase-9 to end-of-Phase-10. Document the deltas in the Phase 10 summary in `timing_discipline.md`.
- Update `/workspace/UNTESTED.md` if any deferred test coverage emerges.
- Build with `-DGPU_SIM_USE_DRAMSIM3=ON` and rerun the regression to validate the cache↔mem_if carve-out under DRAMSim3 holds through both plans.

## Critical files

Source:
- `/workspace/sim/include/gpu_sim/timing/alu_unit.h`, `/workspace/sim/src/timing/alu_unit.cpp` (10A, 10E)
- `/workspace/sim/src/timing/multiply_unit.cpp`, `divide_unit.cpp`, `tlookup_unit.cpp` (10B.1)
- `/workspace/sim/include/gpu_sim/timing/operand_collector.h`, `/workspace/sim/src/timing/operand_collector.cpp` (10A, 10B.2)
- `/workspace/sim/include/gpu_sim/timing/warp_scheduler.h`, `/workspace/sim/src/timing/warp_scheduler.cpp` (10B.3, 10E)
- `/workspace/sim/src/timing/writeback_arbiter.cpp` (10B.1 — coordinate with memory plan)
- `/workspace/sim/src/timing/fetch_stage.cpp`, `.h` (10A, 10E)
- `/workspace/sim/src/timing/decode_stage.cpp`, `.h` (10A, 10E)
- `/workspace/sim/src/timing/timing_model.cpp` (10A, 10B all, 10D, 10E — coordinate with memory plan)
- `/workspace/sim/include/gpu_sim/timing/execution_unit.h` (10A — host the lifted `RedirectRequest`)
- `/workspace/sim/include/gpu_sim/timing/branch_shadow_tracker.h` (10A — `note_resolved_correctly` writer migrates from opcoll to ALU)

Tooling:
- `/workspace/tools/lint_timing_naming.py` (10F)
- `/workspace/tools/diagram_extract_ast.py` (10F)
- `/workspace/tools/diagram_extract_md.py` (10F)
- `/workspace/tools/render_signal_diagram.py` (10F validation)

Tests:
- `/workspace/sim/tests/test_branch.cpp`, `test_timing_components.cpp` (10A wiring + 10F accessor migration)
- `/workspace/sim/tests/test_warp_scheduler.cpp` (10F accessor migration)
- All cycle-asserting tests (10G)
- `/workspace/tests/test_signal_diagram.py` (10F)

Documentation:
- `/workspace/resources/cpp_coding_standard.md` (10F)
- `/workspace/resources/timing_discipline.md` (10F — major rewrite, includes memory plan's phases in its summary)
- `/workspace/resources/perf_sim_arch.md` (10F)
- `/workspace/UNTESTED.md` (10G)

## Reused functions and patterns

- `Scoreboard` (`/workspace/sim/include/gpu_sim/timing/scoreboard.h`) — canonical REGISTERED reference; the new `current_has_result_` field in 10B.1 follows this shape.
- `BranchShadowTracker::note_branch_issued/note_resolved_correctly/note_redirect_applied` — already correctly REGISTERED; only the location of `note_resolved_correctly` writer changes (opcoll → ALU) in 10A.
- `OperandCollector::current_redirect_request_or_override` pattern (`/workspace/sim/include/gpu_sim/timing/operand_collector.h:64-68`) — moved verbatim onto `ALUUnit` in 10A so `tools/diagram_extract_ast.py` keeps a statically resolvable receiver.
- `python3 tools/bench_compare.py --baseline <git-ref>` — A/B benchmark snapshot at every phase boundary that changes cycle counts (10B.1, 10B.2, 10B.3, 10E).
- `bash ./tests/run_workload_benchmarks.sh --build-dir build` — canonical workload-benchmark entry point.

## Verification

Per phase:

1. `cmake --build build -j8` succeeds.
2. `cd build && ctest --output-on-failure` passes.
3. For phases that should be cycle-byte-identical (10A, 10D): `python3 tools/bench_compare.py --baseline <phase-start-ref>` reports zero delta on every workload benchmark. Any non-zero delta indicates a hidden ordering dependency — investigate before commit.
4. For phases that intentionally regress cycle counts (10B.*, 10E): capture per-benchmark delta from `bench_compare.py`, record in commit message and in `timing_discipline.md` Phase 10 summary.
5. `python3 tools/render_signal_diagram.py --validate` passes (AST and markdown extractors agree on every edge).
6. `python3 tools/lint_timing_naming.py` passes (after 10F it should pass strictly with no `--report-only`).

End-to-end:

- Run the full RISC-V ISA compliance suite (`/workspace/tests/riscv-isa/`) and synthetic edge tests (`/workspace/tests/synthetic/`) at end of Phase 10G — both must be green.
- Generate a signal diagram (`python3 tools/render_signal_diagram.py --output /tmp/diagram.svg`) and visually inspect: forward edges should all be the REGISTERED line style; only back-pressure / branch-redirect edges should be COMBINATIONAL.
