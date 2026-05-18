# Phase 10 — Synchronous Pipeline Discipline (issue / execute / branch / sweep)

## Context

The simulator's cross-stage signaling discipline (Phases 1-9 in `/workspace/resources/timing_discipline.md`) shipped two flavors of forward data edge: **REGISTERED** (consumer reads producer's `current_*` next cycle, +1 cycle of pipeline depth) and **COMBINATIONAL** (consumer reads producer's `next_*` same tick, zero handoff). The brief that prompted this plan (independently verified against the code in the prior conversation) identified that **COMBINATIONAL forward is being used pervasively in the issue→execute→writeback path** — `WarpScheduler::output() → OperandCollector::accept() → opcoll.evaluate → dispatch_to_unit → unit.accept → unit.evaluate → wb_arbiter.evaluate` collapses what should be 4-5 cycles into a single tick. To slow branch resolution to a sane delay, Phase 5 grafted a REGISTERED `RedirectRequest` slot onto `OperandCollector`; the slot exists only because the surrounding combinational forward edges resolve the branch too early.

The fix is the standard synchronous-pipeline discipline:

1. **Forward data edges are REGISTERED — always.** Each stage takes exactly one cycle.
2. **Backward control edges (stalls, branch redirect) are COMBINATIONAL backward.** Producer asserts mid-cycle; consumer (earlier in the pipeline) reads same cycle.
3. **The evaluate sweep reverses to back-to-front** so backward producers run before their consumers.
4. **Branch redirect becomes a combinational-backward wire from `ALUUnit`** read same-cycle by `FetchStage`/`DecodeStage`/`WarpScheduler`. The latched redirect slot on `OperandCollector` is deleted.
5. **Writeback arbitration prioritizes loads.** The arbiter retires load (and any future variable-latency) results ahead of fixed-latency results and asserts a combinational-backward stall that freezes the issue/execute region for any cycle a fixed-latency writeback is preempted. The freeze is realized as a `commit()`-gate over double-buffered stage state — which first requires normalizing every issue/execute stage to an explicit `seed_next()`/`commit()` discipline so a stalled stage genuinely holds. The scheduler's writeback-slot bitmap is a **binding schedule** for fixed-latency units: the scheduler refuses any issue that would place a second fixed-latency writeback on an already-claimed cycle, so fixed-vs-fixed writeback contention never reaches the arbiter — the arbiter only ever resolves variable-vs-fixed (load-vs-fixed) contention.

This is a deliberate fidelity gain; per-benchmark cycle counts will regress and the regressions are documented as expected.

**Scope:** this plan covers the issue/execute path, branch resolution, the evaluate sweep reversal, the combinational-backward redirect conversion, and the discipline-wide tooling and documentation tightening. The memory subsystem (LdSt FIFO, Coalescing→Cache, Coalescing.claim→Gather, Gather→WritebackArbiter, cache↔mem_if carve-out) is covered separately by `/workspace/project-plans/phase-10-memory-discipline.md`.

## Sequencing

This plan executes **after the memory plan has fully landed.** The starting baseline is the post-memory state:

- All memory forward-data edges already REGISTERED.
- Cache↔mem_if pair already documented as an ordering unit (memory plan Phase M5), so this plan's Phase 10D sweep reversal can proceed without further memory work.
- Memory unit tests already calibrated to post-memory cycle counts. **Memory plan M6 landed as a no-op** (commit `857bf82`: "no-op, inline calibration in M1-M5") — calibration was done inline in each M-phase, so there are **no defensive `max_cycles` bumps and no "10G will replace this" comments** in cross-cutting tests or benchmarks. This plan's Phase 10G recalibrates cycle assertions against current values from scratch; it does not "replace" anything left by M6.
- Issue/execute path still COMBINATIONAL forward at start (this plan converts it).

Cycle-byte-identical claims in this plan are relative to **the post-memory baseline**, not the original pre-Phase-10 baseline. Use `bench_compare.py --baseline <post-memory-ref>` for the relevant comparisons.

Shared files with the memory plan (memory plan's edits already landed; this plan touches different lines):

- `/workspace/sim/src/timing/writeback_arbiter.cpp` — memory plan flipped the gather source to `current_has_result()`. This plan flips ALU/MUL/DIV/TLOOKUP sources the same way in Phase 10B.3.
- `/workspace/sim/src/timing/timing_model.cpp` — memory plan updated gather/ldst/cache sites in `pipeline_drained()`, `execution_units_drained()`, `discard_writeback_results()`, `build_cycle_snapshot()`. This plan updates the ALU/MUL/DIV/TLOOKUP sites in the same functions.
- `/workspace/resources/timing_discipline.md` Phasing section — memory plan added Phase M-series rows; this plan extends with Phase 10A-G rows.

## Phase plan

The refactor is sequenced so each phase builds successfully and runs the full regression suite. Phases 10A, 10B.0.5, and 10D are byte-identical to the post-memory baseline; every other phase has a deliberate per-benchmark cycle delta captured by `python3 tools/bench_compare.py --baseline <phase-start-ref>`.

*Note on phase numbering: there is no Phase 10C. The letter is intentionally unused — phases here run 10A, 10B (with substeps), 10D, 10E, 10F, 10G. The memory subsystem work is numbered separately as the M-series in `/workspace/project-plans/phase-10-memory-discipline.md`.*

**Per-phase test-repair discipline.** Every phase that renames a public accessor or shifts cycle counts repairs its own affected tests *inside the same atomic commit*. The per-phase verification gate ("`cmake --build` succeeds", "`ctest` passes") is real — test repair is **not** deferred to a later phase. Three categories:

- **Compilation fixes (accessor renames).** When a phase renames a public accessor (`output()`→`current_output()` in 10B.1/10B.2; `next_has_result()`→`current_has_result()` in 10B.3), every test translation unit calling the old name migrates in the same commit. A public-method rename is not atomic without its callers — deferring them leaves the build red. Mandatory for 10B.1, 10B.2, 10B.3, 10E.
- **Exact-match cycle assertions** (`REQUIRE(wb.issue_cycle == 12)` and similar). Re-derived inline in whichever phase moves them. Re-deriving the same assertion across several phases is more work than one batch, but it keeps the per-phase gate meaningful and localizes a wrong delta to the phase that caused it instead of surfacing it as a cumulative mystery at 10G.
- **Loose-bound assertions** (`cycles < N` ceilings in `test_integration`/`test_panic`; workload-benchmark `max_cycles` budgets). Bumped once, generously, in the first phase that would trip them; tightened to precise post-Phase-10 values in 10G. This is the only test work 10G still owns.

This mirrors what the memory plan did in practice: its M6 ("batched recalibration") landed as a no-op because calibration was inlined into M1-M5 — the only way to keep the build green per phase.

### Phase 10A — Branch resolution moves to ALUUnit (byte-identical, interim REGISTERED redirect)

Move branch resolution from inline-in-`TimingModel::tick()` (the post-`opcoll_->output()` block, ~lines 470-495 after the memory plan's edits — re-confirm the range against current code; it still calls `branch_mispredicted()` and writes the redirect via `opcoll_->resolve_branch()`) into `ALUUnit::evaluate()`. Keep the redirect REGISTERED on ALU for now; only the **owner** of the latched slot changes, not its cycle behavior.

**Interim discipline violation, deliberate:** under Principle 6 the redirect is a backward control signal and should be combinational backward, not REGISTERED. Phase 10A keeps it REGISTERED purely as a staging step so the relocation to ALU is byte-identical and reviewable in isolation. The conversion to combinational backward (the discipline-correct form) happens in Phase 10E once the back-to-front sweep is in place. Do **not** treat the REGISTERED-on-ALU shape as a steady-state design.

**Commit-order requirement (byte-identical precondition).** The redirect slot moves from `OperandCollector` to `ALUUnit`, both REGISTERED. For 10A to be byte-identical, `fetch.commit()` and `decode.commit()` must continue to read *last cycle's* latched redirect — so `alu.commit()` must run **after** `fetch.commit()`/`decode.commit()` in the commit phase, exactly as `opcoll.commit()` does today (timing_discipline.md inventory row 12). The current commit-phase order in `TimingModel::tick()` (`fetch, decode, scheduler, opcoll, units, …`) already satisfies this since units commit after fetch/decode; confirm it is preserved.

Files:
- `/workspace/sim/include/gpu_sim/timing/alu_unit.h`, `/workspace/sim/src/timing/alu_unit.cpp` — add `RedirectRequest current_redirect_request_/next_redirect_request_`, `set_branch_tracker(BranchShadowTracker*)`, `current_redirect_request()`, `current_redirect_request_or_override(...)`, `set_redirect_request_override(...)`. Inside `ALUUnit::evaluate()` after the existing result-buffer write: if the ALU is executing a valid instruction this cycle and it is a branch (`next_has_pending_ && next_pending_input_.trace.is_branch`), run the `branch_mispredicted` check (move from `TimingModel`), call `branch_predictor_->update(...)`, and on mispredict write `next_redirect_request_`; on correct prediction call `branch_tracker_->note_resolved_correctly(w)`. The `next_has_pending_` guard is mandatory, not shorthand: `next_pending_input_` is only meaningful when the ALU actually has an instruction this cycle — gating on the bare `is_branch` flag would re-resolve a stale slot on an idle cycle once the seed_next discipline (10B.0.5) is in place. The existing `ALUUnit::accept` already carries `DispatchInput.trace` and `.prediction` into `next_pending_input_` (alu_unit.cpp:5-13), so no payload plumbing is needed.
- `/workspace/sim/include/gpu_sim/timing/operand_collector.h`, `/workspace/sim/src/timing/operand_collector.cpp` — delete `RedirectRequest`, `set_branch_tracker`, `resolve_branch`, `current_redirect_request_or_override`, both `current_/next_redirect_request_` fields, and `branch_tracker_` pointer. Hoist `RedirectRequest` to `/workspace/sim/include/gpu_sim/timing/execution_unit.h`.
- `/workspace/sim/src/timing/timing_model.cpp` — delete the inline branch-resolution block (~lines 470-495, post-memory-plan; re-confirm); delete `TimingModel::branch_mispredicted` (lines 324-337, still exact) and the declaration; replace `opcoll_->set_branch_tracker(...)` and the `set_opcoll(...)` calls on fetch/decode with `alu_->set_branch_tracker(...)`, `fetch_->set_alu(alu_.get())`, `decode_->set_alu(alu_.get())`. Pass `branch_predictor_` reference into ALU via `alu_->set_branch_predictor(...)`.
- `/workspace/sim/src/timing/fetch_stage.cpp`, `.h` — change `opcoll_` field/setter to `alu_` (typed `ALUUnit*`); `FetchStage::commit()` reads `alu_->current_redirect_request_or_override(redirect_override_)` instead of opcoll's. Override semantics on `redirect_override_` unchanged.
- `/workspace/sim/src/timing/decode_stage.cpp`, `.h` — same shape as fetch.
- `/workspace/sim/tests/test_branch.cpp`, `test_timing_components.cpp` — wire `alu->set_branch_tracker(...)` and `alu->set_redirect_request_override(...)` instead of opcoll equivalents. Keep `fetch.set_redirect_request_override` and `decode.set_redirect_request_override` as test-only late-stage overrides.

Atomicity: **single atomic commit.** Both sides write `next_redirect_request_` simultaneously; splitting "add ALU side" from "remove opcoll side" would fire the tracker twice.

Verification: full regression suite green; `bash ./tests/run_workload_benchmarks.sh --build-dir build` byte-identical to the pre-10A baseline.

### Phase 10B — Issue/execute forward data edges become REGISTERED

Convert the `scheduler → opcoll → dispatch → units → wb_arbiter` path from COMBINATIONAL forward to REGISTERED. **Phase 10B.0 lands first** — it introduces the scheduler-side issue scoreboard and the binding writeback-slot bitmap that replace the existing busy-polling discipline. Without 10B.0, the subsequent REGISTERED-forward conversions would create a multi-cycle blind spot in the scheduler's issue gate where it over-issues into busy units. **Phase 10B.0.5** then normalizes every issue/execute stage to an explicit `seed_next()`/`commit()` double-buffering discipline — a byte-identical prerequisite for the writeback stall. **Phases 10B.1 and 10B.2** convert the opcoll→unit and scheduler→opcoll edges via the pull model, deleting the tick-level dispatch/accept glue. **Phase 10B.3** then converts the unit→arbiter edge to REGISTERED, restructures the arbiter to fixed-priority arbitration, and introduces the combinational-backward writeback stall — all in one atomic commit.

**Why the unit→arbiter conversion and the stall land together (and last).** Converting the unit→arbiter edge to REGISTERED while the arbiter is still round-robin would create an intermediate with no holding mechanism for a preempted result: a pipelined unit whose result loses the port to a load has nothing to freeze it, so its next-cycle output would overwrite the unconsumed result. The classic patch — having `consume_result()` invalidate the unit's buffer — forces the arbiter to synchronously mutate another module's committed state, which the timing discipline's Forbidden Patterns section explicitly prohibits. The writeback stall is itself the holding mechanism: a preempted unit is frozen by the gated `commit()`, so its result is held with no invalidation needed. Landing the REGISTERED conversion together with the stall means the edge goes straight from round-robin+COMBINATIONAL (today's shape) to fixed-priority+REGISTERED+stall in one step, never passing through the round-robin+REGISTERED intermediate — so `consume_result()` is a **pure read** from the moment the edge converts and the result buffer stays an ordinary double-buffered pipeline register (see 10B.3). The stall lands last so it is a clean `commit()`-gate: by 10B.3 the pull model (10B.1/10B.2) has folded the dispatch/accept glue into the consumers' `evaluate()`, leaving no glue to special-case under the freeze, and the arbiter moves to the front of the evaluate sweep.

After 10B.0.5 the edge-conversion order is **upstream-first** (opcoll→unit, then scheduler→opcoll, then unit→arbiter) so the unit→arbiter conversion — which carries the arbiter restructure and the stall — lands once the glue is gone. Every substep is a single atomic commit — producer-side accessor flip, every consumer's read flip, plus `pipeline_drained()`, `execution_units_drained()`, `discard_writeback_results()`, and `build_cycle_snapshot()` all together.

#### Writeback arbitration model

The writeback port retires **one** result per cycle. Two classes of source contend for it:

- **Fixed-latency units** — `ALU`, `MULTIPLY`, `DIVIDE`, `TLOOKUP`. Issue→writeback distance is exactly predictable from parameterized latencies, so the scheduler places a per-issue *reservation* in the writeback bitmap (10B.0) and refuses to issue a second fixed-latency op onto an already-claimed cycle. With the bitmap gating issue, **two fixed-latency writebacks never land on the same cycle** — for genuinely fixed-latency units the bitmap is a perfect predictor.
- **Variable-latency sources** — loads (retired via `LoadGatherBufferFile`) and any execution unit that later becomes variable-latency (a data-dependent divider is the expected case). Their writeback cycle is not predictable; they reserve no bitmap slot and are arbitrated reactively.

The arbiter prioritizes variable-latency sources: **a load writeback always wins the port over a fixed-latency writeback.** Loads sit on the critical path of every dependent instruction, so retiring them as early as possible unblocks the most downstream work — making a load wait for a gap in a fixed-latency writeback stream is the wrong trade. When a load takes the port and a fixed-latency unit also has a result that cycle, that fixed-latency unit is **stalled**: the arbiter asserts a combinational-backward signal that freezes every execution unit, the operand collector, and the warp scheduler for the cycle. The frozen stages skip `commit()`, so their committed state is unchanged and the cycle re-evaluates identically on the next cycle — by which point the load has retired and the port is free. The scheduler's bitmap head freezes with everything else, so the fixed-latency schedule simply pauses for the cycle: a preempted fixed-latency unit never loses its reserved slot, it absorbs a one-cycle bubble.

Because the bitmap freezes in lockstep with the units it models, this is correct by construction — there is no skew between the bitmap and the pipeline it predicts. **This lockstep is only real once every issue/execute stage genuinely freezes on a stalled cycle** — which the current double-buffering convention does *not* guarantee: a stalled stage's `evaluate()` keeps mutating its live `next_*` state in place, so gating `commit()` alone does not hold it. Phase 10B.0.5 establishes the guarantee by normalizing every stage to an explicit `seed_next()`/`commit()` discipline; the stall mechanism (10B.3) depends on it. The bitmap is a **correctness mechanism** — the binding writeback schedule for fixed-latency units. Writeback-arbitration correctness has two pillars: (1) the bitmap, enforced at issue, eliminates fixed-vs-fixed contention before it can reach the arbiter; (2) the arbiter's load-preempts-fixed priority plus the combinational-backward stall handles the one contention class the scheduler cannot predict — a variable-latency load retiring the same cycle as a fixed writeback. Pillar (1) rests on `kIssueToWritebackOffset[]` being **exact**: each entry must equal the unit's true issue→writeback latency at the current substep, and an arbiter-side assert (10B.3) catches any offset-table error immediately. A future variable-latency divider needs no new mechanism: drop its bitmap reservation and it joins the load class, handled by pillar (2).

**Liveness.** The arbiter retires exactly one result on every cycle a result is available, so the machine always makes forward progress — there is no deadlock (the loads that preempt a fixed-latency unit are independent, already-in-flight loads draining through the unfrozen memory subsystem; once they drain the fixed-latency unit proceeds). A fixed-latency unit can be *transiently starved* by a sustained load-writeback stream — one stall cycle per preempting load, bounded by the number of outstanding loads. This is the intended priority; there is no anti-starvation override.

#### 10B.0 — Scheduler issue scoreboard + writeback bitmap

Replace the scheduler's availability polling (`unit->current_busy()`, `opcoll->current_busy()`) with scheduler-side bookkeeping that predicts unit availability and writeback contention from issue history and parameterized latencies. This is the canonical READY/STALL pattern in real GPU schedulers — the scheduler doesn't poll its consumers; it knows their pipeline depths and tracks its own issues.

Two pieces of state, both owned by the scheduler:

**(A) Per-unit structural-hazard gate.** The scheduler must not issue to a unit that cannot *accept* a new op this cycle. This is a hazard **only for non-pipelined units** — a fully-pipelined unit accepts one op per cycle by construction, so its only issue limit is the writeback port (the bitmap, part B). Pipeline occupancy is not itself a hazard: multiple ops at different stages of a pipelined unit is exactly what pipelining is for. The gate therefore classifies units by a static per-unit *iteration latency*:

```cpp
// 0 => fully pipelined: no structural input gate (issue limited only by the
//      writeback bitmap). >0 => iterative: the unit cannot accept a new op for
//      this many cycles after an issue to it.
constexpr uint32_t kUnitIterationLatency[ExecUnit::COUNT] = {
    /* ALU      */ 0,               // fully pipelined: 1 op/cycle
    /* MULTIPLY */ 0,               // fully pipelined: 1 op/cycle
    /* DIVIDE   */ kDivideLatency,  // iterative: occupies the unit for LATENCY cycles
    /* TLOOKUP  */ kTlookupLatency, // sequential: occupies the unit for LATENCY cycles
    /* LDST     */ 0,               // structural hazard handled by FIFO accounting
    /* SYSTEM   */ 0,
};

std::array<uint32_t, ExecUnit::COUNT> unit_busy_{};  // only DIVIDE/TLOOKUP ever nonzero
```

`unit_busy_[u]` is a **countdown**, not an occupancy counter: set to `kUnitIterationLatency[u]` when an op is issued to `u`, decremented by one each (non-frozen) scheduler cycle, and issue to `u` is blocked while it is `> 0`. A countdown drains to zero on its own — so, unlike a counter with paired increment/decrement, it **cannot leak** if an issue lacks a matching retirement event. Setting it at issue time (rather than at unit-arrival) is exact: the issue→unit-arrival transit delay is identical for consecutive ops to the same unit, so it cancels.

`ALU` and `MULTIPLY` are fully pipelined — `kUnitIterationLatency = 0`, so `unit_busy_` is never armed for them and they have **no input-side issue gate at all**. They issue at one warp per cycle; the writeback bitmap is the only throughput limiter, and it correctly permits a fresh issue every cycle (each cycle `bitmap_head_` advances, so the reserved slot `bitmap_head_ + offset` is a fresh entry). The bitmap also already bounds total in-flight fixed-latency writebacks to `kWritebackBitmapLen` — the correct global cap, enforced by writeback-slot availability rather than an arbitrary per-unit number. `DIVIDE` and `TLOOKUP` are iterative — they cannot accept a new op until the current one finishes. `LDST` uses separate FIFO accounting (see below); `SYSTEM` (ECALL/EBREAK) has no fixed-latency unit and no gate.

The implementer must confirm this classification against the unit sources: `ALUUnit::accept` and `MultiplyUnit::accept` accepting unconditionally (one op/cycle), and `TLookupUnit` actually being sequential. If a unit's `accept()` carries a precondition that can reject, that precondition *is* a structural hazard — give the unit a nonzero `kUnitIterationLatency`. If `TLOOKUP` turns out pipelined, its entry becomes `0`.

`DIVIDE` is fixed-latency today; when it later becomes data-dependent its iteration latency is no longer known at issue, so it leaves the `kUnitIterationLatency` table — the scheduler then gates its structural hazard on a REGISTERED `current_busy()` poll from the divide unit, and drops its bitmap reservation (writeback handled by the arbiter stall, the load class). Out of scope for Phase 10; the divide path stays isolated so this is a localized future change.

Each non-frozen cycle the scheduler decrements every nonzero `unit_busy_[u]` **and the nonzero `opcoll_cooldown_cycles_`**, both at the top of `evaluate()` before the issue gate reads them (see the sketch below). The decrement-then-check ordering is what makes the countdowns exact: a value set on issue at cycle N is checked one lower at N+1. A cooldown of 1 — every non-VDOT8 op, which includes every ALU op — therefore clears to 0 by the next cycle and **permits a fresh issue every cycle**; a cooldown of 2 (VDOT8) blocks exactly one cycle. Because the scheduler early-returns from `evaluate()` on a writeback-stall cycle (10B.3), `unit_busy_` and `opcoll_cooldown_cycles_` pause in lockstep with the iterative units (which are themselves frozen by the stall — see 10B.0.5/10B.3), and with the bitmap head.

**(B) Writeback slot bitmap — the binding fixed-latency writeback schedule.** A circular array of `std::optional<ExecUnit>` of length `kWritebackBitmapLen`. Each occupied entry marks a near-future cycle on which a fixed-latency writeback is already promised. The scheduler reserves an entry at issue time and **refuses to issue any fixed-latency op whose writeback cycle is already claimed** — so two fixed-latency writebacks can never land on the same cycle. **The bitmap models fixed-latency units only** — `LDST` and any future variable-latency unit reserve nothing. **The arbiter never reads the bitmap** — it does not need to: the schedule is enforced at issue, so the arbiter is entitled to assume (and asserts — see 10B.3) that at most one fixed-latency source presents a result on any cycle. The bitmap is a correctness mechanism, not a hint (see "Writeback arbitration model" above).

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
// The length is one greater than the largest offset: a reservation lands at
// (bitmap_head_ + offset) % kWritebackBitmapLen, so the length must strictly
// exceed every offset, otherwise a max-offset reservation aliases bitmap_head_
// itself — the slot being cleared this cycle.
constexpr uint32_t kWritebackBitmapLen =
    1 + std::max({kIssueToWritebackOffset[ALU],
                  kIssueToWritebackOffset[MULTIPLY] + 1,    // +1 for VDOT8's extra opcoll cycle
                  kIssueToWritebackOffset[DIVIDE],
                  kIssueToWritebackOffset[TLOOKUP]});
```

VDOT8 instructions go to MUL but spend 2 cycles in opcoll instead of 1 (currently — until opcoll becomes always-1-cycle); compute the per-issue offset at issue time using the decoded instruction's `num_src_regs == 3` flag and add 1 to MUL's table value.

**Instructions with no register writeback skip the bitmap reservation.** Detect per-instruction via `decoded.has_rd && decoded.rd != 0` — *not* by instruction class: conditional branches, stores, and ECALL/EBREAK have no writeback, but `JAL`/`JALR` are control-flow instructions that write the link register and *do* reserve a slot. If `writes_back` is false, no slot is claimed. Such instructions have no other issue-side state to leak: non-writeback ALU/branch ops and ECALL/EBREAK target `ALU`/`SYSTEM`, which are ungated; stores target `LDST`, whose FIFO counter is arrival-based and fires symmetrically for loads and stores. The `unit_busy_` countdown is armed only by ops targeting `DIVIDE`/`TLOOKUP`, and every `div`/`rem`/`tlookup` writes `rd` — so no non-writeback instruction can ever arm it.

The bitmap is a circular buffer with `bitmap_head_` advancing one slot per cycle:

```cpp
class WarpScheduler {
    std::array<std::optional<ExecUnit>, kWritebackBitmapLen> writeback_bitmap_{};
    uint32_t bitmap_head_ = 0;

    void evaluate() {
        // 10B.3 adds the writeback-stall freeze guard here: if the arbiter
        // asserts next_writeback_stall(), early-return at the top so
        // bitmap_head_ does not advance and no issue bookkeeping moves.
        writeback_bitmap_[bitmap_head_] = std::nullopt;   // reserved cycle elapsed
        bitmap_head_ = (bitmap_head_ + 1) % kWritebackBitmapLen;
        // Top-of-evaluate countdown bookkeeping — runs before the issue gate
        // reads any of it, so a value set on issue at cycle N is one lower
        // when the gate checks it at N+1:
        for (auto& b : unit_busy_) if (b > 0) --b;          // DIVIDE/TLOOKUP only
        if (opcoll_cooldown_cycles_ > 0) --opcoll_cooldown_cycles_;
        // ... then the issue gate (LDST in-flight is derived, not a countdown) ...
    }
};
```

**Combined issue gate.** The scheduler's `evaluate()` walks each round-robin warp candidate and issues to the first one that passes:

```cpp
ExecUnit target      = decoded.target_unit;
bool     writes_back = decoded.has_rd && decoded.rd != 0;
// offset is read by both the bitmap gate and the reservation below, so it is
// declared once here in the enclosing scope. Harmless for LDST/SYSTEM (offset
// 0, never reserve).
uint32_t offset = compute_issue_to_writeback_offset(target, is_vdot8);

if (scoreboard hazard on rs1/rs2/rd) skip;
if (branch_tracker_->current_in_flight(w)) skip;

if (target == LDST) {
    // FIFO-occupancy accounting: in-transit count = issued − pushed-to-FIFO.
    uint32_t ldst_in_flight =
        ldst_issued_total_ - ldst_->current_fifo_total_pushes();
    if (ldst_->current_fifo_size() + ldst_in_flight + 1 > LDST_FIFO_DEPTH) skip;
} else {
    // Structural hazard: 0 for fully-pipelined ALU/MULTIPLY, so this never
    // blocks them; armed only for iterative DIVIDE/TLOOKUP.
    if (unit_busy_[target] > 0) skip;
    // Writeback-port hazard: only instructions that write back reserve a slot.
    if (writes_back &&
        writeback_bitmap_[(bitmap_head_ + offset) % kWritebackBitmapLen]) skip;
}

// Interim opcoll cooldown — drops when opcoll becomes always-1-cycle.
if (opcoll_cooldown_cycles_ > 0) skip;

// Issue.
if (target == LDST) {
    ldst_issued_total_++;
} else {
    if (kUnitIterationLatency[target] > 0)
        unit_busy_[target] = kUnitIterationLatency[target];
    if (writes_back)
        writeback_bitmap_[(bitmap_head_ + offset) % kWritebackBitmapLen] = target;
}
opcoll_cooldown_cycles_ = (is_vdot8 ? 2 : 1);
```

Failure on any gate stalls *this* warp but the loop continues to the next round-robin warp. Other warps targeting non-blocked units still issue. This is the head-of-line-blocking-fix property — it falls out for free from per-unit gates.

**LDST FIFO-occupancy accounting.** The scheduler must not issue a load/store that would overflow `LdStUnit`'s addr-gen FIFO. It tracks the in-transit population as the difference of two monotonic counters rather than predicting it from a latency constant:

- `ldst_issued_total_` — a scheduler-side count incremented on every issue to `LDST`.
- `ldst_->current_fifo_total_pushes()` — a REGISTERED monotonic count `LdStUnit` exposes of ops ever pushed into the addr-gen FIFO (incremented at the M1 `LdStUnit::commit`-phase push). This is the single event the accounting pivots on — **not** `LDST.accept`, which precedes address generation.

Their difference `ldst_issued_total_ - current_fifo_total_pushes()` is exactly the count of ops issued to `LDST` but **not yet in the FIFO** — ops in scheduler→opcoll→unit transit and ops currently in multi-cycle address generation. Adding `ldst_->current_fifo_size()` (ops already in the FIFO) gives the exact count of ops that will occupy a FIFO slot; the issue gate skips an LDST issue when that sum plus one would exceed `LDST_FIFO_DEPTH`.

This accounting is **event-driven, not predictive** — there is no issue→FIFO-entry latency constant, no per-cycle countdown, and no ring. It is correct by construction: every issue increments `ldst_issued_total_`, every actual FIFO push increments `current_fifo_total_pushes()`, and their difference is the in-transit population regardless of how many cycles transit takes. Consequently it needs **no per-substep adjustment** as the REGISTERED conversions deepen the pipeline (10B.1/10B.2), and it stays correct even if address generation is variable-latency — the DIVIDE-style "fall back to a `current_busy()` poll" caveat does not apply to `LDST`. `ldst_->current_fifo_total_pushes()` and `current_fifo_size()` are REGISTERED committed-state reads, scheduler upstream of LDST — discipline-compliant back-pressure reads. Because an op's FIFO push and the matching `current_fifo_size()` increase both reflect the same `LdStUnit::commit` and so become visible to the scheduler on the same cycle, `current_fifo_size() + (issued − pushed)` is invariant across the op's FIFO-entry transition — no double-count, no gap. (Both counters are the same unsigned width, so the subtraction is modular-correct even across wraparound; the difference is always small.)

**Stat counters added:**
- `scheduler_unit_busy_stall_cycles[unit]` — issue blocked by `unit_busy_[unit] > 0` (a non-pipelined unit still occupied by an iterative op; only `DIVIDE`/`TLOOKUP` ever increment it).
- `scheduler_writeback_contention_stall_cycles[unit]` — issue blocked by a writeback-bitmap conflict (the scheduler proactively avoided a fixed-vs-fixed writeback collision).
- `scheduler_ldst_fifo_full_stall_cycles` — LDST issue blocked by FIFO accounting.

(The `fixed_writeback_preempted_cycles` arbiter-stall counter is added in 10B.3 with the stall mechanism.) These replace the existing `warp_stall_unit_busy[w]` family with finer-grained reasons. Tooling/trace-counter docs (`resources/trace_and_perf_counters.md`) update in 10F.

**Files:**
- `/workspace/sim/include/gpu_sim/timing/execution_unit.h` — add `kIssueToWritebackOffset[]` table, `kWritebackBitmapLen` constant, helper `compute_issue_to_writeback_offset(ExecUnit, bool is_vdot8)`.
- `/workspace/sim/include/gpu_sim/timing/warp_scheduler.h`, `/workspace/sim/src/timing/warp_scheduler.cpp` — add `unit_busy_[]`, `writeback_bitmap_`, `bitmap_head_`, `ldst_issued_total_`, `opcoll_cooldown_cycles_`. Rewrite `evaluate()` issue gate per the sketch. Delete `set_dependencies(...)`'s opcoll/unit-pointer plumbing for busy reads (keep scoreboard and branch_tracker).
- `/workspace/sim/include/gpu_sim/timing/ldst_unit.h` — add REGISTERED `current_fifo_size()` (committed addr-gen FIFO depth) and `current_fifo_total_pushes()` (monotonic committed count of ops ever pushed into the addr-gen FIFO) accessors, if not already present — both read by the scheduler's LDST FIFO-occupancy gate.
- `/workspace/sim/include/gpu_sim/timing/operand_collector.h`, `.cpp` — `current_busy()` accessor stays for the panic-drain query (`execution_units_drained()`) but is no longer read by the scheduler. Note this in the header.
- Each unit's `current_busy()` accessor stays for the same panic-drain query; same note.
- `/workspace/sim/include/gpu_sim/stats.h` — add the four scheduler stall counters.
- Tests: rewrite `set_unit_ready_override` / `set_opcoll_ready_override` test hooks (`test_warp_scheduler.cpp`, `test_timing_components.cpp`) to instead drive the scheduler's `unit_busy_[]` and `writeback_bitmap_` directly through test-only accessors (`scheduler.test_set_unit_busy(...)`, `test_reserve_writeback_slot(...)`). Add a regression that issues `ALU` ops on consecutive cycles and asserts all are accepted at 1/cycle — this pins the fully-pipelined-unit throughput that the removed `in_flight` cap would have broken.

The writeback arbiter is untouched in 10B.0 — it stays today's round-robin combinational arbiter until 10B.3. The scheduler's advisory bitmap only gates issue; it has no consumer in the arbiter yet.

**Atomicity:** single atomic commit. The issue gate, the bitmap reservation at issue time, the in-flight increment/decrement, the test-hook migration, and inline recalibration of the cycle assertions the new issue pattern moves (plus generous bumps to any loose `cycles < N` ceilings) all land together. Splitting would leave the simulator in a state where the scheduler over-issues or under-issues for a window of commits.

**Cycle behavior:** small per-benchmark cycle delta. The scheduler now refuses to issue a fixed-latency op when its predicted writeback cycle is already claimed, where today it issues and the (unchanged-in-10B.0) round-robin arbiter resolves the collision reactively. Net throughput should be similar (the bottleneck is still 1 writeback/cycle), but cycle-by-cycle issue patterns differ. Capture per-benchmark deltas with `bench_compare.py --baseline <pre-10B.0-ref>`. Substantial regressions indicate the bookkeeping is over-conservative; investigate before proceeding.

**Note for the remaining 10B substeps:** busy-polling is gone after 10B.0 — the scheduler no longer reads `unit->current_busy()` or `opcoll->current_busy()` for issue gating. (The scheduler does retain two cross-stage reads, `ldst_->current_fifo_size()` and `ldst_->current_fifo_total_pushes()` — REGISTERED committed-state back-pressure reads, scheduler upstream of LDST; both are discipline-compliant and are not the busy-polling pattern being removed.) 10B.0.5 is byte-identical (stage double-buffering normalization); 10B.1–10B.3 each flip one forward edge to REGISTERED and bump every fixed-latency entry in `kIssueToWritebackOffset[]` by 1 (the LDST FIFO-occupancy gate needs no such bump — it is event-driven, see 10B.0). 10B.3 additionally carries the arbiter restructure and the writeback stall. Each substep is a single atomic commit.

#### 10B.0.5 — Stage double-buffering normalization (byte-identical)

The issue/execute stages — the five execution units (`ALUUnit`, `MultiplyUnit`, `DivideUnit`, `TLookupUnit`, `LdStUnit`) and `OperandCollector` — do **not** use the canonical double-buffer convention. Their `commit()` snapshots `current_* = next_*`; `next_*` is the *persistent live state*, and `evaluate()` mutates it in place, relying on the previous `commit()` having left `next_* == current_*`. Only `Scoreboard` and `BranchShadowTracker` use the canonical form — an explicit `seed_next()` copying `current_* → next_*` at the top of the tick, then `commit()` flipping back.

Under the in-place convention, gating `commit()` does **not** freeze a stage: `evaluate()` still runs and destructively advances `next_*`, and the next cycle advances it again — a divide countdown decremented twice, a multiply pipeline advanced twice, an opcoll countdown that drops a produced output. The writeback stall (10B.3) would corrupt every stalled stage. 10B.0.5 fixes this *before* the stall exists, as a byte-identical prep commit.

**Classification criterion.** Whether a field is seed_next'd is decided **per field, not per stage.** A field must be seed_next'd iff `evaluate()` *consumes its prior-cycle (committed) value* — it reads or read-modify-writes the field expecting last cycle's state to be present (`cycles_remaining_ -= 1`, the `pipeline_` deque shift, a VDOT8's `instr_` still being collected on its second opcoll cycle). `seed_next()` re-establishes that value in `next_*` at the top of the tick. A field that `evaluate()` instead assigns **fresh** every cycle — computing it from this cycle's inputs without consulting its own prior value — must **not** be seed_next'd: doing so would feed a stale value into a cycle where the field should be recomputed from scratch (an idle cycle, or a re-run stalled cycle). This is about whether `evaluate()` *consumes* the prior value, **not** whether the value persists — a result buffer persists until the arbiter takes it, yet `evaluate()` assigns it fresh, so it is not seed_next'd (it is a plain double-buffered pipeline register — see the result-buffer bullet below and 10B.3). The deciding question is the field's role, never its stage's use of the in-place `next_*` convention — every issue/execute stage uses that convention, yet single-cycle execution slots and all result/output registers are fresh-each-cycle. Worked contrast: `ALUUnit`'s execution slot is fresh (1-cycle latency — `evaluate()` latches the op from `opcoll`'s committed output this cycle and never reads a prior value), so it is *not* seed_next'd; `OperandCollector::instr_` is consumed across cycles (a VDOT8 occupies opcoll two cycles, and the scheduler has stopped presenting the op by the second), so it *is*. Classify every field in the two lists below by this test; never infer a field's class from its stage.

**Conversion.** Add `seed_next()` to `ExecutionUnit` (so all five units carry it) and to `OperandCollector`; call it unconditionally for every issue/execute stage at the top of `TimingModel::tick()`, alongside the existing `scoreboard_.seed_next()` / `branch_tracker_.seed_next()`. `seed_next()` copies every **internal carry-forward field** `current_* → next_*`:

- `OperandCollector`: `busy_`, `cycles_remaining_`, `instr_`.
- `DivideUnit` / `TLookupUnit`: `busy_`, `cycles_remaining_`, `pending_result_`.
- `MultiplyUnit`: the `pipeline_` deque (a deque copy — cheap; depth is `kMulPipelineStages`).
- `LdStUnit`: `busy_`, `cycles_remaining_`, `pending_entry_`. **Verified:** LdSt's address generation is multi-cycle — while an address-gen op is in flight `evaluate()` consumes `busy_`/`cycles_remaining_`/`pending_entry_` from the prior cycle — so by the criterion above these are genuine carry-forward and LdSt is a full seed_next participant. (`ldst_unit.cpp:25` confirms it is still on the in-place `next_*` convention — "Operates on `next_*` (seeded equal to `current_*` by `commit()`)" — and the memory plan did not convert it; no special-casing needed. Note "uses the in-place convention" alone does not establish carry-forward — the `ALUUnit` slot used the same convention and is *not* seed_next'd; multi-cycle latency is what makes LdSt's state genuine.)

The `evaluate()` bodies are **unchanged** — they already mutate `next_*` in place assuming `next_* == current_*` on entry; `seed_next()` simply makes that precondition explicit and unconditional, so a stalled cycle (skipped `commit()`) re-establishes it and the next `evaluate()` re-runs identically.

**Fields that are deliberately *not* seed_next'd:**
- **Result buffers** (`current_/next_result_buffer_`) — not seed_next'd: `evaluate()` assigns `next_result_buffer_` fresh every cycle (the result produced this cycle, or `{valid:false}` if none), so there is no prior-cycle value to re-establish. They stay a plain double-buffered pipeline register through the REGISTERED unit→arbiter conversion (10B.3): `consume_result()` is a pure read, and an unconsumed result is held across a writeback stall by the gated `commit()` (10B.3) — no invalidation, no `commit()`-clear, no production gate.
- **`output_` registers** (`scheduler`, `opcoll`) — `evaluate()` already sets `next_output_ = nullopt` at its top and recomputes it from scratch, so it is naturally a pure function of seeded state.
- **`LdStUnit::next_push_`** — the M1 address-gen-FIFO staging slot. Move its `reset()` from `commit()` to the **top of `evaluate()`** so it is a fresh per-cycle staging slot (matching how `opcoll::evaluate` resets `next_output_`); a gated `commit()` would otherwise skip the reset. The `addr_gen_fifo_` deque keeps its M1 commit-phase-mutation discipline and is untouched.
- **`ALUUnit::has_pending_` / `pending_input_`** — the ALU has 1-cycle latency, so its execution slot is *not* multi-cycle carry-forward state; it is a per-cycle latch of `opcoll`'s output, the same category as the `output_` registers, and is **not** seed_next'd. At 10B.0.5 the ALU `evaluate()` is unchanged (byte-identical); once the pull model lands (10B.1) the ALU's `evaluate()` resets `has_pending_` at its top and the pull-model `accept()` sets it, so the slot is recomputed each cycle as a pure function of `opcoll.current_output()` — which is frozen during a writeback stall, so the stalled re-run is identical with no separately-seeded copy to go stale. `ALUUnit::seed_next()` is consequently empty (the `ExecutionUnit` interface still carries it for the iterative units). `pending_cycle_` carries no cross-cycle information for a 1-cycle ALU — confirm against `alu_unit.cpp` and drop it; if it is genuinely used as a multi-cycle counter the ALU is not 1-cycle and this classification must be revisited.

**Stats relocation.** `Stats` counters are non-hardware artifacts; a re-evaluated stalled cycle must not double-count them. Move every `Stats` increment out of `evaluate()` **and `accept()`** into `commit()`:
- `evaluate()`-resident: `alu_stats.busy_cycles`/`instructions`, `mul_stats.busy_cycles`, `div_stats.busy_cycles`, `tlookup_stats.busy_cycles`, `ldst_stats.busy_cycles`, `operand_collector_busy_cycles`.
- `accept()`-resident: `mul_stats.instructions`, `div_stats.instructions`, `tlookup_stats.instructions`, `ldst_stats.instructions` — latched at `commit()` via a per-stage `accepted_this_cycle_` flag (set by `accept()`, consumed and cleared at `commit()`).
- The `WarpScheduler` keeps its stats in `evaluate()`: it gates issue by early-returning at the top of `evaluate()` on a stall (10B.3), so its body never re-runs on a stalled cycle.

**Byte-identical.** When every cycle commits (no stall exists yet), `seed_next()` is a redundant copy — `next_*` already equals `current_*` from the prior `commit()` — and a `Stats` increment counted in `commit()` versus `evaluate()`/`accept()` yields the identical total. Verify with `bench_compare.py --baseline <pre-10B.0.5-ref>`: zero delta on every workload benchmark. A non-zero delta means a field was missed or a stat relocated incorrectly.

**Files:** each unit's `.h/.cpp` (`alu_unit`, `multiply_unit`, `divide_unit`, `tlookup_unit`, `ldst_unit`) and `operand_collector.h/.cpp` (add `seed_next()`, `accepted_this_cycle_` flag, stats relocation; LDST also moves the `next_push_` reset); `execution_unit.h` (add `seed_next()` to the `ExecutionUnit` interface); `timing_model.cpp` (call each stage's `seed_next()` at top of tick). No test changes — byte-identical.

**Atomicity:** single atomic commit.

#### 10B.1 — OperandCollector → unit: REGISTERED via the pull model (one commit)

Convert the opcoll→unit forward edge to REGISTERED, and in doing so **delete the tick-level dispatch glue**. Today `TimingModel::tick()` reads `opcoll_->output()` (live `next_*`) and pushes it into the right unit via the `dispatch_to_unit` switch — a COMBINATIONAL forward edge wired by the orchestrator. Under REGISTERED discipline every other cross-stage edge is a *pull*: the consumer's `evaluate()` reads the producer's `current_*()` accessor. This edge joins that pattern.

- `OperandCollector::output()` is renamed `current_output()` and returns `current_output_` (the committed slot). `evaluate()` continues to write `next_output_`.
- Each unit's `evaluate()` reads `opcoll_->current_output()` at its top; if present and `decoded.target_unit` matches the unit's own type, it calls its own `accept(...)` to latch the payload. Routing is distributed — each unit self-selects — so no central switch is needed. The scheduler's `unit_busy_`/bitmap gating (10B.0) guarantees `opcoll`'s output never targets a unit that cannot accept it, so no instruction is dropped.
- `accept(...)` stays a public method (the unit-test surface depends on it). Units gain a nullptr-tolerant `opcoll_` back-pointer, wired by `TimingModel` (nullptr-tolerant for unit tests that exercise a unit in isolation, matching the existing `scoreboard_ == nullptr` convention).
- `TimingModel::dispatch_to_unit` and the `if (opcoll_->output()) dispatch_to_unit(...)` block are **deleted**. The issue/execute region of `tick()` becomes a bare `evaluate()` sweep.
- **Bump every fixed-latency entry in `kIssueToWritebackOffset[]` by 1.** Bitmap length recomputes accordingly.

Because consumers pull `current_*` (committed end-of-last-cycle), this edge is sweep-order-independent from the moment it converts — no mid-tick ordering constraint survives. The scheduler's bookkeeping accommodates the deeper pipeline automatically because the bitmap offset table grew; no issue-gate logic changes are needed.

Cycle delta: +1 cycle per instruction (one extra cycle traversing the opcoll→unit register).

#### 10B.2 — WarpScheduler → OperandCollector: REGISTERED via the pull model (one commit)

The same conversion for the scheduler→opcoll edge, deleting the last piece of tick-level glue.

- `WarpScheduler::output()` is renamed `current_output()` and returns `current_output_`. `evaluate()` continues to write `next_output_`.
- `OperandCollector::evaluate()` reads `scheduler_->current_output()` at its top; if present and opcoll is free, it calls its own `accept(...)` to latch the issue. opcoll gains a nullptr-tolerant `scheduler_` back-pointer wired by `TimingModel`.
- The `if (scheduler_->output()) opcoll_->accept(...)` block in `TimingModel::tick()` is **deleted**.
- **Bump every fixed-latency entry in `kIssueToWritebackOffset[]` by 1.** Bitmap length recomputes.
- The `opcoll_cooldown_cycles_` countdown in the scheduler (introduced in 10B.0) is unchanged — it already accounts for opcoll's per-cycle holding behavior.

Cycle delta: +1 cycle per instruction.

#### 10B.3 — Units → WritebackArbiter REGISTERED; arbiter → fixed-priority; the combinational-backward writeback stall

This is the final 10B substep and a **single atomic commit**. It converts the last forward edge — unit→arbiter — to REGISTERED, restructures the arbiter from round-robin to fixed-priority, and introduces the combinational-backward writeback stall. The three land **together** deliberately: the writeback stall *is* the holding mechanism for a preempted result, so converting the unit→arbiter edge to REGISTERED while the arbiter is still round-robin (no stall) would leave a pipelined unit's unconsumed result with nothing to freeze it — its next-cycle output would overwrite it. Patching that with a `consume_result()` that invalidates the unit's buffer would force the arbiter to synchronously mutate another module's committed state, which the timing discipline's Forbidden Patterns section explicitly prohibits. Landing the conversion together with the stall takes the edge straight from round-robin+COMBINATIONAL (today's shape) to fixed-priority+REGISTERED+stall, never through the round-robin+REGISTERED intermediate — so `consume_result()` stays a **pure read**. By this substep every other issue/execute edge is already REGISTERED, the tick-level glue is gone (the pull model, 10B.1/10B.2, folded it into the consumers' `evaluate()`), and every stage is explicitly double-buffered (10B.0.5), so the stall is a clean `commit()`-gate with no glue to special-case.

**Unit→arbiter REGISTERED conversion.** For each of `ALUUnit`, `MultiplyUnit`, `DivideUnit`, `TLookupUnit` (the gather-buffer source is owned by the memory plan):

- Rename `next_has_result()` → `current_has_result()`, deriving the predicate from the committed `current_result_buffer_.valid`.
- **The result buffer becomes a plain double-buffered pipeline register.** `evaluate()` writes `next_result_buffer_` fresh every cycle — the result produced this cycle, or `{valid:false}` if none. `commit()` flips `current_result_buffer_ = next_result_buffer_`. It is not seed_next'd (10B.0.5): `evaluate()` assigns it fresh, so there is no prior-cycle value to re-establish.
- **`consume_result()` is a pure read.** It returns the `WritebackEntry` and mutates nothing — no cross-module write, so the Forbidden-Patterns rule against a `consume_*` synchronously mutating another stage's state is satisfied with no carve-out, and no `commit()`-clear or production gate is needed.
- **An unconsumed (preempted) result is held by the gated `commit()`.** Each cycle a fixed-latency unit's result is in exactly one of two states: *consumed* — no contention (or it won the port), so the unit is not stalled, its `commit()` runs, and its `evaluate()` overwrites `next_result_buffer_` with the next pipeline output; or *not consumed* — a load preempted it, which is precisely the writeback-stall condition, so the unit is frozen, `commit()` is gated, and `current_result_buffer_` holds untouched while the frozen `evaluate()` re-derives the identical value into `next_`. A non-frozen unit's prior result was, by definition, already consumed, so the producer can never clobber a live result. This is the "either it stalls on WB contention, or it succeeds and falls out next cycle" property — it needs no invalidation.
- **Sweep-order-independent.** The arbiter reads `current_*` (committed, stable across the whole evaluate phase) and `consume_result()` writes nothing, so the result-buffer handoff is correct under both the front-to-back sweep in effect during 10B and the back-to-front sweep after 10D.

**Arbiter restructure — fixed-priority.** `WritebackArbiter::evaluate()` reads committed `current_has_result()` on every source and classifies each as variable-latency (the gather buffer; later, any variable-latency unit) or fixed-latency. Priority is total:

1. **Variable-latency sources first** — the gather buffer (loads).
2. **Fixed-latency units second** — the 10B.0 binding writeback bitmap guarantees at most one fixed-latency unit presents a result on any cycle, so there is never a fixed-vs-fixed tie to break; the arbiter `assert`s this invariant rather than assuming it silently.

The round-robin `rr_pointer_` and the `writeback_conflicts` stat are deleted; priority is total and deterministic. The arbiter consumes the highest-priority ready source. If it consumes a variable-latency source while a fixed-latency unit also has a result, that fixed-latency unit lost the port — the arbiter asserts `next_writeback_stall()`:

```cpp
void WritebackArbiter::evaluate() {
    writeback_stall_ = false;                          // reset every evaluate
    // The scheduler's binding writeback bitmap (10B.0) guarantees at most one
    // fixed-latency source presents a result on any cycle. Assert it — a failure
    // means kIssueToWritebackOffset[] is wrong for some unit.
    assert(count_fixed_with_result() <= 1 &&
           "scheduler bitmap must prevent fixed-vs-fixed writeback contention");
    ExecutionUnit* fixed = first_fixed_with_result();  // <=1, asserted above
    if (gather_file_->current_has_result()) {
        // consume gather, clear scoreboard, emit writeback
        writeback_stall_ = (fixed != nullptr);         // fixed unit preempted by a load
    } else if (fixed) {
        // consume fixed, clear scoreboard, emit writeback
    }
    // else: idle writeback cycle.
}
bool next_writeback_stall() const { return writeback_stall_; }  // COMBINATIONAL backward
```

**Move `WritebackArbiter` to the front of the evaluate sweep** so `next_writeback_stall()` is readable same-cycle by every consumer. Pre-10D this is a partial reorder (`wb_arbiter` first, the rest of the front-to-back sweep unchanged); 10D's back-to-front reversal keeps it there.

**Gather-source capture is sweep-order-independent.** Moving the arbiter to the front does not change which results it captures. The arbiter reads every source — units and the gather buffer — through `current_has_result()`, a committed-state read; committed state is stable across the entire evaluate phase, so the arbiter at the front of the sweep sees the *identical* set of ready results it would see at any other position. A gather result completed (committed) at cycle N is captured at N+1 regardless of arbiter position — and the 1-cycle "no writeback the cycle a result is generated" delay is enforced by the REGISTERED `current_*` read (memory plan M4), not by the firing order. A completed gather buffer holds `current_has_result()` true until drained, so an older queued result is still captured whenever the arbiter reaches it; nothing ages out. What the front move *does* change is the **`consume_result()` side-effect ordering** — `consume_result()` on the gather source now runs before `gather.evaluate()` instead of after. This does not affect capture (the consume extracts the committed payload before `gather.evaluate()` runs), but this substep must confirm against the memory plan that `consume_result()` on the gather source writes `next_` only — no synchronous committed-state mutation, per the Phase-1 rule — and that `gather.evaluate()`'s buffer-reuse/allocation reads are unaffected by the consume's `next_` release now preceding them. Gate: a byte-identical run of the memory plan's gather-buffer tests (`test_load_gather_buffer.cpp`) with `wb_arbiter` moved ahead of the gather buffer.

**Drain/snapshot sites.** The `next_has_result()` → `current_has_result()` rename propagates to `pipeline_drained()` (timing_model.cpp:278-289), `execution_units_drained()` (lines 291-299), `discard_writeback_results()` (lines 301-313), and `build_cycle_snapshot()` (line 579+ after the memory plan's edits).

**Offset table.** Bump every fixed-latency entry in `kIssueToWritebackOffset[]` by 1 (the unit→arbiter edge gained a cycle of pipeline depth) — end-of-10B values ALU=3, MUL=`pipeline_stages+2`, DIV/TLOOKUP=`LATENCY+2`. Bitmap length recomputes.

**The writeback stall — a combinational clock-enable.** `next_writeback_stall()` is a single-slot COMBINATIONAL-backward signal (asserted-blocking polarity, `next_` prefix per the discipline; reset at the top of every `WritebackArbiter::evaluate()`). It models a pipeline-register clock-enable — it gates *whether state latches*, not *whether logic runs*. There is no centralized freeze: each stage in the issue/execute region reads the signal itself and self-gates its `commit()`. This is sound **only because 10B.0.5 made every stage's `evaluate()` a pure function of seeded `current_*` state** — a stalled `evaluate()` recomputes `next_*` identically from held `current_*` inputs, and a gated `commit()` discards it. Without 10B.0.5 the in-place `evaluate()` would corrupt the held state; with it, the `commit()`-gate is the whole mechanism.

- **Datapath stages (the five units, `OperandCollector`) gate `commit()`.** Each reads `next_writeback_stall()` at the top of `commit()`; if asserted, `commit()` performs no `next_→current_` flip — the stage holds. `evaluate()` runs unconditionally and is re-runnable (10B.0.5). There is **no dispatch/accept glue to gate** — the pull model (10B.1/10B.2) folded it into the consumers' `evaluate()`, where a frozen producer's `current_output()` is simply a frozen input that the re-runnable `evaluate()` reads idempotently.

- **The scheduler gates issue in `evaluate()`, and also gates `commit()`.** Issue is atomic with claiming a scoreboard destination (`set_pending`), and the scoreboard is committed *unconditionally* (it must carry the arbiter's `clear_pending`). The scheduler reads `next_writeback_stall()` at the top of `evaluate()`; if asserted it early-returns — issues nothing and advances no issue-bookkeeping (`bitmap_head_`, reservations, `unit_busy_`, the LDST ring), so the writeback *schedule* freezes in lockstep with the pipeline. `commit()` is gated like the datapath stages so `current_output_` holds — clearing it would lose the already-issued instruction.

- **Not gated:** the writeback arbiter, the memory subsystem, fetch, decode, and every shared registered structure — `scoreboard`, `branch_tracker`, the branch predictor, the register file, the gather buffer — all `seed_next()`/`commit()` every cycle. This resolves the scoreboard case directly: on a stalled cycle the ungated arbiter clears the load's destination (`clear_pending`), the scoreboard commits, and the resumed scheduler sees the freed register next cycle; the scheduler's `set_pending` is absent only because the issue itself did not happen. A stall yields exactly "no new claims, one retirement". Fetch and decode are not frozen either; they are decoupled from the gated scheduler by the per-warp instruction buffer (`WarpState::instr_buffer`), which behaves as a registered FIFO — decode pushes at `decode.commit()`, the scheduler pops at `scheduler.evaluate()`. **Verified safe under the stall** (against `decode_stage.cpp` / `fetch_stage.cpp`): a frozen scheduler's `evaluate()` early-returns before the buffer scan, so it does not pop; `decode.commit()`'s push is explicitly `is_full()`-guarded and clears `pending_` only on a successful push, so a full buffer makes decode *hold* rather than drop an instruction; and fetch's `will_be_full` eligibility check already reserves a slot for every in-flight push (`decode.pending_`, `fetch.current_output_`, the new pick) **without** crediting a same-cycle scheduler pop — so a non-popping frozen scheduler is exactly the conservative case it is built for, and fetch cannot over-fetch. The buffer is a bounded deque: no loss, no overflow. (The decode↔scheduler decoupler is this instruction-buffer FIFO, not a `current_busy()` handshake — the REGISTERED busy handshake is the *fetch↔decode* edge.)

- **ALU branch resolution fires during the stall.** Branch resolution is combinational — the clock-enable gates state latches, not the resolution datapath — so a stalled ALU still resolves its branch and drives the redirect to the (ungated) frontend with no added misprediction latency. But the stall holds the branch in the ALU's resolve-stage register, so `evaluate()` re-runs the *same* branch each stalled cycle, which would re-update the branch predictor and re-write the branch tracker (writes to shared structures, not pipeline registers, so the gated `commit()` does not protect them). The ALU carries a one-bit control register `branch_resolved_`: resolution side-effects fire only when it is clear; firing them sets it; it is cleared whenever the resolve-stage register advances (a non-stalled `commit()`). "Branch still at the resolve stage" ⟺ "ALU was stalled" (the ALU is fully pipelined; the writeback stall is the only thing that holds it), so no instruction tag is needed. `branch_resolved_` is **category-4 control state** — deliberately *not* gated by the stall and *not* seed_next'd — it must update on stalled cycles to record that resolution fired. At 10B.3 the redirect is still REGISTERED (10A's interim form), so the gated `commit()` already dedups it; `branch_resolved_` is required here for the predictor and tracker writes. Phase 10E reuses the same bit once the redirect becomes combinational.

**Panic-flush vs. stall precedence.** A stalled stage's gated `commit()` skips only the normal `next_→current_` flip; the `next_writeback_stall()` check does **not** guard `flush()`. When `pending_panic_flush_` is armed, each panic-flush target runs `flush()` at the commit-phase boundary regardless of the stall, and `flush()` writes the post-flush reset state directly into `current_*` — so the gated flip is moot. A stage that is both stalled and flushed ends the cycle in the flushed state.

The arbiter is moved to the front of the evaluate sweep in this substep (see above), so `next_writeback_stall()` is readable same-cycle by every consumer.

**Stat counters:** add `fixed_writeback_preempted_cycles` — cycles a fixed-latency writeback was held off because a load took the port (equivalently, writeback-stall cycles). Remove `writeback_conflicts` — under fixed-priority arbitration there is no round-robin conflict to count; `fixed_writeback_preempted_cycles` is its semantic successor.

**Offset-table regression:** add a test that interleaves fixed-latency ops across `ALU`/`MULTIPLY`/`DIVIDE`/`TLOOKUP` at maximum issue rate and runs to completion — the arbiter's `count_fixed_with_result() <= 1` assert never tripping is the live check that `kIssueToWritebackOffset[]` is exact. The assert and this regression land in this substep alongside the offset bump; the 10B.1/10B.2 offset bumps are validated retroactively the first time the regression runs here.

**Files:** `writeback_arbiter.h/.cpp` (priority restructure, `writeback_stall_` slot + `next_writeback_stall()`, source classification reads `current_has_result()`, delete `rr_pointer_` and `writeback_conflicts`); each unit's `.h/.cpp` (`next_has_result()`→`current_has_result()` rename, result buffer as a plain double-buffered pipeline register, pure-read `consume_result()`, stall-gated `commit()`; the ALU additionally gets the `branch_resolved_` control bit) and `operand_collector.h/.cpp` (stall-gated `commit()`); `warp_scheduler.h/.cpp` (stall early-return at the top of `evaluate()` + stall-gated `commit()`) — each consumer wires a `WritebackArbiter*`; `timing_model.cpp` (move `wb_arbiter` to the front of the sweep, wire the arbiter pointer into every consumer, drain/snapshot rename); `stats.h` (`fixed_writeback_preempted_cycles`, remove `writeback_conflicts`); tests (migrate every renamed-accessor call site in the same commit).

**Atomicity:** single atomic commit. **Cycle delta:** +1 cycle per writeback from the REGISTERED unit→arbiter edge, plus the new stall behavior. Capture with `bench_compare.py`.

After 10B: every cross-module read in the issue/execute path is `current_*`; the scheduler's bitmap-and-bookkeeping handles all issue gating. End-of-10B `kIssueToWritebackOffset[]` values: ALU=3, MUL=`pipeline_stages+2`, DIV=`DIVIDE_LATENCY+2`, TLOOKUP=`TLOOKUP_LATENCY+2`. Capture cumulative per-benchmark deltas with `python3 tools/bench_compare.py --baseline <pre-10B.0-ref>`.

### Phase 10D — Reverse evaluate sweep to back-to-front

The cache↔mem_if ordering unit is already documented (memory plan's Phase M5). Update `TimingModel::tick()` (timing_model.cpp:411-502) to evaluate stages in reverse pipeline order:

```
wb_arbiter
gather_file
{cache.evaluate → mem_if.evaluate → cache.drain_write_buffer}  [memory ordering unit]
coalescing
execution units (alu, mul, div, tlookup, ldst)
opcoll
scheduler
decode
fetch
```

The cache/mem_if/drain_write_buffer triple stays in its current relative order — naive reversal breaks the same-cycle "submit-then-decrement" interaction (`submit_read` appends to `mem_if.in_flight_`, `mem_if.evaluate` decrements it the same cycle, which is the existing memory-latency arithmetic). Treat as one ordering unit per the memory plan's M5 carve-out.

`gather_file` (`LoadGatherBufferFile`) is placed between `wb_arbiter` and the cache triple: it is downstream of cache in the dataflow (cache `try_write`s fills into it) and upstream of `wb_arbiter`, so back-to-front order puts it second. This satisfies the memory plan's M2 constraint that `gather_file.evaluate()` run before `cache.evaluate()` (so a FILL observes freshly-applied claim metadata) for free.

The memory ordering unit runs **before coalescing** in the back-to-front sweep so that combinational-backward signals from cache flow correctly to coalescing within the same cycle. The memory plan's Phase M3 design depends on this ordering: M3 was refactored (commit `3163147`) from the original cmd-retry/`next_cmd_stall()` model to a valid/ready handshake — coalescing now reads `cache.next_cmd_ready()` combinationally, which requires cache to evaluate earlier in the sweep. The same ordering supports the existing `cache.next_stalled()` back-pressure (M4 in the memory audit), so this is the canonical position regardless of M3.

Under REGISTERED forward edges (this plan's Phase 10B and the memory plan's M1-M3 conversions, both already landed by the time 10D runs), every other stage reads only committed state. The reversal places every combinational-backward producer ahead of its consumer:
- `cache → coalescing` (cache back-pressure / ack — `next_stalled` and the M3-refactor `next_cmd_ready`): cache runs first ✓
- `ALU → fetch/decode/scheduler` (Phase 10E redirect): ALU runs first ✓ (execution units run before opcoll/scheduler/decode/fetch)
- `wb_arbiter → units/opcoll/scheduler` (Phase 10B.3 writeback stall): `wb_arbiter` runs first ✓ (already moved to the front of the sweep in 10B.3; 10D's reversal keeps it there)

Verification: byte-identical to **end of Phase 10B.3** (10D's phase-start ref — 10B.3's writeback-stall cycle delta is already in the baseline). If `bench_compare.py` reports any delta, a hidden order dependency exists — investigate before proceeding.

### Phase 10E — Branch redirect becomes combinational backward

Delete the latched `RedirectRequest` slot on `ALUUnit`. Branch resolution still happens in `ALUUnit::evaluate()`; instead of writing `next_redirect_request_`, ALU asserts a **transient** `next_redirect()` accessor (single-slot, reset at top of `ALUUnit::evaluate`).

- `/workspace/sim/src/timing/alu_unit.cpp` — drop `current_redirect_request_`/`next_redirect_request_` and `commit()`'s flip; keep the override slot for tests but read it directly into the transient signal. The transient now fires even on a writeback-stalled cycle (10B.3's clock-enable gates state latches, not the resolution datapath); gate the transient assertion on the `branch_resolved_` bit (introduced in 10B.3 for the predictor/tracker writes) so a branch held at the resolve stage by a multi-cycle stall asserts the redirect exactly once.
- `/workspace/sim/src/timing/fetch_stage.cpp` — move the redirect-apply logic from `commit()` into `evaluate()`. Fetch reads `alu_->next_redirect()` at the top of evaluate; if asserted, sets `warps_[w].pc = target`, calls `warps_[w].instr_buffer.flush()`, sets `next_output_ = std::nullopt` for that warp, calls `branch_tracker_->note_redirect_applied(w)`. Skip fetching this cycle for the redirected warp.
- `/workspace/sim/src/timing/decode_stage.cpp` — same: move pending-invalidate from `commit` into `evaluate`. Read `alu_->next_redirect()`; if asserted and `pending_.target_warp == warp_id`, invalidate `pending_`.
- `/workspace/sim/src/timing/warp_scheduler.cpp` — the existing `branch_tracker_.current_in_flight(w)` gate is sufficient. **Do not** add an extra `alu_->next_redirect()` issue gate to the scheduler — `current_in_flight` already covers this case and a second gate is redundant.

Mispredict shadow under back-to-front sweep with combinational redirect: ALU resolves at cycle N (writes transient), fetch.evaluate same cycle N applies flush (sets new PC, flushes buffer), fetch.evaluate next cycle N+1 fetches new PC, decode N+2, scheduler issues N+3. This holds whether or not cycle N is writeback-stalled — the redirect is combinational and the frontend is not gated by the stall (see 10B.3). Approximately one cycle shorter than the same scenario at the end of Phase 10D (where redirect was REGISTERED via ALU). Capture deltas via `bench_compare.py`.

Single atomic commit: producer slot deletion + all three consumer migrations + test-redirect updates land together.

### Phase 10F — Tooling and documentation

Atomically tighten the discipline surface so the COMBINATIONAL forward flavor is no longer expressible, and update every documentation artifact to reflect the new architectural model. Test-file accessor migration is **not** part of 10F — under the per-phase test-repair discipline it already landed phase-by-phase with each accessor rename (10B.1–10B.3, 10E), so the lint can be tightened against an already-clean test tree.

**Tooling:**

- **`/workspace/tools/lint_timing_naming.py`** — **the largest single item in 10F. This is a net-new static-analysis pass, not an allowlist tweak — do not under-invest in it.** The current script only inspects *header naming* (prefix / postfix / polarity / field-shape layers); it has **no cross-module-read analysis whatsoever**. 10F adds a whole new analysis pass with libclang-level call-site resolution. Budget it as a substantial implementation task in its own right — comparable in effort to a small phase — and verify it works rather than treating it as a documentation afterthought.

  The new check scans `.cpp` call sites for cross-module `other->next_*()` reads and classifies each by **dataflow direction**, not against an enumerated allowlist. Direction is structural: a `next_*` read is **COMBINATIONAL backward** (legitimate back-pressure / stall) when the reading module is *upstream* of the read module, and **COMBINATIONAL forward** (forbidden — it collapses pipeline depth) when it is *downstream*. The pipeline's dataflow order is already maintained as `MODULE_ORDER` in `tools/diagram_extract_ast.py` (kept in dataflow order by the timing-discipline "New module checklist"). The lint shares that single ordering — factor `MODULE_ORDER` into a small shared module imported by both tools rather than duplicating it — and for each `reader → producer->next_*()` read:
    - `index(reader) < index(producer)` → backward → **OK**
    - `index(reader) > index(producer)` → forward → **violation**
    - same module → self-read of an own staging field → **exempt**

  There is **no `BACK_PRESSURE_CARVEOUTS` allowlist.** Combinational-backward stalls are the *expected, first-class* legitimate use of `next_*` cross-module reads — under the new discipline they are the primary reason a back edge exists, not exceptions to enumerate. Every current backward edge classifies correctly for free: `coalescing → cache->next_stalled`/`next_cmd_ready` (coalescing upstream ✓), `cache → mem_if->next_request_stall` (cache upstream ✓), `units`/`opcoll`/`scheduler → wb_arbiter->next_writeback_stall` (the arbiter is last in dataflow order, so every consumer is upstream ✓), `gather_file → next_port_claimed` (self-read ✓). After Phase 10 there are no legitimate combinational-*forward* edges left, so any forward `next_*` cross-module read is unconditionally a violation.

  Implementation pieces, each non-trivial:
    1. **Call-site extraction.** Walk the timing `.cpp` translation units (reuse the libclang machinery already in `diagram_extract_ast.py` — `build/compile_commands.json` must exist) and find every `expr->next_*()` member-call whose receiver resolves to a different class than the enclosing method's class.
    2. **Static receiver/enclosing-class resolution.** Both the receiver's class and the enclosing method's class must resolve statically. The discipline already forbids cross-stage reads inside lambda bodies / free-function helpers precisely so this resolution is always possible — the lint depends on that invariant and should emit a clear diagnostic (not a silent skip) if a receiver fails to resolve.
    3. **Module identity over the unit hierarchy.** Map both classes to `MODULE_ORDER` entries. Handle the `ExecutionUnit` base / concrete-unit hierarchy — a read through an `ExecutionUnit*` resolves to the base; treat the whole unit hierarchy as one `MODULE_ORDER` position.
    4. **Self-read exemption.** A module reading its own `next_foo_()` — including via a method invoked by another module, where the receiver is still `this` — is internal staging and exempt. Preserve the existing distinction between *declaring* `next_foo_` and *reading* `other->next_foo()`.
    5. **REGISTERED-alias cleanup.** `ExternalMemoryInterface::next_has_response()` is a `next_`-prefixed *compatibility alias* that actually returns committed state (M5) — a naming wart, not a combinational edge. The MODULE_ORDER rule would pass it as "backward" for the wrong reason. Clean it up to `current_has_response()` in 10F (preferred), or give it one explicit commented exception; do not let it pass silently.
    6. **CTest wiring.** The new check joins the existing `timing_naming_lint` target — any finding fails the build — and surfaces under `--report-only` alongside the header-naming findings.

  Verification for this step: the lint runs clean (zero findings, no `--report-only`) against the post-10E tree. A non-empty result means a combinational-forward edge survived a phase — the responsible phase regressed and must be fixed before 10F lands.
- **`/workspace/tools/diagram_extract_ast.py`** — strip `EDGE_CLASSIFICATION_OVERRIDES` entries for forward-data COMBINATIONAL edges that are now REGISTERED; keep entries for true carve-outs (cache stall, gather port arbitration, mem_if stall). Remove obsolete `next_*` entries from `KNOWN_ACCESSOR_NAMES`/`KNOWN_ACCESSOR_RE`.
- **`/workspace/tools/diagram_extract_md.py`** — update `ROW_OVERRIDES` so rows 5, 7, 12, 15 reflect REGISTERED forward classifications.
- **`/workspace/tests/test_signal_diagram.py`** — flip 3-5 entries in the edge floor from COMBINATIONAL to REGISTERED; recompute the floor count from post-refactor extractor output.

**Architectural specification (`/workspace/resources/gpu_architectural_spec.md`):** the bookkeeping change in 10B.0 is an architectural change — the scheduler's interface to the rest of the SM has shifted from polling-based to bookkeeping-based. The spec must reflect this:

- **Warp scheduler section** — describe the per-unit structural-hazard gate (an iteration-latency countdown that gates issue to non-pipelined units only — `DIVIDE`/`TLOOKUP`; fully-pipelined `ALU`/`MULTIPLY` have no input-side gate) and the **binding** writeback-slot bitmap (the fixed-latency writeback schedule, enforced at issue; fixed-latency reservations only; frozen on a writeback-stall cycle; not consulted by the arbiter). Replace any "scheduler polls unit-ready signals" language with "scheduler tracks unit availability from issue history." Note the LDST FIFO accounting on the scheduler side. Note the interim opcoll cooldown and the planned migration to always-1-cycle opcoll.
- **Writeback arbiter section** — describe fixed-priority arbitration: variable-latency sources (loads via the gather buffer, plus any future variable-latency execution unit) win the writeback port over fixed-latency units. Document the combinational-backward writeback stall — when a load preempts a fixed-latency writeback the arbiter freezes the execution units, operand collector, and warp scheduler for the cycle, which then re-evaluates identically next cycle. Note that fixed-vs-fixed contention cannot occur — the binding writeback bitmap prevents it at issue time, and the arbiter asserts the invariant. `kIssueToWritebackOffset[]` is correctness-critical and must be kept exact across the 10B substeps.
- **Branch resolution section** — update to reflect that branches resolve in `ALUUnit::evaluate()` (not in a privileged inline TimingModel block) and that misprediction redirect is a combinational-backward signal from ALU to fetch/decode/scheduler (post-10E). Pre-10E redirect was REGISTERED via OperandCollector; that path is gone.
- **Memory subsystem section** — describe the REGISTERED forward + combinational backward stall pattern at coalescing↔cache (M3) and cache↔mem_if (M5) boundaries. Reference the gather-buffer port arbitration (already documented).
- **Pipeline diagram** — update the cycle-by-cycle issue-to-writeback latency table for each unit type. End-of-Phase-10 values: ALU=3, MUL=`pipeline_stages+2`, DIV=`DIVIDE_LATENCY+2`, TLOOKUP=`TLOOKUP_LATENCY+2` (cycle offsets from issue to arbiter consume).
- **Design principles section** (if it exists, or add one) — synchronous pipeline discipline as documented in `cpp_coding_standard.md` and `timing_discipline.md`.

**Coding standard (`/workspace/resources/cpp_coding_standard.md`):** § Cross-stage signaling discipline: rewrite to define REGISTERED as the only forward-data flavor and document COMBINATIONAL backward as the only same-cycle classification, restricted to back-pressure / control. Update the prefix table and postfix examples. **Add the `seed_next()`/`commit()` double-buffer rule:** every pipelined stage seeds `next_* = current_*` at the top of the tick and flips back at `commit()`, so `evaluate()` is a pure function of committed state — the property a `commit()`-gate stall relies on to freeze a stage. State that `Stats` increments belong in `commit()`, never `evaluate()`/`accept()`, since `evaluate()` may re-run on a stalled cycle. Cross-reference Principle 6 in `CLAUDE.md`.

**Timing discipline doc (`/workspace/resources/timing_discipline.md`):** rewrite the COMBINATIONAL section into "COMBINATIONAL backward control"; rewrite per-boundary inventory rows 5, 7, 12, 15 (and any others affected by 10B's REGISTERED conversions and M-series memory conversions) to reflect new classifications. **Append a new inventory row for the writeback stall** — producer `WritebackArbiter::next_writeback_stall()`; consumers the five execution units and `OperandCollector` (read in `commit()` — the stall gates the register latch) and `WarpScheduler` (read in `evaluate()` — the stall gates issue); COMBINATIONAL on the cycle axis, back-pressure on the direction axis, tick-order constraint "arbiter evaluates first." Add a Phase 10A-G section to the Phasing reference summarizing each subphase's commit boundary and cycle delta. The memory plan's M1-M6 phases get their own subsection. Add a per-stage "scheduler scoreboard" entry describing the bookkeeping that replaces the polling paths, and note that the writeback stall freezes that bookkeeping (the bitmap head does not advance on a stalled cycle). **Add the four-category signal taxonomy** that the writeback stall depends on: (1) **boundary I/O registers** — stage inputs/outputs, double-buffered, frozen by the `commit()`-gate; (2) **internal *multi-cycle* carry-forward state** — iterative-unit counters and busy flags, in-flight payloads that genuinely span cycles, the multiply pipeline deque — explicitly `seed_next()`'d so `evaluate()` is re-runnable, frozen by the `commit()`-gate (a *single-cycle* unit's execution slot — the ALU's `has_pending_`/`pending_input_` — is **not** category 2: it spans no cycle boundary, so it is category-1 boundary I/O recomputed by `evaluate()` and is not seed_next'd); (3) **non-hardware sim artifacts** — `Stats` counters — not double-buffered, applied as a `commit()`-phase side effect so a re-evaluated stalled cycle counts once; (4) **deliberately-ungated control state** — the ALU's `branch_resolved_` bit — single-buffered, updates even during a stall, by design. Categories 2 and 3 are established by Phase 10B.0.5; category 4 by 10B.3.

**Performance simulator architecture (`/workspace/resources/perf_sim_arch.md`):**
- Update `WarpScheduler` description with the new fields (`unit_busy_[]`, `writeback_bitmap_`, `bitmap_head_`, `ldst_issued_total_`, `opcoll_cooldown_cycles_`) and the issue gate (structural-hazard countdown for non-pipelined units + writeback-slot bitmap + event-driven LDST FIFO-occupancy accounting).
- Update `WritebackArbiter` description to reflect fixed-priority arbitration (loads/variable-latency sources before fixed-latency units) and the combinational-backward `next_writeback_stall()` signal — no more round-robin source iteration.
- Update `ALUUnit` description to mention branch resolution responsibility (post-10A).
- Update `OperandCollector` description to drop redirect machinery (deleted in 10A).
- Update `L1Cache` description to mention the M3-refactor valid/ready handshake (`next_cmd_ready()`, repurposed `next_cmd_stall_reason()`) and the cmd/response slot model.
- Update `ExternalMemoryInterface` description to mention `next_request_stall()` and the REGISTERED request slots (M5).
- Update `CoalescingUnit` description to reflect REGISTERED FIFO read (M1) and REGISTERED command submission (M3).
- Update `LoadGatherBufferFile` description to mention REGISTERED claim (M2) and `current_has_result` (M4).
- Update the execution-unit (`ALUUnit`, `MultiplyUnit`, `DivideUnit`, `TLookupUnit`, `LdStUnit`) and `OperandCollector` descriptions to note the explicit `seed_next()`/`commit()` double-buffering (10B.0.5), the pull-model `accept()` driven from the consumer's own `evaluate()` (10B.1/10B.2), and the writeback-stall `commit()`-gate (10B.3).

**Trace and performance counters (`/workspace/resources/trace_and_perf_counters.md`):** add the new `Stats` fields:
- `fixed_writeback_preempted_cycles` (10B.3) — cycles a fixed-latency writeback was held off because a load won the port
- `scheduler_unit_busy_stall_cycles[ExecUnit]` (10B.0; per-unit array — only `DIVIDE`/`TLOOKUP` ever nonzero)
- `scheduler_writeback_contention_stall_cycles[ExecUnit]` (10B.0; per-unit array)
- `scheduler_ldst_fifo_full_stall_cycles` (10B.0)

The existing `warp_stall_unit_busy[w]` counter is deprecated; replaced by the finer-grained per-reason counters above. Decide whether to remove it or keep it as a roll-up. Document the decision in the change. The arbiter's `writeback_conflicts` counter is removed in 10B.3 — under fixed-priority arbitration there is no round-robin conflict to count; `fixed_writeback_preempted_cycles` is its semantic successor.

**Onboarding (`/workspace/resources/onboarding.md`):** brief update to point new contributors at the canonical synchronous-pipeline discipline (Principle 6 in `CLAUDE.md`, full rules in `timing_discipline.md`). Mention the scheduler bookkeeping pattern as the answer to "how does back-pressure work?"

**`AGENTS.md` Key References:** no new entries (`/AGENTS.md` is a symlink to `CLAUDE.md`, so the Principle 6 update there is already in place from the prior commit).

**Test file accessor migration:** already complete. Production-code accessor renames (`next_has_result` / `next_output` → `current_*`) migrated their test call sites (`test_timing_components.cpp`, `test_warp_scheduler.cpp`, `test_branch.cpp`) inside the same atomic commit as the rename (10B.1–10B.3, 10E), per the per-phase test-repair discipline. 10F's only test-file change is `test_signal_diagram.py` (listed under Tooling above). Memory-related test accessor migrations (`test_load_gather_buffer.cpp`, `test_cache.cpp`) belonged to the memory plan and already landed.

### Phase 10G — Final verification, loose-bound tightening, and benchmark snapshot

Exact-match cycle assertions and accessor-rename compilation fixes were already handled inline, phase-by-phase, under the per-phase test-repair discipline. 10G is the closing sweep: tighten the loose bounds, confirm the whole suite against final HEAD, and capture the cumulative benchmark delta.

- **Tighten loose-bound ceilings.** The generous `cycles < N` bumps in `test_integration` / `test_panic` and the workload-benchmark `max_cycles` budgets that earlier phases bumped to survive intermediate cycle deltas get re-derived to precise post-Phase-10 values, each with a brief comment of the final budget's basis.
- **Full-suite audit against final HEAD.** Re-run `ctest` and confirm every exact-match assertion recalibrated inline by an earlier phase still holds — this catches an assertion that a *later* phase shifted again but whose earlier fixer did not revisit. Re-derive any that drifted, across `test_panic.cpp`, `test_timing_components.cpp`, `test_branch.cpp`, `test_warp_scheduler.cpp`, `test_integration.cpp`.
- Run `python3 tools/bench_compare.py --baseline <pre-phase-10-ref>` (the original pre-memory baseline) and capture the cumulative per-benchmark delta from end-of-Phase-9 to end-of-Phase-10. Document the deltas in the Phase 10 summary in `timing_discipline.md`.
- Update `/workspace/UNTESTED.md` if any deferred test coverage emerges.
- Build with `-DGPU_SIM_USE_DRAMSIM3=ON` and rerun the regression to validate the cache↔mem_if carve-out under DRAMSim3 holds through both plans.

## Critical files

Source:
- `/workspace/sim/include/gpu_sim/timing/alu_unit.h`, `/workspace/sim/src/timing/alu_unit.cpp` (10A, 10B.0.5, 10B.1, 10B.3, 10E)
- `/workspace/sim/include/gpu_sim/timing/multiply_unit.h`, `divide_unit.h`, `tlookup_unit.h`, `ldst_unit.h` + `/workspace/sim/src/timing/multiply_unit.cpp`, `divide_unit.cpp`, `tlookup_unit.cpp`, `ldst_unit.cpp` (10B.0.5, 10B.1, 10B.3 — `ldst_unit` is a full participant in 10B.0.5; it uses the same in-place convention as the others, confirmed against the post-memory baseline)
- `/workspace/sim/include/gpu_sim/timing/operand_collector.h`, `/workspace/sim/src/timing/operand_collector.cpp` (10A, 10B.0.5, 10B.1, 10B.2, 10B.3)
- `/workspace/sim/include/gpu_sim/timing/warp_scheduler.h`, `/workspace/sim/src/timing/warp_scheduler.cpp` (10B.0, 10B.2, 10B.3, 10E)
- `/workspace/sim/include/gpu_sim/timing/writeback_arbiter.h`, `/workspace/sim/src/timing/writeback_arbiter.cpp` (10B.3 — coordinate with memory plan)
- `/workspace/sim/src/timing/fetch_stage.cpp`, `.h` (10A, 10E)
- `/workspace/sim/src/timing/decode_stage.cpp`, `.h` (10A, 10E)
- `/workspace/sim/src/timing/timing_model.cpp` (10A, 10B all, 10D, 10E — coordinate with memory plan)
- `/workspace/sim/include/gpu_sim/timing/execution_unit.h` (10A — host the lifted `RedirectRequest`; 10B.0.5 — add `seed_next()` to the `ExecutionUnit` interface)
- `/workspace/sim/include/gpu_sim/timing/branch_shadow_tracker.h` (10A — `note_resolved_correctly` writer migrates from opcoll to ALU)

Tooling:
- `/workspace/tools/lint_timing_naming.py` (10F — **major work:** net-new libclang cross-module-read analysis pass with MODULE_ORDER-based direction inference; the largest single 10F item, not a documentation afterthought)
- `/workspace/tools/diagram_extract_ast.py` (10F — `MODULE_ORDER` factored into a shared module the lint also imports)
- `/workspace/tools/diagram_extract_md.py` (10F)
- `/workspace/tools/render_signal_diagram.py` (10F validation)

Tests:
- `/workspace/sim/tests/test_branch.cpp`, `test_timing_components.cpp` (10A wiring; accessor migration + cycle recalibration inline with 10B.1–10B.3 / 10E)
- `/workspace/sim/tests/test_warp_scheduler.cpp` (test-hook migration in 10B.0; accessor migration inline with 10B.2)
- All cycle-asserting tests — recalibrated inline by each cycle-changing phase (10B.*, 10E); final loose-bound tightening + full-suite audit in 10G
- `/workspace/tests/test_signal_diagram.py` (10F)

Documentation:
- `/workspace/resources/cpp_coding_standard.md` (10F)
- `/workspace/resources/timing_discipline.md` (10F — major rewrite, includes memory plan's phases in its summary)
- `/workspace/resources/perf_sim_arch.md` (10F)
- `/workspace/UNTESTED.md` (10G)

## Reused functions and patterns

- `Scoreboard` / `BranchShadowTracker` (`/workspace/sim/include/gpu_sim/timing/scoreboard.h`, `branch_shadow_tracker.h`) — canonical `seed_next()`/`commit()` double-buffer reference; 10B.0.5 converts every issue/execute stage (the five units + `OperandCollector`) to this explicit shape so the writeback stall can freeze them by gating `commit()` alone.
- `BranchShadowTracker::note_branch_issued/note_resolved_correctly/note_redirect_applied` — already correctly REGISTERED; only the location of `note_resolved_correctly` writer changes (opcoll → ALU) in 10A.
- `OperandCollector::current_redirect_request_or_override` pattern (`/workspace/sim/include/gpu_sim/timing/operand_collector.h:64-68`) — moved verbatim onto `ALUUnit` in 10A so `tools/diagram_extract_ast.py` keeps a statically resolvable receiver.
- `python3 tools/bench_compare.py --baseline <git-ref>` — A/B benchmark snapshot at every phase boundary that changes cycle counts (10B.1, 10B.2, 10B.3, 10E) and at every byte-identical boundary as a zero-delta check (10A, 10B.0.5, 10D).
- `bash ./tests/run_workload_benchmarks.sh --build-dir build` — canonical workload-benchmark entry point.

## Verification

Per phase:

1. `cmake --build build -j8` succeeds.
2. `cd build && ctest --output-on-failure` passes.
3. For phases that should be cycle-byte-identical (10A, 10B.0.5, 10D): `python3 tools/bench_compare.py --baseline <phase-start-ref>` reports zero delta on every workload benchmark. Any non-zero delta indicates a hidden ordering dependency (or, for 10B.0.5, a missed `seed_next()` field or a mis-relocated stat) — investigate before commit.
4. For phases that intentionally regress cycle counts (10B.1–10B.3, 10E): capture per-benchmark delta from `bench_compare.py`, record in commit message and in `timing_discipline.md` Phase 10 summary.
5. `python3 tools/render_signal_diagram.py --validate` passes (AST and markdown extractors agree on every edge).
6. `python3 tools/lint_timing_naming.py` passes (after 10F it should pass strictly with no `--report-only`).

End-to-end:

- Run the full RISC-V ISA compliance suite (`/workspace/tests/riscv-isa/`) and synthetic edge tests (`/workspace/tests/synthetic/`) at end of Phase 10G — both must be green.
- Generate a signal diagram (`python3 tools/render_signal_diagram.py --output /tmp/diagram.svg`) and visually inspect: forward edges should all be the REGISTERED line style; only back-pressure / branch-redirect edges should be COMBINATIONAL.
