# Cross-Stage Signaling Discipline (Timing Model)

## Purpose

The cycle-accurate timing model in `sim/` uses a two-phase `evaluate()` /
`commit()` pattern intended to mimic synchronous-logic hardware: every stage's
`evaluate()` should compute a *new* state (`next_*`) from the *committed*
state (`current_*`), and `commit()` flips `next_* → current_*` at the cycle
boundary. When stages instead read mutable members that other stages write
mid-evaluate, correctness silently depends on the order in which `tick()`
calls those stages — and that fragility produces real bugs. The recently
fixed fetch/decode reorder (commit `0383f04`) is the canonical example: the
fetch stage read `output_consumed_` before decode mutated it, capping
frontend throughput at 1 instr per 2 cycles. This document classifies every
cross-stage signal in the timing model so that future stages can be added
(or reordered) without re-discovering hazards of this kind.

## Tick discipline

Each non-panic tick of `TimingModel::tick()` runs three phases in this
fixed order. The structure is documented in `tick()` itself (see
`sim/src/timing/timing_model.cpp`); the summary below is the contract
every new stage must respect.

1. **compute_ready phase (backward sweep).** Every consumer that exposes
   a READY/STALL signal computes its `ready_out_` from committed
   (`current_*`) state only. Today's participants:
   `OperandCollector`, `ALUUnit`, `MultiplyUnit`, `DivideUnit`,
   `TLookupUnit`, `LdStUnit`, `DecodeStage`. Stages without a ready
   output (`FetchStage`, `WarpScheduler`, `WritebackArbiter`,
   `CoalescingUnit`, `MemoryInterface`, `L1Cache`, `PanicController`)
   inherit the default no-op from `PipelineStage::compute_ready()`
   (formalized in Phase 8) or from the parallel
   `ExecutionUnit::compute_ready()` virtual.

2. **evaluate phase (forward sweep, dataflow order).** Each stage reads
   its inputs (committed state for REGISTERED edges, the live
   `next_*` / `ready_out_` of upstream for COMBINATIONAL or READY/STALL
   edges) and writes its own `next_*` slot. Order:
   `cache -> fetch -> decode -> scheduler -> opcoll.accept -> opcoll
   -> dispatch -> alu/mul/div/tlookup/ldst -> coalescing -> mem_if
   -> cache.drain_write_buffer -> wb_arbiter`.

3. **commit phase.** Every stage flips `next_* -> current_*`. Stages
   are independent at commit time, so the order is for traceability
   only: `fetch, decode, scheduler, opcoll, units, coalescing, cache,
   mem_if, wb_arbiter, gather_file, scoreboard, branch_tracker`,
   followed by the optional Phase 6 panic-flush cascade if armed at
   top-of-tick.

`PipelineStage` (`sim/include/gpu_sim/timing/pipeline_stage.h`)
formally exposes all four methods — `compute_ready`, `evaluate`,
`commit`, `reset` — as part of the discipline contract. `compute_ready`
defaults to a virtual no-op so stages without a READY/STALL output do
not need an empty override. `ExecutionUnit` is a separate hierarchy
(units have a different lifecycle: they produce results consumed by
`WritebackArbiter` rather than participating in the unified
evaluate/commit fan-in) but shares the same four-method convention via
its own `compute_ready` virtual default.

## Signal classifications

Every cross-stage signal in the timing model belongs to exactly one of
these three classes, and the classification must be declared at the call
site.

### REGISTERED

The producer writes its `next_*` slot during `evaluate()`. `commit()`
latches `next_* → current_*`. The consumer reads the producer's
`current_*` only — never `next_*`. Result: a guaranteed 1-cycle handshake,
independent of `tick()` ordering.

```
cycle N    : producer.evaluate()  -> writes producer.next_X
           : consumer.evaluate()  -> reads producer.current_X (still old)
           : producer.commit()    -> next_X → current_X
cycle N+1  : consumer.evaluate()  -> reads producer.current_X (now new)
```

This is the default for any state whose effect should reach the consumer
on a later cycle: PCs, scoreboard bits, `branch_in_flight`, opcoll
busy/cycles, execution-unit `result_buffer_`, `instr_buffer` entries.

### COMBINATIONAL

The producer's `next_*` (or a freshly-computed transient) is read by the
consumer in the *same* `evaluate()` phase. The order of calls in
`tick()` is part of the contract and must be documented at every call
site that depends on it. This is used where the design intends zero-cycle
fanout — for example, the scheduler issue → opcoll accept → unit accept
chain inside a single tick, or the cache fill → gather-buffer write port
arbitration.

```
cycle N    : producer.evaluate()  -> writes producer.next_X
           : consumer.evaluate()  -> reads producer.next_X this cycle
                                    (depends on producer running first)
```

A COMBINATIONAL edge requires a comment at the call site identifying the
producer, the consumer, and why ordering matters.

### READY/STALL

The consumer exposes a `ready_out()` method computed in a
`compute_ready()` step that reads only its own `current_*`. The producer
calls `consumer.ready_out()` during its own `evaluate()` and uses the
result to decide whether to advance or hold. Modeling backpressure this
way matches a real ready/valid handshake: the stall signal flows
backward in the same cycle, but the underlying state from which it was
computed is committed.

```
cycle N    : consumer.compute_ready() -> reads only consumer.current_*
                                          -> exposes consumer.ready_out()
           : producer.evaluate()       -> reads consumer.ready_out()
```

Used for: decode → fetch backpressure, opcoll → scheduler backpressure,
unit → scheduler backpressure, gather-buffer write-port reservation.

## Reference implementation

`Scoreboard` (`sim/include/gpu_sim/timing/scoreboard.h`) is the gold
standard for the REGISTERED pattern. Every new next/current pair in the
timing model should follow this shape verbatim:

```cpp
class Scoreboard {
public:
    // Reads of committed state.
    bool is_pending(WarpId warp, RegIndex reg) const {
        if (reg == 0) return false;
        return current_[warp][reg];
    }

    // Writes go to next_ only.
    void set_pending(WarpId warp, RegIndex reg) {
        if (reg == 0) return;
        next_[warp][reg] = true;
    }
    void clear_pending(WarpId warp, RegIndex reg) {
        if (reg == 0) return;
        next_[warp][reg] = false;
    }

    void seed_next() { std::memcpy(next_, current_, sizeof(next_)); }
    void commit()    { std::memcpy(current_, next_, sizeof(current_)); }

private:
    bool current_[MAX_WARPS][NUM_REGS];
    bool next_[MAX_WARPS][NUM_REGS];
};
```

Key invariants:

- All reads return `current_*`.
- All writes go to `next_*`.
- `seed_next()` is called at the top of every cycle before any writes,
  so unmodified bits carry forward.
- `commit()` is the sole operation that flips state.

`FetchStage::next_output_`/`current_output_` and
`WarpScheduler::next_diagnostics_`/`current_diagnostics_` are existing
precedents for double-buffering an `std::optional` payload and a
fixed-size array, respectively. New REGISTERED signals introduced by the
phasing below reuse those idioms.

## Per-boundary inventory

The table below catalogs every cross-stage edge currently observed in the
timing model. "Compliant" means the signal already follows its
classification; "non-compliant" means the refactor phase listed in the
last column is responsible for fixing it.

| # | Producer | Consumer | Payload | Classification | Tick-order constraint | Current state | Refactor phase |
|---|---|---|---|---|---|---|---|
| 1 | `DecodeStage::ready_to_consume_fetch()` (computed in `compute_ready()`) | `FetchStage::evaluate` | `bool` decode-ready signal | **READY/STALL** | tick order: `decode_.compute_ready` -> `fetch_.evaluate` -> `decode_.evaluate` | compliant (Phase 3) | 3 |
| 2 | `DecodeStage::pending_warp()` | `FetchStage::evaluate` (direct accessor read; the `set_decode_pending_warp` setter has been deleted) | optional warp id of decode's pending entry | **READY/STALL** | same tick order as row 1; fetch holds a `DecodeStage*` wired by `TimingModel` | compliant (Phase 3) | 3 |
| 3 | `OperandCollector::ready_out()` (computed in `compute_ready()` from `current_busy_`) | `WarpScheduler::evaluate` (reads `opcoll_->ready_out()` directly via wired pointer) | `bool` opcoll-free flag | **READY/STALL** | tick order: `opcoll_.compute_ready` -> `scheduler_.evaluate`; `set_opcoll_free` setter deleted | compliant (Phase 4) | 4 |
| 4 | Each `ExecutionUnit::ready_out()` (added to base interface; computed in each unit's `compute_ready()` from committed state) | `WarpScheduler::evaluate` (reads `unit->ready_out()` via typed pointers wired by `set_consumers`) | per-unit ready bit | **READY/STALL** | tick order: each unit's `compute_ready` -> `scheduler_.evaluate`; `unit_ready_fn_` callback deleted | compliant (Phase 4) | 4 |
| 5 | `WarpScheduler::evaluate` writes `branch_tracker_.set_in_flight(w)` (next_); `OperandCollector::resolve_branch` clears (next_) on correct prediction; `FetchStage::commit` clears (next_) when applying a redirect | `WarpScheduler::evaluate` reads `branch_tracker_.is_in_flight(w)` (current_) | per-warp branch-shadow bit | **REGISTERED** per-warp `current_`/`next_` pair via `BranchShadowTracker` (Scoreboard pattern) | tick-order: `branch_tracker_.seed_next` -> writers in evaluates -> commits -> `branch_tracker_.commit` | compliant (Phase 5) | 5 |
| 6 | `OperandCollector::accept()` writes only `next_busy_`/`next_cycles_remaining_`/`next_instr_` | `OperandCollector::evaluate` (reads `next_*` after the prior commit, so equal to committed values until accept overrides) and `OperandCollector::is_free()` (reads `current_busy_`) | opcoll busy/cycles/instr | **REGISTERED** next/current | `accept()` only mutates `next_*`; `commit()` flips next/current for busy, cycles_remaining, instr, and output | compliant (Phase 2) | 2 |
| 7 | `ALUUnit`/`MultiplyUnit`/`DivideUnit`/`TLookupUnit`/`LdStUnit::accept()` (writes only `next_*`); `WritebackArbiter::evaluate` calls `consume_result()` which writes `next_result_buffer_.valid = false` | downstream `evaluate()` (writeback arbiter sees `has_result()` reading live `next_*` for the COMBINATIONAL same-tick edge); `is_ready()` consumed by scheduler reads `current_*` (committed) | per-unit `pending_input_`/`pending_result_`/`pending_entry_`, `pipeline_`, `addr_gen_fifo_`, `result_buffer_` | mixed: cross-cycle state is **REGISTERED** (`next_*`/`current_*` with `commit()` flip); the wb-arbiter and ldst→coalescing edges are **COMBINATIONAL** (live `next_*` reads to preserve zero cycle delta) | every unit's `commit()` flips next/current for all double-buffered fields; `accept()` and `consume_result()` write only `next_*` | compliant (Phase 1) | 1 |
| 8 | `WritebackArbiter::evaluate` writes scoreboard via `scoreboard_.clear_pending()` | scheduler's next-cycle `evaluate()` reads scoreboard | scoreboard pending bits | **REGISTERED** | `Scoreboard` already exposes `next_*`/`current_*`, so `clear_pending()` only writes `next_` | compliant | none |
| 9 | `CoalescingUnit::evaluate` calls `ldst_.fifo_pop()`, `gather_file_.claim()`, `cache_.process_load()`/`process_store()` mid-evaluate | LdStUnit FIFO, gather buffer file, cache | command-style mutations | documented as "internal-subsystem-mutating commands"; cache is treated as a trusted internal subsystem; the gather-buffer write-port boundary now arbitrates via REGISTERED state owned by `LoadGatherBufferFile` (see row 11) | gather-buffer write-port arbitration cleaned up; cache internals intentionally remain direct-mutation per row 10 | compliant (Phase 7) | 7 |
| 10 | `L1Cache` internal members (`tags_`, `mshrs_`, `write_buffer_`, `pending_fill_`) | itself; cache test expectations | direct mutation of cache state | trusted internal subsystem (double-buffering descoped — `test_cache.cpp` and `test_cache_mshr_merging.cpp` assert MSHR/tag state synchronously) | compliant by design | none |
| 11 | `LoadGatherBufferFile::try_write` writes only `next_port_claimed_` (single shared flag, not per-buffer; models §5.3 "one line-to-gather-buffer extraction per cycle"). `L1Cache::handle_responses`, `drain_secondary_chain_head`, and `process_load` HIT path all funnel through `try_write`; the cache-side `gather_extract_port_used_` scratch flag is removed. | `LoadGatherBufferFile::try_write` reads the live `next_port_claimed_` (combinational first-writer-wins); external observers read the post-commit `current_port_claimed_` accessor | intra-cycle write-port arbitration | **REGISTERED** (`next_port_claimed_` / `current_port_claimed_` pair owned by `LoadGatherBufferFile`; `commit()` flips next -> current). FILL > secondary > HIT priority is encoded by tick ordering: `cache_->evaluate()` runs at the top of the non-panic tick (FILL via `handle_responses`, secondary via `drain_secondary_chain_head`); `coalescing_->evaluate()` runs later in the tick (HIT via `process_load`). | tick order: `gather_file.commit` flips next -> current at end-of-cycle; first writer in tick N+1 sees `next_port_claimed_ == false` and wins | compliant (Phase 7) | 7 |
| 12 | `OperandCollector::resolve_branch` writes `next_redirect_request_{valid, warp_id, target_pc}` on misprediction | `FetchStage::commit` and `DecodeStage::commit` read `opcoll.current_redirect_request()` and apply the flush there | flush request and redirect target PC | **REGISTERED** redirect-request via `RedirectRequest` produced by opcoll, consumed by fetch/decode at their own `commit()`. Mispredict-recovery now takes one additional cycle (Option A) | tick-order: producer writes `next_` during evaluate; opcoll.commit flips to `current_` at end of cycle N; fetch.commit and decode.commit read `current_` at cycle N+1 (their commits run before opcoll.commit within the same cycle, so they observe last cycle's latched signal) | compliant (Phase 5) | 5 |
| 13 | `DecodeStage::current_ebreak_request()` (REGISTERED `next_`/`current_` pair, latched by `decode.commit()`); panic-flush cascade `scheduler/opcoll/gather_file/wb_arbiter->flush()` invoked at commit-phase boundary when `pending_panic_flush_` is armed | `TimingModel::tick()` (observes `current_ebreak_request()` at top of cycle to call `panic_->trigger`); scheduler/opcoll/gather buffer/writeback arbiter (consume `flush()` at commit-phase) | EBREAK request and machine flush | **REGISTERED** ebreak side-channel; per-stage `flush()` at commit-phase replaces the prior mid-evaluate `reset()` cascade. Trigger takes one additional cycle vs. the pre-Phase-6 mid-tick path (Option A) | tick order: decode.commit latches ebreak request at end of cycle N; tick top of cycle N+1 reads current_ebreak_request_ and calls panic_->trigger / arms pending_panic_flush_; commit-phase of cycle N+1 invokes flush() on each panic-flush target | compliant (Phase 6) | 6 |
| 14 | `PanicController::set_drained_query` wires a callable that the controller invokes inside `evaluate()`; the callable composes `execution_units_drained()` from each unit's committed-state accessors (`is_ready`, `has_result`, `fifo_empty`, `ready_out`) | `PanicController::evaluate()` (case 2 drain step) | drained bit | wired callable; the prior `set_units_drained()` pre-evaluate setter (which latched live state from another stage) is removed | controller queries drained_query_ inside its own evaluate(); the callable reads only committed-state accessors | compliant (Phase 6) | 6 |

If a future change adds a new cross-stage edge, append a row here with
its classification and the phase that lands it.

## Forbidden patterns

The following are explicit violations of the discipline. Code review
should flag any of them when they appear in the timing model.

- **Plain mutable members read across stages mid-evaluate.** A bool/int
  written by stage A's `evaluate()` and read by stage B's `evaluate()` in
  the same tick. Use REGISTERED next/current, COMBINATIONAL with a
  call-site comment, or READY/STALL via a `ready_out()` method.
- **Pre-evaluate setters that latch live state from another stage.**
  Examples in flight: `set_opcoll_free`, `set_decode_pending_warp`,
  `set_units_drained`, `set_unit_ready_fn`. These hide ordering
  dependencies inside `tick()` and silently invert when the orchestrator
  reorders calls. Replace with `compute_ready()` + `ready_out()`.
- **`consume_*` calls that synchronously mutate the other stage's
  state.** `WritebackArbiter::consume_result` flipping
  `unit.result_buffer_.valid = false` is the canonical case. Mutate only
  the unit's `next_*` slot; `commit()` performs the flip.
- **Mid-tick mutations of committed state that bypass `commit()`.**
  `fetch_->redirect_warp`, `decode_->invalidate_warp`, the panic
  `reset()` cascade, and direct writes to `warps_[w].branch_in_flight`
  fall under this rule. Express the request as a REGISTERED signal and
  let each stage flush at its own commit.

## Phasing reference

The full refactor plan lives in
`/project-plans/i-want-to-build-greedy-noodle.md`. Phase summaries:

- **Phase 0** (this doc): Discipline document, per-boundary inventory,
  coding-standard pointer. No code changes.
- **Phase 1**: Execution-unit double-buffering. Add
  `next_result_buffer_`/`next_pending_input_` to ALU/MUL/DIV/TLOOKUP/LDST;
  `WritebackArbiter::consume_result` writes only `next_*`; each unit's
  `commit()` flips. Inventory rows: 7.
- **Phase 2**: `OperandCollector` double-buffering. `accept()` writes
  `next_busy_`/`next_cycles_remaining_`/`next_current_instr_`; `evaluate()`
  reads `current_*`. Inventory rows: 6.
- **Phase 3** (landed): Fetch/decode `ready_out` boundary. Replaced
  `output_consumed_` (plain bool round-trip) and `set_decode_pending_warp`
  (pre-evaluate setter) with `DecodeStage::ready_to_consume_fetch()`
  computed in `DecodeStage::compute_ready()` and a direct
  `decode->pending_warp()` accessor read by fetch. Tick order is now
  `decode_.compute_ready -> fetch_.evaluate -> decode_.evaluate`. Fetch's
  output is now strictly REGISTERED (`current_output_ = next_output_` in
  `commit()`); evaluate encodes the hold-vs-advance decision into
  `next_output_`. Inventory rows: 1, 2.
- **Phase 4** (landed): Scheduler ready signals. Removed `set_opcoll_free`
  setter and the `UnitReadyFn` callback / `unit_ready_fn_` slot. Added
  `compute_ready()` (default no-op) and pure-virtual `ready_out()` to the
  `ExecutionUnit` base interface; each of ALU/MUL/DIV/TLOOKUP/LDST and
  `OperandCollector` overrides them, writing a `ready_out_` slot from
  committed `current_*` state. `WarpScheduler::set_consumers` wires the
  opcoll plus five typed unit pointers at construction; `evaluate()` reads
  `opcoll_->ready_out()` and each unit's `ready_out()` directly. Tick order
  in `TimingModel::tick()` adds a backward sweep
  (`opcoll/alu/mul/div/tlookup/ldst/decode .compute_ready()`) before
  `fetch_->evaluate()`. Test override hooks (`set_opcoll_ready_override`,
  `set_unit_ready_override`) replace the deleted setters in
  `test_warp_scheduler.cpp`, `test_timing_components.cpp`, and
  `test_branch.cpp`. Inventory rows: 3, 4.
- **Phase 5** (landed): Branch redirect and `branch_in_flight` as REGISTERED
  state. Added `BranchShadowTracker` (Scoreboard-shape per-warp `current_`/
  `next_` pair) replacing the plain `WarpState::branch_in_flight` bool.
  Added `RedirectRequest` plus `OperandCollector::resolve_branch` /
  `current_redirect_request()` so opcoll publishes the redirect through a
  REGISTERED slot. `FetchStage::commit` and `DecodeStage::commit` now read
  the signal and apply the flush from there; the prior mid-tick
  `fetch_->redirect_warp` / `decode_->invalidate_warp` side-channel calls
  from `timing_model.cpp` are gone. Mispredict recovery takes one
  additional cycle (Option A). The `branch_in_flight` clear is split: for
  correctly-predicted branches opcoll clears immediately on resolve; for
  mispredicts opcoll defers the clear and `FetchStage::commit` clears when
  it actually applies the redirect — this prevents the scheduler from
  issuing a shadow instruction in the cycle the redirect propagates from
  opcoll to fetch. Tick order in `TimingModel::tick()` adds
  `branch_tracker_.seed_next()` near `scoreboard_.seed_next()` and
  `branch_tracker_.commit()` near `scoreboard_.commit()`. Inventory rows:
  5, 12.
- **Phase 6** (landed): Panic / EBREAK side-channel. Decode publishes a
  REGISTERED `EBreakRequest{valid, warp_id, pc}` via `next_`/`current_`
  pair; `decode.commit()` latches; `TimingModel::tick()` observes
  `current_ebreak_request()` at the top of the next tick and calls
  `panic_->trigger`. `PanicController` replaces the prior
  `set_units_drained` pre-evaluate setter with a wired callable
  (`set_drained_query`) that composes `execution_units_drained()` from
  committed-state accessors and is invoked inside the controller's
  own `evaluate()`. The mid-evaluate `reset()` cascade
  (`scheduler_/opcoll_/gather_file_/wb_arbiter_->reset()`) is replaced
  by per-stage `flush()` methods invoked at the commit-phase boundary
  when `pending_panic_flush_` is armed. Trigger takes one additional
  cycle vs. pre-Phase-6 (Option A); benchmark cycle counts unchanged
  because workloads do not invoke EBREAK (they ECALL out). Test
  assertions in `test_panic.cpp` use `< 1000` upper bounds and remain
  green; the state-machine test was migrated from `set_units_drained()`
  to the wired callable. Inventory rows: 13, 14.
- **Phase 7**: Gather-buffer port arbitration cleanup. `claim()` writes
  `next_*`; `commit()` flips. Cache internals stay direct-mutation by
  design. Inventory rows: 9 (boundary only), 11.
- **Phase 8** (landed): `PipelineStage::compute_ready()` added as a
  virtual default no-op so the four-method discipline (`compute_ready`,
  `evaluate`, `commit`, `reset`) is formally part of the base-class
  contract. `DecodeStage::compute_ready()` and
  `OperandCollector::compute_ready()` now carry `override`. The
  `ExecutionUnit::compute_ready()` parallel virtual is retained because
  units are a separate hierarchy; both hierarchies share the convention.
  A block comment in `TimingModel::tick()` formalizes the
  compute_ready / evaluate / commit phasing (see "Tick discipline"
  above). No code-flow changes; cycle counts byte-identical to Phase 7.
  Inventory: every row above is now compliant with a phase tag.
