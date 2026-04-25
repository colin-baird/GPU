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
| 3 | `OperandCollector::is_free()` | `WarpScheduler` (via `set_opcoll_free` setter called before `scheduler.evaluate`) | `bool` opcoll-free flag | should be **READY/STALL** via `opcoll.ready_out()` | currently pre-evaluate setter | non-compliant | 4 |
| 4 | Each `ExecutionUnit::is_ready()` | `WarpScheduler` (via `unit_ready_fn_` callback queried during `scheduler.evaluate`) | per-unit ready bit | should be **READY/STALL** via `unit.ready_out()` | callback reads live state mid-tick | non-compliant | 4 |
| 5 | `WarpScheduler` issue (sets `warps_[w].branch_in_flight = true` mid-evaluate) and `OperandCollector` branch resolution (clears `warps_[w].branch_in_flight = false` via direct mutation in `timing_model.cpp` ~line 403) | `WarpScheduler::evaluate` (next cycle) reads `warps_[w].branch_in_flight` | per-warp branch-shadow bit | should be **REGISTERED** per-warp pair (`next_branch_in_flight_[w]` / `current_branch_in_flight_[w]`) | mid-tick mutation of shared `WarpState` field | non-compliant | 5 |
| 6 | `OperandCollector::accept()` writes only `next_busy_`/`next_cycles_remaining_`/`next_instr_` | `OperandCollector::evaluate` (reads `next_*` after the prior commit, so equal to committed values until accept overrides) and `OperandCollector::is_free()` (reads `current_busy_`) | opcoll busy/cycles/instr | **REGISTERED** next/current | `accept()` only mutates `next_*`; `commit()` flips next/current for busy, cycles_remaining, instr, and output | compliant (Phase 2) | 2 |
| 7 | `ALUUnit`/`MultiplyUnit`/`DivideUnit`/`TLookupUnit`/`LdStUnit::accept()` (writes only `next_*`); `WritebackArbiter::evaluate` calls `consume_result()` which writes `next_result_buffer_.valid = false` | downstream `evaluate()` (writeback arbiter sees `has_result()` reading live `next_*` for the COMBINATIONAL same-tick edge); `is_ready()` consumed by scheduler reads `current_*` (committed) | per-unit `pending_input_`/`pending_result_`/`pending_entry_`, `pipeline_`, `addr_gen_fifo_`, `result_buffer_` | mixed: cross-cycle state is **REGISTERED** (`next_*`/`current_*` with `commit()` flip); the wb-arbiter and ldst→coalescing edges are **COMBINATIONAL** (live `next_*` reads to preserve zero cycle delta) | every unit's `commit()` flips next/current for all double-buffered fields; `accept()` and `consume_result()` write only `next_*` | compliant (Phase 1) | 1 |
| 8 | `WritebackArbiter::evaluate` writes scoreboard via `scoreboard_.clear_pending()` | scheduler's next-cycle `evaluate()` reads scoreboard | scoreboard pending bits | **REGISTERED** | `Scoreboard` already exposes `next_*`/`current_*`, so `clear_pending()` only writes `next_` | compliant | none |
| 9 | `CoalescingUnit::evaluate` calls `ldst_.fifo_pop()`, `gather_file_.claim()`, `cache_.process_load()`/`process_store()` mid-evaluate | LdStUnit FIFO, gather buffer file, cache | command-style mutations | documented as "internal-subsystem-mutating commands"; cache is treated as a trusted internal subsystem | unchanged at the cache boundary; gather-buffer write-port arbitration cleaned up | partially compliant | 7 (boundaries only — gather-buffer port arbitration) |
| 10 | `L1Cache` internal members (`tags_`, `mshrs_`, `write_buffer_`, `pending_fill_`) | itself; cache test expectations | direct mutation of cache state | trusted internal subsystem (double-buffering descoped — `test_cache.cpp` and `test_cache_mshr_merging.cpp` assert MSHR/tag state synchronously) | compliant by design | none |
| 11 | `LoadGatherBufferFile::try_write` (sets `port_used_this_cycle = true`); `L1Cache` (sets `gather_extract_port_used_ = true` for FILL/secondary/HIT priority) | each other within the same cycle | intra-cycle write-port arbitration scratch flags | **COMBINATIONAL** (FILL > secondary drain > HIT priority is enforced by tick ordering) | semantically correct but encoded as plain mutable booleans | partially compliant | 7 (cleanup; preserve semantics) |
| 12 | Branch misprediction in `timing_model.cpp` (`fetch_->redirect_warp(...)`, `decode_->invalidate_warp(...)` called mid-tick after `dispatch_to_unit`) | `FetchStage`, `DecodeStage` | flush request and redirect target PC | should be **REGISTERED** redirect-request signal (`next_redirect_request_{valid, warp_id, target_pc}` produced by opcoll, consumed by fetch/decode in their own `commit()`) | side-channel calls bypass commit entirely | non-compliant | 5 |
| 13 | `DecodeStage::ebreak_detected()` and the cascade `panic_->trigger(...)` + `scheduler/opcoll/gather_file/wb_arbiter->reset()` mid-tick from `timing_model.cpp` | `PanicController`, scheduler, opcoll, gather buffer, writeback arbiter | EBREAK request and machine flush | should be **REGISTERED** `next_ebreak_request_{valid, warp_id, pc}`; per-stage `flush()` invoked at commit when the panic signal becomes active | mid-tick `reset()` cascade is the only path | non-compliant | 6 |
| 14 | `TimingModel::execution_units_drained()` (queries each unit's live state) | `PanicController` (via `set_units_drained` setter called pre-`panic.evaluate`) | drained bit | should be **READY/STALL** via per-unit `ready_out()` queries inside `PanicController::compute_ready()` | currently pre-evaluate setter aggregating live state | non-compliant | 6 |

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
- **Phase 4**: Scheduler ready signals. Remove `set_opcoll_free` and
  `unit_ready_fn_`; add `ready_out()` on opcoll and each unit.
  Inventory rows: 3, 4.
- **Phase 5**: Branch redirect and `branch_in_flight` as REGISTERED state.
  Per-warp next/current pair; opcoll publishes a `next_redirect_request_`
  signal; fetch/decode flush at their own commit. Inventory rows: 5, 12.
- **Phase 6**: Panic / EBREAK side-channel. Decode publishes a REGISTERED
  `next_ebreak_request_`; `PanicController::compute_ready()` queries unit
  ready signals; `reset()` cascade replaced by per-stage `flush()` at
  commit. Inventory rows: 13, 14.
- **Phase 7**: Gather-buffer port arbitration cleanup. `claim()` writes
  `next_*`; `commit()` flips. Cache internals stay direct-mutation by
  design. Inventory rows: 9 (boundary only), 11.
- **Phase 8**: Lift `compute_ready()` into `PipelineStage` base; formalize
  the backward sweep in `tick()`. Done last so every stage already
  exposes its own `ready_out()` before the base-class hook is added.
