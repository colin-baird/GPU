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

Each non-panic tick of `TimingModel::tick()` runs two phases in this
fixed order. The structure is documented in `tick()` itself (see
`sim/src/timing/timing_model.cpp`); the summary below is the contract
every new stage must respect.

1. **evaluate phase (forward sweep, dataflow order).** Each stage reads
   its inputs (committed `current_*` state for REGISTERED edges, the
   live `next_*` of upstream for COMBINATIONAL edges) and writes its
   own `next_*` slot. Order:
   `cache -> fetch -> decode -> scheduler -> opcoll.accept -> opcoll
   -> dispatch -> alu/mul/div/tlookup/ldst -> coalescing -> mem_if
   -> cache.drain_write_buffer -> wb_arbiter`.

2. **commit phase.** Every stage flips `next_* -> current_*`. Stages
   are independent at commit time, so the order is for traceability
   only: `fetch, decode, scheduler, opcoll, units, coalescing, cache,
   mem_if, wb_arbiter, gather_file, scoreboard, branch_tracker`,
   followed by the optional Phase 6 panic-flush cascade if armed at
   top-of-tick.

`PipelineStage` (`sim/include/gpu_sim/timing/pipeline_stage.h`)
formally exposes three methods — `evaluate`, `commit`, `reset` — as
part of the discipline contract. Back-pressure signals are exposed as
`const` accessors that read only the producer's own committed
(`current_*`) state, so the value is stable across the entire evaluate
phase regardless of where it is queried; there is no separate
backward-sweep phase. `ExecutionUnit` is a separate hierarchy (units
have a different lifecycle: they produce results consumed by
`WritebackArbiter` rather than participating in the unified
evaluate/commit fan-in) and shares the same convention.

## Signal classifications

Every cross-stage signal in the timing model is classified along **two
independent axes**.

### Cycle axis — when is the read sampled?

Two values, mechanically distinguishable by the accessor's name prefix:

#### REGISTERED (`current_*`)

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
busy/cycles, execution-unit `result_buffer_`, `instr_buffer` entries,
back-pressure `current_busy()` signals.

Accessors that return committed state are named `current_<noun>()` or
`current_<adjective>()` per the postfix design language below.

#### COMBINATIONAL (`next_*`)

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

Accessors that return live mid-tick state are named `next_<noun>()` or
`next_<adjective>()`.

### Direction axis — what does the edge mean architecturally?

This axis is **orthogonal to the cycle axis** and is *architectural
metadata*, not a signal classification. It controls visualization style
in the rendered diagram but does not affect the read pattern at the
call site.

- **forward-data** — the consumer reads payload (instructions, results,
  trace data) flowing along the natural data direction
  upstream→downstream. Examples: fetch → decode `current_output()`,
  opcoll → unit `current_redirect_request()`, alu → wb_arbiter
  `next_has_result()`.
- **back-pressure** — the producer (the upstream stage in the dataflow)
  reads a downstream consumer's "are you ready?" signal so it can
  decide whether to advance or hold this cycle. The diagram edge
  conventionally points consumer→producer, against the data direction.
  Examples: scheduler reading `opcoll_->current_busy()`, scheduler
  reading `unit->current_busy()`, fetch reading
  `decode_->current_busy()`, coalescing reading `cache_->next_stalled()`.

Worked example (back-pressure + REGISTERED is the most common case):
`OperandCollector` exposes `current_busy()` (a `const` accessor that
reads only its own `current_busy_` slot). The scheduler reads it during
its own `evaluate()`; the diagram renders the edge consumer→producer
because the scheduler is *downstream* of opcoll in the data flow but is
the one performing the read. The discipline is REGISTERED on the cycle
axis (committed-state read, stable across the evaluate phase) and
back-pressure on the direction axis (visualization).

`L1Cache::next_stalled()` is the orthogonal case (back-pressure +
COMBINATIONAL): coalescing reads cache's stall mid-tick, after
`cache.evaluate()` has produced this cycle's stall outcome.
Single-cycle backpressure path; carved out per row 9 / 10 below.

## Postfix design language

After the cycle prefix, every cross-stage accessor falls into one of
three grammatical shapes. The shape is derived from what the accessor
returns.

| Shape | Returns | Postfix grammar | Examples |
|-------|---------|-----------------|----------|
| **State predicate** | `bool` | `<prefix>_<adjective>` | `current_busy()`, `current_idle()`, `next_stalled()`, `current_in_flight(w)`, `current_pending(w, r)`, `next_fifo_empty()` |
| **Possession predicate** | `bool` | `<prefix>_has_<noun>` | `next_has_result()`, `next_has_response()` |
| **Payload accessor** | non-`bool` (`std::optional<T>`, payload struct, scalar id) | `<prefix>_<noun>` | `current_output()`, `current_pending_warp()`, `current_redirect_request()`, `current_ebreak_request()`, `next_fifo_front()` |

Rules:

1. **State predicates** describe a *condition* the producer is in.
   Use a bare adjective phrase, no `is_*` / `has_*` filler. Multi-word
   adjectives (`in_flight`, `fifo_empty`) split on underscores.
2. **Possession predicates** describe whether the producer *holds* a
   thing. Use the `has_<noun>` form because the question is about
   ownership, not state. Reserve this shape for accessors that are a
   precondition for a follow-up read of the actual thing
   (`if (next_has_result()) entry = consume_result();`).
3. **Payload accessors** return the thing itself. Bare noun. The
   prefix already conveys cycle discipline; the noun describes what's
   inside.
4. **Scope is carried by parameters, not name suffixes.**
   `current_busy(WarpId w)` instead of `current_busy_for_warp(w)`;
   `current_pending(WarpId w, RegIndex r)` instead of
   `current_pending_for_warp_register(w, r)`. The parameter list
   already names the scope axes.

## Polarity convention: asserted = blocking

Every state predicate returns `true` when the *condition that prevents
forward progress* is in effect. The reader writes `if (predicate)
skip;` to bail out; no negation in the common case.

| Predicate | True means |
|-----------|------------|
| `current_busy()` | producer cannot accept more work this cycle |
| `current_in_flight(w)` | warp `w`'s branch shadow blocks issue |
| `current_pending(w, r)` | warp `w`'s register `r` is reserved (scoreboard hit) |
| `next_stalled()` | producer is stalling its consumer this cycle |
| `next_fifo_empty()` | producer's FIFO is empty (consumer cannot pop) |

Possession predicates have a distinct polarity rule because they
describe ownership, not blocking: `next_has_result()` returns `true`
when the result is available. This reads naturally as
"if-then-consume" rather than "if-blocked-then-skip". The two
polarities don't conflict because the shapes are different (state
adjective vs. `has_<noun>`).

The convention eliminates inverse pairs: only one polarity exists per
concept (`current_busy` survives; `ready_out` does not).

## Reference implementation

`Scoreboard` (`sim/include/gpu_sim/timing/scoreboard.h`) is the gold
standard for the REGISTERED pattern. Every new next/current pair in the
timing model should follow this shape verbatim:

```cpp
class Scoreboard {
public:
    // Reads of committed state.
    bool current_pending(WarpId warp, RegIndex reg) const {
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

## New module checklist

A contributor adding a new timing module must satisfy each of:

1. **Cluster assignment.** Add the new class to `MODULE_CLUSTER` and
   `MODULE_ORDER` in `tools/diagram_extract_ast.py` (and the parallel
   list in `tools/diagram_extract_md.py`). Place it in dataflow order
   within its cluster.
2. **Cycle prefix on every public const accessor.** Each cross-stage
   const accessor returning `bool`, `std::optional<…>`, or a payload
   reference must be named `current_<…>()` (REGISTERED) or
   `next_<…>()` (COMBINATIONAL). Lifecycle hooks (`evaluate`, `commit`,
   `reset`, `flush`, `seed_next`, `accept`, `consume_result`,
   `add_source`, `set_*`) are exempt.
3. **Postfix shape per the design language.** State predicate (bare
   adjective), possession predicate (`has_<noun>`), or payload
   accessor (bare noun).
4. **Asserted=blocking polarity** for every state predicate.
5. **REGISTERED state must be `private`.** Expose only via
   `current_*()` / `next_*()` accessors; no public bare fields.
6. **An entry in the per-boundary inventory below** with the
   classification on each axis.

`tools/lint_timing_naming.py` enforces rules 2–5 mechanically.

## Per-boundary inventory

The table below catalogs every cross-stage edge currently observed in the
timing model. The "Cycle" column is REGISTERED or COMBINATIONAL (or a
documented mixed pair for boundaries that span multiple signals). The
"Direction" column is forward-data or back-pressure. "Compliant" means
the signal already follows its classification; "non-compliant" means
the refactor phase listed in the last column is responsible for fixing
it.

| # | Producer | Consumer | Payload | Cycle | Direction | Tick-order constraint | Current state | Refactor phase |
|---|---|---|---|---|---|---|---|---|
| 1 | `DecodeStage::current_busy()` (`const` accessor reading `pending_.valid`) plus the natural complement `FetchStage::current_output()` (REGISTERED accessor returning `current_output_`) read by `DecodeStage::evaluate` | `FetchStage::evaluate` (reads decode busy) and `DecodeStage::evaluate` (reads fetch's committed output) | `bool` decode-busy signal + REGISTERED fetch payload | **REGISTERED** for both | back-pressure (decode→fetch busy) + forward-data (fetch→decode output) | tick order: `fetch_.evaluate` -> `decode_.evaluate` (so the value fetch reads is committed-state stable; decode reads the prior cycle's `current_output_`) | compliant | 3 |
| 2 | `DecodeStage::current_pending_warp()` | `FetchStage::evaluate` (direct accessor read; the `set_decode_pending_warp` setter has been deleted) | optional warp id of decode's pending entry | **REGISTERED** | back-pressure | same tick order as row 1; fetch holds a `DecodeStage*` wired by `TimingModel` | compliant (Phase 3) | 3 |
| 3 | `OperandCollector::current_busy()` (`const` accessor reading `current_busy_`) | `WarpScheduler::evaluate` (reads `opcoll_->current_busy()` directly via wired pointer) | `bool` opcoll-busy flag | **REGISTERED** | back-pressure | the accessor reads only `current_busy_`, so its value is stable across the entire evaluate phase regardless of where the scheduler queries it; `set_opcoll_free` setter deleted | compliant | 4 |
| 4 | Each `ExecutionUnit::current_busy()` (`const` accessor on the base; each concrete unit reads its own `current_*`) | `WarpScheduler::evaluate` (reads `unit->current_busy()` via typed pointers wired by `set_dependencies`) | per-unit busy bit | **REGISTERED** | back-pressure | as row 3: each accessor reads only the unit's committed state; `unit_ready_fn_` callback deleted | compliant | 4 |
| 5 | `WarpScheduler::evaluate` writes `branch_tracker_.note_branch_issued(w)` (next_); `OperandCollector::resolve_branch` writes `note_resolved_correctly(w)` (next_) on correct prediction; `FetchStage::commit` writes `note_redirect_applied(w)` (next_) when applying a redirect | `WarpScheduler::evaluate` reads `branch_tracker_.current_in_flight(w)` (current_) | per-warp branch-shadow bit | **REGISTERED** per-warp `current_`/`next_` pair via `BranchShadowTracker` (Scoreboard pattern) | forward-data (writers publish, scheduler reads) | tick-order: `branch_tracker_.seed_next` -> writers in evaluates -> commits -> `branch_tracker_.commit` | compliant (Phase 5) | 5 |
| 6 | `OperandCollector::accept()` writes only `next_busy_`/`next_cycles_remaining_`/`next_instr_` | `OperandCollector::evaluate` (reads `next_*` after the prior commit, so equal to committed values until accept overrides) and `OperandCollector::current_busy()` (reads `current_busy_`) | opcoll busy/cycles/instr | **REGISTERED** next/current | (internal) | `accept()` only mutates `next_*`; `commit()` flips next/current for busy, cycles_remaining, instr, and output | compliant (Phase 2) | 2 |
| 7 | `ALUUnit`/`MultiplyUnit`/`DivideUnit`/`TLookupUnit`/`LdStUnit::accept()` (writes only `next_*`); `LoadGatherBufferFile` is also registered as a writeback source via `wb_arbiter_->add_source` and emits a REGISTERED writeback when all 32 slots fill; `WritebackArbiter::evaluate` calls `consume_result()` which writes `next_result_buffer_.valid = false` | downstream `evaluate()` (writeback arbiter sees `next_has_result()` reading live `next_*` for the COMBINATIONAL same-tick edge); `current_busy()` consumed by scheduler reads `current_*` (committed) | per-unit `pending_input_`/`pending_result_`/`pending_entry_`, `pipeline_`, `addr_gen_fifo_`, `result_buffer_` | mixed: cross-cycle state is **REGISTERED** (`next_*`/`current_*` with `commit()` flip); the wb-arbiter and ldst→coalescing edges are **COMBINATIONAL** (live `next_*` reads to preserve zero cycle delta) | forward-data (units→arbiter, ldst→coalescing) | every unit's `commit()` flips next/current for all double-buffered fields; `accept()` and `consume_result()` write only `next_*` | compliant (Phase 1) | 1 |
| 8 | `WritebackArbiter::evaluate` writes scoreboard via `scoreboard_.clear_pending()` (releases destinations); `WarpScheduler::evaluate` writes via `scoreboard_.set_pending()` (claims destinations on issue) | scheduler's `evaluate()` reads scoreboard via `current_pending()` (current_) for issue gating | scoreboard pending bits | **REGISTERED** | forward-data | `Scoreboard` already exposes `next_*`/`current_*`, so `set_pending` / `clear_pending` only write `next_` | compliant | none |
| 9 | `CoalescingUnit::evaluate` calls `ldst_.fifo_pop()`, `gather_file_.claim()`, `cache_.process_load()`/`process_store()` mid-evaluate; also reads `cache_.next_stalled()` at top of its evaluate as a same-cycle backpressure signal, and `gather_file_.current_busy(warp)` as a per-warp back-pressure gate before popping a load from the LdSt FIFO | LdStUnit FIFO, gather buffer file, cache | command-style mutations + COMBINATIONAL stall read + REGISTERED gather-busy read | command-style mutations into LdStUnit/gather/cache are documented "internal-subsystem-mutating commands"; the gather-buffer write-port boundary arbitrates via REGISTERED state owned by `LoadGatherBufferFile` (row 11). The `cache.next_stalled()` read is a Phase 9 COMBINATIONAL same-tick edge: cache's stall signal is combinationally driven from registered tag/write-buffer/pending_fill state, and `cache.evaluate()` runs before `coalescing.evaluate()` in tick order. The `gather_file.current_busy()` read is a REGISTERED back-pressure accessor over the gather buffer's committed `current_*` state. | back-pressure (cache→coalescing stall, gather→coalescing busy) + forward-data (coalescing→ldst/gather/cache commands) | gather-buffer port arbitration cleaned up (Phase 7); cache↔coalescing stall edge formally classified COMBINATIONAL with call-site comment in `coalescing_unit.cpp` (Phase 9) | compliant (Phase 7 + 9) | 7, 9 |
| 10 | `L1Cache` external observable surface: REGISTERED scratch (`pending_fill_`, four `last_*_event_` slots) flipped by `commit()`, accessors return `current_*`. COMBINATIONAL same-tick (`stalled_`, `stall_reason_`) — single slot, reset at top of `evaluate`, observed mid-tick by coalescing (row 9) and post-commit by `record_cycle_trace`. Internal hardware state (`tags_`, `mshrs_`, `write_buffer_`) intentionally direct-mutated. | `TimingModel::record_cycle_trace` (post-commit, REGISTERED reads); `CoalescingUnit::evaluate` (same-tick, COMBINATIONAL stall read); cache itself; cache test expectations | mixed boundary discipline | mixed: REGISTERED for trace/scratch; COMBINATIONAL for stall | back-pressure (cache→coalescing stall) + forward-data (cache→trace) | Phase 9 carve-out: scratch trace events and the cross-cycle `pending_fill_` carrier go through `current_/next_` pairs so `record_cycle_trace` (which runs after `cache.commit()`) sees committed state via the `current_*` slot. Stall flags stay single-slot COMBINATIONAL because pipelining would cost a cycle of backpressure latency every stall transition (~5% on `embedding_gather`) without a hardware payoff at this cache scale. Internal MSHR/tag/write-buffer mutation stays direct because it has no cross-stage observers — only `L1Cache` itself touches it, and `test_cache.cpp` / `test_cache_mshr_merging.cpp` assert MSHR/tag state synchronously. | compliant (Phase 9); cache external boundary now classified per-field, internals carved out by design | 9 |
| 11 | `LoadGatherBufferFile::try_write` writes only `next_port_claimed_` (single shared flag, not per-buffer; models §5.3 "one line-to-gather-buffer extraction per cycle"). `L1Cache::handle_responses`, `drain_secondary_chain_head`, and `process_load` HIT path all funnel through `try_write`. | `LoadGatherBufferFile::try_write` reads the live `next_port_claimed_` (combinational first-writer-wins) | intra-cycle write-port arbitration | **COMBINATIONAL** | (internal arbitration) | first-writer-wins via the shared `next_port_claimed_` flag; `commit()` clears the flag at end-of-cycle so the next tick starts unclaimed. FILL > secondary > HIT priority is encoded by tick ordering: `cache_->evaluate()` runs at the top of the non-panic tick (FILL via `handle_responses`, secondary via `drain_secondary_chain_head`); `coalescing_->evaluate()` runs later in the tick (HIT via `process_load`). | tick order: `gather_file.commit` clears the flag at end-of-cycle; first writer in tick N+1 sees `next_port_claimed_ == false` and wins | compliant (Phase 7) | 7 |
| 12 | `OperandCollector::resolve_branch` writes `next_redirect_request_{valid, warp_id, target_pc}` on misprediction | `FetchStage::commit` and `DecodeStage::commit` read `opcoll.current_redirect_request()` and apply the flush there | flush request and redirect target PC | **REGISTERED** redirect-request via `RedirectRequest` produced by opcoll, consumed by fetch/decode at their own `commit()`. Mispredict-recovery now takes one additional cycle (Option A) | forward-data | tick-order: producer writes `next_` during evaluate; opcoll.commit flips to `current_` at end of cycle N; fetch.commit and decode.commit read `current_` at cycle N+1 (their commits run before opcoll.commit within the same cycle, so they observe last cycle's latched signal) | compliant (Phase 5) | 5 |
| 13 | `DecodeStage::current_ebreak_request()` (REGISTERED `next_`/`current_` pair, latched by `decode.commit()`); panic-flush cascade `scheduler/opcoll/gather_file/wb_arbiter->flush()` invoked at commit-phase boundary when `pending_panic_flush_` is armed | `TimingModel::tick()` (observes `current_ebreak_request()` at top of cycle to call `panic_->trigger`); scheduler/opcoll/gather buffer/writeback arbiter (consume `flush()` at commit-phase) | EBREAK request and machine flush | **REGISTERED** ebreak side-channel; per-stage `flush()` at commit-phase replaces the prior mid-evaluate `reset()` cascade. Trigger takes one additional cycle vs. the pre-Phase-6 mid-tick path (Option A) | forward-data (decode→TimingModel; PanicController→cascade targets) | tick order: decode.commit latches ebreak request at end of cycle N; tick top of cycle N+1 reads current_ebreak_request_ and calls panic_->trigger / arms pending_panic_flush_; commit-phase of cycle N+1 invokes flush() on each panic-flush target | compliant (Phase 6) | 6 |
| 14 | `PanicController::set_drained_query` wires a callable that the controller invokes inside `evaluate()`; the callable composes `execution_units_drained()` from `OperandCollector::current_busy`, each unit's `current_busy` + `next_has_result`/`next_fifo_empty`, and `WritebackArbiter::current_busy` — all const accessors over committed state | `PanicController::evaluate()` (case 2 drain step) | drained bit | **REGISTERED** for committed-state reads (the callable invokes only committed-state accessors) | back-pressure (controller polls accessors on each unit) | wired callable; the prior `set_units_drained()` pre-evaluate setter (which latched live state from another stage) is removed | controller queries drained_query_ inside its own evaluate(); the callable reads only committed-state accessors | compliant (Phase 6) | 6 |
| 15 | `L1Cache` calls `mem_if_.submit_read()` / `submit_write()` (mid-evaluate REGISTERED writes into mem_if's `in_flight_`); `L1Cache::handle_responses` reads `mem_if_.next_has_response()` and dequeues via `get_response()` mid-evaluate | `ExternalMemoryInterface` (FixedLatencyMemory and DRAMSim3Memory implementations) and `L1Cache` (consumes `next_has_response`) | external-memory request/response queue mutations + COMBINATIONAL response accessor | mixed: REGISTERED queue mutations + COMBINATIONAL `next_has_response()` poll | back-pressure (mem_if→cache response poll) + forward-data (cache→mem_if requests) | trusted-subsystem carve-out, mirroring row 10. Tick order is `cache.evaluate -> ... -> mem_if.evaluate`, so a request submitted in cycle N is processed by `mem_if.evaluate` at the end of the same cycle (one cycle of memory latency consumed at submit time). Likewise responses produced by `mem_if.evaluate` in cycle N are dequeued by `cache.handle_responses` at the top of cycle N+1, since `mem_if.evaluate` runs after `cache.evaluate`. The DRAMSim3 backend has tighter constraints (write-region FIFO bounds; see `dramsim3_memory.cpp:67`) that make REGISTERED queue mutation invasive without a clear payoff. | compliant by design (carve-out, Phase 9 documented) | 9 |

If a future change adds a new cross-stage edge, append a row here with
its classification (cycle axis + direction axis) and the phase that lands it.

`tools/render_signal_diagram.py` emits the whole-pipeline architecture
poster (Graphviz DOT + Mermaid) with modules grouped into Frontend &
Issue / Execute / Memory / Writeback / Control clusters and edges styled
by classification (cycle discipline as line style, direction as overlay).
The renderer drives off two extractors:

- **`--source=ast` (default).** The libclang extractor under
  `tools/diagram_extract_ast.py` walks the timing translation units in
  `build/compile_commands.json` (run `cmake -B build && cmake --build
  build` first). The C++ source is the source of truth, so the diagram
  picks up new modules and edges automatically.
- **`--source=markdown`.** The legacy extractor under
  `tools/diagram_extract_md.py` parses the per-boundary inventory above.
  Retained as the cross-check view of the same data.

Run `python3 tools/render_signal_diagram.py --validate` to diff the two
extractors. Documented carve-outs live in
`render_signal_diagram.VALIDATE_ALLOW_LIST`; everything else fails the
diff. The snapshot test at `tests/test_signal_diagram.py` pins module
count and a hand-curated edge floor.

## Forbidden patterns

The following are explicit violations of the discipline. Code review
should flag any of them when they appear in the timing model.

- **Plain mutable members read across stages mid-evaluate.** A bool/int
  written by stage A's `evaluate()` and read by stage B's `evaluate()` in
  the same tick. Use REGISTERED next/current with `current_*` accessor or
  COMBINATIONAL with a call-site comment.
- **Pre-evaluate setters that latch live state from another stage.**
  Examples in flight: `set_opcoll_free`, `set_decode_pending_warp`,
  `set_units_drained`, `set_unit_ready_fn`. These hide ordering
  dependencies inside `tick()` and silently invert when the orchestrator
  reorders calls. Expose the signal as a `const current_*()` accessor on
  the consumer that reads only its own committed state.
- **`consume_*` calls that synchronously mutate the other stage's
  state.** `WritebackArbiter::consume_result` flipping
  `unit.result_buffer_.valid = false` is the canonical case. Mutate only
  the unit's `next_*` slot; `commit()` performs the flip.
- **Mid-tick mutations of committed state that bypass `commit()`.**
  `fetch_->redirect_warp`, `decode_->invalidate_warp`, the panic
  `reset()` cascade, and direct writes to `warps_[w].branch_in_flight`
  fall under this rule. Express the request as a REGISTERED signal and
  let each stage flush at its own commit.
- **REGISTERED accessor not prefixed `current_*`.** A `const` accessor
  returning a committed-state value must use the `current_*()` prefix.
  Names with `is_*` filler (`is_busy`, `is_pending`, `is_in_flight`)
  are violations — drop the filler and add the cycle prefix.
- **COMBINATIONAL accessor not prefixed `next_*`.** A `const` accessor
  returning a live mid-tick value must use the `next_*()` prefix.
- **Predicate that doesn't follow the postfix design language.** No
  `is_*` filler on state predicates; reserve the `has_<noun>` form for
  possession predicates that precede a `consume_*` call. Inverse-polarity
  twins of an existing predicate (e.g. `ready_out` opposite `current_busy`)
  are forbidden — pick the asserted-blocking polarity and stick with it.
- **Cross-stage read inside a lambda body or free-function helper
  where the receiver is a parameter.** Hides the producer endpoint from
  static analysis. Inline the read at the call site or move it to a
  method on the producer module so libclang can resolve it.
- **REGISTERED state field exposed as `public`.** Must go through an
  accessor (`current_*()` / `next_*()`).

`tools/lint_timing_naming.py` enforces the prefix / postfix / polarity
/ field-shape rules mechanically. Enforcement is wired through the
`timing_naming_lint` CTest target — any finding fails the build. Use
`tools/lint_timing_naming.py --report-only` locally to inspect findings
without failing. Suppress a single finding with a per-line comment
`// timing-naming-allow: <reason>`.

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
- **Phase 3** (landed): Fetch/decode `current_busy` boundary. Replaced
  `output_consumed_` (plain bool round-trip) and `set_decode_pending_warp`
  (pre-evaluate setter) with `DecodeStage::current_busy()` computed in
  `DecodeStage::compute_ready()` and a direct `decode->current_pending_warp()`
  accessor read by fetch. Tick order is now
  `decode_.compute_ready -> fetch_.evaluate -> decode_.evaluate`. Fetch's
  output is now strictly REGISTERED (`current_output_ = next_output_` in
  `commit()`); evaluate encodes the hold-vs-advance decision into
  `next_output_`. Inventory rows: 1, 2.
- **Phase 4** (landed): Scheduler busy signals. Removed `set_opcoll_free`
  setter and the `UnitReadyFn` callback / `unit_ready_fn_` slot. Added
  `compute_ready()` (default no-op) and pure-virtual `current_busy()` to the
  `ExecutionUnit` base interface; each of ALU/MUL/DIV/TLOOKUP/LDST and
  `OperandCollector` overrides them, writing a `ready_out_` slot from
  committed `current_*` state. `WarpScheduler::set_dependencies` wires the
  opcoll plus five typed unit pointers at construction; `evaluate()` reads
  `opcoll_->current_busy()` and each unit's `current_busy()` directly. Tick
  order in `TimingModel::tick()` adds a backward sweep
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
- **Phase 9** (landed): Cache external boundary discipline. The cache subsystem
  was the largest remaining undocumented edge; the original refactor's row 9/10
  treated cache internals as a "trusted internal subsystem" but did not classify
  the external-facing observable surface or the cache↔mem_if edge.
  - `L1Cache::pending_fill_` and the four `last_*_event_` slots become
    REGISTERED `current_/next_` pairs; accessors return `current_*`; internal
    mutations write `next_*`; `commit()` flips. `TimingModel::record_cycle_trace`
    (which runs at end-of-tick after commits) reads via the `current_*` slot.
    No cycle-count impact — trace recording was always end-of-tick.
  - `L1Cache::stalled_` / `stall_reason_` stay single-slot COMBINATIONAL with
    a call-site comment in `coalescing_unit.cpp` declaring the same-tick
    backpressure semantics. An exploratory REGISTERED treatment was tried and
    reverted: it cost a cycle of backpressure latency per stall transition,
    showing up as +5.2% on `embedding_gather` (memory-bound; trips
    `WRITE_BUFFER_FULL` deferred fills frequently). The hardware path is
    feasible at this cache scale (small flop-backed tag array; combinational
    depth from registered tag/WB/pending_fill state through stall signal
    into coalescing's FSM closes timing on FPGA), so pipelining the signal
    is over-conservative.
  - Cache internal hardware state (`tags_`, `mshrs_`, `write_buffer_`)
    remains direct-mutation by design; the `test_cache.cpp` /
    `test_cache_mshr_merging.cpp` synchronous-state contract is preserved.
  - The cache↔`mem_if` boundary (`submit_read`/`submit_write`/`next_has_response`/
    `get_response`) is formally carved out as a row-10-style trusted-subsystem
    edge (new inventory row 15). Tick order encodes the latency arithmetic;
    DRAMSim3's tighter request-FIFO constraints make REGISTERED queue
    mutation invasive without a clear payoff.
  - Tests that observed REGISTERED slots (`pending_fill()`, `last_*_event()`)
    immediately after `process_load`/`process_store`/`evaluate` had
    `f.cache.commit()` inserted between the mutator and observer, modeling
    the natural cycle boundary. The `test_dramsim3_memory` write-buffer-full
    saturation loop was restructured to record the rejection bool mid-cycle
    and CHECK observable state after the end-of-cycle commit. All six
    workload benchmarks remain byte-identical to the pre-Phase-9 baseline.
  - Inventory rows 9 and 10 rewritten; new row 15 added for the
    cache↔mem_if carve-out.
- **Consolidation pass** (landed): hygiene cleanup after Phase 8.
  - `compute_ready()` and the `ready_out_` cache slot deleted
    everywhere. Back-pressure signals are now exposed as `const`
    accessors that read only the producer's committed state directly,
    so the value is stable through the entire evaluate phase regardless
    of where it is queried. Tick discipline collapses from three
    phases to two (evaluate + commit). `PipelineStage` and
    `ExecutionUnit` no longer carry `compute_ready()`.
  - `is_ready()` removed from `ExecutionUnit` and all five concrete
    units; callers (scheduler diag, drain checks, tests) use
    `current_busy()`. `OperandCollector::is_free()` removed; callers use
    `current_busy()`.
  - `flush()` on `OperandCollector` / `WarpScheduler` /
    `WritebackArbiter` / `LoadGatherBufferFile` collapsed to a
    `reset()` delegate, eliminating drift between the two bodies.
  - Dead double-buffer mirrors deleted: `LdStUnit::current_addr_gen_fifo_`
    (only writer was `commit()`; readers all read live `next_*`) and
    `LoadGatherBufferFile::current_port_claimed_` + accessor (no
    callers). The shared `next_port_claimed_` flag now models the
    intra-cycle write-port reservation directly with `commit()`
    clearing it at end-of-cycle.
  - Redirect-request read in `FetchStage::commit` and
    `DecodeStage::commit` consolidated into a free helper
    `read_redirect_request(override, opcoll)` declared next to
    `RedirectRequest`. Override fields collapsed from a 3- or 4-tuple
    to a single `std::optional<RedirectRequest>` per stage.
  - `BranchShadowTracker::set_in_flight` / `clear_in_flight` renamed
    to event-shaped methods: `note_branch_issued` (scheduler.evaluate),
    `note_resolved_correctly` (opcoll.resolve_branch on correct
    prediction), `note_redirect_applied` (fetch.apply_redirect on
    mispredict). The header now lists the three writer sites and the
    cycle-discipline reason for each, mirroring inventory row 5.
  - All workload benchmark cycle counts byte-identical to Phase 8.
- **Naming-and-access discipline** (landed): two-axis model (cycle
  discipline + direction), postfix design language (state predicate /
  possession predicate / payload accessor), asserted-blocking polarity
  convention, and `tools/lint_timing_naming.py` enforced via the
  `timing_naming_lint` CTest target. Cross-stage accessors renamed
  uniformly: `ready_out` (and per-unit overrides) → `current_busy()`
  with polarity flip; `is_pending` / `is_in_flight` / `is_busy` →
  `current_pending` / `current_in_flight` / `current_busy`;
  `is_stalled` → `next_stalled`; `fifo_empty` / `fifo_front` →
  `next_fifo_empty` / `next_fifo_front`; `has_result` /
  `has_response` → `next_has_result` / `next_has_response`;
  `committed_entry` → `current_committed_entry`; `pending_warp` →
  `current_pending_warp`; `pending_fill` and the four cache
  `last_*_event` slots → `current_*`. `WarpScheduler` field-access
  shape converted to all-pointer (scoreboard, branch tracker, opcoll,
  units) wired post-construction by `set_dependencies()`. The
  `query_unit_ready` lambda was inlined and the
  `read_redirect_request` free function was moved onto
  `OperandCollector::current_redirect_request_or_override()` so every
  cross-stage read has a statically resolvable receiver — both changes
  drove `tools/diagram_extract_ast.py` `MANUAL_AST_EDGES` down to the
  documented orchestrator-level floor (panic-flush cascade, ebreak,
  trace events, ldst→arbiter result, opcoll→tracker branch resolved).
  Workload benchmark cycle counts byte-identical to the pre-refactor
  baseline at every phase boundary. See
  `/project-plans/naming-and-access-discipline.md`.
