# Cross-Stage Signaling Discipline Refactor

## Context

The cycle-accurate timing model in `/workspace/sim` uses an `evaluate()` /
`commit()` two-phase pattern intended to model parallel hardware: every stage's
`evaluate()` should compute a *new* state (`next_*`) from the *committed* state
(`current_*`), and `commit()` flips `next_* → current_*` at the cycle boundary.

`Scoreboard` (`sim/include/gpu_sim/timing/scoreboard.h`) implements this
correctly: separate `current_[][]` / `next_[][]` arrays, `seed_next()` at the
start of each tick, `commit()` at the end. Most other stages do not. The
recently fixed fetch/decode bug (commit `0383f04`) was a symptom: `FetchStage`
read `output_consumed_` before `DecodeStage` mutated it inside the same
`evaluate()` phase, capping frontend throughput at 1 instr per 2 cycles.

A subsequent audit catalogued **10 distinct cross-stage signaling violations**:
plain mutable members read by one stage and written by another mid-evaluate
(`output_consumed_`, `branch_in_flight`); pre-`evaluate` setters that latch
live state from another stage (`set_opcoll_free`, `set_decode_pending_warp`,
`set_unit_ready_fn`); execution units with empty `commit()` whose
`result_buffer_` is mutated synchronously by `WritebackArbiter::consume_result`;
mid-tick side-channel mutations that bypass `commit` entirely
(`fetch_->redirect_warp`, `decode_->invalidate_warp`, panic flush cascades).
Some of these are merely fragile (correctness depends on `tick()` call order);
some have already produced real throughput regressions.

The intent is to make the signaling discipline uniform, mechanically reviewable,
and robust to stage reordering — without inventing 1-cycle bubbles where the
current code (correctly, by design) routes signals combinationally within a
single tick. The user's design constraint: stall signals generated downstream
must be able to influence upstream's commit-or-hold decision in the same cycle,
mirroring real hardware ready/valid handshakes; intracycle data dependencies
must be preserved where they reflect actual register-to-register paths.

The intended outcome is a refactor in which every cross-stage signal is
explicitly classified, every stage's `commit()` does the work it should, and
the discipline is anchored by a documented contract that future stages can
follow without re-discovering today's hazards.

---

## Approach

### Per-boundary signal classification

Reject a uniform three-phase mandate. Instead, classify each cross-stage edge
as one of:

- **REGISTERED** — `producer.next_*` written in `evaluate()`, latched by
  `commit()`; consumer reads `producer.current_*` only. 1-cycle handshake by
  construction. Default for cross-cycle state (PCs, scoreboard bits,
  `branch_in_flight`, opcoll busy/cycles, unit `result_buffer_`,
  `instr_buffer`).
- **COMBINATIONAL** — producer's `next_*` is read by consumer in the same
  `evaluate()` phase. Order in `tick()` is part of the contract and must be
  documented at the call site. Used where the design intends zero-cycle
  fanout (scheduler issue → opcoll accept → unit accept inside one tick;
  cache fill → gather-buffer write port arbitration).
- **READY/STALL** — consumer exposes `ready_out()` computed in a
  `compute_ready()` method that reads only its own `current_*`. Producer
  reads `consumer.ready_out()` during its own `evaluate()` to decide whether
  to advance or hold. Used for backpressure (decode→fetch, opcoll→scheduler,
  unit→scheduler, gather-buffer write port).

Each cross-stage edge in the timing model gets a one-line entry in a new
discipline doc declaring its classification, producer, consumer, payload, and
any tick-order constraint.

### Enhanced PipelineStage contract

Add an optional `compute_ready()` method to `PipelineStage`
(`sim/include/gpu_sim/timing/pipeline_stage.h`):

```cpp
class PipelineStage {
public:
    virtual ~PipelineStage() = default;
    virtual void compute_ready() {}  // default no-op; override to expose ready_out
    virtual void evaluate() = 0;
    virtual void commit() = 0;
    virtual void reset() = 0;
};
```

`tick()` becomes a backward sweep of `compute_ready()` (terminal sink first,
fetch last) followed by the existing forward `evaluate()` sweep, then
`commit()`. The backward sweep is added only after every concrete stage that
participates in a READY/STALL edge has its own `ready_out()` method — i.e.,
the base-class change is the *last* land, not the first.

`Scoreboard` is the model implementation for REGISTERED state: `current_[]` /
`next_[]` arrays, `seed_next()` at the top of `tick()` (already present),
`commit()` flip. Reuse this pattern verbatim for new next/current pairs in
other stages.

### Phasing

Each phase is a self-contained PR landing the discipline at one cluster of
edges. Cycle counts are baselined before Phase 1 and diffed after each phase;
expected deltas are predicted in writing before running and compared.

**Phase 0 — Discipline doc and per-boundary inventory.** Land
`resources/timing_discipline.md`. For every cross-stage edge identified in the
audit, list classification, producer, consumer, payload, and tick-order
constraint. Add a pointer from `AGENTS.md` Key References. Add a discipline
section to `cpp_coding_standard.md` with the rule "any signal read across
`evaluate()` boundaries must be REGISTERED, COMBINATIONAL, or READY/STALL,
declared at the call site." No code changes; expected cycle-count delta zero.

**Phase 1 — Execution unit double-buffering.** Files:
`sim/{src,include/gpu_sim}/timing/{alu,multiply,divide,tlookup,ldst}_unit.{h,cpp}`,
`writeback_arbiter.cpp`, `execution_unit.h`. Add `next_result_buffer_` and
`next_pending_input_` members to each unit. `accept()` writes the `next_*`
slot. `consume_result()` writes `next_result_buffer_.valid = false`. Each
unit's `commit()` flips `next_*` → `current_*`. `has_result()` reads
`current_*`. Expected cycle-count delta: zero (no observable behavior change
because writeback arbiter and scheduler already read `has_result()` before
`consume_result()` mutates state).

**Phase 2 — OperandCollector double-buffering.** Files:
`operand_collector.{h,cpp}`, the issue path in `timing_model.cpp`. `accept()`
writes `next_busy_`, `next_cycles_remaining_`, `next_current_instr_`.
`evaluate()` reads `current_*` and writes `next_*`. `is_free()` returns
`!current_busy_`. The COMBINATIONAL forward chain
`scheduler.next_output → opcoll.accept → opcoll.evaluate → dispatch_to_unit →
unit.accept` is preserved by reading `current_busy_` (the gating signal) but
writing only into `next_*`. Expected delta: zero.

**Phase 3 — Fetch/decode `ready_out` boundary (re-fix the bug structurally).**
Files: `fetch_stage.{h,cpp}`, `decode_stage.{h,cpp}`, `timing_model.cpp`.
Remove the plain-bool `output_consumed_` and the `consume_output()` direct
mutation. Replace with `DecodeStage::ready_out()` computed in
`compute_ready()` from `current_pending_.valid` and the target warp's
committed buffer occupancy. Replace `set_decode_pending_warp` setter with a
direct read of `decode.ready_out()` during `fetch.evaluate()`. The post-fix
tick order is then `decode.compute_ready` → `fetch.evaluate` →
`decode.evaluate` → ...; the current "decode evaluates before fetch"
reorder is replaced with this structural form. Expected delta: zero
(behavior matches today's reordered tick).

**Phase 4 — Scheduler ready signals.** Files: `warp_scheduler.{h,cpp}`,
`operand_collector.h`, `execution_unit.h`, `timing_model.cpp`. Remove
`set_opcoll_free` setter and `unit_ready_fn_` callback. Add `ready_out()` on
opcoll and on each execution unit, computed in `compute_ready()` from
`current_*` only. Scheduler's `evaluate()` reads `opcoll.ready_out()` and
`unit.ready_out()` directly. Expected delta: zero (current pre-evaluate
setters already capture committed-cycle state).

**Phase 5 — Branch redirect and `branch_in_flight` as registered state.**
Files: `warp_state.h`, `warp_scheduler.cpp`, `operand_collector.{h,cpp}`,
`fetch_stage.cpp`, `decode_stage.cpp`, `timing_model.cpp`. `branch_in_flight`
becomes a per-warp next/current pair: scheduler issue writes `next_=true`,
opcoll branch completion writes `next_=false`, `commit()` flips. Branch
redirect is modeled as a REGISTERED signal: opcoll's `evaluate()` writes
`next_redirect_request_{valid, warp_id, target_pc}` on misprediction;
`commit()` latches; fetch and decode read `current_redirect_request_` on
their next-cycle `compute_ready()` and `evaluate()`, performing the flush
internally during their own `commit()`. The synchronous mutators
(`fetch_.redirect_warp`, `decode_.invalidate_warp`) become private flush
helpers invoked from each stage's own `commit()`, no longer called
mid-tick from `timing_model.cpp`. **Expected delta: small but nonzero on the
misprediction-flush path** (one cycle later than today, which matches real
EX→IF redirect-register hardware behavior). Pre-compute new expected values
for `test_branch.cpp` and update assertions deliberately, with a one-line
rationale per delta in the discipline doc.

**Phase 6 — Panic / EBREAK side-channel.** Files: `panic_controller.{h,cpp}`,
`decode_stage.cpp`, `timing_model.cpp`. Decode's EBREAK detection writes a
`next_ebreak_request_{valid, warp_id, pc}`. `tick()` observes after `commit()`
and calls `panic_.trigger`. The cascade `scheduler_->reset()`,
`opcoll_->reset()`, `gather_file_->reset()`, `wb_arbiter_->reset()` is
replaced with a per-stage `flush()` method called at commit time when the
panic signal becomes active. `set_units_drained` setter is replaced with a
`PanicController::compute_ready()` query of unit ready_out states. **Expected
delta: small** on `test_panic.cpp` (1-cycle shift on the trigger boundary;
all assertions are inequalities with wide margin).

**Phase 7 — Gather-buffer port arbitration cleanup (descoped: cache
internals).** Files: `load_gather_buffer.{h,cpp}`, `cache.cpp` boundary call
sites. The gather-buffer write port is re-modeled as a single arbiter with
two requestors (cache FILL, coalescing HIT) and per-requestor `ready_out`
signals computed in `compute_ready()`. FILL keeps priority. The existing
`port_used_this_cycle` flag is replaced with next/current discipline
internal to `LoadGatherBufferFile`. `claim()` writes `next_*`; `commit()`
flips. **Cache tags / MSHRs / write buffer remain direct-mutation by design.**
The cache is treated as a trusted internal subsystem: its public boundary
calls (`process_load`, `process_store`, `is_stalled`, `gather_file_.try_write`)
are the discipline points; its internals are left alone because
`test_cache.cpp` and `test_cache_mshr_merging.cpp` assert MSHR/tag state
synchronously after `process_load` calls and any double-buffering breaks
those tests catastrophically. Expected delta: zero on tests outside the
cache; cache tests untouched.

**Phase 8 — Lift `compute_ready()` into `PipelineStage` base; formalize
backward sweep in `tick()`.** Done last, after every stage participating in
READY/STALL edges already implements its own `ready_out()` and
`compute_ready()`. Adds the backward sweep call at the top of `tick()`
(below `scoreboard_.seed_next()` and `cache_->evaluate()`). Expected delta:
zero (the explicit sweep replaces ad-hoc setter calls that were already
computing the same signals).

### Critical files

- **Base class & gold-standard reference**:
  `sim/include/gpu_sim/timing/pipeline_stage.h` (extend),
  `sim/include/gpu_sim/timing/scoreboard.h` (model — reuse pattern verbatim).
- **Tick orchestration** (every phase touches it):
  `sim/src/timing/timing_model.cpp` (the `tick()` method, lines 320–453).
- **Phase 1**:
  `sim/{src,include/gpu_sim}/timing/{alu,multiply,divide,tlookup,ldst}_unit.{h,cpp}`,
  `sim/src/timing/writeback_arbiter.cpp`,
  `sim/include/gpu_sim/timing/execution_unit.h`.
- **Phase 2**: `sim/{src,include/gpu_sim}/timing/operand_collector.{h,cpp}`.
- **Phase 3**: `sim/{src,include/gpu_sim}/timing/fetch_stage.{h,cpp}`,
  `sim/{src,include/gpu_sim}/timing/decode_stage.{h,cpp}`.
- **Phase 4**: `sim/{src,include/gpu_sim}/timing/warp_scheduler.{h,cpp}`,
  plus `ready_out()` additions to opcoll and each unit header.
- **Phase 5**: `sim/include/gpu_sim/timing/warp_state.h`,
  `sim/src/timing/warp_scheduler.cpp`,
  `sim/{src,include/gpu_sim}/timing/operand_collector.{h,cpp}`,
  `sim/src/timing/fetch_stage.cpp`, `sim/src/timing/decode_stage.cpp`.
- **Phase 6**: `sim/{src,include/gpu_sim}/timing/panic_controller.{h,cpp}`,
  `sim/src/timing/decode_stage.cpp`.
- **Phase 7**: `sim/{src,include/gpu_sim}/timing/load_gather_buffer.{h,cpp}`,
  `sim/src/timing/cache.cpp` (call-site boundaries only).
- **Documentation** (each phase updates as needed):
  `resources/timing_discipline.md` (new in Phase 0),
  `resources/cpp_coding_standard.md` (discipline section in Phase 0),
  `resources/perf_sim_arch.md` (per-stage compute_ready/evaluate/commit
  responsibilities, updated per phase),
  `resources/gpu_architectural_spec.md` (only if Phase 5/6 introduce
  observable architectural changes — branch redirect register, panic trigger
  register), `AGENTS.md` Key References (add discipline doc).

### Existing utilities to reuse

- **Scoreboard pattern** (`scoreboard.h`): the canonical
  `current_[]`/`next_[]` + `seed_next()` + `commit()` template. Every new
  next/current pair in this refactor follows the same shape.
- **`FetchStage::next_output_`/`current_output_`**: existing precedent for
  optional-typed payload double-buffering. Phase 5 redirect-request signal
  uses the same idiom.
- **`WarpScheduler::next_diagnostics_`/`current_diagnostics_`**: existing
  precedent for per-warp metadata double-buffering. Phase 5 per-warp
  `branch_in_flight` follows the same idiom.
- **Existing tests as canaries**: `test_scoreboard.cpp` (always-green; its
  test-target *is* the gold-standard discipline — should never fail or
  change).

---

## Verification

### Pre-refactor baseline

Before Phase 1, capture a cycle-count manifest:

```
cmake --build build -j8
cd build && ctest -j8 --output-on-failure | tee /tmp/baseline-tests.txt
```

For each integration and benchmark test, dump `(test_name, total_cycles, ipc,
external_memory_reads, external_memory_writes, scheduler_idle_cycles,
fetch_skip_backpressure)` to `tools/.timing_baseline.json`. The full benchmark
suite (`bash ./tests/run_workload_benchmarks.sh --build-dir build`) under both
backends (DRAMSim3 default and `--fixed-memory`) is captured the same way.

### Per-phase validation

After each phase:

1. `cmake --build build -j8` — must succeed without warnings.
2. `cd build && ctest -j8 --output-on-failure` — all 21 tests must pass.
3. `bash ./tests/run_workload_benchmarks.sh --build-dir build` — `SUMMARY`
   line shows `failed=0`.
4. Cycle-count diff vs. baseline. Every nonzero delta must match the
   prediction recorded in the discipline doc and in the PR description for
   that phase.

### Predicted deltas

| Phase | Predicted total-cycles delta | Tests likely to shift |
|---|---|---|
| 0 | 0 (docs only) | none |
| 1 | 0 | none |
| 2 | 0 | none |
| 3 | 0 (replicates current post-bugfix order) | none |
| 4 | 0 | none |
| 5 | +1 cycle per branch misprediction | `test_branch.cpp` misprediction cases; `matmul`/`gemv` IPC unchanged within ±0.2% |
| 6 | +1 cycle per panic trigger | `test_panic.cpp` (assertions are `< 1000`, robust) |
| 7 | 0 | none (cache tests untouched by design) |
| 8 | 0 | none |

### Functional spot-check

After Phase 5 (the largest expected behavior change), re-run the matmul
roofline (`python3 tools/roofline_matmul.py`) and shape sweep
(`python3 tools/matmul_shape_sweep.py`); IPC under DDR3 should match the
post-`0383f04` baseline within ±0.2% for kernels without misprediction-heavy
control flow.

### Test-suite tripwires

- `test_scoreboard.cpp` must never fail or change cycle counts at any phase
  — it tests the gold-standard pattern directly.
- `test_cache.cpp` and `test_cache_mshr_merging.cpp` must not change cycle
  counts at any phase — if they do, discipline has leaked into cache
  internals (Phase 7 explicitly forbids this).
- Any test failure that *can't* be matched to a predicted delta in the table
  above is a regression and the phase rolls back.

### Documentation sync

Each phase updates documentation atomically with the code change, per the
project's `CLAUDE.md` Documentation Sync rules:

- `resources/timing_discipline.md` — new boundary entries, classification
  changes.
- `resources/perf_sim_arch.md` — per-stage `compute_ready`/`evaluate`/`commit`
  responsibilities updated as stages change.
- `resources/cpp_coding_standard.md` — discipline section updated only if a
  new pattern is introduced.
- `resources/gpu_architectural_spec.md` — updated in Phase 5 (1-cycle
  redirect register on misprediction) and Phase 6 (panic trigger
  registration); no other phases.
- `UNTESTED.md` — only if any phase ships behavior change without a targeted
  test (none are expected).
