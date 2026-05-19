# `Reg<T>` Register Abstraction — Design & Migration Plan

## Context

The cycle-accurate timing model in `sim/` mimics synchronous RTL: each
stage's `evaluate()` computes a `next_*` state from committed `current_*`
state, and `commit()` flips `next_* → current_*` at the cycle boundary
(orchestrated per-cycle by `TimingModel::tick()` — seed → back-to-front
evaluate sweep → commit). The discipline is currently hand-rolled per
stage: paired `current_X_`/`next_X_` fields, a `seed_next()`, a `commit()`.

Nothing structurally enforces it. Contributors (frequently LLM agents)
keep introducing **same-cycle-visible state updates** — a plain mutable
member written and read within one cycle — which in real hardware would
be a register that only latches at the cycle edge. The canonical bug
(commit `0383f04`, the fetch/decode `output_consumed_` reorder) is exactly
this class. Hand-rolled double-buffering also leaks subtler bugs: a field
that should be seeded but isn't, a `commit()` that forgets a newly added
field.

This plan introduces typed state primitives — `Reg<T>` (a clock-edge
register), `RegFifo<T>` (a commit-disciplined FIFO), `Wire<T>` (a
combinational signal) — and migrates every state-holding element in the
timing model onto them, so that every member in a timing header has a
*declared kind*. A final lint rule then makes a raw `current_*`/`next_*`
field pair a build error, so the primitives are mandatory for all future
state.

### Design intent — the migration is behavior-neutral

`Reg<T>` is **behavior-neutral storage**, not a semantic change. Wrapping
a field changes behavior *only* if a read site is also re-routed between
the staged and committed value. Therefore:

- **The migration is designed to be byte-identical.** Every read site is
  faithfully translated to observe the exact value it observes today —
  committed reads → `current()`, same-cycle/staged reads (including
  in-place-mutated counters read back within the same `evaluate()`) →
  `next()`/`next_mut()`. The byte-identical benchmark gate applies to
  **every** phase.
- **`Reg<T>` does not assert RTL 1:1 correspondence** — only clock-edge
  latching. A high-level abstraction (an iterative unit's
  `cycles_remaining_` countdown, a round-robin pointer) is still a register:
  it latches at the cycle edge. Wrapping it preserves the modeled
  behavior exactly.
- **The fidelity audit is a separate, later task.** Once every register
  is a visible `Reg<T>`, every cross-stage / latch-point `next()` read is
  greppable. Deciding which of those reads *should* be a committed
  `current()` read (i.e. finding genuine same-cycle fidelity bugs) is a
  distinct follow-up, explicitly out of scope here.

A benchmark delta in any phase is therefore **not expected**. If one
occurs, the §"Delta triage" procedure classifies it as a migration
mistake (fix in-phase) or — rarely — a revealed pre-existing fidelity bug
that a faithful translation cannot avoid (document, accept; its *fix*
belongs to the separate audit task).

This is a behavior-preserving refactor and follows
`resources/refactor_workflow.md`: phased, regression-gated, consolidation
review every ~3 phases, no test-authoring round.

## The primitives (`sim/include/gpu_sim/timing/reg.h`, new file)

`RegBase` is the non-template virtual base of the state primitives, so a
stage can hold them uniformly for seed/commit (see `RegisteredStage`).

### `Reg<T>` — a clock-edge-latched register
- `const T& current() const` — committed read (the normal read).
- `const T& next() const` — staged read; **intra-stage self-reads only**.
  A cross-module `next()` read is the combinational-forward bug the lint
  forbids; `Reg` cannot enforce caller identity, so the lint stays the
  enforcer.
- `T& next_mut()` — in-place mutation of the staged value (deque
  `push_back`, struct field assignment, counter decrement) — avoids the
  full container copy a `set_next(copy)` would force.
- `void set_next(const T&)` — whole-value staged write.
- `void seed()` — `next_ = current_`; idempotent (safe on a stalled
  re-evaluated cycle).
- `void commit()` — `current_ = next_`.
- `void reset(const T& = T{})` — sets **both** `current_` and `next_`.
- Fields value-initialized: `T current_{}; T next_{};`. Plain movable
  value type — no owner back-pointer.

Auto-seed (every `Reg` is seeded each tick via `RegisteredStage`) is the
canonical hardware behavior — a register holds its value unless written.
For a field whose `evaluate()` unconditionally reassigns it (e.g. every
execution unit does `next_result_buffer_ = WritebackEntry{}` at the top
of `evaluate()`), auto-seed is immediately overwritten → byte-identical.
For a field `evaluate()` "holds", auto-seed (`next_=current_`) *is* the
hold → byte-identical. A delta can only arise where old code depended on
an unseeded staged value being stale — which is itself a latent bug.

### `RegFifo<T>` — a commit-disciplined FIFO
`Reg<std::deque>` is wrong for the registered FIFOs: their hardware
semantics are "committed deque + a *staged push intent* + a *staged pop
intent*", not "committed deque + staged deque". `RegFifo<T>`:
- `const std::deque<T>& current() const` — committed read.
- `stage_push(T)`, `stage_pop()`, `claim_port()`, `port_claimed()`.
- `commit()` applies pop-then-push, clears staging.
- `seed()` is a **no-op** (the committed deque is the only state).
- `reset()` clears deque + staging.
Covers `cache write_buffer_` (port claim → `claim_port()`),
`ldst addr_gen_fifo_`, `dramsim3 request_fifo_` — eliminating three
hand-rolled copies of the same discipline.

### `Wire<T>` — a combinational backward signal
For the cross-stage signals that have **no committed twin** — asserted by
a downstream stage during its `evaluate()`, read backward same-cycle by
an upstream stage. `Wire<T>`:
- `const T& value() const` — read the asserted value.
- `void drive(const T&)` — assert (called in the producer's `evaluate()`).
- `void reset()` — back to the default (called at the top of the
  producer's `evaluate()`, replacing today's manual reset).
- No `current()`/`commit()` — it is not a register.
Covers `alu next_redirect_`, `writeback_arbiter writeback_stall_`,
`cache stalled_`/`stall_reason_`/`next_cmd_ready_`. The producer's
`next_*()` accessor (e.g. `next_writeback_stall()`) forwards to
`wire_.value()`; accessor names are unchanged.

### `RegisteredStage` — seed/commit-all mixin
A base/mixin holding `std::vector<RegBase*>` and providing `seed_all()` /
`commit_all()` / `reset_all()` loops. Stages call
`register_state(&r1, &r2, ...)` once in their **constructor body** (runs
after all member initializers — construction-order-safe). Usable as a
mixin on non-`PipelineStage` helpers (`Scoreboard`, `BranchShadowTracker`).

**Explicit registration, not intrusive self-registration:** an owner
back-pointer in `Reg`'s constructor makes `Reg` non-copyable/non-movable
and creates a silent member-construction-order hazard. The "forgot to
register" gap is closed instead by a lint check (final phase) requiring
every `Reg<`/`RegFifo<`/`Wire<` member to appear in a `register_state(`
argument list.

## Member taxonomy — every timing-header member has a declared kind

After migration, every data member of a timing-model class is exactly one
of:

| Kind | Meaning | Examples |
|------|---------|----------|
| `Reg<T>` | clock-edge register | `cycles_remaining_`, `rr_pointer_`, `unit_busy_`, `writeback_bitmap_`, output slots, cache tags, MSHR entries |
| `RegFifo<T>` | commit-disciplined FIFO | `cache write_buffer_`, `ldst addr_gen_fifo_`, `dramsim3 request_fifo_` |
| `Wire<T>` | combinational backward signal | `alu next_redirect_`, `wb writeback_stall_`, `cache stalled_` |
| plain — **config** | set at construction, const-after | `num_warps_`, `fifo_depth_`, back-pointers |
| plain — **sim-instrumentation** (annotated `// sim-instrumentation`) | observational / accounting state that does **not** model a clocked hardware register | see below |
| plain — **per-cycle scratch** (annotated) | within-stage evaluate→commit scratch | `busy_this_cycle_`, `accepted_this_cycle_`, `processed_this_cycle_` |

### Explicitly NOT wrapped — simulator-instrumentation state

These are a high-level simulator encoding, not modeled clocked hardware;
forcing edge-latching discipline on them would be semantically wrong.
They stay plain, annotated `// sim-instrumentation`, and are exempt from
the final lint:

- `dramsim3` `fabric_cycle_`, `dram_ticks_` (free-running / accumulator
  counters), `phase_` (fractional cross-clock-domain accumulator),
  `max_response_queue_` / `max_write_ack_queue_` (peak observation).
- `ldst fifo_total_pushes_`, scheduler `ldst_issued_total_` — monotonic
  accumulators that *encode* FIFO occupancy (real hardware would compare
  pointers / use an up-down counter). Already discipline-correct
  (written at commit, read next cycle); left as-is.

### Deliberate non-register: `alu branch_resolved_`
Documented in `alu_unit.h` as not-seeded/not-gated by design — it fires a
side-effect exactly once across a multi-cycle stall. Stays a plain `bool`,
annotated; an auto-seeded `Reg` would break the intent.

## Scope

- **Public accessor methods and names are preserved unchanged**
  (`current_output()`, `next_writeback_stall()`, …) as one-line
  forwarders. This keeps `timing_naming_lint`, the
  `signal_diagram_ast_snapshot`, and the ~12 Catch2 binaries that call
  accessors green without source changes.
  *Verified:* `diagram_extract_ast.py` keys snapshot edges on the
  cross-module accessor **method name** called from a stage's
  `evaluate()`/`commit()` body, and inlines only same-class helpers — it
  never descends into another class's accessor body, so a renamed
  internal field is invisible to it.
- Per-warp arrays are wrapped **whole**, not per-element:
  `Reg<ScoreboardBits>` (POD wrapping `bool[MAX_WARPS][NUM_REGS]`),
  `Reg<std::array<bool,MAX_WARPS>>`. Copy compiles to the same `memcpy`
  the code does today — zero regression.
- Test hooks that poke state directly (`warp_scheduler`
  `test_set_unit_busy` / `test_reserve_writeback_slot`) are updated to
  drive the wrapped member; internal, minor.
- **Functional model** (`sim/src/functional/`) — the architectural
  correctness model, intentionally updated immediately, not
  cycle-accurate — is untouched.

## Phasing

One byte-identical migration, phased by stage for review size.

**Baseline** (`refactor_workflow.md` §2) — SHA
`da13605110681968ed444eb977e316d93d749c2b`, captured via
`bash ./tests/run_workload_benchmarks.sh --build-dir build` (DRAMSim3
backend). The byte-identical contract for every phase:

| Benchmark | cycles |
|-----------|--------|
| matmul | 101879 |
| gemv | 6363 |
| fused_linear_activation | 2621 |
| softmax_row | 2471 |
| embedding_gather | 49002 |
| layernorm_lite | 9933 |

All 30 CTest targets pass at baseline (incl. `timing_naming_lint`,
`signal_diagram_ast_snapshot`). Contract = 6 workload benchmarks + the
full CTest suite.

**Gate for every phase:** benchmarks byte-identical to baseline; all 16
Catch2 binaries pass; `timing_naming_lint` and `signal_diagram_ast_snapshot`
green. A delta runs the triage procedure below.

- **Phase 0 — Infrastructure.** Create `reg.h` (`RegBase`, `Reg<T>`,
  `RegFifo<T>`, `Wire<T>`, `RegisteredStage`) and `sim/tests/test_reg.cpp`
  (+ register in `sim/tests/CMakeLists.txt`). No stage uses it yet.
  Zero-delta by construction. (`test_reg.cpp` tests new infrastructure,
  not behavior — allowed under the no-test-authoring refactor rule.)
- **Phase 1 — `Scoreboard` + `BranchShadowTracker`.** Smallest, purest,
  fully seeded. Proves whole-array wrapping and the `RegisteredStage`
  mixin on a non-`PipelineStage` helper.
- **Phase 2 — `FetchStage` + `DecodeStage`.** Output slots, decode
  `pending_`/`ebreak_request_`, and `fetch rr_pointer_` (a faithful
  mechanical wrap — byte-identical).
- **Phase 3 — Execution units.** `cycles_remaining_`/`busy_`/`pending_`/
  `result_buffer_` for divide/tlookup/ldst/alu (all `Reg`); `multiply`
  pipeline `Reg<std::deque<PipelineEntry>>`; `ldst addr_gen_fifo_` →
  `RegFifo`. `alu branch_resolved_` left as the documented non-register.
- **Phase 4 — `WarpScheduler` + `OperandCollector`.** Output/diagnostics
  slots, opcoll busy/cycles/instr/output, and the in-place-mutated
  scheduler registers `rr_pointer_`, `unit_busy_[]`, `bitmap_head_`,
  `writeback_bitmap_`, `opcoll_cooldown_cycles_` — each a faithful
  mechanical wrap (`next_mut()` to mutate, `next()` to read back the
  mutated value this cycle). `ldst_issued_total_` stays plain
  sim-instrumentation.
- **Phase 5 — `L1Cache` + `MSHRFile` + `LoadGatherBufferFile`.** Cache
  tags / fill / outstanding-writes / trace-event pairs / cmd slots; mshr
  entries; gather buffers/claim/has_result; `cache write_buffer_` →
  `RegFifo`. Largest phase — split cache vs gather/mshr if the diff is
  unwieldy.
- **Phase 6 — `MemoryInterface` + `DRAMSim3Memory`.** Request slots;
  `dramsim3 request_fifo_` → `RegFifo`; dramsim3 instrumentation counters
  left plain + annotated. Exercise both `-DGPU_SIM_USE_DRAMSIM3=ON/OFF`.
- **Phase 7 — Combinational wires → `Wire<T>`.** `alu next_redirect_`,
  `writeback_arbiter writeback_stall_`, `cache stalled_`/`stall_reason_`/
  `next_cmd_ready_`. Producer accessors keep their names.
- **Final phase — Lint enforcement + fidelity-audit checklist.** Extend
  `tools/lint_timing_naming.py`: (1) a raw `current_X_`/`next_X_` field
  pair in a timing header is an ERROR — state must be a primitive;
  (2) every `Reg<`/`RegFifo<`/`Wire<` member must appear in a
  `register_state(` call; (3) any other plain member must carry a
  `// config` or `// sim-instrumentation` / `// scratch` annotation.
  Also emit `project-plans/reg-fidelity-audit.md`: a checklist of every
  cross-stage / latch-point `Reg::next()` read, grouped by stage, each to
  be classified in the separate follow-up task as a legitimate
  intra-stage staged read or a genuine same-cycle hazard to convert to
  `current()`. Ships with the last migration phase.

Consolidation review after **Phase 3** and **Phase 6**; review #3
(**opus model**) before the final commit.

## Delta triage

A faithful translation is expected to be byte-identical. If a phase
deltas:

1. **Reproduce the wrap with no semantic change.** Re-check every read
   site in the changed file: a committed read must use `current()`, a
   read of an in-evaluate-mutated value must use `next()`. The most
   common mistake is routing a same-cycle staged read through `current()`
   (or vice versa). Fix and re-gate.
2. If byte-identity still fails, **localize** the first divergent cycle
   via the structured Perfetto trace (`resources/trace_and_perf_counters.md`).
3. **Classify.** If the divergence is explainable only as the old code
   having relied on an unseeded stale staged value or an in-place
   double-mutation across a stalled re-evaluation, it is a **revealed
   pre-existing fidelity bug** — document it here (benchmark, cycle,
   field, mechanism) and accept the delta; its *fix* is the separate
   audit task. Otherwise treat it as a migration mistake until proven
   otherwise.

## Key risks

- **Commit gating.** Five units + opcoll + scheduler self-gate `commit()`
  on `WritebackArbiter::next_writeback_stall()`. The gate wraps the whole
  `commit_all()` (`if (stalled) return; commit_all();`) — a stalled stage
  freezes entirely, the correct hardware semantics. `seed_all()` stays
  **unconditional** (called from `tick()`'s seed phase) so a stalled
  re-evaluation is idempotent.
- **`Reg<std::deque>` / `Reg<std::vector>` copy cost.** No *new* cost:
  scoreboard, cache tags, mshr, multiply pipeline already copy pairs every
  tick; `RegFifo::seed()` is a no-op. Group-4 wraps add only small twins
  (scalars, ~6–10-entry `writeback_bitmap_`) — negligible.
  `seed()`/`commit()` use assignment so containers reuse capacity.
- **`reset()`/`flush()` drift.** `Reg::reset()` sets current_ AND next_;
  `reset_all()` loops. Watch for `flush()`↔`reset()` drift in
  consolidation review #3 (a known finding type from the prior refactor).
- **`Wire` reset timing.** Each `Wire` is `reset()` at the top of its
  producer's `evaluate()` — exactly replacing today's manual reset.
  Mechanical; verify the back-to-front sweep still runs each producer
  before its consumers (it does — unchanged).

## Doc sync (per `CLAUDE.md` trigger table)

- **`resources/timing_discipline.md`** — REQUIRED. `Reg<T>` as the
  canonical REGISTERED encoding, `Wire<T>` as the COMBINATIONAL-backward
  encoding, `RegFifo<T>` for registered FIFOs; `RegisteredStage`
  seed/commit-all; the gated `commit_all()` rule; the explicit
  non-registers (`branch_resolved_`, sim-instrumentation counters).
- **`resources/cpp_coding_standard.md`** — REQUIRED. New state MUST be a
  `Reg`/`RegFifo`/`Wire` member registered via `register_state`; plain
  members MUST be annotated `// config` / `// sim-instrumentation` /
  `// scratch`; the lint enforces it.
- **`resources/perf_sim_arch.md`** — REQUIRED. New file `reg.h`; each
  timing header's "key types" change.
- **`AGENTS.md`** — verify; likely no edit (the phasing doc is not a
  permanent reference artifact).
- `lint_timing_naming.py` docstring + `test_signal_diagram.py` doc
  comment updated to mention the primitive-field rules.
- No `UNTESTED.md` entry (refactor, no new behavior).

## Critical files

**Create:** `sim/include/gpu_sim/timing/reg.h`; `sim/tests/test_reg.cpp`
(+ `sim/tests/CMakeLists.txt`); `project-plans/reg-fidelity-audit.md`
(final phase — the follow-up-task checklist).

**Modify (highest-impact):** `tools/lint_timing_naming.py` (final phase);
`scoreboard.h` (Phase 1, the proving ground); the five execution-unit
headers + `.cpp` (Phase 3); `warp_scheduler.h`/`.cpp` (Phase 4);
`cache.h`/`cache.cpp` (Phase 5, largest); the remaining timing
headers/`.cpp` per phase; `resources/timing_discipline.md`,
`resources/cpp_coding_standard.md`, `resources/perf_sim_arch.md`.

## Verification

- **Per phase:** `cmake -B build && cmake --build build -j8`; `ctest -j8`
  from `build/` (16 Catch2 binaries + `timing_naming_lint` +
  `signal_diagram_ast_snapshot`); `python3 tools/bench_compare.py
  --baseline <baseline-sha>` (every phase byte-identical); `bash
  ./tests/run_workload_benchmarks.sh --build-dir build`; `tests/riscv-isa`
  + `tests/synthetic`.
- **dramsim3 phase (6):** build and test both
  `-DGPU_SIM_USE_DRAMSIM3=ON` and `OFF`.
- **`test_reg.cpp`** directly exercises `Reg<T>`/`RegFifo<T>`/`Wire<T>`
  seed/commit/reset/idempotence.
- One phase = one commit (implementation + doc updates bundled).
