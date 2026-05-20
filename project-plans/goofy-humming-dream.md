# Eliminate the `current_mut()` Anti-Pattern — Durable-State-as-Register, Combinational-State-as-Wire

## Context

`Reg<T>::current_mut()` writes to a register's Q output mid-cycle. That operation has no hardware analog: in synthesis, Q is the *clocked output* of a register and does not change between clock edges. Every same-cycle "make the consumer see this value now" effect is in fact a combinational mix — `Q OR new_signal`, with the new signal coming from a wire driven this cycle and consumed by a downstream input. `current_mut()` is the simulator pretending Q changed when what should actually happen is the consumer reading `Q` AND-gated or OR-mixed with a `Wire<T>` signal.

Today there are 11 call sites grouped under 5 documented "exceptions" in `sim/include/gpu_sim/timing/reg.h:50-79`. Each exception is the same shape — a same-cycle write to committed state, justified by stage-ordering reasoning. That's exactly the kind of discipline the `Reg<T>` wrapper was built to remove; `current_mut()` re-admits it through a documented escape hatch.

The deeper principle, which this plan also commits to:

- **Every piece of durable cross-cycle state in a timing-model class is a register.** No plain `std::deque` / `std::vector` / `std::array` / scalar that holds value across `tick()` calls. `Reg<T>` for clocked registers, `RegFifo<T>` for FIFOs, `PulseReg<T>` (new) for one-cycle pulse / command slots whose D defaults to invalid.
- **Every piece of combinational state that doesn't survive the cycle is a `Wire<T>`.** Scratch member fields used for evaluate→commit handoff (today annotated `// scratch`) become `Wire<T>`. The annotation goes away; the type is the annotation.
- Function-body local variables stay local. The wrappers are for class *members*.
- The lint is extended so that every non-`const`, non-`// sim-instrumentation` member in `sim/include/gpu_sim/timing/*.h` must be one of the four kinds: `Reg`-family, `Wire`, `const`-after-construction config, or `// sim-instrumentation`. A bare `bool foo_;` in a timing header becomes a build error.

`current_mut()` is deleted at the end of the refactor and the lint enforces it cannot return.

This is a behavior-correcting refactor following [`/resources/refactor_workflow.md`](/resources/refactor_workflow.md): regression-as-contract, phased commits, consolidation review every ~3 phases. The `reg-fidelity-audit.md` framing applies — most phases are expected byte-identical (the new shape produces the same observation as the old `current_mut()` shortcut), but each phase explicitly runs `bench_compare` and documents any delta as a revealed pre-existing fidelity bug.

## Anti-pattern inventory

The 11 call sites today, grouped by the five reg.h-documented patterns:

| Pattern | Sites | Fix |
|---------|-------|-----|
| (1) Redirect-flush | `fetch_stage.cpp:153` (`output_.current_mut() = nullopt`) | Combinational gating: eligibility scan AND-masks the redirected warp's current_output via the existing `next_redirect_` Wire. No write to current_. |
| (2) Post-commit consumed-mark | `decode_stage.cpp:93` (`pending_.current_mut().valid = false`) | `warp_state.instr_buffer` becomes `RegFifo<BufferEntry>`. Decode co-stages `instr_buffer.stage_push()` and `pending_.set_next({valid=false})` in evaluate. Commit applies both. |
| (3) Memoryless-consumer mid-evaluate invalidation | `cache.cpp:640, 650`; `memory_interface.cpp:76, 86`; `dramsim3_memory.cpp:167, 171`; `load_gather_buffer.cpp:146` (7 sites) | Slots become `PulseReg<T>` — seed defaults next_ to T{}. Consumer reads current_, processes, does not modify current_. Producer's cache-stall handshake stays on the existing `next_cmd_ready_` / `next_stalled_` Wires. |
| (4) Deferred-claim dual-write | `load_gather_buffer.cpp:131` (writes both current and next within one loop) | `Wire<std::bitset<MAX_WARPS>> just_claimed_` driven by `gather_file.evaluate()`. `current_busy()` returns `buffers_.current().busy[w] OR just_claimed_.value().test(w)`. The staged claim still goes to `buffers_.next_mut()` only. |
| (5) Test-hook dual-write | `warp_scheduler.h:116, 120` (`test_set_unit_busy`, `test_reserve_writeback_slot`) | Tests call `scheduler.seed_all()` explicitly (or use a setup helper). Hooks write only `next_mut()`. Drop the `current_mut()` half. |

After the refactor, `Reg<T>` exposes only `current()`, `next()`, `next_mut()`, `set_next()`, `seed()`, `commit()`, `reset()`, `initialize()`. `current_mut()` is gone.

## New primitive: `PulseReg<T>`

```cpp
template <typename T>
class PulseReg : public RegBase {
public:
    PulseReg() = default;
    explicit PulseReg(const T& init) : current_(init) {}

    const T& current() const { return current_; }
    const T& next() const { return next_; }
    T& next_mut() { return next_; }
    void set_next(const T& v) { next_ = v; }

    void seed() override { next_ = T{}; }        // <-- the key difference from Reg<T>
    void commit() override { current_ = next_; }
    void reset() override { current_ = T{}; next_ = T{}; }
    void initialize(const T& v) { current_ = v; next_ = v; }

private:
    T current_{};
    T next_{};
};
```

Semantics: one-cycle pulse / command slot. Q defaults to T{} each cycle unless the producer explicitly drives. Producer reads current_ for the previously-latched value (the last pulse, if any); consumer reads current_ and acts but never modifies it. The producer's "stop driving" is implicit — no `set_next` call, no value latched.

This eliminates the consumer's `current_mut().valid = false` clear (the slot defaults to invalid automatically) and the consumer-tail `set_next(T{})` clear at end of commit (auto-seed handles it).

## Phasing

Per `refactor_workflow.md`: one phase = one commit (implementation + doc updates bundled). Each phase runs `cmake --build build -j8`, full `ctest -j8`, and `python3 tools/bench_compare.py --baseline <baseline-sha>`. Deltas are documented inline in this file's "Per-phase findings" section as they emerge.

### Baseline

Before Phase 0 lands, capture a benchmark baseline from the current HEAD:

```
python3 tools/bench_compare.py --baseline HEAD
```

Record the SHA and the six benchmark cycle counts in this file under "Baseline" below.

### Phase 0 — Infrastructure: `PulseReg<T>` + audit findings

- Add `PulseReg<T>` to `sim/include/gpu_sim/timing/reg.h` (alongside `Reg<T>`, `RegFifo<T>`, `Wire<T>`).
- Extend `sim/tests/test_reg.cpp` with a `PulseReg<T>` section: seed-to-default behavior, commit, reset, producer-drives-then-consumer-sees pattern.
- **Audit all plain (non-`Reg`/`RegFifo`/`Wire`/`const`/`// sim-instrumentation`) members across `sim/include/gpu_sim/timing/*.h`.** Walk every header; for each plain member, classify:
  - Durable cross-cycle → `Reg`-family (note which: `Reg`, `RegFifo`, `PulseReg`).
  - Transient evaluate→commit handoff → `Wire<T>`.
  - Construction-only config → mark `const` if not already.
  - Non-clocked accumulator → annotate `// sim-instrumentation` if not already.
- Write the audit findings into a new section of this plan (`## Audit findings — plain members in timing headers`). This list drives the phase plan for the bulk durable-state conversions.

**Validation:** build + ctest + bench (byte-identical expected — `PulseReg<T>` is unused, audit is documentation only).

### Phase 1 — Pattern 5: test-hook cleanup

- Update `WarpScheduler::test_set_unit_busy` and `test_reserve_writeback_slot` in `sim/include/gpu_sim/timing/warp_scheduler.h:115-122` to write `next_mut()` only.
- Audit callers in `sim/tests/test_warp_scheduler.cpp` (lines 254, 285-286, 307 per exploration). All currently call `f.scoreboard.seed_next()` before evaluate; verify each also calls `scheduler.seed_all()` (or equivalent). If any test doesn't seed the scheduler before evaluate, either add the seed call or restructure setup.
- Update warp_scheduler.h:108-114 comment block to reflect the new semantics.

**Validation:** build + ctest (test_warp_scheduler must stay green) + bench (byte-identical expected).

### Phase 2 — Pattern 1: redirect-flush via combinational gating

- In `FetchStage::evaluate()` (`sim/src/timing/fetch_stage.cpp:26-119`):
  - Read the redirect once at the top (already done: lines 35-45).
  - Delete `output_.current_mut() = std::nullopt;` at line 153 (inside `apply_redirect()`). Keep the staged-slot clear (`output_.next_mut() = nullopt`).
  - Modify the READY/STALL gate (line 52) to mask the redirected warp: the gate triggers only if `output_.current().has_value() && !(redirected_warp && *redirected_warp == output_.current()->warp_id) && !decode_ready`.
  - Modify the `current_output_warp` definition (lines 77-80) to similarly mask the redirected warp, so the eligibility scan's `inflight_to_w` doesn't double-count a doomed output.
- Update `apply_redirect()` to no longer touch committed output_ — only staged.
- Update the doc comment around the redirect-apply to reflect the new shape (Wire-gated rather than Q-modified).

**Validation:** build + ctest (full pipeline tests must pass; redirect tests in particular) + bench. Byte-identical expected — the gating produces the same observation as the Q-write.

### Phase 3 — Pattern 4: gather-buffer dual-write → `Wire<std::bitset>`

- Add `Wire<std::bitset<MAX_WARPS>> just_claimed_` to `LoadGatherBufferFile`.
- In `LoadGatherBufferFile::evaluate()` (`sim/src/timing/load_gather_buffer.cpp:108-148`):
  - Delete the `buffers_.current_mut()` write at line 131 and the loop that dual-writes (lines 135-142 become a single-write to `buffers_.next_mut()` only).
  - Drive `just_claimed_.set(req.warp_id)` (or equivalent bitset operation) when applying a claim.
  - Reset `just_claimed_` at the top of `evaluate()`.
- Modify `LoadGatherBufferFile::current_busy(warp_id)`:
  ```cpp
  bool current_busy(uint32_t warp_id) const {
      return buffers_.current()[warp_id].busy || just_claimed_.value().test(warp_id);
  }
  ```
- `claim_request_.current_mut().valid = false;` at line 146 stays for now — that's a Pattern 3 site, handled in Phase 4.

**Validation:** build + ctest (test_load_gather_buffer + test_integration must pass) + bench. Byte-identical expected — the OR-mix produces the same observation as the dual-write.

### → Consolidation review #1 (Phases 1-3, **opus**)

Range: `<baseline-sha>..HEAD`. Reviewer reads the diff and flags duplication, dead code, drift between similar patterns. No fixes, just a report.

### Phase 4 — Pattern 3: memoryless-consumer slots → `PulseReg<T>`

Seven sites convert in one phase (they share the same shape):

- `L1Cache::load_cmd_`, `store_cmd_` (cache.h:303-304): `Reg<...>` → `PulseReg<...>`.
- `FixedLatencyMemory::read_request_`, `write_request_` (memory_interface.h:138-139): same.
- `DRAMSim3Memory::read_request_`, `write_request_` (dramsim3_memory.h:199-200): same.
- `LoadGatherBufferFile::claim_request_` (load_gather_buffer.h:183): same.

For each:
- Delete the consumer's `current_mut().valid = false` clear in evaluate().
- Delete the consumer's `set_next(T{})` clear at the tail of commit() (no longer needed — `PulseReg::seed()` handles it).
- Add the slot to the owning stage's seed phase if it was opted out (`PulseReg`'s seed-to-default makes opt-out unnecessary).
- Update the doc comments documenting "memoryless consumer pattern" to point at `PulseReg<T>`.
- Update the lint (`tools/lint_timing_naming.py`) to recognize `PulseReg<T>` as a registered-state primitive: must appear in `register_state(...)`, treated the same as `Reg<T>` for the cross-module `next()`-read rule.

**Validation:** build + ctest (test_cache, test_load_gather_buffer, all integration tests must pass; both `-DGPU_SIM_USE_DRAMSIM3=ON/OFF` builds) + bench. Most likely byte-identical (seed-to-default at top of cycle is observationally equivalent to commit-tail clear from previous cycle) but flag any delta.

### Phase 5 — Pattern 2: `warp_state.instr_buffer` → `RegFifo<BufferEntry>`

- Convert `warp_state.instr_buffer` from a plain `std::deque`-backed `InstructionBuffer` to `RegFifo<BufferEntry>`. Update `sim/include/gpu_sim/instruction_buffer.h` accordingly — likely deleted in favor of direct `RegFifo` usage, or kept as a thin wrapper that exposes `is_full()` / `capacity()` over the underlying RegFifo.
- Decide ownership: the per-warp buffers must enroll in some `RegisteredStage`. The natural owner is `DecodeStage` (which pushes) or a new lightweight container. Likely simplest: have `DecodeStage` enroll all `warp_state[].instr_buffer` slots via `register_state` for seed/commit/reset uniformity.
- In `DecodeStage::evaluate()`: co-stage `pending_` advance, `instr_buffer.stage_push(entry)`, and `pending_.set_next({valid=false})` when the push will be accepted. The is-full check runs against committed buffer state (no same-cycle pop credit), preserving today's fetch-side conservatism.
- In `DecodeStage::commit()`: delete the post-`commit_all()` push block (lines 87-95) and the `pending_.current_mut().valid = false;` write at line 93.
- In `WarpScheduler`: convert the buffer pop to `instr_buffer.stage_pop()`.
- Verify `FetchStage::will_be_full` still reads committed-only state (no change needed — the carve-out at `timing_model.cpp:479-488` already enforces this).

**Validation:** build + ctest (every pipeline test; this is the most invasive structural change) + bench. May produce a delta — document if so. The expected behavior is byte-identical (same one-cycle latency from decode to scheduler-visible buffer), but the structural change is large.

### → Consolidation review #2 (Phases 4-5, **opus**)

Range from review-1 end to HEAD. Look especially for `RegFifo` vs `PulseReg` boundary confusion, duplicated handshake patterns across cache/memory/gather, dead `set_next(T{})` commit-tail clears that should have been removed.

### Phase 6 — Audit findings: durable plain members → `Reg`-family

Drive this phase from the Phase 0 audit list. For each plain durable member found:

- Wrap in the appropriate primitive (`Reg<T>` for scalars/POD, `RegFifo<T>` for FIFOs, `PulseReg<T>` for one-cycle pulse semantics — though most non-Pattern-3 cases will be `Reg<T>`).
- Enroll via `register_state(...)`.
- Audit read sites: committed → `current()`, intra-stage staged → `next()` / `next_mut()`.
- Per finding, decide whether it ships as a sub-commit within Phase 6 or its own phase. Findings of low complexity bundle; findings touching multiple files split.

The audit will likely surface durable scalars (PCs, pointers, counters that model real hardware registers, FIFOs that hand-roll their own discipline). Each conversion is mechanically the same as the existing Phase 1-8 `reg.h` migration phases, but for fields the original migration missed.

**Validation:** per-finding-batch build + ctest + bench. Document per-finding deltas (most expected byte-identical).

### Phase 7 — Combinational scratch fields → `Wire<T>`

Driven by the same Phase 0 audit. For every member field annotated `// scratch` (or any equivalent that's written by evaluate and read by commit, never observed by another module):

- Wrap as `Wire<T>` (intra-class wire; no cross-module accessor required).
- Drive at the assignment site in evaluate (typically: replace `processed_this_cycle_ = true;` with `wire_processed_.drive(true);`).
- Reset at the top of the owning stage's evaluate (`Wire::reset()` makes this mechanical).
- Read sites become `wire_foo_.value()`.
- Remove the `// scratch` annotation — `Wire<T>` is now self-documenting.

Known sites (representative, not exhaustive — Phase 0 audit produces the full list):
- `ALUUnit::processed_this_cycle_` (alu_unit.h:147)
- `OperandCollector::busy_this_cycle_`, `accepted_this_cycle_`
- Execution units' `busy_this_cycle_` flags

Note: `ALUUnit::branch_resolved_` is the documented exception (deliberate non-register, fires once across multi-cycle stall) — it stays a plain `bool` with its existing comment, but the lint rule must exempt it explicitly (probably by tightening the annotation form to `// deliberate-non-register: <reason>` and listing it in the exception set).

**Validation:** build + ctest + bench. Byte-identical expected (Wire reset-at-top is exactly the "assign false at top of evaluate" pattern these fields already follow).

### Phase 8 — Delete `current_mut()` + lint enforcement

- Delete `current_mut()` from `Reg<T>` in `sim/include/gpu_sim/timing/reg.h`. The five-pattern comment block (lines 50-79) is replaced with a brief note that `current_mut()` is intentionally absent because Q does not change between clock edges; pointers to `PulseReg<T>` and `Wire<T>` for the patterns it used to support.
- Extend `tools/lint_timing_naming.py`:
  - Existing rule (raw `current_X_`/`next_X_` pair = error) stays.
  - Existing rule (`Reg`/`RegFifo`/`Wire` must enroll via `register_state`) extended to include `PulseReg`.
  - **New rule (strict-compliance taxonomy):** every member of a class in `sim/include/gpu_sim/timing/*.h` must be one of:
    - A primitive: `Reg<T>` / `RegFifo<T>` / `PulseReg<T>` / `Wire<T>`.
    - A reference (`T&`) — inherently non-state.
    - Annotated `// sim-instrumentation` (non-clocked accumulator).
    - Annotated `// test-only-override` (test-fixture hook state).
    - Annotated `// back-pointer` (component wiring).
    - A `const`-qualified config value.

    Any other plain field (`bool foo_;`, `std::deque<T> bar_;`, `T* p_;` without the back-pointer annotation, etc.) is a build error. Deliberate-non-register exceptions are listed by fully-qualified name in an explicit lint exception set if any survive Phase 7; there is no generic `// deliberate-non-register:` carve-out keyword.
  - **New rule:** `current_mut(` appearing anywhere in the codebase is an error (the method no longer exists, but the lint provides a clearer error message than the compiler's "no member named current_mut" when someone tries to add it back).
- Update `tools/lint_timing_naming.py` docstring and the `test_signal_diagram.py` doc comment to describe the strengthened taxonomy.
- Update `resources/timing_discipline.md`: add `PulseReg<T>`; describe the durable-state-as-register and combinational-state-as-Wire principles as the explicit rule.
- Update `resources/cpp_coding_standard.md` § Cross-stage accessor naming and the timing-state-shape rules.
- Update `sim/include/gpu_sim/timing/reg.h` header comment block (the "every state-holding member is exactly one of" list) to reflect the four-kinds taxonomy.
- Delete `project-plans/reg-fidelity-audit.md` once its successor (this plan) is complete — or update it to reflect what the audit no longer needs to cover.

**Validation:** build + ctest + bench. Critically, the lint must catch every violation; spot-check by intentionally introducing a plain `bool foo_;` to a timing header and confirming the lint flags it before reverting.

### → Consolidation review #3 (Phases 6-8, **opus**)

Per `refactor_workflow.md`: the final pre-commit review of a large refactor uses `opus`. Range: review-2 end to HEAD. This is the last chance to catch subtle semantic-equivalence issues, intentional-vs-accidental duplication, lint coverage gaps, and drift between similar `Reg`/`PulseReg`/`Wire` use sites.

## Subagent dispatch policy

This refactor uses the `refactor_workflow.md` shape (regression-as-contract, phased commits, consolidation reviews) — not the formal `multi-agent-workflow` skill, which is for feature additions and assumes new behavior to test-author against. Subagent dispatch is selective and scoped to specific phases.

**Model:** every non-`Explore` subagent runs with `model: opus` (passed via the `model` parameter on the `Agent` call). This includes `general-purpose` agents and the `consolidation-reviewer` agent in all three reviews. The project's default for `consolidation-reviewer` is sonnet; this plan overrides to opus across the board. `Explore` agents use their default model.

**Per-phase dispatch:**

| Phase | Dispatch | Rationale |
|-------|----------|-----------|
| 0 (PulseReg + audit) | Direct for PulseReg; **2-3 Explore agents in parallel** for the audit, split by header group (frontend / execution units / memory subsystem). Findings merged into the plan. | Audit is broad fan-out read-only work — Explore's sweet spot. |
| 1 (test hooks) | Direct. | Single file, surgical. |
| 2 (redirect-flush) | Direct. | Single file, design hinges on conversation context. |
| 3 (gather Wire<bitset>) | Direct. | Contained to one file; design hinges on conversation context. |
| 4 (PulseReg conversions, 7 sites) | Direct. | Multi-file but the discipline (auto-seed-to-T{} + delete consumer clears) is uniform; consistency matters more than parallelism. |
| 5 (instr_buffer → RegFifo) | Direct. | Highest-risk structural change; needs full conversation context. |
| 6 (durable plain members) | **general-purpose agents (opus), parallel by file/finding** once Phase 0 audit has produced independent units. | Mechanical conversions where each finding is contained. Drift between similar conversions caught by consolidation review #2. |
| 7 (scratch → Wire) | **general-purpose agents (opus), parallel by execution unit**. The five execution units share a near-identical scratch pattern; brief is the same for each. | Same mechanical-conversion shape as Phase 6. |
| 8 (delete current_mut + lint) | Direct. | Lint rule design + final delete; surgical and conversation-context-dependent. |

**Reviews:** all three consolidation-reviewer dispatches use `model: opus`.

**Brief template for parallel general-purpose agents (Phases 6 and 7):** include the relevant Phase 0 audit finding(s) verbatim, the file path(s) to modify, the conversion shape (e.g. "wrap `processed_this_cycle_` as `Wire<bool>`; reset at top of evaluate(); replace `processed_this_cycle_ = true` with `wire_processed_.drive(true)`; remove the `// scratch` annotation"), the regression invariant ("byte-identical expected; bench_compare must show zero delta"), and the one-file boundary (the agent does not touch other files). Parallel agents are dispatched in a single message with multiple `Agent` tool calls.

## Per-phase findings

(Populated as phases land. Each phase entry: cycle deltas vs baseline, why, decisions taken.)

### Baseline

```
SHA: 10177f2648b4a878a6a32281c510b58384188bb4
matmul:                  101879
gemv:                      6363
fused_linear_activation:   2621
softmax_row:               2471
embedding_gather:         49002
layernorm_lite:            9933
```

Captured via `bash ./tests/run_workload_benchmarks.sh --build-dir build` on a clean Release build (DRAMSim3 backend not enabled in this build). All 6 benchmarks passed. Working-tree dirty only by `project-plans/goofy-humming-dream.md` (this file).

### Phase 0

- **Delta:** zero across all 6 benchmarks. PulseReg<T> added but unused; audit produced no code changes.
- **ctest:** 31/31 pass (incl. `timing_naming_lint`, `signal_diagram_ast_snapshot`).
- **PulseReg<T>:** added to `sim/include/gpu_sim/timing/reg.h` between `Reg<T>` and `RegFifo<T>`. Inherits `RegBase`; seed-to-T{} semantics documented.
- **Tests:** 8 new test cases in `sim/tests/test_reg.cpp` (`PulseReg: ...` suite) plus one `RegisteredStage` cross-primitive test verifying `PulseReg` enrolls correctly alongside `Reg` / `RegFifo`. All pass.
- **Audit:** consolidated above; ~57 already-conformant members, 13 Phase 7 Wire conversions, 12 Phase 6 Reg/RegFifo conversions (including the known `warp_state.instr_buffer`), 7 Phase 4 PulseReg conversions (matches prior plan scope). Six open classification questions surfaced for operator review before Phase 6/7 execution.

### Phase 1
(to be filled in)

...

## Audit findings — plain members in timing headers

Produced by three parallel `Explore` agents covering frontend / execution-unit / memory-subsystem header groups, classifying every plain (non-`Reg<`, non-`RegFifo<`, non-`Wire<`, non-`const`, non-annotated) member. Findings consolidated below.

### Phase 7 — combinational scratch → `Wire<T>` (16 fields)

Standard `evaluate()`-writes-`commit()`-reads scratch. Mechanical conversion: `Wire<T>` member, `wire_.reset()` at top of evaluate (or implicit via the existing reset disciplines), `wire_.drive(...)` at the assignment site, `wire_.value()` at the read site.

| File:line | Field | Wrap |
|-----------|-------|------|
| `alu_unit.h:140` | `pending_cycle_` | `Wire<uint64_t>` |
| `alu_unit.h:147` | `processed_this_cycle_` | `Wire<bool>` |
| `multiply_unit.h:97` | `busy_this_cycle_` | `Wire<bool>` |
| `multiply_unit.h:98` | `accepted_this_cycle_` | `Wire<bool>` |
| `divide_unit.h:78` | `busy_this_cycle_` | `Wire<bool>` |
| `divide_unit.h:79` | `accepted_this_cycle_` | `Wire<bool>` |
| `tlookup_unit.h:79` | `busy_this_cycle_` | `Wire<bool>` |
| `tlookup_unit.h:80` | `accepted_this_cycle_` | `Wire<bool>` |
| `ldst_unit.h:129` | `next_push_` | `Wire<std::optional<AddrGenFIFOEntry>>` |
| `ldst_unit.h:142` | `busy_this_cycle_` | `Wire<bool>` |
| `ldst_unit.h:143` | `accepted_this_cycle_` | `Wire<bool>` |
| `operand_collector.h:104` | `busy_this_cycle_` | `Wire<bool>` |
| `coalescing_unit.h:51` | `next_pop_` | `Wire<bool>` |

### Phase 6 — durable plain state → `Reg<T>` / `RegFifo<T>` (24 fields)

Real durable cross-cycle state hiding as plain fields. Each is a genuine clocked register / register-array / FIFO in hardware terms. **Strict-compliance policy:** any container that holds data across `tick()` calls wraps, regardless of whether the producer and consumer co-occur within one `evaluate()`. The `MultiplyUnit::pipeline_` precedent (`multiply_unit.h:88` — `Reg<std::deque<PipelineEntry>>` with auto-seed-copy-then-mutate) applies uniformly.

| File:line | Field | Wrap |
|-----------|-------|------|
| **Frontend / control** | | |
| `warp_state.h:9` | `pc` (per-warp PC) | `Reg<uint32_t>` |
| `warp_state.h:10` | `active` (per-warp liveness) | `Reg<bool>` |
| `warp_state.h:11` | `instr_buffer` | `RegFifo<BufferEntry>` (**Phase 5**, structural) |
| `coalescing_unit.h:41` | `processing_` | `Reg<bool>` |
| `coalescing_unit.h:42` | `current_entry_` | `Reg<AddrGenFIFOEntry>` |
| `coalescing_unit.h:43` | `is_coalesced_` | `Reg<bool>` |
| `coalescing_unit.h:44` | `serial_index_` | `Reg<uint32_t>` |
| `coalescing_unit.h:59` | `cmd_in_flight_` | `Reg<bool>` (overrides Explore-agent `Wire` classification; carries across cycle boundary) |
| `writeback_arbiter.h:63` | `committed_` | `Reg<std::optional<WritebackEntry>>` |
| `writeback_arbiter.h:64` | `pending_commit_` | `Reg<std::optional<WritebackEntry>>` |
| `timing_model.h:113` | `cycle_` | `Reg<uint64_t>` (TimingModel becomes `RegisteredStage` or wires the field through `tick()`'s seed/commit machinery) |
| `timing_model.h:119` | `pending_panic_flush_` | `Reg<bool>` (latched at top of tick) |
| **Execution-unit base + LDST** | | |
| `execution_unit.h:233` | `queue_` (writeback queue) | `Reg<std::deque<WritebackEntry>>` (overrides audit "conformant — externally managed"; durable cross-cycle queue) |
| `ldst_unit.h:128` | `addr_gen_fifo_` | `Reg<std::deque<AddrGenFIFOEntry>>` (overrides Phase M1 "deliberate hand-rolled" carve-out under strict compliance; asymmetric-gating semantics achievable by ordering push/pop within the wrapped container) |
| **Memory backends** | | |
| `memory_interface.h:132` | `in_flight_` | `Reg<std::deque<MemoryRequest>>` |
| `memory_interface.h:133` | `responses_` | `Reg<std::deque<MemoryResponse>>` |
| `memory_interface.h:134` | `write_acks_` | `Reg<std::deque<MemoryResponse>>` |
| `dramsim3_memory.h:154` | `request_fifo_` | `Reg<std::deque<PendingChunk>>` |
| `dramsim3_memory.h:158` | `write_chunks_in_fifo_` | `Reg<uint32_t>` (counter tracking FIFO occupancy; durable across cycles) |
| `dramsim3_memory.h:161` | `responses_` | `Reg<std::deque<MemoryResponse>>` |
| `dramsim3_memory.h:164` | `write_acks_` | `Reg<std::deque<MemoryResponse>>` |
| `dramsim3_memory.h:168` | `pending_write_acks_` | `Reg<std::deque<PendingWriteAck>>` |
| `dramsim3_memory.h:171` | `read_assembly_` | `Reg<std::vector<ReadAssembly>>` |
| `dramsim3_memory.h:174` | `read_chunk_to_mshr_` | `Reg<std::unordered_map<uint64_t, uint32_t>>` |

**Performance note:** wrapping the ~10 memory-backend containers and `addr_gen_fifo_` adds a whole-container copy per cycle for each (the auto-seed step). The MultiplyUnit pipeline already pays this cost. The user has explicitly accepted any performance regression as a non-priority — correctness and hardware fidelity are the goal. If profiling later shows a problem, that's a separate optimization concern handled outside this refactor.

### Phase 4 — memoryless-consumer slots → `PulseReg<T>` (7 fields)

Already in the Phase 4 scope; audit confirmed.

| File:line | Field | Wrap |
|-----------|-------|------|
| `cache.h:303` | `load_cmd_` | `PulseReg<LoadCommand>` |
| `cache.h:304` | `store_cmd_` | `PulseReg<StoreCommand>` |
| `memory_interface.h:138` | `read_request_` | `PulseReg<PendingMemoryRequest>` |
| `memory_interface.h:139` | `write_request_` | `PulseReg<PendingMemoryRequest>` |
| `dramsim3_memory.h:199` | `read_request_` | `PulseReg<PendingMemoryRequest>` |
| `dramsim3_memory.h:200` | `write_request_` | `PulseReg<PendingMemoryRequest>` |
| `load_gather_buffer.h:183` | `claim_request_` | `PulseReg<GatherClaimRequest>` |

### Annotation additions (Phase 7 or 8 alongside lint extension)

Fields whose semantics are non-state and stay plain. Three lint-recognized annotation categories — and only these three:

| Category | Meaning | Examples |
|----------|---------|----------|
| `// sim-instrumentation` | Non-clocked accumulator that doesn't model hardware state (monotonic counter, peak observation, free-running fabric cycle). | DRAMSim3 `phase_`, `fabric_cycle_`, `dram_ticks_`, `max_*_queue_`; `ldst_issued_total_`. |
| `// test-only-override` | Set by a test fixture hook, read by the same class across many cycles; not modeling hardware. | Fetch / decode / ALU `*_override_` fields. |
| `// back-pointer` | Wiring to another component (nullable for nullptr-tolerant unit-test patterns); not state. | `decode_->`, `alu_->`, `scoreboard_`, `branch_tracker_`, `ldst_`, `wb_arbiter_`, `opcoll_`, `sim_cycle_`, etc. |

**Specific annotations to add or verify** (most will land in Phase 7 alongside Wire conversions; verification of existing comments matches the lint-recognized form):

| File:line | Field | Annotation |
|-----------|-------|------------|
| `dramsim3_memory.h:134` | `phase_` | `// sim-instrumentation` |
| `dramsim3_memory.h:180` | `fabric_cycle_` | `// sim-instrumentation` (verify; likely already commented) |
| `dramsim3_memory.h:183` | `dram_ticks_` | `// sim-instrumentation` |
| `dramsim3_memory.h:186-187` | `max_response_queue_`, `max_write_ack_queue_` | `// sim-instrumentation` |
| `warp_scheduler.h:211` | `ldst_issued_total_` | `// sim-instrumentation` (verify form) |
| `fetch_stage.h:120-126` | 5 test-hook overrides | `// test-only-override` |
| `decode_stage.h:128` | `redirect_override_` | `// test-only-override` |
| `alu_unit.h:159` | `redirect_override_` | `// test-only-override` |
| `fetch_stage.h:97-104`, `decode_stage.h:96-98`, etc. | All component back-pointers / refs wired post-construction | `// back-pointer` |

References (`Stats& stats_`, `FunctionalModel& func_model_`, `L1Cache& cache_`, etc.) are inherently non-rebindable; the lint recognizes references as not-state without an explicit annotation.

### Deliberate exceptions (potentially empty)

Under strict compliance, there is one open candidate remaining — re-examined and either eliminated or kept as a single named exception:

| File:line | Field | Status |
|-----------|-------|--------|
| `alu_unit.h:174` | `branch_resolved_` | **Re-examination required (Phase 7).** Documented today as "side-effect fires once across multi-cycle stall." Under strict compliance the proposed shape is `Reg<bool>` with auto-seed copying the latched value forward (true persists across a stall — exactly the intended semantics), cleared via `set_next(false)` at the explicit "branch slot advances" point. If this encoding reproduces the existing behavior, `branch_resolved_` becomes `Reg<bool>` and the exception goes away. If a real semantic gap is found, it stays plain with an explicit `// deliberate-non-register: <reason>` annotation listed by fully-qualified name in the lint's exception set (no generic escape hatch). |
| `load_gather_buffer.h:192` | `next_release_` | Defer to Phase 7. Likely `Wire<GatherReleaseRequest>` (asserted during evaluate, consumed within the cycle) — verify the field is reset at evaluate top; if so, `Wire`, no exception. |

**Goal:** the lint's deliberate-non-register exception set is empty or contains at most one explicitly-named field, not a general carve-out.

### Resolved classification questions

1. **Test-only override fields** → `// test-only-override` annotation. Lint-recognized; exempt from primitive-wrapping requirement because they don't model hardware. (Resolved.)
2. **Back-pointers wired post-construction** → `// back-pointer` annotation. Lint-recognized. (Resolved.)
3. **Reference members** → no annotation needed; lint treats `T&` members as inherently non-state. (Resolved.)
4. **Coalescing `cmd_in_flight_`** → `Reg<bool>` (cross-cycle); recorded in the Phase 6 table above. (Resolved.)
5. **`LoadGatherBufferFile::next_release_`** → defer to Phase 7; likely `Wire<GatherReleaseRequest>`. (Resolved-as-deferred.)
6. **Internal scheduling state** (memory-backend deques + `addr_gen_fifo_` + `queue_`) → **all wrap as `Reg<std::deque<T>>` / `Reg<std::vector<T>>` / `Reg<std::unordered_map<...>>`** under strict-compliance policy. No `// internal-scheduling-state` carve-out category exists; the previously-considered exemption is rejected. Performance regressions from per-cycle container copies are acceptable — correctness and hardware fidelity are the priority. (Resolved.)

### Audit totals (strict-compliance final)

- Already conformant: ~57 members (Reg / RegFifo / Wire / annotated).
- **Phase 4 `PulseReg<T>` conversions:** 7 fields.
- **Phase 5 structural conversion:** 1 field (`warp_state.instr_buffer` → `RegFifo<BufferEntry>`).
- **Phase 6 `Reg<T>` / `RegFifo<T>` conversions:** 24 fields (frontend / control / execution-unit base / LDST FIFO / memory-backend containers).
- **Phase 7 `Wire<T>` conversions:** 13 mechanical + 1 deferred (`next_release_`) + 1 candidate (`branch_resolved_` → `Reg<bool>` if encoding reproduces; not a Wire but examined in Phase 7).
- **Annotation additions:** ~9 `// sim-instrumentation`, 7 `// test-only-override`, ~10+ `// back-pointer`. Three categories total, all lint-recognized.
- **Deliberate-non-register exceptions:** target empty; `branch_resolved_` examined and converted to `Reg<bool>` if possible.

The three annotation categories (`// sim-instrumentation`, `// test-only-override`, `// back-pointer`) plus references are the **only** legitimate non-primitive plain-field forms after Phase 8. Anything else in a timing header is a lint error.

## Doc sync

Per `CLAUDE.md` trigger table:

- **`resources/timing_discipline.md`** — REQUIRED. Document `PulseReg<T>` as the canonical primitive for one-cycle pulse / memoryless-consumer slots. Strengthen the four-kinds taxonomy as the explicit rule (durable→Reg-family, transient→Wire, config→const, instrumentation→annotated). Remove or rewrite any text that endorsed `current_mut()` patterns.
- **`resources/cpp_coding_standard.md`** — REQUIRED. Update the timing-state-shape rules. Every non-const, non-instrumentation member of a timing class must be `Reg`/`RegFifo`/`PulseReg`/`Wire`. The `// scratch` annotation is replaced by `Wire<T>`.
- **`resources/perf_sim_arch.md`** — REQUIRED. New primitive `PulseReg<T>` in `reg.h`; per-stage type changes where the audit re-shapes members.
- **`AGENTS.md` / `CLAUDE.md`** — verify; likely no edit (this phasing doc is not a permanent reference artifact). If kept after completion, add it to Key References; otherwise delete after Phase 8.
- **`project-plans/reg-fidelity-audit.md`** — supersede or delete. The fidelity-audit task was explicitly separate from the byte-identical migration; this refactor does the part the audit pointed at (eliminating same-cycle-write-to-committed-state). After Phase 8, the audit checklist is either obsolete or reduces to whatever `.next()` reads remain (none of which should now be cross-class).
- **`UNTESTED.md`** — no entry; the refactor preserves existing behavior (modulo documented per-phase deltas), so the existing regression suite is the contract.
- **`tools/lint_timing_naming.py`** docstring + `tests/test_signal_diagram.py` doc comment — update to mention the strengthened taxonomy and `PulseReg<T>`.

## Critical files

**Create:**
- `project-plans/goofy-humming-dream.md` (this file).

**Modify (largest scope):**
- `sim/include/gpu_sim/timing/reg.h` — add `PulseReg<T>`; delete `current_mut()` from `Reg<T>` (Phase 8); rewrite header comment block.
- `sim/include/gpu_sim/timing/fetch_stage.h` and `sim/src/timing/fetch_stage.cpp` — Phase 2.
- `sim/include/gpu_sim/timing/decode_stage.h` and `sim/src/timing/decode_stage.cpp` — Phase 5.
- `sim/include/gpu_sim/timing/warp_state.h` (and any `instruction_buffer.h`) — Phase 5.
- `sim/include/gpu_sim/timing/warp_scheduler.h` and `sim/src/timing/warp_scheduler.cpp` — Phase 1 (test hooks); Phase 5 (instr_buffer pop).
- `sim/include/gpu_sim/timing/cache.h` and `sim/src/timing/cache.cpp` — Phase 4 (load_cmd_ / store_cmd_ → `PulseReg`).
- `sim/include/gpu_sim/timing/memory_interface.h` and `sim/src/timing/memory_interface.cpp` — Phase 4.
- `sim/include/gpu_sim/timing/dramsim3_memory.h` and `sim/src/timing/dramsim3_memory.cpp` — Phase 4.
- `sim/include/gpu_sim/timing/load_gather_buffer.h` and `sim/src/timing/load_gather_buffer.cpp` — Phase 3 (Wire<bitset>); Phase 4 (claim_request_ → `PulseReg`).
- `sim/include/gpu_sim/timing/alu_unit.h`, `multiply_unit.h`, `divide_unit.h`, `tlookup_unit.h`, `ldst_unit.h`, `operand_collector.h` — Phase 7 (scratch → Wire); Phase 6 (audit findings).
- `tools/lint_timing_naming.py` — Phase 4 (recognize `PulseReg`); Phase 8 (new taxonomy rule, `current_mut(` ban).
- `sim/tests/test_reg.cpp` — Phase 0 (`PulseReg<T>` tests).
- `sim/tests/test_warp_scheduler.cpp` — Phase 1 (verify seed_all calls).

**Documentation (per Doc Sync above):**
- `resources/timing_discipline.md`, `resources/cpp_coding_standard.md`, `resources/perf_sim_arch.md`, `project-plans/reg-fidelity-audit.md`.

## Verification

**Per phase:**
1. `cmake -B build && cmake --build build -j8` — must succeed.
2. `ctest -j8` from `build/` — all targets pass (Catch2 binaries + `timing_naming_lint` + `signal_diagram_ast_snapshot`).
3. `bash ./tests/run_workload_benchmarks.sh --build-dir build` — capture `RESULT` lines.
4. `python3 tools/bench_compare.py --baseline <baseline-sha>` — record deltas in this file's "Per-phase findings" section. Most phases expected byte-identical; any delta documented with the benchmark, the cycle change, the field/site, and the mechanism (per `reg-fidelity-audit.md` framing: revealed pre-existing fidelity bug, not a migration mistake).
5. `tests/riscv-isa` and `tests/synthetic` — must remain green.
6. **Phase 4 (DRAMSim3 backend touched):** build and test both `-DGPU_SIM_USE_DRAMSIM3=ON` and `-DGPU_SIM_USE_DRAMSIM3=OFF`.

**End-to-end (after Phase 8):**
- Re-grep for `current_mut` across the tree — should return only the lint comment / docstring describing why the method doesn't exist.
- Spot-check the lint: introduce a plain `bool test_field_;` to a timing header; confirm lint flags it; revert.
- `python3 tests/test_signal_diagram.py` — the snapshot may need a refresh for new Wire accessors (`just_claimed_`, any others); inspect failures, update the `expected` set in `tests/test_signal_diagram.py`, re-run.

## Out of scope (explicit)

- Functional model (`sim/src/functional/`) — not cycle-accurate; not touched.
- Runner (`runner/`) — not affected.
- The reg-fidelity-audit document's `next()`-read audit — this plan addresses the *write*-side anti-pattern (current_mut), not the read-side (`.next()` reads that should be `.current()`). Those are still in scope for the original audit task; the surface is narrower now that all durable state is registered.
- Performance optimization — `Wire<T>`-ifying every scratch member has marginal runtime cost; cost is uniformly paid post-refactor.
