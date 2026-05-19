# Memory-system series — deviation & assumption log

Record of every deviation from the three plans, every significant assumption,
and every interpretation made while implementing the coordinated series:

- `registered-tag-array.md` → commit `4293f8c`
- `eager-wobbling-pizza.md` → commit `e09bc01`
- `registered-mshr-write-buffer.md` → commit `7853076`
- DRAMSim3 write-ack folding fix → commit `929778e` (gate-caught defect)

Branch `memory-system-registered-series`, off baseline `add9b59`. Recommended
landing order was followed exactly (P1 → P2 → P3); no ordering deviation.

Tags: **[Deviation]** = departs from the plan text · **[Refinement]** = goes
beyond the plan text without contradicting it · **[Interpretation]** = resolves
a plan ambiguity/omission · **[Assumption]** = a value/fact taken as given ·
**[Process]** = workflow/tooling, not the modelled design.

---

## Plan 1 — registered-tag-array (`4293f8c`)

### 1.1 `fill_installed_set_` is reset in `commit()` as well as at evaluate()-top — **[Refinement]**
Plan Step 4: "Reset to −1 at the top of `evaluate()` alongside `stalled_`."
Implemented: reset at evaluate()-top **and** in `commit()`.
Rationale: the field is classified COMBINATIONAL same-tick scratch; clearing it
at the tick boundary makes that lifetime explicit and is required for the
direct-API test contract (a test that calls `handle_responses()` then
`process_*()` directly, with no `evaluate()` between, would otherwise observe a
leaked scratch value). Production-equivalent — `evaluate()`'s top-of-tick reset
governs the production path; the `commit()` reset is redundant there. Surfaced
to the user at the time as a Step 4 refinement.

### 1.2 `drain_secondary_chain_head`'s tag read — corrected to `current_tags_` post-review — **[Deviation, since corrected]**
Plan Step 2 enumerated every reader to switch to `current_tags_` and
**omitted** `drain_secondary_chain_head`'s tag read; Step 3 routed that
function's pin-clear *write* to `next_tags_`.
As first implemented (at commit `4293f8c`, Plan 1 in isolation) the read was
left on `next_tags_`. The MSHR file was still direct-mutated then, so
`complete_fill`'s primary `free` was immediately visible and a `current_tags_`
read *would* have shifted the first chain drain by a cycle and broken
baselined tests — which made `next_tags_` look load-bearing, and the Step-2
omission was read as deliberate.
Once Plan 3 (`7853076`) registered the MSHR file, that justification went
stale: the registered `free` keeps the primary valid in `current_entries_`
through the fill cycle, so `drain_secondary_chain_head`'s head-detection scan
already gates any secondary of the just-filled line out of the fill cycle.
Reading `current_tags_` vs `next_tags_` then became observationally identical
— but the `next_tags_` read remained: a same-cycle read of `complete_fill`'s
registered next-state, and the only tag reader not on `current_tags_`.
A post-series review caught this. **Corrected:** the read is now
`current_tags_` (the pin-clear write stays `next_tags_`, per Step 3). Full
regression stays green, confirming the observational equivalence. The stale
code / `timing_discipline.md` / `perf_sim_arch.md` comments that asserted the
`next_tags_` read was required are corrected in the same change.

### 1.3 Plan Step 3 parenthetical — **[Interpretation, updated post-review]**
Plan Step 3 (parenthetical): "Confirm `complete_fill` and
`drain_secondary_chain_head` cannot write the same `next_tags_[S]` in one cycle
… so they target different sets."
At Plan 1's isolated commit they *could* coincide on `next_tags_[S]` — a
primary fill and that same line's last-secondary-drain in one cycle
(`complete_fill` writes `{valid,tag,pinned=true}`, `drain_secondary_chain_head`
then writes `pinned=false`); harmless, the writes being tick-ordered and
complementary. In the final tree this cannot happen at all: with the MSHR file
registered (Plan 3), the primary stays valid in `current_entries_` through its
fill cycle, so the head-detection scan keeps any secondary of the just-filled
line from draining that cycle (the same fact that made the 1.2 fix
observationally inert). `complete_fill` and `drain_secondary_chain_head`
therefore never write the same `next_tags_[S]` in one cycle — the plan's
conclusion holds outright. No code consequence.

### 1.4 Existing direct-API tests re-baselined — **[Deviation, plan-anticipated]**
Plan "Related concerns / Test-direct-API note" predicted this. Tests in
`test_cache.cpp` and `test_cache_mshr_merging.cpp` that interleave
`handle_responses()` / `process_*()` directly needed `cache.commit()` /
`cache.evaluate()` inserted at cycle boundaries (the tag array is now
REGISTERED). `test_cache_mshr_merging.cpp` Case 13: two `pinned_line_count()`
assertions moved to *after* a commit (pin state is observable only
post-commit). No assertion *values* changed.

---

## Plan 2 — eager-wobbling-pizza (`e09bc01`)

### 2.1 `max_outstanding_writes` default = 32 — **[Assumption]**
Plan Step 1: "A sensible default is `>= write_buffer_depth` and large enough not
to bottleneck store throughput (sweep the benchmark suite)."
Chosen: **32**. It is `>= write_buffer_depth` (4) and comfortably above the
external-memory latency window, so it does not bottleneck. A full benchmark
sweep to confirm the optimum was **not** performed — 32 is a defensible,
non-bottlenecking starting value; flagged as a sweep candidate.

### 2.2 `validate()` *rejects* `max_outstanding_writes < write_buffer_depth` — **[Interpretation]**
Plan Step 1: "`validate()` … should **warn (or reject)**."
Chosen: hard reject (throw), consistent with how `SimConfig::validate()` handles
every other constraint. Also rejects `max_outstanding_writes < 1`.

### 2.3 `any_pinned_tag()` rewritten as an index loop — **[Refinement]**
Plan Step 3 said to swap `tags_[set].pinned` → `is_pinned(set)`. `is_pinned`
takes a set index, but the old `any_pinned_tag()` iterated tag entries by value.
Rewrote it as a `for (set …)` loop calling `is_pinned(set)`. The old loop's
explicit `t.valid` check was dropped (validity is implied by the pin invariant).
Consistent with plan intent.

### 2.4 Stall attribution added at `complete_fill`'s deferred-fill guard — **[Refinement]**
Plan Step 7 said to attribute the stall (`line_pin_stall_cycles` vs
`write_ack_pin_stall_cycles`) "at the 3 pin-enforcement sites." `complete_fill`'s
deferred-fill guard previously bumped `line_pin_stall_cycles` unconditionally; it
now does the chain-pin > write-ack-pin attribution like the other two sites.
Within plan intent (it *is* one of the three sites).

### 2.5 Per-test-file `MAX_OUTSTANDING_WRITES` constants — **[Assumption]**
Test fixtures construct `L1Cache` directly. Picked `MAX_OUTSTANDING_WRITES = 32`
for `test_cache.cpp` / `test_cache_mshr_merging.cpp` and `64` for
`test_coalescing.cpp` / `test_load_gather_buffer.cpp` (larger `WB_DEPTH` there).
Assumption: each is large enough not to perturb the existing tests; the cache
constructor only asserts `>= 1`.

### 2.6 A Plan-1 test re-baselined inside Plan 2's commit — **[Deviation, plan-anticipated]**
`test_cache.cpp` "store hit racing an evicting fill" (added in Plan 1) used a
store-*miss* as the evicting fill. After Plan 2 a store-miss fill write-ack-pins
its set, so the racing store would stall `LINE_PINNED` instead of retrying into
a clean miss. The test's evicting fill was changed to a load-miss (installs the
line unpinned) so the test's original intent holds. Re-baselined within `e09bc01`
because Plan 2 is what changed the behaviour.

### 2.7 Other existing-test re-baselining for the write-ack channel split — **[Deviation, plan-anticipated]**
`test_timing_components.cpp` "fixed latency responses preserve submission order"
— reads/writes now arrive on separate channels (`get_response` vs
`get_write_ack`). `test_dramsim3_memory.cpp` "idle goes true after drain" — write
completions drained via `get_write_ack`. The worst-case stress test rewritten to
model the outstanding-write cap and assert both split queue bounds. All plan-
predicted ("re-baseline any test asserting exact cycles around a pinned
conflict").

### 2.8 Lint allowlist updated for the new accessors — **[Process]**
`tools/lint_timing_naming.py`: added `write_ack_count`,
`read_response_queue_capacity`, `max_observed_write_ack_queue`,
`write_ack_queue_capacity` to the `LIFECYCLE_METHODS` allowlist and removed the
renamed `response_queue_capacity`. The plan required the lint to accept the new
accessors; these are scalar telemetry/capacity reporters — the allowlist's
documented category.

### 2.9 `chunks_per_line >= 2` — **[Assumption, confirmed]**
Plan Step 4 asked to confirm `chunks_per_line >= 2` for the write-ack-queue
cushion. `chunks_per_line = 128/32 = 4` ✓.

### 2.10 Outstanding-write-cap overshoot — **[Assumption, confirmed]**
Plan notes: `outstanding_writes_total_` can transiently overshoot the cap by ≤2
until Plan 3's single-enqueue-port lands (0 after). Confirmed: after Plan 3,
`queue_write_through` is single-port → ≤1 enqueue/cycle → overshoot 0.

---

## Plan 3 — registered-mshr-write-buffer (`7853076`)

### 3.1 Plan's site enumerations treated as illustrative, not exhaustive — **[Interpretation]**
Plan A3 listed some MSHR read/write call sites. The implementation routed **all**
reads → `current_at` and **all** writes → `next_at`, including sites the plan did
not name: `secondary_mshr_count`, `active_mshr_count`, `active_mshr_warps`, and
`complete_fill`'s `auto& mshr` / `drain_secondary_chain_head`'s `auto& cand`
(both became `const auto&` because `current_at` returns `const`). Assumption: the
plan's list was illustrative; the at()-split makes the const/non-const routing
mandatory regardless.

### 3.2 `timing_model.cpp` not in the plan's file list — **[Interpretation]**
`timing_model.cpp`'s trace recording reads `cache_->mshrs().at(i)`. The plan's
Critical-files list did not mention `timing_model.cpp`, but the `at()` → split
forced the change; routed to `current_at` (post-commit committed read). Necessary
consequence of the API split.

### 3.3 Call-site ordering: chose "option (a)" for the fallible `queue_write_through` — **[Interpretation]**
Plan B1 offered two ways to make the fallible enqueue leave no partial state:
(a) call `queue_write_through` first and perform side effects only on success,
or (b) add a `write_buffer_can_enqueue()` pre-check. Chose **(a)** at all three
callers; `write_buffer_can_enqueue()` was therefore **not** added. (a) is the
smaller surface and needs no extra method.

### 3.4 `write_buffer_port_conflict_cycles` — dedicated counter added — **[Refinement]**
Plan B Critical-files: "optional `write_buffer_port_conflict_cycles` counter …
reuse `write_buffer_stall_cycles` if a separate counter is not wanted." Added the
dedicated counter, consistent with the per-reason observability used in Plans 1–2.

### 3.5 `next_write_buffer_port_claimed_` cleared at `commit()` only — **[Interpretation]**
Plan B1 called it "a REGISTERED scratch reset at commit," and B2 said `commit()`
clears it. Implemented exactly that — cleared only at `commit()`, modelled on
`LoadGatherBufferFile::next_port_claimed_` (the gather-buffer port flag, which
`timing_discipline.md` row 11 classifies as a single intra-cycle first-writer-
wins claim cleared at commit). It is *not* additionally reset at evaluate-top;
tests must drive full cycles, consistent with the gather-buffer model.

### 3.6 Lint: `INTERNAL_HELPER_CLASSES` exemption added for `MSHRFile` — **[Process]**
`tools/lint_timing_naming.py`'s cross-module-read layer flagged
`mshrs_.next_at(...)` because `MSHRFile` is not in `MODULE_ORDER`. `MSHRFile` is
*explicitly* "out of scope for the cross-stage accessor convention" per the
lint's own `TIMING_MODULES` doc — but only the prefix layer respected that; the
cross-module layer caught any `next_*`-named call. Added an
`INTERNAL_HELPER_CLASSES = {"MSHRFile"}` set and a matching skip in
`_classify_cross_module_read`. `next_at` is an intra-module registered-pair
write handle, not a cross-*stage* edge. This is how the plan's "lint must accept
the new current_/next_ MSHR pair" was satisfied.

### 3.7 Large existing-test re-baseline delegated to a sub-agent — **[Process]**
The registered MSHR file + registered write buffer made many direct-API tests
need `commit()`s inserted at cycle boundaries (an MSHR allocated in cycle T is
visible only at T+1; a chain primary must be committed before a same-line
secondary's `find_chain_tail` can see it; `write_buffer_size()` reads committed
state). The mechanical re-baseline of `test_cache.cpp`,
`test_cache_mshr_merging.cpp`, `test_load_gather_buffer.cpp` was delegated to a
general-purpose sub-agent under strict guardrails. Verified afterward: the diff
contains only `commit()`/`evaluate()` insertions and reordering — **zero**
assertion-value changes (confirmed by diffing every `REQUIRE`/`CHECK` line) — and
all 30 tests pass. The fixture `tick_mem()` helpers gained a `cache.commit()` at
their head.

---

## Deadlock fix — DRAMSim3 write-ack folding (`929778e`)

### 4.1 A fourth commit exists at all — **[Deviation]**
The task structure was "three plans, three atomic commits." The final benchmark
gate caught a hang: `embedding_gather` and `layernorm_lite` never reached
`is_idle()` under the DRAMSim3 backend (5,000,000-cycle cap). Per the task rules
("on any unexpected test regression or plan/code inconsistency — STOP and
report; do not improvise"), the work stopped and the defect was reported with
its root cause; the user then directed the fix. It is a fourth commit, not part
of any plan's atomic commit — a deliberate, user-approved departure from the
three-commit structure, necessitated by a gate-caught defect.

### 4.2 The defect is rooted in a Plan 2 gap — **[Deviation surfaced]**
Plan 2 Step 4: "`FixedLatencyMemory` already emits write responses … 
`DRAMSim3Memory` likewise." This premise is **false** for DRAMSim3:
`DRAMSim3Memory` folds multiple in-flight same-line write-throughs into one
`write_assembly_` slot and emits one ack for them; the per-write-through
`outstanding_writes_` counter then never returns to zero → the set's write-ack
pin sticks → permanent `LINE_PINNED` → hang. Plan 2's "No underflow" note
assumed a strict 1:1 ack↔enqueue relationship DRAMSim3 violates. The doc text
written during Plan 2 (`gpu_architectural_spec.md` §5.6, "folded into a single
write ack") was the visible symptom — self-inconsistent with the 1:1 counter —
and was corrected in `929778e`.

### 4.3 Fix design: synthetic per-`submit_write` write ack — **[Deviation, user-approved]**
Investigation of the vendored DRAMSim3 source (`controller.cc`) established that
DRAMSim3's write callback fires at `complete_cycle = clk_ + 1` (it models a
*posted* write) — it is a submit+1 acceptance event, **not** a durability
signal, and DRAMSim3 exposes no durable-write event. So the callback cannot be
the ack source. The fix, designed with and approved by the user: ignore
DRAMSim3's write callback (now a no-op; write chunks are still issued to DRAMSim3
for bank/bus contention) and have `submit_write` schedule exactly **one
synthetic write ack per call**, released a fixed DRAM-cycle commit latency later.
1:1 with the cache's outstanding-write counter by construction.

### 4.4 `dramsim3_write_commit_latency_tck` default = 30 tCK — **[Assumption]**
Derived from the DDR3-800 DE-10 Nano `.ini`: for a 4-chunk (128 B) line,
`CWL(5) + 3·tCCD_S(12) + BL/2(4) + tWR(6) = 27` tCK with the row open, `+tRCD(6)
= 33` tCK with an activation — ~30 tCK representative (≈12 fabric cycles at the
400/150 clock ratio). This is an explicit modelling **assumption**: a *fixed*
latency cannot capture variable write-buffer queueing, and DRAMSim3 exposes no
variable durable-write time, so a synthetic constant is unavoidable. Flagged as a
sweep candidate (sane range derived as ~12–17 fabric cycles). The parameter is
denominated in DRAM cycles (tCK) so it is stable under fabric/DRAM clock-ratio
changes.

### 4.5 Synthetic timer anchored at `submit_write` — **[Interpretation]**
The user's framing was "from the time the request is issued to the DRAM."
`submit_write` is when the write enters the DRAM model (chunks pushed to
`request_fifo_`); the chunks reach DRAMSim3's `AddTransaction` a few DRAM ticks
later as the FIFO drains. The timer is anchored at `submit_write`
(`release_dram_tick = dram_ticks_ + write_commit_latency_tck_`) rather than at
the first/last chunk's `AddTransaction` — simpler, and the `request_fifo_`
residency is small. A minor anchor-point choice within the agreed design.

### 4.6 Dead code removed — **[Refinement]**
`write_assembly_`, `write_chunk_to_line_`, `on_write_complete`, and the
`WriteAssembly` struct existed only for the old per-line write-completion ack;
they are removed. The DRAMSim3 write callback is registered as a no-op lambda.
`is_idle()` / `in_flight_count()` now track `pending_write_acks_` instead of
`write_assembly_`.

### 4.7 `validate()` requires `dramsim3_write_commit_latency_tck >= 1` — **[Refinement]**
Degenerate guard — a latency of 0 would be a submit-time ack, which the design
explicitly rejects.

---

## Cross-cutting / process

### 5.1 Default `max_outstanding_writes` and `dramsim3_write_commit_latency_tck` are un-swept — **[Assumption]**
Both defaults (32; 30 tCK) are derived/defensible starting values. Neither was
validated by a full benchmark sweep. Both are flagged in their `config.h`
comments and here as recalibration candidates — exactly as
`external_memory_latency_cycles = 17` was once calibrated.

### 5.2 Consolidation review scope — **[Process]**
The consolidation review was run (opus) over the three-plan range
`add9b59..7853076`, before the fourth (fix) commit. It was **not** re-run over
the full four-commit range. Its findings — stall-attribution triplication at the
3 pin-enforcement sites and write-buffer-admission triplication at the 3 enqueue
callers — were each written per the plans' explicit per-site specification and
were left as-is (not refactored: "be faithful to the plans; do not improvise").
They are logged as optional future cleanups. The fix commit `929778e` is net
dead-code-removing and was not separately reviewed.

### 5.3 Build assumes DRAMSim3 is enabled — **[Assumption, verified]**
The DRAMSim3 backend changes assume `GPU_SIM_USE_DRAMSIM3=ON`. Verified from
`build/CMakeCache.txt` (it is ON by default and was ON throughout).

### 5.4 Git index corruption during bisection — **[Process]**
While bisecting the hang, a stray `git checkout <sha> -- .` invoked with
`--git-dir=/workspace/.git` corrupted `/workspace`'s git **index** (left it
pointing at `e09bc01`'s tree). Detected via `git status` showing `MM` on every
file. Repaired with `git reset` (index → HEAD); the commits and the working tree
were intact. No code impact — purely a workflow mishap, logged for completeness.

### 5.5 `project-plans/*.md` and `package-lock.json` left uncommitted — **[Deviation, deliberate]**
The three input plan files (`registered-tag-array.md`,
`registered-mshr-write-buffer.md` untracked; `eager-wobbling-pizza.md` modified)
and the pre-existing untracked `package-lock.json` were **not** committed — they
are orchestration inputs / pre-existing artifacts, not part of the code change.
`git status` is therefore not bare-clean. This `deviation-log.md` is likewise
left uncommitted unless you direct otherwise. No `.gitignore` change was needed
(no new build-artifact sources were introduced).

### 5.6 New `Stats` counters — **[Refinement, plan-specified]**
Added across the series: `fill_conflict_retry_cycles` (P1),
`write_ack_pin_stall_cycles` + `write_throttle_stall_cycles` (P2),
`write_buffer_port_conflict_cycles` (P3). All wired into `stats.h`,
`Stats::report`, and `Stats::report_json`, and documented in
`trace_and_perf_counters.md`. `write_buffer_port_conflict_cycles` was the only
optional one (see 3.4).

### 5.7 `<algorithm>` include added to `cache.cpp` — **[Refinement]**
`L1Cache::reset()` uses `std::fill` to zero the outstanding-write vectors;
`#include <algorithm>` was added accordingly.

---

## Net behavioural shifts (all plan-predicted, for reference)

- Registered tag array: a load/store racing a same-set fill retries one cycle
  (`fill_conflict_retry_cycles`).
- Write-ack pin: a queued write-through pins its set until the write ack;
  conflict-heavy / store-heavy workloads pay write-latency stalls.
- Registered MSHR file: a slot freed in cycle T is reusable only at T+1;
  chain-drain start shifts +1 cycle.
- Registered write buffer: no longer fall-through (enqueue visible at T+1);
  single enqueue port (FILL > secondary > HIT).
- Final benchmark deltas vs `add9b59` (DRAMSim3): matmul +0.2%, gemv −5.5%,
  fused_linear_activation −1.1%, softmax_row −1.2%, embedding_gather +3.5%,
  layernorm_lite +0.6% — all within the plans' stated IPC-cost expectation.
