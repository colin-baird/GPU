# Plan: Make the L1 cache tag array a registered structure

## Context

The L1 cache tag array `tags_` (`std::vector<CacheTag>`, each entry `{valid, tag, pinned}`) is
currently **direct-mutated**: `complete_fill` writes it and `process_load`/`process_store` read
it in the same `evaluate()`, with `complete_fill` (in `handle_responses`) running first in tick
order. This makes a same-cycle fill visible to a same-cycle lookup — an implicit, undocumented
**write-first tag array** assumption. It was flagged as a pre-existing concern in the write-ack
pinning plan (`eager-wobbling-pizza.md`, "Related pre-existing concerns"): if RTL built the tag
SRAM read-first, a lookup racing an evicting fill would behave differently from the simulator.

This plan makes `tags_` a proper **REGISTERED `current_`/`next_` pair**, consistent with the
synchronous-pipeline discipline (every reader in cycle T sees the value latched at the start of
T; writes latch for T+1). "The cache array" here means the tag array specifically — `mshrs_`
and `write_buffer_` are separate structures and out of scope (see Related concerns).

## The arbitration problem and the chosen resolution

With a registered tag array, a command reads `current_tags_` while a fill writes `next_tags_` —
they no longer collide on access, but a command racing a fill **to the same set** computes its
hit/miss decision against pre-fill state. The cases:

- **Load hit racing an evicting fill** — the load reads the pre-fill `current_tags_` and "hits"
  the line resident at the start of the cycle. Benign for the load *in isolation* — a load hit
  has no cross-cycle tag side effect — but see the next point and the resolution.
- **Command to the just-filled line, racing its own fill** — stale read → spurious miss. For a
  store this reopens the RAW hazard (below). For a load the miss is **not** merely wasteful: a
  load miss allocates an MSHR and, via `find_chain_tail`, may chain onto an existing same-line
  MSHR — and in the fill cycle the line's primary MSHR is being freed.
- **Store hit racing an evicting fill** — the store would "hit" a line being evicted and
  `queue_write_through` a pin for it; after commit the set holds the *other* line, so the
  write-ack pin protects the wrong line and the store→load RAW hazard reopens.

**A load miss is not side-effect-free.** An earlier draft of this plan retried stores only, on
the premise that a load "reading the registered pre-fill `current_tags_` is always correct"
because a load has no cross-cycle side effect. That premise is wrong. A load *miss* allocates an
MSHR and may chain onto an existing same-line MSHR via `find_chain_tail` — a cross-cycle side
effect predicated on a stable MSHR/tag view. In the exact cycle a fill completes a line's *lone
primary* MSHR, a fresh load to that line reads the pre-fill tags (miss), and `find_chain_tail`
— scanning the registered `current_entries_` once `registered-mshr-write-buffer.md` lands —
still sees the just-freed primary as a valid chain tail. The load is then allocated as a
*secondary* chained onto a primary that is freed at commit; because a lone-primary fill installs
the line **unpinned**, `drain_secondary_chain_head` never drains the orphan → MSHR leak, the
load result is never delivered, and the sim never reaches `is_idle()`. It is a hang, not a
wrong-data result, and it appears only once both the tag array and the MSHR file are registered.

Resolution: **fill priority + all-command retry.**

- **Fills always win.** A fill only writes `next_tags_` and is never blocked by a command (it
  can still be deferred by a *pin* — existing behavior, unchanged). The fill / response path is
  therefore never stalled by a lookup.
- **Any command racing a fill to its set is rejected and retried — load *and* store.** When
  `complete_fill` installs into set S this cycle, a load *or* store command to S is refused
  (`next_cmd_ready_ = false`) and coalescing's existing valid/ready handshake re-stages it for
  next cycle. On the retry the command reads the committed post-fill `current_tags_` (and, for a
  miss, the committed `current_entries_` with the primary already freed): a command to the
  just-filled line hits; a command to a conflicting line misses cleanly and allocates a fresh
  primary. The registered design *necessarily* needs this collision stall — registering the tag
  array and the MSHR file removes the implicit same-cycle forwarding (the "write-first tag
  array" and the same-cycle MSHR free→reuse) that used to absorb the collision silently; the
  retry is what replaces it.

This was chosen over "hits win": if a store hit could win and pin its set, a conflicting fill at
the head of the response path would be deferred for the **entire write-through duration** (until
the write ack clears the pin), blocking all other fills. Fill-priority + retry has no such dead
end — the fill always completes immediately and a racing command pays at most a one-cycle retry.

**No coalescing-unit changes are required.** Confirmed in `coalescing_unit.cpp`: `evaluate()`
Step 1 advances `current_entry_`/`serial_index_` only when `cache_.next_cmd_ready()` is true;
Step 3 re-stages the same command every cycle on a hold. A `false` return from
`process_load`/`process_store` already drives `next_cmd_ready_ = false` and triggers the
re-stage. The new fill-conflict bail just adds one more `false`-return condition to *both*
`process_load` and `process_store`.

## Implementation

> Line numbers below are relative to the pre-change tree. These three memory-system plans land
> in sequence, so a plan landing after another should locate sites by symbol, not by line.

### Step 1 — `tags_` becomes a registered pair
`sim/include/gpu_sim/timing/cache.h`, `sim/src/timing/cache.cpp`
- Replace `std::vector<CacheTag> tags_;` with `current_tags_` and `next_tags_`, both sized
  `num_sets_` in the constructor initializer list.
- `evaluate()` seed_next (cache.cpp ~line 398): add `next_tags_ = current_tags_;`.
- `commit()` (~line 509): add `current_tags_ = next_tags_;`.
- `reset()`: reset both vectors.

### Step 2 — readers read `current_tags_`
`sim/src/timing/cache.cpp`
Swap `tags_[set]` → `current_tags_[set]` at every read site: `process_load` hit-check and
pin-check (~lines 35, 57), `process_store` hit-check and pin-check (~lines 122, 138),
`complete_fill` deferred-fill guard (~line 208), `any_pinned_tag` (~line 482),
`pinned_line_count` (~line 579), and `is_pinned` (the helper introduced by the write-ack pinning
plan — `current_tags_[set].pinned`).

### Step 3 — writers write `next_tags_`
`sim/src/timing/cache.cpp`
Swap `tags_[set]` → `next_tags_[set]` at every write site: `complete_fill` install of
`valid`/`tag`/`pinned` (~lines 236-238 store path, ~247-249 load path), and
`drain_secondary_chain_head` pin-clear `pinned = false` (~lines 344, 371).
(Confirm `complete_fill` and `drain_secondary_chain_head` cannot write the same `next_tags_[S]`
in one cycle: a different-tag fill into a set with a draining chain is deferred by the pin, and
the chain's own line has no second fill — so they target different sets.)

### Step 4 — fill-conflict detection and command retry
`sim/include/gpu_sim/timing/cache.h`, `sim/src/timing/cache.cpp`
- Add a COMBINATIONAL same-tick scratch `int32_t fill_installed_set_ = -1;` (−1 = no fill this
  cycle). Reset to −1 at the top of `evaluate()` alongside `stalled_` etc.
- In `complete_fill`, on a successful (non-deferred) install, set `fill_installed_set_ = static_cast<int32_t>(set)`.
  (The deferred-fill early-return path does not set it.)
- In **both** `process_load` and `process_store`, immediately after computing `set`, add:
  `if (fill_installed_set_ == static_cast<int32_t>(set)) return false;` — the command is not
  ready this cycle; coalescing re-stages it. This precedes any tag read, MSHR access, side
  effect, or trace event, so a retried command leaves no partial state. The check is identical
  for loads and stores: a load miss is *not* side-effect-free (it allocates/chains an MSHR — see
  the resolution above), so a load racing a same-set fill must retry exactly as a store does.
- The retry is set-granular and unconditional: it fires for *every* successful fill install into
  set S, whether or not the installed line is pinned. The unpinned (lone-primary) case is the
  one that must be covered for correctness; covering the pinned case too is harmless (the
  command simply retries into a hit) and keeps the rule simple.
- Only `complete_fill`'s install (a `valid`/`tag` change) triggers retry. A
  `drain_secondary_chain_head` pin-clear changes only `pinned`; with the registered array the
  command simply observes the pin one cycle longer (conservative, correct) — no retry needed.

### Step 5 — observability
`sim/include/gpu_sim/stats.h`, `sim/src/timing/cache.cpp`
Add `uint64_t fill_conflict_retry_cycles = 0;` to `Stats`; bump it when a command — load or
store — bails for the fill-conflict reason. No new `CacheStallReason` (keep blast radius small,
consistent with the write-ack pinning plan's observability decision). The fill-conflict bail
returns `false` **without** setting `stalled_` / `stall_reason_`: the retry is driven by
`next_cmd_ready_ = false` (the valid/ready handshake), and a retry cycle is observable through
`fill_conflict_retry_cycles` rather than the `CacheStallReason` trace. The port-busy retry in
`registered-mshr-write-buffer.md` follows the same rule — decide it once, consistently.

### Step 6 — documentation (ships with the code)
- `resources/gpu_architectural_spec.md` §5.3: the tag array is a registered structure; declare
  the same-cycle fill/lookup arbitration — **fill priority; any conflicting command (load or
  store) to a set receiving a fill this cycle is retried one cycle** — as the architectural
  contract (this also retires the implicit write-first tag-array assumption).
- `resources/perf_sim_arch.md`: updated `cache.*` responsibilities (registered tags, retry).
- `resources/timing_discipline.md`: reclassify the tag array from row-10 direct-mutated state to
  a REGISTERED forward `current_`/`next_` pair; classify `fill_installed_set_` as COMBINATIONAL
  same-tick scratch (produced by `complete_fill`, consumed by the command path later in the same
  tick); document the fill/lookup arbitration as a declared resolution.
- `resources/trace_and_perf_counters.md`: new `Stats` field `fill_conflict_retry_cycles`.
- No new doc artifact, no directory change → `AGENTS.md`/`README.md`/`onboarding.md` unchanged.

## Timing-fidelity notes

- **The tag array is now a register.** Readers see the committed start-of-cycle state; fills
  latch for the next cycle. The implicit write-first assumption is gone.
- **Fills are never blocked by a command.** A fill writes only `next_tags_`; the response/fill
  path cannot be stalled by a lookup. This directly removes the "response buffer blocked for the
  write-through duration" failure mode of the rejected hits-win design. (Fills can still be
  deferred by a *pin* — pre-existing, handled by the write-ack pinning plan.)
- **The store-hit-vs-evicting-fill hazard is closed.** The store is retried and, next cycle,
  correctly classified as a miss → write-allocate fetches the line, and `complete_fill` installs
  and pins the re-fetched line. The pin is never applied to the wrong line.
- **Retry is bounded.** A command to set S retries only on cycles a fill installs into S; fills
  into a given set are sporadic. No livelock in practice; resolution is 1–2 cycles.
- **A racing command retries once; loads and stores alike.** A load or store to a set receiving
  a fill this cycle is rejected and re-staged; on the retry it reads the committed post-fill
  tags. A command to the just-filled line then hits; a command to a conflicting line misses
  cleanly against the post-fill tag and (for a miss) against the committed `current_entries_`
  with the primary already freed. Retrying loads as well as stores is what closes the
  orphaned-secondary hang described in the resolution section — a load miss in the fill cycle
  would otherwise chain onto the line's just-freed lone primary. The retry is rare (a fresh
  command landing exactly on its line's fill cycle, not already absorbed as an MSHR secondary)
  and bounded at 1–2 cycles.

## Interaction with the other two plans

This plan touches `is_pinned`, `complete_fill`, `process_load`/`process_store`, and the cache's
`seed_next`/`commit`, all shared with `eager-wobbling-pizza.md` (write-ack pinning) and
`registered-mshr-write-buffer.md` (registered MSHR file + write buffer).

- **Write-ack pinning (`eager-wobbling-pizza.md`).** It flagged the direct-mutated tag/pin as a
  pre-existing concern; this plan resolves it. With this plan landed first, the write-ack plan
  builds `is_pinned` cleanly on `current_tags_[set].pinned` + `current_outstanding_writes_[set]`
  with no asymmetry, and its "acknowledged asymmetry" note is dropped.
- **Registered MSHR file (`registered-mshr-write-buffer.md`).** Once the MSHR file is
  registered, `find_chain_tail` scans `current_entries_`, so a primary freed by `complete_fill`
  this cycle still looks like a valid chain tail until commit. **Step 4's universal fill-conflict
  retry (loads *and* stores)** is what makes that safe: no command reaches `find_chain_tail` in a
  cycle a primary for its set is freed. This is a load-bearing cross-plan invariant — see that
  plan's §A4.

**Ordering constraint (hard).** The dangerous state is *both* the tag array and the MSHR file
registered with *no* universal fill-conflict retry — that is the orphaned-secondary hang. The
all-command retry of Step 4 must therefore be in force no later than the cycle the second of the
two registrations lands. Recommended order: **this plan first**, then `eager-wobbling-pizza.md`,
then `registered-mshr-write-buffer.md` — this plan carries the retry, so every intermediate
state is safe. Landing out of order is workable only if the second registration to land is
co-landed with (or after) Step 4's retry. Whichever plan lands second must also reconcile shared
`seed_next`/`commit` (the registered pairs `*_tags_`, `*_outstanding_writes_`, `*_entries_`) and
the shared `is_pinned` helper.

## Critical files

- `sim/src/timing/cache.cpp`, `sim/include/gpu_sim/timing/cache.h` — registered tag pair,
  reader/writer rewiring, `fill_installed_set_` scratch, retry bail.
- `sim/include/gpu_sim/stats.h` — `fill_conflict_retry_cycles`.
- (No `coalescing_unit.*` changes — the valid/ready retry already covers this.)

## Verification

Build: `cmake -B build && cmake --build build -j8`.

New Catch2 tests (`sim/tests/test_cache.cpp`, direct API with `FixedLatencyMemory`):
- A **store** to a set a fill installs into this cycle is rejected (`next_cmd_ready()` false,
  `fill_conflict_retry_cycles` bumped) and succeeds on the next attempt against the post-fill
  tags.
- A **load** racing a fill to its set is **also** retried (`next_cmd_ready()` false,
  `fill_conflict_retry_cycles` bumped); on the retry, a load to the just-filled line hits and a
  load to a conflicting line misses cleanly against the post-fill tags.
- **Orphaned-secondary regression (the core test):** a fresh load to a line whose *lone primary*
  MSHR is completed by `complete_fill` this cycle is retried — *not* allocated as a secondary
  chained to the just-freed primary. After the retry it hits; no undrained secondary remains;
  the sim reaches `is_idle()` / terminates. This test must fail if Step 4's retry is restricted
  to stores.
- **Store hit racing an evicting fill:** the store does not "hit" the doomed line — on retry it
  is a miss, write-allocates, and `complete_fill` pins the re-fetched line. The write-ack pin is
  on the set holding the store's own line.
- A fill is never blocked by a command: a deferred/normal fill completes regardless of a
  same-set command being staged.
- A command to a set with no fill this cycle is unaffected (no spurious retry).

Lint / snapshot (required — this plan adds a registered pair and a new same-tick scratch):
`ctest --test-dir build -R timing_naming_lint` and `-R signal_diagram_ast_snapshot` must both be
green; `tools/lint_timing_naming.py` must accept `current_tags_`/`next_tags_` and the
COMBINATIONAL classification of `fill_installed_set_`.

Existing tests: fill/lookup-race cases will shift by up to one cycle — re-baseline cycle counts
where tests assert exact cycles around fills. The shift is cumulative with the other two
memory-system plans; if this plan does not land first, re-baseline against the post-previous-plan
tree. Run `ctest --test-dir build`; full regression must stay green after re-baselining.

Benchmarks: `bash ./tests/run_workload_benchmarks.sh --build-dir build`, then
`python3 tools/bench_compare.py --baseline <pre-change ref>`. Small cycle-count deltas expected
at fill/lookup races; quantify retry frequency via `fill_conflict_retry_cycles`.

## Related concerns (out of scope)

- **`mshrs_` and `write_buffer_`** remain direct-mutated internal structures until
  `registered-mshr-write-buffer.md` lands. Converting them to registered form is that plan's
  scope, not "the cache array" here. Note the cross-plan invariant: once the MSHR file is
  registered, Step 4's universal (load + store) retry is what keeps `find_chain_tail` correct.
- **No data array.** The timing model carries no cache-line data, so there is no data array to
  register — only the tag array.
- **Test-direct-API note.** `fill_installed_set_` is reset at the top of `evaluate()`; tests
  driving the granular API (`handle_responses()` then `process_load()`/`process_store()`
  directly, same set) should drive full evaluate/commit cycles or expect the retry.
