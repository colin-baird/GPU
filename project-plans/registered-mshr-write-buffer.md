# Plan: Register the MSHR file and the write buffer

## Context

After the tag array is registered (`registered-tag-array.md`) and `outstanding_writes_` is
registered (`eager-wobbling-pizza.md`), two L1-cache structures remain **direct-mutated** —
written and read within the same `evaluate()` with no `current_`/`next_` separation:

- **The MSHR file** (`MSHRFile`, `sim/.../mshr.{h,cpp}`).
- **The write buffer** (`L1Cache::write_buffer_`, a `std::deque<uint32_t>`).

A survey of the rest of the memory path found nothing else: `LoadGatherBufferFile` is already
double-buffered (Phase 10D), the `LdStUnit` addr-gen FIFO is already commit-mutation-only
(Phase M1), and the `ExternalMemoryInterface` uses registered request slots plus
registered-by-tick-order response delivery. These two are the last structures whose same-cycle
write-then-read behavior is implicit and undeclared.

Neither has a *correctness* bug today, but both embed undeclared same-cycle semantics:

- **MSHR file:** `free()` runs early in `evaluate()` (`complete_fill`, `drain_secondary_chain_head`);
  `allocate()` / `has_free()` / `has_active()` / `find_chain_tail()` run later
  (`process_load`/`process_store`) and observe this-cycle's frees. A miss can reuse an MSHR slot
  the same cycle a fill frees it (one-cycle-optimistic vs. a registered file).
- **Write buffer:** a `push_back` during `cache_->evaluate()` is visible to same-cycle depth
  checks, and to `cache_->drain_write_buffer()` later in the tick — so a line enqueued at cycle
  T can be submitted to memory at cycle T (a fall-through FIFO). It also accepts up to ~3
  `push_back`s per `evaluate()` (store hit + store-miss fill + store-secondary drain), an
  implicit multi-write-port FIFO.

This plan converts both to discipline-compliant structures: the MSHR file becomes a registered
`current_`/`next_` pair; the write buffer becomes a registered, **single-enqueue-port** FIFO.

> Line numbers cited below are relative to the pre-change tree. These three memory-system plans
> land in sequence; locate sites by symbol, not by the cited line number.

## Part A — MSHR file

Make `MSHRFile` internally double-buffered, mirroring `LoadGatherBufferFile`.

### A1 — registered storage
`sim/include/gpu_sim/timing/mshr.h`, `sim/src/timing/mshr.cpp`
- Replace `std::vector<MSHREntry> entries_` with `current_entries_` / `next_entries_`.
- Add `void seed_next()` (copies `current_entries_ → next_entries_`) and `void commit()`
  (flips `current_ ← next_`). `reset()` resets both.

### A2 — readers read `current_`, writers write `next_`
`sim/src/timing/mshr.cpp`
- `has_free()`, `has_active()`, `find_chain_tail()` — scan `current_entries_`.
- `allocate(entry)` — scan **`current_entries_`** for the first `!valid` slot (so a slot freed
  this cycle is *not* reused this cycle — the registered semantics); write the entry into
  `next_entries_[i]`; return `i`.
- `free(index)` — clear `next_entries_[index].valid`.
- Split `at()` into `const MSHREntry& current_at(uint32_t)` (read) and
  `MSHREntry& next_at(uint32_t)` (write).

### A3 — update call sites
`sim/src/timing/cache.cpp`
- Reads → `current_at`: `complete_fill` (primary read + chain walk + chain-length loop),
  `drain_secondary_chain_head` (candidate + head-detection scans), `active_mshr_count` /
  `active_mshr_warps`.
- Writes → `next_at`: the `next_in_chain` link write in `process_load` and `process_store`.
- `L1Cache` drives the new lifecycle: call `mshrs_.seed_next()` in the cache's evaluate-top
  seeding block (next to `next_pending_fill_ = current_pending_fill_`) and `mshrs_.commit()` in
  `L1Cache::commit()`.

### A4 — invariant to confirm (no code)
For any one line, the chain *build* phase (`find_chain_tail` + secondary `allocate`, while the
primary is still in flight) and the chain *drain* phase (`drain_secondary_chain_head`, after the
primary's `complete_fill`) never overlap — a resident line's load is a hit, not a new secondary
allocation. This disjointness is what guarantees a `next_at(tail)` link write and a `free()`
never target the same entry in one cycle. It already holds; verify it survives the change.

**Cross-plan correctness invariant (load-bearing — do not weaken).** `find_chain_tail` scans
`current_entries_` (A2), so a primary freed by `complete_fill` *this cycle* still appears as a
valid chain tail until commit. On its own that is a bug source: in the exact cycle a fill
completes a line's *lone primary*, a fresh miss to that line would chain a secondary onto the
just-freed primary, and — because a lone-primary fill installs the line *unpinned* —
`drain_secondary_chain_head` never drains the orphan (MSHR leak, the load is never delivered,
the sim never reaches `is_idle()`). What makes `find_chain_tail`-on-`current_entries_` correct is
**`registered-tag-array.md` Step 4's universal fill-conflict retry**: a load *or* store to a set
receiving a fill this cycle is rejected and re-staged, so no command ever reaches
`find_chain_tail` in a cycle a primary for its set is freed. This invariant must hold whenever
both the tag array and the MSHR file are registered; if that retry is ever narrowed (e.g. back
to store-only), this plan reopens the orphan hang. Treat it as a hard dependency, not an
incidental interaction — see "Interaction with the other two plans".

## Part B — write buffer

Make `write_buffer_` a registered FIFO with a single enqueue port, mutated only at commit —
the `LdStUnit` addr-gen FIFO discipline (Phase M1), plus a port-claim flag like the gather
buffer's `next_port_claimed_`.

### B1 — single enqueue port
`sim/include/gpu_sim/timing/cache.h`, `sim/src/timing/cache.cpp`
- Add `bool next_write_buffer_port_claimed_` — a REGISTERED scratch reset at commit (modeled on
  `LoadGatherBufferFile::next_port_claimed_`).
- `queue_write_through(line_addr)` (the helper introduced by `eager-wobbling-pizza.md`) becomes
  *fallible*: if `next_write_buffer_port_claimed_` is already set, return failure without
  staging anything; otherwise claim the port, stage the enqueue (B2), and
  `++next_outstanding_writes_[set]` / `++next_outstanding_writes_total_` (the per-set pin counter
  and the global outstanding-write-cap counter, both from `eager-wobbling-pizza.md`). Return
  success/failure.
- The three callers already have a write-buffer-full backpressure path — route the port-busy
  failure into the same path: `complete_fill` defers the fill, `drain_secondary_chain_head`
  leaves the secondary for next cycle, `process_store` returns `false` (command retry). Tick
  order (`complete_fill` → `drain_secondary` → `process_store`) gives FILL > secondary > HIT
  enqueue priority for free, consistent with the existing port model.
- **Call-site ordering (must do).** Today the bare `write_buffer_.push_back` sits *after* the
  tag install and trace-event writes in `complete_fill` (cache.cpp ~236–239) and *before*
  `mshrs_.free` / the pin-clear in `drain_secondary_chain_head` (~333 vs ~342–345). A *fallible*
  `queue_write_through` must be probed before any of those side effects, so each caller commits
  to the enqueue atomically. Either call `queue_write_through` first and perform the tag install
  / `next_last_fill_event_` write / `mshrs_.free` / pin-clear only on success, or add a cheap
  non-mutating `write_buffer_can_enqueue()` pre-check and call it where the existing
  `write_buffer_.size() >= write_buffer_depth_` check sits. Do **not** install the tag (or free
  the MSHR, or clear the pin) and then discover the enqueue failed — that leaves partial state a
  deferred fill / retried command would re-apply.
- The port-busy `false` return follows the same observability rule as `registered-tag-array.md`'s
  fill-conflict retry: it does **not** set `stalled_` / `stall_reason_`; the retry is driven by
  the existing backpressure path, and the cycle is quantified by the optional
  `write_buffer_port_conflict_cycles` counter (see Verification).

### B2 — registered FIFO (commit-mutation-only)
`sim/include/gpu_sim/timing/cache.h`, `sim/src/timing/cache.cpp`
- Add staging slots: `std::optional<uint32_t> next_write_buffer_push_` (the ≤1 enqueue this
  cycle, set by the port winner) and `bool next_write_buffer_pop_` (set by `drain_write_buffer`).
- All evaluate-phase / `drain_write_buffer`-phase reads (`write_buffer_.size()` depth checks,
  `.empty()`, `.front()`) read the committed `write_buffer_` deque unchanged — it is now mutated
  only at commit.
- `L1Cache::commit()` applies the staged ops: pop front if `next_write_buffer_pop_`, then
  push back if `next_write_buffer_push_`; clear both slots and `next_write_buffer_port_claimed_`.
- `drain_write_buffer()` reads `write_buffer_.front()` (committed), submits it to `mem_if_`, and
  stages `next_write_buffer_pop_ = true` instead of popping directly.

### B3 — depth-check note
With ≤1 enqueue per cycle and the depth check reading committed size, the check `size() >=
write_buffer_depth_` is correct unchanged (at most one entry added per cycle).

## Timing-fidelity notes

- **MSHR reuse is now registered:** a slot freed by a fill in cycle T is available to a miss in
  cycle T+1, not T. A miss arriving the cycle the file becomes non-full stalls `MSHR_FULL` one
  extra cycle. `is_idle()` / `has_active()` likewise report the last MSHR freeing one cycle
  later.
- **Chain-drain start shifts by one cycle:** `drain_secondary_chain_head` sees the primary's
  `free` only after commit, so the first secondary drains one cycle later than today.
- **The write buffer is no longer fall-through:** a line enqueued at cycle T is submittable to
  memory no earlier than T+1 — one extra cycle of write-through latency when the buffer was
  empty.
- **The write buffer is single-write-port:** in a cycle where two or three of {store hit,
  store-miss fill, store-secondary drain} want to enqueue, one wins (FILL > secondary > HIT by
  tick order) and the others take their existing backpressure/retry path. This closes the
  "write-buffer multi-enqueue" item flagged in `eager-wobbling-pizza.md` Related concerns.
- All shifts are bounded ±1 cycle and are the expected, faithful consequence of registering.

## Interaction with the other two plans

This plan touches the same store paths and the cache `seed`/`commit` as the other two:

- `eager-wobbling-pizza.md` introduces `queue_write_through`, the per-set `outstanding_writes_`
  vector, and the `outstanding_writes_total_` cap counter (with its own enqueue-admission
  backpressure at the three callers). **This plan makes `queue_write_through` fallible** (B1) —
  whichever lands second must reconcile: the three callers fold a port-busy failure into the
  same backpressure path that already handles write-buffer-full and the outstanding-write cap.
- `registered-tag-array.md` registers `tags_` and adds `fill_installed_set_`. No direct
  conflict, but all three add to the cache's evaluate-top seeding and `commit()`.
- **Hard dependency — the orphan hang.** Registering the MSHR file (Part A) makes
  `find_chain_tail` scan `current_entries_`, which is correct *only* while
  `registered-tag-array.md` Step 4's universal (load + store) fill-conflict retry is in force —
  see §A4. The dangerous state is the tag array and the MSHR file both registered with no such
  retry. So Part A must not land before that retry exists: land `registered-tag-array.md` (with
  the load+store retry) first, or co-land Part A with it.
- **Recommended order:** `registered-tag-array.md` → `eager-wobbling-pizza.md` → this plan.
  Registering the tags first gives a clean `is_pinned` and puts the fill-conflict retry in place
  before this plan registers the MSHR file; the write-ack plan then adds `queue_write_through`;
  this plan finally makes it fallible and registers the FIFO/MSHR file. Landing out of order is
  workable but each later plan must re-touch the shared seed/commit block and
  `queue_write_through`, and Part A must never precede the universal retry.

## Critical files

- `sim/include/gpu_sim/timing/mshr.h`, `sim/src/timing/mshr.cpp` — registered `MSHRFile`.
- `sim/include/gpu_sim/timing/cache.h`, `sim/src/timing/cache.cpp` — `current_at`/`next_at`
  call-site rewiring, MSHR `seed_next`/`commit` wiring, write-buffer port + staged FIFO.
- `sim/include/gpu_sim/stats.h` — optional `write_buffer_port_conflict_cycles` counter (see
  Verification); reuse `write_buffer_stall_cycles` if a separate counter is not wanted.

## Documentation (ships with the code)

- `resources/gpu_architectural_spec.md` §5.3 / §5.3.2: the MSHR file is a registered structure
  (same-cycle free→reuse no longer occurs); the write buffer is a registered, single-enqueue-port
  FIFO with FILL > secondary > HIT enqueue arbitration and no fall-through.
- `resources/perf_sim_arch.md`: updated `mshr.*` and `cache.*` responsibilities.
- `resources/timing_discipline.md`: reclassify the MSHR file and write buffer from row-10
  direct-mutated state to REGISTERED structures; classify `next_write_buffer_port_claimed_` as a
  registered port-claim flag (like the gather-buffer port).
- `resources/trace_and_perf_counters.md`: any new `Stats` field added in Verification.

## Verification

Build: `cmake -B build && cmake --build build -j8`.

New Catch2 tests (`sim/tests/`):
- `test_mshr.cpp` (or the existing MSHR coverage): a slot freed this cycle is *not* reusable
  until next cycle; with the file full, a miss stalls one extra cycle and then allocates.
- `test_cache.cpp`: the write buffer does not fall through — a line enqueued at cycle T is not
  submitted to `mem_if_` before T+1; with two enqueuers in one cycle (e.g. store-miss fill +
  store hit), one is accepted and the other takes its backpressure path; chain-drain start
  shifts by one cycle after a store-miss fill.
- A port-busy `complete_fill` / `drain_secondary_chain_head` leaves no partial state: the tag is
  not installed (and the MSHR not freed / pin not cleared) on the cycle the enqueue is refused;
  the fill / secondary is retried intact next cycle.
- Confirm `is_idle()` / termination still works with the registered MSHR file.
- Orphan regression (cross-plan): with `registered-tag-array.md` landed, a fresh load to a line
  whose lone primary completes this cycle does not produce an undrained secondary — see that
  plan's Verification. If this plan lands without that retry in force, this case hangs.

Lint / snapshot (required — this plan converts two structures to registered form and adds a
port-claim flag): `ctest --test-dir build -R timing_naming_lint` and `-R
signal_diagram_ast_snapshot` must be green; `tools/lint_timing_naming.py` must accept the new
`current_`/`next_` MSHR pair, the staged write-buffer slots, and `next_write_buffer_port_claimed_`.

Existing tests: cycle counts around MSHR reuse, chain drain, and write-through drain shift by up
to one cycle — re-baseline those assertions. The shift is cumulative with the other two plans;
since this plan is recommended to land last, re-baseline against the post-`eager-wobbling-pizza`
tree. Run `ctest --test-dir build`; full regression must be green after re-baselining.

Benchmarks: `bash ./tests/run_workload_benchmarks.sh --build-dir build`, then
`python3 tools/bench_compare.py --baseline <pre-change ref>`. Expect small IPC deltas on
store-heavy / MSHR-pressured workloads from the registered reuse latency and the single enqueue
port. Optionally add a `write_buffer_port_conflict_cycles` counter to quantify enqueue-port
contention.

## Out of scope

- The `ExternalMemoryInterface` backend-internal structures (`in_flight_`, `request_fifo_`,
  DRAMSim3 assembly maps) — touched only by each backend's own `evaluate()`, no cross-consumer
  hazard; left as-is.
- A genuinely multi-write-port write buffer (instead of arbitrating to one port) — not pursued;
  single-write-port is the hardware-faithful default.
