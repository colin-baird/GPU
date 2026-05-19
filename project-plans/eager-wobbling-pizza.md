# Plan: Close the store→load memory-ordering hazard via write-ack line pinning

## Context

The timing model has a documented Phase-1 memory-ordering hole (`resources/gpu_architectural_spec.md`
§5.4, "Not guaranteed", item 2 — "No write-buffer snoop on the load-miss path"): a load that
follows a store to the same address (same warp, program order) can return a **stale** value in
the modeled microarchitecture. The mechanism: a store completes into L1 + the write buffer; if
the line is then evicted by a direct-mapped conflict before the write-through drains to memory,
a later load misses that line and fetches the pre-store value from external memory.

This change closes that hole. A cache line is kept **pinned** (un-evictable) from the moment a
write-through is queued for it until the external-memory **write ack** for that store is
received — i.e. until the data is durable in memory. With the line un-evictable across that
window, the eviction the stale read depends on cannot happen, and once the write is durable any
later miss/fetch is correct. Outcome: store→load to the same address within a warp becomes
safe; §5.4 item 2 moves from "Not guaranteed" to "Guaranteed".

The existing pin **enforcement** (reject a different-tag miss into a pinned set; defer a
different-tag fill) is reused unchanged. The existing pin **state** (`bool CacheTag::pinned`) is
*not* sufficient and is extended with a separate registered counter — see Step 1.

## Validated design findings

- **A single `bool pinned` cannot work.** It conflates two independent pin reasons with
  different lifecycles (the existing MSHR-chain pin vs the new write-ack pin) and cannot count
  multiple outstanding writes to one set. The chain-pin clear in `drain_secondary_chain_head`
  (`tags_[set].pinned = false`) is destructive and would wrongly release a coexisting write-ack
  pin. → Add a per-set outstanding-write **counter**; effective pin = chain-bit OR counter>0.
- **Deadlock if write acks share the read-fill response queue.** `handle_responses` early-returns
  when a read fill is deferred (`current_pending_fill_.valid`), never draining `mem_if_`
  responses. A write-ack pin can *only* be cleared by consuming a write ack; if a fill is
  deferred because its target set holds a write-ack pin, that ack is never consumed → pin never
  clears → permanent deadlock. The shared FIFO cannot be skip-drained (a read response can sit
  ahead of the needed write ack). → External memory must deliver write acks on a **separate
  channel** that the cache drains unconditionally every cycle.
- **`FixedLatencyMemory` already emits write responses** (its `evaluate()` pushes a
  `MemoryResponse{is_write=true}` for every completed in-flight request); `DRAMSim3Memory`
  likewise. Both already carry `line_addr`. No backend needs to "start" producing acks — the
  work is *routing* them to a separate queue and having the cache *consume* them (today
  `handle_responses` discards them via `if (resp.is_write) continue;`).
- **A one-ack-per-cycle drain needs a structural cap on outstanding writes.** Once the cache
  consumes ≤1 write ack/cycle (see Write-ack consumption), the depth of completed-but-unconsumed
  acks is no longer "what finalized since the last drain" — a backend that completes writes in
  bursts (DRAMSim3, multiple DRAM ticks per fabric cycle) can queue acks faster than they drain.
  Nothing else bounds the queue: a write leaves `write_buffer_` at *submit*, so
  `write_buffer_depth` caps only enqueued-but-unsubmitted writes, **not** enqueued-but-unacked
  ("outstanding") writes. `DRAMSim3Memory` asserts on its response-queue capacity, so an
  un-bounded write-ack queue is a hard crash on store-heavy DDR3 runs, not a slowdown. → Add a
  parameterized global cap `max_outstanding_writes` on outstanding write-throughs, enforced as
  enqueue backpressure. The write-ack queue is then provably `≤ max_outstanding_writes` (+ a
  small cushion), so `DRAMSim3Memory`'s write-ack queue capacity is sized directly from the
  parameter.
- **The counter is state, therefore a REGISTERED next/current pair — not direct-mutated.** A
  hardware counter is a register: every reader in cycle T sees the value latched at the start of
  T; the `+inc −dec` update is latched at the next edge. Modeling it as direct-mutated and
  reading the post-decrement value the same cycle would mean reading a register's next-state in
  the cycle that computes it — a violation of the REGISTERED/COMBINATIONAL timing discipline.
  See Timing-fidelity notes.
- **The `valid`/tag enforcement condition is self-enforcing and needs no change.** While
  `current_outstanding_writes_[set] > 0`, the only thing that could clear `tags_[set].valid` or
  change its tag is a conflicting fill — which the pin blocks (`complete_fill` defers on
  `is_pinned`). So the resident tag invariantly equals the in-flight write's tag; the existing
  `tags_[set].valid && tags_[set].tag != tag` condition stays, only `tags_[set].pinned` →
  `is_pinned(set)` changes.

## Confirmed decisions

- **Pin tracking:** per-set outstanding-write counter. `write_buffer_` keeps its current
  pop-at-submit behavior; the counter spans enqueue→ack independently.
- **Counter discipline:** the counter is a **REGISTERED** `current_`/`next_` pair, modeled
  exactly like the existing `pending_fill_` carrier. Enforcement reads `current_`; increments
  and decrements write `next_`; `commit()` flips. (Consequences in Timing-fidelity notes.)
- **Write-ack consumption:** the cache consumes **at most one** write ack per cycle, but does so
  **unconditionally** (independent of the deferred-fill early return). One/cycle matches a real
  write-response channel (Avalon `writeresponse` / AXI B channel — ≤1 completion/cycle); the
  unconditional property is what fixes the deadlock.
- **Outstanding-write cap:** a parameterized global ceiling `max_outstanding_writes` on
  enqueued-but-unacked write-throughs, enforced as enqueue backpressure (a write-through that
  would exceed it is refused; the requesting command / fill / secondary retries). Tracked by a
  REGISTERED scalar `outstanding_writes_total_` — the running sum of the per-set
  `outstanding_writes_` vector. `max_outstanding_writes >= 1` is required for forward progress;
  `max_outstanding_writes >= write_buffer_depth` is the sensible configuration (every
  write-buffer entry is also outstanding).
- **Observability:** new dedicated `Stats` counter `write_ack_pin_stall_cycles` and
  `write_throttle_stall_cycles` (the outstanding-write cap); keep the single
  `CacheStallReason::LINE_PINNED` for pins and reuse `CacheStallReason::WRITE_BUFFER_FULL` for
  the cap (no new stall reason, no `WarpRestReason` change).

## Plan ordering and interaction

This plan is part of a three-plan memory-system series and assumes the recommended landing order
**`registered-tag-array.md` → this plan → `registered-mshr-write-buffer.md`**.

With `registered-tag-array.md` landed first, the L1 tag array is already a REGISTERED
`current_tags_`/`next_tags_` pair and `CacheTag::pinned` is registered along with it. Therefore,
throughout this plan: read `tags_[set]` as `current_tags_[set]` (reads) / `next_tags_[set]`
(writes), and treat the chain pin as a REGISTERED bit, not a direct-mutated one. This *removes*
the chain-pin / write-ack-counter asymmetry an earlier draft of this plan called out — `is_pinned`
then combines two REGISTERED terms (`current_tags_[set].pinned` and
`current_outstanding_writes_[set]`), both visible from the cycle after they are written. Steps 1
and 5 and the Timing-fidelity notes are written for that post-`registered-tag-array.md` world.

If this plan is landed *before* `registered-tag-array.md`, the chain pin is still direct-mutated,
`is_pinned` is a mixed REGISTERED/direct-mutated combination, and Step 5's handoff reasoning must
be read in its pre-registration form (chain pin visible the same cycle). Either way the handoff
is gap-free — see Step 5.

> Line numbers cited below are relative to the pre-change tree; these plans land in sequence, so
> locate sites by symbol, not by the cited line number.

## Implementation

### Step 1 — `L1Cache` state: write-ack counters + outstanding-write cap (REGISTERED)
`sim/include/gpu_sim/timing/cache.h`, `sim/src/timing/cache.cpp`
- Add a REGISTERED pair `std::vector<uint32_t> current_outstanding_writes_;` and
  `std::vector<uint32_t> next_outstanding_writes_;`, both sized `num_sets_` in the constructor
  initializer list. Model them on the existing `current_pending_fill_`/`next_pending_fill_`
  carrier.
- Add a REGISTERED scalar pair `uint32_t current_outstanding_writes_total_` /
  `uint32_t next_outstanding_writes_total_` — the running sum of the per-set vector, i.e. the
  count of all enqueued-but-unacked write-throughs. It is the global outstanding-write cap's
  state; the per-set vector stays the per-set pin's state.
- Add a constructor parameter `uint32_t max_outstanding_writes` and member
  `max_outstanding_writes_`. Plumb it from `SimConfig` (`sim/include/gpu_sim/config.h`, in the
  `// Cache` block next to `write_buffer_depth`) through the `L1Cache` construction path like
  `write_buffer_depth`. `assert(max_outstanding_writes_ >= 1)` in the constructor — a cap of 0
  deadlocks all stores. `SimConfig::validate()` rejects `max_outstanding_writes < 1` and should
  warn (or reject) `max_outstanding_writes < write_buffer_depth`, which leaves part of the write
  buffer structurally unreachable. A sensible default is `>= write_buffer_depth` and large
  enough not to bottleneck store throughput (sweep the benchmark suite).
- `evaluate()` seed_next: add `next_outstanding_writes_ = current_outstanding_writes_;` and
  `next_outstanding_writes_total_ = current_outstanding_writes_total_;` alongside
  `next_pending_fill_ = current_pending_fill_;`.
- `commit()`: add `current_outstanding_writes_ = next_outstanding_writes_;` and
  `current_outstanding_writes_total_ = next_outstanding_writes_total_;`.
- Add `bool is_pinned(uint32_t set) const` → `tags_[set].pinned || current_outstanding_writes_[set] > 0`.
  It reads the REGISTERED `current_` — never `next_`.
- Add `bool outstanding_writes_at_cap() const` → `current_outstanding_writes_total_ >=
  max_outstanding_writes_`. Like `is_pinned`, it reads the REGISTERED `current_` — never `next_`.
- Add `void queue_write_through(uint32_t line_addr)` → `write_buffer_.push_back(line_addr);
  ++next_outstanding_writes_[get_set(line_addr * line_size_)]; ++next_outstanding_writes_total_;`
  — the single wrapper for all write-buffer enqueues; both increments target `next_`. Callers
  must confirm admission (write-buffer depth **and** the outstanding-write cap) *before*
  invoking it — see Step 2.
- `reset()`: zero both vectors and `current_/next_outstanding_writes_total_`.
- Keep `CacheTag::pinned` as the **chain pin**; update its doc comment to note it is now one of
  two pin reasons combined by `is_pinned`. Under the recommended order (`registered-tag-array.md`
  first) `CacheTag::pinned` is a REGISTERED bit, so `is_pinned` combines two REGISTERED terms —
  see Plan ordering and interaction.

### Step 2 — Route the 3 write-through enqueue sites through the wrapper, gate on the cap
`sim/src/timing/cache.cpp`
Replace the three bare `write_buffer_.push_back(...)` calls with `queue_write_through(...)`:
store hit (~line 132), store-miss primary fill (~line 239), store-secondary drain (~line 333).
Each of the three callers already tests `write_buffer_.size() >= write_buffer_depth_` *before*
any enqueue side effect (tag install / MSHR free / pin clear) — widen that condition to
`write_buffer_.size() >= write_buffer_depth_ || outstanding_writes_at_cap()`. Keeping the cap
test at the existing depth-check site (not at the `queue_write_through` call, which in
`complete_fill` sits *after* the tag install) is what guarantees a cap-refused enqueue leaves no
partial state. Each caller routes a refusal into its **existing** write-buffer-full backpressure
path, unchanged: `process_store` returns `false` (command retry); `complete_fill` defers the
fill (it stays in `pending_fill_`); `drain_secondary_chain_head` leaves the secondary for next
cycle. When the cap — not write-buffer depth — is the cause, attribute the stall to
`write_throttle_stall_cycles` instead of `write_buffer_stall_cycles` (precedence: buffer-full >
cap; `drain_secondary_chain_head` bumps no counter, as today). The cap reuses
`CacheStallReason::WRITE_BUFFER_FULL` — no new stall reason.

### Step 3 — Pin enforcement uses `is_pinned`
`sim/src/timing/cache.cpp`
Swap `tags_[set].pinned` → `is_pinned(set)` at the three enforcement sites — `process_load`
(~line 57), `process_store` (~line 138), `complete_fill` deferred-fill guard (~line 208) — and
in `any_pinned_tag()` (~line 482, used by `next_cmd_stall_reason()`). Leave the surrounding
`tags_[set].valid && tags_[set].tag != tag` / `!= new_tag` conditions exactly as they are (the
self-enforcing invariant makes them correct). `pinned_line_count()` stays chain-pin-only
(existing trace semantics preserved).

### Step 4 — Separate write-ack channel + `handle_responses` restructure
`sim/include/gpu_sim/timing/memory_interface.h`, `sim/src/timing/memory_interface.cpp`,
`sim/src/timing/dramsim3_memory.cpp`, `sim/include/gpu_sim/timing/dramsim3_memory.h`,
`sim/src/timing/cache.cpp`

`ExternalMemoryInterface` — add a REGISTERED forward write-ack channel (mirrors the read-fill
`current_has_response`/`get_response` pair; naming per `cpp_coding_standard.md`):
- `virtual bool current_has_write_ack() const = 0;`
- `virtual MemoryResponse get_write_ack() = 0;`
- `virtual size_t write_ack_count() const = 0;` (for tests)

`FixedLatencyMemory`:
- Add `std::deque<MemoryResponse> write_acks_;`.
- In `evaluate()`, route completed `is_write` requests to `write_acks_`; reads still to
  `responses_`.
- Implement the new accessors; add `&& write_acks_.empty()` to `is_idle()`; clear in `reset()`.

`DRAMSim3Memory`:
- Add `std::deque<MemoryResponse> write_acks_;`. `on_write_complete` pushes there instead of
  `responses_`; `on_read_complete` unchanged.
- Split `response_queue_capacity_` into `read_response_queue_capacity_ = num_mshrs` and
  `write_ack_queue_capacity_ = max_outstanding_writes + chunks_per_line`; update the two
  completion asserts and the class-doc comment. The write-ack bound is now provable: the cache
  caps enqueued-but-unacked writes at `max_outstanding_writes` (Steps 1–2) and the write-ack
  queue holds a subset of those. `chunks_per_line` is the cushion for an in-evaluate completion
  plus the bounded same-cycle multi-enqueue overshoot (`outstanding_writes_total_` can exceed
  the cap by ≤ 2 until `registered-mshr-write-buffer.md`'s single-enqueue-port lands, 0 after —
  see Timing-fidelity notes; confirm `chunks_per_line >= 2`). Add `write_ack_queue_capacity()` /
  a `max_observed_write_ack_queue()` mirror for the stress test.
- Add `write_acks_.empty()` to `is_idle()`; clear in `reset()`.

`L1Cache::handle_responses` — restructure:
1. **Unconditional, at most one write ack per cycle.** Before the deferred-fill early return:
   `if (mem_if_.current_has_write_ack())` — pop one ack, derive `set = get_set(line_addr*line_size_)`,
   `assert(tags_[set].valid)` (documents the self-enforcing invariant), `assert(current_outstanding_writes_[set] > 0)`,
   then `--next_outstanding_writes_[set]; --next_outstanding_writes_total_;` — the ack returns
   one credit to the cap. (At this point `next_` has just been seeded from `current_`, so the
   asserts against `current_` and the decrements of `next_` are consistent.)
2. Then the existing deferred-fill retry (`if (current_pending_fill_.valid) { ...; return; }`).
3. Then the existing read-fill path. Delete the `if (resp.is_write) continue;` branch — write
   acks no longer arrive on `responses_`.

The write-ack `if` is placed before the early `return` so it runs every cycle — that is the
deadlock fix. Because the counter is registered, a fill deferred solely on a write-ack pin
lands the **cycle after** its blocking ack is consumed (the decrement hits `next_` at cycle T;
`is_pinned`, via `current_`, observes it at T+1). Still bounded, still deadlock-free.

### Step 5 — Chain-pin → write-ack-pin handoff
No code beyond Steps 1–4. For a store-miss with a dependent chain, `complete_fill` sets the chain
pin (`next_tags_[set].pinned = true`) and `++next_outstanding_writes_[set]` (write-ack pin).
Under the recommended order both are REGISTERED writes latched at the same commit edge, so from
the cycle after the fill both pins are visible via `is_pinned`. When the last chain secondary
drains (≥1 cycle later), `drain_secondary_chain_head` clears only the chain pin
(`next_tags_[set].pinned = false`); `is_pinned` stays true because
`current_outstanding_writes_[set] > 0`. The handoff is register-to-register with no gap. The
fill cycle itself is eviction-safe independently of pin visibility — see the gap-cycle argument
in Timing-fidelity notes (`complete_fill` is the only fill per cycle and runs before command
processing, so the just-filled line cannot be evicted in its own fill cycle). **Confirm** the
two chain-clear lines are left untouched apart from the `tags_` → `next_tags_` rename that
`registered-tag-array.md` already applied.

### Step 6 — `is_idle()` for termination
`sim/src/timing/cache.cpp`
`L1Cache::is_idle()`: add `&& current_outstanding_writes_total_ == 0` — a write that left
`write_buffer_` but is not yet acked must keep the cache non-idle. The scalar makes this an O(1)
check, no scan over the per-set vector. `mem_if_` idle is covered in Step 4. `TimingModel::pipeline_drained()` needs no change (already ANDs `cache_->is_idle() &&
mem_if_->is_idle()`).

### Step 7 — Stats counters
`sim/include/gpu_sim/stats.h`, `sim/src/timing/cache.cpp`
Add `uint64_t write_ack_pin_stall_cycles = 0;` to `Stats`. At the 3 pin-enforcement sites,
attribute the stall: bump `line_pin_stall_cycles` when the chain bit is the cause, else
`write_ack_pin_stall_cycles` (precedence: chain-pin > write-ack-pin; document it). Keep
`CacheStallReason::LINE_PINNED` as the single pin stall reason.
Add `uint64_t write_throttle_stall_cycles = 0;` for the outstanding-write cap: bump it at the
Step 2 enqueue-admission sites when `outstanding_writes_at_cap()` — not write-buffer depth —
refused the enqueue. The cap reuses `CacheStallReason::WRITE_BUFFER_FULL`; no new stall reason.

### Step 8 — Documentation (ships with the code)
- `resources/gpu_architectural_spec.md`: §5.3.1 line-pinning — describe the two pin reasons;
  §5.3.2 write buffer — a queued write-through pins its set until the ack, and the
  `max_outstanding_writes` parameter caps enqueued-but-unacked write-throughs (enqueue
  backpressure at the cap, with the `>= write_buffer_depth` recommendation); §5.3.3 — new
  write-ack registered forward edge; §5.4 — move "No write-buffer snoop" from "Not guaranteed"
  to "Guaranteed" and rewrite the surrounding framing (only the cross-line store→store reorder
  remains a Phase-1 limitation); §5.6 — new interface methods + split queue bounds (the
  write-ack queue capacity is sized directly from `max_outstanding_writes`).
- `resources/perf_sim_arch.md`: updated responsibilities for `cache.*`, `memory_interface.*`,
  `dramsim3_memory.*`.
- `resources/trace_and_perf_counters.md`: new `Stats` fields `write_ack_pin_stall_cycles` and
  `write_throttle_stall_cycles`.
- `resources/timing_discipline.md`: extend the cache↔mem_if boundary row with the write-ack
  queue (second REGISTERED forward response edge, drained unconditionally, ≤1/cycle) and the
  deadlock rationale. Classify `current_/next_outstanding_writes_` and the
  `current_/next_outstanding_writes_total_` scalar as **REGISTERED** forward state (modeled like
  `pending_fill_`) — explicitly *not* row-10 direct-mutated state. Under the recommended order
  the chain pin is likewise REGISTERED (via `registered-tag-array.md`), so `is_pinned` combines
  two REGISTERED terms.
- New `SimConfig` parameter `max_outstanding_writes`: if cache parameters (`write_buffer_depth`,
  `num_mshrs`) are documented in `README.md` / `resources/onboarding.md` or exposed as CLI
  flags, add `max_outstanding_writes` there too; otherwise no new doc artifact and no directory
  change → `AGENTS.md` unchanged.

## Timing-fidelity notes

- **The counter is a register.** `is_pinned` (every enforcement site) reads
  `current_outstanding_writes_`; increments/decrements write `next_outstanding_writes_`;
  `commit()` flips. Within an `evaluate()` the order of an increment and a decrement to the same
  set is immaterial — both modify `next_`, nothing reads the partial `next_` — which faithfully
  models a synchronous up/down counter `next = current + inc − dec`.
- **Pin releases one cycle after the write ack** (decrement latched at T, `current_`-visible at
  T+1). Conservative and safe: correctness requires the pin be held *at least* until the write
  is durable, and the ack means it already is.
- **Pin engages one cycle after the write-through is queued** (increment `current_`-visible at
  T+1). The gap cycle T is provably eviction-free: for a store *hit* the cycle-T command slot is
  the store itself (no conflicting command runs); for a store-miss *fill* `complete_fill` runs
  before command processing in tick order. The real eviction gate — `complete_fill`'s defer on
  a conflicting line's fill — always observes the registered pin with ≥ memory-latency margin.
  Worst case is a conflicting miss slipping past the pin check in the single gap cycle and
  allocating an MSHR that is then deferred; the line is not evicted; correctness holds.
- **No underflow.** A decrement is always the ack of a write queued in a strictly earlier cycle,
  so `current_outstanding_writes_[set] ≥ 1` at the start of any cycle that decrements it. The
  same holds for `current_outstanding_writes_total_ ≥ 1`.
- **The outstanding-write cap is registered, like the pin counter.** `outstanding_writes_at_cap()`
  reads `current_outstanding_writes_total_`; `queue_write_through` increments and the write-ack
  consumer decrements `next_outstanding_writes_total_`. Because the check reads `current_`, two
  or three `queue_write_through` calls in one `evaluate()` (the multi-enqueue case in Related
  concerns) all observe the same pre-increment total, so `outstanding_writes_total_` can
  transiently overshoot the cap by up to (same-cycle enqueues − 1) ≤ 2.
  `registered-mshr-write-buffer.md`'s single-enqueue-port reduces this to 0. The
  `write_ack_queue_capacity_` cushion (Step 4) absorbs the overshoot, so the write-ack queue
  stays bounded in every landing order.
- **The cap cannot deadlock.** Backpressure gates only *new* write-through enqueues; draining
  `write_buffer_`, submitting to memory, and consuming write acks for already-enqueued writes
  are all independent of the cap. Each ack returns a credit (`--outstanding_writes_total_`), so
  for any `max_outstanding_writes ≥ 1` the counter always drains and a refused enqueue
  eventually succeeds. The write-ack drain that returns the credit is itself unconditional (the
  deadlock fix above), so the credit path is never blocked behind a deferred fill.
- **Both pin terms are REGISTERED.** Under the recommended order (`registered-tag-array.md`
  first), `is_pinned` combines `current_outstanding_writes_` and `current_tags_[set].pinned`,
  both REGISTERED and both visible from the cycle after they are written. The chain→write-ack
  handoff is register-to-register with no gap (Step 5). An earlier draft of this plan noted an
  asymmetry here because it assumed a direct-mutated chain pin; `registered-tag-array.md` retires
  that — see Plan ordering and interaction.

## Critical files

- `sim/src/timing/cache.cpp`, `sim/include/gpu_sim/timing/cache.h` — registered counters
  (per-set vector + `outstanding_writes_total_` scalar), helpers (`is_pinned`,
  `outstanding_writes_at_cap`), `max_outstanding_writes_` member + constructor parameter,
  `handle_responses`, `is_idle`, enqueue admission, stall attribution.
- `sim/include/gpu_sim/config.h` — new `SimConfig::max_outstanding_writes` field +
  `validate()` constraint; the `L1Cache` construction path (`TimingModel`) threads it through.
- `sim/include/gpu_sim/timing/memory_interface.h`, `sim/src/timing/memory_interface.cpp` —
  write-ack channel on the abstract interface + `FixedLatencyMemory`.
- `sim/src/timing/dramsim3_memory.cpp`, `sim/include/gpu_sim/timing/dramsim3_memory.h` —
  write-ack channel + split queue capacities (write-ack queue sized from `max_outstanding_writes`).
- `sim/include/gpu_sim/stats.h` — `write_ack_pin_stall_cycles`, `write_throttle_stall_cycles`.

## Verification

Build: `cmake -B build && cmake --build build -j8`.

New Catch2 tests (`sim/tests/`):
- `test_cache.cpp` (drive via direct API with `FixedLatencyMemory` for deterministic cycles):
  - store-hit pins its set; a conflicting different-tag access stalls `LINE_PINNED` and bumps
    `write_ack_pin_stall_cycles`, accepted only after the pin releases.
  - **core test:** a conflicting fill is *deferred* and the written line is *not evicted* while
    the write is in flight (`current_last_fill_event().deferred == true`, tag still resident);
    it lands after the pin clears.
  - the pin releases the **cycle after** the write ack is consumed (registered counter) — not
    before, and not the same cycle.
  - N outstanding writes to one set stay pinned until the Nth ack (consumed one/cycle).
  - **deadlock regression:** a read fill deferred by a write-ack-pinned set makes progress —
    `handle_responses` consumes the ack unconditionally, the pin clears, the fill lands the next
    cycle, the sim reaches `is_idle()`.
  - chain-pin→write-ack-pin handoff: the set stays pinned after the chain fully drains, releases
    only after the write acks.
  - **outstanding-write cap:** construct the cache with a small `max_outstanding_writes`; drive
    `max_outstanding_writes` distinct un-acked write-throughs, then confirm the next enqueue is
    refused — the requesting store retries `WRITE_BUFFER_FULL`, `write_throttle_stall_cycles` is
    bumped, and `write_buffer_stall_cycles` is *not* — and is accepted the cycle after an ack
    returns a credit. The sim still reaches `is_idle()`.
- `test_dramsim3_memory.cpp`: write completions arrive on the write-ack channel (not the read
  channel); extend the worst-case stress test to assert both new queue bounds — in particular
  that the write-ack queue never exceeds `write_ack_queue_capacity_` under a store-heavy burst
  with the cap in force; `is_idle()` stays false while a write ack is queued.
- An integration case: store-then-load to one address with a deliberate intervening conflicting
  access, run under both `fixed` and `dramsim3` backends — assert termination and
  `write_ack_pin_stall_cycles > 0`.

Run: `ctest --test-dir build` (full regression must stay green; re-baseline any test asserting
exact cycles around a pinned conflict — the pin engages one cycle after the write-through is
queued — and note the shift is cumulative with the other two memory-system plans).

Lint / snapshot (required — this plan adds a registered counter pair and three new
`ExternalMemoryInterface` cross-stage accessors): `ctest --test-dir build -R timing_naming_lint`
and `-R signal_diagram_ast_snapshot` must be green; `tools/lint_timing_naming.py` must accept
`current_/next_outstanding_writes_` and the `current_has_write_ack` / `get_write_ack` /
`write_ack_count` accessors.

Benchmarks: `bash ./tests/run_workload_benchmarks.sh --build-dir build`, then
`python3 tools/bench_compare.py --baseline <pre-change ref>`. Expect IPC cost on
conflict-heavy, store-heavy workloads — conflicting misses now stall for write latency, and a
fill deferred by a write-ack pin holds the fill port for the pin's lifetime. Quantify via
`write_ack_pin_stall_cycles`.

## Related pre-existing concerns (out of scope, flagged for a separate decision)

These were surfaced while validating this plan. They are not introduced by this change and are
not fixed here, but the write-ack pin is built on top of them:

- **Read + write issued to the backend in the same cycle.** `ExternalMemoryInterface` has
  independent `next_read_request_` / `next_write_request_` slots, modeling a dual-channel
  (AXI-like) bus. The Phase-1 target is single-ported Avalon-MM, which carries one command per
  cycle — issuing a read and a write together is not realistic there and would need
  command-channel arbitration. Pre-existing; `drain_write_buffer` and the command path are
  untouched by this plan.
- **Same-cycle fill-vs-lookup to one set.** Tick order runs `complete_fill` before
  `process_load`/`process_store`, so with a direct-mutated tag array a lookup racing a fill that
  evicts its line sees the post-eviction tag — an implicit *write-first tag array*. This concern
  is **resolved by `registered-tag-array.md`**, which registers the tag array and replaces the
  implicit forwarding with an explicit fill-priority + one-cycle command retry. Under the
  recommended landing order that plan is already in before this one; no action here.
- **Write-buffer multi-enqueue.** Up to ~2 write-throughs can be queued to one set in a single
  `evaluate()` (`complete_fill` + a store-hit command; or a secondary drain + a store-hit
  command), so `write_buffer_` implicitly has multiple write ports and the counter increment is
  `+0..2`. Across all sets up to three enqueues can occur in one `evaluate()` (`complete_fill` +
  `drain_secondary_chain_head` + a store-hit command) — this is the source of the bounded
  outstanding-write-cap overshoot (`outstanding_writes_total_` exceeds `max_outstanding_writes`
  by ≤ 2; see Timing-fidelity notes). If a single-write-port write buffer is intended, the
  enqueue sites should be arbitrated to ≤1/cycle, which makes the counter a clean ±1 up/down
  counter and the cap overshoot zero — `registered-mshr-write-buffer.md` does exactly this.
- **Deferred-fill serialization.** While a fill is deferred (`current_pending_fill_.valid`),
  `handle_responses` services no other read fill — a fill deferred by a write-ack pin holds the
  fill port for the pin's lifetime. A Phase-2 change could let a write-ack-deferred fill step
  aside for an independent read fill; larger restructure, not included here.
