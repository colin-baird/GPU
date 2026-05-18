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
window, the eviction that the stale read depends on cannot happen, and once the write is durable
any later miss/fetch is correct. Outcome: store→load to the same address within a warp becomes
safe; §5.4 item 2 moves from "Not guaranteed" to "Guaranteed".

The existing pin **enforcement** (reject a different-tag miss into a pinned set; defer a
different-tag fill) is reused unchanged. The existing pin **state** (`bool CacheTag::pinned`) is
*not* sufficient and is extended — see Step 1.

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
- **The `valid`/tag enforcement condition is self-enforcing and needs no change.** While
  `outstanding_writes_[set] > 0`, the only thing that could clear `tags_[set].valid` or change
  its tag is a conflicting fill — which the pin blocks. So the resident tag invariantly equals
  the in-flight write's tag; the existing `tags_[set].valid && tags_[set].tag != tag` condition
  stays, only `tags_[set].pinned` → `is_pinned(set)` changes.

## Confirmed decisions

- **Pin tracking:** per-set outstanding-write counter. `write_buffer_` keeps its current
  pop-at-submit behavior; the counter spans enqueue→ack independently.
- **Observability:** new dedicated `Stats` counter `write_ack_pin_stall_cycles`; keep the single
  `CacheStallReason::LINE_PINNED` (no new stall reason, no `WarpRestReason` change).

## Implementation

### Step 1 — `L1Cache` state: per-set counter + helpers
`sim/include/gpu_sim/timing/cache.h`, `sim/src/timing/cache.cpp`
- Add `std::vector<uint32_t> outstanding_writes_;`, sized `num_sets_` in the constructor
  initializer list (alongside `tags_`).
- Keep `CacheTag::pinned` as the **chain pin**; update its doc comment to note it is now one of
  two pin reasons.
- Add `bool is_pinned(uint32_t set) const` → `tags_[set].pinned || outstanding_writes_[set] > 0`.
- Add `void queue_write_through(uint32_t line_addr)` → `write_buffer_.push_back(line_addr);
  ++outstanding_writes_[get_set(line_addr * line_size_)];` — the single wrapper for all
  write-buffer enqueues.
- `reset()`: zero `outstanding_writes_`.

### Step 2 — Route the 3 write-through enqueue sites through the wrapper
`sim/src/timing/cache.cpp`
Replace the three bare `write_buffer_.push_back(...)` calls with `queue_write_through(...)`:
store hit (~line 132), store-miss primary fill (~line 239), store-secondary drain (~line 333).
The existing write-buffer depth checks (`write_buffer_.size() >= write_buffer_depth_`) are
unchanged.

### Step 3 — Pin enforcement uses `is_pinned`
`sim/src/timing/cache.cpp`
Swap `tags_[set].pinned` → `is_pinned(set)` at the three enforcement sites — `process_load`
(~line 57), `process_store` (~line 138), `complete_fill` deferred-fill guard (~line 208) — and
in `any_pinned_tag()` (~line 482, used by `next_cmd_stall_reason()`). Leave the surrounding
`tags_[set].valid && tags_[set].tag != tag` / `!= new_tag` conditions exactly as they are (the
invariant above makes them correct). `pinned_line_count()` stays chain-pin-only (existing trace
semantics preserved).

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
  `write_ack_queue_capacity_ = write_buffer_depth + chunks_per_line`; update the two completion
  asserts and the class-doc comment. Add `write_ack_queue_capacity()` / a
  `max_observed_write_ack_queue()` mirror for the stress test.
- Add `write_acks_.empty()` to `is_idle()`; clear in `reset()`.

`L1Cache::handle_responses` — restructure so write acks drain first, unconditionally:
1. `while (mem_if_.current_has_write_ack())`: pop, derive `set = get_set(line_addr*line_size_)`,
   `assert(tags_[set].valid)` (documents the self-enforcing invariant), `assert(outstanding_writes_[set] > 0)`,
   `--outstanding_writes_[set]`.
2. Then the existing deferred-fill retry (`if (current_pending_fill_.valid) { ...; return; }`) —
   now safe, since a blocking write-ack pin was just decremented; a fill blocked solely by a
   write-ack pin lands the same cycle.
3. Then the existing read-fill path. Delete the `if (resp.is_write) continue;` branch — write
   acks no longer arrive on `responses_`.

### Step 5 — Chain-pin → write-ack-pin handoff
No code beyond Steps 1–4. For a store-miss with a dependent chain, `complete_fill` sets both
`tags_[set].pinned` (chain) and `++outstanding_writes_[set]` (via `queue_write_through`). When
the last chain secondary drains, `drain_secondary_chain_head` clears only `tags_[set].pinned`;
`is_pinned` stays true because `outstanding_writes_[set] > 0`, and goes false only after the
last write ack. **Confirm** the two chain-clear lines (`tags_[set].pinned = false`) are left
untouched.

### Step 6 — `is_idle()` for termination
`sim/src/timing/cache.cpp`
`L1Cache::is_idle()`: add `&& std::none_of(outstanding_writes_...)` — a write that left
`write_buffer_` but is not yet acked must keep the cache non-idle. `mem_if_` idle is covered in
Step 4. `TimingModel::pipeline_drained()` needs no change (already ANDs `cache_->is_idle() &&
mem_if_->is_idle()`).

### Step 7 — Stats counter
`sim/include/gpu_sim/stats.h`, `sim/src/timing/cache.cpp`
Add `uint64_t write_ack_pin_stall_cycles = 0;` to `Stats`. At the 3 enforcement sites, attribute
the stall: bump `line_pin_stall_cycles` when the chain bit is the cause, else
`write_ack_pin_stall_cycles` (precedence: chain-pin > write-ack-pin; document it). Keep
`CacheStallReason::LINE_PINNED` as the single stall reason.

### Step 8 — Documentation (ships with the code)
- `resources/gpu_architectural_spec.md`: §5.3.1 line-pinning — describe the two pin reasons;
  §5.3.2 write buffer — a queued write-through pins its set until the ack; §5.3.3 — new
  write-ack registered forward edge; §5.4 — move "No write-buffer snoop" from "Not guaranteed"
  to "Guaranteed" and rewrite the surrounding framing (only the cross-line store→store reorder
  remains a Phase-1 limitation); §5.6 — new interface methods + split queue bounds.
- `resources/perf_sim_arch.md`: updated responsibilities for `cache.*`, `memory_interface.*`,
  `dramsim3_memory.*`.
- `resources/trace_and_perf_counters.md`: new `Stats` field `write_ack_pin_stall_cycles`.
- `resources/timing_discipline.md`: extend the cache↔mem_if boundary row with the write-ack
  queue (second REGISTERED forward response edge, drained unconditionally) and the deadlock
  rationale; note `outstanding_writes_` as direct-mutated internal state.
- No new doc artifact, no directory change → `AGENTS.md`/`README.md`/`onboarding.md` unchanged.

## Critical files

- `sim/src/timing/cache.cpp`, `sim/include/gpu_sim/timing/cache.h` — pin state, helpers,
  `handle_responses`, `is_idle`, stall attribution.
- `sim/include/gpu_sim/timing/memory_interface.h`, `sim/src/timing/memory_interface.cpp` —
  write-ack channel on the abstract interface + `FixedLatencyMemory`.
- `sim/src/timing/dramsim3_memory.cpp`, `sim/include/gpu_sim/timing/dramsim3_memory.h` —
  write-ack channel + split queue capacities.
- `sim/include/gpu_sim/stats.h` — `write_ack_pin_stall_cycles`.

## Verification

Build: `cmake -B build && cmake --build build -j8`.

New Catch2 tests (`sim/tests/`):
- `test_cache.cpp` (drive via direct API with `FixedLatencyMemory` for deterministic cycles):
  - store-hit pins its set until the write ack; a conflicting different-tag access stalls
    `LINE_PINNED` and bumps `write_ack_pin_stall_cycles`, accepted only after the ack.
  - **core test:** a conflicting fill is *deferred* and the written line is *not evicted* while
    the write is in flight (`current_last_fill_event().deferred == true`, tag still resident);
    it lands after the ack.
  - the pin releases exactly on the ack-delivery cycle, not before.
  - N outstanding writes to one set stay pinned until the Nth ack.
  - **deadlock regression:** a read fill deferred by a write-ack-pinned set makes progress —
    `handle_responses` drains the ack, the pin clears, the fill lands, the sim reaches
    `is_idle()`.
  - chain-pin→write-ack-pin handoff: set stays pinned after the chain fully drains, releases
    only after all write acks.
- `test_dramsim3_memory.cpp`: write completions arrive on the write-ack channel (not the read
  channel); extend the worst-case stress test to assert both new queue bounds; `is_idle()`
  stays false while a write ack is queued.
- An integration case: store-then-load to one address with a deliberate intervening conflicting
  access, run under both `fixed` and `dramsim3` backends — assert termination and
  `write_ack_pin_stall_cycles > 0`.

Run: `ctest --test-dir build` (full regression must stay green).

Benchmarks: `bash ./tests/run_workload_benchmarks.sh --build-dir build`, then
`python3 tools/bench_compare.py --baseline <pre-change ref>`. Expect IPC cost on
conflict-heavy, store-heavy workloads — conflicting misses now stall for write latency, and a
fill deferred by a write-ack pin holds the fill port for the pin's lifetime. Quantify via
`write_ack_pin_stall_cycles`.

## Out of scope (possible follow-up)

While a fill is deferred (`current_pending_fill_.valid`), `handle_responses` services no other
read fill — so a fill deferred by a write-ack pin serializes the fill port behind write latency.
A Phase-2 change could let a write-ack-deferred fill step aside for an independent read fill;
that is a larger restructure and is not included here.
