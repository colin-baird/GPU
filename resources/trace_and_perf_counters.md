# Trace Generation and Performance Counters

This document is the single reference for the simulator's observability surface:
how to turn tracing on, what the traces contain, how the performance counters
are laid out, and how the two pieces fit together. The authoritative source of
truth is always the code; cross-references are given throughout. Architectural
motivation lives in [`gpu_architectural_spec.md`](./gpu_architectural_spec.md);
this document is the operator- and tooling-facing manual.

There are three distinct observability channels:

1. **Human-readable text trace** — a one-line-per-cycle summary of pipeline
   state on `stderr`. Enabled by `--trace` / `--trace-text`.
2. **Structured Chrome/Perfetto trace** — a JSON file of tracks, slices, instant
   events, and counters suitable for GUI inspection and SQL queries. Enabled by
   `--trace-file=<path>`.
3. **Performance counters** — an aggregate end-of-run summary emitted as text or
   JSON. The `Stats` struct is always populated; `--json` switches the format.

The three channels are orthogonal. Any combination can be enabled on a single
run; they do not interfere with one another or with normal `stdout` reporting.

---

## 1. Command-Line Surface

All tracing flags are parsed by the runner. Source of truth:
[`runner/src/main.cpp`](../runner/src/main.cpp) and
[`runner/src/backends/perf_sim_backend.cpp`](../runner/src/backends/perf_sim_backend.cpp).

| Flag | Channel | Effect |
|------|---------|--------|
| `--trace` | text trace | Enables the per-cycle pipeline text trace on `stderr`. Boolean alias that routes through `SimConfig::apply_cli_overrides` and sets `SimConfig::trace_enabled = true`. |
| `--trace-text` | text trace | Alias for `--trace`. |
| `--trace-file=<path>` | structured trace | Writes a Chrome-format JSON trace to `<path>`. Consumed by `PerfSimBackend::run` into `TimingTraceOptions::output_path`, which the `TimingModel` constructor uses to instantiate a `ChromeTraceWriter`. |
| `--json` | perf counters | Switches end-of-run stats from the human-readable text format to machine-parseable JSON (written to `stdout`). |
| `--max-cycles=<N>` | (lifecycle) | Bounds simulation; any partially written trace file is still finalized on shutdown. |
| `--functional-only` | (lifecycle) | Skips the timing model entirely. Disables both the text trace and structured trace (neither is produced without a `TimingModel`). |

The text and structured channels are independent: you can pass both, one, or
neither. Example invocation producing all three outputs at once:

```sh
./build/runner/gpu_sim program.elf \
    --trace \
    --trace-file=/tmp/gpu_trace.json \
    --json > stats.json 2> text_trace.log
```

---

## 2. Text Trace (`--trace` / `--trace-text`)

### 2.1 What it is

A compact one-line-per-cycle pipeline snapshot printed to `stderr`. Intended
for eyeballing short runs or grepping for specific cycles; not intended to be
programmatically parsed.

### 2.2 Wiring

- **Source:** [`TimingModel::trace_cycle()`](../sim/src/timing/timing_model.cpp) at the
  end of `tick()`. It runs only when `trace_enabled_` is true (set from
  `SimConfig::trace_enabled`, which the CLI alias `--trace` / `--trace-text`
  populates).
- **Output stream:** `std::cerr`.
- **Emission cadence:** once per non-panic `tick()`. During panic drain, the
  text trace is not emitted (the structured trace continues — see §3.8).

### 2.3 Line format

```
cycle=NNNNNN fetch=Wx issue=Wy opcoll=free|busy alu=rdy|bsy mul=rdy|bsy div=rdy|bsy tlk=rdy|bsy ldst=rdy|bsy wb=Wz:xR
```

Field semantics:

| Token | Meaning |
|-------|---------|
| `cycle=NNNNNN` | Current cycle counter (6-digit right-aligned). Monotonic from 1. |
| `fetch=Wx` / `fetch=--` | Warp id currently holding the fetch-stage output, or `--` when fetch is idle (backpressure or all warps ineligible). |
| `issue=Wy` / `issue=--` | Warp id whose instruction was issued by the scheduler this cycle, or `--` when no warp issued. |
| `opcoll=free\|busy` | Operand-collector occupancy. |
| `alu`, `mul`, `div`, `tlk`, `ldst` | Per-unit readiness: `rdy` means the unit can accept a new instruction, `bsy` means it cannot. |
| `wb=Wz:xR` / `wb=--` | If the writeback arbiter committed a result this cycle, the producing warp id and destination register. `--` otherwise. |

The text trace is deliberately minimal. Anything finer-grained (stall reasons,
individual MSHR activity, coalesced vs. serialized memory, pipeline contents
per stage) belongs in the structured trace.

---

## 3. Structured Chrome/Perfetto Trace (`--trace-file=<path>`)

### 3.1 What it is

A JSON trace in the [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU)
that opens in Perfetto and the legacy `chrome://tracing` viewer. It carries
three categories of events:

- **Complete events** (`ph:"X"`): coalesced duration slices for warp states and
  hardware-block occupancy.
- **Instant events** (`ph:"i"`): zero-duration markers for discrete pipeline
  actions (issue, writeback, cache miss allocation, etc.).
- **Counter events** (`ph:"C"`): per-cycle numeric samples for hardware
  occupancy and memory-system depth metrics.

Time is measured in **cycles**; there is no attempt to map to wall time. One
tick of the timing model produces one unit of trace time.

### 3.2 Wiring

- **Options struct:** `TimingTraceOptions { std::string output_path; }` in
  [`timing_trace.h`](../sim/include/gpu_sim/timing/timing_trace.h). Deliberately
  separate from `SimConfig` so trace-file emission remains a backend/CLI concern
  rather than a simulator-library concern.
- **Writer:** `ChromeTraceWriter` in the same header; a dependency-free JSON
  emitter supporting process/thread metadata, complete events, instant events,
  and counters.
- **Lifetime:** The `TimingModel` constructor instantiates the writer when
  `trace_options.output_path` is non-empty. `TimingModel::~TimingModel` calls
  `finalize_trace()`, which flushes every still-open slice at `cycle + 1` and
  closes the JSON array. The writer's destructor also finalizes defensively in
  case of exceptions.
- **Per-cycle path:** `TimingModel::tick()` calls `record_cycle_trace(...)`,
  which builds a `CycleTraceSnapshot` via `build_cycle_snapshot()` and then
  calls `emit_cycle_events()` to update slices, emit instant events, and write
  counters.

### 3.3 Track layout (processes, threads, and PIDs/TIDs)

The writer registers three synthetic processes. The numeric PIDs and TIDs are
internal identifiers; the names are what Perfetto displays.

| Process (PID) | Thread (TID) | Display name | Purpose |
|---------------|--------------|--------------|---------|
| Warp States (100) | `w + 1` for each resident warp | `Warp <n>` | One track per resident warp showing coalesced state slices. Sort index equals the warp id. |
| Hardware Blocks (200) | 1 | Operand Collector | Busy/idle slices of the operand collector. |
| Hardware Blocks (200) | 2 | ALU | ALU activity slices. |
| Hardware Blocks (200) | 3 | Multiply | Pipelined multiplier occupancy slices. |
| Hardware Blocks (200) | 4 | Divide | Iterative divider activity slices. |
| Hardware Blocks (200) | 5 | TLookup | TLOOKUP activity slices. |
| Hardware Blocks (200) | 6 | LD/ST | Address-generation activity slices. |
| Hardware Blocks (200) | 7 | Coalescer | Coalesced vs. serialized memory work. |
| Hardware Blocks (200) | 8 | Cache | MSHR / write-buffer / pending-fill aggregate slice. |
| Hardware Blocks (200) | 9 | Writeback | Writeback arbiter activity. |
| Hardware Blocks (200) | 10 | Panic | Panic drain state. |
| Counters (300) | 1 | Timing Counters | Host track for all per-cycle counter samples (§3.6). |

### 3.4 Warp state track

Each warp track shows coalesced slices labeled with a `WarpTraceState`. States
are exhaustively enumerated in
[`timing_trace.h`](../sim/include/gpu_sim/timing/timing_trace.h):

| `WarpTraceState` | Meaning |
|------------------|---------|
| `FETCH` | Warp owns the committed fetch output this cycle. |
| `DECODE_PENDING` | Decode is holding a staged instruction for this warp (either because the buffer is full or the staged entry has not yet committed). |
| `OPERAND_COLLECT` | Warp is resident in the operand collector. |
| `EXECUTE_ALU` | An ALU instruction for this warp is in the ALU's pending-input slot. |
| `EXECUTE_MUL` | This warp has one or more entries in the multiplier pipeline snapshot. |
| `EXECUTE_DIV` | Warp is the resident of the iterative divide unit. |
| `EXECUTE_TLOOKUP` | Warp is the resident of the TLOOKUP unit. |
| `ADDR_GEN` | Warp is in the LD/ST address-generation stage. |
| `LDST_FIFO` | Warp has an entry in the LD/ST output FIFO. |
| `COALESCING` | Warp is in the coalescing unit, actively issuing cache requests. |
| `MEMORY_WAIT` | Warp has one or more live MSHR entries waiting on external memory. |
| `WRITEBACK_WAIT` | Warp has a result parked in an execution unit's result buffer, or a gather buffer is full and waiting on the arbiter. |
| `AT_REST` | Warp is active but not owning any pipeline resource — see `WarpRestReason`. |
| `PANIC_DRAIN` | Simulation is draining after an EBREAK. |
| `RETIRED` | Warp has deactivated (ECALL or explicit termination). |

When a warp is `AT_REST` its slice name is the `WarpRestReason` instead of the
literal string `at_rest`, so rest-cause is visible at a glance in the GUI:

| `WarpRestReason` | Set when |
|------------------|----------|
| `NONE` | Default; means "no explicit reason" (warp is resting but backend made no diagnosis). |
| `WAIT_FRONTEND` | Instruction buffer was empty, or the scheduler reported `BUFFER_EMPTY`. |
| `WAIT_BRANCH_SHADOW` | Scheduler reported `BRANCH_SHADOW` (a branch/JAL/JALR is in flight). |
| `WAIT_SCOREBOARD` | Scheduler reported `SCOREBOARD` (source register still pending writeback). |
| `WAIT_OPCOLL` | Scheduler reported `OPCOLL_BUSY`. |
| `WAIT_UNIT_ALU` / `WAIT_UNIT_MULTIPLY` / `WAIT_UNIT_DIVIDE` / `WAIT_UNIT_TLOOKUP` / `WAIT_UNIT_LDST` | Scheduler reported the corresponding `UNIT_BUSY_*`. |
| `WAIT_ROUND_ROBIN` | Scheduler reported `READY_NOT_SELECTED` — warp was eligible but lost the round-robin arbitration. |
| `WAIT_L1_MSHR` | The cache reported `CacheStallReason::MSHR_FULL` while this warp was in the coalescer. |
| `WAIT_L1_WRITE_BUFFER` | Cache reported `WRITE_BUFFER_FULL`, or a store fill is blocked by the write buffer. |
| `WAIT_MEMORY_RESPONSE` | Warp is in `MEMORY_WAIT` (see above); set automatically on MSHR-populated slices. |
| `WAIT_WRITEBACK` | Warp is in `WRITEBACK_WAIT` (result parked, arbiter has not yet consumed). |

#### Slice args (warp tracks)

Every warp slice carries a JSON `args` object with:

- `cycle` (uint) — snapshot cycle.
- `state` (string) — `to_string(WarpTraceState)`.
- `rest_reason` (string, optional) — emitted iff `rest_reason != NONE`.
- `pc`, `raw_instruction`, `dest_reg` (uint) — instruction identifying fields.
- `target_unit` (string) — one of `alu`, `multiply`, `divide`, `tlookup`,
  `ldst`, `system`, `none`.
- `branch_target` (uint, optional) — only when `branch_taken`.
- `first_memory_address`, `coalesced_memory` (optional pair) — only when the
  instruction has a memory address; `coalesced_memory` is true iff all 32 lanes
  share a cache line.

#### Slice coalescing

Adjacent cycles with the same `(state, rest_reason, pc, raw_instruction, target_unit, dest_reg, branch_taken, branch_target, has_memory_address, first_memory_address, coalesced_memory)` key collapse into one complete event. The key function is
`make_warp_key` in `timing_model.cpp`. A change in any field flushes the prior
slice and starts a new one at the current cycle. This keeps trace size
proportional to state transitions, not cycle count.

### 3.5 Hardware-block tracks

Slices on the hardware tracks describe per-block occupancy. Each has a track-specific
`key` schema (used for coalescing) and a small args payload:

| Track | Slice name | Args |
|-------|-----------|------|
| Operand Collector | `busy` while occupied, empty otherwise | `cycle`, `warp`, `cycles_remaining` |
| ALU | `busy` when an entry is pending or a result is parked | `cycle`, `warp`, `cycles_remaining`, `writeback_warp` (if result parked) |
| Multiply | `busy` when any pipeline slot is occupied | `cycle`, `occupancy`, `writeback_warp` (if result parked) |
| Divide | `busy` | `cycle`, `warp`, `cycles_remaining`, `writeback_warp` |
| TLookup | `busy` | `cycle`, `warp`, `cycles_remaining`, `writeback_warp` |
| LD/ST | `busy` when address-gen has a pending entry | `cycle`, `warp`, `cycles_remaining` |
| Coalescer | `coalesced` or `serialized` based on mode | `cycle`, `warp`, `mode`, `serial_index`, `stall_reason` (optional: `mshr_full` or `write_buffer_full`) |
| Cache | `active` whenever any MSHR or write-buffer entry exists or a fill is pending | `cycle`, `active_mshrs`, `secondary_mshrs`, `pinned_lines`, `write_buffer_depth`, `pending_fill_mshr` (optional) |
| Writeback | `commit` if arbiter committed, `pending` if ≥1 source ready but not committed, empty otherwise | `cycle`, `warp`, `dest_reg` (on commit) or `cycle`, `ready_sources` (on pending) |
| Panic | `panic_drain` while active | `cycle`, `panic_warp`, `step` |

Empty hardware slices (key = `""`) flush the currently open slice and produce
no new one — this is how the tracks become dashed/empty in Perfetto.

### 3.6 Counter tracks

Every counter below is emitted **once per cycle** on the `Timing Counters`
thread of the `Counters` process. Values come directly from the
`CycleTraceSnapshot` built after commit. The counter keys map one-to-one with
Perfetto counter tracks named after the keys.

| Counter | Value | Source field |
|---------|-------|--------------|
| `active_warps` | count of warps with `active == true` | `CycleTraceSnapshot::active_warps` |
| `opcoll_busy` | 1 if operand collector busy, else 0 | `opcoll_busy` |
| `alu_busy` | 1/0 | `alu_busy` |
| `mul_occupancy` | 0..`multiply_pipeline_stages` | `mul_pipeline_occupancy` |
| `div_busy` | 1/0 | `div_busy` |
| `tlookup_busy` | 1/0 | `tlookup_busy` |
| `ldst_busy` | 1/0 | `ldst_busy` |
| `ldst_fifo_depth` | current FIFO size | `ldst_fifo_depth` |
| `active_mshrs` | live MSHR entries | `active_mshrs` |
| `secondary_mshrs` | secondary (merged) entries | `secondary_mshrs` |
| `pinned_lines` | number of cache lines pinned by dependent chains | `pinned_lines` |
| `write_buffer_depth` | write-buffer occupancy | `write_buffer_depth` |

### 3.7 Instant events

Instant events are `ph:"i"` markers with `s:"t"` (thread scope). They mark
discrete pipeline actions. All carry a base args payload of
`(cycle, warp, pc, raw_instruction, target_unit, dest_reg)` except where noted.

| Event | Track | When emitted | Extra args |
|-------|-------|--------------|------------|
| `issue` | Warp track of issuing warp | Every cycle the scheduler committed an issue output | — |
| `branch_redirect` | Warp track of branching warp | When the operand-collector output corresponds to a branch that mispredicted | `branch_target` |
| `writeback` | Warp track of writing warp | Every cycle the writeback arbiter committed a result | — |
| `panic_trigger` | Panic hardware track | Once, the cycle decode detected EBREAK and triggered panic | `cycle`, `panic_warp`, `panic_pc` |
| `cache_miss_alloc` | Cache hardware track | When a cache miss allocated an MSHR | `line_addr`, `is_store`, `merged_secondary` |
| `memory_response_complete` | Cache hardware track | When an MSHR fill completed | `line_addr`, `is_store`, `chain_length_at_fill`, `deferred` |
| `secondary_drain` | Cache hardware track | When a secondary MSHR retired (one chain step) | `line_addr`, `is_store` |
| `line_pin_stall` | Cache hardware track | When a miss is deferred because the target set is pinned by a different line | `warp_id`, `requested_line_addr`, `pinned_line_addr`, `is_store` |

### 3.8 Lifecycle and panic behavior

- `record_cycle_trace()` runs on every `tick()` path, including the panic-drain
  path. The panic trigger cycle fires once; subsequent panic-drain cycles keep
  emitting warp and hardware slices (warps show `PANIC_DRAIN`).
- `finalize_trace()` runs from `run()` and from the destructor. It flushes each
  still-open warp and hardware slice at `cycle + 1` (so the final slice has
  non-zero duration) and then closes the JSON trace array.
- A run that aborts via `--max-cycles` still produces a valid, consumable
  trace file because `finalize_trace()` always runs.
- `--functional-only` bypasses the `TimingModel` entirely, so no trace file is
  produced regardless of `--trace-file`.

### 3.9 File format details

- Root JSON object: `{"traceEvents":[ ... ]}` with one event per element.
- Events are written in emission order; no sort step.
- Strings are JSON-escaped (`\\`, `\"`, `\n`, `\r`, `\t`).
- Booleans render as `true`/`false`; all other numerics render unquoted via
  `operator<<`.
- Trace emission is self-contained — no external JSON library, no schema file.

---

## 4. Viewing and Querying Structured Traces

### 4.1 Perfetto

1. Run the simulator with `--trace-file=/tmp/gpu_trace.json`.
2. Open [ui.perfetto.dev](https://ui.perfetto.dev) in a browser.
3. Click **Open trace file** and select the JSON file. Perfetto auto-detects
   the legacy Chrome-trace format.
4. The left panel will show three process groups: **Warp States** (one track
   per warp), **Hardware Blocks** (one track per unit/cache/writeback), and
   **Counters** (numeric tracks).
5. Click any slice to see its args payload.

For long runs, Perfetto's SQL pane is more productive than scrolling. Example
queries live in
[`perfetto_trace_queries.sql`](./perfetto_trace_queries.sql). The current set
covers:

- **Stall breakdown by warp** — sums slice durations per warp and state.
- **Blocked vs. in-flight time** — aggregates `wait_*` slices against active
  slices.
- **Counter averages and peaks** — summarizes each counter track.

Write new queries against the tables Perfetto populates from Chrome traces:
`slice`, `thread_track`, `thread`, `counter`, and `counter_track`. Warp threads
are named `Warp N`; counter tracks are named after the counter keys listed in
§3.6.

### 4.2 chrome://tracing (legacy)

Chromium-based browsers still ship the legacy `chrome://tracing` viewer. Open
it, click **Load**, and pick the JSON file. The same tracks are visible; there
is no SQL pane.

---

## 5. Performance Counters (`Stats`)

### 5.1 What it is

An aggregate end-of-run summary of simulator-wide and per-warp metrics. The
`Stats` struct is always populated during a timing run (instrumentation is
mandatory, not optional). The CLI chooses only the report format.

### 5.2 Wiring

- **Definition:** `struct Stats` in
  [`include/gpu_sim/stats.h`](../sim/include/gpu_sim/stats.h).
- **Producer:** components reach `Stats&` by reference (injected via their
  constructors). Every execution unit, the cache, MSHRs, memory interface,
  fetch stage, scheduler, operand collector, and writeback arbiter update
  counters directly during their `evaluate()`/`commit()` paths.
- **Reports:** `Stats::report(ostream, num_warps)` and
  `Stats::report_json(ostream, num_warps)` in
  [`src/stats.cpp`](../sim/src/stats.cpp).
- **CLI:** the backend calls `report()` by default and `report_json()` when
  `--json` is passed. Output goes to `stdout`.

### 5.3 Counter catalog

Every field in `Stats` is listed below, grouped by section. The JSON report
uses the same field names; the text report groups them under labeled headings.

#### Global

| Field | Meaning |
|-------|---------|
| `total_cycles` | Total cycles executed by the timing model. Equal to the final value of `TimingModel::cycle_count()`. |
| `total_instructions_issued` | Number of instructions that reached a dispatch unit. |
| `ipc` (derived) | `total_instructions_issued / total_cycles`. Emitted in both reports when `total_cycles > 0`. Text uses 3 decimals; JSON uses 6. |

#### Per-warp (arrays of length `MAX_WARPS`; only the first `num_warps` are reported)

| Field | Meaning |
|-------|---------|
| `warp_instructions[w]` | Count of instructions issued by warp `w`. |
| `warp_cycles_active[w]` | Cycles during which warp `w` was active (incremented each tick while `warps_[w].active`). |
| `warp_stall_scoreboard[w]` | Scheduler scans in which warp `w` was blocked by scoreboard. |
| `warp_stall_buffer_empty[w]` | (Reserved) warp stalled because its instruction buffer was empty. |
| `warp_stall_branch_shadow[w]` | Scheduler scans in which warp `w` was blocked by an in-flight branch/JAL/JALR. |
| `warp_stall_unit_busy[w]` | Scheduler scans in which warp `w` was blocked by operand-collector-busy or target-unit-busy. |

#### Pipeline

| Field | Meaning |
|-------|---------|
| `fetch_skip_count` | Total cycles the fetch stage produced no output. |
| `fetch_skip_backpressure` | Subset: fetch stalled because decode had not consumed the previous fetch output. |
| `fetch_skip_all_full` | Subset: fetch stalled because every warp's instruction buffer was at capacity. |
| `scheduler_idle_cycles` | Cycles the scheduler issued no instruction. |
| `scheduler_frontend_stall_cycles` | Subset: at least one active warp had an empty buffer. |
| `scheduler_stall_backend_cycles` | Subset: all active warps had buffered instructions but none could issue. |
| `operand_collector_busy_cycles` | Cycles during which the operand collector was occupied. |
| `branch_predictions` | Predictions scored (equal to branch/JAL/JALR completions). |
| `branch_mispredictions` | Predictions whose direction/target disagreed with the execution result. |
| `branch_flushes` | Fetch redirects caused by mispredictions. |

#### Per execution unit (`UnitStats { busy_cycles, instructions }`)

Emitted for each of `alu_stats`, `mul_stats`, `div_stats`, `ldst_stats`,
`tlookup_stats`.

| Field | Meaning |
|-------|---------|
| `<unit>_busy_cycles` | Cycles the unit was not `is_ready()`. |
| `<unit>_instructions` | Instructions accepted by the unit. |
| Utilization (derived) | `100 × busy_cycles / total_cycles`, emitted in the text report only. |

#### Memory system

| Field | Meaning |
|-------|---------|
| `cache_hits` / `cache_misses` | Request-level totals (cache-line requests from the coalescer). Counted only on accepted requests, never on retries blocked by MSHR-full or write-buffer-full. |
| `load_hits` / `load_misses` / `store_hits` / `store_misses` | Split by operation type. Same accept-only accounting as the totals above. |
| Hit rate (derived) | `100 × cache_hits / (cache_hits + cache_misses)`, emitted in the text report only. |
| `mshr_stall_cycles` | Cycles the coalescer was unable to issue into a full MSHR file. |
| `write_buffer_stall_cycles` | Cycles a store hit or store fill could not proceed because the write buffer was full. |
| `coalesced_requests` | Memory requests where all 32 lanes fell in one cache line. |
| `serialized_requests` | Memory requests that serialized into 32 per-lane accesses (totals per lane, not per warp). |
| `external_memory_reads` / `external_memory_writes` | Requests submitted to the external memory interface. |
| `total_load_latency` / `total_loads_completed` | Used to derive average load latency (`total_load_latency / total_loads_completed`), emitted in the text report. |
| `gather_buffer_stall_cycles` | Cycles the coalescer stalled because the target warp's gather buffer was busy. |
| `gather_buffer_port_conflict_cycles` | Cycles a HIT lost the gather-buffer write port to a same-cycle FILL. |
| `mshr_merged_loads` / `mshr_merged_stores` | Secondary-MSHR allocations (same-line merges). |
| `line_pin_stall_cycles` | Cycles a miss stalled because the target set was pinned by a different line. |
| `secondary_drain_cycles` | Cycles spent draining dependent chains after a primary fill. |

#### Writeback

| Field | Meaning |
|-------|---------|
| `writeback_conflicts` | Cycles in which more than one arbiter source was ready; only the round-robin winner proceeds. |

### 5.4 Report formats

- **Text (`Stats::report`)**: `=== GPU Simulator Statistics ===` with named
  sections for Global, Per-Warp, Pipeline, Execution Units, Memory System, and
  Writeback. Percentages and derived metrics (IPC, utilization, hit rate, avg
  load latency) appear where applicable.
- **JSON (`Stats::report_json`, enabled by `--json`)**: a single flat object.
  All scalar fields appear as top-level keys; per-warp arrays are emitted as
  JSON arrays of length `num_warps`. The key set is stable and intended to be
  consumed by the benchmark and comparison tooling
  (`tools/bench_compare.py`, `tests/run_workload_benchmarks.sh`, etc.).

### 5.5 Relationship to the structured trace

Counters emitted on the Counters track (§3.6) are **instantaneous per-cycle
samples**. Stats counters (§5.3) are **aggregates over the whole run**. Where
they measure the same underlying quantity, a cycle-by-cycle sum of the trace
counter should approach the corresponding `Stats` aggregate (subject to
definition differences — e.g. `mul_occupancy` is pipeline slots, while
`mul_stats.busy_cycles` counts any busy cycle).

Use the trace counters to understand *when* a metric changed; use the stats
counters to understand *how much* it changed across the whole run.

---

## 6. Tests

Observability has dedicated coverage in
[`sim/tests/test_integration.cpp`](../sim/tests/test_integration.cpp):

- **`Trace snapshot: load miss is classified as memory_wait`** — asserts that
  a warp with an outstanding MSHR appears in `MEMORY_WAIT` in the committed
  `CycleTraceSnapshot`.
- **`Trace snapshot: MSHR pressure is classified as wait_l1_mshr`** — asserts
  that a coalescer stall on a full MSHR file produces `AT_REST` +
  `WAIT_L1_MSHR`.
- **`Trace file: emits Chrome trace JSON with warp states and counters`** —
  smoke test: runs the timing model with a `TimingTraceOptions` pointing at a
  temp file and verifies the file contains `traceEvents`, warp thread names,
  the `active_warps` counter, a `memory_wait` slice, and an `issue` instant
  event.

Unit tests for `Stats` live implicitly across the component tests, which are
the sole producers of the counters they verify. End-to-end validation of
counter values happens in `test_integration.cpp` (the "stats collection" case).

---

## 7. Pointers to Source of Truth

| Concept | File |
|---------|------|
| Structured trace types, snapshot structs, Chrome writer | [`sim/include/gpu_sim/timing/timing_trace.h`](../sim/include/gpu_sim/timing/timing_trace.h), [`sim/src/timing/timing_trace.cpp`](../sim/src/timing/timing_trace.cpp) |
| Per-cycle snapshot construction, event emission, slice coalescing | [`sim/src/timing/timing_model.cpp`](../sim/src/timing/timing_model.cpp) (`build_cycle_snapshot`, `emit_cycle_events`, `update_track_slice`, `flush_track_slice`, `finalize_trace`, `trace_cycle`) |
| Cache trace-event structs | [`sim/include/gpu_sim/timing/cache.h`](../sim/include/gpu_sim/timing/cache.h) (`CacheMissTraceEvent`, `CacheFillTraceEvent`, `CacheSecondaryDrainTraceEvent`, `CachePinStallTraceEvent`) |
| Stats struct and reports | [`sim/include/gpu_sim/stats.h`](../sim/include/gpu_sim/stats.h), [`sim/src/stats.cpp`](../sim/src/stats.cpp) |
| CLI wiring for traces and `--json` | [`runner/src/main.cpp`](../runner/src/main.cpp), [`runner/src/backends/perf_sim_backend.cpp`](../runner/src/backends/perf_sim_backend.cpp) |
| Example Perfetto SQL queries | [`resources/perfetto_trace_queries.sql`](./perfetto_trace_queries.sql) |
| Architectural context | [`resources/gpu_architectural_spec.md`](./gpu_architectural_spec.md) |
