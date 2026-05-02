# Autonomous Timing-Bug Hunt — Plan

## Context

The functional model is trusted; the timing model is not. Two recent timing bugs were caught only after they shipped:

- **0383f04** "Fix frontend throughput: evaluate decode before fetch in tick()" — `FetchStage::output_consumed_` was mutated by `DecodeStage::evaluate()` but read by `FetchStage::evaluate()` earlier in the same tick. Symptom: IPC capped at ~0.5 across all benchmarks. Pattern: **intra-tick read-before-write hazard**.
- **7b5f713** "Fetch: count current_output toward will_be_full in-flight to fix HOL stall" — fetch eligibility gate counted `decode.pending` toward inflight reservations for a target warp but not `fetch.current_output_`. Symptom: 71339 backpressure cycles on matmul; misclassified backend stalls as frontend stalls. Pattern: **capacity gate that omits a held-state register**.

Both bugs lived in the cross-stage signaling layer formalized in `/workspace/resources/timing_discipline.md`. They were caught reactively. This plan executes a proactive structural + dynamic + spec audit to surface anything else of the same shape, **without fixing**. Spec ambiguities count.

User has selected the **aggressive + read-only instrumentation** scope tier. All phases below are in-scope. Read-only counter additions are permitted when they close an observability gap that would otherwise require speculative findings.

## Outputs

- **Findings ledger:** `/workspace/project-plans/timing-bug-hunt-findings.md` (appendable, single file, schema below).
- **Raw data dump:** `/workspace/traces/bughunt/{stats,trace}/<config>/<bench>.json`.
- **Read-only instrumentation diffs:** committed only after user review; staged in a working branch `bughunt/observability`.
- **Synthetic kernels (Phase 6):** added under `/workspace/tests/synthetic/<kernel>_bench/` mirroring existing benchmark layout.

## Finding schema

```
## Finding F-NN: <one-line title>
- Severity: Definite | Probable | Spec-Ambiguity | Observability-Gap
- Category: Cross-stage hazard | Held-state leak | Counter inconsistency
            | Trace anomaly | Spec ambiguity | Resource arbitration
- Evidence: <data point, equation that fails, trace excerpt, or code path>
- Code citation: file:line (writer); file:line (consumer / gate)
- Repro hint: <benchmark + config flag combo, or "static; no repro">
- Spec section: §X.Y (or N/A)
- Discovered: phase N, YYYY-MM-DD
```

Verification bar:
- **Definite** = code citation pinpoints a discipline violation OR a spec contract mathematically violated by counters.
- **Probable** = anomaly in counters/trace + plausible mechanism in code, but spec is silent on the exact behavior.
- **Spec-Ambiguity** = spec wording cited verbatim + implementation pinned to one interpretation.
- **Observability-Gap** = current counters/trace cannot disambiguate; recommended new counter named.

## Critical files

- `/workspace/resources/gpu_architectural_spec.md` — golden reference; Phase 7 walks it end-to-end.
- `/workspace/resources/timing_discipline.md` — 15-row inventory; Phase 4 walks it row-by-row.
- `/workspace/resources/trace_and_perf_counters.md` — counter and trace-event catalog.
- `/workspace/sim/src/timing/timing_model.cpp` — tick() ordering at lines 426–525; ground truth for evaluate/commit sequence.
- `/workspace/sim/include/gpu_sim/timing/{fetch_stage,decode_stage,operand_collector,warp_scheduler,*_unit,coalescing_unit,cache,load_gather_buffer,branch_shadow_tracker,writeback_arbiter}.h` — held-state registers live here.
- `/workspace/sim/include/gpu_sim/stats.h` + `/workspace/sim/src/stats.cpp` — counter catalog; Phase 8 instrumentation lands here.
- `/workspace/tests/run_workload_benchmarks.sh` — canonical benchmark entry point.
- `/workspace/sim/configs/dram/DDR3_4Gb_x16_800.ini` — default DRAMSim3 config.

## Session budget & resume protocol

The 5h rolling usage window cannot be queried mid-session, so the hunt is structured for atomic resumability and bounded session footprint. Every sub-task is an "atom" tracked in a single durable progress file; sessions read it, execute as much as fits the budget, persist progress, and exit cleanly.

### Progress tracker

`/workspace/project-plans/timing-bug-hunt-progress.md` is the single source of resume state. Schema:

```
# Timing Bug Hunt — Progress

## Calibration
- mean atom wall: <updated after each session>
- p95 atom wall: <updated>
- last session budget actual: <wall, atoms>

## Completed
- [phase 0.bootstrap] @ 2026-04-27T14:32 — sha=<git>, branch bughunt/observability
- [phase 1.matmul.default] @ 2026-04-27T14:51 — stats/default/matmul.json, trace/default/matmul.json
- ...

## Next
- [phase 1.matmul.fixed-memory] cmd: build/tests/matmul/matmul_bench --memory-backend=fixed --json --trace-file=traces/bughunt/trace/fixed-memory/matmul.json > traces/bughunt/stats/fixed-memory/matmul.json

## Deferred / blocked
- (none)

## Pending findings (rolled up at end of session)
- F-NN: <title>
```

The agent reads `## Next`, executes that atom, moves it to `## Completed` with wall time, pushes the next atom into `## Next`, continues or exits per budget.

### Atom decomposition

- **Phase 0**: 1 atom (bootstrap).
- **Phase 1**: 48 atoms (8 configs × 6 benches). Each writes one stats JSON + one trace JSON.
- **Phase 2**: 6 atoms (one per bench), executable once Phase 1 default-config atoms exist.
- **Phase 3**: 6 atoms (per bench), depends on Phase 1.
- **Phase 4**: 15 atoms (per inventory row). Each may spawn one Explore sub-agent.
- **Phase 5**: 8 atoms (per held register). Each may spawn one Explore sub-agent.
- **Phase 6**: 5 atoms (per synthetic kernel: author + run + analyze).
- **Phase 7**: ~10 atoms (per spec section walked).
- **Phase 8**: ad-hoc; 1 atom per added counter.
- **Phase 9**: 1 atom (consolidation).

Total ≈ 100 atoms.

### Per-session budget

Each session:

1. Read progress.md; record session start time.
2. Execute next atom listed under `## Next`.
3. After each atom: append `[phase X.atom-id] @ ts — wall=Nm — outputs` to `## Completed`; pop next atom into `## Next`.
4. Check exit conditions, in priority order:
   - elapsed wall > **90 min** → exit
   - atoms completed ≥ **5** → exit (raised after calibration if mean atom wall is small)
   - any single atom exceeded **30 min** → exit (anomaly; flag for investigation)
5. On exit: ensure all data files flushed, progress.md saved, and append a `## Session log` entry to findings ledger noting wall, atoms, model effort, anomalies.

The 90 min wall + 5 atom cap leaves ~3.5h headroom in the rolling window, so:
- Two back-to-back sessions still don't saturate.
- A hard limit-hit mid-atom loses ≤ 30 min of work; the in-flight atom is re-executed on next session (atoms are idempotent — Phase 1 re-runs overwrite JSON; Phase 4/5/7 dedup at Phase 9 consolidation).

**Calibration:** the first session updates the `## Calibration` block in progress.md with observed mean/p95 atom wall. If mean < 8 min, raise the 5-atom cap to 8-10 for subsequent sessions. If mean > 20 min, lower to 3.

### Model-effort routing

Per the user's mix preference:

| Phase | Recommended effort | Reason |
|-------|-------------------|--------|
| 0 (bootstrap) | high | Mechanical (cmake, mkdir, git branch). |
| 1 (data collection) | high | Launching binaries, parsing summary lines; minimal reasoning while binaries run. |
| 2 (counter audit) | xhigh | Cross-references spec contracts to numerical anomalies. Catches mathematical violations of fairness/closure invariants. |
| 3 (trace audit) | xhigh | Pattern recognition over per-cycle event streams; failure modes are subtle. |
| 4 (cross-stage signal audit) | **xhigh** | Heaviest reasoning; each row requires understanding REGISTERED vs COMBINATIONAL discipline + tick order. |
| 5 (held-state register audit) | **xhigh** | Each register requires walking N consumers and verifying capacity-gate accounting (the 7b5f713 template). |
| 6 (synthetic kernels) | xhigh | Author RV32IM kernels that exercise specific spec ambiguities — needs ISA precision. |
| 7 (spec ambiguity catalog) | xhigh | Reading + interpretation; flagging subtle wording issues. |
| 8 (instrumentation) | high | Counter additions are mechanical once specified. |
| 9 (consolidation) | high | Renumbering, deduplication, summary table. |

Sub-agents inherit parent model; for fan-out in Phase 4/5, use Explore agents (cheaper) for the read-only walk and report findings back.

### Scheduled resume

User option: `/schedule` a daily routine to fire unattended.

Recommended routine:
- **Cron**: daily at 04:00 user-local time (least likely to overlap with interactive use; ensures the 5h window has fully decayed).
- **Effort**: high by default; bump to xhigh manually before sessions that will run Phase 4/5/7.
- **Prompt**:
  > Resume the timing bug hunt at `/workspace/project-plans/lively-conjuring-neumann.md`. Read `timing-bug-hunt-progress.md`, execute the next atom under `## Next`, update progress, and stop after 90 min wall OR 5 atoms (or whatever calibration sets). Do not re-do completed atoms. Do not create new atoms or change the plan. End with a session-log entry in the findings ledger.

User can also launch interactive sessions any time; the same protocol applies. If weekly cap is depleted partway, finished atoms persist and Phase 9 can run cheaply at the start of the next week.

### Recovery on hard limit-hit

If the rolling-window limit is hit mid-atom, the in-flight atom is incomplete. On next session: re-read progress.md, observe the in-flight atom is not in `## Completed`, and re-execute from scratch. Atoms are idempotent by design. Phase 9 deduplication handles any duplicate findings introduced by re-runs.

## Phase 0 — Bootstrap (~5 min)

```sh
cmake -S /workspace -B /workspace/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build /workspace/build -j8
mkdir -p /workspace/traces/bughunt/{stats,trace}
git checkout -b bughunt/observability   # for staged Phase 8 instrumentation
```

Initialize the findings ledger with header + finding schema. Record git SHA.

## Phase 1 — Baseline data collection (~30 min wall)

Goal: produce a dataset rich enough to surface counter and trace anomalies in Phases 2–3.

**Configurations to sweep** (3 axes × default benchmarks):

| Config tag | Memory backend | Warps | MSHRs | Buffer depth |
|-----------|---------------|-------|-------|--------------|
| `default` | dramsim3 | 4 | 4 | 3 |
| `fixed-memory` | fixed (17 cyc) | 4 | 4 | 3 |
| `warps-2` | dramsim3 | 2 | 4 | 3 |
| `warps-8` | dramsim3 | 8 | 4 | 3 |
| `mshr-2` | dramsim3 | 4 | 2 | 3 |
| `mshr-8` | dramsim3 | 4 | 8 | 3 |
| `buffer-2` | dramsim3 | 4 | 4 | 2 |
| `buffer-6` | dramsim3 | 4 | 4 | 6 |

Per (config, bench) tuple: emit `--json` to `stats/<config>/<bench>.json`; emit `--trace-file=trace/<config>/<bench>.json`. Some flags may not exist; verify CLI parsing in `runner/src/main.cpp` first; if a knob isn't exposed, drop the config and log it as Observability-Gap F-NN.

Save the runner's `RESULT name=…` summary to `stats/<config>/SUMMARY.txt` per config.

**Five highest-leverage signals to retain:** per-bench (cycles, IPC, frontend_stall%, backend_stall%); per-warp instruction balance (max/min ratio); load_latency p50/p99; coalesced/serialized ratio; mul_busy/mul_instructions ratio.

## Phase 2 — Counter-anomaly pass (~30 min)

Run a Python analysis script over each `stats/<config>/<bench>.json`. Each invariant violation is a finding.

**Top-5 highest-leverage invariants:**

1. **HOL bug profile (the 7b5f713 signature).** On backend-bound workloads (matmul, fused_linear_activation), `fetch_skip_backpressure / total_cycles > 0.10` while a backend unit is >90% busy is the exact pre-fix profile. Flag any rebound.

2. **Per-unit busy reconciliation.** For each of {alu, mul, div, ldst, tlookup}: `unit_busy_cycles ≈ unit_instructions × declared_latency` to within 1.10× upper bound. Declared latencies: ALU=1, MUL=3 (parameterizable STAGES), DIV=32 iterative, TLOOKUP=17 warp / pipelined 1/cycle, LDST=variable but ≥1 per addr-gen cycle. Sustained excess = held-state-leak (unit pinned beyond actual work).

3. **IPC ceiling and stall-partition closure.** `ipc ≤ 1.0` (spec §4.3); `total_cycles ≈ active_cycles + scheduler_idle_cycles + panic_drain_cycles` (sum with rounding for partial-active cycles); `scheduler_idle_cycles == scheduler_frontend_stall_cycles + scheduler_stall_backend_cycles` exactly. Gaps signal unaccounted dead cycles.

4. **Memory accounting closure.** `cache_hits + cache_misses == load_hits + load_misses + store_hits + store_misses`; `external_memory_reads ≥ load_misses - mshr_merged_loads` (after MSHR merging); `external_memory_writes ≥ store_hits + store_misses` (write-through every store; spec §5.3); `coalesced_requests + serialized_requests/32 ≥ all loads/stores`.

5. **Per-warp fairness.** Round-robin should yield `max(warp_instructions[w]) / min(warp_instructions[w]) ≤ 1.5` for active warps on uniform workloads. Outliers signal scheduler bias or fetch-pointer-advance bugs (spec §4.2, §4.3).

**Secondary checks (lower leverage but cheap):**
- `branch_predictions ≥ branch_mispredictions ≥ branch_flushes`; mispredictions == flushes.
- `mul_busy_cycles ≥ vdot8_inst × pipeline_depth` (VDOT8 shares MUL pipeline).
- `gather_buffer_port_conflict_cycles ≤ cache_hits` (HIT can only lose port to FILL).
- `secondary_drain_cycles ≤ mshr_merged_loads + mshr_merged_stores`.
- Cross-config scaling: doubling warps from 4→8 should not decrease aggregate IPC (with sufficient backend); if it does, scheduler arbitration is buggy.

Anomalies → findings. Each finding cites the failing equation, the JSON file, and the suspected code site (grep for the counter's increment to locate).

## Phase 3 — Trace-anomaly pass (~45 min)

Python over trace JSONs. Use `resources/perfetto_trace_queries.sql` patterns where applicable; otherwise direct JSON traversal. Cite event types from `trace_and_perf_counters.md`.

**Top-5 highest-leverage anomalies:**

1. **AT_REST without `rest_reason`.** Walk complete events with `name == "AT_REST"`; flag any with missing/null/`NONE` reason. This is the major observability gap — the simulator currently can't explain *why* a warp is resting in some cases. Cross-cut with Phase 8 (add `rest_reason_unset_cycles` counter).

2. **Same-warp consecutive issue.** Walk `issue` instant events (`ph:"i"`); if warp X issues at cycle N and N+1, check whether opcoll was held by X (legitimate VDOT8 2-cycle). If not legit, flag — this is the 7b5f713 shape (fetch picks same warp twice due to capacity gate).

3. **Scheduler-idle while warp eligible.** Cross-reference `scheduler_idle_cycles` ticks against per-warp slices; any cycle where scheduler is idle AND ≥1 warp is in DECODE_PENDING/AT_REST without `WAIT_OPCOLL`/`WAIT_UNIT_*`/`WAIT_BRANCH_SHADOW`/`WAIT_SCOREBOARD` rest reason = scheduler arbitration bug or unaccounted stall reason.

4. **Per-load latency outliers.** For each load (extract from `EXECUTE_*`/`MEMORY_WAIT` slice durations), compute distribution. Flag any load > 4× p50 unaccompanied by `WAIT_L1_MSHR`, `WAIT_L1_WRITE_BUFFER`, or `WAIT_MEMORY_RESPONSE` reason. Outliers without reason = head-of-line blocking or arbitration lossless to the trace.

5. **Pipeline-bubble cycles unexplained by stall reasons.** Cycle where unit U is idle but a warp Y is in `WAIT_UNIT_<U>` rest state. Should be impossible — flag as either ready-out logic bug or rest_reason mis-attribution.

**Secondary trace checks:**
- `gather_buffer_port_conflict_cycles` bursts > 5 consecutive cycles signal first-writer-wins arbitration starving HIT path (row 11 in discipline doc).
- `branch_redirect` events not preceded by a matching `issue` for the same warp 1+ cycles earlier = redirect-tracking bug.
- `ldst_fifo_depth` counter saturating against capacity → check for HOL on coalescer/cache stall.
- `pinned_lines` counter spikes coincident with `line_pin_stall` events confirm pin discipline; absence of stalls when counter > 0 = stall miscounting.

## Phase 4 — Static cross-stage signal audit (~60 min, fan-out via sub-agents)

Walk every row of `resources/timing_discipline.md`'s inventory (rows 1–14 + row 15 = 15 rows total). For each row, spawn one sub-agent with this contract (parallel up to 5 at a time):

> Row N: `<producer>` → `<consumer>` carrying `<payload>` classified as `<class>`. Verify in code:
> 1. Locate writer call site(s). Cite file:line.
> 2. Locate reader call site(s). Cite file:line.
> 3. For REGISTERED edges: confirm reads use `current_*` only. Grep the consumer's eval body for `next_<field>` references — any hit is a finding.
> 4. For COMBINATIONAL edges: confirm `tick()` order in `timing_model.cpp:426–500` puts producer's `evaluate()` before consumer's. Confirm a comment at the call site declares the COMBINATIONAL edge per discipline §76–93.
> 5. For READY/STALL edges: confirm the accessor is `const`, reads only `current_*` (or single-slot fields with documented same-tick semantics like `cache.stalled_`), and is queried during the consumer's evaluate.
> 6. Flag any "Forbidden pattern" (discipline §192–215): plain mutable members read mid-evaluate, pre-evaluate setters, `consume_*` writing the other stage's `current_*`, mid-tick mutation of committed state.
>
> Output: row N status, citations, and any findings with severity Definite/Probable.

**Top-5 highest-leverage rows** (run these even if fan-out is dropped):

- **Row 7**: per-unit `current_pending_*`/`current_result_buffer_` — largest concurrent-write surface; `WritebackArbiter::consume_result()` path is the canonical "consume_* writes the other stage" hazard, must verify it writes only `next_*`.
- **Row 9**: cache↔coalescing same-tick stall (newest carve-out from Phase 9); verify `cache.evaluate()` strictly precedes `coalescing.evaluate()` and that `cache.is_stalled()` is fully derived from committed state.
- **Row 11**: `next_port_claimed_` first-writer-wins — three writers (cache fill, secondary drain, coalescing hit). Verify priority encoding by tick order matches spec §5.3 (FILL > secondary > HIT).
- **Row 5**: BranchShadowTracker — three writers (scheduler issue, opcoll resolve, fetch redirect-apply). Verify scheduler reads `current_` before opcoll/fetch clear `next_`. Trace through a mispredict cycle in `test_branch.cpp` to confirm.
- **Row 12**: RedirectRequest — opcoll writes `next_redirect_request_` during evaluate, fetch.commit / decode.commit read `current_redirect_request()`. Verify commit ordering puts fetch.commit/decode.commit before opcoll.commit.

## Phase 5 — Held-state register audit (~45 min, fan-out via sub-agents)

For each held-state register, spawn a sub-agent with this contract:

> Register `R` lives in `<owner>::<field>`. Bug pattern (7b5f713 template): some consumer C gates on whether warp W has too many in-flight instructions, but C's gate omits R from the count.
>
> 1. Find all writers of R. Cite file:line.
> 2. Find all consumers that gate on R or related state via grep across `sim/src/timing/`. Cite file:line for each.
> 3. For each consumer, examine its capacity / eligibility / arbitration logic. Does it count R toward in-flight totals where appropriate? Note any consumer that gates on a *related* field (e.g., `decode.pending_`) but not R itself.
> 4. Output a table: `consumer | gates_on | counts_R? (Y/N) | citation`. Any "N" where Y is expected = Definite finding.

**Registers to audit** (top-5 highest-leverage first):

1. `FetchStage::current_output_` (already-bitten; verify current state reflects fix).
2. `DecodeStage::pending_` (paired writer).
3. `OperandCollector::current_busy_` + `current_instr_` (held 1–2 cycles for VDOT8).
4. Each `ExecutionUnit::current_pending_*` and `current_result_buffer_` (5 units; main consumer is `WarpScheduler::evaluate` via `unit->ready_out()`).
5. `LoadGatherBufferFile::buffers_[warp]` (per-warp; arbitrated via `next_port_claimed_`).

**Secondary registers:**
- `LdStUnit::next_addr_gen_fifo_` (FIFO mid-fill across cycles; coalescer drains).
- `MultiplyUnit::current_pipeline_` (deque).
- `BranchShadowTracker::current_[w]` (already audited as part of row 5).

For each consumer, additionally check its **relationship to scheduler eligibility predicates** in `WarpScheduler::evaluate` and `FetchStage::evaluate`. Spec §4.2 enumerates: buffer-not-empty AND no-branch-shadow AND scoreboard-clear AND opcoll-free AND target-unit-ready. Walk each predicate and verify it accounts for held registers.

## Phase 6 — Synthetic stress workloads (~3–4 hours)

Author 4–6 minimal kernels under `/workspace/tests/synthetic/` mirroring existing benchmark structure (`<name>_bench/<name>_bench.cpp` + assembled ELF). Each kernel targets one ambiguous spec area.

**Top-5 highest-leverage kernels:**

1. **`rr_tiebreak`** — two warps with identical PCs, identical instruction streams, no memory ops. Probes spec §4.2/§4.3 "first eligible warp" tie-break. Expected counter check: per-warp instruction count ratio = 1.00 ± 0.02. Deviation = scheduler bias.

2. **`line_boundary_load`** — coalesced load with 32-thread addresses straddling cache-line boundary 0x80. Probes spec §5.2 "all 32 in single 128-byte line" boundary handling. Expected: `serialized_requests` increments (not coalesced). Off-by-one in the boundary check would coalesce incorrectly.

3. **`mshr_same_line_race`** — 4 warps issue loads to the same cache-line address near-simultaneously. Probes spec §5.3.1 MSHR linear scan + secondary chain. Expected: 1 primary + 3 secondary; chain ordered by allocation cycle. Verify via trace `cache_miss_alloc` events.

4. **`jalr_storm`** — JALR on every other instruction. Probes spec §4.2 static-mispredict path + redirect propagation. Expected: `branch_mispredictions == jalr_count`; `branch_flushes == branch_mispredictions`. Cycle count should match analytical: `jalr_count × (resolve_cycles + 2 refill cycles)`.

5. **`divide_chain`** — chain of dependent divides. Probes spec §4.6 div-by-zero behavior (`-1`, no panic) + 32-cycle iterative latency. Verify trace shows EXECUTE_DIV slices of exactly 32 cycles; verify scoreboard releases at cycle 32.

**Optional sixth (`panic_drain_test`):** kernel that hits EBREAK with various pipeline-depth in-flight states. Probes spec §4.8.1 panic drain bound and "in-flight" definition. Expected drain ≤ 32 cycles. Likely surfaces ambiguity rather than bug.

For each: run; compare measured behavior against analytical expectation derived from spec; record either match (no finding) or mismatch (Definite or Spec-Ambiguity finding).

## Phase 7 — Spec-ambiguity catalog (~60 min, manual walk)

Walk `/workspace/resources/gpu_architectural_spec.md` end-to-end. For each ambiguous statement (vague wording, undefined tie-break, missing edge case), record:

```
## Finding F-NN: Spec ambiguity in §X.Y "<topic>"
- Severity: Spec-Ambiguity
- Quoted text: "<verbatim from spec>"
- Implementation pinning: <file:line where impl makes a specific choice>
- Implementation behavior: <what it actually does>
- Suggested clarification target: <which sentence to make precise; do not write the clarification>
```

**Top-5 pre-flagged ambiguities** (from Explore phase):

1. §4.2 "first eligible warp" — tie-break rule undefined.
2. §5.3.1 MSHR linear scan — direction (low-to-high index?) and first-match-vs-first-empty undefined.
3. §5.2 "single 128-byte cache line" — boundary handling at 0x7F→0x80 transitions.
4. §4.8.1 "drain in-flight instructions" — does "in-flight" include coalescer/cache pending fills?
5. §4.7 writeback arbiter — pointer wrap behavior on last unit; tie-break across simultaneous unit completions.

**Additional candidates to walk:** §3 warp model "ready" predicate definition; §4.3 "loose round-robin" formal definition; §5.6 read/write FIFO sharing; §4.2 fetch pointer advance vs. fetch decision ordering; §4.8.1 panic-drain 32-cycle timeout reasoning.

## Phase 8 — Read-only instrumentation (as needed)

Add new counters when Phase 2/3 surfaces an observability gap. Counters land in `stats.h` + `stats.cpp` + the appropriate timing stage. Branch: `bughunt/observability`. **No behavioral changes.** Each new counter:
- Documented in `resources/trace_and_perf_counters.md` (per project rule).
- Justified in the corresponding finding (severity Observability-Gap, with the gap it closes).

**Pre-identified candidates** (add only if needed):
- `rest_reason_unset_cycles` (per-warp): cycles a warp is in AT_REST with no `WarpRestReason`.
- `consecutive_same_warp_issue_cycles`: count of cycles where warp(N) == warp(N-1) at issue.
- `unit_idle_with_dependent_warp_cycles[unit]`: cycles unit is idle but ≥1 warp is in `WAIT_UNIT_<unit>`.
- `mshr_secondary_chain_max_depth`: largest secondary-chain depth observed.
- `gather_buffer_hit_starvation_cycles[warp]`: per-warp count of FILL beating HIT in port arbitration.

Do not add counters speculatively. Each must trace to a finding that needs it.

## Phase 9 — Findings consolidation & report (continuous + final pass, ~30 min)

The findings file is appended throughout. Final pass:
- Renumber findings sequentially (F-01 onward).
- Tag each with all three of: severity, category, phase-of-discovery.
- Cross-reference duplicates (e.g., a Phase 2 counter anomaly and Phase 4 code citation both pointing to the same root cause merge into one finding).
- Emit a top-of-file summary table: `# | Severity | Title | File`.
- Emit a "Spec-clarification recommendations" section listing all Spec-Ambiguity findings together.
- Emit an "Observability gaps" section with proposed new counters.

Do **not** propose fixes. Do not write spec text. Hand off to user for triage.

## Execution graph & dependencies

```
Phase 0 (bootstrap)
   |
   v
Phase 1 (data collection)
   |          \
   v           v
Phase 2     Phase 3      Phase 4 (independent of 1)     Phase 7 (independent)
(counters)  (traces)     (cross-stage audit)            (spec walk)
   |          |              |                              |
   |          v              v                              |
   |       Phase 8 (counter additions, only if 2/3 finds gaps)
   |          |              |                              |
   v          v              v                              v
Phase 5 (held-state audit; can start in parallel with 4)
   |
   v
Phase 6 (synthetic kernels — depends on understanding from 1-5)
   |
   v
Phase 9 (consolidation)
```

Phases 4, 5, 7 can run in parallel (sub-agent fan-out for 4 and 5). Phases 2, 3 depend on Phase 1. Phase 6 is sequenced last for full context.

## Verification end-to-end

To validate this plan executed correctly:

1. **Build clean:** `cmake --build /workspace/build -j8` succeeds.
2. **Phase 1 dataset complete:** `find /workspace/traces/bughunt -name '*.json' | wc -l` ≥ 96 (8 configs × 6 benches × 2 [stats+trace]). Lower count = missing tuples; investigate.
3. **Findings ledger populated:** `/workspace/project-plans/timing-bug-hunt-findings.md` exists with at least one finding per phase that ran (excluding "all clean" outcomes, which are still recorded as `## Phase N: clean`).
4. **All phase logs present:** each phase's `## Phase N session log` block recorded with start time, end time, files inspected, sub-agent IDs (if used).
5. **Tests still pass:** `ctest --test-dir /workspace/build` green (regression protection if Phase 8 instrumentation landed).
6. **Spec & docs untouched:** `git diff main -- resources/gpu_architectural_spec.md` empty (the hunt does not modify spec; clarifications are findings only).
7. **Spot-check a Definite finding:** pick one Definite finding, walk the citations, confirm the violation reproduces (counter value matches reported value; or static violation appears at the cited file:line).

## Stopping condition

All in-scope phases (0–9) complete. Stopping is not gated on a target finding count — zero-finding outcomes are valid and still recorded ("Phase N: clean"). The user reviews the consolidated ledger and decides which findings to triage / fix in subsequent sessions.
