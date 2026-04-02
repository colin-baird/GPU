# Performance Alignment Validation

This document defines the active workflow for validating that the timing model is sufficiently aligned with the architectural specification to support architectural experimentation.

The validation policy is **spec-first**:

- The architecture spec is authoritative for visible timing behavior.
- Simulator behavior that affects timing but is not covered by the spec is a defect until the spec is explicitly updated.
- The retired analytical Python reference flow is no longer a gating oracle.

## Primary Gate

The primary gate is the manifest-driven Catch2 target:

- Test binary: `build/sim/tests/test_alignment`
- Test source: `/sim/tests/test_alignment.cpp`
- Manifests: `/tests/alignment/manifests/*.manifest`

Run it with:

```bash
ctest --test-dir build -R test_alignment --output-on-failure
```

The intent is that this gate stays green before substantial architectural experiments resume.

## Validation Surfaces

The active gate validates against these simulator surfaces:

- `Stats` / `report_json()` counters for total cycles, issue counts, branch flushes, unit instruction counts, cache counters, external memory traffic, writeback conflicts, and selected per-warp stall counters
- committed `CycleTraceSnapshot` classifications from `TimingModel::last_cycle_snapshot()`
- `WarpTraceState` / `WarpRestReason` names from `timing_trace.h`
- final architected register state
- panic diagnostics (`warp`, `pc`, `cause`)

The following counters remain **informational by default** and are not part of every manifest:

- `branch_predictions`
- `branch_mispredictions`
- `fetch_skip_count`

They are still useful for frontend diagnostics, but they are not the core architectural contract.

## Manifest Schema

Each manifest is a line-oriented key/value file. Blank lines and lines beginning with `#` are ignored.

Required keys:

- `scenario = <name>`
- `builder = <scenario_builder_name>`
- `citation = <spec reference>` may appear more than once

Configuration overrides:

- `config.<field> = <value>`
- Supported fields today: `num_warps`, `instruction_buffer_depth`, `multiply_pipeline_stages`, `num_ldst_units`, `addr_gen_fifo_depth`, `l1_cache_size_bytes`, `cache_line_size_bytes`, `num_mshrs`, `write_buffer_depth`, `external_memory_latency_cycles`, `start_pc`, `arg0`..`arg3`

Expectations:

- `stat.<field> = <value>`
- `warp.<warp_id>.<field> = <value>`
- `cycle.<cycle>.<field> = <value>`
- `cycle.<cycle>.warp<warp_id>.<field> = <value>`
- `reg.<warp_id>.<lane_id>.<reg_id> = <value>`
- `panic.<field> = <value>`

Supported numeric comparison operators:

- `= 42`
- `= >= 1`
- `= <= 12`

Supported cycle fields:

- snapshot fields: `active_warps`, `opcoll_busy`, `alu_busy`, `mul_busy`, `div_busy`, `ldst_busy`, `active_mshrs`, `write_buffer_depth`, `panic_active`
- warp fields: `state`, `rest_reason`, `pc`, `dest_reg`, `active`, `branch_taken`

Supported warp stats fields:

- `instructions`
- `cycles_active`
- `stall_scoreboard`
- `stall_buffer_empty`
- `stall_unit_busy`

Supported panic fields:

- `panicked`
- `warp`
- `cause`
- `pc`

## Current Scenario Set

The current manifest set covers the highest-risk timing rules:

| Scenario | Focus |
|----------|-------|
| `simple_pipeline` | 1-warp fill/drain baseline |
| `alu_chain` | dependent ALU chain without extra scoreboard stalls |
| `mul_dependency` | multiply latency and RAW visibility |
| `div_dependency` | divide latency and scoreboard stall duration |
| `branch_taken` | forward-taken branch mispredict recovery and buffer flush |
| `jal_predicted_taken` | direct JAL predicted taken without frontend flush |
| `load_miss_use` | stall-on-use for a cold load miss |
| `store_then_load_same_line` | duplicate same-line misses plus write-through drain |
| `writeback_conflict` | round-robin writeback arbitration delay |
| `panic_drain` | EBREAK panic sequencing and diagnostics |
| `serialized_load` | all-or-nothing coalescing fallback |
| `four_warp_round_robin` | 4-warp fetch/issue fairness baseline |
| `four_warp_mshr_pressure` | 4-warp MSHR exhaustion backpressure |

## Legacy Analytical Flow

`/tests/references/` and `/resources/perf_reference_methodology.md` are retained as historical context only.

They remain useful for exploratory comparison, but they are **not** the primary timing oracle and are **not** the release gate for performance-model alignment.
