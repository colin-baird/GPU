# Untested Changes

Changes that passed regression but have not yet received targeted test coverage. Each entry should be removed once targeted tests are written and committed, or once the change is reverted.

## Format

```
### <short description>
- **Date:** YYYY-MM-DD
- **Commit:** <short hash>
- **What changed:** <brief description of the architectural/implementation change>
- **Why untested:** <"pending user decision" | "deferred" | other reason>
- **Test priority:** <high | medium | low>
```

## Entries

### External-memory read-latency instrumentation
- **Date:** 2026-04-25
- **Commit:** (logged during fixed-latency calibration to DRAMSim3 mean)
- **What changed:** Added `Stats::external_read_latency_total` / `external_read_latency_count`, populated by both `FixedLatencyMemory` (adds `latency_` per completion) and `DRAMSim3Memory` (measures `fabric_cycle_ - submit_cycle` per completed read). Used to recalibrate the fixed-backend default from 100 → 17 cycles to match the weighted-mean DDR3 read latency across the workload benchmark suite.
- **Why untested:** deferred — counters are exercised end-to-end by every benchmark run, but no targeted Catch2 case asserts the accumulation arithmetic on either backend.
- **Test priority:** low

### TLOOKUP intra-warp 2-lane/cycle BRAM pipelining
- **Date:** 2026-04-23
- **Commit:** (logged during TLOOKUP timing-test retrofit)
- **What changed:** The architectural spec §2.3 / §4.5 describes the TLOOKUP unit as a pipelined dual-port BRAM reading 2 lanes/cycle (16 issue + 1 drain = 17 cycles). The timing model collapses the 32 lanes into a single 17-cycle countdown with no per-lane state exposed at the unit interface.
- **Why untested:** structural gap — intra-warp per-lane progress is not observable without either instrumentation exposing per-lane state or a lower-level structural model. Only the 17-cycle aggregate warp latency is bound.
- **Test priority:** low

### Writeback-stall freeze: multi-cycle re-evaluation idempotence and branch-resolution coincidence
- **Date:** 2026-05-18
- **Commit:** `29dd299` (writeback stall), `c97b7ac` (`branch_resolved_` reuse)
- **What changed:** The combinational-backward writeback stall (Phase 10B.3) freezes the five execution units, the operand collector, and the warp scheduler by gating `commit()` when a load preempts a fixed-latency writeback; a frozen stage's `evaluate()` re-runs and must produce byte-identical `next_*` state, which the `seed_next()`/`commit()` discipline guarantees by construction. Phase 10E reuses the ALU `branch_resolved_` bit so a branch held at the resolve stage across a multi-cycle stall asserts the redirect (and the predictor/tracker side-effects) exactly once.
- **Why untested:** deferred — the stall has arbiter-level targeted tests (fixed-priority scan, load-preempts-fixed assertion) plus the offset-table interleave regression, and is exercised end-to-end by every workload benchmark (`fixed_writeback_preempted_cycles` is non-zero). `test_instruction_latency.cpp` ("Writeback contention" case) now pins corner (a) for the **ALUUnit**: a result held across one and two consecutive stall cycles is retired unchanged with its issue→writeback latency extended by exactly the stall count. Still deferred: (a) for the *iterative/pipelined* units whose `seed_next()` re-establishes non-trivial carry-forward state (the divide/tlookup countdowns, the multiply pipeline deque) across a ≥2-cycle freeze, and (b) once-only redirect/predictor/tracker side-effects when a branch resolves on a writeback-stalled cycle. These remaining corners are argued by construction and covered statistically, not asserted by a dedicated test.
- **Test priority:** medium
