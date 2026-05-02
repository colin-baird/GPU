# Timing Bug Hunt — Progress

> Plan: `/workspace/project-plans/lively-conjuring-neumann.md`
> Findings: `/workspace/project-plans/timing-bug-hunt-findings.md`

## Calibration

- mean atom wall: ~30 ms (Phase 1 fixed atoms); ~50 ms (DRAMSim3); reasoning atoms (Phase 4/5/7) TBD per-atom
- p95 atom wall (Phase 1): ~150 ms
- last session 1 budget: 42 Phase 1 atoms + Phase 2 audit completed in <2 minutes wall + reasoning time
- **Updated session budget cap**: 90 min wall OR 8 reasoning atoms OR any single atom > 20 min. Phase 1 atoms ignore the cap entirely (batch as many as wanted).

## CLI knob inventory (verified 2026-04-27, build sha 7b5f713)

Bench binaries (matmul_bench etc.) expose: `--num-warps`, `--memory-backend`, `--memory-latency`, `--max-cycles`, `--dramsim3-config-path`, `--json`. They do NOT expose `--trace-file`, `--num-mshrs`, `--buffer-depth`.

Runner `build/runner/gpu_sim` exposes `--trace-file`, `--config=<file.json>` (JSON SimConfig), `--num-warps` (capped at 8), but requires explicit kernel args via `--arg0..3` and explicit kernel ELF.

**Implications for Phase 1:**
- MSHR/buffer-depth sweeps require either a JSON config + runner OR new CLI flags. Logged as F-01 Observability-Gap (deferred).
- Trace collection requires runner-with-ELF or new bench flag. Phase 3 deferred until traces are available; will plumb on bughunt/observability branch later.

## Sweep matrix (revised, all bench-supported)

| Config tag | Args |
|-----------|------|
| `fixed-default` | (defaults: --memory-backend=fixed) |
| `dramsim3-default` | `--memory-backend=dramsim3 --dramsim3-config-path=/workspace/sim/configs/dram/DDR3_4Gb_x16_800.ini` |
| `fixed-warps-2` | `--num-warps=2` |
| `fixed-warps-4` | `--num-warps=4` |
| `fixed-warps-8` | `--num-warps=8` |
| `fixed-warps-16` | `--num-warps=16` |
| `fixed-latency-8` | `--memory-latency=8` |
| `fixed-latency-64` | `--memory-latency=64` |

48 Phase 1 atoms across 8 configs × 6 benches.

## Completed (session 1: 2026-04-27)

- [phase 0.bootstrap] @ 2026-04-27 — sha 7b5f713; build clean; dirs created; ledger + progress initialized
- [phase 1.all-42-tuples] @ 2026-04-27 — wall ~1.5s total; outputs in /workspace/traces/bughunt/stats/{fixed-default,dramsim3-default,fixed-warps-{2,4,8},fixed-latency-{8,64}}/
- [phase 2.counter-audit] @ 2026-04-27 — analysis script /tmp/phase2_audit.py; surfaced F-01, F-02, F-03, F-04, F-05
- [phase 4.row-7] @ 2026-04-27 — execution units + WB arbiter; COMPLIANT, no findings
- [phase 4.row-11] @ 2026-04-27 — gather buffer port arbitration; COMPLIANT + F-07 doc-stale
- [phase 4.row-5] @ 2026-04-27 — BranchShadowTracker; COMPLIANT
- [phase 4.row-9] @ 2026-04-27 — cache↔coalescing COMBINATIONAL stall; COMPLIANT
- [phase 4.row-12] @ 2026-04-27 — RedirectRequest cross-commit; COMPLIANT
- [phase 5.fetch.current_output_] @ 2026-04-27 — 7b5f713 fix verified complete
- Inline disposition: F-03 root cause identified (drain bucket); F-05 confirmed (counter/spec semantic mismatch); F-04 → F-06 reclassified
- [phase 6.kernels] @ 2026-04-28 — 5 synthetic kernels authored (rr_tiebreak, line_boundary_load, mshr_same_line_race, jalr_storm, divide_chain) under tests/synthetic/; built; all `add_test`'d (ctest 26/26 pass); ran with --json + --trace-file; analyzed against analytical spec expectations. Findings: F-18 (Definite, serialized_requests counter ↔ doc mismatch), F-19 (Spec-Ambiguity, JALR "always mispredicted" wording vs conditional impl), F-20 (Definite, EXECUTE_DIV trace slice 1 cyc short of busy_cycles counter). Plus 4 positive spec-conformance verifications recorded as non-findings.
- [phase 6.kernel.panic_drain_test] @ 2026-04-28 — optional 6th kernel authored (panic_drain_test). Probes spec §4.8.1 panic drain bound + "in-flight" definition. Single-warp DIV→mv-r31→EBREAK forces a 32-cyc DIV in-flight at panic trigger; bench manually ticks and records `last_cycle_snapshot()->panic_active` to measure the panic-active span. Result: panic span = 32 cyc (within MAX_DRAIN_CYCLES + 2 fixed-overhead bound), panic_cause correctly latches r31 = 0x101, div_busy_cycles = 32 (DIV completes normally during drain). Trace confirms spec §4.8.1 steps 1-4 take 3 cycles (cycle 7 decode-detect → cycle 8 latch → cycle 9 drain step begins). All 27 ctest tests pass. No new finding — confirms spec-conformant behavior.
- [phase 6.audit-of-prior-work] @ 2026-04-28 — re-verified all 5 prior kernels' code citations (F-18 at coalescing_unit.cpp:50 + trace_and_perf_counters.md:415; F-19 at branch_predictor.cpp:24-26 + timing_model.cpp:322-335; F-20 at divide_unit.cpp:20-32 + divide_unit.h:30-32 + timing_model.cpp:670-673). Each citation walked end-to-end and confirmed accurate. Re-ran each kernel against its analytical pass criterion: rr_tiebreak (W0..W3=164 each, imbalance=0, 0 consec same-warp issues), line_boundary_load (1 coalesced + 1 serialized, 9 cache_misses = 3 primaries + 6 secondaries), mshr_same_line_race (1 primary + 3 secondaries on line 32, allocs at 18/22/26/30 = 4-cyc cadence), jalr_storm (W0=33 mispredicts at iter=32, IPC 0.50→0.99 from 1→4 warps), divide_chain (576 div_busy = 18×32 exact, all 18 trace slices dur=31 confirming F-20).

## Next

- [phase 5.opcoll.current_busy_+current_instr_] — Explore audit per the held-state register template; trace consumers gating on opcoll being free
- [phase 5.exec_units.current_pending_+result_buffer_] — Explore audit; main consumer is WarpScheduler::evaluate via unit->ready_out()
- [phase 5.gather_file.buffers_per_warp] — Explore audit; per-warp gather buffer; arbitrated via row 11 port
- [phase 4.row-1] — DecodeStage::ready_to_consume_fetch / FetchStage
- [phase 4.row-2] — DecodeStage::pending_warp accessor / FetchStage
- [phase 4.row-3] — OperandCollector::ready_out / WarpScheduler
- [phase 4.row-4] — ExecutionUnit::ready_out / WarpScheduler
- [phase 4.row-6] — OperandCollector::accept (REGISTERED)
- [phase 4.row-8] — WritebackArbiter -> Scoreboard
- [phase 4.row-10] — L1Cache external surface (Phase-9 boundary discipline)
- [phase 4.row-13] — EBreakRequest / panic flush cascade
- [phase 4.row-14] — PanicController::set_drained_query
- [phase 4.row-15] — L1Cache <-> mem_if (DRAMSim3 carve-out)
- [phase 7.spec-walk-§4.2] — fetch wording; first-eligible tie-break; pointer-advance ordering
- [phase 7.spec-walk-§4.3] — scheduler "loose round-robin" definition
- [phase 7.spec-walk-§5.3.1] — MSHR linear-scan order (direction, first-match-vs-first-empty)
- [phase 7.spec-walk-§5.2] — coalescing line-boundary handling at 0x7F→0x80
- [phase 7.spec-walk-§4.8.1] — panic drain "in-flight" definition
- [phase 7.spec-walk-§4.7] — writeback arbiter wrap behavior + F-05 follow-up
- [phase 8.observability-counter-additions] — only if new gaps surface
- [phase 9.consolidation] — final pass, renumber findings, summary table

## Pending (queued in order)

- phase 1.gemv.fixed-default
- phase 1.fused_linear_activation.fixed-default
- phase 1.softmax_row.fixed-default
- phase 1.embedding_gather.fixed-default
- phase 1.layernorm_lite.fixed-default
- phase 1.matmul.dramsim3-default
- phase 1.gemv.dramsim3-default
- phase 1.fused_linear_activation.dramsim3-default
- phase 1.softmax_row.dramsim3-default
- phase 1.embedding_gather.dramsim3-default
- phase 1.layernorm_lite.dramsim3-default
- phase 1.matmul.fixed-warps-2
- phase 1.matmul.fixed-warps-4
- phase 1.matmul.fixed-warps-8
- phase 1.matmul.fixed-warps-16
- phase 1.matmul.fixed-latency-8
- phase 1.matmul.fixed-latency-64
- (… repeat warp/latency variants for gemv, fused, softmax, embedding, layernorm)
- phase 4.row-1 — DecodeStage::ready_to_consume_fetch / FetchStage
- phase 4.row-2 — DecodeStage::pending_warp / FetchStage
- phase 4.row-3 — OperandCollector::ready_out / WarpScheduler
- phase 4.row-4 — ExecutionUnit::ready_out / WarpScheduler
- phase 4.row-5 — BranchShadowTracker (3 writers)
- phase 4.row-6 — OperandCollector::accept (REGISTERED)
- phase 4.row-7 — ExecutionUnit accept/consume_result (largest concurrent surface)
- phase 4.row-8 — WritebackArbiter -> Scoreboard
- phase 4.row-9 — CoalescingUnit / cache.is_stalled COMBINATIONAL (Phase-9 carve-out)
- phase 4.row-10 — L1Cache external surface
- phase 4.row-11 — LoadGatherBufferFile::next_port_claimed_ (3 writers)
- phase 4.row-12 — RedirectRequest cross-commit ordering
- phase 4.row-13 — EBreakRequest / panic flush cascade
- phase 4.row-14 — PanicController::set_drained_query
- phase 4.row-15 — L1Cache <-> mem_if (DRAMSim3 carve-out)
- phase 5.fetch.current_output_
- phase 5.decode.pending_
- phase 5.opcoll.current_busy_+current_instr_
- phase 5.execution_units.current_pending_+result_buffer_
- phase 5.gather_file.buffers_per_warp
- phase 5.ldst.next_addr_gen_fifo_
- phase 5.multiply.current_pipeline_
- phase 5.scheduler-eligibility-predicate-walk
- phase 7.spec-walk-§3-warp-model
- phase 7.spec-walk-§4.2-fetch
- phase 7.spec-walk-§4.3-scheduler
- phase 7.spec-walk-§4.6-execution-units
- phase 7.spec-walk-§4.7-writeback
- phase 7.spec-walk-§4.8-panic
- phase 7.spec-walk-§5.2-coalescing
- phase 7.spec-walk-§5.3-cache
- phase 7.spec-walk-§5.6-external-memory
- (Phase 2/3/6/8/9 atoms queued after Phase 1 completes)

## Deferred / blocked

- Phase 3 (trace audit): blocked on adding `--trace-file` to bench binaries OR using runner with ELF + extracted args. Will plumb on bughunt/observability branch in a future session.

## Pending findings (rolled up at end of session)

- F-01 (Observability-Gap, queued): bench binaries do not expose `--num-mshrs` or `--buffer-depth`, so MSHR-depth and buffer-depth sweeps are not directly executable. The runner accepts `--config=<file.json>` for SimConfig overrides, but bench binaries do not. To enable these sweeps, either (a) add `--num-mshrs` / `--buffer-depth` to bench CLI parsing, or (b) extract kernel ELFs + args and use the runner with custom JSON config. Phase 1 sweep substituted with warp-count and memory-latency variants in the meantime.

## Phase 6 status (consolidated 2026-04-28)

Phase 6 is **complete**. All 5 mandatory + 1 optional kernels authored, built, registered with ctest, and analyzed against analytical pass criteria derived from spec §4.2/§4.3/§4.6/§4.7/§4.8.1/§5.2/§5.3.1.

| Kernel | Spec target | Pass criterion | Status |
|--------|-------------|----------------|--------|
| `rr_tiebreak` | §4.2/§4.3 RR tie-break | per-warp imbalance == 0 | PASS |
| `line_boundary_load` | §5.2 boundary detection | coalesced=1 + serialized=1 per warp | PASS (surfaces F-18) |
| `mshr_same_line_race` | §5.3.1 same-line merging | 1 primary + N-1 secondaries, ordered by alloc | PASS |
| `jalr_storm` | §4.2 mispredict + redirect | mispredicts == N+1 per warp, flushes match | PASS (surfaces F-19) |
| `divide_chain` | §4.6 32-cyc DIV + div-by-zero | div_busy = 32 × instr; div0 = -1 | PASS (surfaces F-20) |
| `panic_drain_test` | §4.8.1 drain bound + "in-flight" | panic_span ≤ MAX_DRAIN+2; r31 latch correct; DIV completes | PASS (no new finding) |

Resolves the earlier session-log contradiction: an early 2026-04-28 entry stated "Phase 6 (synthetic kernels): DEFERRED to user triage; not pursued"; that statement is stale. The 5 mandatory kernels were authored later that same day, and the optional 6th was added on 2026-04-28 (verification pass) — Phase 6 is now fully complete.

