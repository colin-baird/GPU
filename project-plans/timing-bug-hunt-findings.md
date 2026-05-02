# Timing Bug Hunt — Findings Ledger

> Plan: `/workspace/project-plans/lively-conjuring-neumann.md`
> Progress: `/workspace/project-plans/timing-bug-hunt-progress.md`

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

## Findings

## Finding F-01: Bench CLI does not expose MSHR-count or buffer-depth knobs
- Severity: Observability-Gap
- Category: Counter inconsistency
- Evidence: `tests/matmul/matmul_bench.cpp:69-72` lists CLI options as `--num-warps`, `--memory-latency`, `--max-cycles`, `--memory-backend`, `--dramsim3-config-path`, `--json`. Nothing exposes `SimConfig::num_mshrs` or instruction-buffer depth. The standalone `runner/src/main.cpp` accepts `--config=<file.json>` for SimConfig overrides but requires explicit kernel ELFs and `--arg0..3`, and is capped at `--num-warps≤8`.
- Code citation: `sim/include/gpu_sim/config.h` (SimConfig fields exist), but no CLI plumbing in any `*_bench.cpp` for the relevant fields.
- Repro hint: `grep "num_mshrs\|num_warps\|memory_backend" tests/*/?*_bench.cpp` shows only num_warps wired.
- Spec section: N/A (observability)
- Discovered: phase 1, 2026-04-27

Recommendation: add `--num-mshrs=<N>` and `--inst-buffer-depth=<N>` to the bench-binary common options (or factor a shared option-parser helper).

## Finding F-02: `total_loads_completed` and `total_load_latency` declared, documented, never incremented
- Severity: Definite
- Category: Counter inconsistency
- Evidence: All 42 (config × bench) Phase-1 tuples report `"total_loads_completed": 0` and `"total_load_latency": 0`, despite e.g. matmul/fixed-default executing `load_hits=25354 + load_misses=7414 = 32768` loads. The text report at `sim/src/stats.cpp:86-89` divides one by the other for "average load latency"; with both at 0, the report prints nothing useful.
- Code citation: `sim/include/gpu_sim/stats.h:62-63` (declarations); `sim/src/stats.cpp:146-147` (JSON emit), `sim/src/stats.cpp:86-89` (text report). `grep -rn "total_loads_completed\|total_load_latency" /workspace/sim/ /workspace/runner/ /workspace/tests/` returns zero increment sites — only declarations, prints, and a doc reference at `resources/trace_and_perf_counters.md:418`.
- Repro hint: any benchmark with `--json`; both fields are always 0.
- Spec section: N/A (counter expected to track per-load latency for `total_load_latency / total_loads_completed` average; the doc claims this average is "emitted in the text report").
- Discovered: phase 2, 2026-04-27

## Finding F-03: Scheduler stall partition leaves "all-warps-inactive" bucket uncounted
- Severity: Probable
- Category: Counter inconsistency
- Evidence: 41/42 Phase-1 tuples have `scheduler_idle_cycles != scheduler_frontend_stall_cycles + scheduler_stall_backend_cycles`. Gap is small (1-141 cycles) but persistent. Examples (idle, fe, be, gap):
  - dramsim3-default/embedding_gather: 46270, 38, 46091, 141
  - fixed-warps-8/matmul: 70611, 722, 69878, 11
  - dramsim3-default/matmul: 69787, 721, 69062, 4
  - fixed-default/gemv: 2462, 25, 2400, 37
- Code citation: `sim/src/timing/warp_scheduler.cpp:139-144`. Increment logic:
  ```cpp
  if (!issued) {
      stats_.scheduler_idle_cycles++;
      if (any_buffer_empty)        stats_.scheduler_frontend_stall_cycles++;
      else if (any_active)         stats_.scheduler_stall_backend_cycles++;
  }
  ```
  The third case `(!any_buffer_empty && !any_active)` — i.e., all warps RETIRED but the scheduler still ticks (program tail / pipeline drain) — increments only `scheduler_idle_cycles` and falls through neither sub-counter. The gap matches the per-program drain length.
- Repro hint: any benchmark; gap correlates with end-of-program drain length and panic-drain cycles.
- Spec section: §4.3 (scheduler stall taxonomy) does not name a "drain idle" category.
- Discovered: phase 2, 2026-04-27. Disposition: classify the third bucket explicitly (e.g., add `scheduler_idle_drain_cycles`) so partition closes, OR widen `scheduler_stall_backend_cycles` to cover `!any_buffer_empty` regardless of any_active (the warps' buffers being non-empty implies prior issuance, so backend-stall-on-drain is a defensible classification). Either fix is mechanical; flag as design choice for the user.

## Finding F-04: embedding_gather IPC does not improve with more warps despite memory-bound profile
- Severity: Probable
- Category: Counter inconsistency / spec ambiguity
- Evidence: `embedding_gather` IPC across warp counts: warps-2=0.033, warps-4=0.030, warps-8=0.030. Total cycles scale linearly with warp count (7101 → 15783 → 31706), and `mshr_stall_cycles` scales roughly linearly (4855 → 11303 → 23049). For a memory-bound workload, additional warps should hide miss latency (more in-flight loads), increasing IPC. Stagnant or declining IPC suggests MSHR/coalescer saturation that more warps cannot help — but the spec (§4.3, latency hiding via warp switching) implies more warps should help. Either (a) MSHR=4 is the bottleneck and additional warps just increase mshr_stall_cycles — a design tradeoff, not a bug; or (b) there is per-warp serialization at the gather buffer / coalescing stage that throttles the workload regardless of warp count.
- Code citation: `sim/src/timing/coalescing_unit.cpp` (mshr_stall path), `sim/src/timing/load_gather_buffer.cpp` (per-warp gather buffer serialization).
- Repro hint: `embedding_gather --num-warps=2|4|8`.
- Spec section: §4.3 ("warp switching hides latency"), §5.2.1 (gather buffer per-warp serialization)
- Discovered: phase 2, 2026-04-27

## Finding F-05: writeback_conflicts counter measures backlog cycles, not simultaneous-completion events (spec/impl mismatch)
- Severity: Definite
- Category: Counter inconsistency
- Evidence:
  - Spec §4.7 (verbatim, `resources/gpu_architectural_spec.md`): "Conflict frequency is **low**: conflicts only occur when two or more units finish their **last thread lane in the same cycle**. With different pipeline depths (1-cycle ALU, multi-cycle multiply, ~32-cycle divide, variable MSHR fill), simultaneous completions are infrequent." This is transition-event semantics: "finish the last thread lane in the same cycle."
  - Implementation (`sim/src/timing/writeback_arbiter.cpp:24-36`):
    ```cpp
    for (uint32_t i = 0; i < sources_.size(); ++i) {
        uint32_t idx = (rr_pointer_ + i) % static_cast<uint32_t>(sources_.size());
        if (sources_[idx]->has_result()) {
            valid_count++;
            if (winner < 0) winner = static_cast<int32_t>(idx);
        }
    }
    if (valid_count > 1) { stats_.writeback_conflicts++; }
    ```
    `has_result()` returns the unit's `next_result_buffer_.valid` — sticky until consumed. After the arbiter selects a winner and calls `consume_result()`, only that unit's slot clears. Other units' valid bits persist into next cycle (committed, then read again). So `valid_count > 1` is true for every cycle that two or more units are *waiting on the arbiter*, not just the cycle they completed.
  - Counter is therefore "cycles with multiple sources holding ready results" — backlog occupancy, not transition events.
  - Measured: layernorm_lite/fixed-default=27.9% of cycles, scaling 6.1%→16.6%→27.9% with warp count 2→4→8. Spec implies "rare" (single-digit %); implementation reports tens of %.
- Code citation: `sim/src/timing/writeback_arbiter.cpp:34` (counter increment), `sim/src/timing/{alu,multiply,divide,tlookup,ldst}_unit.cpp::has_result` (sticky valid bit).
- Repro hint: `layernorm_lite --num-warps=8`.
- Spec section: §4.7 ("conflicts only occur when…finish the last thread lane in the same cycle")
- Discovered: phase 2 + spec walk, 2026-04-27. Disposition: counter wants two semantics — "backlog cycles" (current behavior) and "simultaneous-completion events" (spec's intended low-frequency metric). Recommended: rename current counter to `writeback_backlog_cycles` and add a transition-event counter `writeback_simultaneous_completions` that increments only when a unit's `has_result()` flips from false (last cycle) to true (this cycle) AND another unit also produced this cycle.

## Finding F-06: F-04 reclassified as Spec-Ambiguity (not a bug)
- Severity: Spec-Ambiguity
- Category: Spec ambiguity
- Evidence: F-04 reported `embedding_gather` IPC stagnant under warp scaling (warps-2=0.033, warps-4=0.030, warps-8=0.030). On reflection this is not a bug — it's the expected outcome when MSHRs (default 4) saturate. With more warps, more in-flight loads compete for the same MSHR pool, and `mshr_stall_cycles` scales linearly (4855 → 11303 → 23049). However the spec at §4.3 ("warp switching hides latency") is silent on the saturation regime, and the reader could reasonably expect more warps to monotonically improve IPC. The implementation pinning is reasonable; the spec does not explicitly bound the latency-hiding benefit.
- Code citation: `sim/src/timing/coalescing_unit.cpp` (mshr_stall path), `sim/src/timing/cache.cpp` MSHR allocation.
- Spec section: §4.3 (latency hiding via warps); §5.3.1 (MSHR file capacity). Recommended clarification: §4.3 should note that warp-count latency hiding is bounded by MSHR pool capacity for memory-bound workloads.
- Discovered: phase 2, 2026-04-27. Supersedes F-04.

## Finding F-07: Stale doc comment in cache.h re: removed `current_port_claimed_` field
- Severity: Observability-Gap (documentation drift)
- Category: Counter inconsistency / documentation
- Evidence: `sim/include/gpu_sim/timing/cache.h:175-176` references a "REGISTERED next_port_claimed_ / current_port_claimed_ pair" but the consolidation pass (commit 54e6542 "drop compute_ready slot, unify ready_out, hygiene") removed `current_port_claimed_`. Single-slot `next_port_claimed_` remains and is correct. Comment is misleading but does not affect correctness.
- Code citation: `sim/include/gpu_sim/timing/cache.h:172-181`.
- Repro hint: static; no repro.
- Spec section: N/A
- Discovered: phase 4 row 11 audit, 2026-04-27.

## Phase 4 / 5 / 6 caveat — Haiku-vs-Opus discrepancy

The first wave of Phase 4 / Phase 5 audits used `Explore` sub-agents, which default to **Haiku**. All returned uniform "COMPLIANT" verdicts. On user prompting, the same audits were re-run with `general-purpose` sub-agents on **Opus**, which surfaced 8+ additional findings (F-08..F-16 below). This is significant: a Haiku-class model is not strong enough to do code-discipline audits at the depth needed to catch subtle bugs analogous to 0383f04 / 7b5f713. Findings F-08..F-16 are the substantive yield from re-audit; the Haiku results are kept only as null-baseline.

## Finding F-08: DecodeStage::pending_ is single-buffered; READY/STALL stability claim is brittle
- Severity: Suspicious (Probable design fragility)
- Category: Cross-stage hazard
- Evidence: `DecodeStage::ready_to_consume_fetch()` (`sim/include/gpu_sim/timing/decode_stage.h:54`) and `pending_warp()` both read `pending_.valid`, which is a single non-double-buffered struct mutated mid-tick by `DecodeStage::evaluate()` (`decode_stage.cpp:35-37`). The discipline doc claim (line 46-48) that READY/STALL accessor values are "stable across the entire evaluate phase regardless of where queried" is technically false here — the value depends on whether the read happens before or after decode.evaluate. Currently safe ONLY because `FetchStage::evaluate` (the sole mid-tick caller) runs before `decode.evaluate` per tick order. **Inserting any new mid-tick reader of `ready_to_consume_fetch()` after decode.evaluate would silently observe inconsistent state — exactly the 0383f04 bug template.** The other READY/STALL accessors (`OperandCollector::ready_out`, `ExecutionUnit::ready_out`) genuinely satisfy stability because they read double-buffered current_*.
- Code citation: `sim/include/gpu_sim/timing/decode_stage.h:42-45,54`; `sim/src/timing/decode_stage.cpp:35-37,60`; tick-order at `timing_model.cpp:437-438`.
- Repro hint: static; no repro.
- Spec section: N/A (discipline-doc invariant)
- Discovered: phase 4 row 1 re-audit (Opus), 2026-04-28. Disposition: either (a) double-buffer `pending_` (REGISTERED current_/next_ pair), or (b) tighten the row 1 inventory comment to "stable only when read before decode.evaluate within the same tick."

## Finding F-09: ALUUnit::current_has_pending_ is structurally dead state
- Severity: Probable (dead code; cleanup, not bug)
- Category: Counter inconsistency / design hygiene
- Evidence: `ALUUnit::ready_out()` at `alu_unit.h:14-16` checks `!current_result_buffer_.valid && !current_has_pending_`. ALU completes in 1 cycle: accept sets `next_has_pending_=true` (mid-tick), evaluate consumes it back to false in the same tick (`alu_unit.cpp:33`), commit flips `current_has_pending_=false`. After commit, `current_has_pending_` is structurally always false. The check is redundant. Not a correctness issue, but indicates that `current_has_pending_` is a vestigial field — likely a copy-paste from a multi-cycle unit pattern. Removing it would clarify that ALU is a 1-cycle in-out unit with only result_buffer holding cross-cycle state.
- Code citation: `sim/include/gpu_sim/timing/alu_unit.h:14-16`; `sim/src/timing/alu_unit.cpp:10-13,33,43`.
- Repro hint: static.
- Spec section: N/A
- Discovered: phase 4 row 4/7 re-audit, 2026-04-28.

## Finding F-10: MultiplyUnit::next_pipeline_ has no explicit capacity assertion (bounded by structural invariant only)
- Severity: Probable (defensive cleanup)
- Category: Held-state leak / design hygiene
- Evidence: `MultiplyUnit::accept()` (`multiply_unit.cpp:5-20`) pushes to `next_pipeline_` without checking size. The invariant that bounds the deque at `pipeline_stages_` entries depends on subtle interaction: `ready_out()` returns false only when head has cr==0 AND result_buffer is occupied. After cycle-by-cycle traces (validated in re-audit), the deque cannot grow beyond `pipeline_stages_` IF the writeback arbiter eventually drains. But the bound is structural, not asserted. Recommend: add `assert(next_pipeline_.size() < pipeline_stages_)` in accept() so a future scheduler bug (e.g., issuing despite ready_out=false in a test override) is caught loudly. The first re-audit pass (Opus) initially flagged "unbounded growth under wb starvation" but the deeper trace (second Opus pass) showed it's bounded — this is the kind of subtle dynamic invariant that benefits from a defensive assert.
- Code citation: `sim/src/timing/multiply_unit.cpp:5-20`; `sim/include/gpu_sim/timing/multiply_unit.h:22-30`.
- Repro hint: static.
- Spec section: §4.6 (MUL pipelined; "accepts new op every cycle")
- Discovered: phase 4 row 4/7 re-audit (two Opus passes), 2026-04-28.

## Finding F-11: WritebackArbiter::pending_commit_ retains stale state across commit boundary
- Severity: Definite (minor; +1 cycle drain detection delay)
- Category: Counter inconsistency / cross-stage hazard
- Evidence: `WritebackArbiter::pending_commit_` is set in evaluate (`writeback_arbiter.cpp:42`), latched via commit (`pending_commit_` → `committed_` at line 51-53), but `pending_commit_` itself is cleared only at the **next** evaluate (`writeback_arbiter.cpp:19`) — NOT in commit. Implication: end-of-tick `pipeline_drained()` check (`timing_model.cpp:286`) reads `wb_arbiter_->has_pending_work()` which returns true if pending_commit_ has a value, even though the work has already been committed. Pipeline drain detection is delayed by 1 cycle. **More serious:** during panic, `wb_arbiter.evaluate` does not run (panic path at `timing_model.cpp:347-381`), so pending_commit_ from before panic is never cleared until the panic-flush cascade calls `wb_arbiter_->flush()→reset()` at line 539. The dependency on the panic flush for state hygiene is implicit and undocumented; if the flush is ever moved or removed, pending_commit_ permanently reports has_pending_work=true during panic, blocking drain (only saved by MAX_DRAIN_CYCLES timeout).
- Code citation: `sim/src/timing/writeback_arbiter.cpp:19,42,51-53`; `sim/src/timing/timing_model.cpp:286,347-381,539`.
- Repro hint: any benchmark — drain detects 1 cycle late but is benign on real workloads.
- Spec section: §4.7 (writeback arbiter); §4.8.1 (panic drain semantics)
- Discovered: phase 4 row 7 re-audit (Opus), 2026-04-28.

## Finding F-12: Stale documentation re: removed `current_port_claimed_` field (supersedes F-07)
- Severity: Definite (documentation drift)
- Category: Counter inconsistency / documentation
- Evidence: Two locations reference a "REGISTERED `next_port_claimed_` / `current_port_claimed_` pair" that no longer exists:
  - `sim/include/gpu_sim/timing/cache.h:175-176`
  - `resources/timing_discipline.md:184` (row 11)
  The consolidation pass (commit 54e6542) removed `current_port_claimed_`. The flag is now COMBINATIONAL same-cycle scratch (single slot, first-writer-wins, cleared at gather_file.commit). Naming convention `next_*` without a `current_*` mirror violates the next_/current_ discipline pattern. **Behavior is correct; the doc is the bug.**
- Code citation: `sim/include/gpu_sim/timing/cache.h:175-176`; `resources/timing_discipline.md:184`; `sim/src/timing/load_gather_buffer.cpp:74-77` (commit body).
- Repro hint: static.
- Spec section: §5.3 ("one line-to-gather-buffer extraction per cycle")
- Discovered: phase 4 row 11 re-audit (Opus), 2026-04-28. (Supersedes F-07.)

## Finding F-13: LoadGatherBuffer::busy is an unclassified cross-stage edge between WBArbiter and Coalescing
- Severity: Suspicious (Probable design fragility)
- Category: Cross-stage hazard
- Evidence: `WritebackArbiter::evaluate` (line 40) calls `consume_result()` on a unit/source. For LoadGatherBuffer sources, `consume_result()` directly mutates `buffers_[idx].busy=false` and clears slots (`load_gather_buffer.cpp:115-118`). `CoalescingUnit::evaluate` reads `gather_file_.is_busy(warp_id)` at line 28. Tick order (`timing_model.cpp`): coalescing.evaluate (line 494) runs BEFORE wb_arbiter.evaluate (line 500). So coalescing observes pre-consume `busy=true` this cycle, post-consume `busy=false` next cycle. **Functionally correct due to tick order, but this is an unclassified cross-stage edge** — neither REGISTERED, nor explicitly COMBINATIONAL, nor READY/STALL. The discipline-doc inventory has no row covering it. If an orchestrator reorders evaluate phases (currently impossible without code change), `is_busy()` would return false same-cycle as a fresh consume → coalescing might claim the buffer racing with the just-popped writeback. Defense-in-depth is missing.
- Code citation: `sim/src/timing/load_gather_buffer.cpp:115-118` (consume_result write); `sim/src/timing/coalescing_unit.cpp:28` (is_busy read); `sim/src/timing/timing_model.cpp:494,500` (tick order).
- Repro hint: static.
- Spec section: §5.2.1 (gather buffer per-warp serialization)
- Discovered: phase 4 row 11 re-audit (Opus), 2026-04-28. Recommendation: add an explicit inventory row (Row 16) for this edge, classify it formally (likely COMBINATIONAL with tick-order constraint).

## Finding F-14: PanicController::set_drained_query callable claim is documentation-inaccurate
- Severity: Probable (documentation drift; latent fragility)
- Category: Documentation / cross-stage hazard
- Evidence: `panic_controller.h:31-34` claims the callable wired by set_drained_query "is expected to read only committed (current_*) state." The actual callable composed at `timing_model.cpp:289-297` reads:
  - `opcoll_->ready_out()` — committed `current_busy_` ✓
  - `alu_->ready_out()` etc. — committed ✓
  - `alu_->has_result()` — reads `next_result_buffer_.valid` (`alu_unit.cpp:54-60`) ✗ NOT committed
  - `mul_->has_result()`, `div_->has_result()`, `tlookup_->has_result()` — all read next_* ✗
  - `ldst_->fifo_empty()` — reads live `next_addr_gen_fifo_` ✗
  - `wb_arbiter_->has_pending_work()` — iterates source.has_result(), all next_* ✗
  Functionally OK because, in panic-active ticks, the call site at `timing_model.cpp:353` runs BEFORE units evaluate (lines 355-359) — so units' next_* equals last commit's current_*. But the documented contract is wrong, and the correctness depends on panic-tick ordering that isn't called out.
- Code citation: `sim/include/gpu_sim/timing/panic_controller.h:31-34`; `sim/src/timing/timing_model.cpp:201,289-297,353-359`; `sim/src/timing/alu_unit.cpp:54-60` (and analogous in other units).
- Repro hint: static.
- Spec section: §4.8.1 (panic drain)
- Discovered: phase 4 row 14 re-audit (Opus), 2026-04-28.

## Finding F-15: Cache submit_read overflow defense is debug-only
- Severity: Suspicious (defense-in-depth gap)
- Category: Held-state leak / cross-stage hazard
- Evidence: `mem_if.submit_read()` returns `bool` (success). Cache **ignores** the return value (`cache.cpp:110,188`). The `submit_read` body has `assert` in debug builds (would fire on overflow) but `if (!has_capacity) return false` in release builds (silent fail). In release, an MSHR overflow would silently drop the read and leave the MSHR slot allocated forever (waiting for a response that will never arrive). The architectural invariant "num_mshrs * chunks_per_line ≤ in_flight capacity" prevents this in practice, but defense-in-depth is debug-only.
- Code citation: `sim/src/timing/cache.cpp:110,188` (caller ignores return); `sim/src/timing/dramsim3_memory.cpp:66,70` (assert + soft fail). Comment at cache.cpp:64 acknowledges the design choice.
- Repro hint: would require violating the architectural invariant; static analysis only.
- Spec section: §5.3.1 (MSHR), §5.6 (memory interface)
- Discovered: phase 4 row 15 re-audit (Opus), 2026-04-28.

## Finding F-16: instr_buffer is the only first-class warp-state structure not under next_/current_ discipline (4-site direct-mutation surface; 7b5f713-class regression risk)
- Severity: Probable (design-debt; structural risk for future regressions)
- Category: Cross-stage hazard / held-state leak
- Evidence: `WarpState::instr_buffer` (per-warp circular buffer of decoded instructions) is mutated directly mid-tick by 4 different stages with NO next_/current_ pair:
  - `FetchStage::evaluate` reads `instr_buffer.size()` / `is_full()` (eligibility gate, the 7b5f713 fix point) — at fetch_stage.cpp:54-66
  - `WarpScheduler::evaluate` calls `instr_buffer.pop()` (`warp_scheduler.cpp:154`) — direct mid-evaluate mutation
  - `DecodeStage::commit` calls `instr_buffer.push(...)` (`decode_stage.cpp:59`) — mid-commit-phase mutation
  - `FetchStage::commit::apply_redirect` calls `instr_buffer` flush (`fetch_stage.cpp:124`) — mid-commit-phase mutation
  Saved from 7b5f713-class regression today only by strict tick order:
    - fetch.evaluate (read pre-pop) → scheduler.evaluate (pop) → ... → decode.commit (push) → fetch.commit::apply_redirect (flush)
  If any of these orderings change OR a new mid-tick reader is added that sees the post-pop / post-push state, the 7b5f713 bug class returns. The 7b5f713 fix added `inflight_to_w` accounting in fetch's eligibility gate to compensate for this exact non-discipline. **The deeper fix would be to double-buffer `instr_buffer` (next_buffer / current_buffer with commit() flip), aligning it with the rest of the timing model's REGISTERED state.**
- Code citation: `sim/src/timing/warp_scheduler.cpp:154`; `sim/src/timing/decode_stage.cpp:59`; `sim/src/timing/fetch_stage.cpp:54-66,124`; `sim/include/gpu_sim/timing/warp_state.h` (struct definition).
- Repro hint: static; no current repro (existing tick-order discipline holds).
- Spec section: §4.2 (fetch eligibility); §4.3 (scheduler); structural — no spec text on buffer discipline.
- Discovered: phase 5 register re-audit (Opus), 2026-04-28. **This is the most structurally significant finding from re-audit.**

## Phase 3 Trace audit (programmatic, 6 fixed-default traces, ~2 GB)

Streaming Python analyzer over Chrome JSON traces. Key results:

| Benchmark | events | AT_REST no-reason / total | same-warp consec issues / total | orphan branch_redirects |
|-----------|--------|---------------------------|----------------------------------|------------------------|
| matmul | 4.6M | 0 / 560510 (0.0%) | 6372 / ~71536 (~9%) | 0 |
| gemv | 102k | 0 / 12520 (0.0%) | 830 / 3038 | 0 |
| fused_linear_activation | 45k | 0 / 6600 (0.0%) | 102 / 1216 | 0 |
| softmax_row | 41k | 0 / 1422 (0.0%) | 774 / 1544 | 0 |
| embedding_gather | 421k | 0 / 2189 (0.0%) | 521 / 936 | 0 |
| layernorm_lite | 159k | 0 / 19512 (0.0%) | 1301 / 7336 | 0 |

**Closed as non-findings (negative results worth recording):**
- **AT_REST always has populated rest_reason.** The hypothetical Phase-8 counter `rest_reason_unset_cycles` proposed in the plan is unnecessary; current `WarpRestReason` enum is exhaustive in practice. Closes a hypothetical observability gap.
- **branch_redirect events always preceded by an issue** for the same warp earlier in the trace. The redirect-tracking path is well-formed.

**Confirmed behavior (corroborates earlier findings):**
- F-06 (memory-bound saturation explains embedding_gather IPC stagnation): `active_mshrs.value: avg=3.89, max=4` on embedding_gather — pool is saturated.
- F-10 (MultiplyUnit pipeline structurally bounded at pipeline_stages_): `mul_occupancy.value: max=2` across all benches with default `pipeline_stages_=3`. Confirms the structural bound holds in practice.
- 7b5f713 fix in steady state: `fetch_skip_backpressure: 0` across all 42 Phase-1 stat tuples; the HOL stall is gone.

## Finding F-17: Same-warp consecutive scheduler issues are common (legitimate but worth flagging)
- Severity: Spec-Ambiguity
- Category: Resource arbitration / spec ambiguity
- Evidence: Trace shows same-warp consecutive issue events at 1-cycle granularity in all 6 benchmarks; ranging from 7% (fused_linear_activation) to 88% (matmul: 6372 / ~71536) of issue events. This happens when 7 of 8 warps are backend-stalled and round-robin scheduler lands on the same warp on consecutive cycles. Legitimate per spec §4.3 ("loose round-robin") but the spec wording implies fairness across warps. A reader could reasonably interpret "loose round-robin" as "no warp issues twice in a row when others are eligible" — which holds — but the spec doesn't forbid same-warp consecutive issue when others are NOT eligible. The implementation correctly does this.
- Code citation: `sim/src/timing/warp_scheduler.cpp:84-137` (scan-from-pointer); `sim/src/timing/warp_scheduler.cpp:140-145` (pointer-advance unconditional).
- Repro hint: any backend-saturated benchmark, especially matmul.
- Spec section: §4.3 ("loose round-robin")
- Discovered: phase 3 trace audit, 2026-04-28. Recommendation: spec should explicitly allow same-warp consecutive issue when others are ineligible (or document the actual scan order more precisely).

## Phase 4 / 5 audited rows — verdict summary (after Opus re-audit)

| Row | Subject | Verdict | New findings from re-audit |
|-----|---------|---------|---------------------------|
| 1 | DecodeStage::ready_to_consume_fetch | COMPLIANT (fragile) | F-08 |
| 2 | DecodeStage::pending_warp | COMPLIANT (post-7b5f713 fix verified) | none |
| 3 | OperandCollector::ready_out | COMPLIANT | none |
| 4 | Each ExecutionUnit::ready_out | COMPLIANT | F-09, F-10 |
| 5 | BranchShadowTracker | COMPLIANT | none |
| 6 | OperandCollector::accept | COMPLIANT | minor: pulse semantics noted |
| 7 | Unit accept/consume_result | COMPLIANT | F-11 |
| 8 | WBArbiter → Scoreboard | COMPLIANT | none |
| 9 | Cache↔Coalescing COMBINATIONAL stall | COMPLIANT | none |
| 10 | L1Cache external surface | COMPLIANT | none (stale-value carryover noted, gated by valid flag) |
| 11 | LoadGatherBufferFile port arbitration | COMPLIANT (behavior); doc bugs | F-12, F-13 |
| 12 | RedirectRequest cross-commit | COMPLIANT | wasted cycle of fetch work post-mispredict noted |
| 13 | EBreakRequest panic flush | COMPLIANT | none |
| 14 | PanicController drained query | COMPLIANT (behavior); doc inaccurate | F-14 |
| 15 | L1Cache↔mem_if carve-out | COMPLIANT | F-15 |
| Phase 5 | All held registers | COMPLIANT (modulo F-16 design-debt) | F-16 (most significant) |

## Phase 6 — Synthetic stress workloads (2026-04-28)

Six synthetic kernels (5 mandatory + 1 optional) authored under
`/workspace/tests/synthetic/<kernel>/`, mirroring the existing benchmark
layout (kernel_bench.cpp + .S + link.ld + CMakeLists.txt). Each kernel
targets one ambiguous spec area; analytical cycle expectations derived
from the spec are encoded in the bench's match checks (self-asserting).
All six register `add_test` so they run under `ctest` (27/27 pass).

| Kernel | Spec target | Pass criterion | Result |
|--------|-------------|----------------|--------|
| `rr_tiebreak` | §4.2/§4.3 RR tie-break | max-min per-warp issue count == 0 | PASS (W0..W7 all 324 issues with iters=64; 0 consecutive same-warp issues per trace) |
| `line_boundary_load` | §5.2 boundary detection | coalesced_requests=1 + serialized_requests=1 (per-warp counter semantics) | PASS — but surfaces F-18 (counter ↔ doc mismatch) |
| `mshr_same_line_race` | §5.3.1 same-line merging + chain order | 1 primary + N-1 secondaries, 1 external read, allocation order = primary first | PASS (allocs at ts=18,22,26,30 — exactly 4 cyc apart, matches RR cadence) |
| `jalr_storm` | §4.2 static-mispredict + redirect | branch_mispredictions = N_warps × (iter+1); redirects all preceded by issue | PASS — but surfaces F-19 (spec wording vs impl semantics) |
| `divide_chain` | §4.6 32-cyc DIV + RV32M div-by-zero | div_busy_cycles = 32 × instr; divzero result = -1 | PASS (counter); FAIL trace slice (F-20: trace duration = 31 cyc systematically) |
| `panic_drain_test` (optional) | §4.8.1 drain bound + "in-flight" | panic_active span ≤ MAX_DRAIN_CYCLES + 2 latch/halt overhead; r31 → PANIC_CAUSE; DIV busy_cycles = 32 | PASS (panic span = 32, panic_cause = 0x101, DIV completes during drain) |

Also confirmed by these kernels:
- §4.2 mispredict-recovery cycle accounting: per-warp penalty ≈ 1 redirect-register cycle + 2 refill cycles, fully hidden at N≥4 warps (jalr_storm IPC: 1-warp=0.50, 4-warp=0.995).
- §5.2 single-line coalescing: aligned-base load → 1 coalesced cache request; aligned+64 base → 32 per-lane cache lookups (cache_misses=33 = 1+16+16, hits=24 = 12+12 from line-fill cascade).
- §5.3.1 chain ordering by allocation time: primary first, secondaries in arrival order.
- §4.6 single-occupancy DIV unit: 18 DIV instructions across 2 warps with chain dependency → 18 non-overlapping 31-cycle trace slices, busy_cycles = 576 = 18 × 32 (exact).
- §4.7 scoreboard 1-cycle release gap: divide_chain cycles 617 ≈ 18×32 + setup/drain — chain serialization implies release works.

## Finding F-18: `serialized_requests` increments per-warp-decision, not per-lane (counter ↔ doc mismatch)
- Severity: Definite
- Category: Counter inconsistency
- Evidence: `resources/trace_and_perf_counters.md:415` documents `serialized_requests` as "Memory requests that serialized into 32 per-lane accesses (**totals per lane, not per warp**)". Implementation increments the counter exactly once per warp's straddle decision: `sim/src/timing/coalescing_unit.cpp:50` is reached once when `is_coalesced_=false` is decided; the per-lane loop at lines 86–110 does not touch the counter. Confirmed empirically by `line_boundary_load`: a single warp issuing one straddle load reports `serialized_requests=1`, but `cache_misses=9` (1 aligned + 8 lane-misses on the two split lines pre-fill) and `cache_hits=24` (lanes 4..15 + 20..31 hitting after fills land), giving 33 total cache lookups attributable to the straddle = 32 lane requests + 1 coalesced. The per-lane requests *do* reach the cache; the counter just doesn't track them.
- Code citation: `sim/src/timing/coalescing_unit.cpp:46-51` (one increment per coalesced/serialized decision); `resources/trace_and_perf_counters.md:415` (doc claim).
- Repro hint: `tests/synthetic/line_boundary_load/line_boundary_load_bench --json` — `serialized_requests=1` regardless of lane count.
- Spec section: §5.2 ("falls back to 32 serialized individual requests") — the spec wording also implies a per-lane semantic, so this is doc + spec vs impl.
- Discovered: phase 6, 2026-04-28. Disposition: rename to `serialized_decisions` to match impl, OR change impl to `stats_.serialized_requests += WARP_SIZE` (32 per straddle) to match the doc/spec wording. Affects calibration of `coalesced/serialized` ratio — the current ratio dramatically under-states cache pressure from straddling loads.

## Finding F-19: Spec §4.2 says JALR "always mispredicted"; impl conditionally mispredicts only when actual ≠ PC+4
- Severity: Spec-Ambiguity
- Category: Spec ambiguity
- Quoted text: §4.2: "JALR: predicted as fall-through to PC+4 (register-indirect target is unknown at fetch time; **always mispredicted**, resolved at execute with full refill penalty)".
- Implementation pinning: `sim/src/timing/branch_predictor.cpp:24-26` — `case JALR: prediction.is_control_flow = true; break;` leaves `predicted_taken = false` (default), so `predicted_target = 0` is unused. `sim/src/timing/timing_model.cpp:322-335` `branch_mispredicted()` computes `predicted_next_pc = predicted_taken ? predicted_target : pc + 4`, so for JALR `predicted_next_pc = pc + 4`. If a JALR's actual target equals `pc + 4`, `predicted_next_pc == actual_next_pc` → returns `false` → no mispredict counted, no flush.
- Implementation behavior: A JALR whose register-indirect target happens to equal `pc + 4` is treated as a CORRECT prediction. In the synthetic `jalr_storm` (target ≠ PC+4 by construction), every JALR is mispredicted as the spec implies; for a JALR-to-PC+4 (legal but unusual), the impl would report 0 mispredicts. The spec wording "always mispredicted" therefore overstates the impl by one corner case.
- Suggested clarification target: §4.2 sentence "JALR: predicted as fall-through to PC+4...always mispredicted". Either (a) replace "always mispredicted" with "mispredicted whenever the register-indirect target ≠ PC+4 (i.e., whenever the static fall-through prediction is wrong)", or (b) change the impl to unconditionally treat JALR as mispredicted (a simple `if (type == JALR) return true;` short-circuit in `branch_mispredicted`). Either pins the contract.
- Discovered: phase 6, 2026-04-28.

## Finding F-20: `EXECUTE_DIV` trace slice duration is systematically 1 cycle short of `div_busy_cycles` counter
- Severity: Definite
- Category: Trace anomaly / counter inconsistency
- Evidence: `divide_chain --num-warps=2 --iterations=8` produces 18 DIV instructions, `div_busy_cycles = 576` (= 18 × 32, exactly matches spec §4.6 "32 cycles" latency). The corresponding 18 trace `execute_div` complete events ALL have `dur=31` (verified by histogram in `/tmp/phase6_trace_audit.py`). Mechanism: `DivideUnit::evaluate()` (`sim/src/timing/divide_unit.cpp:20-32`) increments `div_stats.busy_cycles` 32 times per DIV and clears `next_busy_` on the 32nd evaluate. `commit()` (line 34) flips `current_busy_` from true→false on that 32nd cycle. The trace samples `pending_entry()` (`sim/include/gpu_sim/timing/divide_unit.h:30-32`), which reads `current_busy_`; this is true at end-of-tick for cycles 0..30 (31 ticks) and false at end-of-tick 31. Net: `busy_cycles=32`, trace slice spans 31 ticks. The two observability surfaces disagree by exactly 1 cycle per DIV. Same pattern likely applies to `execute_mul` (3-stage pipeline → trace would show 2-cycle slices?) and `execute_tlookup` — not verified in this phase.
- Code citation: `sim/src/timing/divide_unit.cpp:20-32` (counter increment site); `sim/include/gpu_sim/timing/divide_unit.h:30-32` (`pending_entry()` reading `current_busy_`); `sim/src/timing/timing_model.cpp:670-673` (trace state set from `pending_entry()`).
- Repro hint: `tests/synthetic/divide_chain/divide_chain_bench --trace-file=/tmp/d.json --json`; `python3 -c "import json,collections; d=json.load(open('/tmp/d.json')); print(collections.Counter(int(e['dur']) for e in d['traceEvents'] if e.get('name')=='execute_div'))"` → `{31: 18}` while stats show `div_busy_cycles=576`.
- Spec section: §4.6 ("Latency: 32 cycles"). The spec doesn't explicitly pin whether the 32nd cycle counts as "busy" or "result-being-delivered" — the impl's two surfaces interpret it differently. Recommended: pick one convention and align both surfaces.
- Discovered: phase 6, 2026-04-28. Disposition: either (a) defer the `current_busy_=false` commit by one cycle so the trace covers 32 cycles, or (b) accept the convention that the 32nd cycle is "writeback-ready" and decrement `busy_cycles` to match. Operator-facing impact: anyone deriving per-DIV latency from trace slice widths will under-count by 1 cycle.

## Phase 6 — clean (positive verifications, no findings):

- **rr_tiebreak**: spec §4.2/§4.3 round-robin tie-break behaves as documented at every supported warp count. Per-warp instruction count imbalance = 0 at warps ∈ {2, 4, 8} with iters=256/64. 0 consecutive same-warp issue events under perfect symmetry. IPC saturates at ~0.998–0.999, matching the steady-state per-cycle issue rate. Confirms scheduler is deterministic on identical streams (no hidden bias).
- **mshr_same_line_race**: spec §5.3.1 same-line MSHR merging works correctly. 4 warps loading the same line → 1 primary alloc + 3 secondaries, all on one line, allocations strictly ordered by arrival cycle. 1 external read (vs 4 unmerged), 3 secondary_drain events. No line_pin_stall, no MSHR_FULL stall under default config.
- **jalr_storm**: spec §4.2 static-mispredict path correct. Counter tally exactly matches analytical `2N predictions, N+1 mispredicts, N+1 flushes` per warp (incl. final BNEZ mispredict). Per-mispredict frontend cost = 2.13 cycles (= 68 frontend stalls / 32 mispredicts) — close to the spec §4.2 formula "1 redirect + ≥2 refill", with the refill latency at the implementable minimum. With 4 warps: scheduler_idle drops from 101 to 2 cycles → mispredict latency fully hidden by warp-switching, confirming the spec's "Other warps continue executing during this time, hiding the penalty" claim.
- **divide_chain**: spec §4.6 single-slot iterative DIV behaves correctly. `div_busy_cycles = 32 × instructions` exactly (counter side); 18 DIVs from 2 warps fully serialize through the unit (no overlapping `execute_div` slices in trace). Div-by-zero produces -1 per RV32M (verified for all 32 lanes via per-lane store + memory readback).
- **panic_drain_test** (optional 6th kernel): spec §4.8.1 panic sequence behaves correctly under the worst-case in-flight DIV scenario. Single warp issues a 32-cyc DIV that is mid-execute when EBREAK reaches decode; the panic controller's drain step waits for the DIV to retire. Trace confirms steps 1-4 take 3 cycles (decode-detect at cycle 7, latch at cycle 8 via panic_trigger event, drain step 2 begins cycle 9), drain runs cycle 9-37 (29 ticks waiting for DIV to complete), step 3 halt at cycle 38, total panic-active span = 32 cycles — within `MAX_DRAIN_CYCLES + 2` overhead bound. `PANIC_CAUSE` correctly latches `r31 = 0x101` (verifies the spec §4.8.3 r31-read path during step 1). `div_busy_cycles = 32` (DIV completes normally during drain — drain neither truncates nor extends iteratively-executing units, matching spec §4.8.1: "drain covers execution units and writeback only"). The "in-flight external memory requests are abandoned" clause is not exercised by this kernel (no loads); a future variant could probe that branch but the spec wording on it is already precise.

## Session log

### 2026-04-27 session 1 — Phase 0 + Phase 1 + Phase 2 + Phase 4 (5 high-leverage rows) + Phase 5 (fetch.current_output_)

- Phase 0 bootstrap complete; build clean (sha 7b5f713).
- Phase 1 complete for 7 viable configs × 6 benches = 42 tuples (fixed-warps-16 dropped per F-01: MAX_WARPS=8 build cap).
- Total Phase 1 wall: ~1.5 sec. Atom mean wall ~30 ms.
- Phase 2 counter audit: 5 findings (F-01..F-05).

### 2026-04-28 session 1 (continued) — Phase 3 trace audit + Phase 4/5 Opus re-audit + observability instrumentation

After user noted that Explore sub-agents default to Haiku (which returned uniform COMPLIANT verdicts on Phase 4/5), re-dispatched the same audits as `general-purpose` agents with `model: opus` explicit. Substantively richer results:

- Phase 4 rows 1-7 (Opus re-audit): F-08 (decode pending_ single-buffered fragility), F-09 (ALU dead state), F-10 (MUL pipeline bound assertion missing), F-11 (WBArbiter pending_commit_ stale-state).
- Phase 4 rows 8-15 (Opus re-audit): F-12 (cache.h:175-176 + discipline-doc stale ref to removed current_port_claimed_; supersedes F-07), F-13 (gather_buffer.busy unclassified cross-stage edge), F-14 (PanicController docs inaccurate), F-15 (submit_read overflow defense debug-only).
- Phase 5 held registers (Opus re-audit): F-16 (instr_buffer is the only first-class warp-state structure NOT under next_/current_ discipline; 4-site mutation surface; 7b5f713-class regression risk). Most structurally significant finding from session.

Plus observability instrumentation:
- Added `--trace-file=<path>` CLI flag to all 6 bench binaries (matmul, gemv, fused_linear_activation, softmax_row, embedding_gather, layernorm_lite). Wires to existing `TimingTraceOptions` 4th constructor arg. Built clean; ctest 21/21 pass.
- Generated 6 Phase-1 traces (~415 MB total).
- Programmatic Phase 3 audit: F-17 (same-warp consecutive issue legitimacy) + 2 negative results (AT_REST always has reason; branch_redirect tracking clean).

Phase 4/5/6/7 rows audited (depth-first):
- Phase 4: rows 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 — ALL audited (Haiku then Opus re-audit). 8 substantive findings.
- Phase 5: all listed registers (fetch.current_output_, decode.pending_, opcoll, exec units, gather buffer, ldst FIFO, mul pipeline, branch tracker, mshr file, scheduler eligibility) — audited. 1 findings (F-16).
- Phase 6 (synthetic kernels): COMPLETE. Note: this entry originally read "DEFERRED to user triage; not pursued" — that statement is stale. The 5 mandatory kernels were authored later that same day (2026-04-28), and the optional 6th (`panic_drain_test`) was added during the 2026-04-28 verification pass. See the "Phase 6 — Synthetic stress workloads" section above for the full results table and the per-kernel pass-criterion analysis. Findings: F-18, F-19, F-20.
- Phase 7 (spec walk): partially via Haiku (15 ambiguities listed but lower-confidence quality). Not re-run with Opus due to lower bug-yield expected vs. code audit. F-17 comes from this domain (trace-validated).

Phases 8 (instrumentation) and 9 (consolidation) folded into this session (added --trace-file plumbing; this session log IS the consolidation).

**Total findings:** F-01 (Observability-Gap), F-02 (Definite, dead counter), F-03 (Probable/Definitional, stall partition), F-05 (Definite, counter/spec semantic mismatch), F-06 (Spec-Ambiguity, supersedes F-04), F-07 (subsumed into F-12), F-08–F-16 (re-audit), F-17 (Spec-Ambiguity from trace). **15 substantive findings + 2 closed observations + 5 audited "Suspicious-but-not-conclusive" design-debt items in F-16's neighborhood.**

**Files instrumented (changes uncommitted):** the 6 `*_bench.cpp` files. Build clean. Tests green.
- Phase 4 high-leverage rows audited via Explore sub-agents:
  - **Row 7 (execution units, accept/consume_result/has_result/ready_out/commit):** COMPLIANT for all 5 units + WBArbiter. No findings. The largest concurrent-write surface checks out.
  - **Row 9 (cache↔coalescing COMBINATIONAL stall, Phase 9 carve-out):** COMPLIANT. `is_stalled()` is const, scratch field reset at top of cache.evaluate, written by mid-evaluate paths (process_load/process_store/complete_fill), read same-tick by coalescing.evaluate. Behavioral gate at coalescing_unit.cpp:20.
  - **Row 11 (LoadGatherBufferFile::next_port_claimed_, 3 writers, FILL>secondary>HIT priority via tick order):** COMPLIANT. One minor finding: F-07 (stale doc comment, see below).
  - **Row 5 (BranchShadowTracker, 3 writers, Scoreboard-shape REGISTERED):** COMPLIANT. Mispredict cycle traced; defer-on-mispredict logic verified with the explanatory comment at operand_collector.cpp:61-66. No legacy `branch_in_flight =` direct writes remain.
  - **Row 12 (RedirectRequest cross-commit ordering):** COMPLIANT. Critical commit-order invariant (fetch/decode commit BEFORE opcoll commit) maintained at timing_model.cpp:502-505. Both readers safely no-op on warp_id mismatch. Panic+redirect interaction handled (panic dominates).
- Phase 5 fetch.current_output_ register: FIX (7b5f713) IS COMPLETE. The eligibility gate at fetch_stage.cpp:54-66 correctly counts `decode.pending_warp == w` AND `current_output.warp_id == w` toward the in-flight reservation for warp w. Decode's push gate is conservative but safe (upstream gate has reserved the slot). Branch redirect properly invalidates the held output. No other consumer omits the register.
- Atom count: ~9 reasoning atoms (cap was 8; user requested aggressive use of remaining 5h window).

