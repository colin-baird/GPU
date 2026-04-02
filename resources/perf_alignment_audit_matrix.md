# Performance Alignment Audit Matrix

This matrix records the current spec-to-model alignment for performance-affecting rules.

Status meanings:

- `aligned`: simulator behavior matches the architectural contract and is validated
- `underspecified`: behavior exists, but the architectural contract intentionally does not freeze it yet
- `untested`: intended behavior is described, but not yet validated by an explicit check
- `drift`: simulator behavior conflicts with the current architectural contract

| Rule | Spec Reference | Simulator Owner | Observable Surface | Validation | Status |
|------|----------------|-----------------|--------------------|------------|--------|
| 1-warp pipeline fill before first issue | §4.2 | `fetch_stage`, `decode_stage`, `warp_scheduler` | `total_cycles`, `scheduler_idle_cycles` | `simple_pipeline` manifest | aligned |
| ALU RAW visibility after committed writeback | §4.3, §4.7, §8 | `warp_scheduler`, `writeback_arbiter`, `scoreboard` | `warp_stall_scoreboard`, final registers | `alu_chain` manifest | aligned |
| Multiply latency and scoreboard release timing | §4.6, §4.7 | `multiply_unit`, `writeback_arbiter`, `scoreboard` | `mul_instructions`, cycle snapshots, final registers | `mul_dependency` manifest | aligned |
| Divide latency and scoreboard release timing | §4.6, §4.7 | `divide_unit`, `writeback_arbiter`, `scoreboard` | `div_instructions`, `warp_stall_scoreboard`, cycle snapshots | `div_dependency` manifest | aligned |
| Taken conditional branch flushes the warp buffer | §4.2 | `timing_model`, `fetch_stage`, `decode_stage` | `branch_flushes`, final registers | `branch_taken` manifest | aligned |
| JAL/JALR redirect at execute-stage resolution | §4.2 | `timing_model`, `fetch_stage`, `decode_stage` | `branch_flushes`, final registers | `jal_redirect` manifest | aligned |
| Stall-on-use for load miss | §5.3, §5.4 | `ldst_unit`, `coalescing_unit`, `cache`, `scoreboard` | `load_misses`, `warp_stall_scoreboard`, cycle snapshots | `load_miss_use` manifest | aligned |
| No duplicate miss merging for same cache line | §5.3.1 | `cache`, `mshr`, `coalescing_unit` | `cache_misses`, `external_memory_reads`, `active_mshrs` | `store_then_load_same_line` manifest | aligned |
| Write-through drain delays completion | §5.3.2, §5.4 | `cache`, `memory_interface`, `timing_model` | `external_memory_writes`, `total_cycles` | `store_then_load_same_line` manifest | aligned |
| Round-robin writeback arbitration delays scoreboard clear | §4.7, §8 | `writeback_arbiter`, `scoreboard` | `writeback_conflicts`, `total_cycles` | `writeback_conflict` manifest | aligned |
| Panic sequencing and diagnostic latch | §4.8 | `panic_controller`, `timing_model` | cycle snapshots, `panic.*` fields | `panic_drain` manifest | aligned |
| All-or-nothing coalescing fallback serializes divergent lanes | §5.2 | `coalescing_unit`, `cache` | `serialized_requests`, `external_memory_reads`, final registers | `serialized_load` manifest | aligned |
| 4-warp strict round-robin fetch / loose RR issue baseline | §4.2, §4.3 | `fetch_stage`, `warp_scheduler` | `warp_instructions`, `total_cycles` | `four_warp_round_robin` manifest | aligned |
| MSHR exhaustion backpressures additional misses | §5.3.1 | `cache`, `mshr`, `coalescing_unit` | `mshr_stall_cycles`, `external_memory_reads`, final registers | `four_warp_mshr_pressure` manifest | aligned |
| Write-buffer-full backpressure | §5.3.2 | `cache`, `coalescing_unit` | `write_buffer_stall_cycles` | `test_cache.cpp`, `test_integration.cpp` | aligned |
| DONE waits for memory/writeback drain | §4.8, §5.3.2 | `timing_model`, `cache`, `writeback_arbiter` | `total_cycles`, completion behavior | `store_then_load_same_line` manifest, `test_integration.cpp` | aligned |
| Branch prediction counters | not architecturally frozen | `branch_predictor`, `timing_model` | `branch_predictions`, `branch_mispredictions` | unit tests only | underspecified |
| Fetch skip counter | not architecturally frozen | `fetch_stage` | `fetch_skip_count` | unit tests only | underspecified |
