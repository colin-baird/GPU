#pragma once

#include "gpu_sim/types.h"
#include <cstdint>
#include <iostream>
#include <array>
#include <string>

namespace gpu_sim {

struct Stats {
    // Global
    uint64_t total_cycles = 0;
    uint64_t total_instructions_issued = 0;

    // Per-warp
    std::array<uint64_t, MAX_WARPS> warp_instructions{};
    std::array<uint64_t, MAX_WARPS> warp_cycles_active{};
    std::array<uint64_t, MAX_WARPS> warp_stall_scoreboard{};
    std::array<uint64_t, MAX_WARPS> warp_stall_buffer_empty{};
    std::array<uint64_t, MAX_WARPS> warp_stall_branch_shadow{};
    std::array<uint64_t, MAX_WARPS> warp_stall_unit_busy{};

    // Pipeline
    uint64_t fetch_skip_count = 0;
    uint64_t fetch_skip_backpressure = 0;   // fetch stalled: decode hasn't consumed previous output
    uint64_t fetch_skip_all_full = 0;        // fetch stalled: all warp instruction buffers at capacity
    uint64_t scheduler_idle_cycles = 0;
    uint64_t scheduler_frontend_stall_cycles = 0;  // idle: ≥1 active warp has empty buffer
    uint64_t scheduler_stall_backend_cycles = 0;   // idle: all active warps have instructions but can't issue

    // Phase 10B.0 issue-scoreboard stall reasons. These give finer-grained
    // breakdowns of the warp_stall_unit_busy[] family above. Indexed by the
    // six real ExecUnit values (ALU..SYSTEM); the SYSTEM slot stays zero.
    //   - unit_busy: issue blocked because a non-pipelined unit (DIVIDE /
    //     TLOOKUP) is still occupied by an iterative op (unit_busy_[u] > 0).
    //   - writeback_contention: issue blocked because the predicted writeback
    //     cycle is already claimed in the writeback bitmap (the scheduler
    //     proactively avoided a fixed-vs-fixed writeback collision).
    //   - ldst_fifo_full: an LDST issue blocked by FIFO-occupancy accounting.
    std::array<uint64_t, 6> scheduler_unit_busy_stall_cycles{};
    std::array<uint64_t, 6> scheduler_writeback_contention_stall_cycles{};
    uint64_t scheduler_ldst_fifo_full_stall_cycles = 0;
    uint64_t operand_collector_busy_cycles = 0;
    uint64_t branch_predictions = 0;
    uint64_t branch_mispredictions = 0;
    uint64_t branch_flushes = 0;

    // Per execution unit
    struct UnitStats {
        uint64_t busy_cycles = 0;
        uint64_t instructions = 0;
    };
    UnitStats alu_stats;
    UnitStats mul_stats;
    UnitStats div_stats;
    UnitStats ldst_stats;
    UnitStats tlookup_stats;

    // Memory system
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t load_hits = 0;
    uint64_t load_misses = 0;
    uint64_t store_hits = 0;
    uint64_t store_misses = 0;
    uint64_t mshr_stall_cycles = 0;
    uint64_t write_buffer_stall_cycles = 0;
    uint64_t coalesced_requests = 0;
    uint64_t serialized_requests = 0;
    uint64_t external_memory_reads = 0;
    uint64_t external_memory_writes = 0;
    uint64_t external_read_latency_total = 0;   // sum of submit→response cycles for completed external reads
    uint64_t external_read_latency_count = 0;   // number of external reads contributing to the total
    uint64_t total_load_latency = 0;
    uint64_t total_loads_completed = 0;
    uint64_t gather_buffer_stall_cycles = 0;
    uint64_t gather_buffer_port_conflict_cycles = 0;
    uint64_t mshr_merged_loads = 0;
    uint64_t mshr_merged_stores = 0;
    uint64_t line_pin_stall_cycles = 0;
    uint64_t secondary_drain_cycles = 0;

    // Writeback. Phase 10B.3: fixed_writeback_preempted_cycles counts cycles a
    // fixed-latency writeback was held off because a variable-latency load
    // took the port (equivalently, writeback-stall cycles). It is the semantic
    // successor of the removed writeback_conflicts counter — under fixed-
    // priority arbitration there is no round-robin conflict to count.
    uint64_t fixed_writeback_preempted_cycles = 0;

    void report(std::ostream& out, uint32_t num_warps) const;
    void report_json(std::ostream& out, uint32_t num_warps) const;
};

} // namespace gpu_sim
