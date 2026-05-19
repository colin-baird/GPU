#pragma once

#include "gpu_sim/types.h"
#include <cstdint>
#include <string>
#include <stdexcept>

namespace gpu_sim {

struct SimConfig {
    // Warp configuration
    uint32_t num_warps = 4;

    // Instruction memory
    uint32_t instruction_mem_size_bytes = 8192;  // 8 KB = 2048 instructions

    // Pipeline
    uint32_t instruction_buffer_depth = 3;
    uint32_t multiply_pipeline_stages = 3;

    // Load/Store
    uint32_t num_ldst_units = 8;
    uint32_t addr_gen_fifo_depth = 4;

    // Cache
    uint32_t l1_cache_size_bytes = 4096;
    uint32_t cache_line_size_bytes = 128;
    uint32_t num_mshrs = 4;
    uint32_t write_buffer_depth = 4;
    // Global ceiling on enqueued-but-unacked write-throughs. A write-through
    // pins its cache set from the moment it is queued until its external
    // write ack is received; this caps how many such writes can be in flight
    // at once (enqueue backpressure). Must be >= 1 (a cap of 0 deadlocks all
    // stores) and should be >= write_buffer_depth (otherwise part of the
    // write buffer is structurally unreachable). The default is sized well
    // above write_buffer_depth and the external-memory latency window so it
    // does not bottleneck store throughput.
    uint32_t max_outstanding_writes = 32;

    // Lookup table
    uint32_t lookup_table_entries = 1024;

    // External memory
    // Default chosen to match the weighted-mean DRAMSim3 read latency across
    // the workload benchmark suite (DDR3-800 DE-10 Nano config), so the fixed
    // backend approximates the DDR3 backend's average per-request cost. Sweep
    // the suite under both backends and recompute if the suite or DDR3 config
    // changes materially.
    uint32_t external_memory_latency_cycles = 17;
    uint32_t external_memory_size_bytes = 64 * 1024 * 1024;  // 64 MB

    // External-memory backend selection.
    //   "fixed"    — FixedLatencyMemory (default; used by all unit tests)
    //   "dramsim3" — DRAMSim3Memory backed by the DE-10 Nano DDR3 model
    std::string memory_backend = "fixed";
    std::string dramsim3_config_path = "";
    std::string dramsim3_output_dir = "/tmp/dramsim3";
    double      fpga_clock_mhz = 150.0;
    double      dram_clock_mhz = 400.0;             // DDR3-800 I/O = 400 MHz
    // Sized to exactly absorb the cache's worst-case simultaneous in-flight
    // submits: (num_mshrs + write_buffer_depth) chunks_per_line. validate()
    // rejects any value below this minimum. The default tracks the simulator
    // defaults (4 MSHRs + 4-deep write buffer + 4 chunks/line = 32). Sizing
    // larger than the minimum is wasteful — the cache cannot produce more
    // outstanding chunks than the architectural bound.
    uint32_t    dramsim3_request_fifo_depth = 32;
    uint32_t    dramsim3_bytes_per_burst = 32;      // BL8 x 32-bit = 32 B
    // Synthetic write-commit latency, in DRAM cycles (tCK): how long after a
    // write-through is issued to DRAM its write ack is released to the cache
    // (releasing the write-ack pin). DRAMSim3's own write callback models a
    // posted write at submit+1 and folds same-line writes, so it is neither
    // 1:1 with write-throughs nor durability-faithful; DRAMSim3Memory
    // synthesizes the ack instead. The default ~= CWL + 3*tCCD_S + BL/2 + tWR
    // for a 4-chunk line under the DDR3-800 DE-10 Nano timings. Device-anchored
    // (tCK), so it is stable under fabric/DRAM clock-ratio changes.
    uint32_t    dramsim3_write_commit_latency_tck = 30;

    // Kernel arguments — loaded into x1..x6 of every lane at kernel launch.
    uint32_t kernel_args[6] = {0, 0, 0, 0, 0, 0};
    uint32_t start_pc = 0;

    // Simulation options
    bool trace_enabled = false;
    bool functional_only = false;

    void validate() const;

    static SimConfig from_json(const std::string& path);
    void apply_cli_overrides(int argc, char* argv[]);
};

} // namespace gpu_sim
