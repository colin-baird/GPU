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

    // Lookup table
    uint32_t lookup_table_entries = 1024;

    // External memory
    uint32_t external_memory_latency_cycles = 100;
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

    // Kernel arguments
    uint32_t kernel_args[4] = {0, 0, 0, 0};
    uint32_t start_pc = 0;

    // Simulation options
    bool trace_enabled = false;
    bool functional_only = false;

    void validate() const;

    static SimConfig from_json(const std::string& path);
    void apply_cli_overrides(int argc, char* argv[]);
};

} // namespace gpu_sim
