#pragma once

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
    uint32_t instruction_buffer_depth = 2;
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
