#include "gpu_sim/config.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace gpu_sim {

static bool is_power_of_two(uint32_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

void SimConfig::validate() const {
    if (num_warps < 1 || num_warps > 8) {
        throw std::invalid_argument("num_warps must be in [1, 8]");
    }
    if (instruction_mem_size_bytes == 0 || instruction_mem_size_bytes % 4 != 0) {
        throw std::invalid_argument("instruction_mem_size_bytes must be a positive multiple of 4");
    }
    if (instruction_buffer_depth < 1) {
        throw std::invalid_argument("instruction_buffer_depth must be >= 1");
    }
    if (multiply_pipeline_stages < 1) {
        throw std::invalid_argument("multiply_pipeline_stages must be >= 1");
    }
    if (num_ldst_units < 1 || num_ldst_units > 32) {
        throw std::invalid_argument("num_ldst_units must be in [1, 32]");
    }
    if (addr_gen_fifo_depth < 1) {
        throw std::invalid_argument("addr_gen_fifo_depth must be >= 1");
    }
    if (!is_power_of_two(l1_cache_size_bytes) || l1_cache_size_bytes < cache_line_size_bytes) {
        throw std::invalid_argument("l1_cache_size_bytes must be a power of 2 >= cache_line_size_bytes");
    }
    if (cache_line_size_bytes != 128) {
        throw std::invalid_argument("cache_line_size_bytes must be 128");
    }
    if (num_mshrs < 1) {
        throw std::invalid_argument("num_mshrs must be >= 1");
    }
    if (write_buffer_depth < 1) {
        throw std::invalid_argument("write_buffer_depth must be >= 1");
    }
    if (lookup_table_entries < 1) {
        throw std::invalid_argument("lookup_table_entries must be >= 1");
    }
    if (external_memory_latency_cycles < 1) {
        throw std::invalid_argument("external_memory_latency_cycles must be >= 1");
    }
    if (external_memory_size_bytes < 1024) {
        throw std::invalid_argument("external_memory_size_bytes must be >= 1024");
    }
}

// Minimal JSON parser - handles flat key-value pairs for config
// For production use, replace with nlohmann/json
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

SimConfig SimConfig::from_json(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    SimConfig config;
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '{' || line[0] == '}' || line[0] == '/') continue;

        // Find "key": value
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;

        std::string key = trim(line.substr(0, colon));
        std::string val = trim(line.substr(colon + 1));

        // Remove quotes from key
        if (key.size() >= 2 && key.front() == '"' && key.back() == '"') {
            key = key.substr(1, key.size() - 2);
        }
        // Remove trailing comma from value
        if (!val.empty() && val.back() == ',') val.pop_back();
        val = trim(val);
        // Remove quotes from value if present
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
            val = val.substr(1, val.size() - 2);
        }

        // Remove "true"/"false" handling
        bool bool_val = (val == "true");

        try {
            uint32_t num = static_cast<uint32_t>(std::stoul(val));
            if (key == "num_warps") config.num_warps = num;
            else if (key == "instruction_mem_size_bytes") config.instruction_mem_size_bytes = num;
            else if (key == "instruction_buffer_depth") config.instruction_buffer_depth = num;
            else if (key == "multiply_pipeline_stages") config.multiply_pipeline_stages = num;
            else if (key == "num_ldst_units") config.num_ldst_units = num;
            else if (key == "addr_gen_fifo_depth") config.addr_gen_fifo_depth = num;
            else if (key == "l1_cache_size_bytes") config.l1_cache_size_bytes = num;
            else if (key == "cache_line_size_bytes") config.cache_line_size_bytes = num;
            else if (key == "num_mshrs") config.num_mshrs = num;
            else if (key == "write_buffer_depth") config.write_buffer_depth = num;
            else if (key == "lookup_table_entries") config.lookup_table_entries = num;
            else if (key == "external_memory_latency_cycles") config.external_memory_latency_cycles = num;
            else if (key == "external_memory_size_bytes") config.external_memory_size_bytes = num;
            else if (key == "start_pc") config.start_pc = num;
            else if (key == "arg0") config.kernel_args[0] = num;
            else if (key == "arg1") config.kernel_args[1] = num;
            else if (key == "arg2") config.kernel_args[2] = num;
            else if (key == "arg3") config.kernel_args[3] = num;
        } catch (...) {
            // Try boolean fields
            if (key == "trace_enabled") config.trace_enabled = bool_val;
            else if (key == "functional_only") config.functional_only = bool_val;
        }
    }

    return config;
}

void SimConfig::apply_cli_overrides(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.substr(0, 2) != "--") continue;

        auto eq = arg.find('=');
        if (eq == std::string::npos) {
            // Boolean flags
            if (arg == "--trace") trace_enabled = true;
            else if (arg == "--trace-text") trace_enabled = true;
            else if (arg == "--functional-only") functional_only = true;
            continue;
        }

        std::string key = arg.substr(2, eq - 2);
        std::string val = arg.substr(eq + 1);

        // Replace hyphens with underscores for matching
        std::replace(key.begin(), key.end(), '-', '_');

        try {
            uint32_t num = static_cast<uint32_t>(std::stoul(val));
            if (key == "num_warps") num_warps = num;
            else if (key == "instruction_mem_size_bytes") instruction_mem_size_bytes = num;
            else if (key == "instruction_buffer_depth") instruction_buffer_depth = num;
            else if (key == "multiply_pipeline_stages") multiply_pipeline_stages = num;
            else if (key == "num_ldst_units") num_ldst_units = num;
            else if (key == "addr_gen_fifo_depth") addr_gen_fifo_depth = num;
            else if (key == "l1_cache_size_bytes") l1_cache_size_bytes = num;
            else if (key == "num_mshrs") num_mshrs = num;
            else if (key == "write_buffer_depth") write_buffer_depth = num;
            else if (key == "lookup_table_entries") lookup_table_entries = num;
            else if (key == "external_memory_latency_cycles") external_memory_latency_cycles = num;
            else if (key == "external_memory_size_bytes") external_memory_size_bytes = num;
            else if (key == "start_pc") start_pc = num;
            else if (key == "arg0") kernel_args[0] = num;
            else if (key == "arg1") kernel_args[1] = num;
            else if (key == "arg2") kernel_args[2] = num;
            else if (key == "arg3") kernel_args[3] = num;
        } catch (...) {
            // Ignore invalid values
        }
    }
}

} // namespace gpu_sim
