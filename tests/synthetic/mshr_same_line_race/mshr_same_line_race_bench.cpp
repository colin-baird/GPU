#include "gpu_sim/config.h"
#include "gpu_sim/elf_loader.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/timing/timing_model.h"
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace gpu_sim;

namespace {

// Shared cache-line base — 128B-aligned. All warps load from this line.
constexpr uint32_t kSharedBase = 0x00001000;

struct Options {
    uint32_t num_warps = 4;
    uint32_t memory_latency = SimConfig{}.external_memory_latency_cycles;
    uint64_t max_cycles = 100000;
    bool json_output = false;
    std::string memory_backend = "fixed";
    std::string dramsim3_config_path = "";
    std::string trace_file_path = "";
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--num-warps=<1-" << MAX_WARPS << ">] [--memory-latency=<cycles>]\n"
              << "         [--max-cycles=<N>] [--memory-backend=<fixed|dramsim3>]\n"
              << "         [--dramsim3-config-path=<file.ini>] [--trace-file=<path>] [--json]\n";
}

uint32_t parse_u32(const std::string& v, const std::string& n) {
    try { return static_cast<uint32_t>(std::stoul(v, nullptr, 0)); }
    catch (const std::exception&) { throw std::invalid_argument("invalid " + n + ": " + v); }
}
uint64_t parse_u64(const std::string& v, const std::string& n) {
    try { return std::stoull(v, nullptr, 0); }
    catch (const std::exception&) { throw std::invalid_argument("invalid " + n + ": " + v); }
}

Options parse_options(int argc, char* argv[]) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help") { print_usage(argv[0]); std::exit(0); }
        if (a.rfind("--num-warps=", 0) == 0)         { opts.num_warps = parse_u32(a.substr(12), "num-warps"); continue; }
        if (a.rfind("--memory-latency=", 0) == 0)    { opts.memory_latency = parse_u32(a.substr(17), "memory-latency"); continue; }
        if (a.rfind("--max-cycles=", 0) == 0)        { opts.max_cycles = parse_u64(a.substr(13), "max-cycles"); continue; }
        if (a == "--json")                            { opts.json_output = true; continue; }
        if (a.rfind("--memory-backend=", 0) == 0)    { opts.memory_backend = a.substr(17); continue; }
        if (a.rfind("--dramsim3-config-path=", 0) == 0) { opts.dramsim3_config_path = a.substr(23); continue; }
        if (a.rfind("--trace-file=", 0) == 0)        { opts.trace_file_path = a.substr(13); continue; }
        throw std::invalid_argument("unknown argument: " + a);
    }
    return opts;
}

bool all_warps_inactive(const FunctionalModel& m, uint32_t n) {
    for (uint32_t w = 0; w < n; ++w) if (m.is_warp_active(w)) return false;
    return true;
}

void seed_memory(FlatMemory& mem) {
    for (uint32_t lane = 0; lane < 32; ++lane) {
        mem.write32(kSharedBase + lane * 4, 0xC0000000u | lane);
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const Options opts = parse_options(argc, argv);

        SimConfig config;
        config.num_warps = opts.num_warps;
        config.external_memory_latency_cycles = opts.memory_latency;
        config.memory_backend = opts.memory_backend;
        config.dramsim3_config_path = opts.dramsim3_config_path;
        config.kernel_args[0] = kSharedBase;
        config.kernel_args[1] = 0;
        config.kernel_args[2] = 0;
        config.kernel_args[3] = 0;
        config.validate();

        FunctionalModel model(config);
        config.start_pc = load_program(model, MSLR_KERNEL_ELF);
        seed_memory(model.memory());
        model.init_kernel(config);

        Stats stats;
        TimingTraceOptions trace_options{opts.trace_file_path};
        TimingModel timing(config, model, stats, trace_options);
        timing.run(opts.max_cycles);

        if (!all_warps_inactive(model, config.num_warps)) {
            std::cerr << "Kernel did not complete within " << opts.max_cycles << " cycles\n";
            return 2;
        }
        if (model.is_panicked()) {
            std::cerr << "Kernel panicked at pc=0x" << std::hex << model.panic_pc()
                      << " cause=" << std::dec << model.panic_cause() << "\n";
            return 3;
        }

        // All warps target the same line: 1 primary MSHR + (N-1) secondaries.
        const uint64_t expected_coalesced  = opts.num_warps;
        const uint64_t expected_external_reads = 1;
        const uint64_t expected_mshr_merged = (opts.num_warps > 0) ? (opts.num_warps - 1) : 0;

        const bool ok_coalesced  = stats.coalesced_requests == expected_coalesced;
        const bool ok_external   = stats.external_memory_reads == expected_external_reads;
        const bool ok_merged     = stats.mshr_merged_loads == expected_mshr_merged;

        if (opts.json_output) {
            std::ostringstream ss;
            stats.report_json(ss, config.num_warps);
            std::string json = ss.str();
            const auto pos = json.rfind('}');
            std::ostringstream extra;
            extra << ",\n  \"benchmark\": \"mshr_same_line_race\""
                  << ",\n  \"expected_coalesced\": " << expected_coalesced
                  << ",\n  \"expected_external_reads\": " << expected_external_reads
                  << ",\n  \"expected_mshr_merged_loads\": " << expected_mshr_merged
                  << ",\n  \"match_coalesced\": " << (ok_coalesced ? "true" : "false")
                  << ",\n  \"match_external_reads\": " << (ok_external ? "true" : "false")
                  << ",\n  \"match_mshr_merged\": " << (ok_merged ? "true" : "false");
            json.insert(pos, extra.str());
            std::cout << json;
        } else {
            std::cout << "mshr_same_line_race completed\n";
            std::cout << "  warps: " << opts.num_warps
                      << "  cycles: " << timing.cycle_count() << "\n";
            std::cout << "  coalesced/serialized: " << stats.coalesced_requests
                      << "/" << stats.serialized_requests
                      << "  (expected coalesced=" << expected_coalesced
                      << (ok_coalesced ? " OK" : " MISMATCH") << ")\n";
            std::cout << "  external_reads: " << stats.external_memory_reads
                      << " (expected " << expected_external_reads << ")"
                      << (ok_external ? " OK" : " MISMATCH") << "\n";
            std::cout << "  cache_hits/misses: " << stats.cache_hits << "/" << stats.cache_misses << "\n";
            std::cout << "  mshr_merged_loads: " << stats.mshr_merged_loads
                      << " (expected " << expected_mshr_merged << ")"
                      << (ok_merged ? " OK" : " MISMATCH") << "\n";
            std::cout << "  mshr_stall: " << stats.mshr_stall_cycles
                      << "  secondary_drain: " << stats.secondary_drain_cycles
                      << "  line_pin_stall: " << stats.line_pin_stall_cycles << "\n";
        }

        return (ok_coalesced && ok_external && ok_merged) ? 0 : 4;
    } catch (const std::exception& e) {
        std::cerr << "mshr_same_line_race_bench error: " << e.what() << "\n";
        return 1;
    }
}
