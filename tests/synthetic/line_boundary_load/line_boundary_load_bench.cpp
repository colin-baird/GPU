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

// Aligned base: 128B-aligned, far above the program text.
constexpr uint32_t kAlignedBase = 0x00001000;
// Straddle base: aligned + 64. Lane-0 reads from a different line than
// lane-31, forcing the coalescer to fall back to 32 serialized requests.
constexpr uint32_t kStraddleBase = 0x00002040;

struct Options {
    uint32_t num_warps = 1;
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
    // Populate the regions touched by both loads with deterministic values so
    // the loads do not hit uninitialized memory (FlatMemory's default is 0
    // which would still work, but seeding makes traces more readable).
    for (uint32_t lane = 0; lane < 32; ++lane) {
        mem.write32(kAlignedBase + lane * 4, 0xA0000000u | lane);
    }
    for (uint32_t lane = 0; lane < 32; ++lane) {
        mem.write32(kStraddleBase + lane * 4, 0xB0000000u | lane);
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
        config.kernel_args[0] = kAlignedBase;
        config.kernel_args[1] = kStraddleBase;
        config.kernel_args[2] = 0;
        config.kernel_args[3] = 0;
        config.validate();

        FunctionalModel model(config);
        config.start_pc = load_program(model, LBL_KERNEL_ELF);
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

        // Per-warp expectation: 1 coalesced (aligned load) + 1 serialized
        // (straddle load) — counter increments once per warp-level decision,
        // NOT per individual lane request, despite the doc wording at
        // resources/trace_and_perf_counters.md:415 claiming "totals per lane,
        // not per warp". This counter / doc mismatch is a separate finding.
        const uint64_t expected_coalesced  = static_cast<uint64_t>(opts.num_warps) * 1u;
        const uint64_t expected_serialized = static_cast<uint64_t>(opts.num_warps) * 1u;
        const bool match_coalesced = stats.coalesced_requests == expected_coalesced;
        const bool match_serialized = stats.serialized_requests == expected_serialized;

        if (opts.json_output) {
            std::ostringstream ss;
            stats.report_json(ss, config.num_warps);
            std::string json = ss.str();
            const auto pos = json.rfind('}');
            std::ostringstream extra;
            extra << ",\n  \"benchmark\": \"line_boundary_load\""
                  << ",\n  \"expected_coalesced\": " << expected_coalesced
                  << ",\n  \"expected_serialized\": " << expected_serialized
                  << ",\n  \"match_coalesced\": " << (match_coalesced ? "true" : "false")
                  << ",\n  \"match_serialized\": " << (match_serialized ? "true" : "false");
            json.insert(pos, extra.str());
            std::cout << json;
        } else {
            std::cout << "line_boundary_load completed\n";
            std::cout << "  warps: " << opts.num_warps
                      << "  cycles: " << timing.cycle_count() << "\n";
            std::cout << "  coalesced_requests: " << stats.coalesced_requests
                      << "  (expected " << expected_coalesced << ")"
                      << (match_coalesced ? " OK" : " MISMATCH") << "\n";
            std::cout << "  serialized_requests: " << stats.serialized_requests
                      << "  (expected " << expected_serialized << ")"
                      << (match_serialized ? " OK" : " MISMATCH") << "\n";
            std::cout << "  cache_hits/misses: " << stats.cache_hits << "/" << stats.cache_misses
                      << "  load_hits/misses: " << stats.load_hits << "/" << stats.load_misses << "\n";
            std::cout << "  external_reads: " << stats.external_memory_reads
                      << "  mshr_merged_loads: " << stats.mshr_merged_loads << "\n";
            std::cout << "  mshr_stall_cycles: " << stats.mshr_stall_cycles
                      << "  gather_buffer_stall: " << stats.gather_buffer_stall_cycles << "\n";
        }

        return (match_coalesced && match_serialized) ? 0 : 4;
    } catch (const std::exception& e) {
        std::cerr << "line_boundary_load_bench error: " << e.what() << "\n";
        return 1;
    }
}
