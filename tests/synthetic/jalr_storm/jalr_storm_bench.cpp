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

struct Options {
    uint32_t num_warps = 1;
    uint32_t iterations = 32;
    uint32_t memory_latency = SimConfig{}.external_memory_latency_cycles;
    uint64_t max_cycles = 100000;
    bool json_output = false;
    std::string memory_backend = "fixed";
    std::string dramsim3_config_path = "";
    std::string trace_file_path = "";
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--num-warps=<1-" << MAX_WARPS << ">] [--iterations=<N>] [--memory-latency=<cycles>]\n"
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
        if (a.rfind("--iterations=", 0) == 0)        { opts.iterations = parse_u32(a.substr(13), "iterations"); continue; }
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

} // namespace

int main(int argc, char* argv[]) {
    try {
        const Options opts = parse_options(argc, argv);

        SimConfig config;
        config.num_warps = opts.num_warps;
        config.external_memory_latency_cycles = opts.memory_latency;
        config.memory_backend = opts.memory_backend;
        config.dramsim3_config_path = opts.dramsim3_config_path;
        config.kernel_args[0] = opts.iterations;
        config.kernel_args[1] = 0;
        config.kernel_args[2] = 0;
        config.kernel_args[3] = 0;
        config.validate();

        FunctionalModel model(config);
        config.start_pc = load_program(model, JALR_KERNEL_ELF);
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

        // Per warp: one JALR + one BNEZ per iteration → 2N branch_predictions.
        // Each JALR mispredicts (target ≠ PC+4); BNEZ correctly predicted on
        // iters 1..N-1 and mispredicted on iter N (counter reaches 0).
        // → branch_mispredictions = N + 1 per warp; branch_flushes ditto.
        const uint64_t expected_branch_predictions = static_cast<uint64_t>(opts.num_warps) * 2u * opts.iterations;
        const uint64_t expected_mispredicts        = static_cast<uint64_t>(opts.num_warps) * (opts.iterations + 1);
        const uint64_t expected_flushes            = expected_mispredicts;

        const bool ok_pred  = stats.branch_predictions == expected_branch_predictions;
        const bool ok_mis   = stats.branch_mispredictions == expected_mispredicts;
        const bool ok_flush = stats.branch_flushes == expected_flushes;

        if (opts.json_output) {
            std::ostringstream ss;
            stats.report_json(ss, config.num_warps);
            std::string json = ss.str();
            const auto pos = json.rfind('}');
            std::ostringstream extra;
            extra << ",\n  \"benchmark\": \"jalr_storm\""
                  << ",\n  \"iterations\": " << opts.iterations
                  << ",\n  \"expected_branch_predictions\": " << expected_branch_predictions
                  << ",\n  \"expected_mispredicts\": " << expected_mispredicts
                  << ",\n  \"match_branch_predictions\": " << (ok_pred ? "true" : "false")
                  << ",\n  \"match_mispredicts\": " << (ok_mis ? "true" : "false")
                  << ",\n  \"match_flushes\": " << (ok_flush ? "true" : "false");
            json.insert(pos, extra.str());
            std::cout << json;
        } else {
            std::cout << "jalr_storm completed\n";
            std::cout << "  warps: " << opts.num_warps
                      << "  iterations: " << opts.iterations
                      << "  cycles: " << timing.cycle_count() << "\n";
            std::cout << "  branch_predictions: " << stats.branch_predictions
                      << " (expected " << expected_branch_predictions << ")"
                      << (ok_pred ? " OK" : " MISMATCH") << "\n";
            std::cout << "  branch_mispredictions: " << stats.branch_mispredictions
                      << " (expected " << expected_mispredicts << ")"
                      << (ok_mis ? " OK" : " MISMATCH") << "\n";
            std::cout << "  branch_flushes: " << stats.branch_flushes
                      << " (expected " << expected_flushes << ")"
                      << (ok_flush ? " OK" : " MISMATCH") << "\n";
            std::cout << "  total_instr: " << stats.total_instructions_issued
                      << "  IPC: " << static_cast<double>(stats.total_instructions_issued) /
                                       static_cast<double>(timing.cycle_count()) << "\n";
            std::cout << "  scheduler_idle: " << stats.scheduler_idle_cycles
                      << " (frontend=" << stats.scheduler_frontend_stall_cycles
                      << " backend=" << stats.scheduler_stall_backend_cycles << ")\n";
            std::cout << "  fetch_skips: " << stats.fetch_skip_count
                      << " (backpressure=" << stats.fetch_skip_backpressure
                      << " all_full=" << stats.fetch_skip_all_full << ")\n";
        }

        return (ok_pred && ok_mis && ok_flush) ? 0 : 4;
    } catch (const std::exception& e) {
        std::cerr << "jalr_storm_bench error: " << e.what() << "\n";
        return 1;
    }
}
