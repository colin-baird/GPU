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
#include <vector>

using namespace gpu_sim;

namespace {

struct Options {
    uint32_t num_warps = 2;
    uint32_t iterations = 256;
    uint32_t memory_latency = SimConfig{}.external_memory_latency_cycles;
    uint64_t max_cycles = 200000;
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

uint32_t parse_u32(const std::string& v, const std::string& name) {
    try { return static_cast<uint32_t>(std::stoul(v, nullptr, 0)); }
    catch (const std::exception&) { throw std::invalid_argument("invalid value for " + name + ": " + v); }
}

uint64_t parse_u64(const std::string& v, const std::string& name) {
    try { return std::stoull(v, nullptr, 0); }
    catch (const std::exception&) { throw std::invalid_argument("invalid value for " + name + ": " + v); }
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
        config.start_pc = load_program(model, RR_TIEBREAK_KERNEL_ELF);
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

        // Setup (mv, li, li) = 3 instructions; loop body = 5 (3 ALU + addi + bnez);
        // plus ECALL at the end.
        const uint64_t expected_per_warp = 3ull + 5ull * opts.iterations + 1ull;

        // Verify: per-warp instruction counts equal across warps (within ±1 for
        // ECALL drift). Spec §4.2/§4.3 strict round-robin should produce exactly
        // equal counts on identical streams.
        uint64_t min_w = stats.warp_instructions[0];
        uint64_t max_w = stats.warp_instructions[0];
        for (uint32_t w = 1; w < opts.num_warps; ++w) {
            min_w = std::min(min_w, stats.warp_instructions[w]);
            max_w = std::max(max_w, stats.warp_instructions[w]);
        }
        const uint64_t imbalance = max_w - min_w;

        if (opts.json_output) {
            std::ostringstream ss;
            stats.report_json(ss, config.num_warps);
            std::string json = ss.str();
            const auto pos = json.rfind('}');
            std::ostringstream extra;
            extra << std::setprecision(6)
                  << ",\n  \"benchmark\": \"rr_tiebreak\""
                  << ",\n  \"iterations\": " << opts.iterations
                  << ",\n  \"expected_per_warp_instructions\": " << expected_per_warp
                  << ",\n  \"warp_instruction_imbalance\": " << imbalance;
            json.insert(pos, extra.str());
            std::cout << json;
        } else {
            std::cout << "rr_tiebreak completed\n";
            std::cout << "  warps: " << opts.num_warps
                      << "  iterations: " << opts.iterations << "\n";
            std::cout << "  cycles: " << timing.cycle_count()
                      << "  total instr: " << stats.total_instructions_issued
                      << "  IPC: " << static_cast<double>(stats.total_instructions_issued) /
                                       static_cast<double>(timing.cycle_count()) << "\n";
            std::cout << "  expected per-warp instr: " << expected_per_warp << "\n";
            std::cout << "  per-warp issued:";
            for (uint32_t w = 0; w < opts.num_warps; ++w) {
                std::cout << " W" << w << "=" << stats.warp_instructions[w];
            }
            std::cout << "\n";
            std::cout << "  imbalance (max-min): " << imbalance << "\n";
            std::cout << "  branch_predictions=" << stats.branch_predictions
                      << " mispredicts=" << stats.branch_mispredictions
                      << " flushes=" << stats.branch_flushes << "\n";
            std::cout << "  scheduler_idle=" << stats.scheduler_idle_cycles
                      << " (frontend=" << stats.scheduler_frontend_stall_cycles
                      << " backend=" << stats.scheduler_stall_backend_cycles << ")\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "rr_tiebreak_bench error: " << e.what() << "\n";
        return 1;
    }
}
