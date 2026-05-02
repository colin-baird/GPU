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

// Spec §4.6 DIV latency. The kernel issues one DIV that must still be in
// the divide unit when EBREAK reaches decode.
constexpr uint32_t kDivLatency = 32;
// Spec §4.8.1 panic drain bound (PanicController::MAX_DRAIN_CYCLES).
constexpr uint32_t kMaxDrainCycles = 32;
// Software panic cause (spec §4.8.3 reserves 0x100+ for software use).
constexpr uint32_t kPanicCause = 0x101;

struct Options {
    uint32_t num_warps = 1;
    uint64_t max_cycles = 1024;
    bool json_output = false;
    std::string memory_backend = "fixed";
    std::string dramsim3_config_path = "";
    std::string trace_file_path = "";
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--num-warps=<1-" << MAX_WARPS << ">] [--max-cycles=<N>]\n"
              << "         [--memory-backend=<fixed|dramsim3>]\n"
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
        if (a.rfind("--max-cycles=", 0) == 0)        { opts.max_cycles = parse_u64(a.substr(13), "max-cycles"); continue; }
        if (a == "--json")                            { opts.json_output = true; continue; }
        if (a.rfind("--memory-backend=", 0) == 0)    { opts.memory_backend = a.substr(17); continue; }
        if (a.rfind("--dramsim3-config-path=", 0) == 0) { opts.dramsim3_config_path = a.substr(23); continue; }
        if (a.rfind("--trace-file=", 0) == 0)        { opts.trace_file_path = a.substr(13); continue; }
        throw std::invalid_argument("unknown argument: " + a);
    }
    return opts;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const Options opts = parse_options(argc, argv);

        SimConfig config;
        config.num_warps = opts.num_warps;
        config.memory_backend = opts.memory_backend;
        config.dramsim3_config_path = opts.dramsim3_config_path;
        config.kernel_args[0] = kPanicCause;
        config.kernel_args[1] = 0;
        config.kernel_args[2] = 0;
        config.kernel_args[3] = 0;
        config.validate();

        FunctionalModel model(config);
        config.start_pc = load_program(model, PANIC_DRAIN_TEST_KERNEL_ELF);
        model.init_kernel(config);

        Stats stats;
        TimingTraceOptions trace_options{opts.trace_file_path};
        TimingModel timing(config, model, stats, trace_options);

        // Drive the timing model one tick at a time so we can capture the
        // exact cycle range across which the panic controller is active.
        // The drain duration we report is the panic-active span; spec §4.8.1
        // bounds the drain step by MAX_DRAIN_CYCLES = 32, plus a small
        // fixed overhead for steps 1-4 (latch) and step 6-7 (halt).
        uint64_t panic_first_active_cycle = 0;
        uint64_t panic_last_active_cycle = 0;
        bool saw_panic = false;
        bool finished = false;
        for (uint64_t i = 0; i < opts.max_cycles; ++i) {
            const bool more = timing.tick();
            const auto& snap = timing.current_cycle_snapshot();
            if (snap && snap->panic_active) {
                if (!saw_panic) {
                    saw_panic = true;
                    panic_first_active_cycle = snap->cycle;
                }
                panic_last_active_cycle = snap->cycle;
            }
            if (!more) { finished = true; break; }
        }

        if (!finished) {
            std::cerr << "Kernel did not complete within " << opts.max_cycles << " cycles\n";
            return 2;
        }
        if (!model.is_panicked()) {
            std::cerr << "Kernel did not panic (expected EBREAK panic)\n";
            return 3;
        }
        if (!saw_panic) {
            std::cerr << "Timing snapshot never reported panic_active=true\n";
            return 4;
        }

        // panic_active is true from PanicController::trigger() through the
        // tick that sets done_=true. The panic span therefore covers
        // step-1 latch (1 tick) + step-2 drain (≤ MAX_DRAIN_CYCLES) + step-3
        // halt (1 tick).
        const uint64_t panic_span = panic_last_active_cycle - panic_first_active_cycle + 1;

        const bool ok_panic_cause  = model.panic_cause() == kPanicCause;
        // Drain bound (spec §4.8.1 step 5): drain ≤ 32 cycles. Empirically
        // the in-flight DIV started at issue and must retire within drain;
        // span includes 1 latch + drain + 1 halt.
        const bool ok_drain_bound  = panic_span <= (kMaxDrainCycles + 2);
        // The DIV's busy_cycles counter must still equal 32 (spec §4.6
        // exact latency) — drain neither truncates nor extends it.
        const bool ok_div_busy     = stats.div_stats.busy_cycles == kDivLatency;

        if (opts.json_output) {
            std::ostringstream ss;
            stats.report_json(ss, config.num_warps);
            std::string json = ss.str();
            const auto pos = json.rfind('}');
            std::ostringstream extra;
            extra << ",\n  \"benchmark\": \"panic_drain_test\""
                  << ",\n  \"panic_first_active_cycle\": " << panic_first_active_cycle
                  << ",\n  \"panic_last_active_cycle\": " << panic_last_active_cycle
                  << ",\n  \"panic_span_cycles\": " << panic_span
                  << ",\n  \"max_drain_cycles\": " << kMaxDrainCycles
                  << ",\n  \"panic_cause\": " << model.panic_cause()
                  << ",\n  \"expected_panic_cause\": " << kPanicCause
                  << ",\n  \"match_panic_cause\": " << (ok_panic_cause ? "true" : "false")
                  << ",\n  \"match_drain_bound\": " << (ok_drain_bound ? "true" : "false")
                  << ",\n  \"match_div_busy_cycles\": " << (ok_div_busy ? "true" : "false");
            json.insert(pos, extra.str());
            std::cout << json;
        } else {
            std::cout << "panic_drain_test completed\n";
            std::cout << "  warps: " << opts.num_warps
                      << "  cycles: " << timing.cycle_count() << "\n";
            std::cout << "  panic_active first->last cycle: "
                      << panic_first_active_cycle << " -> "
                      << panic_last_active_cycle
                      << "  span=" << panic_span
                      << " (bound: " << (kMaxDrainCycles + 2) << ")"
                      << (ok_drain_bound ? " OK" : " EXCEEDS BOUND") << "\n";
            std::cout << "  panic_cause: 0x" << std::hex << model.panic_cause()
                      << std::dec << " (expected 0x" << std::hex << kPanicCause
                      << std::dec << ")"
                      << (ok_panic_cause ? " OK" : " MISMATCH") << "\n";
            std::cout << "  div_busy_cycles: " << stats.div_stats.busy_cycles
                      << " (expected " << kDivLatency << ")"
                      << (ok_div_busy ? " OK" : " MISMATCH") << "\n";
            std::cout << "  div_instructions: " << stats.div_stats.instructions << "\n";
            std::cout << "  panic_warp: " << model.panic_warp()
                      << "  panic_pc: 0x" << std::hex << model.panic_pc() << std::dec << "\n";
        }

        return (ok_panic_cause && ok_drain_bound && ok_div_busy) ? 0 : 6;
    } catch (const std::exception& e) {
        std::cerr << "panic_drain_test_bench error: " << e.what() << "\n";
        return 1;
    }
}
