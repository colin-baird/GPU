#include "gpu_sim/config.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/timing/timing_model.h"
#include "gpu_sim/elf_loader.h"
#include "gpu_sim/stats.h"
#include <iostream>
#include <string>
#include <cstring>

using namespace gpu_sim;

static void print_usage(const char* progname) {
    std::cerr << "Usage: " << progname << " <program.bin|program.elf> [options]\n\n"
              << "Options:\n"
              << "  --config=<file.json>       Load configuration from JSON\n"
              << "  --lookup-table=<file.bin>   Load lookup table from binary\n"
              << "  --data=<file.bin>@<addr>    Load data into memory at address\n"
              << "  --functional-only           Run functional model only (no timing)\n"
              << "  --trace                     Enable per-cycle pipeline trace\n"
              << "  --json                      Output stats in JSON format\n"
              << "  --num-warps=<N>             Set number of warps (1-8)\n"
              << "  --arg0=<N> ... --arg3=<N>   Set kernel arguments\n"
              << "  --start-pc=<N>              Override start PC\n"
              << "  --max-cycles=<N>            Limit simulation cycles (0=unlimited)\n"
              << "  --help                      Show this help\n";
}

static int run_functional_only(FunctionalModel& model, const SimConfig& config,
                                uint32_t max_cycles) {
    // Simple functional-only execution: round-robin through warps
    uint32_t pcs[MAX_WARPS];
    for (uint32_t w = 0; w < config.num_warps; ++w) {
        pcs[w] = config.start_pc;
    }

    uint64_t total_instructions = 0;
    uint64_t cycle = 0;

    while (true) {
        bool any_active = false;
        for (uint32_t w = 0; w < config.num_warps; ++w) {
            if (!model.is_warp_active(w)) continue;
            any_active = true;

            TraceEvent evt = model.execute(w, pcs[w]);
            ++total_instructions;

            // Update PC
            if (evt.is_branch && evt.branch_taken) {
                pcs[w] = evt.branch_target;
            } else {
                pcs[w] += 4;
            }

            if (evt.is_ecall || evt.is_ebreak) {
                // Warp completed or panicked
                if (evt.is_ebreak) {
                    std::cerr << "PANIC: warp=" << w << " pc=0x" << std::hex << evt.pc
                              << " cause=" << std::dec << evt.panic_cause << "\n";
                    // Print final register state for panicking warp
                    std::cout << "\nPanic warp " << w << " registers (lane 0):\n";
                    for (int r = 0; r < 32; ++r) {
                        std::cout << "  x" << r << " = 0x" << std::hex
                                  << model.register_file().read(w, 0, r) << std::dec << "\n";
                    }
                    return 1;
                }
            }
        }

        if (!any_active) break;
        if (model.is_panicked()) break;

        ++cycle;
        if (max_cycles > 0 && cycle >= max_cycles) {
            std::cerr << "Max cycles reached (" << max_cycles << ")\n";
            break;
        }
    }

    std::cout << "Functional simulation complete: " << total_instructions
              << " instructions executed across " << config.num_warps << " warps\n\n";

    // Print register state for each warp (lane 0)
    for (uint32_t w = 0; w < config.num_warps; ++w) {
        std::cout << "Warp " << w << " registers (lane 0):\n";
        for (int r = 0; r < 32; ++r) {
            uint32_t val = model.register_file().read(w, 0, r);
            if (val != 0) {
                std::cout << "  x" << r << " = 0x" << std::hex << val
                          << " (" << std::dec << static_cast<int32_t>(val) << ")\n";
            }
        }
    }

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Find program file (first non-option argument)
    std::string program_path;
    std::string config_path;
    std::string lookup_table_path;
    bool json_output = false;
    uint64_t max_cycles = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg.substr(0, 9) == "--config=") {
            config_path = arg.substr(9);
        } else if (arg.substr(0, 16) == "--lookup-table=") {
            lookup_table_path = arg.substr(16);
        } else if (arg == "--json") {
            json_output = true;
        } else if (arg.substr(0, 13) == "--max-cycles=") {
            max_cycles = std::stoull(arg.substr(13));
        } else if (arg.substr(0, 2) != "--" && program_path.empty()) {
            program_path = arg;
        }
    }

    if (program_path.empty()) {
        std::cerr << "Error: no program file specified\n";
        print_usage(argv[0]);
        return 1;
    }

    // Load config
    SimConfig config;
    if (!config_path.empty()) {
        config = SimConfig::from_json(config_path);
    }
    config.apply_cli_overrides(argc, argv);

    try {
        config.validate();
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    // Create functional model
    FunctionalModel model(config);

    // Load program
    try {
        uint32_t entry = load_program(model, program_path);
        if (config.start_pc == 0 && entry != 0) {
            config.start_pc = entry;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading program: " << e.what() << "\n";
        return 1;
    }

    // Load lookup table if specified
    if (!lookup_table_path.empty()) {
        try {
            load_lookup_table(model, lookup_table_path);
        } catch (const std::exception& e) {
            std::cerr << "Error loading lookup table: " << e.what() << "\n";
            return 1;
        }
    }

    // Re-init kernel with final config (in case start_pc was updated from ELF)
    model.init_kernel(config);

    if (config.functional_only) {
        return run_functional_only(model, config, max_cycles);
    }

    // Timing model execution
    Stats stats;
    TimingModel timing(config, model, stats);
    timing.run(max_cycles);

    if (json_output) {
        stats.report_json(std::cout, config.num_warps);
    } else {
        stats.report(std::cout, config.num_warps);
    }

    // Print register state for each warp (lane 0)
    if (!json_output) {
        std::cout << "\n--- Final Register State ---\n";
        for (uint32_t w = 0; w < config.num_warps; ++w) {
            std::cout << "Warp " << w << " (lane 0):\n";
            for (int r = 0; r < 32; ++r) {
                uint32_t val = model.register_file().read(w, 0, r);
                if (val != 0) {
                    std::cout << "  x" << r << " = 0x" << std::hex << val
                              << " (" << std::dec << static_cast<int32_t>(val) << ")\n";
                }
            }
        }
    }

    return model.is_panicked() ? 1 : 0;
}
