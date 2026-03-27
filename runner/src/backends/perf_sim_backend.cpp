#include "runner/backends/perf_sim_backend.h"
#include "gpu_sim/elf_loader.h"
#include "gpu_sim/timing/timing_model.h"
#include "gpu_sim/stats.h"
#include <iostream>
#include <string>
#include <cstring>

namespace gpu_sim {

void PerfSimBackend::load_image(FunctionalModel& model,
                                const ProgramImage& image) {
    load_image_into_model(model, image);
}

int PerfSimBackend::run_functional_only(FunctionalModel& model,
                                        const SimConfig& config,
                                        uint64_t max_cycles) {
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

            if (evt.is_branch && evt.branch_taken) {
                pcs[w] = evt.branch_target;
            } else {
                pcs[w] += 4;
            }

            if (evt.is_ecall || evt.is_ebreak) {
                if (evt.is_ebreak) {
                    std::cerr << "PANIC: warp=" << w << " pc=0x" << std::hex
                              << evt.pc << " cause=" << std::dec
                              << evt.panic_cause << "\n";
                    std::cout << "\nPanic warp " << w
                              << " registers (lane 0):\n";
                    for (int r = 0; r < 32; ++r) {
                        std::cout << "  x" << r << " = 0x" << std::hex
                                  << model.register_file().read(w, 0, r)
                                  << std::dec << "\n";
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
              << " instructions executed across " << config.num_warps
              << " warps\n\n";

    for (uint32_t w = 0; w < config.num_warps; ++w) {
        std::cout << "Warp " << w << " registers (lane 0):\n";
        for (int r = 0; r < 32; ++r) {
            uint32_t val = model.register_file().read(w, 0, r);
            if (val != 0) {
                std::cout << "  x" << r << " = 0x" << std::hex << val
                          << " (" << std::dec << static_cast<int32_t>(val)
                          << ")\n";
            }
        }
    }

    return 0;
}

int PerfSimBackend::run(const ProgramImage& image, SimConfig& config,
                        int argc, char* argv[]) {
    // Parse backend-specific options from CLI
    std::string lookup_table_path;
    bool json_output = false;
    uint64_t max_cycles = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.substr(0, 16) == "--lookup-table=") {
            lookup_table_path = arg.substr(16);
        } else if (arg == "--json") {
            json_output = true;
        } else if (arg.substr(0, 13) == "--max-cycles=") {
            max_cycles = std::stoull(arg.substr(13));
        }
    }

    // Create functional model and load program image
    FunctionalModel model(config);
    load_image(model, image);

    if (config.start_pc == 0 && image.entry_pc != 0) {
        config.start_pc = image.entry_pc;
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

    // Load data files from CLI (--data=<file>@<addr>)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.substr(0, 7) == "--data=") {
            std::string spec = arg.substr(7);
            auto at_pos = spec.find('@');
            if (at_pos == std::string::npos) {
                std::cerr << "Error: --data requires format <file>@<addr>\n";
                return 1;
            }
            std::string data_path = spec.substr(0, at_pos);
            uint32_t addr = static_cast<uint32_t>(
                std::stoul(spec.substr(at_pos + 1), nullptr, 0));
            try {
                load_data(model, data_path, addr);
            } catch (const std::exception& e) {
                std::cerr << "Error loading data: " << e.what() << "\n";
                return 1;
            }
        }
    }

    // Initialize kernel with final config
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
                              << " (" << std::dec << static_cast<int32_t>(val)
                              << ")\n";
                }
            }
        }
    }

    return model.is_panicked() ? 1 : 0;
}

} // namespace gpu_sim
