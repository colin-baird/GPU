#include "gpu_sim/config.h"
#include "gpu_sim/elf_loader.h"
#include "gpu_sim/backend.h"
#include <iostream>
#include <string>

using namespace gpu_sim;

static void print_usage(const char* progname) {
    std::cerr << "Usage: " << progname << " <program.bin|program.elf> [options]\n\n"
              << "Options:\n"
              << "  --backend=<name>            Execution backend (default: perf_sim)\n"
              << "  --config=<file.json>        Load configuration from JSON\n"
              << "  --lookup-table=<file.bin>    Load lookup table from binary\n"
              << "  --data=<file.bin>@<addr>     Load data into memory at address\n"
              << "  --functional-only            Run functional model only (no timing)\n"
              << "  --trace                      Enable per-cycle text pipeline trace\n"
              << "  --trace-text                 Alias for --trace\n"
              << "  --trace-file=<path>          Write structured Chrome trace JSON\n"
              << "  --json                       Output stats in JSON format\n"
              << "  --num-warps=<N>              Set number of warps (1-8)\n"
              << "  --arg0=<N> ... --arg3=<N>    Set kernel arguments\n"
              << "  --start-pc=<N>               Override start PC\n"
              << "  --max-cycles=<N>             Limit simulation cycles (0=unlimited)\n"
              << "  --help                       Show this help\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse top-level options
    std::string program_path;
    std::string config_path;
    std::string backend_name = "perf_sim";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg.substr(0, 10) == "--backend=") {
            backend_name = arg.substr(10);
        } else if (arg.substr(0, 9) == "--config=") {
            config_path = arg.substr(9);
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

    // Load program image
    ProgramImage image;
    try {
        image = load_program_image(program_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading program: " << e.what() << "\n";
        return 1;
    }

    // Create and run backend
    auto backend = create_backend(backend_name);
    if (!backend) {
        std::cerr << "Error: unknown backend '" << backend_name << "'\n";
        return 1;
    }

    return backend->run(image, config, argc, argv);
}
