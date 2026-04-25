#include "gpu_sim/config.h"
#include "gpu_sim/elf_loader.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/timing/timing_model.h"
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SOFTMAX_ROW_KERNEL_ELF
#define SOFTMAX_ROW_KERNEL_ELF "softmax_row_kernel.elf"
#endif

using namespace gpu_sim;

namespace {

constexpr uint32_t kRows = 64;
constexpr uint32_t kCols = 32;
constexpr uint32_t kInputBase = 0x00002000;
constexpr uint32_t kRowMaxBase = 0x00004000;
constexpr uint32_t kRowSumBase = 0x00005000;
constexpr uint32_t kOutputBase = 0x00018000;
constexpr uint32_t kOutputFracBits = 12;
constexpr uint32_t kLookupEntries = 256;

struct Options {
    uint32_t num_warps = MAX_WARPS;
    uint32_t memory_latency = SimConfig{}.external_memory_latency_cycles;
    uint64_t max_cycles = 5000000;
    bool json_output = false;
    std::string memory_backend = "fixed";
    std::string dramsim3_config_path = "";
};

struct SoftmaxRowCase {
    std::vector<int8_t> logits;
    std::vector<int32_t> row_max;
    std::vector<uint32_t> row_sum;
    std::vector<uint32_t> reference;
    std::vector<uint32_t> lookup_table;
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--num-warps=<1-" << MAX_WARPS << ">] [--memory-latency=<cycles>] [--max-cycles=<N>]\n"
              << "         [--memory-backend=<fixed|dramsim3>] [--dramsim3-config-path=<file.ini>]\n";
    std::cerr << "Defaults: --num-warps=" << MAX_WARPS
              << " --memory-latency=" << SimConfig{}.external_memory_latency_cycles << " --max-cycles=5000000 --memory-backend=fixed\n";
}

uint32_t parse_u32(const std::string& value, const std::string& name) {
    try {
        return static_cast<uint32_t>(std::stoul(value, nullptr, 0));
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid value for " + name + ": " + value);
    }
}

uint64_t parse_u64(const std::string& value, const std::string& name) {
    try {
        return std::stoull(value, nullptr, 0);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid value for " + name + ": " + value);
    }
}

Options parse_options(int argc, char* argv[]) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg.rfind("--num-warps=", 0) == 0) {
            options.num_warps = parse_u32(arg.substr(12), "num-warps");
            continue;
        }
        if (arg.rfind("--memory-latency=", 0) == 0) {
            options.memory_latency = parse_u32(arg.substr(17), "memory-latency");
            continue;
        }
        if (arg.rfind("--max-cycles=", 0) == 0) {
            options.max_cycles = parse_u64(arg.substr(13), "max-cycles");
            continue;
        }
        if (arg == "--json") {
            options.json_output = true;
            continue;
        }
        if (arg.rfind("--memory-backend=", 0) == 0) {
            options.memory_backend = arg.substr(17);
            continue;
        }
        if (arg.rfind("--dramsim3-config-path=", 0) == 0) {
            options.dramsim3_config_path = arg.substr(23);
            continue;
        }
        throw std::invalid_argument("unknown argument: " + arg);
    }

    return options;
}

int8_t make_logit(uint32_t row, uint32_t col) {
    const uint32_t raw = (row * 37u + col * 11u + (row ^ (col * 3u)) + 19u) % 128u;
    return static_cast<int8_t>(static_cast<int32_t>(raw) - 64);
}

std::vector<uint32_t> build_lookup_table() {
    std::vector<uint32_t> table(kLookupEntries);
    for (uint32_t i = 0; i < kLookupEntries; ++i) {
        const double value = std::exp(-static_cast<double>(i) / 32.0) * 4096.0;
        const auto rounded = static_cast<uint32_t>(std::lround(value));
        table[i] = rounded == 0 ? 1u : rounded;
    }
    return table;
}

SoftmaxRowCase build_case() {
    SoftmaxRowCase test_case;
    test_case.logits.resize(kRows * kCols);
    test_case.row_max.resize(kRows);
    test_case.row_sum.resize(kRows);
    test_case.reference.assign(kRows * kCols, 0);
    test_case.lookup_table = build_lookup_table();

    for (uint32_t row = 0; row < kRows; ++row) {
        int32_t max_value = static_cast<int32_t>(make_logit(row, 0));
        for (uint32_t col = 0; col < kCols; ++col) {
            const int8_t value = make_logit(row, col);
            test_case.logits[row * kCols + col] = value;
            if (static_cast<int32_t>(value) > max_value) {
                max_value = static_cast<int32_t>(value);
            }
        }

        test_case.row_max[row] = max_value;

        uint32_t sum = 0;
        for (uint32_t col = 0; col < kCols; ++col) {
            const int32_t delta = max_value - static_cast<int32_t>(test_case.logits[row * kCols + col]);
            const uint32_t lookup = test_case.lookup_table[static_cast<uint32_t>(delta)];
            sum += lookup;
        }
        test_case.row_sum[row] = sum;

        for (uint32_t col = 0; col < kCols; ++col) {
            const int32_t delta = max_value - static_cast<int32_t>(test_case.logits[row * kCols + col]);
            const uint32_t lookup = test_case.lookup_table[static_cast<uint32_t>(delta)];
            const uint64_t scaled = static_cast<uint64_t>(lookup) << kOutputFracBits;
            test_case.reference[row * kCols + col] = static_cast<uint32_t>(scaled / sum);
        }
    }

    return test_case;
}

void load_logits(const std::vector<int8_t>& logits, FlatMemory& memory) {
    for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t col = 0; col < kCols; ++col) {
            const uint32_t addr = kInputBase + row * kCols + col;
            memory.write8(addr, static_cast<uint8_t>(logits[row * kCols + col]));
        }
    }
}

void load_row_scalars(const std::vector<int32_t>& row_max,
                      const std::vector<uint32_t>& row_sum,
                      FlatMemory& memory) {
    for (uint32_t row = 0; row < kRows; ++row) {
        memory.write32(kRowMaxBase + row * sizeof(uint32_t),
                       static_cast<uint32_t>(row_max[row]));
        memory.write32(kRowSumBase + row * sizeof(uint32_t), row_sum[row]);
    }
}

void load_lookup_table(const std::vector<uint32_t>& table, FunctionalModel& model) {
    for (uint32_t i = 0; i < table.size(); ++i) {
        model.lookup_table().write(i, table[i]);
    }
}

bool all_warps_inactive(const FunctionalModel& model, uint32_t num_warps) {
    for (uint32_t warp = 0; warp < num_warps; ++warp) {
        if (model.is_warp_active(warp)) {
            return false;
        }
    }
    return true;
}

bool verify_output(const FunctionalModel& model, const std::vector<uint32_t>& reference) {
    uint32_t mismatch_count = 0;

    for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t col = 0; col < kCols; ++col) {
            const uint32_t addr = kOutputBase + row * (kCols * sizeof(uint32_t)) + col * sizeof(uint32_t);
            const uint32_t observed = model.memory().read32(addr);
            const uint32_t expected = reference[row * kCols + col];
            if (observed == expected) {
                continue;
            }

            if (mismatch_count < 8) {
                std::cerr << "Mismatch at row " << row
                          << ", col " << col
                          << ": expected " << expected
                          << ", observed " << observed << "\n";
            }
            ++mismatch_count;
        }
    }

    if (mismatch_count == 0) {
        return true;
    }

    std::cerr << "Total mismatches: " << mismatch_count << "\n";
    return false;
}

void print_summary(const Options& options, const FunctionalModel& model,
                   const TimingModel& timing, const Stats& stats) {
    const double elements = static_cast<double>(kRows) * static_cast<double>(kCols);
    const double cycles = static_cast<double>(timing.cycle_count());
    const double elements_per_cycle = elements / cycles;
    const double tlookup_ops_per_cycle = elements / cycles;

    uint64_t checksum = 0;
    for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t col = 0; col < kCols; ++col) {
            checksum += model.memory().read32(
                kOutputBase + row * (kCols * sizeof(uint32_t)) + col * sizeof(uint32_t));
        }
    }

    std::cout << "Softmax-row kernel verified\n";
    std::cout << "  rows x cols: " << kRows << " x " << kCols << "\n";
    std::cout << "  resident warps: " << options.num_warps << "\n";
    std::cout << "  memory backend: " << options.memory_backend << "\n";
    std::cout << "  external memory latency: " << options.memory_latency << " cycles\n";
    std::cout << "  cycles: " << timing.cycle_count() << "\n";
    std::cout << "  issued instructions: " << stats.total_instructions_issued << "\n";
    std::cout << "  fetch skips: " << stats.fetch_skip_count
              << " (backpressure=" << stats.fetch_skip_backpressure
              << " all_full=" << stats.fetch_skip_all_full << ")\n";
    std::cout << "  scheduler idle: " << stats.scheduler_idle_cycles
              << " (frontend=" << stats.scheduler_frontend_stall_cycles
              << " backend=" << stats.scheduler_stall_backend_cycles << ")\n";
    std::cout << "  elements/cycle: " << elements_per_cycle << "\n";
    std::cout << "  tlookup ops/cycle: " << tlookup_ops_per_cycle << "\n";
    std::cout << "  checksum: " << checksum << "\n";
    std::cout << "  cache hits/misses: " << stats.cache_hits
              << "/" << stats.cache_misses << "\n";
    std::cout << "  coalesced/serialized memory ops: " << stats.coalesced_requests
              << "/" << stats.serialized_requests << "\n";
    std::cout << "  external reads/writes: " << stats.external_memory_reads
              << "/" << stats.external_memory_writes << "\n";
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const Options options = parse_options(argc, argv);
        const SoftmaxRowCase test_case = build_case();

        SimConfig config;
        config.num_warps = options.num_warps;
        config.external_memory_latency_cycles = options.memory_latency;
        config.memory_backend = options.memory_backend;
        config.dramsim3_config_path = options.dramsim3_config_path;
        config.kernel_args[0] = kInputBase;
        config.kernel_args[1] = kRowMaxBase;
        config.kernel_args[2] = kRowSumBase;
        config.kernel_args[3] = kRows;
        config.validate();

        FunctionalModel model(config);
        config.start_pc = load_program(model, SOFTMAX_ROW_KERNEL_ELF);

        load_logits(test_case.logits, model.memory());
        load_row_scalars(test_case.row_max, test_case.row_sum, model.memory());
        model.init_kernel(config);
        load_lookup_table(test_case.lookup_table, model);

        Stats stats;
        TimingModel timing(config, model, stats);
        timing.run(options.max_cycles);

        if (!all_warps_inactive(model, config.num_warps)) {
            std::cerr << "Kernel did not complete within " << options.max_cycles
                      << " cycles\n";
            return 2;
        }

        if (model.is_panicked()) {
            std::cerr << "Kernel panicked at pc=0x" << std::hex << model.panic_pc()
                      << " cause=" << std::dec << model.panic_cause() << "\n";
            return 3;
        }

        if (!verify_output(model, test_case.reference)) {
            return 4;
        }

        if (options.json_output) {
            std::ostringstream ss;
            stats.report_json(ss, config.num_warps);
            std::string json = ss.str();
            const double cycles = static_cast<double>(timing.cycle_count());
            const double elements = static_cast<double>(kRows) * static_cast<double>(kCols);
            auto pos = json.rfind('}');
            std::ostringstream derived;
            derived << std::setprecision(6)
                    << ",\n  \"benchmark\": \"softmax_row\""
                    << ",\n  \"elements_per_cycle\": " << elements / cycles;
            json.insert(pos, derived.str());
            std::cout << json;
        } else {
            print_summary(options, model, timing, stats);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "softmax_row_bench error: " << e.what() << "\n";
        return 1;
    }
}
