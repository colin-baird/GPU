#include "gpu_sim/config.h"
#include "gpu_sim/elf_loader.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/timing/timing_model.h"
#include "gpu_sim/types.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gpu_sim;

#ifndef LAYERNORM_LITE_KERNEL_ELF
#define LAYERNORM_LITE_KERNEL_ELF "layernorm_lite_kernel.elf"
#endif

namespace {

constexpr uint32_t kVectorLen = 64;
constexpr uint32_t kInputBase = 0x00002000;
constexpr uint32_t kLookupBase = 768;

struct Options {
    uint32_t num_warps = MAX_WARPS;
    uint32_t memory_latency = 100;
    uint64_t max_cycles = 5000000;
};

struct LayernormLiteCase {
    std::vector<int8_t> input;
    std::vector<int8_t> reference;
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--num-warps=<1-" << MAX_WARPS << ">] [--memory-latency=<cycles>] [--max-cycles=<N>]\n";
    std::cerr << "Defaults: --num-warps=" << MAX_WARPS << " --memory-latency=100 --max-cycles=5000000\n";
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
        throw std::invalid_argument("unknown argument: " + arg);
    }

    return options;
}

int8_t make_input_value(uint32_t row, uint32_t col) {
    const uint32_t value = (row * 17u + col * 5u + 3u) % 15u;
    return static_cast<int8_t>(static_cast<int32_t>(value) - 7);
}

uint32_t make_rsqrt_value(uint32_t idx) {
    const double scale = static_cast<double>(1u << 15);
    const double input = 1.0 + static_cast<double>(idx);
    const double rsqrt = 1.0 / std::sqrt(input);
    const auto value = static_cast<int64_t>(std::llround(rsqrt * scale));
    return static_cast<uint32_t>(
        std::clamp<int64_t>(value, 0, static_cast<int64_t>(std::numeric_limits<uint32_t>::max())));
}

std::array<uint32_t, 256> build_rsqrt_table() {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < table.size(); ++i) {
        table[i] = make_rsqrt_value(i);
    }
    return table;
}

LayernormLiteCase build_case(uint32_t rows) {
    LayernormLiteCase test_case;
    test_case.input.resize(rows * kVectorLen);
    test_case.reference.resize(rows * kVectorLen);

    const auto rsqrt_table = build_rsqrt_table();

    for (uint32_t row = 0; row < rows; ++row) {
        int32_t sumsq = 0;
        for (uint32_t col = 0; col < kVectorLen; ++col) {
            const int8_t value = make_input_value(row, col);
            test_case.input[row * kVectorLen + col] = value;
            sumsq += static_cast<int32_t>(value) * static_cast<int32_t>(value);
        }

        uint32_t idx = static_cast<uint32_t>(sumsq >> 4);
        if (idx > 255u) {
            idx = 255u;
        }
        const int32_t factor = static_cast<int32_t>(rsqrt_table[idx]);

        for (uint32_t col = 0; col < kVectorLen; ++col) {
            const int32_t x = static_cast<int32_t>(test_case.input[row * kVectorLen + col]);
            const int32_t y = static_cast<int32_t>((static_cast<int64_t>(x) * factor) >> 15);
            test_case.reference[row * kVectorLen + col] = static_cast<int8_t>(y);
        }
    }

    return test_case;
}

void load_transposed_input(const std::vector<int8_t>& input, uint32_t rows, FlatMemory& memory) {
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < kVectorLen; ++col) {
            const uint32_t addr = kInputBase + col * rows + row;
            memory.write8(addr, static_cast<uint8_t>(input[row * kVectorLen + col]));
        }
    }
}

void clear_transposed_output(uint32_t rows, uint32_t output_base, FlatMemory& memory) {
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < kVectorLen; ++col) {
            const uint32_t addr = output_base + col * rows + row;
            memory.write8(addr, 0);
        }
    }
}

void load_rsqrt_table(LookupTable& table) {
    const auto rsqrt_table = build_rsqrt_table();
    for (uint32_t i = 0; i < rsqrt_table.size(); ++i) {
        table.write(kLookupBase + i, rsqrt_table[i]);
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

void dump_timeout_snapshot(const TimingModel& timing) {
    const auto& snapshot_opt = timing.last_cycle_snapshot();
    if (!snapshot_opt.has_value()) {
        return;
    }

    const auto& snapshot = *snapshot_opt;
    std::cerr << "Last cycle snapshot at cycle " << snapshot.cycle << ":\n";
    std::cerr << "  opcoll busy=" << snapshot.opcoll_busy
              << " alu busy=" << snapshot.alu_busy
              << " mul busy=" << snapshot.mul_busy
              << " div busy=" << snapshot.div_busy
              << " tlookup busy=" << snapshot.tlookup_busy
              << " ldst busy=" << snapshot.ldst_busy
              << " ldst fifo=" << snapshot.ldst_fifo_depth
              << " mshrs=" << snapshot.active_mshrs
              << " write buffer=" << snapshot.write_buffer_depth << "\n";

    for (uint32_t warp = 0; warp < snapshot.num_warps; ++warp) {
        const auto& warp_state = snapshot.warps[warp];
        if (!warp_state.active) {
            continue;
        }

        std::cerr << "  warp " << warp
                  << " state=" << to_string(warp_state.state)
                  << " rest=" << to_string(warp_state.rest_reason)
                  << " pc=0x" << std::hex << warp_state.pc << std::dec
                  << " rd=x" << static_cast<int>(warp_state.dest_reg) << "\n";
    }
}

bool verify_output(const FunctionalModel& model, uint32_t rows, uint32_t output_base,
                   const std::vector<int8_t>& reference) {
    uint32_t mismatch_count = 0;

    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < kVectorLen; ++col) {
            const uint32_t addr = output_base + col * rows + row;
            const int8_t observed = static_cast<int8_t>(model.memory().read8(addr));
            const int8_t expected = reference[row * kVectorLen + col];
            if (observed == expected) {
                continue;
            }

            if (mismatch_count < 8) {
                std::cerr << "Mismatch at row " << row
                          << ", col " << col
                          << ": expected " << static_cast<int32_t>(expected)
                          << ", observed " << static_cast<int32_t>(observed) << "\n";
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

void print_summary(const Options& options, uint32_t rows, const TimingModel& timing, const Stats& stats) {
    const double elements = static_cast<double>(rows) * static_cast<double>(kVectorLen);
    const double cycles = static_cast<double>(timing.cycle_count());
    const double elements_per_cycle = elements / cycles;

    std::cout << "Layernorm-lite kernel verified\n";
    std::cout << "  rows: " << rows << "  vector length: " << kVectorLen << "\n";
    std::cout << "  resident warps: " << options.num_warps << "\n";
    std::cout << "  external memory latency: " << options.memory_latency << " cycles\n";
    std::cout << "  cycles: " << timing.cycle_count() << "\n";
    std::cout << "  issued instructions: " << stats.total_instructions_issued << "\n";
    std::cout << "  elements/cycle: " << elements_per_cycle << "\n";
    std::cout << "  MUL instructions: " << stats.mul_stats.instructions << "\n";
    std::cout << "  TLOOKUP instructions: " << stats.tlookup_stats.instructions << "\n";
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
        const uint32_t rows = options.num_warps * WARP_SIZE;

        const uint32_t input_bytes = kVectorLen * rows;
        const uint32_t output_base = (kInputBase + input_bytes + 0xFFFu) & ~0xFFFu;

        SimConfig config;
        config.num_warps = options.num_warps;
        config.external_memory_latency_cycles = options.memory_latency;
        config.kernel_args[0] = kInputBase;
        config.kernel_args[1] = output_base;
        config.kernel_args[2] = rows;
        config.kernel_args[3] = 0;
        config.validate();

        FunctionalModel model(config);
        config.start_pc = load_program(model, LAYERNORM_LITE_KERNEL_ELF);

        const LayernormLiteCase test_case = build_case(rows);
        load_transposed_input(test_case.input, rows, model.memory());
        clear_transposed_output(rows, output_base, model.memory());
        load_rsqrt_table(model.lookup_table());
        model.init_kernel(config);

        Stats stats;
        TimingModel timing(config, model, stats);
        timing.run(options.max_cycles);

        if (!all_warps_inactive(model, config.num_warps)) {
            std::cerr << "Kernel did not complete within " << options.max_cycles
                      << " cycles\n";
            dump_timeout_snapshot(timing);
            return 2;
        }

        if (model.is_panicked()) {
            std::cerr << "Kernel panicked at pc=0x" << std::hex << model.panic_pc()
                      << " cause=" << std::dec << model.panic_cause() << "\n";
            return 3;
        }

        if (stats.serialized_requests != 0) {
            std::cerr << "Expected fully coalesced memory traffic, observed "
                      << stats.serialized_requests << " serialized accesses\n";
            return 4;
        }

        if (!verify_output(model, rows, output_base, test_case.reference)) {
            return 5;
        }

        print_summary(options, rows, timing, stats);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "layernorm_lite_bench error: " << e.what() << "\n";
        return 1;
    }
}
