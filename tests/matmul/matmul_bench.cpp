#include "gpu_sim/config.h"
#include "gpu_sim/elf_loader.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/timing/timing_model.h"
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gpu_sim;

namespace {

constexpr uint32_t kMatmulM = 128;
constexpr uint32_t kMatmulN = 128;
constexpr uint32_t kMatmulK = 128;
constexpr uint32_t kTileCols = 32;
constexpr uint32_t kABytesPerRow = kMatmulK;
constexpr uint32_t kBTileStrideBytes = (kMatmulK / 4) * kTileCols * sizeof(uint32_t);
constexpr uint32_t kCBytesPerRow = kMatmulN * sizeof(int32_t);

constexpr uint32_t kABase = 0x00002000;
constexpr uint32_t kBBase = 0x00008000;
constexpr uint32_t kCBase = 0x00010000;

struct Options {
    uint32_t num_warps = MAX_WARPS;
    uint32_t memory_latency = 100;
    uint64_t max_cycles = 5000000;
};

struct MatmulCase {
    std::vector<int8_t> a;
    std::vector<int8_t> b;
    std::vector<int32_t> reference;
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

int8_t make_a_value(uint32_t row, uint32_t col) {
    const uint32_t value = (row * 17u + col * 5u + 3u) % 15u;
    return static_cast<int8_t>(static_cast<int32_t>(value) - 7);
}

int8_t make_b_value(uint32_t row, uint32_t col) {
    const uint32_t value = (row * 13u + col * 7u + 1u) % 15u;
    return static_cast<int8_t>(static_cast<int32_t>(value) - 7);
}

uint32_t pack_int8x4(int8_t v0, int8_t v1, int8_t v2, int8_t v3) {
    return static_cast<uint32_t>(static_cast<uint8_t>(v0)) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v1)) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v2)) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v3)) << 24);
}

MatmulCase build_case() {
    MatmulCase test_case;
    test_case.a.resize(kMatmulM * kMatmulK);
    test_case.b.resize(kMatmulK * kMatmulN);
    test_case.reference.assign(kMatmulM * kMatmulN, 0);

    for (uint32_t row = 0; row < kMatmulM; ++row) {
        for (uint32_t col = 0; col < kMatmulK; ++col) {
            test_case.a[row * kMatmulK + col] = make_a_value(row, col);
        }
    }

    for (uint32_t row = 0; row < kMatmulK; ++row) {
        for (uint32_t col = 0; col < kMatmulN; ++col) {
            test_case.b[row * kMatmulN + col] = make_b_value(row, col);
        }
    }

    for (uint32_t row = 0; row < kMatmulM; ++row) {
        for (uint32_t col = 0; col < kMatmulN; ++col) {
            int32_t accum = 0;
            for (uint32_t k = 0; k < kMatmulK; ++k) {
                accum += static_cast<int32_t>(test_case.a[row * kMatmulK + k]) *
                         static_cast<int32_t>(test_case.b[k * kMatmulN + col]);
            }
            test_case.reference[row * kMatmulN + col] = accum;
        }
    }

    return test_case;
}

void load_a_matrix(const std::vector<int8_t>& a, FlatMemory& memory) {
    for (uint32_t row = 0; row < kMatmulM; ++row) {
        for (uint32_t k = 0; k < kMatmulK; k += 4) {
            const uint32_t packed = pack_int8x4(
                a[row * kMatmulK + k + 0],
                a[row * kMatmulK + k + 1],
                a[row * kMatmulK + k + 2],
                a[row * kMatmulK + k + 3]);
            const uint32_t addr = kABase + row * kABytesPerRow + k;
            memory.write32(addr, packed);
        }
    }
}

void load_b_tiles(const std::vector<int8_t>& b, FlatMemory& memory) {
    for (uint32_t tile = 0; tile < kMatmulN / kTileCols; ++tile) {
        for (uint32_t chunk = 0; chunk < kMatmulK / 4; ++chunk) {
            for (uint32_t lane = 0; lane < kTileCols; ++lane) {
                const uint32_t col = tile * kTileCols + lane;
                const uint32_t k = chunk * 4;
                const uint32_t packed = pack_int8x4(
                    b[(k + 0) * kMatmulN + col],
                    b[(k + 1) * kMatmulN + col],
                    b[(k + 2) * kMatmulN + col],
                    b[(k + 3) * kMatmulN + col]);
                const uint32_t addr = kBBase + tile * kBTileStrideBytes +
                                      chunk * kTileCols * sizeof(uint32_t) +
                                      lane * sizeof(uint32_t);
                memory.write32(addr, packed);
            }
        }
    }
}

void clear_c_matrix(FlatMemory& memory) {
    for (uint32_t row = 0; row < kMatmulM; ++row) {
        for (uint32_t col = 0; col < kMatmulN; ++col) {
            const uint32_t addr = kCBase + row * kCBytesPerRow + col * sizeof(uint32_t);
            memory.write32(addr, 0);
        }
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

bool verify_output(const FunctionalModel& model, const std::vector<int32_t>& reference) {
    uint32_t mismatch_count = 0;

    for (uint32_t row = 0; row < kMatmulM; ++row) {
        for (uint32_t col = 0; col < kMatmulN; ++col) {
            const uint32_t addr = kCBase + row * kCBytesPerRow + col * sizeof(uint32_t);
            const int32_t observed = static_cast<int32_t>(model.memory().read32(addr));
            const int32_t expected = reference[row * kMatmulN + col];
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

void print_summary(const Options& options, const TimingModel& timing, const Stats& stats) {
    const double macs = static_cast<double>(kMatmulM) * kMatmulN * kMatmulK;
    const double cycles = static_cast<double>(timing.cycle_count());
    const double macs_per_cycle = macs / cycles;

    std::cout << "Matmul kernel verified\n";
    std::cout << "  dimensions: " << kMatmulM << "x" << kMatmulN << "x" << kMatmulK << "\n";
    std::cout << "  resident warps: " << options.num_warps << "\n";
    std::cout << "  external memory latency: " << options.memory_latency << " cycles\n";
    std::cout << "  cycles: " << timing.cycle_count() << "\n";
    std::cout << "  issued instructions: " << stats.total_instructions_issued << "\n";
    std::cout << "  fetch skips: " << stats.fetch_skip_count << "\n";
    std::cout << "  frontend stall cycles: " << stats.scheduler_frontend_stall_cycles << "\n";
    std::cout << "  MACs/cycle: " << macs_per_cycle << "\n";
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

        SimConfig config;
        config.num_warps = options.num_warps;
        config.external_memory_latency_cycles = options.memory_latency;
        config.kernel_args[0] = kABase;
        config.kernel_args[1] = kBBase;
        config.kernel_args[2] = kCBase;
        config.kernel_args[3] = kMatmulM;
        config.validate();

        FunctionalModel model(config);
        config.start_pc = load_program(model, MATMUL_KERNEL_ELF);

        const MatmulCase test_case = build_case();
        load_a_matrix(test_case.a, model.memory());
        load_b_tiles(test_case.b, model.memory());
        clear_c_matrix(model.memory());
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

        if (!verify_output(model, test_case.reference)) {
            return 5;
        }

        print_summary(options, timing, stats);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "matmul_bench error: " << e.what() << "\n";
        return 1;
    }
}
