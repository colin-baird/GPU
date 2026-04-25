#include "gpu_sim/config.h"
#include "gpu_sim/elf_loader.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/timing/timing_model.h"
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gpu_sim;

namespace {

constexpr uint32_t kRowsPerWarp = 32;
constexpr uint32_t kCols = 256;
constexpr uint32_t kPackedCols = kCols / 4;

constexpr uint32_t kMatrixBase = 0x00002000;

struct Options {
    uint32_t num_warps = MAX_WARPS;
    uint32_t memory_latency = 100;
    uint64_t max_cycles = 2000000;
    bool json_output = false;
    std::string memory_backend = "fixed";
    std::string dramsim3_config_path = "";
};

struct GemvCase {
    std::vector<int8_t> matrix;
    std::vector<int8_t> vector;
    std::vector<int32_t> reference;
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--num-warps=<1-" << MAX_WARPS << ">] [--memory-latency=<cycles>] [--max-cycles=<N>]\n"
              << "         [--memory-backend=<fixed|dramsim3>] [--dramsim3-config-path=<file.ini>]\n";
    std::cerr << "Defaults: --num-warps=" << MAX_WARPS
              << " --memory-latency=100 --max-cycles=2000000 --memory-backend=fixed\n";
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

int8_t make_matrix_value(uint32_t row, uint32_t col) {
    const uint32_t value = (row * 11u + col * 7u + 5u) % 21u;
    return static_cast<int8_t>(static_cast<int32_t>(value) - 10);
}

int8_t make_vector_value(uint32_t index) {
    const uint32_t value = (index * 13u + 3u) % 17u;
    return static_cast<int8_t>(static_cast<int32_t>(value) - 8);
}

uint32_t pack_int8x4(int8_t v0, int8_t v1, int8_t v2, int8_t v3) {
    return static_cast<uint32_t>(static_cast<uint8_t>(v0)) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v1)) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v2)) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v3)) << 24);
}

GemvCase build_case(uint32_t rows) {
    GemvCase test_case;
    test_case.matrix.resize(rows * kCols);
    test_case.vector.resize(kCols);
    test_case.reference.assign(rows, 0);

    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < kCols; ++col) {
            test_case.matrix[row * kCols + col] = make_matrix_value(row, col);
        }
    }

    for (uint32_t col = 0; col < kCols; ++col) {
        test_case.vector[col] = make_vector_value(col);
    }

    for (uint32_t row = 0; row < rows; ++row) {
        int32_t accum = 0;
        for (uint32_t col = 0; col < kCols; ++col) {
            accum += static_cast<int32_t>(test_case.matrix[row * kCols + col]) *
                     static_cast<int32_t>(test_case.vector[col]);
        }
        test_case.reference[row] = accum;
    }

    return test_case;
}

void load_matrix(const std::vector<int8_t>& matrix, uint32_t rows, FlatMemory& memory) {
    // Lane-major tile layout: within each warp's tile, packed words for all 32
    // lanes are stored contiguously per column chunk so that adjacent lanes are
    // 4 bytes apart and all 32 addresses fall within one 128-byte cache line.
    const uint32_t num_warps = rows / kRowsPerWarp;
    const uint32_t tile_stride = kPackedCols * kRowsPerWarp * sizeof(uint32_t);
    const uint32_t chunk_stride = kRowsPerWarp * sizeof(uint32_t);

    for (uint32_t warp = 0; warp < num_warps; ++warp) {
        for (uint32_t chunk = 0; chunk < kPackedCols; ++chunk) {
            for (uint32_t lane = 0; lane < kRowsPerWarp; ++lane) {
                const uint32_t row = warp * kRowsPerWarp + lane;
                const uint32_t col = chunk * 4;
                const uint32_t packed = pack_int8x4(
                    matrix[row * kCols + col + 0],
                    matrix[row * kCols + col + 1],
                    matrix[row * kCols + col + 2],
                    matrix[row * kCols + col + 3]);
                const uint32_t addr = kMatrixBase +
                                      warp * tile_stride +
                                      chunk * chunk_stride +
                                      lane * sizeof(uint32_t);
                memory.write32(addr, packed);
            }
        }
    }
}

void load_vector(const std::vector<int8_t>& vector, uint32_t vector_base, FlatMemory& memory) {
    for (uint32_t packed_col = 0; packed_col < kPackedCols; ++packed_col) {
        const uint32_t col = packed_col * 4;
        const uint32_t packed = pack_int8x4(
            vector[col + 0],
            vector[col + 1],
            vector[col + 2],
            vector[col + 3]);
        memory.write32(vector_base + col, packed);
    }
}

void clear_output(uint32_t rows, uint32_t output_base, FlatMemory& memory) {
    for (uint32_t row = 0; row < rows; ++row) {
        memory.write32(output_base + row * sizeof(uint32_t), 0);
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
                   const std::vector<int32_t>& reference) {
    uint32_t mismatch_count = 0;

    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t addr = output_base + row * sizeof(uint32_t);
        const int32_t observed = static_cast<int32_t>(model.memory().read32(addr));
        const int32_t expected = reference[row];
        if (observed == expected) {
            continue;
        }

        if (mismatch_count < 8) {
            std::cerr << "Mismatch at row " << row
                      << ": expected " << expected
                      << ", observed " << observed << "\n";
        }
        ++mismatch_count;
    }

    if (mismatch_count == 0) {
        return true;
    }

    std::cerr << "Total mismatches: " << mismatch_count << "\n";
    return false;
}

void print_summary(const Options& options, uint32_t rows, const TimingModel& timing, const Stats& stats) {
    const double macs = static_cast<double>(rows) * static_cast<double>(kCols);
    const double cycles = static_cast<double>(timing.cycle_count());
    const double macs_per_cycle = macs / cycles;
    const double outputs_per_cycle = static_cast<double>(rows) / cycles;

    std::cout << "GEMV kernel verified\n";
    std::cout << "  dimensions: " << rows << "x" << kCols << "\n";
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
    std::cout << "  MACs/cycle: " << macs_per_cycle << "\n";
    std::cout << "  outputs/cycle: " << outputs_per_cycle << "\n";
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
        const uint32_t rows = options.num_warps * kRowsPerWarp;

        const uint32_t matrix_bytes = rows * kCols;
        const uint32_t vector_base = (kMatrixBase + matrix_bytes + 0xFFFu) & ~0xFFFu;
        const uint32_t output_base = (vector_base + kCols + 0xFFFu) & ~0xFFFu;

        SimConfig config;
        config.num_warps = options.num_warps;
        config.external_memory_latency_cycles = options.memory_latency;
        config.memory_backend = options.memory_backend;
        config.dramsim3_config_path = options.dramsim3_config_path;
        config.kernel_args[0] = kMatrixBase;
        config.kernel_args[1] = vector_base;
        config.kernel_args[2] = output_base;
        config.kernel_args[3] = rows;
        config.validate();

        FunctionalModel model(config);
        config.start_pc = load_program(model, GEMV_KERNEL_ELF);

        const GemvCase test_case = build_case(rows);
        load_matrix(test_case.matrix, rows, model.memory());
        load_vector(test_case.vector, vector_base, model.memory());
        clear_output(rows, output_base, model.memory());
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

        if (!verify_output(model, rows, output_base, test_case.reference)) {
            return 4;
        }

        if (options.json_output) {
            std::ostringstream ss;
            stats.report_json(ss, config.num_warps);
            std::string json = ss.str();
            const double cycles = static_cast<double>(timing.cycle_count());
            const double macs = static_cast<double>(rows) * static_cast<double>(kCols);
            auto pos = json.rfind('}');
            std::ostringstream derived;
            derived << std::setprecision(6)
                    << ",\n  \"benchmark\": \"gemv\""
                    << ",\n  \"macs_per_cycle\": " << macs / cycles
                    << ",\n  \"outputs_per_cycle\": " << static_cast<double>(rows) / cycles;
            json.insert(pos, derived.str());
            std::cout << json;
        } else {
            print_summary(options, rows, timing, stats);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "gemv_bench error: " << e.what() << "\n";
        return 1;
    }
}
