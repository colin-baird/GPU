#include "gpu_sim/config.h"
#include "gpu_sim/elf_loader.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/timing/timing_model.h"
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

constexpr uint32_t kTileCols = 32;
constexpr uint32_t kABaseDefault = 0x00002000;

struct Shape {
    uint32_t m = 128;
    uint32_t n = 128;
    uint32_t k = 128;

    uint32_t a_base = kABaseDefault;
    uint32_t b_base = 0;
    uint32_t c_base = 0;

    uint32_t a_bytes_per_row() const { return k; }
    uint32_t b_tile_stride_bytes() const {
        return (k / 4u) * kTileCols * static_cast<uint32_t>(sizeof(uint32_t));
    }
    uint32_t c_bytes_per_row() const {
        return n * static_cast<uint32_t>(sizeof(int32_t));
    }

    void recompute_bases() {
        const uint32_t a_bytes = m * k;
        const uint32_t b_bytes = k * n;
        auto align_up = [](uint32_t x, uint32_t align) {
            return (x + align - 1) & ~(align - 1);
        };
        b_base = align_up(a_base + a_bytes, 0x100);
        c_base = align_up(b_base + b_bytes, 0x100);
    }
};

struct Options {
    uint32_t num_warps = MAX_WARPS;
    uint32_t memory_latency = SimConfig{}.external_memory_latency_cycles;
    uint64_t max_cycles = 5000000;
    bool json_output = false;
    std::string memory_backend = "fixed";
    std::string dramsim3_config_path = "";
    std::string kernel_elf = MATMUL_KERNEL_ELF;
    Shape shape;
};

struct MatmulCase {
    std::vector<int8_t> a;
    std::vector<int8_t> b;
    std::vector<int32_t> reference;
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--num-warps=<1-" << MAX_WARPS << ">] [--memory-latency=<cycles>] [--max-cycles=<N>]\n"
              << "         [--memory-backend=<fixed|dramsim3>] [--dramsim3-config-path=<file.ini>]\n"
              << "         [--m=<rows>] [--n=<cols>] [--k=<reduction>] [--kernel-elf=<path>]\n";
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
        if (arg.rfind("--m=", 0) == 0) {
            options.shape.m = parse_u32(arg.substr(4), "m");
            continue;
        }
        if (arg.rfind("--n=", 0) == 0) {
            options.shape.n = parse_u32(arg.substr(4), "n");
            continue;
        }
        if (arg.rfind("--k=", 0) == 0) {
            options.shape.k = parse_u32(arg.substr(4), "k");
            continue;
        }
        if (arg.rfind("--kernel-elf=", 0) == 0) {
            options.kernel_elf = arg.substr(13);
            continue;
        }
        throw std::invalid_argument("unknown argument: " + arg);
    }

    if (options.shape.n % kTileCols != 0) {
        throw std::invalid_argument("--n must be a multiple of " +
                                    std::to_string(kTileCols));
    }
    if (options.shape.k % 16u != 0u) {
        throw std::invalid_argument("--k must be a multiple of 16");
    }
    options.shape.recompute_bases();

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

MatmulCase build_case(const Shape& shape) {
    MatmulCase test_case;
    test_case.a.resize(shape.m * shape.k);
    test_case.b.resize(shape.k * shape.n);
    test_case.reference.assign(shape.m * shape.n, 0);

    for (uint32_t row = 0; row < shape.m; ++row) {
        for (uint32_t col = 0; col < shape.k; ++col) {
            test_case.a[row * shape.k + col] = make_a_value(row, col);
        }
    }

    for (uint32_t row = 0; row < shape.k; ++row) {
        for (uint32_t col = 0; col < shape.n; ++col) {
            test_case.b[row * shape.n + col] = make_b_value(row, col);
        }
    }

    for (uint32_t row = 0; row < shape.m; ++row) {
        for (uint32_t col = 0; col < shape.n; ++col) {
            int32_t accum = 0;
            for (uint32_t k = 0; k < shape.k; ++k) {
                accum += static_cast<int32_t>(test_case.a[row * shape.k + k]) *
                         static_cast<int32_t>(test_case.b[k * shape.n + col]);
            }
            test_case.reference[row * shape.n + col] = accum;
        }
    }

    return test_case;
}

void load_a_matrix(const std::vector<int8_t>& a, const Shape& shape,
                   FlatMemory& memory) {
    for (uint32_t row = 0; row < shape.m; ++row) {
        for (uint32_t k = 0; k < shape.k; k += 4) {
            const uint32_t packed = pack_int8x4(
                a[row * shape.k + k + 0],
                a[row * shape.k + k + 1],
                a[row * shape.k + k + 2],
                a[row * shape.k + k + 3]);
            const uint32_t addr = shape.a_base + row * shape.a_bytes_per_row() + k;
            memory.write32(addr, packed);
        }
    }
}

void load_b_tiles(const std::vector<int8_t>& b, const Shape& shape,
                  FlatMemory& memory) {
    for (uint32_t tile = 0; tile < shape.n / kTileCols; ++tile) {
        for (uint32_t chunk = 0; chunk < shape.k / 4; ++chunk) {
            for (uint32_t lane = 0; lane < kTileCols; ++lane) {
                const uint32_t col = tile * kTileCols + lane;
                const uint32_t k = chunk * 4;
                const uint32_t packed = pack_int8x4(
                    b[(k + 0) * shape.n + col],
                    b[(k + 1) * shape.n + col],
                    b[(k + 2) * shape.n + col],
                    b[(k + 3) * shape.n + col]);
                const uint32_t addr = shape.b_base +
                                      tile * shape.b_tile_stride_bytes() +
                                      chunk * kTileCols * sizeof(uint32_t) +
                                      lane * sizeof(uint32_t);
                memory.write32(addr, packed);
            }
        }
    }
}

void clear_c_matrix(const Shape& shape, FlatMemory& memory) {
    for (uint32_t row = 0; row < shape.m; ++row) {
        for (uint32_t col = 0; col < shape.n; ++col) {
            const uint32_t addr = shape.c_base + row * shape.c_bytes_per_row() +
                                  col * sizeof(uint32_t);
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

bool verify_output(const FunctionalModel& model, const Shape& shape,
                   const std::vector<int32_t>& reference) {
    uint32_t mismatch_count = 0;

    for (uint32_t row = 0; row < shape.m; ++row) {
        for (uint32_t col = 0; col < shape.n; ++col) {
            const uint32_t addr = shape.c_base + row * shape.c_bytes_per_row() +
                                  col * sizeof(uint32_t);
            const int32_t observed = static_cast<int32_t>(model.memory().read32(addr));
            const int32_t expected = reference[row * shape.n + col];
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
    const Shape& s = options.shape;
    const double macs = static_cast<double>(s.m) * s.n * s.k;
    const double cycles = static_cast<double>(timing.cycle_count());
    const double macs_per_cycle = macs / cycles;

    std::cout << "Matmul kernel verified\n";
    std::cout << "  dimensions: " << s.m << "x" << s.n << "x" << s.k << "\n";
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
        config.memory_backend = options.memory_backend;
        config.dramsim3_config_path = options.dramsim3_config_path;
        config.kernel_args[0] = options.shape.a_base;
        config.kernel_args[1] = options.shape.b_base;
        config.kernel_args[2] = options.shape.c_base;
        config.kernel_args[3] = options.shape.m;
        config.validate();

        FunctionalModel model(config);
        config.start_pc = load_program(model, options.kernel_elf.c_str());

        const MatmulCase test_case = build_case(options.shape);
        load_a_matrix(test_case.a, options.shape, model.memory());
        load_b_tiles(test_case.b, options.shape, model.memory());
        clear_c_matrix(options.shape, model.memory());
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

        if (!verify_output(model, options.shape, test_case.reference)) {
            return 5;
        }

        if (options.json_output) {
            std::ostringstream ss;
            stats.report_json(ss, config.num_warps);
            std::string json = ss.str();
            auto pos = json.rfind('}');
            json.insert(pos, ",\n  \"benchmark\": \"matmul\""
                             ",\n  \"macs_per_cycle\": " +
                             ([&]{
                                 std::ostringstream v;
                                 v << std::setprecision(6)
                                   << (static_cast<double>(options.shape.m) *
                                       options.shape.n * options.shape.k)
                                      / static_cast<double>(timing.cycle_count());
                                 return v.str();
                             })());
            std::cout << json;
        } else {
            print_summary(options, timing, stats);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "matmul_bench error: " << e.what() << "\n";
        return 1;
    }
}
