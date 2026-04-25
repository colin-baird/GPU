#include "gpu_sim/config.h"
#include "gpu_sim/elf_loader.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/timing/timing_model.h"
#include <cmath>
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

#ifndef FUSED_LINEAR_ACTIVATION_KERNEL_ELF
#define FUSED_LINEAR_ACTIVATION_KERNEL_ELF ""
#endif

namespace {

constexpr uint32_t kInputLength = 128;
constexpr uint32_t kTileOutputs = 32;
constexpr uint32_t kInputWords = kInputLength / 4;
constexpr uint32_t kTileStrideBytes = kInputWords * kTileOutputs * sizeof(uint32_t);
constexpr uint32_t kActivationBase = 512;
constexpr uint32_t kActivationEntries = 256;
constexpr uint32_t kActivationScale = 256;
constexpr int32_t kIndexBias = 128;
constexpr int32_t kIndexShift = 5;

constexpr uint32_t kInputBase = 0x00002000;

struct Options {
    uint32_t num_warps = MAX_WARPS;
    uint32_t memory_latency = 100;
    uint64_t max_cycles = 5000000;
    bool json_output = false;
    std::string memory_backend = "fixed";
    std::string dramsim3_config_path = "";
};

struct WorkloadCase {
    std::vector<int8_t> input;
    std::vector<int8_t> weights;
    std::vector<int32_t> lookup_table;
    std::vector<int32_t> reference;
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--num-warps=<1-" << MAX_WARPS << ">] [--memory-latency=<cycles>] [--max-cycles=<N>]\n"
              << "         [--memory-backend=<fixed|dramsim3>] [--dramsim3-config-path=<file.ini>]\n";
    std::cerr << "Defaults: --num-warps=" << MAX_WARPS
              << " --memory-latency=100 --max-cycles=5000000 --memory-backend=fixed\n";
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

uint32_t pack_int8x4(int8_t v0, int8_t v1, int8_t v2, int8_t v3) {
    return static_cast<uint32_t>(static_cast<uint8_t>(v0)) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v1)) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v2)) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v3)) << 24);
}

int8_t make_input_value(uint32_t index) {
    const uint32_t value = (index * 7u + 3u) % 9u;
    return static_cast<int8_t>(static_cast<int32_t>(value) - 4);
}

int8_t make_weight_value(uint32_t output_index, uint32_t input_index) {
    const uint32_t value = (output_index * 5u + input_index * 11u + 1u) % 9u;
    return static_cast<int8_t>(static_cast<int32_t>(value) - 4);
}

int32_t arithmetic_shift_right(int32_t value, uint32_t shift) {
    return value >> shift;
}

int32_t silu_q8_from_index(int32_t index) {
    const double x = static_cast<double>(index - kIndexBias) / 32.0;
    const double silu = x / (1.0 + std::exp(-x));
    return static_cast<int32_t>(std::lround(silu * static_cast<double>(kActivationScale)));
}

WorkloadCase build_case(uint32_t output_count) {
    WorkloadCase test_case;
    test_case.input.resize(kInputLength);
    test_case.weights.resize(output_count * kInputLength);
    test_case.lookup_table.assign(1024, 0);
    test_case.reference.assign(output_count, 0);

    for (uint32_t i = 0; i < kInputLength; ++i) {
        test_case.input[i] = make_input_value(i);
    }

    for (uint32_t out = 0; out < output_count; ++out) {
        for (uint32_t in = 0; in < kInputLength; ++in) {
            test_case.weights[out * kInputLength + in] = make_weight_value(out, in);
        }
    }

    for (uint32_t i = 0; i < kActivationEntries; ++i) {
        test_case.lookup_table[kActivationBase + i] = silu_q8_from_index(static_cast<int32_t>(i));
    }

    for (uint32_t out = 0; out < output_count; ++out) {
        int32_t accum = 0;
        for (uint32_t in = 0; in < kInputLength; ++in) {
            accum += static_cast<int32_t>(test_case.input[in]) *
                     static_cast<int32_t>(test_case.weights[out * kInputLength + in]);
        }
        const int32_t index = arithmetic_shift_right(accum, kIndexShift) + kIndexBias;
        test_case.reference[out] = test_case.lookup_table[kActivationBase + static_cast<uint32_t>(index)];
    }

    return test_case;
}

void load_input_vector(const std::vector<int8_t>& input, FlatMemory& memory) {
    for (uint32_t chunk = 0; chunk < kInputWords; ++chunk) {
        const uint32_t base = chunk * 4;
        const uint32_t packed = pack_int8x4(input[base + 0],
                                            input[base + 1],
                                            input[base + 2],
                                            input[base + 3]);
        memory.write32(kInputBase + chunk * sizeof(uint32_t), packed);
    }
}

void load_weight_tiles(const std::vector<int8_t>& weights, uint32_t output_count,
                       uint32_t weights_base, FlatMemory& memory) {
    const uint32_t tile_count = output_count / kTileOutputs;

    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        for (uint32_t chunk = 0; chunk < kInputWords / 4; ++chunk) {
            for (uint32_t lane = 0; lane < kTileOutputs; ++lane) {
                const uint32_t output_index = tile * kTileOutputs + lane;
                const uint32_t input_base = chunk * 16;
                const uint32_t chunk_base = weights_base +
                                            tile * kTileStrideBytes +
                                            chunk * 4 * kTileOutputs * sizeof(uint32_t) +
                                            lane * sizeof(uint32_t);

                // Match the kernel's lane-major tile layout: four packed words per chunk,
                // spaced one 32-lane stride apart so each lane can stream them with fixed offsets.
                for (uint32_t pack = 0; pack < 4; ++pack) {
                    const uint32_t pack_base = input_base + pack * 4;
                    const uint32_t packed = pack_int8x4(
                        weights[output_index * kInputLength + pack_base + 0],
                        weights[output_index * kInputLength + pack_base + 1],
                        weights[output_index * kInputLength + pack_base + 2],
                        weights[output_index * kInputLength + pack_base + 3]);
                    memory.write32(chunk_base + pack * kTileOutputs * sizeof(uint32_t), packed);
                }
            }
        }
    }
}

void clear_output(const uint32_t output_count, uint32_t output_base, FlatMemory& memory) {
    for (uint32_t out = 0; out < output_count; ++out) {
        memory.write32(output_base + out * sizeof(uint32_t), 0);
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

bool verify_output(const FunctionalModel& model, uint32_t output_base,
                   const std::vector<int32_t>& reference) {
    uint32_t mismatch_count = 0;

    for (uint32_t out = 0; out < reference.size(); ++out) {
        const uint32_t addr = output_base + out * sizeof(uint32_t);
        const int32_t observed = static_cast<int32_t>(model.memory().read32(addr));
        const int32_t expected = reference[out];
        if (observed == expected) {
            continue;
        }

        if (mismatch_count < 8) {
            std::cerr << "Mismatch at output " << out
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

void print_summary(const Options& options, const TimingModel& timing, const Stats& stats,
                   uint32_t output_count) {
    const double macs = static_cast<double>(output_count) * static_cast<double>(kInputLength);
    const double cycles = static_cast<double>(timing.cycle_count());
    const double macs_per_cycle = macs / cycles;
    const double outputs_per_cycle = static_cast<double>(output_count) / cycles;

    std::cout << "Fused linear + activation kernel verified\n";
    std::cout << "  outputs: " << output_count << "\n";
    std::cout << "  input length: " << kInputLength << "\n";
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
        const uint32_t output_count = options.num_warps * kTileOutputs;

        const uint32_t input_bytes = kInputWords * sizeof(uint32_t);
        const uint32_t weights_base = (kInputBase + input_bytes + 0xFFFu) & ~0xFFFu;
        const uint32_t tile_count = output_count / kTileOutputs;
        const uint32_t weights_bytes = tile_count * kTileStrideBytes;
        const uint32_t output_base = (weights_base + weights_bytes + 0xFFFu) & ~0xFFFu;

        SimConfig config;
        config.num_warps = options.num_warps;
        config.external_memory_latency_cycles = options.memory_latency;
        config.memory_backend = options.memory_backend;
        config.dramsim3_config_path = options.dramsim3_config_path;
        config.kernel_args[0] = kInputBase;
        config.kernel_args[1] = weights_base;
        config.kernel_args[2] = output_base;
        config.kernel_args[3] = output_count;
        config.validate();

        FunctionalModel model(config);
        config.start_pc = load_program(model, FUSED_LINEAR_ACTIVATION_KERNEL_ELF);

        const WorkloadCase test_case = build_case(output_count);
        load_input_vector(test_case.input, model.memory());
        load_weight_tiles(test_case.weights, output_count, weights_base, model.memory());
        clear_output(output_count, output_base, model.memory());
        model.init_kernel(config);
        for (uint32_t i = 0; i < test_case.lookup_table.size(); ++i) {
            model.lookup_table().write(i, static_cast<uint32_t>(test_case.lookup_table[i]));
        }

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

        if (!verify_output(model, output_base, test_case.reference)) {
            return 4;
        }

        if (options.json_output) {
            std::ostringstream ss;
            stats.report_json(ss, config.num_warps);
            std::string json = ss.str();
            const double cycles = static_cast<double>(timing.cycle_count());
            const double macs = static_cast<double>(output_count) * static_cast<double>(kInputLength);
            auto pos = json.rfind('}');
            std::ostringstream derived;
            derived << std::setprecision(6)
                    << ",\n  \"benchmark\": \"fused_linear_activation\""
                    << ",\n  \"macs_per_cycle\": " << macs / cycles
                    << ",\n  \"outputs_per_cycle\": " << static_cast<double>(output_count) / cycles;
            json.insert(pos, derived.str());
            std::cout << json;
        } else {
            print_summary(options, timing, stats, output_count);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fused_linear_activation_bench error: " << e.what() << "\n";
        return 1;
    }
}
