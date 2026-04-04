#include <cstdlib>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_sim/config.h"
#include "gpu_sim/elf_loader.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/timing/timing_model.h"
#include "gpu_sim/types.h"

using namespace gpu_sim;

namespace {

constexpr uint32_t kEmbeddingWords = 16;
constexpr uint32_t kVocabSize = 256;
constexpr uint32_t kTokensPerWarp = 32;

constexpr uint32_t kEmbeddingBase = 0x00002000;
constexpr uint32_t kTokensBase = 0x00008000;
constexpr uint32_t kOutputBase = 0x0000C000;

struct Options {
    uint32_t num_warps = MAX_WARPS;
    uint32_t memory_latency = 100;
    uint64_t max_cycles = 5000000;
};

struct Workload {
    std::vector<uint32_t> tokens;
    std::vector<uint32_t> embeddings;
    std::vector<uint32_t> reference;
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

    if (options.num_warps == 0 || options.num_warps > MAX_WARPS) {
        throw std::invalid_argument("num-warps must be in [1, " + std::to_string(MAX_WARPS) + "]");
    }

    return options;
}

uint32_t mix_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

Workload build_workload(uint32_t token_count) {
    Workload workload;
    workload.tokens.resize(token_count);
    workload.embeddings.resize(kVocabSize * kEmbeddingWords);
    workload.reference.resize(token_count * kEmbeddingWords);

    for (uint32_t row = 0; row < kVocabSize; ++row) {
        for (uint32_t col = 0; col < kEmbeddingWords; ++col) {
            const uint32_t seed = mix_u32(row * 131u + col * 17u + 0x9e3779b9u);
            workload.embeddings[row * kEmbeddingWords + col] = seed;
        }
    }

    for (uint32_t token = 0; token < token_count; ++token) {
        const uint32_t token_id = mix_u32(token * 19u + 7u) % kVocabSize;
        workload.tokens[token] = token_id;
        for (uint32_t col = 0; col < kEmbeddingWords; ++col) {
            workload.reference[token * kEmbeddingWords + col] =
                workload.embeddings[token_id * kEmbeddingWords + col];
        }
    }

    return workload;
}

void load_words(uint32_t base_addr, const std::vector<uint32_t>& words, FlatMemory& memory) {
    for (uint32_t i = 0; i < words.size(); ++i) {
        memory.write32(base_addr + i * sizeof(uint32_t), words[i]);
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

bool verify_output(const FunctionalModel& model, const std::vector<uint32_t>& reference) {
    uint32_t mismatch_count = 0;

    for (uint32_t i = 0; i < reference.size(); ++i) {
        const uint32_t addr = kOutputBase + i * sizeof(uint32_t);
        const uint32_t observed = model.memory().read32(addr);
        const uint32_t expected = reference[i];
        if (observed == expected) {
            continue;
        }

        if (mismatch_count < 8) {
            std::cerr << "Mismatch at word " << i
                      << ": expected 0x" << std::hex << expected
                      << ", observed 0x" << observed << std::dec << "\n";
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
                   uint32_t token_count) {
    const double copied_bytes = static_cast<double>(token_count) * kEmbeddingWords * sizeof(uint32_t);
    const double cycles = static_cast<double>(timing.cycle_count());
    const double bytes_per_cycle = copied_bytes / cycles;

    std::cout << "Embedding gather kernel verified\n";
    std::cout << "  tokens: " << token_count << "\n";
    std::cout << "  embedding words/row: " << kEmbeddingWords << "\n";
    std::cout << "  resident warps: " << options.num_warps << "\n";
    std::cout << "  external memory latency: " << options.memory_latency << " cycles\n";
    std::cout << "  cycles: " << timing.cycle_count() << "\n";
    std::cout << "  issued instructions: " << stats.total_instructions_issued << "\n";
    std::cout << "  fetch skips: " << stats.fetch_skip_count << "\n";
    std::cout << "  frontend stall cycles: " << stats.scheduler_frontend_stall_cycles << "\n";
    std::cout << "  copied bytes/cycle: " << bytes_per_cycle << "\n";
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
        config.kernel_args[0] = kEmbeddingBase;
        config.kernel_args[1] = kTokensBase;
        config.kernel_args[2] = kOutputBase;
        config.kernel_args[3] = kEmbeddingWords;
        config.validate();

        FunctionalModel model(config);
        config.start_pc = load_program(model, EMBEDDING_GATHER_KERNEL_ELF);

        const uint32_t token_count = config.num_warps * kTokensPerWarp;
        const Workload workload = build_workload(token_count);
        load_words(kEmbeddingBase, workload.embeddings, model.memory());
        load_words(kTokensBase, workload.tokens, model.memory());
        load_words(kOutputBase, std::vector<uint32_t>(token_count * kEmbeddingWords, 0), model.memory());
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

        if (!verify_output(model, workload.reference)) {
            return 4;
        }

        print_summary(options, timing, stats, token_count);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "embedding_gather_bench error: " << e.what() << "\n";
        return 1;
    }
}
