#include "catch.hpp"
#include "gpu_sim/config.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/isa.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/timing/timing_model.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gpu_sim;

namespace {

enum class CompareOp {
    kEqual,
    kAtLeast,
    kAtMost,
};

struct NumericExpectation {
    CompareOp op = CompareOp::kEqual;
    uint64_t value = 0;
};

struct ManifestExpectation {
    std::string key;
    std::string raw_value;
    CompareOp op = CompareOp::kEqual;
};

struct AlignmentManifest {
    std::filesystem::path path;
    std::string scenario_name;
    std::string builder_name;
    std::map<std::string, std::string> config_overrides;
    std::vector<ManifestExpectation> expectations;
    std::vector<std::string> citations;
};

struct ScenarioProgram {
    std::vector<uint32_t> instructions;
    std::function<void(const SimConfig&, FunctionalModel&)> initialize;
};

struct ExecutedScenario {
    SimConfig config;
    Stats stats;
    std::vector<CycleTraceSnapshot> snapshots;
    std::unique_ptr<FunctionalModel> model;
};

std::string trim(const std::string& input) {
    size_t start = 0;
    while (start < input.size() &&
           std::isspace(static_cast<unsigned char>(input[start])) != 0) {
        start++;
    }

    size_t end = input.size();
    while (end > start &&
           std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
        end--;
    }

    return input.substr(start, end - start);
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

uint64_t parse_uint64(const std::string& value) {
    return std::stoull(value, nullptr, 0);
}

bool parse_bool(const std::string& value) {
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    throw std::runtime_error("expected true/false, got '" + value + "'");
}

CompareOp parse_compare_op(std::string& value) {
    if (starts_with(value, ">=")) {
        value = trim(value.substr(2));
        return CompareOp::kAtLeast;
    }
    if (starts_with(value, "<=")) {
        value = trim(value.substr(2));
        return CompareOp::kAtMost;
    }
    return CompareOp::kEqual;
}

std::vector<std::string> split(const std::string& value, char delim) {
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, delim)) {
        parts.push_back(item);
    }
    return parts;
}

void apply_config_override(SimConfig& config, const std::string& key, const std::string& value) {
    if (key == "num_warps") {
        config.num_warps = static_cast<uint32_t>(parse_uint64(value));
    } else if (key == "instruction_buffer_depth") {
        config.instruction_buffer_depth = static_cast<uint32_t>(parse_uint64(value));
    } else if (key == "multiply_pipeline_stages") {
        config.multiply_pipeline_stages = static_cast<uint32_t>(parse_uint64(value));
    } else if (key == "num_ldst_units") {
        config.num_ldst_units = static_cast<uint32_t>(parse_uint64(value));
    } else if (key == "addr_gen_fifo_depth") {
        config.addr_gen_fifo_depth = static_cast<uint32_t>(parse_uint64(value));
    } else if (key == "l1_cache_size_bytes") {
        config.l1_cache_size_bytes = static_cast<uint32_t>(parse_uint64(value));
    } else if (key == "cache_line_size_bytes") {
        config.cache_line_size_bytes = static_cast<uint32_t>(parse_uint64(value));
    } else if (key == "num_mshrs") {
        config.num_mshrs = static_cast<uint32_t>(parse_uint64(value));
    } else if (key == "write_buffer_depth") {
        config.write_buffer_depth = static_cast<uint32_t>(parse_uint64(value));
    } else if (key == "external_memory_latency_cycles") {
        config.external_memory_latency_cycles = static_cast<uint32_t>(parse_uint64(value));
    } else if (key == "start_pc") {
        config.start_pc = static_cast<uint32_t>(parse_uint64(value));
    } else if (key == "arg0") {
        config.kernel_args[0] = static_cast<uint32_t>(parse_uint64(value));
    } else if (key == "arg1") {
        config.kernel_args[1] = static_cast<uint32_t>(parse_uint64(value));
    } else if (key == "arg2") {
        config.kernel_args[2] = static_cast<uint32_t>(parse_uint64(value));
    } else if (key == "arg3") {
        config.kernel_args[3] = static_cast<uint32_t>(parse_uint64(value));
    } else {
        throw std::runtime_error("unknown config override '" + key + "'");
    }
}

AlignmentManifest parse_manifest(const std::filesystem::path& path) {
    AlignmentManifest manifest;
    manifest.path = path;

    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open manifest: " + path.string());
    }

    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        line_number++;
        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        const size_t eq = trimmed.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("invalid manifest line " + std::to_string(line_number) +
                                     " in " + path.string());
        }

        const std::string key = trim(trimmed.substr(0, eq));
        std::string value = trim(trimmed.substr(eq + 1));

        if (key == "scenario") {
            manifest.scenario_name = value;
            continue;
        }
        if (key == "builder") {
            manifest.builder_name = value;
            continue;
        }
        if (key == "citation") {
            manifest.citations.push_back(value);
            continue;
        }
        if (starts_with(key, "config.")) {
            manifest.config_overrides[key.substr(7)] = value;
            continue;
        }

        ManifestExpectation expectation;
        expectation.key = key;
        expectation.op = parse_compare_op(value);
        expectation.raw_value = value;
        manifest.expectations.push_back(expectation);
    }

    if (manifest.scenario_name.empty()) {
        throw std::runtime_error("manifest missing scenario name: " + path.string());
    }
    if (manifest.builder_name.empty()) {
        throw std::runtime_error("manifest missing builder name: " + path.string());
    }

    return manifest;
}

std::vector<AlignmentManifest> load_manifests(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".manifest") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());

    std::vector<AlignmentManifest> manifests;
    manifests.reserve(paths.size());
    for (const auto& path : paths) {
        manifests.push_back(parse_manifest(path));
    }
    return manifests;
}

uint32_t i_type(int32_t imm, uint32_t rs1, uint32_t funct3, uint32_t rd, uint32_t opcode) {
    return (static_cast<uint32_t>(imm & 0xFFF) << 20) | (rs1 << 15) |
           (funct3 << 12) | (rd << 7) | opcode;
}

uint32_t r_type(uint32_t funct7, uint32_t rs2, uint32_t rs1, uint32_t funct3, uint32_t rd,
                uint32_t opcode) {
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) |
           (funct3 << 12) | (rd << 7) | opcode;
}

uint32_t s_type(int32_t imm, uint32_t rs2, uint32_t rs1, uint32_t funct3, uint32_t opcode) {
    const uint32_t imm_hi = (imm >> 5) & 0x7F;
    const uint32_t imm_lo = imm & 0x1F;
    return (imm_hi << 25) | (rs2 << 20) | (rs1 << 15) |
           (funct3 << 12) | (imm_lo << 7) | opcode;
}

uint32_t b_type(int32_t imm, uint32_t rs2, uint32_t rs1, uint32_t funct3, uint32_t opcode) {
    const uint32_t bit12 = (imm >> 12) & 1;
    const uint32_t bit11 = (imm >> 11) & 1;
    const uint32_t bits10_5 = (imm >> 5) & 0x3F;
    const uint32_t bits4_1 = (imm >> 1) & 0xF;
    return (bit12 << 31) | (bits10_5 << 25) | (rs2 << 20) | (rs1 << 15) |
           (funct3 << 12) | (bits4_1 << 8) | (bit11 << 7) | opcode;
}

uint32_t j_type(int32_t imm, uint32_t rd, uint32_t opcode) {
    const uint32_t bit20 = (imm >> 20) & 1;
    const uint32_t bits10_1 = (imm >> 1) & 0x3FF;
    const uint32_t bit11 = (imm >> 11) & 1;
    const uint32_t bits19_12 = (imm >> 12) & 0xFF;
    return (bit20 << 31) | (bits10_1 << 21) | (bit11 << 20) |
           (bits19_12 << 12) | (rd << 7) | opcode;
}

uint32_t encode_addi(uint32_t rd, uint32_t rs1, int32_t imm) {
    return i_type(imm, rs1, isa::FUNCT3_ADD_SUB, rd, isa::OP_ALU_I);
}

uint32_t encode_slli(uint32_t rd, uint32_t rs1, uint32_t shamt) {
    return i_type(static_cast<int32_t>(shamt), rs1, isa::FUNCT3_SLL, rd, isa::OP_ALU_I);
}

uint32_t encode_add(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    return r_type(isa::FUNCT7_BASE, rs2, rs1, isa::FUNCT3_ADD_SUB, rd, isa::OP_ALU_R);
}

uint32_t encode_mul(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    return r_type(isa::FUNCT7_MULDIV, rs2, rs1, isa::FUNCT3_MUL, rd, isa::OP_ALU_R);
}

uint32_t encode_div(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    return r_type(isa::FUNCT7_MULDIV, rs2, rs1, isa::FUNCT3_DIV, rd, isa::OP_ALU_R);
}

uint32_t encode_lw(uint32_t rd, uint32_t rs1, int32_t imm) {
    return i_type(imm, rs1, isa::FUNCT3_LW, rd, isa::OP_LOAD);
}

uint32_t encode_sw(uint32_t rs2, uint32_t rs1, int32_t imm) {
    return s_type(imm, rs2, rs1, isa::FUNCT3_SW, isa::OP_STORE);
}

uint32_t encode_beq(uint32_t rs1, uint32_t rs2, int32_t imm) {
    return b_type(imm, rs2, rs1, isa::FUNCT3_BEQ, isa::OP_BRANCH);
}

uint32_t encode_jal(uint32_t rd, int32_t imm) {
    return j_type(imm, rd, isa::OP_JAL);
}

uint32_t encode_fence() {
    return i_type(0, 0, 0, 0, isa::OP_FENCE);
}

uint32_t encode_ebreak() {
    return i_type(1, 0, 0, 0, isa::OP_SYSTEM);
}

uint32_t encode_ecall() {
    return i_type(0, 0, 0, 0, isa::OP_SYSTEM);
}

uint32_t encode_csrrs(uint32_t rd, uint16_t csr_addr, uint32_t rs1) {
    return (static_cast<uint32_t>(csr_addr) << 20) | (rs1 << 15) |
           (isa::FUNCT3_CSRRS << 12) | (rd << 7) | isa::OP_SYSTEM;
}

std::map<std::string, std::function<ScenarioProgram()>> build_programs() {
    std::map<std::string, std::function<ScenarioProgram()>> builders;

    builders["simple_pipeline"] = []() {
        ScenarioProgram program;
        program.instructions = {
            encode_fence(),
            encode_addi(5, 0, 1),
            encode_ecall(),
        };
        return program;
    };

    builders["alu_chain"] = []() {
        ScenarioProgram program;
        program.instructions = {
            encode_addi(5, 0, 1),
            encode_addi(6, 5, 1),
            encode_addi(7, 6, 1),
            encode_ecall(),
        };
        return program;
    };

    builders["mul_dependency"] = []() {
        ScenarioProgram program;
        program.instructions = {
            encode_mul(5, 1, 2),
            encode_addi(6, 5, 1),
            encode_ecall(),
        };
        return program;
    };

    builders["div_dependency"] = []() {
        ScenarioProgram program;
        program.instructions = {
            encode_div(5, 1, 2),
            encode_addi(6, 5, 1),
            encode_ecall(),
        };
        return program;
    };

    builders["branch_taken"] = []() {
        ScenarioProgram program;
        program.instructions = {
            encode_addi(5, 0, 1),
            encode_beq(0, 0, 8),
            encode_addi(6, 0, 99),
            encode_addi(7, 0, 42),
            encode_ecall(),
        };
        return program;
    };

    builders["jal_predicted_taken"] = []() {
        ScenarioProgram program;
        program.instructions = {
            encode_jal(5, 8),
            encode_addi(6, 0, 99),
            encode_addi(7, 0, 42),
            encode_ecall(),
        };
        return program;
    };

    builders["load_miss_use"] = []() {
        ScenarioProgram program;
        program.instructions = {
            encode_lw(5, 1, 0),
            encode_addi(6, 5, 1),
            encode_ecall(),
        };
        program.initialize = [](const SimConfig&, FunctionalModel& model) {
            model.memory().write32(0x1000, 41);
        };
        return program;
    };

    builders["store_then_load_same_line"] = []() {
        ScenarioProgram program;
        program.instructions = {
            encode_addi(5, 0, 42),
            encode_sw(5, 1, 0),
            encode_lw(6, 1, 0),
            encode_ecall(),
        };
        return program;
    };

    builders["writeback_conflict"] = []() {
        ScenarioProgram program;
        program.instructions = {
            encode_lw(5, 1, 0),
            encode_mul(6, 2, 3),
            encode_ecall(),
        };
        program.initialize = [](const SimConfig&, FunctionalModel& model) {
            model.memory().write32(0x1000, 41);
        };
        return program;
    };

    builders["panic_drain"] = []() {
        ScenarioProgram program;
        program.instructions = {
            encode_addi(31, 0, 42),
            encode_addi(0, 0, 0),
            encode_ebreak(),
        };
        return program;
    };

    builders["serialized_load"] = []() {
        ScenarioProgram program;
        program.instructions = {
            encode_lw(5, 1, 0),
            encode_ecall(),
        };
        program.initialize = [](const SimConfig&, FunctionalModel& model) {
            for (LaneId lane = 0; lane < WARP_SIZE; ++lane) {
                const uint32_t addr = 0x1000 + lane * 128;
                model.register_file().write(0, lane, 1, addr);
                model.memory().write32(addr, 100 + lane);
            }
        };
        return program;
    };

    builders["four_warp_round_robin"] = []() {
        ScenarioProgram program;
        program.instructions = {
            encode_addi(5, 0, 1),
            encode_ecall(),
        };
        return program;
    };

    builders["four_warp_mshr_pressure"] = []() {
        ScenarioProgram program;
        program.instructions = {
            encode_csrrs(5, isa::CSR_WARP_ID, 0),
            encode_slli(5, 5, 7),
            encode_add(6, 1, 5),
            encode_lw(7, 6, 0),
            encode_ecall(),
        };
        program.initialize = [](const SimConfig& config, FunctionalModel& model) {
            for (uint32_t warp = 0; warp < config.num_warps; ++warp) {
                const uint32_t addr = 0x1000 + warp * 128;
                model.memory().write32(addr, 1000 + warp);
            }
        };
        return program;
    };

    return builders;
}

ExecutedScenario run_manifest(const AlignmentManifest& manifest) {
    static const auto builders = build_programs();

    const auto builder_it = builders.find(manifest.builder_name);
    if (builder_it == builders.end()) {
        throw std::runtime_error("unknown scenario builder '" + manifest.builder_name + "'");
    }

    SimConfig config;
    for (const auto& [key, value] : manifest.config_overrides) {
        apply_config_override(config, key, value);
    }
    config.validate();

    ScenarioProgram program = builder_it->second();
    auto model = std::make_unique<FunctionalModel>(config);
    for (size_t i = 0; i < program.instructions.size(); ++i) {
        model->instruction_memory().write(static_cast<uint32_t>(i), program.instructions[i]);
    }
    model->init_kernel(config);
    if (program.initialize) {
        program.initialize(config, *model);
    }

    ExecutedScenario executed;
    executed.config = config;
    executed.model = std::move(model);

    TimingModel timing(config, *executed.model, executed.stats);
    bool keep_running = true;
    while (keep_running) {
        keep_running = timing.tick();
        REQUIRE(timing.last_cycle_snapshot().has_value());
        executed.snapshots.push_back(*timing.last_cycle_snapshot());
    }

    return executed;
}

const CycleTraceSnapshot& snapshot_for_cycle(const ExecutedScenario& executed, uint64_t cycle) {
    for (const auto& snapshot : executed.snapshots) {
        if (snapshot.cycle == cycle) {
            return snapshot;
        }
    }
    throw std::runtime_error("missing snapshot for cycle " + std::to_string(cycle));
}

uint64_t stat_value(const ExecutedScenario& executed, const std::string& key) {
    const Stats& stats = executed.stats;
    if (key == "total_cycles") return stats.total_cycles;
    if (key == "total_instructions_issued") return stats.total_instructions_issued;
    if (key == "fetch_skip_count") return stats.fetch_skip_count;
    if (key == "scheduler_idle_cycles") return stats.scheduler_idle_cycles;
    if (key == "operand_collector_busy_cycles") return stats.operand_collector_busy_cycles;
    if (key == "branch_predictions") return stats.branch_predictions;
    if (key == "branch_mispredictions") return stats.branch_mispredictions;
    if (key == "branch_flushes") return stats.branch_flushes;
    if (key == "alu_instructions") return stats.alu_stats.instructions;
    if (key == "mul_instructions") return stats.mul_stats.instructions;
    if (key == "div_instructions") return stats.div_stats.instructions;
    if (key == "ldst_instructions") return stats.ldst_stats.instructions;
    if (key == "tlookup_instructions") return stats.tlookup_stats.instructions;
    if (key == "cache_hits") return stats.cache_hits;
    if (key == "cache_misses") return stats.cache_misses;
    if (key == "load_hits") return stats.load_hits;
    if (key == "load_misses") return stats.load_misses;
    if (key == "store_hits") return stats.store_hits;
    if (key == "store_misses") return stats.store_misses;
    if (key == "mshr_stall_cycles") return stats.mshr_stall_cycles;
    if (key == "write_buffer_stall_cycles") return stats.write_buffer_stall_cycles;
    if (key == "coalesced_requests") return stats.coalesced_requests;
    if (key == "serialized_requests") return stats.serialized_requests;
    if (key == "external_memory_reads") return stats.external_memory_reads;
    if (key == "external_memory_writes") return stats.external_memory_writes;
    if (key == "writeback_conflicts") return stats.writeback_conflicts;
    throw std::runtime_error("unknown stat field '" + key + "'");
}

uint64_t warp_value(const ExecutedScenario& executed, uint32_t warp, const std::string& key) {
    const Stats& stats = executed.stats;
    if (key == "instructions") return stats.warp_instructions[warp];
    if (key == "cycles_active") return stats.warp_cycles_active[warp];
    if (key == "stall_scoreboard") return stats.warp_stall_scoreboard[warp];
    if (key == "stall_buffer_empty") return stats.warp_stall_buffer_empty[warp];
    if (key == "stall_unit_busy") return stats.warp_stall_unit_busy[warp];
    throw std::runtime_error("unknown warp field '" + key + "'");
}

void compare_numeric(const ManifestExpectation& expectation, uint64_t actual) {
    const uint64_t expected = parse_uint64(expectation.raw_value);
    INFO("Expectation key: " << expectation.key);
    INFO("Expected value: " << expectation.raw_value);
    INFO("Actual value: " << actual);

    switch (expectation.op) {
        case CompareOp::kEqual:
            REQUIRE(actual == expected);
            break;
        case CompareOp::kAtLeast:
            REQUIRE(actual >= expected);
            break;
        case CompareOp::kAtMost:
            REQUIRE(actual <= expected);
            break;
    }
}

void compare_bool(const ManifestExpectation& expectation, bool actual) {
    if (expectation.op != CompareOp::kEqual) {
        throw std::runtime_error("boolean expectations require exact equality");
    }
    const bool expected = parse_bool(expectation.raw_value);
    INFO("Expectation key: " << expectation.key);
    INFO("Expected value: " << expectation.raw_value);
    INFO("Actual value: " << (actual ? "true" : "false"));
    REQUIRE(actual == expected);
}

void compare_string(const ManifestExpectation& expectation, const std::string& actual) {
    if (expectation.op != CompareOp::kEqual) {
        throw std::runtime_error("string expectations require exact equality");
    }
    INFO("Expectation key: " << expectation.key);
    INFO("Expected value: " << expectation.raw_value);
    INFO("Actual value: " << actual);
    REQUIRE(actual == expectation.raw_value);
}

void check_expectation(const ExecutedScenario& executed, const ManifestExpectation& expectation) {
    if (starts_with(expectation.key, "stat.")) {
        compare_numeric(expectation, stat_value(executed, expectation.key.substr(5)));
        return;
    }

    if (starts_with(expectation.key, "warp.")) {
        const auto parts = split(expectation.key, '.');
        if (parts.size() != 3) {
            throw std::runtime_error("invalid warp expectation key '" + expectation.key + "'");
        }
        const uint32_t warp = static_cast<uint32_t>(parse_uint64(parts[1]));
        compare_numeric(expectation, warp_value(executed, warp, parts[2]));
        return;
    }

    if (starts_with(expectation.key, "reg.")) {
        const auto parts = split(expectation.key, '.');
        if (parts.size() != 4) {
            throw std::runtime_error("invalid reg expectation key '" + expectation.key + "'");
        }
        const uint32_t warp = static_cast<uint32_t>(parse_uint64(parts[1]));
        const uint32_t lane = static_cast<uint32_t>(parse_uint64(parts[2]));
        const uint32_t reg = static_cast<uint32_t>(parse_uint64(parts[3]));
        compare_numeric(expectation, executed.model->register_file().read(warp, lane, reg));
        return;
    }

    if (starts_with(expectation.key, "panic.")) {
        const std::string field = expectation.key.substr(6);
        if (field == "panicked") {
            compare_bool(expectation, executed.model->is_panicked());
            return;
        }
        if (field == "warp") {
            compare_numeric(expectation, executed.model->panic_warp());
            return;
        }
        if (field == "cause") {
            compare_numeric(expectation, executed.model->panic_cause());
            return;
        }
        if (field == "pc") {
            compare_numeric(expectation, executed.model->panic_pc());
            return;
        }
        throw std::runtime_error("unknown panic field '" + field + "'");
    }

    if (starts_with(expectation.key, "cycle.")) {
        const auto parts = split(expectation.key, '.');
        if (parts.size() < 3) {
            throw std::runtime_error("invalid cycle expectation key '" + expectation.key + "'");
        }
        const uint64_t cycle = parse_uint64(parts[1]);
        const auto& snapshot = snapshot_for_cycle(executed, cycle);

        if (parts[2] == "active_warps") {
            compare_numeric(expectation, snapshot.active_warps);
            return;
        }
        if (parts[2] == "opcoll_busy") {
            compare_bool(expectation, snapshot.opcoll_busy);
            return;
        }
        if (parts[2] == "alu_busy") {
            compare_bool(expectation, snapshot.alu_busy);
            return;
        }
        if (parts[2] == "mul_busy") {
            compare_bool(expectation, snapshot.mul_busy);
            return;
        }
        if (parts[2] == "div_busy") {
            compare_bool(expectation, snapshot.div_busy);
            return;
        }
        if (parts[2] == "ldst_busy") {
            compare_bool(expectation, snapshot.ldst_busy);
            return;
        }
        if (parts[2] == "active_mshrs") {
            compare_numeric(expectation, snapshot.active_mshrs);
            return;
        }
        if (parts[2] == "write_buffer_depth") {
            compare_numeric(expectation, snapshot.write_buffer_depth);
            return;
        }
        if (parts[2] == "panic_active") {
            compare_bool(expectation, snapshot.panic_active);
            return;
        }
        if (parts.size() == 4 && starts_with(parts[2], "warp")) {
            const uint32_t warp = static_cast<uint32_t>(parse_uint64(parts[2].substr(4)));
            const auto& warp_snapshot = snapshot.warps[warp];
            if (parts[3] == "state") {
                compare_string(expectation, to_string(warp_snapshot.state));
                return;
            }
            if (parts[3] == "rest_reason") {
                compare_string(expectation, to_string(warp_snapshot.rest_reason));
                return;
            }
            if (parts[3] == "pc") {
                compare_numeric(expectation, warp_snapshot.pc);
                return;
            }
            if (parts[3] == "dest_reg") {
                compare_numeric(expectation, warp_snapshot.dest_reg);
                return;
            }
            if (parts[3] == "active") {
                compare_bool(expectation, warp_snapshot.active);
                return;
            }
            if (parts[3] == "branch_taken") {
                compare_bool(expectation, warp_snapshot.branch_taken);
                return;
            }
        }

        throw std::runtime_error("unknown cycle field '" + expectation.key + "'");
    }

    throw std::runtime_error("unknown expectation key '" + expectation.key + "'");
}

} // namespace

TEST_CASE("Alignment manifests hold spec-facing timing contracts", "[alignment]") {
    const auto manifests = load_manifests(ALIGNMENT_MANIFEST_DIR);
    REQUIRE_FALSE(manifests.empty());

    for (const auto& manifest : manifests) {
        DYNAMIC_SECTION(manifest.scenario_name) {
            INFO("Manifest: " << manifest.path);
            INFO("Builder: " << manifest.builder_name);
            if (!manifest.citations.empty()) {
                INFO("Citations: " << manifest.citations.front());
            }

            const ExecutedScenario executed = run_manifest(manifest);
            for (const auto& expectation : manifest.expectations) {
                check_expectation(executed, expectation);
            }
        }
    }
}
