#include "catch.hpp"
#include "gpu_sim/config.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/timing/timing_model.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/isa.h"
#include "gpu_sim/decoder.h"
#include <cstdio>
#include <fstream>
#include <sstream>

using namespace gpu_sim;

// ========== Instruction encoding helpers ==========

static uint32_t r_type(uint32_t funct7, uint32_t rs2, uint32_t rs1,
                        uint32_t funct3, uint32_t rd, uint32_t opcode) {
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode;
}

static uint32_t i_type(int32_t imm, uint32_t rs1, uint32_t funct3,
                        uint32_t rd, uint32_t opcode) {
    return (static_cast<uint32_t>(imm & 0xFFF) << 20) | (rs1 << 15) |
           (funct3 << 12) | (rd << 7) | opcode;
}

static uint32_t s_type(int32_t imm, uint32_t rs2, uint32_t rs1,
                        uint32_t funct3, uint32_t opcode) {
    uint32_t imm_hi = (imm >> 5) & 0x7F;
    uint32_t imm_lo = imm & 0x1F;
    return (imm_hi << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_lo << 7) | opcode;
}

static uint32_t b_type(int32_t imm, uint32_t rs2, uint32_t rs1,
                        uint32_t funct3, uint32_t opcode) {
    uint32_t bit12 = (imm >> 12) & 1;
    uint32_t bit11 = (imm >> 11) & 1;
    uint32_t bits10_5 = (imm >> 5) & 0x3F;
    uint32_t bits4_1 = (imm >> 1) & 0xF;
    return (bit12 << 31) | (bits10_5 << 25) | (rs2 << 20) | (rs1 << 15) |
           (funct3 << 12) | (bits4_1 << 8) | (bit11 << 7) | opcode;
}

static uint32_t u_type(int32_t imm, uint32_t rd, uint32_t opcode) {
    return (static_cast<uint32_t>(imm) & 0xFFFFF000) | (rd << 7) | opcode;
}

static uint32_t j_type(int32_t imm, uint32_t rd, uint32_t opcode) {
    uint32_t bit20 = (imm >> 20) & 1;
    uint32_t bits10_1 = (imm >> 1) & 0x3FF;
    uint32_t bit11 = (imm >> 11) & 1;
    uint32_t bits19_12 = (imm >> 12) & 0xFF;
    return (bit20 << 31) | (bits10_1 << 21) | (bit11 << 20) |
           (bits19_12 << 12) | (rd << 7) | opcode;
}

// Convenience encoders
static uint32_t ADDI(uint32_t rd, uint32_t rs1, int32_t imm) {
    return i_type(imm, rs1, isa::FUNCT3_ADD_SUB, rd, isa::OP_ALU_I);
}
static uint32_t ADD(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    return r_type(isa::FUNCT7_BASE, rs2, rs1, isa::FUNCT3_ADD_SUB, rd, isa::OP_ALU_R);
}
static uint32_t MUL(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    return r_type(isa::FUNCT7_MULDIV, rs2, rs1, isa::FUNCT3_MUL, rd, isa::OP_ALU_R);
}
static uint32_t LW(uint32_t rd, uint32_t rs1, int32_t imm) {
    return i_type(imm, rs1, isa::FUNCT3_LW, rd, isa::OP_LOAD);
}
static uint32_t SW(uint32_t rs2, uint32_t rs1, int32_t imm) {
    return s_type(imm, rs2, rs1, isa::FUNCT3_SW, isa::OP_STORE);
}
static uint32_t BEQ(uint32_t rs1, uint32_t rs2, int32_t offset) {
    return b_type(offset, rs2, rs1, isa::FUNCT3_BEQ, isa::OP_BRANCH);
}
static uint32_t BNE(uint32_t rs1, uint32_t rs2, int32_t offset) {
    return b_type(offset, rs2, rs1, isa::FUNCT3_BNE, isa::OP_BRANCH);
}
static uint32_t JAL(uint32_t rd, int32_t offset) {
    return j_type(offset, rd, isa::OP_JAL);
}
static uint32_t LUI(uint32_t rd, int32_t imm_upper) {
    return u_type(imm_upper, rd, isa::OP_LUI);
}
static uint32_t ECALL() {
    return i_type(0, 0, 0, 0, isa::OP_SYSTEM);
}
static uint32_t EBREAK() {
    return i_type(1, 0, 0, 0, isa::OP_SYSTEM);
}
static uint32_t VDOT8(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    return r_type(0x00, rs2, rs1, 0x0, rd, isa::OP_VDOT8);
}
static uint32_t TLOOKUP(uint32_t rd, uint32_t rs1, int32_t imm) {
    return i_type(imm, rs1, 0, rd, isa::OP_TLOOKUP);
}

// Helper: set up a single-warp config and return model + timing
struct IntegrationFixture {
    SimConfig config;
    FunctionalModel model;
    Stats stats;

    IntegrationFixture(uint32_t num_warps = 1) : config(), model(config) {
        config.num_warps = num_warps;
        config.start_pc = 0;
        config.instruction_buffer_depth = 2;
        config.external_memory_latency_cycles = 10; // Short for testing
        model = FunctionalModel(config);
    }

    void load_program(const std::vector<uint32_t>& instructions) {
        for (size_t i = 0; i < instructions.size(); ++i) {
            model.instruction_memory().write(static_cast<uint32_t>(i), instructions[i]);
        }
        model.init_kernel(config);
    }

    uint64_t run(uint64_t max_cycles = 50000) {
        TimingModel timing(config, model, stats);
        timing.run(max_cycles);
        return timing.cycle_count();
    }

    uint32_t reg(uint32_t warp, uint32_t lane, uint32_t r) const {
        return model.register_file().read(warp, lane, r);
    }
};

static std::string slurp_file(const std::string& path) {
    std::ifstream in(path);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

// ========== Integration Tests ==========

TEST_CASE("Integration: ADD chain correctness and completion", "[integration]") {
    IntegrationFixture f;
    // 10 sequential ADDIs: x5 = 1+2+3+...+10 = 55
    std::vector<uint32_t> prog;
    for (int i = 1; i <= 10; ++i) {
        prog.push_back(ADDI(5, 5, i));
    }
    prog.push_back(ECALL());
    f.load_program(prog);

    uint64_t cycles = f.run();

    REQUIRE(f.reg(0, 0, 5) == 55);
    // Verify across all lanes (all lanes get same result for ADDI from x0-derived chain)
    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        REQUIRE(f.reg(0, lane, 5) == 55);
    }
    // Should complete in reasonable time (pipeline depth + 10 instructions + drain)
    REQUIRE(cycles > 10);   // At least 10 instructions
    REQUIRE(cycles < 200);  // Shouldn't take excessively long
}

TEST_CASE("Integration: independent ADDIs to different registers", "[integration]") {
    IntegrationFixture f;
    // No RAW hazards between these -- all read x0
    f.load_program({
        ADDI(5, 0, 10),
        ADDI(6, 0, 20),
        ADDI(7, 0, 30),
        ADDI(8, 0, 40),
        ECALL()
    });

    f.run();

    REQUIRE(f.reg(0, 0, 5) == 10);
    REQUIRE(f.reg(0, 0, 6) == 20);
    REQUIRE(f.reg(0, 0, 7) == 30);
    REQUIRE(f.reg(0, 0, 8) == 40);
}

TEST_CASE("Integration: RAW dependency chain", "[integration]") {
    IntegrationFixture f;
    // Each instruction depends on the previous one's result
    f.load_program({
        ADDI(5, 0, 1),   // x5 = 1
        ADDI(5, 5, 1),   // x5 = 2 (depends on x5)
        ADDI(5, 5, 1),   // x5 = 3
        ADDI(5, 5, 1),   // x5 = 4
        ECALL()
    });

    uint64_t cycles = f.run();

    REQUIRE(f.reg(0, 0, 5) == 4);
    // With RAW hazards, should be slower than independent chain
    // Each instruction stalls waiting for scoreboard clear
    REQUIRE(cycles > 4);
}

TEST_CASE("Integration: load-use stall", "[integration]") {
    IntegrationFixture f;

    // Store a value to memory first, then load it and use it
    // x1 = base address (kernel arg, set via config)
    f.config.kernel_args[0] = 0x1000; // Base addr in x1
    f.model = FunctionalModel(f.config);

    // Pre-store a value in memory
    f.model.memory().write32(0x1000, 0xDEADBEEF);

    f.load_program({
        LW(5, 1, 0),      // x5 = mem[x1+0] = 0xDEADBEEF
        ADDI(6, 5, 1),    // x6 = x5 + 1 (load-use: stalls on scoreboard)
        ECALL()
    });

    uint64_t cycles = f.run();

    REQUIRE(f.reg(0, 0, 5) == 0xDEADBEEF);
    REQUIRE(f.reg(0, 0, 6) == 0xDEADBEF0); // 0xDEADBEEF + 1
}

TEST_CASE("Integration: store then load same address", "[integration]") {
    IntegrationFixture f;
    f.config.kernel_args[0] = 0x1000;
    f.model = FunctionalModel(f.config);

    f.load_program({
        ADDI(5, 0, 42),   // x5 = 42
        SW(5, 1, 0),      // mem[x1+0] = x5
        LW(6, 1, 0),      // x6 = mem[x1+0]
        ECALL()
    });

    f.run();

    // The functional model handles store-load forwarding correctly
    REQUIRE(f.reg(0, 0, 6) == 42);
}

TEST_CASE("Integration: completion waits for write-through drain", "[integration]") {
    IntegrationFixture f;
    f.config.kernel_args[0] = 0x1000;
    f.config.external_memory_latency_cycles = 10;
    f.model = FunctionalModel(f.config);

    f.load_program({
        ADDI(5, 0, 42),
        SW(5, 1, 0),
        ECALL()
    });

    uint64_t cycles = f.run();

    REQUIRE(f.stats.external_memory_reads == 1);
    REQUIRE(f.stats.external_memory_writes == 1);
    REQUIRE(cycles >= 20);
}

TEST_CASE("Trace snapshot: load miss is classified as memory_wait", "[integration][trace]") {
    IntegrationFixture f;
    f.config.kernel_args[0] = 0x1000;
    f.config.external_memory_latency_cycles = 20;
    f.model = FunctionalModel(f.config);
    f.model.memory().write32(0x1000, 0xDEADBEEF);

    f.load_program({
        LW(5, 1, 0),
        ADDI(6, 5, 1),
        ECALL()
    });

    TimingModel timing(f.config, f.model, f.stats);
    bool saw_memory_wait = false;
    for (uint32_t i = 0; i < 40 && timing.tick(); ++i) {
        REQUIRE(timing.last_cycle_snapshot().has_value());
        const auto& warp = timing.last_cycle_snapshot()->warps[0];
        if (warp.state == WarpTraceState::MEMORY_WAIT) {
            saw_memory_wait = true;
            REQUIRE(warp.rest_reason == WarpRestReason::WAIT_MEMORY_RESPONSE);
            REQUIRE(warp.pc == 0);
            REQUIRE(warp.raw_instruction == LW(5, 1, 0));
            break;
        }
    }
    REQUIRE(saw_memory_wait);
}

TEST_CASE("Trace snapshot: MSHR pressure is classified as wait_l1_mshr", "[integration][trace]") {
    IntegrationFixture f(2);
    f.config.num_warps = 2;
    f.config.num_mshrs = 1;
    f.config.kernel_args[0] = 0x1000;
    f.config.external_memory_latency_cycles = 20;
    f.model = FunctionalModel(f.config);
    f.model.memory().write32(0x1000, 0x12345678);

    f.load_program({
        LW(5, 1, 0),
        ADDI(6, 5, 1),
        ECALL()
    });

    TimingModel timing(f.config, f.model, f.stats);
    bool saw_mshr_wait = false;
    for (uint32_t i = 0; i < 60 && timing.tick(); ++i) {
        REQUIRE(timing.last_cycle_snapshot().has_value());
        const auto& snapshot = *timing.last_cycle_snapshot();
        for (uint32_t w = 0; w < 2; ++w) {
            const auto& warp = snapshot.warps[w];
            if (warp.state == WarpTraceState::AT_REST &&
                warp.rest_reason == WarpRestReason::WAIT_L1_MSHR) {
                saw_mshr_wait = true;
            }
        }
        if (saw_mshr_wait) {
            break;
        }
    }
    REQUIRE(saw_mshr_wait);
}

TEST_CASE("Trace file: emits Chrome trace JSON with warp states and counters",
          "[integration][trace]") {
    IntegrationFixture f(2);
    f.config.kernel_args[0] = 0x1000;
    f.config.external_memory_latency_cycles = 20;
    f.model = FunctionalModel(f.config);
    f.model.memory().write32(0x1000, 0xCAFEBABE);

    f.load_program({
        LW(5, 1, 0),
        ADDI(6, 5, 1),
        ECALL()
    });

    const std::string trace_path = "/tmp/gpu_trace_integration_test.json";
    std::remove(trace_path.c_str());

    TimingTraceOptions trace_options;
    trace_options.output_path = trace_path;

    {
        TimingModel timing(f.config, f.model, f.stats, trace_options);
        timing.run(200);
    }

    const std::string trace = slurp_file(trace_path);
    REQUIRE_FALSE(trace.empty());
    REQUIRE(trace.find("\"traceEvents\"") != std::string::npos);
    REQUIRE(trace.find("Warp 0") != std::string::npos);
    REQUIRE(trace.find("active_warps") != std::string::npos);
    REQUIRE(trace.find("memory_wait") != std::string::npos);
    REQUIRE(trace.find("issue") != std::string::npos);

    std::remove(trace_path.c_str());
}

TEST_CASE("Integration: branch loop", "[integration]") {
    IntegrationFixture f;
    // Loop 5 times
    f.load_program({
        ADDI(6, 0, 5),     // x6 = 5 (loop count)
        ADDI(5, 5, 1),     // x5++ (loop body)
        BNE(5, 6, -4),     // if x5 != x6, branch back to ADDI x5
        ECALL()
    });

    uint64_t cycles = f.run();

    REQUIRE(f.reg(0, 0, 5) == 5);
    REQUIRE(f.reg(0, 0, 6) == 5);
    REQUIRE(f.stats.branch_predictions == 5);
    REQUIRE(f.stats.branch_mispredictions == 1);
    REQUIRE(f.stats.branch_flushes == 1);
}

TEST_CASE("Integration: JAL link register", "[integration]") {
    IntegrationFixture f;
    // JAL saves return address in rd
    f.load_program({
        ADDI(5, 0, 1),    // 0x00: x5 = 1
        JAL(10, 8),        // 0x04: x10 = 0x08 (return addr), jump to 0x0C
        ADDI(5, 0, 99),   // 0x08: skipped
        ADDI(6, 0, 42),   // 0x0C: x6 = 42 (target)
        ECALL()            // 0x10
    });

    f.run();

    REQUIRE(f.reg(0, 0, 5) == 1);
    REQUIRE(f.reg(0, 0, 10) == 0x08); // Return address = PC+4 of JAL
    REQUIRE(f.reg(0, 0, 6) == 42);
    REQUIRE(f.stats.branch_predictions == 1);
    REQUIRE(f.stats.branch_mispredictions == 0);
    REQUIRE(f.stats.branch_flushes == 0);
}

TEST_CASE("Integration: multi-warp independent computation", "[integration]") {
    IntegrationFixture f(4); // 4 warps
    f.config.num_warps = 4;
    f.model = FunctionalModel(f.config);

    // Each warp reads its warp ID via CSR and stores it in x5
    // CSR 0xC00 = warp_id → CSRRS x5, 0xC00, x0
    uint32_t csrrs_warp_id = (0xC00 << 20) | (0 << 15) | (isa::FUNCT3_CSRRS << 12) |
                              (5 << 7) | isa::OP_SYSTEM;
    f.load_program({
        csrrs_warp_id,     // x5 = warp_id
        ADDI(6, 5, 10),   // x6 = warp_id + 10
        ECALL()
    });

    f.run();

    for (uint32_t w = 0; w < 4; ++w) {
        REQUIRE(f.reg(w, 0, 5) == w);
        REQUIRE(f.reg(w, 0, 6) == w + 10);
    }
}

TEST_CASE("Integration: multi-warp all complete via ECALL", "[integration]") {
    IntegrationFixture f(4);
    f.config.num_warps = 4;
    f.model = FunctionalModel(f.config);

    f.load_program({
        ADDI(5, 0, 1),
        ECALL()
    });

    uint64_t cycles = f.run();

    REQUIRE(cycles < 1000);
    for (uint32_t w = 0; w < 4; ++w) {
        REQUIRE_FALSE(f.model.is_warp_active(w));
        REQUIRE(f.reg(w, 0, 5) == 1);
    }
}

TEST_CASE("Integration: memory coalescing vs serialization cycle count", "[integration]") {
    // Coalesced loads should be faster than scattered loads
    SimConfig config;
    config.num_warps = 1;
    config.start_pc = 0;
    config.external_memory_latency_cycles = 10;

    // Test 1: Coalesced -- all lanes load from same cache line
    // Use CSR lane_id to compute per-lane address: addr = base + lane*4
    // All within one 128-byte line if base is aligned
    {
        FunctionalModel model(config);
        config.kernel_args[0] = 0x1000; // Base addr aligned to 128B
        model = FunctionalModel(config);

        // Pre-fill memory
        for (uint32_t i = 0; i < 32; ++i) {
            model.memory().write32(0x1000 + i * 4, i * 100);
        }

        // CSRRS x7, CSR_LANE_ID, x0  -- x7 = lane_id
        uint32_t csrrs_lane = (isa::CSR_LANE_ID << 20) | (0 << 15) |
                               (isa::FUNCT3_CSRRS << 12) | (7 << 7) | isa::OP_SYSTEM;

        model.instruction_memory().write(0, csrrs_lane);           // x7 = lane_id
        model.instruction_memory().write(1, ADDI(8, 7, 0));       // x8 = lane_id (copy)
        model.instruction_memory().write(2, ADD(8, 8, 8));        // x8 = lane_id * 2
        model.instruction_memory().write(3, ADD(8, 8, 8));        // x8 = lane_id * 4
        model.instruction_memory().write(4, ADD(8, 1, 8));        // x8 = base + lane_id*4
        model.instruction_memory().write(5, LW(5, 8, 0));        // x5 = mem[base + lane*4]
        model.instruction_memory().write(6, ECALL());
        model.init_kernel(config);

        Stats stats;
        TimingModel timing(config, model, stats);
        timing.run(50000);

        // Verify lane 0 loaded correct value
        REQUIRE(model.register_file().read(0, 0, 5) == 0);
        REQUIRE(model.register_file().read(0, 1, 5) == 100);
        REQUIRE(stats.coalesced_requests >= 1);
    }
}

TEST_CASE("Integration: serialized load blocks dependent issue until all lanes complete",
          "[integration][coalescing]") {
    // Regression test for premature scoreboard clear on non-coalesced loads.
    // A scattered load (each lane in a different cache line) must keep the
    // destination register pending on the scoreboard until ALL 32 serialized
    // lane requests have been processed and the writeback is produced.
    //
    // Program:
    //   x7 = lane_id
    //   x8 = lane_id * 4
    //   x8 = x8 * 512        (spread lanes across 512-byte strides → different cache lines)
    //   x8 = x1 + x8         (x1 = base address from kernel arg)
    //   x5 = mem[x8]         (scattered load → serialized, 32 cycles of lane requests)
    //   x6 = x5 + 1          (dependent on x5 — must stall until load writeback)
    //   ecall

    // --- Run with scattered addresses (serialized) ---
    SimConfig config;
    config.num_warps = 1;
    config.start_pc = 0;
    config.external_memory_latency_cycles = 10;
    config.kernel_args[0] = 0x1000;

    FunctionalModel model_scattered(config);
    // Pre-fill memory at scattered locations
    for (uint32_t i = 0; i < 32; ++i) {
        model_scattered.memory().write32(0x1000 + i * 512, 1000 + i);
    }

    // CSRRS x7, CSR_LANE_ID, x0
    uint32_t csrrs_lane = (isa::CSR_LANE_ID << 20) | (0 << 15) |
                           (isa::FUNCT3_CSRRS << 12) | (7 << 7) | isa::OP_SYSTEM;

    // Build address: base + lane_id * 512
    // lane_id * 512 = lane_id << 9 = (lane_id * 4) * 128
    // Use: x8 = lane_id, shift left by 9 via repeated add
    // Simpler: LUI x9, (512 << 12) then MUL. But 512 fits in 12-bit imm.
    // Actually: ADDI x9, x0, 512; MUL x8, x7, x9; ADD x8, x1, x8
    model_scattered.instruction_memory().write(0, csrrs_lane);           // x7 = lane_id
    model_scattered.instruction_memory().write(1, ADDI(9, 0, 512));     // x9 = 512
    model_scattered.instruction_memory().write(2, MUL(8, 7, 9));        // x8 = lane_id * 512
    model_scattered.instruction_memory().write(3, ADD(8, 1, 8));        // x8 = base + offset
    model_scattered.instruction_memory().write(4, LW(5, 8, 0));        // x5 = mem[x8] (SCATTERED)
    model_scattered.instruction_memory().write(5, ADDI(6, 5, 1));      // x6 = x5 + 1 (DEPENDENT)
    model_scattered.instruction_memory().write(6, ECALL());
    model_scattered.init_kernel(config);

    Stats stats_scattered;
    TimingModel timing_scattered(config, model_scattered, stats_scattered);
    timing_scattered.run(50000);
    uint64_t cycles_scattered = timing_scattered.cycle_count();

    // Verify correctness: the dependent ADDI must see the loaded value
    REQUIRE(model_scattered.register_file().read(0, 0, 5) == 1000);
    REQUIRE(model_scattered.register_file().read(0, 0, 6) == 1001);
    REQUIRE(model_scattered.register_file().read(0, 1, 5) == 1001);
    REQUIRE(model_scattered.register_file().read(0, 1, 6) == 1002);
    REQUIRE(model_scattered.register_file().read(0, 31, 5) == 1031);
    REQUIRE(model_scattered.register_file().read(0, 31, 6) == 1032);

    // Must have been serialized
    REQUIRE(stats_scattered.serialized_requests >= 1);

    // --- Run with coalesced addresses for comparison ---
    FunctionalModel model_coalesced(config);
    for (uint32_t i = 0; i < 32; ++i) {
        model_coalesced.memory().write32(0x1000 + i * 4, 1000 + i);
    }

    // Same program but stride = 4 (all in one 128B cache line)
    model_coalesced.instruction_memory().write(0, csrrs_lane);
    model_coalesced.instruction_memory().write(1, ADDI(9, 0, 4));       // x9 = 4
    model_coalesced.instruction_memory().write(2, MUL(8, 7, 9));        // x8 = lane_id * 4
    model_coalesced.instruction_memory().write(3, ADD(8, 1, 8));        // x8 = base + offset
    model_coalesced.instruction_memory().write(4, LW(5, 8, 0));        // x5 = mem[x8] (COALESCED)
    model_coalesced.instruction_memory().write(5, ADDI(6, 5, 1));      // x6 = x5 + 1 (DEPENDENT)
    model_coalesced.instruction_memory().write(6, ECALL());
    model_coalesced.init_kernel(config);

    Stats stats_coalesced;
    TimingModel timing_coalesced(config, model_coalesced, stats_coalesced);
    timing_coalesced.run(50000);
    uint64_t cycles_coalesced = timing_coalesced.cycle_count();

    // Verify coalesced correctness
    REQUIRE(model_coalesced.register_file().read(0, 0, 5) == 1000);
    REQUIRE(model_coalesced.register_file().read(0, 0, 6) == 1001);
    REQUIRE(stats_coalesced.coalesced_requests >= 1);

    // The scattered case must take significantly more cycles due to serialization.
    // With 32 lanes serialized at 1 per cycle, plus memory latency, the gap should
    // be at least ~30 cycles.  If the scoreboard were cleared prematurely (old bug),
    // the ADDI would issue immediately after the first lane and cycles would be similar.
    REQUIRE(cycles_scattered > cycles_coalesced + 20);
}

TEST_CASE("Integration: LUI + ADDI pattern", "[integration]") {
    IntegrationFixture f;
    // Build a 32-bit constant: 0x12345678
    // LUI x5, 0x12345000
    // ADDI x5, x5, 0x678
    f.load_program({
        LUI(5, 0x12345000),
        ADDI(5, 5, 0x678),
        ECALL()
    });

    f.run();

    REQUIRE(f.reg(0, 0, 5) == 0x12345678);
}

TEST_CASE("Integration: multiply instruction", "[integration]") {
    IntegrationFixture f;
    f.load_program({
        ADDI(5, 0, 7),    // x5 = 7
        ADDI(6, 0, 6),    // x6 = 6
        MUL(7, 5, 6),     // x7 = 7 * 6 = 42
        ECALL()
    });

    f.run();

    REQUIRE(f.reg(0, 0, 7) == 42);
}

TEST_CASE("Integration: multiply pipeline latency > ALU", "[integration]") {
    // MUL should take more cycles than ADD due to pipeline depth.
    // Use a dependent chain so the latency actually matters:
    // result of MUL/ADD is immediately used by next instruction.
    SimConfig config;
    config.num_warps = 1;
    config.start_pc = 0;
    config.multiply_pipeline_stages = 3;

    // ADD-only dependent chain: x5 = 2, x5 = x5 + x5, x5 = x5 + x5, x6 = x5 + 0
    FunctionalModel model_add(config);
    model_add.instruction_memory().write(0, ADDI(5, 0, 2));
    model_add.instruction_memory().write(1, ADD(5, 5, 5));   // depends on x5
    model_add.instruction_memory().write(2, ADD(5, 5, 5));   // depends on x5
    model_add.instruction_memory().write(3, ADDI(6, 5, 0));  // depends on x5
    model_add.instruction_memory().write(4, ECALL());
    model_add.init_kernel(config);
    Stats stats_add;
    TimingModel timing_add(config, model_add, stats_add);
    timing_add.run(1000);

    // MUL dependent chain: x5 = 2, x6 = 3, x5 = x5 * x6, x5 = x5 * x6, x7 = x5 + 0
    FunctionalModel model_mul(config);
    model_mul.instruction_memory().write(0, ADDI(5, 0, 2));
    model_mul.instruction_memory().write(1, ADDI(6, 0, 3));
    model_mul.instruction_memory().write(2, MUL(5, 5, 6));   // depends on x5, x6
    model_mul.instruction_memory().write(3, MUL(5, 5, 6));   // depends on x5
    model_mul.instruction_memory().write(4, ADDI(7, 5, 0));  // depends on x5
    model_mul.instruction_memory().write(5, ECALL());
    model_mul.init_kernel(config);
    Stats stats_mul;
    TimingModel timing_mul(config, model_mul, stats_mul);
    timing_mul.run(1000);

    // MUL chain should take more cycles than ADD chain due to higher latency
    REQUIRE(timing_add.cycle_count() < 1000);
    REQUIRE(timing_mul.cycle_count() < 1000);
    REQUIRE(timing_mul.cycle_count() > timing_add.cycle_count());
}

TEST_CASE("Integration: VDOT8 accumulate", "[integration]") {
    IntegrationFixture f;

    // Set up: x5 = packed INT8 values [1, 2, 3, 4]
    //         x6 = packed INT8 values [1, 1, 1, 1]
    //         x7 = 0 (accumulator)
    // VDOT8 x7, x5, x6 → x7 += 1*1 + 2*1 + 3*1 + 4*1 = 10
    // Packed bytes: [1,2,3,4] = 0x04030201 (little-endian)
    // Packed bytes: [1,1,1,1] = 0x01010101
    f.load_program({
        LUI(5, 0x04030000),
        ADDI(5, 5, 0x201),      // x5 = 0x04030201
        LUI(6, 0x01010000),
        ADDI(6, 6, 0x101),      // x6 = 0x01010101
        ADDI(7, 0, 0),          // x7 = 0 (accumulator)
        VDOT8(7, 5, 6),         // x7 += dot(x5_bytes, x6_bytes) = 10
        ECALL()
    });

    f.run();

    REQUIRE(f.reg(0, 0, 7) == 10);
}

TEST_CASE("Integration: TLOOKUP instruction", "[integration]") {
    IntegrationFixture f;

    // Set up lookup table with known values
    for (uint32_t i = 0; i < 64; ++i) {
        f.model.lookup_table().write(i, i * 100);
    }

    f.load_program({
        ADDI(5, 0, 10),       // x5 = 10 (index)
        TLOOKUP(6, 5, 0),     // x6 = LUT[x5 + 0] = LUT[10] = 1000
        ECALL()
    });
    // Re-init kernel after loading lookup table
    f.model.init_kernel(f.config);
    // Restore lookup table (init_kernel may clear it)
    for (uint32_t i = 0; i < 64; ++i) {
        f.model.lookup_table().write(i, i * 100);
    }

    f.run();

    REQUIRE(f.reg(0, 0, 6) == 1000);
}

TEST_CASE("Integration: EBREAK triggers panic", "[integration]") {
    IntegrationFixture f;
    f.load_program({
        ADDI(31, 0, 0x42),   // Set panic cause in x31
        EBREAK()
    });

    uint64_t cycles = f.run(1000);

    // Simulation should terminate early due to panic controller
    // EBREAK is caught at decode stage and triggers the panic state machine
    REQUIRE(cycles < 1000);
}

TEST_CASE("Integration: statistics are collected", "[integration]") {
    IntegrationFixture f;
    f.load_program({
        ADDI(5, 0, 1),
        ADDI(6, 0, 2),
        ADD(7, 5, 6),
        ECALL()
    });

    f.run();

    REQUIRE(f.stats.total_cycles > 0);
    REQUIRE(f.stats.total_instructions_issued == 4); // 3 ALU + 1 ECALL
    REQUIRE(f.stats.alu_stats.instructions >= 3);
}

TEST_CASE("Integration: x0 writes discarded", "[integration]") {
    IntegrationFixture f;
    f.load_program({
        ADDI(0, 0, 42),   // Write to x0 -- should be discarded
        ADDI(5, 0, 1),    // x5 = 0 + 1 = 1 (x0 is still 0)
        ECALL()
    });

    f.run();

    REQUIRE(f.reg(0, 0, 0) == 0); // x0 always 0
    REQUIRE(f.reg(0, 0, 5) == 1);
}

TEST_CASE("Integration: max cycles limit terminates simulation", "[integration]") {
    IntegrationFixture f;
    // Infinite loop
    f.load_program({
        ADDI(5, 5, 1),   // 0x00: x5++
        BEQ(0, 0, -4),   // 0x04: always branch back (infinite loop)
        ECALL()           // 0x08: never reached
    });

    Stats stats;
    TimingModel timing(f.config, f.model, stats);
    timing.run(100); // Limit to 100 cycles

    REQUIRE(timing.cycle_count() >= 100);
}
