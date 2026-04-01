#include "catch.hpp"
#include "gpu_sim/config.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/timing/timing_model.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/isa.h"

using namespace gpu_sim;

// Instruction encoding helpers
static uint32_t i_type(int32_t imm, uint32_t rs1, uint32_t funct3,
                        uint32_t rd, uint32_t opcode) {
    return (static_cast<uint32_t>(imm & 0xFFF) << 20) | (rs1 << 15) |
           (funct3 << 12) | (rd << 7) | opcode;
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

static uint32_t encode_addi(uint32_t rd, uint32_t rs1, int32_t imm) {
    return i_type(imm, rs1, isa::FUNCT3_ADD_SUB, rd, isa::OP_ALU_I);
}

static uint32_t encode_beq(uint32_t rs1, uint32_t rs2, int32_t offset) {
    return b_type(offset, rs2, rs1, isa::FUNCT3_BEQ, isa::OP_BRANCH);
}

static uint32_t encode_bne(uint32_t rs1, uint32_t rs2, int32_t offset) {
    return b_type(offset, rs2, rs1, isa::FUNCT3_BNE, isa::OP_BRANCH);
}

static uint32_t encode_ecall() {
    return i_type(0, 0, 0, 0, isa::OP_SYSTEM);
}

TEST_CASE("Branch: taken branch redirects fetch and flushes", "[branch]") {
    SimConfig config;
    config.num_warps = 1;
    config.start_pc = 0;
    config.instruction_buffer_depth = 2;

    FunctionalModel model(config);

    // Program:
    // 0x00: ADDI x5, x0, 1
    // 0x04: BEQ  x0, x0, +8   (always taken, jumps to 0x0C)
    // 0x08: ADDI x6, x0, 99   (should be skipped)
    // 0x0C: ADDI x7, x0, 42
    // 0x10: ECALL
    model.instruction_memory().write(0, encode_addi(5, 0, 1));
    model.instruction_memory().write(1, encode_beq(0, 0, 8));
    model.instruction_memory().write(2, encode_addi(6, 0, 99));
    model.instruction_memory().write(3, encode_addi(7, 0, 42));
    model.instruction_memory().write(4, encode_ecall());
    model.init_kernel(config);

    Stats stats;
    TimingModel timing(config, model, stats);
    timing.run(1000);

    // x5 should be 1 (executed before branch)
    REQUIRE(model.register_file().read(0, 0, 5) == 1);
    // x6 should be 0 (skipped by branch)
    REQUIRE(model.register_file().read(0, 0, 6) == 0);
    // x7 should be 42 (branch target)
    REQUIRE(model.register_file().read(0, 0, 7) == 42);

    REQUIRE(stats.branch_flushes >= 1);
    REQUIRE(stats.branch_predictions == 1);
    REQUIRE(stats.branch_mispredictions == 1);
}

TEST_CASE("Branch: not-taken branch has no flush penalty", "[branch]") {
    SimConfig config;
    config.num_warps = 1;
    config.start_pc = 0;

    FunctionalModel model(config);

    // Program:
    // 0x00: ADDI x5, x0, 1
    // 0x04: BNE  x0, x0, +8   (never taken: x0 == x0)
    // 0x08: ADDI x6, x0, 42   (falls through)
    // 0x0C: ECALL
    model.instruction_memory().write(0, encode_addi(5, 0, 1));
    model.instruction_memory().write(1, encode_bne(0, 0, 8));
    model.instruction_memory().write(2, encode_addi(6, 0, 42));
    model.instruction_memory().write(3, encode_ecall());
    model.init_kernel(config);

    Stats stats;
    TimingModel timing(config, model, stats);
    timing.run(1000);

    REQUIRE(model.register_file().read(0, 0, 5) == 1);
    REQUIRE(model.register_file().read(0, 0, 6) == 42);
    REQUIRE(stats.branch_flushes == 0);
    REQUIRE(stats.branch_predictions == 1);
    REQUIRE(stats.branch_mispredictions == 0);
}

TEST_CASE("Branch: loop counts correctly", "[branch]") {
    SimConfig config;
    config.num_warps = 1;
    config.start_pc = 0;

    FunctionalModel model(config);

    // Program: loop 5 times, counting in x5
    // 0x00: ADDI x6, x0, 5     # loop count
    // 0x04: ADDI x5, x5, 1     # increment counter
    // 0x08: BNE  x5, x6, -4    # branch back to 0x04 if x5 != x6
    // 0x0C: ECALL
    model.instruction_memory().write(0, encode_addi(6, 0, 5));
    model.instruction_memory().write(1, encode_addi(5, 5, 1));
    model.instruction_memory().write(2, encode_bne(5, 6, -4));
    model.instruction_memory().write(3, encode_ecall());
    model.init_kernel(config);

    Stats stats;
    TimingModel timing(config, model, stats);
    timing.run(10000);

    REQUIRE(model.register_file().read(0, 0, 5) == 5);
    REQUIRE(model.register_file().read(0, 0, 6) == 5);

    // Backward-taken iterations are predicted correctly; the final fall-through misses.
    REQUIRE(stats.branch_predictions == 5);
    REQUIRE(stats.branch_mispredictions == 1);
    REQUIRE(stats.branch_flushes == 1);
}

TEST_CASE("Branch: taken branch incurs pipeline flush penalty", "[branch]") {
    SimConfig config;
    config.num_warps = 1;
    config.start_pc = 0;

    FunctionalModel model(config);

    // Compare a straight-line program with a branch program
    // Straight-line: 3 ADDIs + ECALL
    model.instruction_memory().write(0, encode_addi(5, 0, 1));
    model.instruction_memory().write(1, encode_addi(6, 0, 2));
    model.instruction_memory().write(2, encode_addi(7, 0, 3));
    model.instruction_memory().write(3, encode_ecall());
    model.init_kernel(config);

    Stats stats_straight;
    TimingModel timing_straight(config, model, stats_straight);
    timing_straight.run(1000);
    uint64_t straight_cycles = timing_straight.cycle_count();

    // Branch program: ADDI + BEQ(taken) + ADDI + ECALL
    FunctionalModel model2(config);
    model2.instruction_memory().write(0, encode_addi(5, 0, 1));
    model2.instruction_memory().write(1, encode_beq(0, 0, 8));   // jump to 0x0C
    model2.instruction_memory().write(2, encode_addi(6, 0, 99));  // skipped
    model2.instruction_memory().write(3, encode_addi(7, 0, 3));
    model2.instruction_memory().write(4, encode_ecall());
    model2.init_kernel(config);

    Stats stats_branch;
    TimingModel timing_branch(config, model2, stats_branch);
    timing_branch.run(1000);
    uint64_t branch_cycles = timing_branch.cycle_count();

    // The branch version should take more cycles due to flush penalty
    REQUIRE(branch_cycles > straight_cycles);
}
