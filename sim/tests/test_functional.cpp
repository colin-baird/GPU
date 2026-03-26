#include "catch.hpp"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/isa.h"
#include <array>

using namespace gpu_sim;

// Instruction encoding helpers
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

static SimConfig make_config() {
    SimConfig c;
    c.num_warps = 1;
    c.kernel_args[0] = 0;
    c.kernel_args[1] = 0;
    c.kernel_args[2] = 0;
    c.kernel_args[3] = 0;
    c.start_pc = 0;
    return c;
}

TEST_CASE("Functional: ADDI chain", "[functional]") {
    auto config = make_config();
    FunctionalModel model(config);

    // Program: ADDI x5, x0, 10; ADDI x6, x5, 20; ADDI x7, x6, 30
    model.instruction_memory().write(0, i_type(10, 0, 0, 5, isa::OP_ALU_I));  // x5 = 0 + 10
    model.instruction_memory().write(1, i_type(20, 5, 0, 6, isa::OP_ALU_I));  // x6 = 10 + 20
    model.instruction_memory().write(2, i_type(30, 6, 0, 7, isa::OP_ALU_I));  // x7 = 30 + 30

    auto evt1 = model.execute(0, 0);
    REQUIRE(evt1.results[0] == 10);
    REQUIRE(model.register_file().read(0, 0, 5) == 10);

    auto evt2 = model.execute(0, 4);
    REQUIRE(evt2.results[0] == 30);
    REQUIRE(model.register_file().read(0, 0, 6) == 30);

    auto evt3 = model.execute(0, 8);
    REQUIRE(evt3.results[0] == 60);
    REQUIRE(model.register_file().read(0, 0, 7) == 60);
}

TEST_CASE("Functional: Write to x0 discarded", "[functional]") {
    auto config = make_config();
    FunctionalModel model(config);

    model.instruction_memory().write(0, i_type(42, 0, 0, 0, isa::OP_ALU_I));  // x0 = 0 + 42
    model.execute(0, 0);
    REQUIRE(model.register_file().read(0, 0, 0) == 0);
}

TEST_CASE("Functional: Load and Store", "[functional]") {
    auto config = make_config();
    FunctionalModel model(config);

    // Store 0x12345678 to address 0x1000, then load it back
    // First put the value in a register: LUI x5, 0x12345; ADDI x5, x5, 0x678
    // Simpler: use kernel arg
    config.kernel_args[0] = 0x12345678;  // r1 = 0x12345678
    config.kernel_args[1] = 0x1000;       // r2 = 0x1000
    model.init_kernel(config);

    // SW x1, 0(x2)  -- store r1 to [r2+0]
    model.instruction_memory().write(0, s_type(0, 1, 2, 2, isa::OP_STORE));  // SW
    // LW x5, 0(x2)  -- load from [r2+0] to x5
    model.instruction_memory().write(1, i_type(0, 2, 2, 5, isa::OP_LOAD));   // LW

    auto evt_store = model.execute(0, 0);
    REQUIRE(evt_store.is_store == true);
    REQUIRE(evt_store.mem_addresses[0] == 0x1000);

    auto evt_load = model.execute(0, 4);
    REQUIRE(evt_load.is_load == true);
    REQUIRE(evt_load.results[0] == 0x12345678);
    REQUIRE(model.register_file().read(0, 0, 5) == 0x12345678);
}

TEST_CASE("Functional: Branch taken", "[functional]") {
    auto config = make_config();
    FunctionalModel model(config);

    // BEQ x0, x0, +8 -- always taken (0 == 0)
    model.instruction_memory().write(0, b_type(8, 0, 0, 0, isa::OP_BRANCH));
    auto evt = model.execute(0, 0);
    REQUIRE(evt.is_branch == true);
    REQUIRE(evt.branch_taken == true);
    REQUIRE(evt.branch_target == 8);
}

TEST_CASE("Functional: Branch not taken", "[functional]") {
    auto config = make_config();
    config.kernel_args[0] = 1;  // r1 = 1
    FunctionalModel model(config);

    // BEQ x1, x0, +8 -- not taken (1 != 0)
    model.instruction_memory().write(0, b_type(8, 0, 1, 0, isa::OP_BRANCH));
    auto evt = model.execute(0, 0);
    REQUIRE(evt.is_branch == true);
    REQUIRE(evt.branch_taken == false);
}

TEST_CASE("Functional: ECALL marks warp inactive", "[functional]") {
    auto config = make_config();
    FunctionalModel model(config);

    // ECALL
    model.instruction_memory().write(0, i_type(0, 0, 0, 0, isa::OP_SYSTEM));
    REQUIRE(model.is_warp_active(0) == true);
    auto evt = model.execute(0, 0);
    REQUIRE(evt.is_ecall == true);
    REQUIRE(model.is_warp_active(0) == false);
}

TEST_CASE("Functional: EBREAK panics", "[functional]") {
    auto config = make_config();
    FunctionalModel model(config);

    // Set r31 (panic cause) first
    model.instruction_memory().write(0, i_type(42, 0, 0, 31, isa::OP_ALU_I));  // x31 = 42
    model.instruction_memory().write(1, i_type(1, 0, 0, 0, isa::OP_SYSTEM));    // EBREAK

    model.execute(0, 0);  // Set x31
    auto evt = model.execute(0, 4);  // EBREAK
    REQUIRE(evt.is_ebreak == true);
    REQUIRE(evt.panic_cause == 42);
    REQUIRE(model.is_panicked() == true);
    REQUIRE(model.panic_warp() == 0);
    REQUIRE(model.panic_cause() == 42);
}

TEST_CASE("Functional: CSR reads", "[functional]") {
    SimConfig config;
    config.num_warps = 4;
    FunctionalModel model(config);

    // CSRRS x5, warp_id, x0
    model.instruction_memory().write(0, i_type(0xC00, 0, 2, 5, isa::OP_SYSTEM));
    // CSRRS x6, lane_id, x0
    model.instruction_memory().write(1, i_type(0xC01, 0, 2, 6, isa::OP_SYSTEM));
    // CSRRS x7, num_warps, x0
    model.instruction_memory().write(2, i_type(0xC02, 0, 2, 7, isa::OP_SYSTEM));

    // Execute for warp 2
    auto evt1 = model.execute(2, 0);
    REQUIRE(evt1.results[0] == 2);   // warp_id = 2
    REQUIRE(evt1.is_csr == true);

    auto evt2 = model.execute(2, 4);
    REQUIRE(evt2.results[0] == 0);   // lane_id for lane 0 = 0
    REQUIRE(evt2.results[15] == 15); // lane_id for lane 15 = 15
    REQUIRE(evt2.results[31] == 31); // lane_id for lane 31 = 31

    auto evt3 = model.execute(2, 8);
    REQUIRE(evt3.results[0] == 4);   // num_warps = 4
}

TEST_CASE("Functional: VDOT8 execution", "[functional]") {
    auto config = make_config();
    config.kernel_args[0] = 0x04030201;  // r1 = packed [1, 2, 3, 4]
    config.kernel_args[1] = 0x08070605;  // r2 = packed [5, 6, 7, 8]
    FunctionalModel model(config);

    // VDOT8 x5, x1, x2 (x5 starts at 0)
    model.instruction_memory().write(0, r_type(0x00, 2, 1, 0, 5, isa::OP_VDOT8));
    auto evt = model.execute(0, 0);
    REQUIRE(evt.results[0] == 70);  // 1*5 + 2*6 + 3*7 + 4*8 = 70
    REQUIRE(model.register_file().read(0, 0, 5) == 70);

    // Execute again: accumulates
    auto evt2 = model.execute(0, 0);
    REQUIRE(evt2.results[0] == 140);
}

TEST_CASE("Functional: TLOOKUP", "[functional]") {
    auto config = make_config();
    config.kernel_args[0] = 5;  // r1 = 5 (table index)
    FunctionalModel model(config);

    // Set up lookup table
    model.lookup_table().write(261, 0xDEADBEEF);  // table[5 + 256] = 0xDEADBEEF

    // TLOOKUP x5, x1, 256
    model.instruction_memory().write(0, i_type(256, 1, 0, 5, isa::OP_TLOOKUP));
    auto evt = model.execute(0, 0);
    REQUIRE(evt.is_tlookup == true);
    REQUIRE(evt.results[0] == 0xDEADBEEF);
    REQUIRE(model.register_file().read(0, 0, 5) == 0xDEADBEEF);
}

TEST_CASE("Functional: Kernel args loaded", "[functional]") {
    SimConfig config;
    config.num_warps = 2;
    config.kernel_args[0] = 100;
    config.kernel_args[1] = 200;
    config.kernel_args[2] = 300;
    config.kernel_args[3] = 400;
    FunctionalModel model(config);

    for (WarpId w = 0; w < 2; ++w) {
        for (LaneId l = 0; l < WARP_SIZE; ++l) {
            REQUIRE(model.register_file().read(w, l, 0) == 0);
            REQUIRE(model.register_file().read(w, l, 1) == 100);
            REQUIRE(model.register_file().read(w, l, 2) == 200);
            REQUIRE(model.register_file().read(w, l, 3) == 300);
            REQUIRE(model.register_file().read(w, l, 4) == 400);
            REQUIRE(model.register_file().read(w, l, 5) == 0);
        }
    }
}

TEST_CASE("Functional: LUI + ADDI", "[functional]") {
    auto config = make_config();
    FunctionalModel model(config);

    // LUI x5, 0x12345
    model.instruction_memory().write(0, (0x12345 << 12) | (5 << 7) | isa::OP_LUI);
    // ADDI x5, x5, 0x678
    model.instruction_memory().write(1, i_type(0x678, 5, 0, 5, isa::OP_ALU_I));

    model.execute(0, 0);
    REQUIRE(model.register_file().read(0, 0, 5) == 0x12345000);

    model.execute(0, 4);
    REQUIRE(model.register_file().read(0, 0, 5) == 0x12345678);
}

TEST_CASE("Functional: Byte load sign extension", "[functional]") {
    auto config = make_config();
    config.kernel_args[0] = 0x1000;  // r1 = base address
    FunctionalModel model(config);

    // Write 0xFF to address 0x1000
    model.memory().write8(0x1000, 0xFF);

    // LB x5, 0(x1) -- sign-extended: 0xFF -> -1 -> 0xFFFFFFFF
    model.instruction_memory().write(0, i_type(0, 1, 0, 5, isa::OP_LOAD));
    model.execute(0, 0);
    REQUIRE(model.register_file().read(0, 0, 5) == 0xFFFFFFFF);

    // LBU x6, 0(x1) -- zero-extended: 0xFF -> 0x000000FF
    model.instruction_memory().write(1, i_type(0, 1, 4, 6, isa::OP_LOAD));
    model.execute(0, 4);
    REQUIRE(model.register_file().read(0, 0, 6) == 0x000000FF);
}

TEST_CASE("Functional: JAL", "[functional]") {
    auto config = make_config();
    FunctionalModel model(config);

    // JAL x1, +16: encode imm[20|10:1|11|19:12]
    uint32_t imm20 = 0;
    uint32_t imm10_1 = 8;  // 16 >> 1 = 8
    uint32_t imm11 = 0;
    uint32_t imm19_12 = 0;
    uint32_t instr = (imm20 << 31) | (imm10_1 << 21) | (imm11 << 20) |
                     (imm19_12 << 12) | (1 << 7) | isa::OP_JAL;
    model.instruction_memory().write(0, instr);

    auto evt = model.execute(0, 0);
    REQUIRE(evt.is_branch == true);
    REQUIRE(evt.branch_taken == true);
    REQUIRE(evt.branch_target == 16);
    REQUIRE(evt.results[0] == 4);  // Return address = PC + 4
    REQUIRE(model.register_file().read(0, 0, 1) == 4);
}

TEST_CASE("Functional: Multi-warp independence", "[functional]") {
    SimConfig config;
    config.num_warps = 4;
    config.kernel_args[0] = 0;
    FunctionalModel model(config);

    // ADDI x5, x0, 100
    model.instruction_memory().write(0, i_type(100, 0, 0, 5, isa::OP_ALU_I));

    // Execute for each warp -- each should get independent result
    model.execute(0, 0);
    model.execute(1, 0);
    model.execute(2, 0);
    model.execute(3, 0);

    for (WarpId w = 0; w < 4; ++w) {
        REQUIRE(model.register_file().read(w, 0, 5) == 100);
    }
}

TEST_CASE("Functional: Store byte/halfword", "[functional]") {
    auto config = make_config();
    config.kernel_args[0] = 0x12345678;
    config.kernel_args[1] = 0x2000;
    FunctionalModel model(config);

    // SB x1, 0(x2) -- store low byte (0x78) at address 0x2000
    model.instruction_memory().write(0, s_type(0, 1, 2, 0, isa::OP_STORE));
    model.execute(0, 0);
    REQUIRE(model.memory().read8(0x2000) == 0x78);

    // SH x1, 4(x2) -- store low halfword (0x5678) at address 0x2004
    model.instruction_memory().write(1, s_type(4, 1, 2, 1, isa::OP_STORE));
    model.execute(0, 4);
    REQUIRE(model.memory().read16(0x2004) == 0x5678);
}
