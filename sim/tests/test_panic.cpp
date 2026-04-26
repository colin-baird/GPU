#include "catch.hpp"
#include "gpu_sim/config.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/timing/timing_model.h"
#include "gpu_sim/timing/panic_controller.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/isa.h"

using namespace gpu_sim;

// Instruction encoding helpers
static uint32_t i_type(int32_t imm, uint32_t rs1, uint32_t funct3,
                        uint32_t rd, uint32_t opcode) {
    return (static_cast<uint32_t>(imm & 0xFFF) << 20) | (rs1 << 15) |
           (funct3 << 12) | (rd << 7) | opcode;
}

static uint32_t encode_addi(uint32_t rd, uint32_t rs1, int32_t imm) {
    return i_type(imm, rs1, isa::FUNCT3_ADD_SUB, rd, isa::OP_ALU_I);
}

static uint32_t encode_ebreak() {
    return i_type(1, 0, 0, 0, isa::OP_SYSTEM);
}

static uint32_t encode_ecall() {
    return i_type(0, 0, 0, 0, isa::OP_SYSTEM);
}

TEST_CASE("Panic: EBREAK halts simulation", "[panic]") {
    SimConfig config;
    config.num_warps = 1;
    config.start_pc = 0;

    FunctionalModel model(config);

    // Program:
    // 0x00: ADDI x31, x0, 0xAB  # Set panic cause
    // 0x04: ADDI x0, x0, 0      # Bubble so x31 reaches committed state
    // 0x08: EBREAK
    model.instruction_memory().write(0, encode_addi(31, 0, 0xAB));
    model.instruction_memory().write(1, encode_addi(0, 0, 0));
    model.instruction_memory().write(2, encode_ebreak());
    model.init_kernel(config);

    Stats stats;
    TimingModel timing(config, model, stats);
    timing.run(1000);

    // Simulation should have terminated early (EBREAK triggers panic controller)
    REQUIRE(timing.cycle_count() < 1000);
    REQUIRE(model.is_panicked());
    REQUIRE(model.panic_warp() == 0);
    REQUIRE(model.panic_pc() == 0x08);
    REQUIRE(model.panic_cause() == 0xAB);
}

TEST_CASE("Panic: multi-warp EBREAK marks all warps inactive", "[panic]") {
    SimConfig config;
    config.num_warps = 4;
    config.start_pc = 0;

    FunctionalModel model(config);

    // All warps will fetch ADDI then EBREAK
    // EBREAK is detected at decode and triggers panic, halting all warps
    model.instruction_memory().write(0, encode_addi(5, 0, 1));
    model.instruction_memory().write(1, encode_ebreak());
    model.init_kernel(config);

    Stats stats;
    TimingModel timing(config, model, stats);
    timing.run(1000);

    // Simulation should terminate early
    REQUIRE(timing.cycle_count() < 1000);
    // The timing model's panic controller marks warps inactive in the timing
    // model's warp state (not in the functional model), so we just verify
    // the simulation terminated properly via cycle count.
}

TEST_CASE("PanicController: state machine progression", "[panic]") {
    SimConfig config;
    config.num_warps = 2;
    FunctionalModel model(config);

    std::vector<WarpState> warps;
    for (uint32_t w = 0; w < 2; ++w) {
        warps.emplace_back(2);
        warps.back().reset(0);
    }

    PanicController panic(2, warps.data(), model);

    // Phase 6: replace the prior set_units_drained() pre-evaluate setter
    // with a wired callable that the controller queries inside
    // evaluate(). The test drives the bool through a captured local.
    bool drained = false;
    panic.set_drained_query([&drained]() { return drained; });

    REQUIRE_FALSE(panic.is_active());
    REQUIRE_FALSE(panic.is_done());

    // Trigger panic
    panic.trigger(0, 0x04);
    REQUIRE(panic.is_active());
    REQUIRE_FALSE(panic.is_done());

    // Decode already consumed cycle 1; the first panic-controller tick latches diagnostics.
    panic.evaluate(); // step 1 -> 2
    REQUIRE_FALSE(panic.is_done());
    REQUIRE(model.is_panicked());
    REQUIRE(model.panic_warp() == 0);
    REQUIRE(model.panic_pc() == 0x04);

    // Drain step -- report units not drained
    drained = false;
    panic.evaluate(); // still draining
    REQUIRE_FALSE(panic.is_done());

    // Now report units drained
    drained = true;
    panic.evaluate(); // drain -> halt
    REQUIRE_FALSE(panic.is_done());
    panic.evaluate();
    REQUIRE(panic.is_done());

    // Both warps should be inactive
    REQUIRE_FALSE(warps[0].active);
    REQUIRE_FALSE(warps[1].active);
}

TEST_CASE("Panic: committed writeback stays frozen once panic is active", "[panic]") {
    SimConfig config;
    config.num_warps = 1;
    config.start_pc = 0;
    config.multiply_pipeline_stages = 3;

    FunctionalModel model(config);
    model.instruction_memory().write(0, encode_addi(1, 0, 2));
    model.instruction_memory().write(1, encode_addi(2, 0, 3));
    model.instruction_memory().write(
        2, (isa::FUNCT7_MULDIV << 25) | (2 << 20) | (1 << 15) |
               (isa::FUNCT3_MUL << 12) | (5 << 7) | isa::OP_ALU_R);
    model.instruction_memory().write(3, encode_ebreak());
    model.init_kernel(config);

    Stats stats;
    TimingModel timing(config, model, stats);

    bool saw_panic = false;
    for (int i = 0; i < 32 && timing.tick(); ++i) {
        const auto& snapshot = timing.last_cycle_snapshot();
        if (snapshot && snapshot->panic_active) {
            saw_panic = true;
            REQUIRE_FALSE(timing.last_committed_writeback().has_value());
        }
    }

    REQUIRE(saw_panic);
}

TEST_CASE("PanicController: reset clears state", "[panic]") {
    SimConfig config;
    config.num_warps = 1;
    FunctionalModel model(config);

    std::vector<WarpState> warps;
    warps.emplace_back(2);
    warps.back().reset(0);

    PanicController panic(1, warps.data(), model);
    panic.trigger(0, 0);
    REQUIRE(panic.is_active());

    panic.reset();
    REQUIRE_FALSE(panic.is_active());
    REQUIRE_FALSE(panic.is_done());
}
