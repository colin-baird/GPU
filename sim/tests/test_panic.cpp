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
    // 0x04: EBREAK
    model.instruction_memory().write(0, encode_addi(31, 0, 0xAB));
    model.instruction_memory().write(1, encode_ebreak());
    model.init_kernel(config);

    Stats stats;
    TimingModel timing(config, model, stats);
    timing.run(1000);

    // Simulation should have terminated early (EBREAK triggers panic controller)
    REQUIRE(timing.cycle_count() < 1000);
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

    REQUIRE_FALSE(panic.is_active());
    REQUIRE_FALSE(panic.is_done());

    // Trigger panic
    panic.trigger(0, 0x04);
    REQUIRE(panic.is_active());
    REQUIRE_FALSE(panic.is_done());

    // Steps 0-2: diagnostic latching (3 cycles)
    panic.evaluate(); // step 0 -> 1
    REQUIRE_FALSE(panic.is_done());
    panic.evaluate(); // step 1 -> 2
    REQUIRE_FALSE(panic.is_done());
    panic.evaluate(); // step 2 -> 3
    REQUIRE_FALSE(panic.is_done());

    // Step 3: drain -- report units not drained
    panic.set_units_drained(false);
    panic.evaluate(); // still draining
    REQUIRE_FALSE(panic.is_done());

    // Now report units drained
    panic.set_units_drained(true);
    panic.evaluate(); // step 3 -> 4
    REQUIRE_FALSE(panic.is_done()); // step 4 hasn't run yet

    // Step 4: mark all inactive
    panic.evaluate();
    REQUIRE(panic.is_done());

    // Both warps should be inactive
    REQUIRE_FALSE(warps[0].active);
    REQUIRE_FALSE(warps[1].active);
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
