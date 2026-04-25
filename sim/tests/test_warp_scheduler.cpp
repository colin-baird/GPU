#include "catch.hpp"
#include "gpu_sim/timing/warp_scheduler.h"
#include "gpu_sim/timing/scoreboard.h"
#include "gpu_sim/timing/branch_shadow_tracker.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/decoder.h"
#include "gpu_sim/isa.h"

using namespace gpu_sim;

// Instruction encoding helpers
static uint32_t i_type(int32_t imm, uint32_t rs1, uint32_t funct3,
                        uint32_t rd, uint32_t opcode) {
    return (static_cast<uint32_t>(imm & 0xFFF) << 20) | (rs1 << 15) |
           (funct3 << 12) | (rd << 7) | opcode;
}

// ADDI rd, rs1, imm
static uint32_t encode_addi(uint32_t rd, uint32_t rs1, int32_t imm) {
    return i_type(imm, rs1, isa::FUNCT3_ADD_SUB, rd, isa::OP_ALU_I);
}

// ECALL
static uint32_t encode_ecall() {
    return i_type(0, 0, 0, 0, isa::OP_SYSTEM);
}

static uint32_t encode_vdot8(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    return (0x00 << 25) | (rs2 << 20) | (rs1 << 15) | (0x0 << 12) |
           (rd << 7) | isa::OP_VDOT8;
}

struct SchedulerFixture {
    SimConfig config;
    FunctionalModel func_model;
    Stats stats;
    Scoreboard scoreboard;
    BranchShadowTracker branch_tracker;
    std::vector<WarpState> warps;
    std::unique_ptr<WarpScheduler> scheduler;

    SchedulerFixture(uint32_t num_warps = 4)
        : config(), func_model(config) {
        config.num_warps = num_warps;
        config.start_pc = 0;
        func_model = FunctionalModel(config);

        for (uint32_t w = 0; w < num_warps; ++w) {
            warps.emplace_back(2); // buffer depth 2
            warps.back().reset(0);
        }

        scheduler = std::make_unique<WarpScheduler>(
            num_warps, warps.data(), scoreboard, branch_tracker, func_model, stats);
        // Phase 4: no consumers wired -> default "all ready". Tests that
        // need to gate opcoll or a specific unit busy use the override hooks
        // (set_opcoll_ready_override / set_unit_ready_override) below.
    }

    // Push a decoded instruction into a warp's buffer
    void push_instr(uint32_t warp, uint32_t raw, uint32_t pc) {
        BufferEntry entry;
        entry.decoded = Decoder::decode(raw);
        entry.warp_id = warp;
        entry.pc = pc;
        warps[warp].instr_buffer.push(entry);
    }

    // Load instructions into instruction memory and push decoded versions to buffers
    void load_and_push(uint32_t warp, uint32_t pc, uint32_t raw) {
        func_model.instruction_memory().write(pc / 4, raw);
        push_instr(warp, raw, pc);
    }
};

TEST_CASE("WarpScheduler: issues from non-empty buffer", "[scheduler]") {
    SchedulerFixture f(1);
    f.load_and_push(0, 0, encode_addi(5, 0, 42));

    f.scoreboard.seed_next();
    f.scheduler->evaluate();

    REQUIRE(f.scheduler->output().has_value());
    REQUIRE(f.scheduler->output()->warp_id == 0);
    REQUIRE(f.stats.total_instructions_issued == 1);
}

TEST_CASE("WarpScheduler: skips warp with empty buffer", "[scheduler]") {
    SchedulerFixture f(2);
    // Only warp 1 has an instruction
    f.load_and_push(1, 0, encode_addi(5, 0, 42));

    f.scoreboard.seed_next();
    f.scheduler->evaluate();

    REQUIRE(f.scheduler->output().has_value());
    REQUIRE(f.scheduler->output()->warp_id == 1);
    REQUIRE(f.stats.warp_stall_buffer_empty[0] == 1);
}

TEST_CASE("WarpScheduler: stalls on scoreboard hazard", "[scheduler]") {
    SchedulerFixture f(1);

    // x5 is pending (e.g., from a previous load)
    f.scoreboard.seed_next();
    f.scoreboard.set_pending(0, 5);
    f.scoreboard.commit();

    // Push ADDI x6, x5, 1 -- reads x5 which is pending
    f.load_and_push(0, 0, encode_addi(6, 5, 1));

    f.scoreboard.seed_next();
    f.scheduler->evaluate();

    REQUIRE_FALSE(f.scheduler->output().has_value());
    REQUIRE(f.stats.warp_stall_scoreboard[0] == 1);
}

TEST_CASE("WarpScheduler: VDOT8 checks rd as a source hazard", "[scheduler]") {
    SchedulerFixture f(1);

    f.scoreboard.seed_next();
    f.scoreboard.set_pending(0, 7);
    f.scoreboard.commit();

    f.load_and_push(0, 0, encode_vdot8(7, 5, 6));

    f.scoreboard.seed_next();
    f.scheduler->evaluate();

    REQUIRE_FALSE(f.scheduler->output().has_value());
    REQUIRE(f.stats.warp_stall_scoreboard[0] == 1);
}

TEST_CASE("WarpScheduler: stalls when opcoll busy", "[scheduler]") {
    SchedulerFixture f(1);
    f.load_and_push(0, 0, encode_addi(5, 0, 42));

    f.scoreboard.seed_next();
    f.scheduler->set_opcoll_ready_override(false); // OpColl is busy
    f.scheduler->evaluate();

    // Should not issue
    REQUIRE_FALSE(f.scheduler->output().has_value());
}

TEST_CASE("WarpScheduler: stalls when target unit busy", "[scheduler]") {
    SchedulerFixture f(1);
    f.load_and_push(0, 0, encode_addi(5, 0, 42));

    f.scheduler->set_unit_ready_override(ExecUnit::ALU, false); // ALU not ready

    f.scoreboard.seed_next();
    f.scheduler->evaluate();

    REQUIRE_FALSE(f.scheduler->output().has_value());
    REQUIRE(f.stats.warp_stall_unit_busy[0] == 1);
}

TEST_CASE("WarpScheduler: round-robin fairness across warps", "[scheduler]") {
    SchedulerFixture f(4);

    // Load instructions for all 4 warps
    for (uint32_t w = 0; w < 4; ++w) {
        f.load_and_push(w, 0, encode_addi(5, 0, static_cast<int32_t>(w + 1)));
    }

    // Issue 4 instructions -- should see different warps
    uint32_t issued_warps[4];
    for (int i = 0; i < 4; ++i) {
        f.scoreboard.seed_next();
        f.scheduler->evaluate();
        REQUIRE(f.scheduler->output().has_value());
        issued_warps[i] = f.scheduler->output()->warp_id;
        f.scheduler->commit();
    }

    // All 4 warps should have been issued (RR order starting from 0)
    // The exact order depends on the initial rr_pointer, but all should appear
    bool seen[4] = {};
    for (int i = 0; i < 4; ++i) {
        seen[issued_warps[i]] = true;
    }
    for (int i = 0; i < 4; ++i) {
        REQUIRE(seen[i]);
    }
}

TEST_CASE("WarpScheduler: diagnostics mark issued and ready_not_selected warps", "[scheduler]") {
    SchedulerFixture f(2);
    f.load_and_push(0, 0, encode_addi(5, 0, 1));
    f.load_and_push(1, 0, encode_addi(6, 0, 2));

    f.scoreboard.seed_next();
    f.scheduler->evaluate();
    f.scheduler->commit();

    const auto& diag = f.scheduler->current_diagnostics();
    REQUIRE(diag[0] == SchedulerIssueOutcome::ISSUED);
    REQUIRE(diag[1] == SchedulerIssueOutcome::READY_NOT_SELECTED);
}

TEST_CASE("WarpScheduler: sets scoreboard pending on issue", "[scheduler]") {
    SchedulerFixture f(1);
    f.load_and_push(0, 0, encode_addi(5, 0, 42)); // rd=5

    f.scoreboard.seed_next();
    f.scheduler->evaluate();
    f.scoreboard.commit();

    // After commit, x5 should be pending
    REQUIRE(f.scoreboard.is_pending(0, 5));
}

TEST_CASE("WarpScheduler: RR pointer advances even when idle", "[scheduler]") {
    SchedulerFixture f(4);
    // No instructions in any buffer

    f.scoreboard.seed_next();
    f.scheduler->evaluate();

    REQUIRE_FALSE(f.scheduler->output().has_value());
    REQUIRE(f.stats.scheduler_idle_cycles == 1);

    // Now put instruction in warp 1 only -- should be found after RR wraps
    f.scheduler->commit();
    f.load_and_push(1, 0, encode_addi(5, 0, 42));

    f.scoreboard.seed_next();
    f.scheduler->evaluate();

    REQUIRE(f.scheduler->output().has_value());
    REQUIRE(f.scheduler->output()->warp_id == 1);
}

TEST_CASE("WarpScheduler: scoreboard-stalled warp becomes eligible next cycle when hazard clears",
          "[scheduler][timing]") {
    // Binds issue-stage eligibility to scoreboard state transitions. Existing
    // tests verify the initial stall in isolation — nothing asserts that the
    // *same* warp issues on the very next cycle once the hazard clears.
    // Without this binding, a scheduler that cached an "ineligible" verdict
    // across cycles would still pass every other scheduler test.
    SchedulerFixture f(1);

    // Cycle 0: warp 0 reads x5 which is pending → SCOREBOARD stall.
    f.scoreboard.seed_next();
    f.scoreboard.set_pending(0, 5);
    f.scoreboard.commit();

    f.load_and_push(0, 0, encode_addi(6, 5, 1));

    f.scoreboard.seed_next();
    f.scheduler->evaluate();
    f.scheduler->commit();
    REQUIRE_FALSE(f.scheduler->current_output().has_value());
    REQUIRE(f.scheduler->current_diagnostics()[0] == SchedulerIssueOutcome::SCOREBOARD);
    REQUIRE(f.stats.total_instructions_issued == 0);

    // Between cycles: writeback clears the hazard.
    f.scoreboard.seed_next();
    f.scoreboard.clear_pending(0, 5);
    f.scoreboard.commit();

    // Cycle 1: same warp, same instruction at head — must now issue.
    f.scoreboard.seed_next();
    f.scheduler->evaluate();
    f.scheduler->commit();
    REQUIRE(f.scheduler->current_output().has_value());
    REQUIRE(f.scheduler->current_output()->warp_id == 0);
    REQUIRE(f.scheduler->current_diagnostics()[0] == SchedulerIssueOutcome::ISSUED);
    REQUIRE(f.stats.total_instructions_issued == 1);
}
