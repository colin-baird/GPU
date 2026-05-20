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

// R-type encoder (used for the M-extension ops below).
static uint32_t r_type(uint32_t funct7, uint32_t rs2, uint32_t rs1,
                       uint32_t funct3, uint32_t rd, uint32_t opcode) {
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) |
           (funct3 << 12) | (rd << 7) | opcode;
}

// DIV rd, rs1, rs2 — targets the iterative DivideUnit.
static uint32_t encode_div(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    return r_type(isa::FUNCT7_MULDIV, rs2, rs1, isa::FUNCT3_DIV, rd,
                  isa::OP_ALU_R);
}

// MUL rd, rs1, rs2 — targets the fully-pipelined MultiplyUnit.
static uint32_t encode_mul(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    return r_type(isa::FUNCT7_MULDIV, rs2, rs1, isa::FUNCT3_MUL, rd,
                  isa::OP_ALU_R);
}

struct SchedulerFixture {
    SimConfig config;
    FunctionalModel func_model;
    Stats stats;
    Scoreboard scoreboard;
    BranchShadowTracker branch_tracker;
    std::vector<WarpState> warps;
    std::unique_ptr<WarpScheduler> scheduler;

    SchedulerFixture(uint32_t num_warps = 4,
                     uint32_t multiply_pipeline_stages = kMulPipelineStages)
        : config(), func_model(config) {
        config.num_warps = num_warps;
        config.multiply_pipeline_stages = multiply_pipeline_stages;
        config.start_pc = 0;
        func_model = FunctionalModel(config);

        for (uint32_t w = 0; w < num_warps; ++w) {
            warps.emplace_back(2); // buffer depth 2
            warps.back().reset(0);
        }

        scheduler = std::make_unique<WarpScheduler>(
            num_warps, warps.data(), func_model, stats,
            config.multiply_pipeline_stages);
        // Phase 10B.0: scoreboard and branch_tracker wired via
        // set_dependencies(). The opcoll / unit busy-poll pointers were
        // removed — the scheduler predicts unit availability from its own
        // issue scoreboard (unit_busy_ countdowns + the writeback bitmap).
        // ldst stays null (the LDST FIFO gate sees an empty FIFO). Tests
        // that need to gate a specific unit busy or a writeback slot drive
        // the scoreboard directly via test_set_unit_busy() /
        // test_reserve_writeback_slot().
        scheduler->set_dependencies(&scoreboard, &branch_tracker, nullptr);
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
    f.scheduler->commit();

    // Phase 10B.2: scheduler->opcoll output is REGISTERED — current_output()
    // exposes the committed slot, so the test commits before observing it.
    REQUIRE(f.scheduler->current_output().has_value());
    REQUIRE(f.scheduler->current_output()->warp_id == 0);
    REQUIRE(f.stats.total_instructions_issued == 1);
}

TEST_CASE("WarpScheduler: skips warp with empty buffer", "[scheduler]") {
    SchedulerFixture f(2);
    // Only warp 1 has an instruction
    f.load_and_push(1, 0, encode_addi(5, 0, 42));

    f.scoreboard.seed_next();
    f.scheduler->evaluate();
    f.scheduler->commit();

    REQUIRE(f.scheduler->current_output().has_value());
    REQUIRE(f.scheduler->current_output()->warp_id == 1);
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
    f.scheduler->commit();

    REQUIRE_FALSE(f.scheduler->current_output().has_value());
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
    f.scheduler->commit();

    REQUIRE_FALSE(f.scheduler->current_output().has_value());
    REQUIRE(f.stats.warp_stall_scoreboard[0] == 1);
}

TEST_CASE("WarpScheduler: stalls a WAW hazard to keep same-register writeback in order",
          "[scheduler][timing]") {
    // Pathological intra-warp WAW: two instructions write the SAME destination
    // register, and the second one's source operands are clear -- so the RAW
    // source checks all pass and only a destination-register check can catch
    // it. Execution-unit latencies are asymmetric: MUL's issue->writeback
    // offset is 5, ALU's is 3. Without the WAW gate the scheduler would issue
    //
    //   cycle 0:  MUL  x5, x1, x2     (writeback at cycle 5)
    //   cycle 1:  ADDI x5, x0, 99     (writeback at cycle 4)
    //
    // back-to-back, and the ALU result would reach the writeback arbiter one
    // cycle BEFORE the MUL -- the MUL would then clobber x5, so the older
    // instruction's write lands last (out-of-order writeback). The WAW gate
    // must hold the ADDI until the MUL's write commits.
    SchedulerFixture f(1);

    // Cycle 0: MUL x5, x1, x2 issues freely (x1/x2 not pending) and marks x5
    // pending for the warp.
    f.load_and_push(0, 0, encode_mul(5, 1, 2));

    f.scoreboard.seed_next();
    f.scheduler->evaluate();
    f.scheduler->commit();
    f.scoreboard.commit();
    REQUIRE(f.scheduler->current_output().has_value());
    REQUIRE(f.scheduler->current_output()->pc == 0);
    REQUIRE(f.scoreboard.current_pending(0, 5));

    // Cycle 1: ADDI x5, x0, 99 -- writes x5 (still pending) but reads only x0.
    // Pure WAW: a source-only scoreboard check would let it through. It must
    // be stalled, and reported as a scoreboard stall.
    f.load_and_push(0, 4, encode_addi(5, 0, 99));

    f.scoreboard.seed_next();
    f.scheduler->evaluate();
    f.scheduler->commit();
    f.scoreboard.commit();
    REQUIRE_FALSE(f.scheduler->current_output().has_value());
    REQUIRE(f.scheduler->current_diagnostics()[0]
            == SchedulerIssueOutcome::SCOREBOARD);
    REQUIRE(f.stats.warp_stall_scoreboard[0] == 1);

    // Cycle 2: x5 still pending -- the stall is a standing gate, not a
    // one-shot verdict cached from cycle 1.
    f.scoreboard.seed_next();
    f.scheduler->evaluate();
    f.scheduler->commit();
    f.scoreboard.commit();
    REQUIRE_FALSE(f.scheduler->current_output().has_value());
    REQUIRE(f.stats.warp_stall_scoreboard[0] == 2);

    // The MUL's write commits -- the writeback arbiter clears x5.
    f.scoreboard.seed_next();
    f.scoreboard.clear_pending(0, 5);
    f.scoreboard.commit();

    // Cycle 3: with the prior write retired, the ADDI is finally free to
    // issue -- strictly after the MUL, so the two writes to x5 retire in
    // program order.
    f.scoreboard.seed_next();
    f.scheduler->evaluate();
    f.scheduler->commit();
    f.scoreboard.commit();
    REQUIRE(f.scheduler->current_output().has_value());
    REQUIRE(f.scheduler->current_output()->pc == 4);
    REQUIRE(f.scheduler->current_diagnostics()[0]
            == SchedulerIssueOutcome::ISSUED);
    REQUIRE(f.stats.total_instructions_issued == 2);
}

TEST_CASE("WarpScheduler: stalls on writeback-bitmap contention", "[scheduler]") {
    // Phase 10B.0: the scheduler refuses to issue a fixed-latency op whose
    // predicted writeback cycle is already claimed in the bitmap. Reserve the
    // ALU writeback slot for this cycle's issue, then verify a writing ALU op
    // cannot issue (its offset-0 reservation collides).
    SchedulerFixture f(1);
    f.load_and_push(0, 0, encode_addi(5, 0, 42)); // ALU op, writes x5

    // test_reserve_writeback_slot stamps writeback_bitmap_[head + offset]
    // pre-evaluate. evaluate() first clears the old head and advances
    // bitmap_head_ by 1, then the gate checks writeback_bitmap_[head + 3] for
    // an ALU op (Phase 10B end-of-phase ALU issue->writeback offset is 3 —
    // three REGISTERED forward edges). Reserving at offset 4 here lands
    // exactly on (post-advance head) + 3, so the gate sees the collision.
    f.scheduler->seed_next();
    f.scheduler->test_reserve_writeback_slot(ExecUnit::ALU, 4);

    f.scoreboard.seed_next();
    f.scheduler->evaluate();
    f.scheduler->commit();

    REQUIRE_FALSE(f.scheduler->current_output().has_value());
    REQUIRE(f.stats.warp_stall_unit_busy[0] == 1);
    REQUIRE(f.stats.scheduler_writeback_contention_stall_cycles[
                static_cast<size_t>(ExecUnit::ALU)] == 1);
}

TEST_CASE("WarpScheduler: multiply writeback bitmap uses configured pipeline depth",
          "[scheduler]") {
    // The multiply pipeline depth is a SimConfig parameter. The scheduler's
    // binding writeback bitmap must therefore reserve/check the configured
    // multiply offset, not the default-depth constant.
    constexpr uint32_t kConfiguredMulStages = 5;
    SchedulerFixture f(1, kConfiguredMulStages);
    f.load_and_push(0, 0, encode_mul(5, 1, 2));

    const uint32_t runtime_offset = compute_issue_to_writeback_offset(
        ExecUnit::MULTIPLY, kConfiguredMulStages, /*is_vdot8=*/false);
    REQUIRE(runtime_offset ==
            kConfiguredMulStages + 2);
    REQUIRE(runtime_offset !=
            kIssueToWritebackOffset[exec_unit_index(ExecUnit::MULTIPLY)]);

    // test_reserve_writeback_slot stamps writeback_bitmap_[head + offset]
    // before evaluate() advances the head. Reserve at +1 relative to the
    // runtime issue offset so the post-advance gate sees the collision.
    f.scheduler->seed_next();
    f.scheduler->test_reserve_writeback_slot(ExecUnit::MULTIPLY,
                                             runtime_offset + 1);

    f.scoreboard.seed_next();
    f.scheduler->evaluate();
    f.scheduler->commit();

    REQUIRE_FALSE(f.scheduler->current_output().has_value());
    REQUIRE(f.stats.warp_stall_unit_busy[0] == 1);
    REQUIRE(f.stats.scheduler_writeback_contention_stall_cycles[
                static_cast<size_t>(ExecUnit::MULTIPLY)] == 1);
}

TEST_CASE("WarpScheduler: stalls when iterative target unit busy", "[scheduler]") {
    // Phase 10B.0: an iterative unit (DIVIDE) occupied by an in-flight op
    // cannot accept a new one. The structural-hazard countdown unit_busy_
    // models this; arm it directly via the test hook.
    SchedulerFixture f(1);
    f.load_and_push(0, 0, encode_div(5, 1, 2)); // DIV -> DivideUnit

    // Arm the DIVIDE countdown. evaluate() decrements it once at the top, so
    // arm with 2 to leave it at 1 (still > 0) when the gate reads it.
    f.scheduler->seed_next();
    f.scheduler->test_set_unit_busy(ExecUnit::DIVIDE, 2);

    f.scoreboard.seed_next();
    f.scheduler->evaluate();
    f.scheduler->commit();

    REQUIRE_FALSE(f.scheduler->current_output().has_value());
    REQUIRE(f.stats.warp_stall_unit_busy[0] == 1);
    REQUIRE(f.stats.scheduler_unit_busy_stall_cycles[
                static_cast<size_t>(ExecUnit::DIVIDE)] == 1);

    REQUIRE(f.scheduler->current_diagnostics()[0]
            == SchedulerIssueOutcome::UNIT_BUSY_DIVIDE);
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
        f.scheduler->commit();
        REQUIRE(f.scheduler->current_output().has_value());
        issued_warps[i] = f.scheduler->current_output()->warp_id;
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
    REQUIRE(f.scoreboard.current_pending(0, 5));
}

TEST_CASE("WarpScheduler: RR pointer advances even when idle", "[scheduler]") {
    SchedulerFixture f(4);
    // No instructions in any buffer

    f.scoreboard.seed_next();
    f.scheduler->evaluate();
    f.scheduler->commit();

    REQUIRE_FALSE(f.scheduler->current_output().has_value());
    REQUIRE(f.stats.scheduler_idle_cycles == 1);

    // Now put instruction in warp 1 only -- should be found after RR wraps
    f.load_and_push(1, 0, encode_addi(5, 0, 42));

    f.scoreboard.seed_next();
    f.scheduler->evaluate();
    f.scheduler->commit();

    REQUIRE(f.scheduler->current_output().has_value());
    REQUIRE(f.scheduler->current_output()->warp_id == 1);
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

TEST_CASE("WarpScheduler: fully-pipelined ALU issues one op per cycle",
          "[scheduler][timing]") {
    // Phase 10B.0 regression: ALU is a fully-pipelined unit
    // (kUnitIterationLatency[ALU] == 0), so it has no structural input gate —
    // its only issue limiter is the writeback bitmap, which advances one slot
    // per cycle and therefore frees a fresh offset-0 entry every cycle. A
    // bookkeeping bug that armed unit_busy_ for ALU, or that failed to clear
    // the elapsed bitmap slot, would cap ALU throughput below 1/cycle. This
    // pins the back-to-back-issue property that the removed busy poll
    // (which never blocked the fully-pipelined ALU) used to guarantee.
    SchedulerFixture f(1);
    constexpr int kCount = 8;

    // The fixture's instruction buffer is depth 2, so refill one entry per
    // cycle (modeling a steady frontend) rather than front-loading all 8.
    for (int i = 0; i < kCount; ++i) {
        f.load_and_push(0, static_cast<uint32_t>(i * 4),
                        encode_addi(static_cast<uint32_t>(5 + i), 0, i + 1));

        f.scoreboard.seed_next();
        f.scheduler->evaluate();
        f.scheduler->commit();
        // Writeback clears the destination next cycle in the real machine; in
        // isolation, clear it here so the scoreboard never blocks the next op.
        f.scoreboard.seed_next();
        f.scoreboard.clear_pending(0, static_cast<RegIndex>(5 + i));
        f.scoreboard.commit();
        REQUIRE(f.scheduler->current_output().has_value());
        REQUIRE(f.scheduler->current_diagnostics()[0]
                == SchedulerIssueOutcome::ISSUED);
    }
    REQUIRE(f.stats.total_instructions_issued == kCount);
}
