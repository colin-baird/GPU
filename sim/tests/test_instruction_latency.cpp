// Instruction-latency tests for the Phase 10 synchronous-pipeline machine.
//
// Two concerns:
//  1. Each execution pipeline (ALU / MULTIPLY / TLOOKUP / DIVIDE) retires its
//     writeback at the documented latency. Latency is measured end-to-end
//     through the full TimingModel as (writeback-commit cycle - the unit's
//     stamped issue_cycle), then cross-checked against the scheduler's binding
//     issue-to-writeback offset helper.
//  2. Writeback contention: when a variable-latency load preempts a
//     fixed-latency unit for the single writeback port, the arbiter's
//     combinational-backward stall freezes the unit, so its writeback — and
//     therefore its observed issue->writeback latency — slips by exactly one
//     cycle per preemption.

#include "catch.hpp"

#include "gpu_sim/config.h"
#include "gpu_sim/decoder.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/isa.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/timing/timing_model.h"

#include <vector>

using namespace gpu_sim;

namespace {

// --- minimal RV32IM + custom-extension encoders -----------------------------

uint32_t i_type(int32_t imm, uint32_t rs1, uint32_t funct3, uint32_t rd,
                uint32_t opcode) {
    return (static_cast<uint32_t>(imm & 0xFFF) << 20) | (rs1 << 15) |
           (funct3 << 12) | (rd << 7) | opcode;
}

uint32_t r_type(uint32_t funct7, uint32_t rs2, uint32_t rs1, uint32_t funct3,
                uint32_t rd, uint32_t opcode) {
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) |
           (rd << 7) | opcode;
}

uint32_t encode_addi(uint32_t rd, uint32_t rs1, int32_t imm) {
    return i_type(imm, rs1, isa::FUNCT3_ADD_SUB, rd, isa::OP_ALU_I);
}
uint32_t encode_mul(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    return r_type(isa::FUNCT7_MULDIV, rs2, rs1, isa::FUNCT3_MUL, rd, isa::OP_ALU_R);
}
uint32_t encode_div(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    return r_type(isa::FUNCT7_MULDIV, rs2, rs1, isa::FUNCT3_DIV, rd, isa::OP_ALU_R);
}
uint32_t encode_tlookup(uint32_t rd, uint32_t rs1, int32_t imm) {
    return i_type(imm, rs1, 0, rd, isa::OP_TLOOKUP);
}
uint32_t encode_lw(uint32_t rd, uint32_t rs1, int32_t imm) {
    return i_type(imm, rs1, isa::FUNCT3_LW, rd, isa::OP_LOAD);
}
uint32_t encode_vdot8(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    // VDOT8 reads rd as a third source operand (num_src_regs == 3), so it
    // spends two cycles in the operand collector instead of one.
    return r_type(0x00, rs2, rs1, 0x0, rd, isa::OP_VDOT8);
}
uint32_t encode_ecall() {
    return i_type(0, 0, 0, 0, isa::OP_SYSTEM);
}

// External-memory latency used by run_and_capture's config — referenced by the
// load-latency lower bound below.
constexpr uint32_t kProbeMemoryLatency = 10;

// The single writeback observed for a probe program, the cycle the arbiter
// committed it, and the total operand-collection cycles consumed by the run
// (Stats::operand_collector_busy_cycles — incremented once per cycle the
// operand collector evaluates an instruction, including the trailing ECALL).
struct WbObservation {
    bool seen = false;
    uint64_t commit_cycle = 0;
    WritebackEntry entry{};
    uint64_t operand_collect_cycles = 0;
};

// Run `program` on a fresh single-warp TimingModel and capture the (expected
// unique) committed writeback to register `dest_reg`.
WbObservation run_and_capture(const std::vector<uint32_t>& program,
                              uint32_t dest_reg,
                              uint32_t multiply_pipeline_stages = kMulPipelineStages) {
    SimConfig config;
    config.num_warps = 1;
    config.start_pc = 0;
    config.instruction_buffer_depth = 2;
    config.multiply_pipeline_stages = multiply_pipeline_stages;
    config.external_memory_latency_cycles = kProbeMemoryLatency;

    FunctionalModel model(config);
    for (size_t i = 0; i < program.size(); ++i) {
        model.instruction_memory().write(static_cast<uint32_t>(i), program[i]);
    }
    model.init_kernel(config);

    Stats stats;
    TimingModel timing(config, model, stats);

    WbObservation obs;
    for (int i = 0; i < 500; ++i) {
        const bool keep_going = timing.tick();
        const auto& wb = timing.current_committed_writeback();
        if (wb.has_value() && wb->dest_reg == dest_reg) {
            obs.seen = true;
            obs.commit_cycle = timing.cycle_count();
            obs.entry = *wb;
        }
        if (!keep_going) {
            break;
        }
    }
    obs.operand_collect_cycles = stats.operand_collector_busy_cycles;
    return obs;
}

}  // namespace

TEST_CASE("Instruction latency: fixed-latency pipelines match writeback offsets",
          "[timing][latency]") {
    // A unit's scheduler-issue -> writeback offset is its measured execute
    // latency plus the fixed scheduler->opcoll->unit front-pipeline transit:
    // two REGISTERED forward edges (Phase 10B.1 / 10B.2).
    constexpr uint32_t kFrontPipelineTransit = 2;

    const SimConfig defaults;  // multiply_pipeline_stages default (3)

    struct Case {
        const char* name;
        uint32_t raw;
        ExecUnit unit;
        uint32_t expected_exec_latency;  // unit-accept -> writeback-commit
    };
    const std::vector<Case> cases = {
        {"ALU/ADDI",  encode_addi(5, 0, 7),    ExecUnit::ALU,
         /*1-cycle ALU + the REGISTERED unit->arbiter edge*/ 1},
        {"MULTIPLY",  encode_mul(5, 1, 2),     ExecUnit::MULTIPLY,
         defaults.multiply_pipeline_stages},
        {"TLOOKUP",   encode_tlookup(5, 1, 0), ExecUnit::TLOOKUP, kTlookupLatency},
        {"DIVIDE",    encode_div(5, 1, 2),     ExecUnit::DIVIDE,  kDivideLatency},
    };

    uint64_t prev_latency = 0;
    for (const auto& c : cases) {
        CAPTURE(c.name);
        const WbObservation obs = run_and_capture({c.raw, encode_ecall()},
                                                  /*dest_reg=*/5);
        REQUIRE(obs.seen);
        REQUIRE(obs.entry.source_unit == c.unit);

        const uint64_t latency = obs.commit_cycle - obs.entry.issue_cycle;
        REQUIRE(latency == c.expected_exec_latency);

        // Cross-check the measured latency against the scheduler's binding
        // writeback-slot offset — the value the issue gate reserves against
        // and that the arbiter asserts (count_fixed_with_result()<=1).
        const uint32_t offset = compute_issue_to_writeback_offset(
            c.unit, defaults.multiply_pipeline_stages, /*is_vdot8=*/false);
        REQUIRE(offset == latency + kFrontPipelineTransit);

        // Distinct pipelines, here strictly increasing latency in table order.
        REQUIRE(latency > prev_latency);
        prev_latency = latency;
    }
}

TEST_CASE("Instruction latency: configured multiply depth changes writeback offset",
          "[timing][latency]") {
    constexpr uint32_t kConfiguredMulStages = 5;
    const WbObservation obs = run_and_capture(
        {encode_mul(5, 1, 2), encode_ecall()}, /*dest_reg=*/5,
        kConfiguredMulStages);
    REQUIRE(obs.seen);
    REQUIRE(obs.entry.source_unit == ExecUnit::MULTIPLY);

    const uint64_t latency = obs.commit_cycle - obs.entry.issue_cycle;
    REQUIRE(latency == kConfiguredMulStages);

    constexpr uint32_t kFrontPipelineTransit = 2;
    const uint32_t runtime_offset = compute_issue_to_writeback_offset(
        ExecUnit::MULTIPLY, kConfiguredMulStages, /*is_vdot8=*/false);
    REQUIRE(runtime_offset == latency + kFrontPipelineTransit);
    REQUIRE(runtime_offset !=
            kIssueToWritebackOffset[exec_unit_index(ExecUnit::MULTIPLY)]);
}

TEST_CASE("Instruction latency: issue and operand-collection cycles are accounted for",
          "[timing][latency]") {
    // The test above folds the scheduler->opcoll->unit front pipeline into a
    // +2 constant. This case opens that constant up and confirms the two
    // front-end cycles are genuinely spent where expected:
    //
    //   front_span := issue_to_writeback_offset - (committed - issue_cycle)
    //              == 1  (the single REGISTERED scheduler->opcoll issue edge)
    //               + operand-collection cycles  (measured, not assumed)
    //
    // The operand-collection cycles are read directly from
    // Stats::operand_collector_busy_cycles. The issue stage is a single
    // REGISTERED handoff — the scheduler never holds an issued instruction for
    // more than one cycle — so its contribution is the irreducible "1" above,
    // isolated here as (front_span - operand-collection cycles).

    // A program of just [ECALL] isolates the trailing ECALL's own one-cycle
    // operand-collection occupancy, which is subtracted from every
    // [target, ECALL] program's total below.
    const WbObservation ecall_only = run_and_capture({encode_ecall()}, /*dest=*/5);
    REQUIRE(ecall_only.operand_collect_cycles == 1);
    const uint64_t ecall_opcoll = ecall_only.operand_collect_cycles;

    struct Case {
        const char* name;
        uint32_t raw;
        ExecUnit unit;
        bool is_vdot8;
        uint64_t expected_operand_collect_cycles;
    };
    const std::vector<Case> cases = {
        {"ALU/ADDI", encode_addi(5, 0, 7),    ExecUnit::ALU,      false, 1},
        {"MULTIPLY", encode_mul(5, 1, 2),     ExecUnit::MULTIPLY, false, 1},
        {"DIVIDE",   encode_div(5, 1, 2),     ExecUnit::DIVIDE,   false, 1},
        {"TLOOKUP",  encode_tlookup(5, 1, 0), ExecUnit::TLOOKUP,  false, 1},
        // VDOT8 reads three source registers and therefore spends two cycles
        // in operand collection rather than one.
        {"VDOT8",    encode_vdot8(5, 1, 2),   ExecUnit::MULTIPLY, true,  2},
    };

    for (const auto& c : cases) {
        CAPTURE(c.name);
        const WbObservation obs = run_and_capture({c.raw, encode_ecall()},
                                                  /*dest_reg=*/5);
        REQUIRE(obs.seen);

        // (1) Operand-collection cycles, measured directly.
        const uint64_t operand_collect =
            obs.operand_collect_cycles - ecall_opcoll;
        REQUIRE(operand_collect == c.expected_operand_collect_cycles);

        // (2) Decompose the front pipeline: issue->writeback offset minus the
        // measured unit-accept->writeback span is the issue->unit-accept span,
        // which must equal one issue edge plus the measured opcoll cycles.
        const uint32_t offset =
            compute_issue_to_writeback_offset(c.unit, c.is_vdot8);
        const uint64_t exec_span = obs.commit_cycle - obs.entry.issue_cycle;
        REQUIRE(offset > exec_span);
        const uint64_t front_span = offset - exec_span;
        REQUIRE(front_span == 1 + operand_collect);

        // (3) The issue stage itself is exactly one cycle — a single
        // REGISTERED scheduler->opcoll handoff, isolated from the opcoll span.
        REQUIRE(front_span - operand_collect == 1);
    }

    // (4) End-to-end punchline: MUL and VDOT8 share the MULTIPLY pipeline and
    // an identical [op, ECALL] front (same fetch/decode/issue cycle), differing
    // only in operand collection — 1 cycle vs 2. The extra opcoll cycle must
    // surface as exactly +1 cycle of issue->writeback latency, observable as
    // the writeback simply committing one cycle later.
    const WbObservation mul =
        run_and_capture({encode_mul(5, 1, 2), encode_ecall()}, /*dest=*/5);
    const WbObservation vdot8 =
        run_and_capture({encode_vdot8(5, 1, 2), encode_ecall()}, /*dest=*/5);
    REQUIRE(mul.seen);
    REQUIRE(vdot8.seen);
    REQUIRE(vdot8.commit_cycle == mul.commit_cycle + 1);
}

TEST_CASE("Instruction latency: the load pipeline is variable and exceeds the memory latency",
          "[timing][latency]") {
    // A first-touch load misses the L1 and traverses
    // coalescing -> cache -> external memory -> gather buffer -> arbiter, so
    // its issue->writeback latency is variable and necessarily exceeds both a
    // fixed-latency ALU op and the external-memory latency itself.
    const WbObservation load =
        run_and_capture({encode_lw(5, 0, 0), encode_ecall()}, /*dest_reg=*/5);
    REQUIRE(load.seen);

    const uint64_t load_latency = load.commit_cycle - load.entry.issue_cycle;
    REQUIRE(load_latency >= kProbeMemoryLatency);
    REQUIRE(load_latency > 1);  // strictly longer than the 1-cycle ALU pipeline
}

TEST_CASE("Writeback contention: load preemption extends a fixed-latency op's latency",
          "[timing][latency][contention]") {
    // Component-level: an ALUUnit and a variable-latency load source share one
    // writeback arbiter. When both present a result the same cycle the load
    // wins, the arbiter asserts the combinational-backward writeback stall, and
    // the ALU's gated commit() holds its result — slipping its writeback (and
    // its measured issue->writeback latency) by exactly one cycle per stall.

    struct Measurement {
        uint64_t latency = 0;
        uint64_t preempted_cycles = 0;
    };

    // Build a fresh ALU+arbiter, accept one ADDI x5 at issue cycle 0, and
    // inject `num_loads` contending loads on cycles 1..num_loads (once the ALU
    // result is committed and visible to the arbiter). Returns the ALU op's
    // measured issue->writeback latency and the preemption count.
    auto measure = [](int num_loads) -> Measurement {
        Stats stats;
        Scoreboard scoreboard;
        WritebackArbiter arbiter(scoreboard, stats);
        ALUUnit alu(stats);
        QueuedWritebackSource load(ExecUnit::LDST);
        arbiter.add_source(&alu);
        arbiter.add_source(&load);
        alu.set_writeback_arbiter(&arbiter);  // commit() self-gates on the stall

        // Pre-mark the ALU op's destination pending so the arbiter's
        // clear_pending() on retirement is well-formed.
        scoreboard.seed_next();
        scoreboard.set_pending(0, 5);
        scoreboard.commit();

        // The ALU accepts ADDI x5 at issue cycle 0 (opcoll_ is null in this
        // isolated harness, so the unit is driven through accept() directly).
        DispatchInput in;
        in.decoded = Decoder::decode(encode_addi(5, 0, 1));
        in.warp_id = 0;
        in.pc = 0;
        in.trace.warp_id = 0;
        in.trace.pc = 0;
        in.trace.decoded = in.decoded;
        in.trace.results.fill(0);
        const uint64_t issue_cycle = 0;
        alu.accept(in, issue_cycle);

        Measurement m;
        bool retired = false;
        for (uint64_t cycle = 0; cycle < 64 && !retired; ++cycle) {
            // Inject one contending load per cycle for cycles 1..num_loads.
            if (cycle >= 1 && cycle <= static_cast<uint64_t>(num_loads)) {
                WritebackEntry le;
                le.valid = true;
                le.warp_id = 0;
                le.dest_reg = 0;  // x0 destination: no scoreboard interaction
                le.source_unit = ExecUnit::LDST;
                load.enqueue(le);
            }
            // Evaluate sweep order: the arbiter is sequenced first so its
            // combinational-backward stall is visible to the ALU's commit().
            alu.seed_next();
            scoreboard.seed_next();
            arbiter.evaluate();
            alu.evaluate();
            arbiter.commit();
            alu.commit();  // gated on arbiter.next_writeback_stall()
            scoreboard.commit();

            const auto& wb = arbiter.current_committed_entry();
            if (wb.has_value() && wb->source_unit == ExecUnit::ALU) {
                m.latency = cycle - issue_cycle;
                retired = true;
            }
        }
        REQUIRE(retired);
        m.preempted_cycles = stats.fixed_writeback_preempted_cycles;
        return m;
    };

    // Baseline: no contention — the ALU op retires at its 1-cycle latency.
    const Measurement baseline = measure(0);
    REQUIRE(baseline.preempted_cycles == 0);
    REQUIRE(baseline.latency == 1);

    // One preempting load: the writeback slips exactly one cycle.
    const Measurement one = measure(1);
    REQUIRE(one.preempted_cycles == 1);
    REQUIRE(one.latency == baseline.latency + 1);

    // Two consecutive preempting loads: the writeback slips two cycles — the
    // stall count is reflected one-for-one in the instruction's latency.
    const Measurement two = measure(2);
    REQUIRE(two.preempted_cycles == 2);
    REQUIRE(two.latency == baseline.latency + 2);
}
