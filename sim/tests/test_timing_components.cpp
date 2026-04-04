#include "catch.hpp"
#include "gpu_sim/config.h"
#include "gpu_sim/decoder.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/isa.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/timing/alu_unit.h"
#include "gpu_sim/timing/branch_predictor.h"
#include "gpu_sim/timing/decode_stage.h"
#include "gpu_sim/timing/divide_unit.h"
#include "gpu_sim/timing/fetch_stage.h"
#include "gpu_sim/timing/ldst_unit.h"
#include "gpu_sim/timing/memory_interface.h"
#include "gpu_sim/timing/multiply_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/timing/tlookup_unit.h"
#include "gpu_sim/timing/writeback_arbiter.h"

using namespace gpu_sim;

static uint32_t i_type(int32_t imm, uint32_t rs1, uint32_t funct3,
                       uint32_t rd, uint32_t opcode) {
    return (static_cast<uint32_t>(imm & 0xFFF) << 20) | (rs1 << 15) |
           (funct3 << 12) | (rd << 7) | opcode;
}

static uint32_t r_type(uint32_t funct7, uint32_t rs2, uint32_t rs1,
                       uint32_t funct3, uint32_t rd, uint32_t opcode) {
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) |
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

static uint32_t j_type(int32_t imm, uint32_t rd, uint32_t opcode) {
    uint32_t bit20 = (imm >> 20) & 0x1;
    uint32_t bits10_1 = (imm >> 1) & 0x3FF;
    uint32_t bit11 = (imm >> 11) & 0x1;
    uint32_t bits19_12 = (imm >> 12) & 0xFF;
    return (bit20 << 31) | (bits19_12 << 12) | (bit11 << 20) |
           (bits10_1 << 21) | (rd << 7) | opcode;
}

static IssueOutput make_issue_output(uint32_t raw, uint32_t warp_id = 0, uint32_t pc = 0) {
    IssueOutput issue;
    issue.decoded = Decoder::decode(raw);
    issue.warp_id = warp_id;
    issue.pc = pc;
    issue.trace.warp_id = warp_id;
    issue.trace.pc = pc;
    issue.trace.decoded = issue.decoded;
    issue.trace.results.fill(0);
    issue.trace.results[0] = 0x1234;
    return issue;
}

class StubExecutionUnit : public ExecutionUnit {
public:
    explicit StubExecutionUnit(ExecUnit type) : type_(type) {}

    void evaluate() override {}
    void commit() override {}
    void reset() override { result_.valid = false; }
    bool is_ready() const override { return true; }
    bool has_result() const override { return result_.valid; }
    WritebackEntry consume_result() override {
        WritebackEntry entry = result_;
        result_.valid = false;
        return entry;
    }
    ExecUnit get_type() const override { return type_; }

    void set_result(uint32_t warp, uint8_t reg) {
        result_.valid = true;
        result_.warp_id = warp;
        result_.dest_reg = reg;
        result_.source_unit = type_;
    }

private:
    ExecUnit type_;
    WritebackEntry result_;
};

TEST_CASE("FetchStage: redirect flushes buffered state for a warp", "[timing]") {
    Stats stats;
    StaticDirectionalBranchPredictor predictor;
    InstructionMemory imem(64);
    imem.write(0, i_type(1, 0, isa::FUNCT3_ADD_SUB, 5, isa::OP_ALU_I));

    std::vector<WarpState> warps;
    warps.emplace_back(2);
    warps[0].reset(0);
    warps[0].instr_buffer.push(BufferEntry{});

    FetchStage fetch(1, warps.data(), imem, predictor, stats);
    fetch.evaluate();
    fetch.commit();

    REQUIRE(fetch.current_output().has_value());
    fetch.redirect_warp(0, 16);

    REQUIRE_FALSE(fetch.current_output().has_value());
    REQUIRE(warps[0].pc == 16);
    REQUIRE(warps[0].instr_buffer.is_empty());
}

TEST_CASE("FetchStage: backward branch prediction steers the warp PC", "[timing]") {
    Stats stats;
    StaticDirectionalBranchPredictor predictor;
    InstructionMemory imem(512);
    imem.write(64, b_type(-8, 2, 1, isa::FUNCT3_BNE, isa::OP_BRANCH));

    std::vector<WarpState> warps;
    warps.emplace_back(2);
    warps[0].reset(0x100);

    FetchStage fetch(1, warps.data(), imem, predictor, stats);
    fetch.evaluate();
    fetch.commit();

    REQUIRE(fetch.current_output().has_value());
    REQUIRE(fetch.current_output()->prediction.predicted_taken);
    REQUIRE(fetch.current_output()->prediction.predicted_target == 0x0F8);
    REQUIRE(warps[0].pc == 0x0F8);
}

TEST_CASE("DecodeStage: EBREAK is detected and not enqueued", "[timing]") {
    Stats stats;
    StaticDirectionalBranchPredictor predictor;
    InstructionMemory imem(64);
    imem.write(0, i_type(1, 0, 0, 0, isa::OP_SYSTEM));

    std::vector<WarpState> warps;
    warps.emplace_back(2);
    warps[0].reset(0);

    FetchStage fetch(1, warps.data(), imem, predictor, stats);
    fetch.evaluate();
    fetch.commit();

    DecodeStage decode(warps.data(), fetch);
    decode.evaluate();
    decode.commit();

    REQUIRE(decode.ebreak_detected());
    REQUIRE(decode.ebreak_warp() == 0);
    REQUIRE(decode.ebreak_pc() == 0);
    REQUIRE(warps[0].instr_buffer.is_empty());
}

TEST_CASE("DecodeStage: invalidate_warp drops pending decode before commit", "[timing]") {
    Stats stats;
    StaticDirectionalBranchPredictor predictor;
    InstructionMemory imem(64);
    imem.write(0, i_type(7, 0, isa::FUNCT3_ADD_SUB, 5, isa::OP_ALU_I));

    std::vector<WarpState> warps;
    warps.emplace_back(2);
    warps[0].reset(0);

    FetchStage fetch(1, warps.data(), imem, predictor, stats);
    fetch.evaluate();
    fetch.commit();

    DecodeStage decode(warps.data(), fetch);
    decode.evaluate();
    decode.invalidate_warp(0);
    decode.commit();

    REQUIRE(warps[0].instr_buffer.is_empty());
}

TEST_CASE("Fetch skips warp with full buffer", "[timing]") {
    Stats stats;
    StaticDirectionalBranchPredictor predictor;
    InstructionMemory imem(64);
    imem.write(0, i_type(7, 0, isa::FUNCT3_ADD_SUB, 5, isa::OP_ALU_I));
    imem.write(1, i_type(9, 0, isa::FUNCT3_ADD_SUB, 6, isa::OP_ALU_I));

    std::vector<WarpState> warps;
    warps.emplace_back(1);  // buffer depth 1
    warps[0].reset(0);

    FetchStage fetch(1, warps.data(), imem, predictor, stats);

    // First fetch succeeds
    fetch.evaluate();
    fetch.commit();
    REQUIRE(fetch.current_output().has_value());

    // Simulate decode consuming the output
    fetch.consume_output();

    // Fill the buffer so the warp becomes ineligible for fetch
    warps[0].instr_buffer.push(BufferEntry{});
    REQUIRE(warps[0].instr_buffer.is_full());

    // Fetch evaluates but skips the warp (buffer full) — no output produced
    fetch.evaluate();
    fetch.commit();
    REQUIRE_FALSE(fetch.current_output().has_value());

    // Free a buffer slot — fetch should succeed again
    warps[0].instr_buffer.pop();
    fetch.evaluate();
    fetch.commit();
    REQUIRE(fetch.current_output().has_value());
}

TEST_CASE("Fetch stalls when decode has unconsumed output", "[timing]") {
    Stats stats;
    StaticDirectionalBranchPredictor predictor;
    InstructionMemory imem(64);
    imem.write(0, i_type(7, 0, isa::FUNCT3_ADD_SUB, 5, isa::OP_ALU_I));
    imem.write(1, i_type(9, 0, isa::FUNCT3_ADD_SUB, 6, isa::OP_ALU_I));

    std::vector<WarpState> warps;
    warps.emplace_back(2);  // buffer depth 2
    warps[0].reset(0);

    FetchStage fetch(1, warps.data(), imem, predictor, stats);

    // First fetch succeeds — PC advances to 4
    fetch.evaluate();
    fetch.commit();
    REQUIRE(fetch.current_output().has_value());
    uint32_t pc_after_first = warps[0].pc;
    REQUIRE(pc_after_first == 4);

    // Do NOT consume — simulate decode being blocked
    // Second fetch should stall (backpressure) — PC must not advance
    fetch.evaluate();
    fetch.commit();
    REQUIRE(fetch.current_output().has_value());  // retains first output
    REQUIRE(warps[0].pc == pc_after_first);       // PC unchanged

    // Now consume — next fetch should proceed
    fetch.consume_output();
    fetch.evaluate();
    fetch.commit();
    REQUIRE(fetch.current_output().has_value());
    REQUIRE(warps[0].pc == 8);  // PC advanced to next instruction
}

TEST_CASE("BranchPredictor: backward branches taken, forward branches not taken", "[timing]") {
    StaticDirectionalBranchPredictor predictor;

    const uint32_t forward_branch = b_type(8, 2, 1, isa::FUNCT3_BNE, isa::OP_BRANCH);
    const auto forward_prediction = predictor.predict(0x100, forward_branch);
    REQUIRE(forward_prediction.is_control_flow);
    REQUIRE_FALSE(forward_prediction.predicted_taken);
    REQUIRE(forward_prediction.predicted_target == 0x108);

    const uint32_t backward_branch = b_type(-8, 2, 1, isa::FUNCT3_BNE, isa::OP_BRANCH);
    const auto backward_prediction = predictor.predict(0x100, backward_branch);
    REQUIRE(backward_prediction.is_control_flow);
    REQUIRE(backward_prediction.predicted_taken);
    REQUIRE(backward_prediction.predicted_target == 0x0F8);

    const uint32_t jal = j_type(8, 5, isa::OP_JAL);
    const auto jal_prediction = predictor.predict(0x100, jal);
    REQUIRE(jal_prediction.is_control_flow);
    REQUIRE(jal_prediction.predicted_taken);
    REQUIRE(jal_prediction.predicted_target == 0x108);

    const uint32_t jalr = i_type(0, 1, 0, 5, isa::OP_JALR);
    const auto jalr_prediction = predictor.predict(0x100, jalr);
    REQUIRE(jalr_prediction.is_control_flow);
    REQUIRE_FALSE(jalr_prediction.predicted_taken);
}

TEST_CASE("OperandCollector: standard ops take one cycle and VDOT8 takes two", "[timing]") {
    Stats stats;
    OperandCollector opcoll(stats);

    auto addi_issue = make_issue_output(i_type(1, 0, isa::FUNCT3_ADD_SUB, 5, isa::OP_ALU_I));
    opcoll.accept(addi_issue);
    REQUIRE_FALSE(opcoll.is_free());
    opcoll.evaluate();
    REQUIRE(opcoll.output().has_value());
    REQUIRE(opcoll.is_free());

    opcoll.reset();

    auto vdot_issue = make_issue_output(r_type(0x00, 2, 1, 0x0, 5, isa::OP_VDOT8));
    opcoll.accept(vdot_issue);
    opcoll.evaluate();
    REQUIRE_FALSE(opcoll.output().has_value());
    REQUIRE_FALSE(opcoll.is_free());
    opcoll.evaluate();
    REQUIRE(opcoll.output().has_value());
    REQUIRE(opcoll.is_free());
}

TEST_CASE("ALUUnit: accepted work produces one-cycle writeback", "[timing]") {
    Stats stats;
    ALUUnit alu(stats);
    auto issue = make_issue_output(i_type(1, 0, isa::FUNCT3_ADD_SUB, 5, isa::OP_ALU_I));

    alu.accept(DispatchInput{issue.decoded, issue.trace, issue.warp_id, issue.pc}, 12);
    REQUIRE_FALSE(alu.is_ready());
    alu.evaluate();
    REQUIRE(alu.has_result());

    auto wb = alu.consume_result();
    REQUIRE(wb.valid);
    REQUIRE(wb.dest_reg == 5);
    REQUIRE(wb.issue_cycle == 12);
    REQUIRE(alu.is_ready());
}

TEST_CASE("MultiplyUnit: stalled head result resumes after writeback consumption", "[timing]") {
    Stats stats;
    MultiplyUnit mul(1, stats);
    auto first = make_issue_output(r_type(isa::FUNCT7_MULDIV, 2, 1, isa::FUNCT3_MUL, 5,
                                          isa::OP_ALU_R));
    auto second = make_issue_output(r_type(isa::FUNCT7_MULDIV, 2, 1, isa::FUNCT3_MUL, 6,
                                           isa::OP_ALU_R));

    mul.accept(DispatchInput{first.decoded, first.trace, first.warp_id, first.pc}, 1);
    mul.evaluate();
    REQUIRE(mul.has_result());

    mul.accept(DispatchInput{second.decoded, second.trace, second.warp_id, second.pc}, 2);
    mul.evaluate();
    REQUIRE(mul.has_result());
    REQUIRE_FALSE(mul.is_ready());

    auto first_wb = mul.consume_result();
    REQUIRE(first_wb.dest_reg == 5);

    mul.evaluate();
    REQUIRE(mul.has_result());
    auto second_wb = mul.consume_result();
    REQUIRE(second_wb.dest_reg == 6);
    REQUIRE(mul.is_ready());
}

TEST_CASE("DivideUnit: result appears after fixed latency", "[timing]") {
    Stats stats;
    DivideUnit div(stats);
    auto issue = make_issue_output(r_type(isa::FUNCT7_MULDIV, 2, 1, isa::FUNCT3_DIV, 5,
                                          isa::OP_ALU_R));

    div.accept(DispatchInput{issue.decoded, issue.trace, issue.warp_id, issue.pc}, 3);
    for (int i = 0; i < 31; ++i) {
        REQUIRE_FALSE(div.has_result());
        div.evaluate();
    }
    REQUIRE_FALSE(div.has_result());
    div.evaluate();
    REQUIRE(div.has_result());
}

TEST_CASE("TLookupUnit: result appears after 17 cycles", "[timing]") {
    Stats stats;
    TLookupUnit tlookup(stats);
    auto issue = make_issue_output(i_type(16, 1, 0, 5, isa::OP_TLOOKUP));

    tlookup.accept(DispatchInput{issue.decoded, issue.trace, issue.warp_id, issue.pc}, 4);
    for (int i = 0; i < 16; ++i) {
        REQUIRE_FALSE(tlookup.has_result());
        tlookup.evaluate();
    }
    REQUIRE_FALSE(tlookup.has_result());
    tlookup.evaluate();
    REQUIRE(tlookup.has_result());
}

TEST_CASE("TLookupUnit: busy signal true for exactly 17 cycles", "[timing][tlookup]") {
    Stats stats;
    TLookupUnit tlookup(stats);
    auto issue = make_issue_output(i_type(16, 1, 0, 5, isa::OP_TLOOKUP));

    REQUIRE_FALSE(tlookup.busy());

    tlookup.accept(DispatchInput{issue.decoded, issue.trace, issue.warp_id, issue.pc}, 0);
    REQUIRE(tlookup.busy());

    // busy must remain true for all 17 evaluate calls
    for (uint32_t i = 0; i < 17; ++i) {
        REQUIRE(tlookup.busy());
        tlookup.evaluate();
    }
    // After 17 evaluations, busy must be false
    REQUIRE_FALSE(tlookup.busy());
}

TEST_CASE("TLookupUnit: result not available at cycle 16, available at 17", "[timing][tlookup]") {
    Stats stats;
    TLookupUnit tlookup(stats);
    auto issue = make_issue_output(i_type(16, 1, 0, 5, isa::OP_TLOOKUP));

    tlookup.accept(DispatchInput{issue.decoded, issue.trace, issue.warp_id, issue.pc}, 0);

    // Advance 16 evaluations -- result must NOT be ready
    for (uint32_t i = 0; i < 16; ++i) {
        tlookup.evaluate();
    }
    REQUIRE_FALSE(tlookup.has_result());

    // One more evaluation (17th) -- result must appear
    tlookup.evaluate();
    REQUIRE(tlookup.has_result());
}

TEST_CASE("TLookupUnit: cycles_remaining counts down from 17 to 0", "[timing][tlookup]") {
    Stats stats;
    TLookupUnit tlookup(stats);
    auto issue = make_issue_output(i_type(16, 1, 0, 5, isa::OP_TLOOKUP));

    tlookup.accept(DispatchInput{issue.decoded, issue.trace, issue.warp_id, issue.pc}, 0);
    REQUIRE(tlookup.cycles_remaining() == 17);

    for (uint32_t i = 17; i > 0; --i) {
        REQUIRE(tlookup.cycles_remaining() == i);
        tlookup.evaluate();
    }
    REQUIRE(tlookup.cycles_remaining() == 0);
}

TEST_CASE("TLookupUnit: is_ready blocks while busy and while result unconsumed", "[timing][tlookup]") {
    Stats stats;
    TLookupUnit tlookup(stats);
    auto issue = make_issue_output(i_type(16, 1, 0, 5, isa::OP_TLOOKUP));

    // Initially ready
    REQUIRE(tlookup.is_ready());

    // Not ready while busy
    tlookup.accept(DispatchInput{issue.decoded, issue.trace, issue.warp_id, issue.pc}, 0);
    REQUIRE_FALSE(tlookup.is_ready());

    for (uint32_t i = 0; i < 17; ++i) {
        REQUIRE_FALSE(tlookup.is_ready());
        tlookup.evaluate();
    }

    // Result is available but unconsumed -- still not ready
    REQUIRE(tlookup.has_result());
    REQUIRE_FALSE(tlookup.is_ready());

    // After consuming, should be ready again
    tlookup.consume_result();
    REQUIRE(tlookup.is_ready());
}

TEST_CASE("TLookupUnit: back-to-back dispatch completes in 17 cycles each", "[timing][tlookup]") {
    Stats stats;
    TLookupUnit tlookup(stats);
    auto first = make_issue_output(i_type(16, 1, 0, 5, isa::OP_TLOOKUP));
    auto second = make_issue_output(i_type(16, 1, 0, 6, isa::OP_TLOOKUP), 1);

    // First TLOOKUP
    tlookup.accept(DispatchInput{first.decoded, first.trace, first.warp_id, first.pc}, 0);
    for (uint32_t i = 0; i < 17; ++i) {
        tlookup.evaluate();
    }
    REQUIRE(tlookup.has_result());
    auto wb1 = tlookup.consume_result();
    REQUIRE(wb1.dest_reg == 5);
    REQUIRE(wb1.warp_id == 0);
    REQUIRE(tlookup.is_ready());

    // Second TLOOKUP immediately after consuming
    tlookup.accept(DispatchInput{second.decoded, second.trace, second.warp_id, second.pc}, 17);

    // Must not produce result before 17 evaluations
    for (uint32_t i = 0; i < 16; ++i) {
        REQUIRE_FALSE(tlookup.has_result());
        tlookup.evaluate();
    }
    REQUIRE_FALSE(tlookup.has_result());
    tlookup.evaluate();
    REQUIRE(tlookup.has_result());

    auto wb2 = tlookup.consume_result();
    REQUIRE(wb2.dest_reg == 6);
    REQUIRE(wb2.warp_id == 1);
    REQUIRE(wb2.issue_cycle == 17);
}

TEST_CASE("TLookupUnit: stats track busy_cycles=17 and instructions=1 per dispatch", "[timing][tlookup]") {
    Stats stats;
    TLookupUnit tlookup(stats);
    auto issue = make_issue_output(i_type(16, 1, 0, 5, isa::OP_TLOOKUP));

    REQUIRE(stats.tlookup_stats.busy_cycles == 0);
    REQUIRE(stats.tlookup_stats.instructions == 0);

    tlookup.accept(DispatchInput{issue.decoded, issue.trace, issue.warp_id, issue.pc}, 0);
    REQUIRE(stats.tlookup_stats.instructions == 1);

    for (uint32_t i = 0; i < 17; ++i) {
        tlookup.evaluate();
    }
    REQUIRE(stats.tlookup_stats.busy_cycles == 17);
    REQUIRE(stats.tlookup_stats.instructions == 1);

    // Second dispatch -- stats should accumulate
    tlookup.consume_result();
    auto second = make_issue_output(i_type(16, 1, 0, 6, isa::OP_TLOOKUP));
    tlookup.accept(DispatchInput{second.decoded, second.trace, second.warp_id, second.pc}, 17);
    REQUIRE(stats.tlookup_stats.instructions == 2);

    for (uint32_t i = 0; i < 17; ++i) {
        tlookup.evaluate();
    }
    REQUIRE(stats.tlookup_stats.busy_cycles == 34);
    REQUIRE(stats.tlookup_stats.instructions == 2);
}

TEST_CASE("TLookupUnit: reset clears all state", "[timing][tlookup]") {
    Stats stats;
    TLookupUnit tlookup(stats);
    auto issue = make_issue_output(i_type(16, 1, 0, 5, isa::OP_TLOOKUP));

    // Put unit in the middle of processing
    tlookup.accept(DispatchInput{issue.decoded, issue.trace, issue.warp_id, issue.pc}, 0);
    for (uint32_t i = 0; i < 8; ++i) {
        tlookup.evaluate();
    }
    REQUIRE(tlookup.busy());

    tlookup.reset();
    REQUIRE_FALSE(tlookup.busy());
    REQUIRE(tlookup.cycles_remaining() == 0);
    REQUIRE_FALSE(tlookup.has_result());
    REQUIRE(tlookup.is_ready());
}

TEST_CASE("TLookupUnit: consume_result returns correct writeback metadata", "[timing][tlookup]") {
    Stats stats;
    TLookupUnit tlookup(stats);
    auto issue = make_issue_output(i_type(16, 1, 0, 7, isa::OP_TLOOKUP), 3, 0x100);

    tlookup.accept(DispatchInput{issue.decoded, issue.trace, issue.warp_id, issue.pc}, 42);

    for (uint32_t i = 0; i < 17; ++i) {
        tlookup.evaluate();
    }

    auto wb = tlookup.consume_result();
    REQUIRE(wb.valid);
    REQUIRE(wb.warp_id == 3);
    REQUIRE(wb.dest_reg == 7);
    REQUIRE(wb.source_unit == ExecUnit::TLOOKUP);
    REQUIRE(wb.pc == 0x100);
    REQUIRE(wb.issue_cycle == 42);
    REQUIRE(wb.values[0] == 0x1234);  // from make_issue_output
}

TEST_CASE("TLookupUnit: active_warp and pending_entry accessors", "[timing][tlookup]") {
    Stats stats;
    TLookupUnit tlookup(stats);
    auto issue = make_issue_output(i_type(16, 1, 0, 5, isa::OP_TLOOKUP), 2);

    // Before accept: no active warp
    REQUIRE_FALSE(tlookup.active_warp().has_value());
    REQUIRE(tlookup.pending_entry() == nullptr);
    REQUIRE(tlookup.result_entry() == nullptr);

    tlookup.accept(DispatchInput{issue.decoded, issue.trace, issue.warp_id, issue.pc}, 0);
    REQUIRE(tlookup.active_warp().has_value());
    REQUIRE(tlookup.active_warp().value() == 2);
    REQUIRE(tlookup.pending_entry() != nullptr);
    REQUIRE(tlookup.pending_entry()->warp_id == 2);

    // Complete the operation
    for (uint32_t i = 0; i < 17; ++i) {
        tlookup.evaluate();
    }

    // No longer busy, so active_warp and pending_entry should be cleared
    REQUIRE_FALSE(tlookup.active_warp().has_value());
    REQUIRE(tlookup.pending_entry() == nullptr);
    // But result_entry should be available
    REQUIRE(tlookup.result_entry() != nullptr);
    REQUIRE(tlookup.result_entry()->warp_id == 2);

    // After consume, result_entry cleared
    tlookup.consume_result();
    REQUIRE(tlookup.result_entry() == nullptr);
}

TEST_CASE("TLookupUnit: get_type returns TLOOKUP", "[timing][tlookup]") {
    Stats stats;
    TLookupUnit tlookup(stats);
    REQUIRE(tlookup.get_type() == ExecUnit::TLOOKUP);
}

TEST_CASE("LdStUnit: full address FIFO backpressures completion until a pop", "[timing]") {
    Stats stats;
    LdStUnit ldst(32, 1, stats);
    auto first = make_issue_output(i_type(0, 1, isa::FUNCT3_LW, 5, isa::OP_LOAD));
    first.trace.is_load = true;
    auto second = make_issue_output(i_type(4, 1, isa::FUNCT3_LW, 6, isa::OP_LOAD));
    second.trace.is_load = true;

    ldst.accept(DispatchInput{first.decoded, first.trace, first.warp_id, first.pc}, 1);
    ldst.evaluate();
    REQUIRE_FALSE(ldst.fifo_empty());
    REQUIRE(ldst.is_ready());

    ldst.accept(DispatchInput{second.decoded, second.trace, second.warp_id, second.pc}, 2);
    ldst.evaluate();
    REQUIRE_FALSE(ldst.is_ready());

    ldst.fifo_pop();
    ldst.evaluate();
    REQUIRE_FALSE(ldst.fifo_empty());
    REQUIRE(ldst.is_ready());
}

TEST_CASE("MemoryInterface: fixed latency responses preserve submission order", "[timing]") {
    Stats stats;
    ExternalMemoryInterface mem_if(2, stats);

    mem_if.submit_read(10, 1);
    mem_if.submit_write(20);
    REQUIRE_FALSE(mem_if.is_idle());

    mem_if.evaluate();
    REQUIRE_FALSE(mem_if.has_response());
    mem_if.evaluate();
    REQUIRE(mem_if.has_response());

    auto first = mem_if.get_response();
    REQUIRE_FALSE(first.is_write);
    REQUIRE(first.line_addr == 10);
    REQUIRE(first.mshr_id == 1);

    auto second = mem_if.get_response();
    REQUIRE(second.is_write);
    REQUIRE(second.line_addr == 20);
    REQUIRE(mem_if.is_idle());
}

TEST_CASE("WritebackArbiter: round-robin arbitration clears scoreboard in order", "[timing]") {
    Stats stats;
    Scoreboard scoreboard;
    WritebackArbiter arbiter(scoreboard, stats);
    StubExecutionUnit alu(ExecUnit::ALU);
    StubExecutionUnit mul(ExecUnit::MULTIPLY);
    arbiter.add_source(&alu);
    arbiter.add_source(&mul);

    scoreboard.seed_next();
    scoreboard.set_pending(0, 5);
    scoreboard.set_pending(0, 6);
    scoreboard.commit();

    alu.set_result(0, 5);
    mul.set_result(0, 6);

    scoreboard.seed_next();
    arbiter.evaluate();
    arbiter.commit();
    scoreboard.commit();
    REQUIRE(arbiter.committed_entry().has_value());
    REQUIRE(arbiter.committed_entry()->dest_reg == 5);
    REQUIRE_FALSE(scoreboard.is_pending(0, 5));
    REQUIRE(scoreboard.is_pending(0, 6));
    REQUIRE(stats.writeback_conflicts == 1);

    scoreboard.seed_next();
    arbiter.evaluate();
    arbiter.commit();
    scoreboard.commit();
    REQUIRE(arbiter.committed_entry().has_value());
    REQUIRE(arbiter.committed_entry()->dest_reg == 6);
    REQUIRE_FALSE(scoreboard.is_pending(0, 6));
}

TEST_CASE("WritebackArbiter: queued memory sources preserve simultaneous hit and fill",
          "[timing]") {
    Stats stats;
    Scoreboard scoreboard;
    WritebackArbiter arbiter(scoreboard, stats);
    QueuedWritebackSource hit_source(ExecUnit::LDST);
    QueuedWritebackSource fill_source(ExecUnit::LDST);
    arbiter.add_source(&hit_source);
    arbiter.add_source(&fill_source);

    scoreboard.seed_next();
    scoreboard.set_pending(0, 5);
    scoreboard.set_pending(0, 6);
    scoreboard.commit();

    WritebackEntry hit_entry;
    hit_entry.valid = true;
    hit_entry.warp_id = 0;
    hit_entry.dest_reg = 5;
    hit_entry.source_unit = ExecUnit::LDST;
    hit_source.enqueue(hit_entry);

    WritebackEntry fill_entry;
    fill_entry.valid = true;
    fill_entry.warp_id = 0;
    fill_entry.dest_reg = 6;
    fill_entry.source_unit = ExecUnit::LDST;
    fill_source.enqueue(fill_entry);

    scoreboard.seed_next();
    arbiter.evaluate();
    arbiter.commit();
    scoreboard.commit();
    REQUIRE(arbiter.committed_entry().has_value());
    REQUIRE(arbiter.committed_entry()->dest_reg == 5);
    REQUIRE_FALSE(scoreboard.is_pending(0, 5));
    REQUIRE(scoreboard.is_pending(0, 6));
    REQUIRE(fill_source.has_result());
    REQUIRE(stats.writeback_conflicts == 1);

    scoreboard.seed_next();
    arbiter.evaluate();
    arbiter.commit();
    scoreboard.commit();
    REQUIRE(arbiter.committed_entry().has_value());
    REQUIRE(arbiter.committed_entry()->dest_reg == 6);
    REQUIRE_FALSE(scoreboard.is_pending(0, 6));
    REQUIRE_FALSE(fill_source.has_result());
}

// ---------------------------------------------------------------------------
// Instruction buffer depth-3 default: adversarial tests
// ---------------------------------------------------------------------------

TEST_CASE("InstructionBuffer: default SimConfig has depth 3", "[ibuffer]") {
    SimConfig cfg;
    REQUIRE(cfg.instruction_buffer_depth == 3);
}

TEST_CASE("InstructionBuffer: default WarpState creates buffer of depth 3", "[ibuffer]") {
    WarpState ws;
    REQUIRE(ws.instr_buffer.capacity() == 3);
    REQUIRE(ws.instr_buffer.is_empty());
    REQUIRE_FALSE(ws.instr_buffer.is_full());
}

TEST_CASE("InstructionBuffer: accepts exactly 3 entries then reports full", "[ibuffer]") {
    InstructionBuffer buf(3);
    REQUIRE(buf.capacity() == 3);

    buf.push(BufferEntry{});
    REQUIRE(buf.size() == 1);
    REQUIRE_FALSE(buf.is_full());

    buf.push(BufferEntry{});
    REQUIRE(buf.size() == 2);
    REQUIRE_FALSE(buf.is_full());

    buf.push(BufferEntry{});
    REQUIRE(buf.size() == 3);
    REQUIRE(buf.is_full());

    // 4th push must be silently dropped (backpressure)
    buf.push(BufferEntry{});
    REQUIRE(buf.size() == 3);
    REQUIRE(buf.is_full());
}

TEST_CASE("InstructionBuffer: depth-1 boundary — 2 entries not full, 3 entries full", "[ibuffer]") {
    InstructionBuffer buf(3);

    buf.push(BufferEntry{});
    buf.push(BufferEntry{});
    REQUIRE(buf.size() == 2);
    REQUIRE_FALSE(buf.is_full());

    buf.push(BufferEntry{});
    REQUIRE(buf.size() == 3);
    REQUIRE(buf.is_full());
}

TEST_CASE("InstructionBuffer: drain one from full restores space", "[ibuffer]") {
    InstructionBuffer buf(3);
    buf.push(BufferEntry{});
    buf.push(BufferEntry{});
    buf.push(BufferEntry{});
    REQUIRE(buf.is_full());

    buf.pop();
    REQUIRE(buf.size() == 2);
    REQUIRE_FALSE(buf.is_full());
    REQUIRE_FALSE(buf.is_empty());

    // Can accept another entry after draining
    buf.push(BufferEntry{});
    REQUIRE(buf.size() == 3);
    REQUIRE(buf.is_full());
}

TEST_CASE("InstructionBuffer: flush from full empties completely", "[ibuffer]") {
    InstructionBuffer buf(3);
    buf.push(BufferEntry{});
    buf.push(BufferEntry{});
    buf.push(BufferEntry{});
    REQUIRE(buf.is_full());

    buf.flush();
    REQUIRE(buf.is_empty());
    REQUIRE(buf.size() == 0);
    REQUIRE_FALSE(buf.is_full());
}

TEST_CASE("InstructionBuffer: FIFO ordering preserved through depth-3 buffer", "[ibuffer]") {
    InstructionBuffer buf(3);

    BufferEntry e0{};
    e0.pc = 0x100;
    BufferEntry e1{};
    e1.pc = 0x104;
    BufferEntry e2{};
    e2.pc = 0x108;

    buf.push(e0);
    buf.push(e1);
    buf.push(e2);

    REQUIRE(buf.front().pc == 0x100);
    buf.pop();
    REQUIRE(buf.front().pc == 0x104);
    buf.pop();
    REQUIRE(buf.front().pc == 0x108);
    buf.pop();
    REQUIRE(buf.is_empty());
}

TEST_CASE("Fetch: warp with 2 of 3 buffer slots filled is still eligible", "[ibuffer][timing]") {
    Stats stats;
    StaticDirectionalBranchPredictor predictor;
    InstructionMemory imem(64);
    imem.write(0, i_type(7, 0, isa::FUNCT3_ADD_SUB, 5, isa::OP_ALU_I));

    std::vector<WarpState> warps;
    warps.emplace_back(3);  // depth 3
    warps[0].reset(0);

    // Fill 2 of 3 slots — warp should still be eligible
    warps[0].instr_buffer.push(BufferEntry{});
    warps[0].instr_buffer.push(BufferEntry{});
    REQUIRE(warps[0].instr_buffer.size() == 2);
    REQUIRE_FALSE(warps[0].instr_buffer.is_full());

    FetchStage fetch(1, warps.data(), imem, predictor, stats);
    fetch.evaluate();
    fetch.commit();
    REQUIRE(fetch.current_output().has_value());
}

TEST_CASE("Fetch: warp with 3 of 3 buffer slots filled is skipped", "[ibuffer][timing]") {
    Stats stats;
    StaticDirectionalBranchPredictor predictor;
    InstructionMemory imem(64);
    imem.write(0, i_type(7, 0, isa::FUNCT3_ADD_SUB, 5, isa::OP_ALU_I));

    std::vector<WarpState> warps;
    warps.emplace_back(3);  // depth 3
    warps[0].reset(0);

    // Fill all 3 slots — warp should be skipped
    warps[0].instr_buffer.push(BufferEntry{});
    warps[0].instr_buffer.push(BufferEntry{});
    warps[0].instr_buffer.push(BufferEntry{});
    REQUIRE(warps[0].instr_buffer.is_full());

    FetchStage fetch(1, warps.data(), imem, predictor, stats);
    fetch.evaluate();
    fetch.commit();
    REQUIRE_FALSE(fetch.current_output().has_value());
}

TEST_CASE("Fetch: decode-pending warp with 2 of 3 slots treated as full", "[ibuffer][timing]") {
    // Per spec: if decode has a pending instruction targeting warp W and W's
    // buffer has only one free slot, fetch treats W's buffer as full.
    Stats stats;
    StaticDirectionalBranchPredictor predictor;
    InstructionMemory imem(64);
    imem.write(0, i_type(7, 0, isa::FUNCT3_ADD_SUB, 5, isa::OP_ALU_I));

    std::vector<WarpState> warps;
    warps.emplace_back(3);  // depth 3
    warps[0].reset(0);

    // 2 entries in buffer + 1 pending from decode = effectively full
    warps[0].instr_buffer.push(BufferEntry{});
    warps[0].instr_buffer.push(BufferEntry{});
    REQUIRE(warps[0].instr_buffer.size() == 2);

    FetchStage fetch(1, warps.data(), imem, predictor, stats);
    fetch.set_decode_pending_warp(0);

    fetch.evaluate();
    fetch.commit();
    REQUIRE_FALSE(fetch.current_output().has_value());
}

TEST_CASE("Fetch: decode-pending warp with 1 of 3 slots is still eligible", "[ibuffer][timing]") {
    // 1 entry + 1 pending = 2, which is less than capacity 3 — should be eligible
    Stats stats;
    StaticDirectionalBranchPredictor predictor;
    InstructionMemory imem(64);
    imem.write(0, i_type(7, 0, isa::FUNCT3_ADD_SUB, 5, isa::OP_ALU_I));

    std::vector<WarpState> warps;
    warps.emplace_back(3);  // depth 3
    warps[0].reset(0);

    warps[0].instr_buffer.push(BufferEntry{});
    REQUIRE(warps[0].instr_buffer.size() == 1);

    FetchStage fetch(1, warps.data(), imem, predictor, stats);
    fetch.set_decode_pending_warp(0);

    fetch.evaluate();
    fetch.commit();
    REQUIRE(fetch.current_output().has_value());
}

TEST_CASE("InstructionBuffer: repeated fill-drain cycles at depth 3", "[ibuffer]") {
    InstructionBuffer buf(3);

    for (int cycle = 0; cycle < 5; ++cycle) {
        // Fill to capacity
        for (uint32_t i = 0; i < 3; ++i) {
            BufferEntry e{};
            e.pc = static_cast<uint32_t>(cycle * 12 + i * 4);
            buf.push(e);
        }
        REQUIRE(buf.is_full());
        REQUIRE(buf.size() == 3);

        // Drain completely
        for (uint32_t i = 0; i < 3; ++i) {
            REQUIRE_FALSE(buf.is_empty());
            buf.pop();
        }
        REQUIRE(buf.is_empty());
    }
}

TEST_CASE("WarpState: explicit depth override still works", "[ibuffer]") {
    // Verify that constructing with a non-default depth is honored
    WarpState ws2(2);
    REQUIRE(ws2.instr_buffer.capacity() == 2);

    WarpState ws5(5);
    REQUIRE(ws5.instr_buffer.capacity() == 5);

    // Depth 1 edge case
    WarpState ws1(1);
    REQUIRE(ws1.instr_buffer.capacity() == 1);
    ws1.instr_buffer.push(BufferEntry{});
    REQUIRE(ws1.instr_buffer.is_full());
}
