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

TEST_CASE("Fetch and Decode: pending decode stalls fetch until buffer space frees", "[timing]") {
    Stats stats;
    StaticDirectionalBranchPredictor predictor;
    InstructionMemory imem(64);
    imem.write(0, i_type(7, 0, isa::FUNCT3_ADD_SUB, 5, isa::OP_ALU_I));
    imem.write(1, i_type(9, 0, isa::FUNCT3_ADD_SUB, 6, isa::OP_ALU_I));

    std::vector<WarpState> warps;
    warps.emplace_back(1);
    warps[0].reset(0);

    FetchStage fetch(1, warps.data(), imem, predictor, stats);
    DecodeStage decode(warps.data(), fetch);

    fetch.evaluate();
    fetch.commit();
    decode.evaluate();
    REQUIRE(decode.has_pending());

    warps[0].instr_buffer.push(BufferEntry{});
    decode.commit();
    REQUIRE(decode.has_pending());

    fetch.set_stall(decode.has_pending());
    fetch.evaluate();
    fetch.commit();
    REQUIRE_FALSE(fetch.current_output().has_value());

    warps[0].instr_buffer.pop();
    decode.commit();
    REQUIRE_FALSE(decode.has_pending());
    REQUIRE(warps[0].instr_buffer.size() == 1);
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

TEST_CASE("TLookupUnit: result appears after 64 cycles", "[timing]") {
    Stats stats;
    TLookupUnit tlookup(stats);
    auto issue = make_issue_output(i_type(16, 1, 0, 5, isa::OP_TLOOKUP));

    tlookup.accept(DispatchInput{issue.decoded, issue.trace, issue.warp_id, issue.pc}, 4);
    for (int i = 0; i < 63; ++i) {
        REQUIRE_FALSE(tlookup.has_result());
        tlookup.evaluate();
    }
    REQUIRE_FALSE(tlookup.has_result());
    tlookup.evaluate();
    REQUIRE(tlookup.has_result());
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
