#include "catch.hpp"
#include "gpu_sim/decoder.h"
#include "gpu_sim/isa.h"
#include "gpu_sim/types.h"

using namespace gpu_sim;

// Helper to build R-type instruction
static uint32_t r_type(uint32_t funct7, uint32_t rs2, uint32_t rs1,
                        uint32_t funct3, uint32_t rd, uint32_t opcode) {
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode;
}

// Helper to build I-type instruction
static uint32_t i_type(int32_t imm, uint32_t rs1, uint32_t funct3,
                        uint32_t rd, uint32_t opcode) {
    return (static_cast<uint32_t>(imm & 0xFFF) << 20) | (rs1 << 15) |
           (funct3 << 12) | (rd << 7) | opcode;
}

// Helper to build S-type instruction
static uint32_t s_type(int32_t imm, uint32_t rs2, uint32_t rs1,
                        uint32_t funct3, uint32_t opcode) {
    uint32_t imm_hi = (imm >> 5) & 0x7F;
    uint32_t imm_lo = imm & 0x1F;
    return (imm_hi << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_lo << 7) | opcode;
}

// Helper to build B-type instruction
static uint32_t b_type(int32_t imm, uint32_t rs2, uint32_t rs1,
                        uint32_t funct3, uint32_t opcode) {
    uint32_t bit12 = (imm >> 12) & 1;
    uint32_t bit11 = (imm >> 11) & 1;
    uint32_t bits10_5 = (imm >> 5) & 0x3F;
    uint32_t bits4_1 = (imm >> 1) & 0xF;
    return (bit12 << 31) | (bits10_5 << 25) | (rs2 << 20) | (rs1 << 15) |
           (funct3 << 12) | (bits4_1 << 8) | (bit11 << 7) | opcode;
}

TEST_CASE("Decode ADD", "[decoder]") {
    // ADD x5, x1, x2: funct7=0x00, rs2=2, rs1=1, funct3=0, rd=5, opcode=0110011
    uint32_t instr = r_type(0x00, 2, 1, 0, 5, isa::OP_ALU_R);
    auto d = Decoder::decode(instr);
    REQUIRE(d.type == InstructionType::ALU_R);
    REQUIRE(d.alu_op == AluOp::ADD);
    REQUIRE(d.rd == 5);
    REQUIRE(d.rs1 == 1);
    REQUIRE(d.rs2 == 2);
    REQUIRE(d.target_unit == ExecUnit::ALU);
    REQUIRE(d.has_rd == true);
    REQUIRE(d.num_src_regs == 2);
}

TEST_CASE("Decode SUB", "[decoder]") {
    uint32_t instr = r_type(0x20, 3, 2, 0, 4, isa::OP_ALU_R);
    auto d = Decoder::decode(instr);
    REQUIRE(d.type == InstructionType::ALU_R);
    REQUIRE(d.alu_op == AluOp::SUB);
    REQUIRE(d.rd == 4);
    REQUIRE(d.rs1 == 2);
    REQUIRE(d.rs2 == 3);
}

TEST_CASE("Decode SLT", "[decoder]") {
    uint32_t instr = r_type(0x00, 2, 1, 2, 5, isa::OP_ALU_R);
    auto d = Decoder::decode(instr);
    REQUIRE(d.alu_op == AluOp::SLT);
}

TEST_CASE("Decode SLTU", "[decoder]") {
    uint32_t instr = r_type(0x00, 2, 1, 3, 5, isa::OP_ALU_R);
    auto d = Decoder::decode(instr);
    REQUIRE(d.alu_op == AluOp::SLTU);
}

TEST_CASE("Decode XOR, OR, AND", "[decoder]") {
    REQUIRE(Decoder::decode(r_type(0, 2, 1, 4, 5, isa::OP_ALU_R)).alu_op == AluOp::XOR);
    REQUIRE(Decoder::decode(r_type(0, 2, 1, 6, 5, isa::OP_ALU_R)).alu_op == AluOp::OR);
    REQUIRE(Decoder::decode(r_type(0, 2, 1, 7, 5, isa::OP_ALU_R)).alu_op == AluOp::AND);
}

TEST_CASE("Decode shifts R-type", "[decoder]") {
    REQUIRE(Decoder::decode(r_type(0x00, 2, 1, 1, 5, isa::OP_ALU_R)).alu_op == AluOp::SLL);
    REQUIRE(Decoder::decode(r_type(0x00, 2, 1, 5, 5, isa::OP_ALU_R)).alu_op == AluOp::SRL);
    REQUIRE(Decoder::decode(r_type(0x20, 2, 1, 5, 5, isa::OP_ALU_R)).alu_op == AluOp::SRA);
}

TEST_CASE("Decode ADDI", "[decoder]") {
    uint32_t instr = i_type(42, 1, 0, 5, isa::OP_ALU_I);
    auto d = Decoder::decode(instr);
    REQUIRE(d.type == InstructionType::ALU_I);
    REQUIRE(d.alu_op == AluOp::ADD);
    REQUIRE(d.imm == 42);
    REQUIRE(d.rs1 == 1);
    REQUIRE(d.rd == 5);
    REQUIRE(d.num_src_regs == 1);
}

TEST_CASE("Decode ADDI negative immediate", "[decoder]") {
    uint32_t instr = i_type(-10, 1, 0, 5, isa::OP_ALU_I);
    auto d = Decoder::decode(instr);
    REQUIRE(d.imm == -10);
}

TEST_CASE("Decode LUI", "[decoder]") {
    // LUI x5, 0x12345: upper 20 bits = 0x12345
    uint32_t instr = (0x12345 << 12) | (5 << 7) | isa::OP_LUI;
    auto d = Decoder::decode(instr);
    REQUIRE(d.type == InstructionType::LUI);
    REQUIRE(d.rd == 5);
    REQUIRE(d.imm == static_cast<int32_t>(0x12345000));
    REQUIRE(d.num_src_regs == 0);
}

TEST_CASE("Decode AUIPC", "[decoder]") {
    uint32_t instr = (0x12345 << 12) | (5 << 7) | isa::OP_AUIPC;
    auto d = Decoder::decode(instr);
    REQUIRE(d.type == InstructionType::AUIPC);
    REQUIRE(d.imm == static_cast<int32_t>(0x12345000));
}

TEST_CASE("Decode JAL", "[decoder]") {
    // JAL x1, +8: imm[20|10:1|11|19:12]
    // +8 = 0b1000, bits: imm[20]=0, imm[10:1]=0000000100, imm[11]=0, imm[19:12]=00000000
    uint32_t imm20 = 0;
    uint32_t imm10_1 = 4;
    uint32_t imm11 = 0;
    uint32_t imm19_12 = 0;
    uint32_t instr = (imm20 << 31) | (imm10_1 << 21) | (imm11 << 20) |
                     (imm19_12 << 12) | (1 << 7) | isa::OP_JAL;
    auto d = Decoder::decode(instr);
    REQUIRE(d.type == InstructionType::JAL);
    REQUIRE(d.rd == 1);
    REQUIRE(d.imm == 8);
    REQUIRE(d.target_unit == ExecUnit::ALU);
}

TEST_CASE("Decode JALR", "[decoder]") {
    uint32_t instr = i_type(100, 3, 0, 1, isa::OP_JALR);
    auto d = Decoder::decode(instr);
    REQUIRE(d.type == InstructionType::JALR);
    REQUIRE(d.rd == 1);
    REQUIRE(d.rs1 == 3);
    REQUIRE(d.imm == 100);
    REQUIRE(d.num_src_regs == 1);
}

TEST_CASE("Decode BEQ", "[decoder]") {
    uint32_t instr = b_type(8, 2, 1, 0, isa::OP_BRANCH);
    auto d = Decoder::decode(instr);
    REQUIRE(d.type == InstructionType::BRANCH);
    REQUIRE(d.branch_op == BranchOp::BEQ);
    REQUIRE(d.rs1 == 1);
    REQUIRE(d.rs2 == 2);
    REQUIRE(d.has_rd == false);
    REQUIRE(d.num_src_regs == 2);
}

TEST_CASE("Decode all branch types", "[decoder]") {
    REQUIRE(Decoder::decode(b_type(8, 2, 1, 0, isa::OP_BRANCH)).branch_op == BranchOp::BEQ);
    REQUIRE(Decoder::decode(b_type(8, 2, 1, 1, isa::OP_BRANCH)).branch_op == BranchOp::BNE);
    REQUIRE(Decoder::decode(b_type(8, 2, 1, 4, isa::OP_BRANCH)).branch_op == BranchOp::BLT);
    REQUIRE(Decoder::decode(b_type(8, 2, 1, 5, isa::OP_BRANCH)).branch_op == BranchOp::BGE);
    REQUIRE(Decoder::decode(b_type(8, 2, 1, 6, isa::OP_BRANCH)).branch_op == BranchOp::BLTU);
    REQUIRE(Decoder::decode(b_type(8, 2, 1, 7, isa::OP_BRANCH)).branch_op == BranchOp::BGEU);
}

TEST_CASE("Decode loads", "[decoder]") {
    REQUIRE(Decoder::decode(i_type(0, 1, 0, 5, isa::OP_LOAD)).mem_op == MemOp::LB);
    REQUIRE(Decoder::decode(i_type(0, 1, 1, 5, isa::OP_LOAD)).mem_op == MemOp::LH);
    REQUIRE(Decoder::decode(i_type(0, 1, 2, 5, isa::OP_LOAD)).mem_op == MemOp::LW);
    REQUIRE(Decoder::decode(i_type(0, 1, 4, 5, isa::OP_LOAD)).mem_op == MemOp::LBU);
    REQUIRE(Decoder::decode(i_type(0, 1, 5, 5, isa::OP_LOAD)).mem_op == MemOp::LHU);

    auto d = Decoder::decode(i_type(16, 3, 2, 7, isa::OP_LOAD));
    REQUIRE(d.type == InstructionType::LOAD);
    REQUIRE(d.target_unit == ExecUnit::LDST);
    REQUIRE(d.has_rd == true);
    REQUIRE(d.num_src_regs == 1);
    REQUIRE(d.imm == 16);
}

TEST_CASE("Decode stores", "[decoder]") {
    REQUIRE(Decoder::decode(s_type(0, 2, 1, 0, isa::OP_STORE)).mem_op == MemOp::SB);
    REQUIRE(Decoder::decode(s_type(0, 2, 1, 1, isa::OP_STORE)).mem_op == MemOp::SH);
    REQUIRE(Decoder::decode(s_type(0, 2, 1, 2, isa::OP_STORE)).mem_op == MemOp::SW);

    auto d = Decoder::decode(s_type(8, 5, 3, 2, isa::OP_STORE));
    REQUIRE(d.type == InstructionType::STORE);
    REQUIRE(d.has_rd == false);
    REQUIRE(d.num_src_regs == 2);
}

TEST_CASE("Decode M-extension multiply", "[decoder]") {
    auto d = Decoder::decode(r_type(0x01, 2, 1, 0, 5, isa::OP_ALU_R));
    REQUIRE(d.type == InstructionType::MUL);
    REQUIRE(d.muldiv_op == MulDivOp::MUL);
    REQUIRE(d.target_unit == ExecUnit::MULTIPLY);

    REQUIRE(Decoder::decode(r_type(0x01, 2, 1, 1, 5, isa::OP_ALU_R)).muldiv_op == MulDivOp::MULH);
    REQUIRE(Decoder::decode(r_type(0x01, 2, 1, 2, 5, isa::OP_ALU_R)).muldiv_op == MulDivOp::MULHSU);
    REQUIRE(Decoder::decode(r_type(0x01, 2, 1, 3, 5, isa::OP_ALU_R)).muldiv_op == MulDivOp::MULHU);
}

TEST_CASE("Decode M-extension divide", "[decoder]") {
    auto d = Decoder::decode(r_type(0x01, 2, 1, 4, 5, isa::OP_ALU_R));
    REQUIRE(d.type == InstructionType::DIV);
    REQUIRE(d.muldiv_op == MulDivOp::DIV);
    REQUIRE(d.target_unit == ExecUnit::DIVIDE);

    REQUIRE(Decoder::decode(r_type(0x01, 2, 1, 5, 5, isa::OP_ALU_R)).muldiv_op == MulDivOp::DIVU);
    REQUIRE(Decoder::decode(r_type(0x01, 2, 1, 6, 5, isa::OP_ALU_R)).muldiv_op == MulDivOp::REM);
    REQUIRE(Decoder::decode(r_type(0x01, 2, 1, 7, 5, isa::OP_ALU_R)).muldiv_op == MulDivOp::REMU);
}

TEST_CASE("Decode ECALL", "[decoder]") {
    uint32_t instr = i_type(0, 0, 0, 0, isa::OP_SYSTEM);
    auto d = Decoder::decode(instr);
    REQUIRE(d.type == InstructionType::ECALL);
    REQUIRE(d.target_unit == ExecUnit::SYSTEM);
    REQUIRE(d.has_rd == false);
}

TEST_CASE("Decode EBREAK", "[decoder]") {
    uint32_t instr = i_type(1, 0, 0, 0, isa::OP_SYSTEM);
    auto d = Decoder::decode(instr);
    REQUIRE(d.type == InstructionType::EBREAK);
    REQUIRE(d.target_unit == ExecUnit::SYSTEM);
}

TEST_CASE("Decode CSRRS (warp_id)", "[decoder]") {
    // CSRRS x5, 0xC00, x0: funct3=2, imm=0xC00, rs1=0, rd=5
    uint32_t instr = i_type(0xC00, 0, 2, 5, isa::OP_SYSTEM);
    auto d = Decoder::decode(instr);
    REQUIRE(d.type == InstructionType::CSR);
    REQUIRE(d.csr_addr == isa::CSR_WARP_ID);
    REQUIRE(d.rd == 5);
    REQUIRE(d.has_rd == true);
}

TEST_CASE("Decode VDOT8", "[decoder]") {
    // VDOT8 x5, x1, x2: R-type, opcode=0001011, funct7=0, funct3=0
    uint32_t instr = r_type(0x00, 2, 1, 0, 5, isa::OP_VDOT8);
    auto d = Decoder::decode(instr);
    REQUIRE(d.type == InstructionType::VDOT8);
    REQUIRE(d.target_unit == ExecUnit::MULTIPLY);
    REQUIRE(d.has_rd == true);
    REQUIRE(d.reads_rd == true);
    REQUIRE(d.num_src_regs == 3);
    REQUIRE(d.rd == 5);
    REQUIRE(d.rs1 == 1);
    REQUIRE(d.rs2 == 2);
}

TEST_CASE("Decode TLOOKUP", "[decoder]") {
    // TLOOKUP x5, x1, 256: I-type, opcode=0101011
    uint32_t instr = i_type(256, 1, 0, 5, isa::OP_TLOOKUP);
    auto d = Decoder::decode(instr);
    REQUIRE(d.type == InstructionType::TLOOKUP);
    REQUIRE(d.target_unit == ExecUnit::TLOOKUP);
    REQUIRE(d.has_rd == true);
    REQUIRE(d.reads_rd == false);
    REQUIRE(d.num_src_regs == 1);
    REQUIRE(d.imm == 256);
    REQUIRE(d.rd == 5);
    REQUIRE(d.rs1 == 1);
}

TEST_CASE("Decode invalid VDOT8 (wrong funct7)", "[decoder]") {
    uint32_t instr = r_type(0x01, 2, 1, 0, 5, isa::OP_VDOT8);
    auto d = Decoder::decode(instr);
    REQUIRE(d.type == InstructionType::INVALID);
}
