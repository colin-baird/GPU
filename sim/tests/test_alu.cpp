#include "catch.hpp"
#include "gpu_sim/functional/alu.h"
#include <climits>

using namespace gpu_sim;

TEST_CASE("ALU ADD", "[alu]") {
    REQUIRE(execute_alu(AluOp::ADD, 5, 3) == 8);
    REQUIRE(execute_alu(AluOp::ADD, 0xFFFFFFFF, 1) == 0);  // Overflow wraps
    REQUIRE(execute_alu(AluOp::ADD, 0, 0) == 0);
}

TEST_CASE("ALU SUB", "[alu]") {
    REQUIRE(execute_alu(AluOp::SUB, 10, 3) == 7);
    REQUIRE(execute_alu(AluOp::SUB, 0, 1) == 0xFFFFFFFF);
    REQUIRE(execute_alu(AluOp::SUB, 5, 5) == 0);
}

TEST_CASE("ALU XOR", "[alu]") {
    REQUIRE(execute_alu(AluOp::XOR, 0xFF00FF00, 0x0F0F0F0F) == 0xF00FF00F);
}

TEST_CASE("ALU OR", "[alu]") {
    REQUIRE(execute_alu(AluOp::OR, 0xFF000000, 0x00FF0000) == 0xFFFF0000);
}

TEST_CASE("ALU AND", "[alu]") {
    REQUIRE(execute_alu(AluOp::AND, 0xFF00FF00, 0x0F0F0F0F) == 0x0F000F00);
}

TEST_CASE("ALU SLL", "[alu]") {
    REQUIRE(execute_alu(AluOp::SLL, 1, 0) == 1);
    REQUIRE(execute_alu(AluOp::SLL, 1, 4) == 16);
    REQUIRE(execute_alu(AluOp::SLL, 1, 31) == 0x80000000);
    REQUIRE(execute_alu(AluOp::SLL, 1, 32) == 1);  // Only lower 5 bits used
}

TEST_CASE("ALU SRL", "[alu]") {
    REQUIRE(execute_alu(AluOp::SRL, 0x80000000, 31) == 1);
    REQUIRE(execute_alu(AluOp::SRL, 16, 4) == 1);
    REQUIRE(execute_alu(AluOp::SRL, 0xFFFFFFFF, 1) == 0x7FFFFFFF);
}

TEST_CASE("ALU SRA", "[alu]") {
    // Arithmetic right shift preserves sign
    REQUIRE(execute_alu(AluOp::SRA, 0x80000000, 1) == 0xC0000000);
    REQUIRE(execute_alu(AluOp::SRA, 0x80000000, 31) == 0xFFFFFFFF);
    REQUIRE(execute_alu(AluOp::SRA, 0x7FFFFFFF, 1) == 0x3FFFFFFF);
}

TEST_CASE("ALU SLT", "[alu]") {
    REQUIRE(execute_alu(AluOp::SLT, 0, 1) == 1);  // 0 < 1
    REQUIRE(execute_alu(AluOp::SLT, 1, 0) == 0);  // 1 not < 0
    REQUIRE(execute_alu(AluOp::SLT, 0xFFFFFFFF, 0) == 1);  // -1 < 0 (signed)
    REQUIRE(execute_alu(AluOp::SLT, 5, 5) == 0);  // Equal
}

TEST_CASE("ALU SLTU", "[alu]") {
    REQUIRE(execute_alu(AluOp::SLTU, 0, 1) == 1);
    REQUIRE(execute_alu(AluOp::SLTU, 0xFFFFFFFF, 0) == 0);  // Large unsigned not < 0
    REQUIRE(execute_alu(AluOp::SLTU, 0, 0xFFFFFFFF) == 1);
}

TEST_CASE("MUL", "[alu]") {
    REQUIRE(execute_mul(MulDivOp::MUL, 6, 7) == 42);
    // -1 * -1 = 1
    REQUIRE(execute_mul(MulDivOp::MUL, 0xFFFFFFFF, 0xFFFFFFFF) == 1);
    // -3 * 5 = -15
    REQUIRE(execute_mul(MulDivOp::MUL, static_cast<uint32_t>(-3), 5) == static_cast<uint32_t>(-15));
}

TEST_CASE("MULH signed", "[alu]") {
    // High 32 bits of 0x7FFFFFFF * 0x7FFFFFFF
    REQUIRE(execute_mul(MulDivOp::MULH, 0x7FFFFFFF, 0x7FFFFFFF) == 0x3FFFFFFF);
    // -1 * -1 -> upper bits = 0
    REQUIRE(execute_mul(MulDivOp::MULH, 0xFFFFFFFF, 0xFFFFFFFF) == 0);
}

TEST_CASE("MULHU unsigned", "[alu]") {
    REQUIRE(execute_mul(MulDivOp::MULHU, 0xFFFFFFFF, 0xFFFFFFFF) == 0xFFFFFFFE);
}

TEST_CASE("DIV signed", "[alu]") {
    REQUIRE(execute_div(MulDivOp::DIV, 20, 6) == 3);
    REQUIRE(execute_div(MulDivOp::DIV, static_cast<uint32_t>(-20), 6) == static_cast<uint32_t>(-3));
}

TEST_CASE("DIV by zero", "[alu]") {
    REQUIRE(execute_div(MulDivOp::DIV, 42, 0) == 0xFFFFFFFF);   // -1
    REQUIRE(execute_div(MulDivOp::DIVU, 42, 0) == 0xFFFFFFFF);
}

TEST_CASE("REM by zero", "[alu]") {
    REQUIRE(execute_div(MulDivOp::REM, 42, 0) == 42);   // dividend
    REQUIRE(execute_div(MulDivOp::REMU, 42, 0) == 42);
}

TEST_CASE("DIV overflow (INT32_MIN / -1)", "[alu]") {
    REQUIRE(execute_div(MulDivOp::DIV, 0x80000000, 0xFFFFFFFF) == 0x80000000);
}

TEST_CASE("REM overflow (INT32_MIN % -1)", "[alu]") {
    REQUIRE(execute_div(MulDivOp::REM, 0x80000000, 0xFFFFFFFF) == 0);
}

TEST_CASE("DIVU", "[alu]") {
    REQUIRE(execute_div(MulDivOp::DIVU, 20, 6) == 3);
    REQUIRE(execute_div(MulDivOp::DIVU, 0xFFFFFFFF, 2) == 0x7FFFFFFF);
}

TEST_CASE("REMU", "[alu]") {
    REQUIRE(execute_div(MulDivOp::REMU, 20, 6) == 2);
}

TEST_CASE("VDOT8 basic", "[alu]") {
    // rs1 = [1, 2, 3, 4] as packed INT8
    // rs2 = [5, 6, 7, 8] as packed INT8
    // result = 0 + 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    uint32_t rs1 = 0x04030201;
    uint32_t rs2 = 0x08070605;
    REQUIRE(execute_vdot8(rs1, rs2, 0) == 70);
}

TEST_CASE("VDOT8 with accumulate", "[alu]") {
    uint32_t rs1 = 0x04030201;
    uint32_t rs2 = 0x08070605;
    REQUIRE(execute_vdot8(rs1, rs2, 100) == 170);
}

TEST_CASE("VDOT8 signed values", "[alu]") {
    // rs1 = [-1, -2, 3, 4] => [0xFF, 0xFE, 0x03, 0x04]
    // rs2 = [5, 6, 7, 8]   => [0x05, 0x06, 0x07, 0x08]
    // result = (-1*5) + (-2*6) + (3*7) + (4*8) = -5 + -12 + 21 + 32 = 36
    uint32_t rs1 = 0x0403FEFF;
    uint32_t rs2 = 0x08070605;
    REQUIRE(execute_vdot8(rs1, rs2, 0) == 36);
}

TEST_CASE("VDOT8 max magnitude", "[alu]") {
    // -128 * -128 = 16384 per pair, 4 pairs = 65536
    uint32_t rs1 = 0x80808080;  // All -128
    uint32_t rs2 = 0x80808080;  // All -128
    REQUIRE(execute_vdot8(rs1, rs2, 0) == 65536);
}

TEST_CASE("Branch BEQ", "[alu]") {
    REQUIRE(evaluate_branch(BranchOp::BEQ, 5, 5) == true);
    REQUIRE(evaluate_branch(BranchOp::BEQ, 5, 6) == false);
}

TEST_CASE("Branch BNE", "[alu]") {
    REQUIRE(evaluate_branch(BranchOp::BNE, 5, 6) == true);
    REQUIRE(evaluate_branch(BranchOp::BNE, 5, 5) == false);
}

TEST_CASE("Branch BLT signed", "[alu]") {
    REQUIRE(evaluate_branch(BranchOp::BLT, 0xFFFFFFFF, 0) == true);  // -1 < 0
    REQUIRE(evaluate_branch(BranchOp::BLT, 0, 0xFFFFFFFF) == false);
}

TEST_CASE("Branch BGE signed", "[alu]") {
    REQUIRE(evaluate_branch(BranchOp::BGE, 5, 5) == true);
    REQUIRE(evaluate_branch(BranchOp::BGE, 6, 5) == true);
    REQUIRE(evaluate_branch(BranchOp::BGE, 4, 5) == false);
}

TEST_CASE("Branch BLTU unsigned", "[alu]") {
    REQUIRE(evaluate_branch(BranchOp::BLTU, 0, 0xFFFFFFFF) == true);
    REQUIRE(evaluate_branch(BranchOp::BLTU, 0xFFFFFFFF, 0) == false);
}

TEST_CASE("Branch BGEU unsigned", "[alu]") {
    REQUIRE(evaluate_branch(BranchOp::BGEU, 0xFFFFFFFF, 0) == true);
    REQUIRE(evaluate_branch(BranchOp::BGEU, 5, 5) == true);
}
