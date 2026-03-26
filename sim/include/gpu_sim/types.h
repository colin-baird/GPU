#pragma once

#include <array>
#include <cstdint>

namespace gpu_sim {

using WarpId = uint32_t;
using LaneId = uint32_t;
using RegIndex = uint8_t;
using Address = uint32_t;
using Word = uint32_t;

static constexpr uint32_t WARP_SIZE = 32;
static constexpr uint32_t MAX_WARPS = 8;
static constexpr uint32_t NUM_REGS = 32;

using WarpData = std::array<Word, WARP_SIZE>;

enum class ExecUnit : uint8_t {
    ALU,
    MULTIPLY,
    DIVIDE,
    LDST,
    TLOOKUP,
    SYSTEM,
    NONE
};

enum class InstructionType : uint8_t {
    ALU_R,
    ALU_I,
    LUI,
    AUIPC,
    LOAD,
    STORE,
    BRANCH,
    JAL,
    JALR,
    MUL,
    DIV,
    VDOT8,
    TLOOKUP,
    ECALL,
    EBREAK,
    CSR,
    INVALID
};

enum class AluOp : uint8_t {
    ADD, SUB, XOR, OR, AND, SLL, SRL, SRA, SLT, SLTU, NONE
};

enum class MulDivOp : uint8_t {
    MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU, NONE
};

enum class MemOp : uint8_t {
    LB, LH, LW, LBU, LHU, SB, SH, SW, NONE
};

enum class BranchOp : uint8_t {
    BEQ, BNE, BLT, BGE, BLTU, BGEU, NONE
};

} // namespace gpu_sim
