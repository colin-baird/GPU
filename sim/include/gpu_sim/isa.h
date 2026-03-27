#pragma once

#include <cstdint>

namespace gpu_sim {
namespace isa {

// RV32I base opcodes
static constexpr uint32_t OP_LUI      = 0b0110111;
static constexpr uint32_t OP_AUIPC    = 0b0010111;
static constexpr uint32_t OP_JAL      = 0b1101111;
static constexpr uint32_t OP_JALR     = 0b1100111;
static constexpr uint32_t OP_BRANCH   = 0b1100011;
static constexpr uint32_t OP_LOAD     = 0b0000011;
static constexpr uint32_t OP_STORE    = 0b0100011;
static constexpr uint32_t OP_ALU_I    = 0b0010011;
static constexpr uint32_t OP_ALU_R    = 0b0110011;
static constexpr uint32_t OP_FENCE    = 0b0001111;
static constexpr uint32_t OP_SYSTEM   = 0b1110011;

// Custom opcodes
static constexpr uint32_t OP_VDOT8    = 0b0001011;  // custom-0
static constexpr uint32_t OP_TLOOKUP  = 0b0101011;  // custom-1

// funct7 values
static constexpr uint32_t FUNCT7_BASE  = 0x00;
static constexpr uint32_t FUNCT7_ALT   = 0x20;  // SUB, SRA
static constexpr uint32_t FUNCT7_MULDIV = 0x01; // M-extension

// funct3 values for ALU
static constexpr uint32_t FUNCT3_ADD_SUB = 0x0;
static constexpr uint32_t FUNCT3_SLL     = 0x1;
static constexpr uint32_t FUNCT3_SLT     = 0x2;
static constexpr uint32_t FUNCT3_SLTU    = 0x3;
static constexpr uint32_t FUNCT3_XOR     = 0x4;
static constexpr uint32_t FUNCT3_SRL_SRA = 0x5;
static constexpr uint32_t FUNCT3_OR      = 0x6;
static constexpr uint32_t FUNCT3_AND     = 0x7;

// funct3 values for M-extension
static constexpr uint32_t FUNCT3_MUL    = 0x0;
static constexpr uint32_t FUNCT3_MULH   = 0x1;
static constexpr uint32_t FUNCT3_MULHSU = 0x2;
static constexpr uint32_t FUNCT3_MULHU  = 0x3;
static constexpr uint32_t FUNCT3_DIV    = 0x4;
static constexpr uint32_t FUNCT3_DIVU   = 0x5;
static constexpr uint32_t FUNCT3_REM    = 0x6;
static constexpr uint32_t FUNCT3_REMU   = 0x7;

// funct3 values for loads
static constexpr uint32_t FUNCT3_LB  = 0x0;
static constexpr uint32_t FUNCT3_LH  = 0x1;
static constexpr uint32_t FUNCT3_LW  = 0x2;
static constexpr uint32_t FUNCT3_LBU = 0x4;
static constexpr uint32_t FUNCT3_LHU = 0x5;

// funct3 values for stores
static constexpr uint32_t FUNCT3_SB = 0x0;
static constexpr uint32_t FUNCT3_SH = 0x1;
static constexpr uint32_t FUNCT3_SW = 0x2;

// funct3 values for branches
static constexpr uint32_t FUNCT3_BEQ  = 0x0;
static constexpr uint32_t FUNCT3_BNE  = 0x1;
static constexpr uint32_t FUNCT3_BLT  = 0x4;
static constexpr uint32_t FUNCT3_BGE  = 0x5;
static constexpr uint32_t FUNCT3_BLTU = 0x6;
static constexpr uint32_t FUNCT3_BGEU = 0x7;

// System funct3
static constexpr uint32_t FUNCT3_ECALL_EBREAK = 0x0;
static constexpr uint32_t FUNCT3_CSRRW  = 0x1;
static constexpr uint32_t FUNCT3_CSRRS  = 0x2;
static constexpr uint32_t FUNCT3_CSRRC  = 0x3;

// CSR addresses for GPU thread identity
static constexpr uint16_t CSR_WARP_ID   = 0xC00;
static constexpr uint16_t CSR_LANE_ID   = 0xC01;
static constexpr uint16_t CSR_NUM_WARPS = 0xC02;

// Instruction field extraction helpers
inline uint32_t opcode(uint32_t instr)  { return instr & 0x7F; }
inline uint32_t rd(uint32_t instr)      { return (instr >> 7) & 0x1F; }
inline uint32_t funct3(uint32_t instr)  { return (instr >> 12) & 0x7; }
inline uint32_t rs1(uint32_t instr)     { return (instr >> 15) & 0x1F; }
inline uint32_t rs2(uint32_t instr)     { return (instr >> 20) & 0x1F; }
inline uint32_t funct7(uint32_t instr)  { return (instr >> 25) & 0x7F; }

// Immediate extraction (sign-extended)
inline int32_t imm_i(uint32_t instr) {
    return static_cast<int32_t>(instr) >> 20;
}

inline int32_t imm_s(uint32_t instr) {
    int32_t imm = ((instr >> 25) << 5) | ((instr >> 7) & 0x1F);
    // Sign extend from bit 11
    if (imm & 0x800) imm |= 0xFFFFF000;
    return imm;
}

inline int32_t imm_b(uint32_t instr) {
    int32_t imm = 0;
    imm |= ((instr >> 31) & 1) << 12;
    imm |= ((instr >> 7) & 1) << 11;
    imm |= ((instr >> 25) & 0x3F) << 5;
    imm |= ((instr >> 8) & 0xF) << 1;
    // Sign extend from bit 12
    if (imm & 0x1000) imm |= 0xFFFFE000;
    return imm;
}

inline int32_t imm_u(uint32_t instr) {
    return static_cast<int32_t>(instr & 0xFFFFF000);
}

inline int32_t imm_j(uint32_t instr) {
    int32_t imm = 0;
    imm |= ((instr >> 31) & 1) << 20;
    imm |= ((instr >> 12) & 0xFF) << 12;
    imm |= ((instr >> 20) & 1) << 11;
    imm |= ((instr >> 21) & 0x3FF) << 1;
    // Sign extend from bit 20
    if (imm & 0x100000) imm |= 0xFFE00000;
    return imm;
}

} // namespace isa
} // namespace gpu_sim
