#include "gpu_sim/functional/alu.h"
#include <cstdint>

namespace gpu_sim {

uint32_t execute_alu(AluOp op, uint32_t a, uint32_t b) {
    int32_t sa = static_cast<int32_t>(a);
    int32_t sb = static_cast<int32_t>(b);

    switch (op) {
        case AluOp::ADD:  return a + b;
        case AluOp::SUB:  return a - b;
        case AluOp::XOR:  return a ^ b;
        case AluOp::OR:   return a | b;
        case AluOp::AND:  return a & b;
        case AluOp::SLL:  return a << (b & 0x1F);
        case AluOp::SRL:  return a >> (b & 0x1F);
        case AluOp::SRA:  return static_cast<uint32_t>(sa >> (b & 0x1F));
        case AluOp::SLT:  return (sa < sb) ? 1u : 0u;
        case AluOp::SLTU: return (a < b) ? 1u : 0u;
        default:          return 0;
    }
}

uint32_t execute_mul(MulDivOp op, uint32_t a, uint32_t b) {
    int32_t sa = static_cast<int32_t>(a);
    int32_t sb = static_cast<int32_t>(b);

    switch (op) {
        case MulDivOp::MUL: {
            // Lower 32 bits of signed multiply
            return static_cast<uint32_t>(sa * sb);
        }
        case MulDivOp::MULH: {
            // Upper 32 bits of signed x signed
            int64_t result = static_cast<int64_t>(sa) * static_cast<int64_t>(sb);
            return static_cast<uint32_t>(result >> 32);
        }
        case MulDivOp::MULHSU: {
            // Upper 32 bits of signed x unsigned
            int64_t result = static_cast<int64_t>(sa) * static_cast<uint64_t>(b);
            return static_cast<uint32_t>(result >> 32);
        }
        case MulDivOp::MULHU: {
            // Upper 32 bits of unsigned x unsigned
            uint64_t result = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
            return static_cast<uint32_t>(result >> 32);
        }
        default:
            return 0;
    }
}

uint32_t execute_div(MulDivOp op, uint32_t a, uint32_t b) {
    int32_t sa = static_cast<int32_t>(a);
    int32_t sb = static_cast<int32_t>(b);

    switch (op) {
        case MulDivOp::DIV: {
            if (b == 0) return 0xFFFFFFFF;  // -1
            // Overflow: INT32_MIN / -1
            if (sa == INT32_MIN && sb == -1) return static_cast<uint32_t>(INT32_MIN);
            return static_cast<uint32_t>(sa / sb);
        }
        case MulDivOp::DIVU: {
            if (b == 0) return 0xFFFFFFFF;
            return a / b;
        }
        case MulDivOp::REM: {
            if (b == 0) return a;  // dividend
            if (sa == INT32_MIN && sb == -1) return 0;
            return static_cast<uint32_t>(sa % sb);
        }
        case MulDivOp::REMU: {
            if (b == 0) return a;
            return a % b;
        }
        default:
            return 0;
    }
}

uint32_t execute_vdot8(uint32_t rs1, uint32_t rs2, uint32_t rd_accum) {
    int32_t accum = static_cast<int32_t>(rd_accum);

    for (int i = 0; i < 4; ++i) {
        int8_t a = static_cast<int8_t>((rs1 >> (i * 8)) & 0xFF);
        int8_t b = static_cast<int8_t>((rs2 >> (i * 8)) & 0xFF);
        accum += static_cast<int32_t>(a) * static_cast<int32_t>(b);
    }

    return static_cast<uint32_t>(accum);
}

bool evaluate_branch(BranchOp op, uint32_t a, uint32_t b) {
    int32_t sa = static_cast<int32_t>(a);
    int32_t sb = static_cast<int32_t>(b);

    switch (op) {
        case BranchOp::BEQ:  return a == b;
        case BranchOp::BNE:  return a != b;
        case BranchOp::BLT:  return sa < sb;
        case BranchOp::BGE:  return sa >= sb;
        case BranchOp::BLTU: return a < b;
        case BranchOp::BGEU: return a >= b;
        default:             return false;
    }
}

} // namespace gpu_sim
