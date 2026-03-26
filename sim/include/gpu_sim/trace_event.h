#pragma once

#include "gpu_sim/types.h"
#include <array>
#include <cstdint>

namespace gpu_sim {

struct DecodedInstruction {
    InstructionType type = InstructionType::INVALID;
    AluOp alu_op = AluOp::NONE;
    MulDivOp muldiv_op = MulDivOp::NONE;
    MemOp mem_op = MemOp::NONE;
    BranchOp branch_op = BranchOp::NONE;
    ExecUnit target_unit = ExecUnit::NONE;
    uint8_t rd = 0;
    uint8_t rs1 = 0;
    uint8_t rs2 = 0;
    int32_t imm = 0;
    bool has_rd = false;
    bool reads_rd = false;       // True only for VDOT8
    uint8_t num_src_regs = 0;   // For scoreboard: how many regs to check
    uint16_t csr_addr = 0;
    uint32_t raw = 0;
};

struct TraceEvent {
    uint32_t warp_id = 0;
    uint32_t pc = 0;
    DecodedInstruction decoded;

    // Per-thread results (32 lanes)
    std::array<uint32_t, WARP_SIZE> results{};
    std::array<uint32_t, WARP_SIZE> mem_addresses{};
    std::array<uint32_t, WARP_SIZE> store_data{};
    std::array<uint8_t, WARP_SIZE> mem_size{};
    bool is_load = false;
    bool is_store = false;

    // Branch info
    bool is_branch = false;
    bool branch_taken = false;
    uint32_t branch_target = 0;

    // Control flow
    bool is_ecall = false;
    bool is_ebreak = false;
    uint32_t panic_cause = 0;

    // TLOOKUP
    bool is_tlookup = false;
    std::array<uint32_t, WARP_SIZE> tlookup_indices{};

    // CSR
    bool is_csr = false;
    uint16_t csr_addr = 0;

    uint64_t sequence_number = 0;
};

} // namespace gpu_sim
