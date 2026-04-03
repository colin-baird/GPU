#include "gpu_sim/decoder.h"
#include "gpu_sim/isa.h"

namespace gpu_sim {

DecodedInstruction Decoder::decode(uint32_t instruction) {
    DecodedInstruction d;
    d.raw = instruction;

    uint32_t op = isa::opcode(instruction);
    d.rd  = static_cast<uint8_t>(isa::rd(instruction));
    d.rs1 = static_cast<uint8_t>(isa::rs1(instruction));
    d.rs2 = static_cast<uint8_t>(isa::rs2(instruction));

    uint32_t f3 = isa::funct3(instruction);
    uint32_t f7 = isa::funct7(instruction);

    switch (op) {

    // ---- R-type ALU / M-extension ----
    case isa::OP_ALU_R: {
        if (f7 == isa::FUNCT7_MULDIV) {
            // M-extension
            if (f3 <= isa::FUNCT3_MULHU) {
                d.type = InstructionType::MUL;
                d.target_unit = ExecUnit::MULTIPLY;
            } else {
                d.type = InstructionType::DIV;
                d.target_unit = ExecUnit::DIVIDE;
            }
            d.has_rd = true;
            d.num_src_regs = 2;
            switch (f3) {
                case isa::FUNCT3_MUL:    d.muldiv_op = MulDivOp::MUL;    break;
                case isa::FUNCT3_MULH:   d.muldiv_op = MulDivOp::MULH;   break;
                case isa::FUNCT3_MULHSU: d.muldiv_op = MulDivOp::MULHSU; break;
                case isa::FUNCT3_MULHU:  d.muldiv_op = MulDivOp::MULHU;  break;
                case isa::FUNCT3_DIV:    d.muldiv_op = MulDivOp::DIV;    break;
                case isa::FUNCT3_DIVU:   d.muldiv_op = MulDivOp::DIVU;   break;
                case isa::FUNCT3_REM:    d.muldiv_op = MulDivOp::REM;    break;
                case isa::FUNCT3_REMU:   d.muldiv_op = MulDivOp::REMU;   break;
            }
        } else {
            // Base ALU R-type — validate funct7
            d.type = InstructionType::ALU_R;
            d.target_unit = ExecUnit::ALU;
            d.has_rd = true;
            d.num_src_regs = 2;
            switch (f3) {
                case isa::FUNCT3_ADD_SUB:
                    if (f7 == isa::FUNCT7_BASE) d.alu_op = AluOp::ADD;
                    else if (f7 == isa::FUNCT7_ALT) d.alu_op = AluOp::SUB;
                    else d.type = InstructionType::INVALID;
                    break;
                case isa::FUNCT3_SLL:
                    if (f7 != isa::FUNCT7_BASE) { d.type = InstructionType::INVALID; break; }
                    d.alu_op = AluOp::SLL;
                    break;
                case isa::FUNCT3_SLT:
                    if (f7 != isa::FUNCT7_BASE) { d.type = InstructionType::INVALID; break; }
                    d.alu_op = AluOp::SLT;
                    break;
                case isa::FUNCT3_SLTU:
                    if (f7 != isa::FUNCT7_BASE) { d.type = InstructionType::INVALID; break; }
                    d.alu_op = AluOp::SLTU;
                    break;
                case isa::FUNCT3_XOR:
                    if (f7 != isa::FUNCT7_BASE) { d.type = InstructionType::INVALID; break; }
                    d.alu_op = AluOp::XOR;
                    break;
                case isa::FUNCT3_SRL_SRA:
                    if (f7 == isa::FUNCT7_BASE) d.alu_op = AluOp::SRL;
                    else if (f7 == isa::FUNCT7_ALT) d.alu_op = AluOp::SRA;
                    else d.type = InstructionType::INVALID;
                    break;
                case isa::FUNCT3_OR:
                    if (f7 != isa::FUNCT7_BASE) { d.type = InstructionType::INVALID; break; }
                    d.alu_op = AluOp::OR;
                    break;
                case isa::FUNCT3_AND:
                    if (f7 != isa::FUNCT7_BASE) { d.type = InstructionType::INVALID; break; }
                    d.alu_op = AluOp::AND;
                    break;
            }
        }
        break;
    }

    // ---- I-type ALU ----
    case isa::OP_ALU_I: {
        d.type = InstructionType::ALU_I;
        d.target_unit = ExecUnit::ALU;
        d.has_rd = true;
        d.num_src_regs = 1;
        d.imm = isa::imm_i(instruction);
        switch (f3) {
            case isa::FUNCT3_ADD_SUB: d.alu_op = AluOp::ADD;  break;  // ADDI
            case isa::FUNCT3_SLT:     d.alu_op = AluOp::SLT;  break;  // SLTI
            case isa::FUNCT3_SLTU:    d.alu_op = AluOp::SLTU; break;  // SLTIU
            case isa::FUNCT3_XOR:     d.alu_op = AluOp::XOR;  break;  // XORI
            case isa::FUNCT3_OR:      d.alu_op = AluOp::OR;   break;  // ORI
            case isa::FUNCT3_AND:     d.alu_op = AluOp::AND;  break;  // ANDI
            case isa::FUNCT3_SLL:
                if (f7 != isa::FUNCT7_BASE) { d.type = InstructionType::INVALID; break; }
                d.alu_op = AluOp::SLL;
                d.imm = d.imm & 0x1F;  // shamt
                break;
            case isa::FUNCT3_SRL_SRA:
                if (f7 == isa::FUNCT7_BASE) d.alu_op = AluOp::SRL;
                else if (f7 == isa::FUNCT7_ALT) d.alu_op = AluOp::SRA;
                else { d.type = InstructionType::INVALID; break; }
                d.imm = d.imm & 0x1F;  // shamt
                break;
        }
        break;
    }

    // ---- LUI ----
    case isa::OP_LUI:
        d.type = InstructionType::LUI;
        d.target_unit = ExecUnit::ALU;
        d.has_rd = true;
        d.num_src_regs = 0;
        d.imm = isa::imm_u(instruction);
        break;

    // ---- AUIPC ----
    case isa::OP_AUIPC:
        d.type = InstructionType::AUIPC;
        d.target_unit = ExecUnit::ALU;
        d.has_rd = true;
        d.num_src_regs = 0;
        d.imm = isa::imm_u(instruction);
        break;

    // ---- JAL ----
    case isa::OP_JAL:
        d.type = InstructionType::JAL;
        d.target_unit = ExecUnit::ALU;
        d.has_rd = true;
        d.num_src_regs = 0;
        d.imm = isa::imm_j(instruction);
        break;

    // ---- JALR ----
    case isa::OP_JALR:
        if (f3 != 0) {
            d.type = InstructionType::INVALID;
            break;
        }
        d.type = InstructionType::JALR;
        d.target_unit = ExecUnit::ALU;
        d.has_rd = true;
        d.num_src_regs = 1;
        d.imm = isa::imm_i(instruction);
        break;

    // ---- Branches ----
    case isa::OP_BRANCH:
        d.type = InstructionType::BRANCH;
        d.target_unit = ExecUnit::ALU;
        d.has_rd = false;
        d.num_src_regs = 2;
        d.imm = isa::imm_b(instruction);
        switch (f3) {
            case isa::FUNCT3_BEQ:  d.branch_op = BranchOp::BEQ;  break;
            case isa::FUNCT3_BNE:  d.branch_op = BranchOp::BNE;  break;
            case isa::FUNCT3_BLT:  d.branch_op = BranchOp::BLT;  break;
            case isa::FUNCT3_BGE:  d.branch_op = BranchOp::BGE;  break;
            case isa::FUNCT3_BLTU: d.branch_op = BranchOp::BLTU; break;
            case isa::FUNCT3_BGEU: d.branch_op = BranchOp::BGEU; break;
            default: d.type = InstructionType::INVALID; break;
        }
        break;

    // ---- Loads ----
    case isa::OP_LOAD:
        d.type = InstructionType::LOAD;
        d.target_unit = ExecUnit::LDST;
        d.has_rd = true;
        d.num_src_regs = 1;
        d.imm = isa::imm_i(instruction);
        switch (f3) {
            case isa::FUNCT3_LB:  d.mem_op = MemOp::LB;  break;
            case isa::FUNCT3_LH:  d.mem_op = MemOp::LH;  break;
            case isa::FUNCT3_LW:  d.mem_op = MemOp::LW;  break;
            case isa::FUNCT3_LBU: d.mem_op = MemOp::LBU; break;
            case isa::FUNCT3_LHU: d.mem_op = MemOp::LHU; break;
            default: d.type = InstructionType::INVALID; break;
        }
        break;

    // ---- Stores ----
    case isa::OP_STORE:
        d.type = InstructionType::STORE;
        d.target_unit = ExecUnit::LDST;
        d.has_rd = false;
        d.num_src_regs = 2;
        d.imm = isa::imm_s(instruction);
        switch (f3) {
            case isa::FUNCT3_SB: d.mem_op = MemOp::SB; break;
            case isa::FUNCT3_SH: d.mem_op = MemOp::SH; break;
            case isa::FUNCT3_SW: d.mem_op = MemOp::SW; break;
            default: d.type = InstructionType::INVALID; break;
        }
        break;

    // ---- FENCE (treated as NOP per RV32I base spec for single-hart) ----
    case isa::OP_FENCE:
        d.type = InstructionType::FENCE;
        d.target_unit = ExecUnit::ALU;
        d.has_rd = false;
        d.num_src_regs = 0;
        break;

    // ---- SYSTEM (ECALL, EBREAK, CSR) ----
    case isa::OP_SYSTEM:
        if (f3 == isa::FUNCT3_ECALL_EBREAK) {
            uint32_t imm_val = (instruction >> 20) & 0xFFF;
            if (imm_val == 0) {
                d.type = InstructionType::ECALL;
                d.target_unit = ExecUnit::SYSTEM;
                d.has_rd = false;
                d.num_src_regs = 0;
            } else if (imm_val == 1) {
                d.type = InstructionType::EBREAK;
                d.target_unit = ExecUnit::SYSTEM;
                d.has_rd = false;
                d.num_src_regs = 0;
            } else {
                d.type = InstructionType::INVALID;
            }
        } else if (f3 == isa::FUNCT3_CSRRS && d.rs1 == 0) {
            uint16_t csr = static_cast<uint16_t>((instruction >> 20) & 0xFFF);
            // Only accept known GPU identity CSRs
            if (csr == isa::CSR_WARP_ID || csr == isa::CSR_LANE_ID ||
                csr == isa::CSR_NUM_WARPS) {
                d.type = InstructionType::CSR;
                d.target_unit = ExecUnit::ALU;
                d.has_rd = true;
                d.num_src_regs = 0;
                d.csr_addr = csr;
            } else {
                d.type = InstructionType::INVALID;
            }
        } else {
            d.type = InstructionType::INVALID;
        }
        break;

    // ---- VDOT8 (custom-0) ----
    case isa::OP_VDOT8:
        if (f7 == 0x00 && f3 == 0x00) {
            d.type = InstructionType::VDOT8;
            d.target_unit = ExecUnit::MULTIPLY;
            d.has_rd = true;
            d.reads_rd = true;   // rd is both source and destination
            d.num_src_regs = 3;  // rs1, rs2, rd all need scoreboard check
        } else {
            d.type = InstructionType::INVALID;
        }
        break;

    // ---- TLOOKUP (custom-1) ----
    case isa::OP_TLOOKUP:
        d.type = InstructionType::TLOOKUP;
        d.target_unit = ExecUnit::TLOOKUP;
        d.has_rd = true;
        d.num_src_regs = 1;  // Only rs1 needs scoreboard check
        d.imm = isa::imm_i(instruction);
        break;

    default:
        d.type = InstructionType::INVALID;
        break;
    }

    return d;
}

} // namespace gpu_sim
