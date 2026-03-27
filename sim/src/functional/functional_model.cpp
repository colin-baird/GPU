#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/decoder.h"
#include "gpu_sim/functional/alu.h"
#include "gpu_sim/isa.h"

namespace gpu_sim {

FunctionalModel::FunctionalModel(const SimConfig& config)
    : config_(config)
    , memory_(config.external_memory_size_bytes)
    , instr_mem_(config.instruction_mem_size_bytes)
    , lookup_table_(config.lookup_table_entries) {
    init_kernel(config);
}

void FunctionalModel::init_kernel(const SimConfig& config) {
    config_ = config;
    panicked_ = false;
    panic_warp_ = 0;
    panic_cause_ = 0;
    panic_pc_ = 0;
    sequence_counter_ = 0;

    for (uint32_t w = 0; w < MAX_WARPS; ++w) {
        if (w < config.num_warps) {
            warp_active_[w] = true;
            reg_file_.init_warp(w, config.kernel_args);
        } else {
            warp_active_[w] = false;
        }
    }
}

void FunctionalModel::reset() {
    reg_file_.reset();
    panicked_ = false;
    panic_warp_ = 0;
    panic_cause_ = 0;
    panic_pc_ = 0;
    sequence_counter_ = 0;
    for (uint32_t w = 0; w < MAX_WARPS; ++w) {
        warp_active_[w] = false;
    }
}

uint32_t FunctionalModel::execute_load(MemOp op, uint32_t addr) {
    switch (op) {
        case MemOp::LB: {
            int8_t val = static_cast<int8_t>(memory_.read8(addr));
            return static_cast<uint32_t>(static_cast<int32_t>(val));
        }
        case MemOp::LH: {
            int16_t val = static_cast<int16_t>(memory_.read16(addr));
            return static_cast<uint32_t>(static_cast<int32_t>(val));
        }
        case MemOp::LW:
            return memory_.read32(addr);
        case MemOp::LBU:
            return static_cast<uint32_t>(memory_.read8(addr));
        case MemOp::LHU:
            return static_cast<uint32_t>(memory_.read16(addr));
        default:
            return 0;
    }
}

void FunctionalModel::execute_store(MemOp op, uint32_t addr, uint32_t data) {
    switch (op) {
        case MemOp::SB: memory_.write8(addr, static_cast<uint8_t>(data)); break;
        case MemOp::SH: memory_.write16(addr, static_cast<uint16_t>(data)); break;
        case MemOp::SW: memory_.write32(addr, data); break;
        default: break;
    }
}

uint8_t FunctionalModel::mem_access_size(MemOp op) const {
    switch (op) {
        case MemOp::LB: case MemOp::LBU: case MemOp::SB: return 1;
        case MemOp::LH: case MemOp::LHU: case MemOp::SH: return 2;
        case MemOp::LW: case MemOp::SW: return 4;
        default: return 0;
    }
}

TraceEvent FunctionalModel::execute(WarpId warp_id, uint32_t pc) {
    TraceEvent evt;
    evt.warp_id = warp_id;
    evt.pc = pc;
    evt.sequence_number = sequence_counter_++;

    uint32_t instr_word = instr_mem_.read(pc);
    evt.decoded = Decoder::decode(instr_word);
    const auto& d = evt.decoded;

    for (LaneId lane = 0; lane < WARP_SIZE; ++lane) {
        uint32_t rs1_val = reg_file_.read(warp_id, lane, d.rs1);
        uint32_t rs2_val = reg_file_.read(warp_id, lane, d.rs2);
        uint32_t rd_val  = reg_file_.read(warp_id, lane, d.rd);
        uint32_t result = 0;

        switch (d.type) {
        case InstructionType::ALU_R:
            result = execute_alu(d.alu_op, rs1_val, rs2_val);
            break;

        case InstructionType::ALU_I:
            result = execute_alu(d.alu_op, rs1_val, static_cast<uint32_t>(d.imm));
            break;

        case InstructionType::LUI:
            result = static_cast<uint32_t>(d.imm);
            break;

        case InstructionType::AUIPC:
            result = pc + static_cast<uint32_t>(d.imm);
            break;

        case InstructionType::JAL:
            result = pc + 4;  // Return address
            if (lane == 0) {
                evt.is_branch = true;
                evt.branch_taken = true;
                evt.branch_target = pc + static_cast<uint32_t>(d.imm);
            }
            break;

        case InstructionType::JALR:
            result = pc + 4;  // Return address
            if (lane == 0) {
                evt.is_branch = true;
                evt.branch_taken = true;
                evt.branch_target = (rs1_val + static_cast<uint32_t>(d.imm)) & ~1u;
            }
            break;

        case InstructionType::BRANCH:
            if (lane == 0) {
                evt.is_branch = true;
                evt.branch_taken = evaluate_branch(d.branch_op, rs1_val, rs2_val);
                evt.branch_target = pc + static_cast<uint32_t>(d.imm);
            }
            break;

        case InstructionType::LOAD: {
            uint32_t addr = rs1_val + static_cast<uint32_t>(d.imm);
            evt.mem_addresses[lane] = addr;
            evt.mem_size[lane] = mem_access_size(d.mem_op);
            evt.is_load = true;
            result = execute_load(d.mem_op, addr);
            break;
        }

        case InstructionType::STORE: {
            uint32_t addr = rs1_val + static_cast<uint32_t>(d.imm);
            evt.mem_addresses[lane] = addr;
            evt.store_data[lane] = rs2_val;
            evt.mem_size[lane] = mem_access_size(d.mem_op);
            evt.is_store = true;
            execute_store(d.mem_op, addr, rs2_val);
            break;
        }

        case InstructionType::MUL:
            result = execute_mul(d.muldiv_op, rs1_val, rs2_val);
            break;

        case InstructionType::DIV:
            result = execute_div(d.muldiv_op, rs1_val, rs2_val);
            break;

        case InstructionType::VDOT8:
            result = execute_vdot8(rs1_val, rs2_val, rd_val);
            break;

        case InstructionType::TLOOKUP: {
            uint32_t index = rs1_val + static_cast<uint32_t>(d.imm);
            evt.tlookup_indices[lane] = index;
            evt.is_tlookup = true;
            result = lookup_table_.read(index);
            break;
        }

        case InstructionType::CSR: {
            evt.is_csr = true;
            evt.csr_addr = d.csr_addr;
            switch (d.csr_addr) {
                case isa::CSR_WARP_ID:   result = warp_id;           break;
                case isa::CSR_LANE_ID:   result = lane;              break;
                case isa::CSR_NUM_WARPS: result = config_.num_warps; break;
                default:                 result = 0;                  break;
            }
            break;
        }

        case InstructionType::ECALL:
            if (lane == 0) {
                evt.is_ecall = true;
                warp_active_[warp_id] = false;
            }
            break;

        case InstructionType::EBREAK:
            if (lane == 0) {
                evt.is_ebreak = true;
                evt.panic_cause = reg_file_.read(warp_id, 0, 31);  // r31 lane 0
                panicked_ = true;
                panic_warp_ = warp_id;
                panic_cause_ = evt.panic_cause;
                panic_pc_ = pc;
            }
            break;

        case InstructionType::FENCE:
            break;

        case InstructionType::INVALID:
            break;
        }

        evt.results[lane] = result;

        // Write result to register file
        if (d.has_rd && d.rd != 0 && d.type != InstructionType::ECALL &&
            d.type != InstructionType::EBREAK && d.type != InstructionType::BRANCH &&
            d.type != InstructionType::STORE && d.type != InstructionType::FENCE &&
            d.type != InstructionType::INVALID) {
            reg_file_.write(warp_id, lane, d.rd, result);
        }
    }

    return evt;
}

} // namespace gpu_sim
