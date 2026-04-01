#include "gpu_sim/timing/decode_stage.h"

namespace gpu_sim {

DecodeStage::DecodeStage(WarpState* warps, FetchStage& fetch)
    : warps_(warps), fetch_(fetch) {}

void DecodeStage::evaluate() {
    ebreak_detected_ = false;

    if (pending_.valid) return;

    const auto& fetch_out = fetch_.current_output();
    if (!fetch_out) return;

    DecodedInstruction decoded = Decoder::decode(fetch_out->raw_instruction);

    if (decoded.type == InstructionType::EBREAK) {
        ebreak_detected_ = true;
        ebreak_warp_id_ = fetch_out->warp_id;
        ebreak_pc_ = fetch_out->pc;
        return;
    }

    BufferEntry entry;
    entry.decoded = decoded;
    entry.warp_id = fetch_out->warp_id;
    entry.pc = fetch_out->pc;
    entry.prediction = fetch_out->prediction;

    pending_.entry = entry;
    pending_.target_warp = fetch_out->warp_id;
    pending_.valid = true;
}

void DecodeStage::commit() {
    if (pending_.valid) {
        if (!warps_[pending_.target_warp].instr_buffer.is_full()) {
            warps_[pending_.target_warp].instr_buffer.push(pending_.entry);
            pending_.valid = false;
        }
    }
}

void DecodeStage::reset() {
    ebreak_detected_ = false;
    pending_.valid = false;
}

void DecodeStage::invalidate_warp(uint32_t warp_id) {
    if (pending_.valid && pending_.target_warp == warp_id) {
        pending_.valid = false;
    }
}

} // namespace gpu_sim
