#include "gpu_sim/timing/decode_stage.h"
#include "gpu_sim/timing/operand_collector.h"

namespace gpu_sim {

DecodeStage::DecodeStage(WarpState* warps, FetchStage& fetch)
    : warps_(warps), fetch_(fetch) {}

void DecodeStage::evaluate() {
    // Phase 6: write only into next_ slot. commit() flips it; TimingModel
    // observes current_ at the top of the *next* tick. Reset next_ to
    // invalid each evaluate to prevent a stale request from carrying.
    next_ebreak_request_ = EBreakRequest{};

    if (pending_.valid) return;

    const auto& fetch_out = fetch_.current_output();
    if (!fetch_out) return;

    DecodedInstruction decoded = Decoder::decode(fetch_out->raw_instruction);

    if (decoded.type == InstructionType::EBREAK) {
        next_ebreak_request_.valid = true;
        next_ebreak_request_.warp_id = fetch_out->warp_id;
        next_ebreak_request_.pc = fetch_out->pc;
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
    // Phase 6: latch the REGISTERED ebreak side-channel. TimingModel reads
    // current_ebreak_request() at the *top* of the next tick to decide
    // whether to panic_->trigger().
    current_ebreak_request_ = next_ebreak_request_;

    // Phase 5: apply REGISTERED redirect-request from the OperandCollector
    // BEFORE the pending->buffer push. The signal here is opcoll's
    // current_redirect_request_ latched by opcoll.commit() on the previous
    // cycle. Applying the invalidate first prevents this commit from
    // pushing a shadow instruction (the pending entry that decode.evaluate
    // accepted while reading from the wrong path) into the warp's buffer.
    const RedirectRequest req = read_redirect_request(redirect_override_, opcoll_);
    if (req.valid) {
        apply_redirect_invalidate(req.warp_id);
    }

    if (pending_.valid) {
        if (!warps_[pending_.target_warp].instr_buffer.is_full()) {
            warps_[pending_.target_warp].instr_buffer.push(pending_.entry);
            pending_.valid = false;
        }
    }
}

void DecodeStage::reset() {
    current_ebreak_request_ = EBreakRequest{};
    next_ebreak_request_ = EBreakRequest{};
    pending_.valid = false;
    redirect_override_.reset();
}

void DecodeStage::apply_redirect_invalidate(uint32_t warp_id) {
    if (pending_.valid && pending_.target_warp == warp_id) {
        pending_.valid = false;
    }
}

} // namespace gpu_sim
