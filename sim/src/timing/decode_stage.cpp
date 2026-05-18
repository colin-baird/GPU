#include "gpu_sim/timing/decode_stage.h"
#include "gpu_sim/timing/alu_unit.h"

namespace gpu_sim {

DecodeStage::DecodeStage(WarpState* warps, FetchStage& fetch)
    : warps_(warps), fetch_(fetch) {}

void DecodeStage::seed_next() {
    // Phase 10D.0: re-establish the carry-forward pending slot in next_* at
    // the top of the tick. While the current evaluate sweep order is
    // preserved this is a redundant copy (the prior commit() already left
    // next_pending_ == current_pending_), so it is byte-identical.
    next_pending_ = current_pending_;
}

void DecodeStage::evaluate() {
    // Phase 6: write only into next_ slot. commit() flips it; TimingModel
    // observes current_ at the top of the *next* tick. Reset next_ to
    // invalid each evaluate to prevent a stale request from carrying.
    next_ebreak_request_ = EBreakRequest{};

    // Phase 10D.0: evaluate() consumes and mutates next_pending_ (seeded
    // equal to current_pending_ by seed_next()).
    if (next_pending_.valid) return;

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

    next_pending_.entry = entry;
    next_pending_.target_warp = fetch_out->warp_id;
    next_pending_.valid = true;
}

void DecodeStage::commit() {
    // Phase 6: latch the REGISTERED ebreak side-channel. TimingModel reads
    // current_ebreak_request() at the *top* of the next tick to decide
    // whether to panic_->trigger().
    current_ebreak_request_ = next_ebreak_request_;

    // Phase 10D.0: flip the double-buffered pending slot. evaluate() wrote
    // next_pending_; the redirect-invalidate and the pending->buffer push
    // below are committed-state mutations and so operate on current_pending_
    // after the flip — byte-identical to the prior in-place mutation of the
    // single pending_ field, because evaluate() runs before commit() in the
    // same tick and seed_next() leaves next_pending_ == current_pending_ on a
    // cycle that pulls nothing.
    current_pending_ = next_pending_;

    // Phase 10A: apply REGISTERED redirect-request from the ALU BEFORE the
    // pending->buffer push. The signal here is the ALU's
    // current_redirect_request_ latched by alu.commit() on the previous
    // cycle. Applying the invalidate first prevents this commit from
    // pushing a shadow instruction (the pending entry that decode.evaluate
    // accepted while reading from the wrong path) into the warp's buffer.
    // Branch resolution moved from OperandCollector to ALUUnit in Phase 10A.
    RedirectRequest req;
    if (redirect_override_) {
        req = *redirect_override_;
    } else if (alu_) {
        req = alu_->current_redirect_request_or_override(std::nullopt);
    }
    if (req.valid) {
        apply_redirect_invalidate(req.warp_id);
    }

    if (current_pending_.valid) {
        if (!warps_[current_pending_.target_warp].instr_buffer.is_full()) {
            warps_[current_pending_.target_warp].instr_buffer.push(
                current_pending_.entry);
            current_pending_.valid = false;
        }
    }
}

void DecodeStage::reset() {
    current_ebreak_request_ = EBreakRequest{};
    next_ebreak_request_ = EBreakRequest{};
    current_pending_.valid = false;
    next_pending_.valid = false;
    redirect_override_.reset();
}

void DecodeStage::apply_redirect_invalidate(uint32_t warp_id) {
    // Phase 10D.0: called from commit() after the next_->current_ flip, so it
    // operates on the committed slot.
    if (current_pending_.valid && current_pending_.target_warp == warp_id) {
        current_pending_.valid = false;
    }
}

} // namespace gpu_sim
