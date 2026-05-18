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

    // Phase 10E: apply the COMBINATIONAL-backward redirect-invalidate at the
    // top of evaluate(), BEFORE the carry-forward guard. The ALU resolved the
    // branch earlier in this same tick (back-to-front sweep); fetch.evaluate()
    // also already ran this cycle and flushed the redirected warp's buffer and
    // cleared its fetch output. The invalidate targets next_pending_ — the
    // slot evaluate() owns and commit() will flip into current_pending_ and
    // push to the buffer. A shadow entry decoded last cycle is carried here in
    // next_pending_ (seed_next() copied it from current_pending_); dropping it
    // before the guard both prevents the shadow push and lets evaluate() pull
    // a fresh correct-path fetch output (which fetch already cleared to
    // nullopt for the redirected warp, so nothing wrong is re-staged).
    RedirectRequest req;
    if (redirect_override_) {
        req = *redirect_override_;
    } else if (alu_) {
        req = alu_->next_redirect();
    }
    if (req.valid) {
        apply_redirect_invalidate(req.warp_id);
    }

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
    // next_pending_ (and Phase 10E: already applied any redirect-invalidate
    // to it the same cycle the branch resolved). The pending->buffer push
    // below is a committed-state mutation and operates on current_pending_
    // after the flip.
    current_pending_ = next_pending_;

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
    // Phase 10E: called from the top of evaluate(), so it operates on
    // next_pending_ — the slot evaluate() owns and commit() flips into
    // current_pending_ before the pending->buffer push. Dropping the shadow
    // entry here prevents commit() from pushing it into the warp buffer.
    if (next_pending_.valid && next_pending_.target_warp == warp_id) {
        next_pending_.valid = false;
    }
}

} // namespace gpu_sim
