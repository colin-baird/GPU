#include "gpu_sim/timing/decode_stage.h"
#include "gpu_sim/timing/alu_unit.h"

namespace gpu_sim {

DecodeStage::DecodeStage(WarpState* warps, FetchStage& fetch)
    : warps_(warps), fetch_(fetch) {
    register_state(&ebreak_request_, &pending_);
}

void DecodeStage::seed_next() {
    // Phase 10D.0: re-establish the carry-forward pending slot at the top of
    // the tick. While the current evaluate sweep order is preserved this is a
    // redundant copy (the prior commit() already left the slots equal), so it
    // is byte-identical. seed_all() also seeds ebreak_request_; that is
    // byte-identical because evaluate() unconditionally restages it
    // (EBreakRequest{}) at its top before any conditional write.
    seed_all();
}

void DecodeStage::evaluate() {
    // Phase 6: write only into the staged slot. commit() latches it;
    // TimingModel observes the committed value at the top of the *next* tick.
    // Reset the staged slot to invalid each evaluate to prevent a stale
    // request from carrying (this unconditional restage is also what makes
    // seeding ebreak_request_ via seed_all() byte-identical).
    ebreak_request_.set_next(EBreakRequest{});

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

    // Phase 10D.0: evaluate() consumes and mutates the staged pending slot
    // (seeded equal to the committed slot by seed_next()). This is an
    // intra-stage self-read of seeded carry-forward state — next().
    if (pending_.next().valid) return;

    const auto& fetch_out = fetch_.current_output();
    if (!fetch_out) return;

    DecodedInstruction decoded = Decoder::decode(fetch_out->raw_instruction);

    if (decoded.type == InstructionType::EBREAK) {
        ebreak_request_.next_mut().valid = true;
        ebreak_request_.next_mut().warp_id = fetch_out->warp_id;
        ebreak_request_.next_mut().pc = fetch_out->pc;
        return;
    }

    BufferEntry entry;
    entry.decoded = decoded;
    entry.warp_id = fetch_out->warp_id;
    entry.pc = fetch_out->pc;
    entry.prediction = fetch_out->prediction;

    pending_.next_mut().entry = entry;
    pending_.next_mut().target_warp = fetch_out->warp_id;
    pending_.next_mut().valid = true;
}

void DecodeStage::commit() {
    // Latch the registered state. Phase 6: the REGISTERED ebreak side-channel
    // — TimingModel reads current_ebreak_request() at the *top* of the next
    // tick to decide whether to panic_->trigger(). Phase 10D.0: the
    // double-buffered pending slot — evaluate() wrote the staged value (and
    // Phase 10E: already applied any redirect-invalidate to it the same cycle
    // the branch resolved).
    commit_all();

    // The pending->buffer push is a committed-state mutation and operates on
    // the committed pending slot after the latch above.
    if (pending_.current().valid) {
        if (!warps_[pending_.current().target_warp].instr_buffer.is_full()) {
            warps_[pending_.current().target_warp].instr_buffer.push(
                pending_.current().entry);
            pending_.current_mut().valid = false;
        }
    }
}

void DecodeStage::reset() {
    // reset_all() clears both the committed and staged slots of
    // ebreak_request_ and pending_ (each back to its value-initialized state,
    // which is the prior reset's {} / valid=false).
    reset_all();
    redirect_override_.reset();
}

void DecodeStage::apply_redirect_invalidate(uint32_t warp_id) {
    // Phase 10E: called from the top of evaluate(), so it operates on the
    // staged pending slot — the slot evaluate() owns and commit() latches
    // into the committed slot before the pending->buffer push. Dropping the
    // shadow entry here prevents commit() from pushing it into the warp
    // buffer.
    if (pending_.next().valid && pending_.next().target_warp == warp_id) {
        pending_.next_mut().valid = false;
    }
}

} // namespace gpu_sim
