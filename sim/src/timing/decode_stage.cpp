#include "gpu_sim/timing/decode_stage.h"
#include "gpu_sim/timing/operand_collector.h"

namespace gpu_sim {

DecodeStage::DecodeStage(WarpState* warps, FetchStage& fetch)
    : warps_(warps), fetch_(fetch) {}

void DecodeStage::compute_ready() {
    // Read only committed state. pending_ at this point reflects what last
    // cycle's commit() left behind (commit pushes to instr_buffer when
    // possible; otherwise pending_ persists). evaluate() runs after
    // compute_ready() this cycle and will only consume a new fetch output
    // when pending_.valid is false at that point — which equals committed
    // state. Hence the ready signal is just !pending_.valid.
    ready_to_consume_fetch_ = !pending_.valid;
}

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
    bool redirect_valid = false;
    uint32_t redirect_warp = 0;
    if (has_redirect_override_) {
        redirect_valid = redirect_override_valid_;
        redirect_warp = redirect_override_warp_;
    } else if (opcoll_) {
        const auto& req = opcoll_->current_redirect_request();
        redirect_valid = req.valid;
        redirect_warp = req.warp_id;
    }
    if (redirect_valid) {
        apply_redirect_invalidate(redirect_warp);
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
    ready_to_consume_fetch_ = true;
    has_redirect_override_ = false;
    redirect_override_valid_ = false;
}

void DecodeStage::apply_redirect_invalidate(uint32_t warp_id) {
    if (pending_.valid && pending_.target_warp == warp_id) {
        pending_.valid = false;
    }
}

} // namespace gpu_sim
