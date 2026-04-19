#include "gpu_sim/timing/fetch_stage.h"

namespace gpu_sim {

FetchStage::FetchStage(uint32_t num_warps, WarpState* warps,
                       const InstructionMemory& imem, BranchPredictor& predictor, Stats& stats)
    : num_warps_(num_warps), warps_(warps), imem_(imem), predictor_(predictor), stats_(stats) {}

void FetchStage::evaluate() {
    next_output_ = std::nullopt;

    // Backpressure: don't produce if decode hasn't consumed previous output
    if (current_output_.has_value() && !output_consumed_) {
        stats_.fetch_skip_count++;
        stats_.fetch_skip_backpressure++;
        rr_pointer_ = (rr_pointer_ + 1) % num_warps_;
        return;
    }

    // Scan forward from the RR pointer to find the first eligible warp
    bool fetched = false;
    for (uint32_t i = 0; i < num_warps_; ++i) {
        uint32_t w = (rr_pointer_ + i) % num_warps_;
        auto& buf = warps_[w].instr_buffer;
        bool will_be_full = buf.is_full() ||
            (decode_pending_warp_.has_value() && *decode_pending_warp_ == w &&
             buf.size() + 1 >= buf.capacity());
        if (warps_[w].active && !will_be_full) {
            uint32_t pc = warps_[w].pc;
            FetchOutput out;
            out.raw_instruction = imem_.read(pc);
            out.warp_id = w;
            out.pc = pc;
            out.prediction = predictor_.predict(pc, out.raw_instruction);
            next_output_ = out;
            warps_[w].pc = out.prediction.predicted_taken ? out.prediction.predicted_target
                                                          : (pc + 4);
            fetched = true;
            break;
        }
    }

    if (!fetched) {
        stats_.fetch_skip_count++;
        stats_.fetch_skip_all_full++;
    }

    // Pointer always advances to (original + 1), regardless of which warp was fetched
    rr_pointer_ = (rr_pointer_ + 1) % num_warps_;
}

void FetchStage::commit() {
    if (next_output_.has_value()) {
        current_output_ = next_output_;
        output_consumed_ = false;
    } else if (output_consumed_) {
        current_output_ = std::nullopt;
    }
    // else: retain current_output_ for decode to consume
}

void FetchStage::reset() {
    rr_pointer_ = 0;
    output_consumed_ = true;
    current_output_ = std::nullopt;
    next_output_ = std::nullopt;
    decode_pending_warp_ = std::nullopt;
}

void FetchStage::redirect_warp(uint32_t warp_id, uint32_t target_pc) {
    warps_[warp_id].pc = target_pc;
    warps_[warp_id].instr_buffer.flush();
    // Invalidate any in-flight fetch for this warp
    if (current_output_ && current_output_->warp_id == warp_id) {
        current_output_ = std::nullopt;
        output_consumed_ = true;
    }
    if (next_output_ && next_output_->warp_id == warp_id) {
        next_output_ = std::nullopt;
    }
}

} // namespace gpu_sim
