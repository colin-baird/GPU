#include "gpu_sim/timing/fetch_stage.h"

namespace gpu_sim {

FetchStage::FetchStage(uint32_t num_warps, WarpState* warps,
                       const InstructionMemory& imem, BranchPredictor& predictor, Stats& stats)
    : num_warps_(num_warps), warps_(warps), imem_(imem), predictor_(predictor), stats_(stats) {}

void FetchStage::evaluate() {
    next_output_ = std::nullopt;

    if (stalled_) {
        stats_.fetch_skip_count++;
        return;
    }

    uint32_t w = rr_pointer_;
    if (w < num_warps_ && warps_[w].active && !warps_[w].instr_buffer.is_full()) {
        uint32_t pc = warps_[w].pc;
        FetchOutput out;
        out.raw_instruction = imem_.read(pc);
        out.warp_id = w;
        out.pc = pc;
        out.prediction = predictor_.predict(pc, out.raw_instruction);
        next_output_ = out;
        warps_[w].pc = out.prediction.predicted_taken ? out.prediction.predicted_target
                                                      : (pc + 4);
    } else {
        stats_.fetch_skip_count++;
    }

    rr_pointer_ = (rr_pointer_ + 1) % num_warps_;
}

void FetchStage::commit() {
    current_output_ = next_output_;
}

void FetchStage::reset() {
    rr_pointer_ = 0;
    stalled_ = false;
    current_output_ = std::nullopt;
    next_output_ = std::nullopt;
}

void FetchStage::redirect_warp(uint32_t warp_id, uint32_t target_pc) {
    warps_[warp_id].pc = target_pc;
    warps_[warp_id].instr_buffer.flush();
    // Invalidate any in-flight fetch for this warp
    if (current_output_ && current_output_->warp_id == warp_id) {
        current_output_ = std::nullopt;
    }
    if (next_output_ && next_output_->warp_id == warp_id) {
        next_output_ = std::nullopt;
    }
}

} // namespace gpu_sim
