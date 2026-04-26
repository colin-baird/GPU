#include "gpu_sim/timing/fetch_stage.h"
#include "gpu_sim/timing/decode_stage.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/timing/branch_shadow_tracker.h"

namespace gpu_sim {

FetchStage::FetchStage(uint32_t num_warps, WarpState* warps,
                       const InstructionMemory& imem, BranchPredictor& predictor, Stats& stats)
    : num_warps_(num_warps), warps_(warps), imem_(imem), predictor_(predictor), stats_(stats) {}

bool FetchStage::query_decode_ready() const {
    if (has_ready_override_) return decode_ready_override_;
    if (decode_) return decode_->ready_to_consume_fetch();
    return true;  // no decode wired (unit-test default): never backpressure
}

std::optional<uint32_t> FetchStage::query_decode_pending_warp() const {
    if (has_pending_override_) return decode_pending_warp_override_;
    if (decode_) return decode_->pending_warp();
    return std::nullopt;
}

void FetchStage::evaluate() {
    // READY/STALL gate: if last cycle's output is still in current_output_
    // and decode is not ready to consume it this cycle, hold the output
    // (carry into next_output_) and do not fetch a new instruction.
    const bool decode_ready = query_decode_ready();
    if (current_output_.has_value() && !decode_ready) {
        next_output_ = current_output_;  // retain — REGISTERED hold
        stats_.fetch_skip_count++;
        stats_.fetch_skip_backpressure++;
        rr_pointer_ = (rr_pointer_ + 1) % num_warps_;
        return;
    }

    next_output_ = std::nullopt;

    const std::optional<uint32_t> decode_pending_warp = query_decode_pending_warp();

    // Scan forward from the RR pointer to find the first eligible warp
    bool fetched = false;
    for (uint32_t i = 0; i < num_warps_; ++i) {
        uint32_t w = (rr_pointer_ + i) % num_warps_;
        auto& buf = warps_[w].instr_buffer;
        bool will_be_full = buf.is_full() ||
            (decode_pending_warp.has_value() && *decode_pending_warp == w &&
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
    // current_output_ is REGISTERED: evaluate() has already encoded the
    // hold-vs-advance decision into next_output_ (carrying current forward
    // when backpressured, producing nullopt or a fresh fetch otherwise).
    current_output_ = next_output_;

    // Phase 5: apply REGISTERED redirect-request from the OperandCollector
    // (or test override). The signal we read here is the producer's
    // current_redirect_request_, latched by opcoll.commit() on the previous
    // cycle (opcoll.commit() runs after fetch.commit() within the same
    // tick). So a misprediction observed in opcoll.evaluate() at cycle N
    // becomes visible to fetch.commit() at cycle N+1, applying the flush
    // there. This adds +1 cycle to mispredict-recovery vs. the prior
    // mid-tick mutation — accepted by Option A.
    const RedirectRequest req = read_redirect_request(redirect_override_, opcoll_);
    if (req.valid) {
        apply_redirect(req.warp_id, req.target_pc);
    }
}

void FetchStage::reset() {
    rr_pointer_ = 0;
    current_output_ = std::nullopt;
    next_output_ = std::nullopt;
    has_pending_override_ = false;
    decode_pending_warp_override_ = std::nullopt;
    has_ready_override_ = false;
    decode_ready_override_ = true;
    redirect_override_.reset();
}

void FetchStage::apply_redirect(uint32_t warp_id, uint32_t target_pc) {
    warps_[warp_id].pc = target_pc;
    warps_[warp_id].instr_buffer.flush();
    // Invalidate any in-flight fetch for this warp.
    if (current_output_ && current_output_->warp_id == warp_id) {
        current_output_ = std::nullopt;
    }
    if (next_output_ && next_output_->warp_id == warp_id) {
        next_output_ = std::nullopt;
    }
    // Phase 5: clear branch_in_flight in the tracker's next_ slot at the
    // same moment we apply the redirect-flush. This is the deferred half
    // of the mispredict-resolve described at OperandCollector::resolve_branch:
    // through the cycle where the redirect is applied, the scheduler must
    // still see current_branch_in_flight==true so it does not issue a
    // shadow instruction from the soon-to-be-flushed buffer. branch_tracker
    // .commit() at end-of-cycle then makes current_=false visible to next
    // cycle's scheduler.
    if (branch_tracker_) {
        branch_tracker_->note_redirect_applied(warp_id);
    }
}

} // namespace gpu_sim
