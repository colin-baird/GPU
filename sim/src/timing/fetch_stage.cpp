#include "gpu_sim/timing/fetch_stage.h"
#include "gpu_sim/timing/decode_stage.h"
#include "gpu_sim/timing/alu_unit.h"
#include "gpu_sim/timing/branch_shadow_tracker.h"

namespace gpu_sim {

FetchStage::FetchStage(uint32_t num_warps, WarpState* warps,
                       const InstructionMemory& imem, BranchPredictor& predictor, Stats& stats)
    : num_warps_(num_warps), warps_(warps), imem_(imem), predictor_(predictor), stats_(stats) {}

bool FetchStage::query_decode_ready() const {
    if (has_ready_override_) return decode_ready_override_;
    if (decode_) return !decode_->current_busy();
    return true;  // no decode wired (unit-test default): never backpressure
}

std::optional<uint32_t> FetchStage::query_decode_pending_warp() const {
    if (has_pending_override_) return decode_pending_warp_override_;
    if (decode_) return decode_->current_pending_warp();
    return std::nullopt;
}

void FetchStage::evaluate() {
    // Phase 10E: apply the COMBINATIONAL-backward redirect at the top of
    // evaluate(). The ALU resolved the branch earlier in this same tick (the
    // back-to-front sweep runs execution units before the frontend), so
    // alu_->next_redirect() carries this cycle's fresh transient. The
    // test-only override takes precedence for isolated FetchStage tests.
    // apply_redirect() mutates committed warp/buffer/output state directly —
    // this is the redirect-flush moment; the redirected warp is then skipped
    // by the fetch scan below (its buffer was just flushed and PC reset).
    RedirectRequest req;
    if (redirect_override_) {
        req = *redirect_override_;
    } else if (alu_) {
        req = alu_->next_redirect();
    }
    const std::optional<uint32_t> redirected_warp =
        req.valid ? std::optional<uint32_t>(req.warp_id) : std::nullopt;
    if (req.valid) {
        apply_redirect(req.warp_id, req.target_pc);
    }

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

    // Scan forward from the RR pointer to find the first eligible warp.
    // A warp is ineligible if pushing this cycle's pick would land in a full
    // buffer once *all* upstream in-flight instructions targeting that warp
    // commit. Two slots can already hold an instruction destined for w:
    //   1. decode.pending_  — will push at end of this cycle.
    //   2. fetch.current_output_ — held in the fetch→decode register; decode
    //      will pull it next cycle and push it the cycle after.
    // The new pick adds a third push, so the eligibility check must reserve
    // a slot for each of the three. Missing the current_output_ term causes
    // head-of-line decode stalls when the scheduler is backend-bound (e.g.
    // LDST saturated): fetch picks the same warp twice in a row, the second
    // push fails, decode goes pending, and fetch backpressures every cycle
    // until the warp drains.
    const std::optional<uint32_t> current_output_warp =
        current_output_.has_value() ? std::optional<uint32_t>(current_output_->warp_id)
                                    : std::nullopt;

    bool fetched = false;
    for (uint32_t i = 0; i < num_warps_; ++i) {
        uint32_t w = (rr_pointer_ + i) % num_warps_;
        // Phase 10E: skip the warp that was redirected this cycle. Its buffer
        // was just flushed and its PC reset to the resolved target; the
        // earliest a correct-path fetch may issue for it is the NEXT cycle
        // (the mispredict shadow: resolve+flush N -> fetch N+1).
        if (redirected_warp.has_value() && *redirected_warp == w) continue;
        auto& buf = warps_[w].instr_buffer;
        uint32_t inflight_to_w = 0;
        if (decode_pending_warp.has_value() && *decode_pending_warp == w) inflight_to_w++;
        if (current_output_warp.has_value() && *current_output_warp == w) inflight_to_w++;
        const bool will_be_full = buf.is_full() ||
            (buf.size() + inflight_to_w + 1 > buf.capacity());
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
    // Phase 10E: the redirect-apply moved into evaluate() — it reads the
    // ALU's COMBINATIONAL-backward next_redirect() the same cycle the branch
    // resolves. commit() now only flips the REGISTERED output register.
    current_output_ = next_output_;
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
