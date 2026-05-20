#include "gpu_sim/timing/fetch_stage.h"
#include "gpu_sim/timing/decode_stage.h"
#include "gpu_sim/timing/alu_unit.h"
#include "gpu_sim/timing/branch_shadow_tracker.h"

namespace gpu_sim {

FetchStage::FetchStage(uint32_t num_warps, WarpState* warps,
                       const InstructionMemory& imem, BranchPredictor& predictor, Stats& stats)
    : num_warps_(num_warps), warps_(warps), imem_(imem), predictor_(predictor), stats_(stats) {
    register_state(&rr_pointer_, &output_);
}

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
    // apply_redirect() mutates committed warp PC and buffer state and clears
    // the staged output slot; the committed output slot is NOT touched mid-
    // cycle (Q does not change between clock edges). The READY/STALL gate
    // and the eligibility scan below combinationally mask the redirected
    // warp so the residual committed output is treated as cleared for this
    // cycle's decisions; decode applies the same combinational mask when
    // reading fetch.current_output(). The redirected warp is also skipped
    // by the fetch scan (its buffer was just flushed, PC reset).
    RedirectRequest req;
    if (redirect_override_) {
        req = *redirect_override_;
    } else if (alu_) {
        req = alu_->next_redirect();
    }
    if (req.valid) {
        apply_redirect(req.warp_id, req.target_pc);
    }

    // READY/STALL gate: if last cycle's output is still in the committed
    // output slot and decode is not ready to consume it this cycle, hold the
    // output (carry it into the staged slot) and do not fetch a new
    // instruction.
    //
    // Combinational redirect mask: a current_ output whose warp matches this
    // cycle's redirect is doomed — its buffer was flushed, its PC reset, and
    // decode will combinationally gate its read of fetch.current_output()
    // against the same redirect signal. Treat the slot as cleared for the
    // gate's purposes so the hold path doesn't re-stage the doomed value.
    const bool decode_ready = query_decode_ready();
    const bool current_is_doomed =
        output_.current().has_value() && req.targets(output_.current()->warp_id);
    if (output_.current().has_value() && !current_is_doomed && !decode_ready) {
        output_.set_next(output_.current());  // retain — REGISTERED hold
        stats_.fetch_skip_count++;
        stats_.fetch_skip_backpressure++;
        rr_pointer_.set_next((rr_pointer_.current() + 1) % num_warps_);
        return;
    }

    output_.set_next(std::nullopt);

    const std::optional<uint32_t> decode_pending_warp = query_decode_pending_warp();

    // Scan forward from the RR pointer to find the first eligible warp.
    // A warp is ineligible if pushing this cycle's pick would land in a full
    // buffer once *all* upstream in-flight instructions targeting that warp
    // commit. Two slots can already hold an instruction destined for w:
    //   1. decode.pending_  — will push at end of this cycle.
    //   2. fetch.output_ (committed) — held in the fetch→decode register;
    //      decode will pull it next cycle and push it the cycle after.
    // The new pick adds a third push, so the eligibility check must reserve
    // a slot for each of the three. Missing the committed-output term causes
    // head-of-line decode stalls when the scheduler is backend-bound (e.g.
    // LDST saturated): fetch picks the same warp twice in a row, the second
    // push fails, decode goes pending, and fetch backpressures every cycle
    // until the warp drains.
    // Same combinational mask as the READY/STALL gate above: a current_
    // output whose warp matches this cycle's redirect is doomed and won't
    // commit to decode, so it doesn't claim a buffer slot in the eligibility
    // scan's inflight_to_w accounting.
    const std::optional<uint32_t> current_output_warp =
        (output_.current().has_value() && !current_is_doomed)
            ? std::optional<uint32_t>(output_.current()->warp_id)
            : std::nullopt;

    bool fetched = false;
    for (uint32_t i = 0; i < num_warps_; ++i) {
        uint32_t w = (rr_pointer_.current() + i) % num_warps_;
        // Phase 10E: skip the warp that was redirected this cycle. Its buffer
        // was just flushed and its PC reset to the resolved target; the
        // earliest a correct-path fetch may issue for it is the NEXT cycle
        // (the mispredict shadow: resolve+flush N -> fetch N+1).
        if (req.targets(w)) continue;
        auto& buf = warps_[w].instr_buffer;
        uint32_t inflight_to_w = 0;
        if (decode_pending_warp.has_value() && *decode_pending_warp == w) inflight_to_w++;
        if (current_output_warp.has_value() && *current_output_warp == w) inflight_to_w++;
        const bool will_be_full = buf.is_full() ||
            (buf.size() + inflight_to_w + 1 > buf.capacity());
        // Phase 2 (close-the-Reg-family-migration): active_ is Reg<bool>;
        // committed read via .current(). The ECALL-retirement path runs
        // earlier in this same tick and drives next_=false; the
        // deactivation_request_ Wire combinationally forwards that intent so
        // the deactivated warp is masked out the same cycle (the staged-false
        // will become visible via .current() at the next cycle's commit
        // boundary).
        const bool deactivating =
            deactivation_request_ && (*deactivation_request_).value()[w];
        if (warps_[w].active_.current() && !deactivating && !will_be_full) {
            uint32_t pc = warps_[w].pc_.current();
            FetchOutput out;
            out.raw_instruction = imem_.read(pc);
            out.warp_id = w;
            out.pc = pc;
            out.prediction = predictor_.predict(pc, out.raw_instruction);
            output_.set_next(out);
            // pc_ is the sole-writer Reg for this stage: stage next-cycle's PC.
            warps_[w].pc_.set_next(out.prediction.predicted_taken
                                       ? out.prediction.predicted_target
                                       : (pc + 4));
            fetched = true;
            break;
        }
    }

    if (!fetched) {
        stats_.fetch_skip_count++;
        stats_.fetch_skip_all_full++;
    }

    // Pointer always advances to (original + 1), regardless of which warp was
    // fetched. The advanced value is staged and latched by commit().
    rr_pointer_.set_next((rr_pointer_.current() + 1) % num_warps_);
}

void FetchStage::commit() {
    // The output slot is REGISTERED: evaluate() has already encoded the
    // hold-vs-advance decision into the staged slot (carrying current forward
    // when backpressured, producing nullopt or a fresh fetch otherwise) and
    // staged the advanced rr_pointer_. Phase 10E: the redirect-apply moved
    // into evaluate() — it reads the ALU's COMBINATIONAL-backward
    // next_redirect() the same cycle the branch resolves. commit() now only
    // latches the REGISTERED state.
    commit_all();
}

void FetchStage::reset() {
    // reset_all() clears both the committed and staged slots of rr_pointer_
    // and output_ (rr_pointer_ -> 0, output_ -> nullopt).
    reset_all();
    has_pending_override_ = false;
    decode_pending_warp_override_ = std::nullopt;
    has_ready_override_ = false;
    decode_ready_override_ = true;
    redirect_override_.reset();
}

void FetchStage::apply_redirect(uint32_t warp_id, uint32_t target_pc) {
    // Phase 2 (close-the-Reg-family-migration): pc_ is Reg<uint32_t>; stage
    // next-cycle's PC. The redirected warp is also skipped by the eligibility
    // scan above (via req.targets(w)), so no fetch this cycle reads the
    // staged value before commit() flips it.
    warps_[warp_id].pc_.set_next(target_pc);
    warps_[warp_id].instr_buffer.flush();
    // Invalidate any in-flight fetch for this warp. The staged-slot clear
    // suffices: the rest of evaluate() either re-stages (advance path) or holds
    // current_ (the gate further down masks the redirected warp so the doomed
    // committed output is treated as cleared and the hold path is skipped).
    // The committed slot is *not* modified here — Q does not change between
    // clock edges. Downstream consumers (decode) combinationally gate their
    // read of output_.current() against the same redirect Wire (the
    // synthesis-faithful encoding of "consumer sees the slot as cleared
    // mid-cycle"). The committed slot rolls to nullopt or the new fetch at
    // this cycle's commit boundary.
    if (output_.next() && output_.next()->warp_id == warp_id) {
        output_.next_mut() = std::nullopt;
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
