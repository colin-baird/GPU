#include "gpu_sim/timing/alu_unit.h"

namespace gpu_sim {

void ALUUnit::accept(const DispatchInput& input, uint64_t cycle) {
    // Phase 1 discipline: accept() writes only the next_* slot. The unit's
    // own evaluate() runs after accept() in the same tick and treats the
    // freshly-accepted input as combinational ("in flight" through the unit
    // this cycle, latched at commit).
    next_has_pending_ = true;
    next_pending_input_ = input;
    pending_cycle_ = cycle;
}

void ALUUnit::evaluate() {
    // Phase 10B.0.5: assign the per-cycle scratch flag fresh every cycle so a
    // re-evaluated stalled cycle recomputes it from scratch. commit()
    // consumes it to perform the Stats increments.
    processed_this_cycle_ = next_has_pending_;

    // Read next_has_pending_ here: this is the input written THIS cycle by
    // accept() (combinational forward inside the unit). Cross-cycle state
    // would be read from current_*; ALU has no committed pending input
    // because it always drains its accept in a single cycle.
    if (next_has_pending_) {
        // 1-cycle latency: result is available after this evaluate. We write
        // into next_result_buffer_; commit() will publish it as
        // current_result_buffer_, which is what writeback_arbiter.next_has_result()
        // reads next cycle.
        next_result_buffer_.valid = true;
        next_result_buffer_.warp_id = next_pending_input_.warp_id;
        next_result_buffer_.dest_reg = next_pending_input_.decoded.rd;
        next_result_buffer_.values = next_pending_input_.trace.results;
        next_result_buffer_.source_unit = ExecUnit::ALU;
        next_result_buffer_.pc = next_pending_input_.pc;
        next_result_buffer_.raw_instruction = next_pending_input_.decoded.raw;
        next_result_buffer_.issue_cycle = pending_cycle_;
        // Phase 10B.0.5: alu_stats.busy_cycles / .instructions are incremented
        // in commit() (gated on processed_this_cycle_), not here — a
        // re-evaluated stalled cycle must not double-count Stats artifacts.

        // Phase 10A: branch resolution, moved here from TimingModel::tick().
        // The next_has_pending_ guard above is mandatory: next_pending_input_
        // is only meaningful when the ALU actually has an instruction this
        // cycle. Gate the branch work on the trace flag of *this* cycle's
        // input. This is byte-identical to the prior post-opcoll block in
        // TimingModel: opcoll.evaluate -> dispatch_to_unit -> alu.accept ->
        // alu.evaluate all run in the same tick, so the branch the ALU sees
        // here is exactly the branch the old block resolved off opcoll.output().
        // The redirect stays REGISTERED (next_ -> current_ at commit), so
        // fetch/decode still observe last cycle's latched redirect.
        if (next_pending_input_.trace.is_branch) {
            const DispatchInput& in = next_pending_input_;
            stats_.branch_predictions++;
            if (branch_predictor_) {
                branch_predictor_->update(in.pc, in.decoded, in.prediction,
                                          in.trace.branch_taken,
                                          in.trace.branch_target);
            }
            const bool mispredicted = branch_mispredicted(in);
            const uint32_t actual_target = in.trace.branch_taken
                ? in.trace.branch_target
                : (in.pc + 4);
            if (mispredicted) {
                // Mispredict: publish the REGISTERED redirect. The branch-
                // shadow clear is deferred to FetchStage::commit() when it
                // applies the redirect — clearing here would unblock the
                // scheduler to issue a shadow instruction from a not-yet-
                // flushed buffer.
                next_redirect_request_.valid = true;
                next_redirect_request_.warp_id = in.warp_id;
                next_redirect_request_.target_pc = actual_target;
                stats_.branch_mispredictions++;
                stats_.branch_flushes++;
            } else if (branch_tracker_) {
                // Correct prediction: no shadow path, clear the in-flight bit
                // immediately (writes into the tracker's next_ slot).
                branch_tracker_->note_resolved_correctly(in.warp_id);
            }
        }

        next_has_pending_ = false;
    }
}

void ALUUnit::commit() {
    // Phase 10B.0.5: Stats increments relocated here from evaluate(). Counting
    // them at commit() (which a stalled cycle skips) means a re-evaluated
    // cycle is not double-counted. Byte-identical to the prior evaluate()-side
    // count while no stall exists.
    if (processed_this_cycle_) {
        stats_.alu_stats.busy_cycles++;
        stats_.alu_stats.instructions++;
        processed_this_cycle_ = false;
    }

    // Flip next_* -> current_* for every double-buffered field.
    current_result_buffer_ = next_result_buffer_;
    current_has_pending_ = next_has_pending_;
    current_pending_input_ = next_pending_input_;
    // Phase 10A: flip the REGISTERED redirect-request slot, then clear next_
    // so a single mispredict does not repeat-fire on subsequent cycles.
    // Fetch and decode read current_redirect_request_ during their own
    // commit() (which runs before alu.commit() within the same tick), so the
    // redirect they apply is last cycle's, latched by THIS commit a tick ago.
    current_redirect_request_ = next_redirect_request_;
    next_redirect_request_.valid = false;
}

void ALUUnit::reset() {
    current_result_buffer_.valid = false;
    next_result_buffer_.valid = false;
    current_has_pending_ = false;
    next_has_pending_ = false;
    current_redirect_request_ = RedirectRequest{};
    next_redirect_request_ = RedirectRequest{};
}

bool ALUUnit::branch_mispredicted(const DispatchInput& input) {
    if (!input.trace.is_branch) {
        return false;
    }

    const uint32_t predicted_next_pc = input.prediction.predicted_taken
        ? input.prediction.predicted_target
        : (input.pc + 4);
    const uint32_t actual_next_pc = input.trace.branch_taken
        ? input.trace.branch_target
        : (input.pc + 4);

    return predicted_next_pc != actual_next_pc;
}

bool ALUUnit::next_has_result() const {
    // COMBINATIONAL edge with the writeback arbiter: it queries next_has_result()
    // AFTER this unit's evaluate in the same tick, and must see the result
    // produced this cycle (zero cycle-count delta). Read next_* (the live,
    // post-evaluate value).
    return next_result_buffer_.valid;
}

WritebackEntry ALUUnit::consume_result() {
    // Return the live entry; invalidate only the next_* slot. commit() at
    // tick-end latches the empty buffer into current_*.
    WritebackEntry entry = next_result_buffer_;
    next_result_buffer_.valid = false;
    return entry;
}

} // namespace gpu_sim
