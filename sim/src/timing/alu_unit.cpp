#include "gpu_sim/timing/alu_unit.h"
#include "gpu_sim/timing/writeback_arbiter.h"

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
    // Phase 10B.1: REGISTERED opcoll->unit edge via the pull model. The ALU
    // reads opcoll's committed output at the top of evaluate() and self-selects
    // by decoded.target_unit; if it matches, it latches the payload via its own
    // accept(). The scheduler's unit_busy_/bitmap gating (10B.0) guarantees the
    // output never targets a unit that cannot accept it. opcoll_ is null in
    // isolated unit tests, which drive accept() directly.
    if (opcoll_ != nullptr) {
        const auto& dispatched = opcoll_->current_output();
        if (dispatched && dispatched->decoded.target_unit == ExecUnit::ALU) {
            accept(*dispatched, sim_cycle_ != nullptr ? *sim_cycle_ : 0);
        }
    }

    // Phase 10B.3: the result buffer is a plain double-buffered pipeline
    // register — evaluate() assigns next_result_buffer_ fresh every cycle. A
    // 1-cycle ALU produces at most one result per cycle, so the default is
    // "no result"; the accepted op (if any) overwrites it below.
    next_result_buffer_ = WritebackEntry{};

    // Phase 10B.0.5: assign the per-cycle scratch flag fresh every cycle so a
    // re-evaluated stalled cycle recomputes it from scratch. commit()
    // consumes it to perform the Stats increments.
    processed_this_cycle_ = next_has_pending_;

    // Read next_has_pending_ here: this is the input written THIS cycle by
    // accept() (combinational forward inside the unit). Cross-cycle state
    // would be read from current_*; ALU has no committed pending input
    // because it always drains its accept in a single cycle.
    if (next_has_pending_) {
        // Phase 10B.3: only writeback ops (has_rd && rd != 0) present a result
        // to the writeback arbiter. A non-writeback ALU op — a conditional
        // branch, an ALU op writing x0 — reserves NO writeback-bitmap slot at
        // issue (10B.0: writes_back = has_rd && rd != 0), so it must not
        // occupy the writeback port either. Producing a result buffer entry
        // for it would make the arbiter see an unscheduled fixed-latency
        // result that can collide with a genuine bitmap-scheduled writeback
        // (the count_fixed_with_result() <= 1 invariant). Branch RESOLUTION
        // still runs below regardless — it is independent of the writeback
        // datapath. JAL/JALR write the link register and DO reserve a slot.
        const bool writes_back = next_pending_input_.decoded.has_rd &&
                                 next_pending_input_.decoded.rd != 0;
        // 1-cycle latency: result is available after this evaluate. We write
        // into next_result_buffer_; commit() will publish it as
        // current_result_buffer_, which is what the writeback arbiter reads
        // (via current_has_result()) on the next cycle.
        next_result_buffer_.valid = writes_back;
        next_result_buffer_.warp_id = next_pending_input_.warp_id;
        next_result_buffer_.dest_reg = next_pending_input_.decoded.rd;
        next_result_buffer_.values = next_pending_input_.trace.results;
        next_result_buffer_.source_unit = ExecUnit::ALU;
        next_result_buffer_.pc = next_pending_input_.pc;
        next_result_buffer_.raw_instruction = next_pending_input_.decoded.raw;
        next_result_buffer_.issue_cycle = pending_cycle_;
        // Phase 10B.0.5: alu_stats.busy_cycles / .instructions are incremented
        // in commit() (gated on processed_this_cycle_), not here.

        // Phase 10A: branch resolution. Phase 10B.3: branch resolution is
        // combinational — it still resolves and drives the redirect each
        // stalled cycle — but its SIDE-EFFECTS (predictor update, tracker
        // write, branch Stats) are writes to shared structures the gated
        // commit() does not protect, so they must fire EXACTLY ONCE. The
        // branch_resolved_ control bit (category-4: not gated, not seeded)
        // guards them: fire only when clear, then set it. commit() clears it
        // whenever the resolve-stage register advances (a non-stalled cycle).
        if (next_pending_input_.trace.is_branch) {
            const DispatchInput& in = next_pending_input_;
            const bool mispredicted = branch_mispredicted(in);
            const uint32_t actual_target = in.trace.branch_taken
                ? in.trace.branch_target
                : (in.pc + 4);
            if (mispredicted) {
                // Mispredict: publish the REGISTERED redirect. Recomputed
                // identically on a stalled re-run; the gated commit() dedups
                // the redirect itself, branch_resolved_ dedups the side
                // effects below.
                next_redirect_request_.valid = true;
                next_redirect_request_.warp_id = in.warp_id;
                next_redirect_request_.target_pc = actual_target;
            }
            if (!branch_resolved_) {
                // Side-effects: fire exactly once across a multi-cycle stall.
                stats_.branch_predictions++;
                if (branch_predictor_) {
                    branch_predictor_->update(in.pc, in.decoded, in.prediction,
                                              in.trace.branch_taken,
                                              in.trace.branch_target);
                }
                if (mispredicted) {
                    stats_.branch_mispredictions++;
                    stats_.branch_flushes++;
                } else if (branch_tracker_) {
                    // Correct prediction: clear the in-flight bit (writes the
                    // tracker's next_ slot).
                    branch_tracker_->note_resolved_correctly(in.warp_id);
                }
                branch_resolved_ = true;
            }
        }

        next_has_pending_ = false;
    }
}

void ALUUnit::commit() {
    // Phase 10B.3: writeback-stall self-gate. The arbiter (sequenced first in
    // the evaluate sweep) asserts next_writeback_stall() when a load preempted
    // a fixed-latency writeback. On a stalled cycle this stage holds: no
    // next_->current_ flip, no Stats increment, branch_resolved_ untouched —
    // so the cycle re-evaluates identically next tick. evaluate() ran
    // unconditionally and is re-runnable (10B.0.5). flush() (panic) is NOT
    // gated by the stall and is handled separately by the panic cascade.
    if (wb_arbiter_ != nullptr && wb_arbiter_->next_writeback_stall()) {
        return;
    }

    // Phase 10B.0.5: Stats increments relocated here from evaluate(). Counting
    // them at commit() (which a stalled cycle skips) means a re-evaluated
    // cycle is not double-counted.
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
    current_redirect_request_ = next_redirect_request_;
    next_redirect_request_.valid = false;
    // Phase 10B.3: a non-stalled commit() advances the resolve-stage register,
    // so the branch (if any) leaves the ALU — clear branch_resolved_ so the
    // next branch to occupy the stage resolves its side-effects afresh.
    branch_resolved_ = false;
}

void ALUUnit::reset() {
    current_result_buffer_.valid = false;
    next_result_buffer_.valid = false;
    current_has_pending_ = false;
    next_has_pending_ = false;
    current_redirect_request_ = RedirectRequest{};
    next_redirect_request_ = RedirectRequest{};
    branch_resolved_ = false;
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

bool ALUUnit::current_has_result() const {
    // Phase 10B.3: REGISTERED unit->arbiter edge. The arbiter reads this one
    // cycle after the unit produced the result — committed (current_*) state.
    return current_result_buffer_.valid;
}

WritebackEntry ALUUnit::consume_result() {
    // Phase 10B.3: pure read. Returns the committed entry and mutates nothing.
    // A consumed result clears naturally — the non-stalled commit() lets
    // evaluate() overwrite next_result_buffer_ fresh; a preempted result is
    // held by the stalled (gated) commit().
    return current_result_buffer_;
}

} // namespace gpu_sim
