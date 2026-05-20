#include "gpu_sim/timing/alu_unit.h"
#include "gpu_sim/timing/writeback_arbiter.h"

namespace gpu_sim {

void ALUUnit::accept(const DispatchInput& input, uint64_t cycle) {
    // Phase 1 discipline: accept() writes only the staged slot. The unit's
    // own evaluate() runs after accept() in the same tick and treats the
    // freshly-accepted input as combinational ("in flight" through the unit
    // this cycle, latched at commit).
    has_pending_.set_next(true);
    pending_input_.set_next(input);
    pending_cycle_.drive(cycle);
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
    // register — assign it fresh every cycle. A 1-cycle ALU produces at most
    // one result per cycle, so the default is "no result"; the accepted op
    // (if any) overwrites it below.
    result_buffer_.set_next(WritebackEntry{});

    // Phase 10E: the redirect is a COMBINATIONAL-backward transient. Reset it
    // at the top of every evaluate() so a stalled re-run, or a cycle with no
    // mispredict, asserts nothing. The test-only override (if set) drives it;
    // a real resolved branch below may overwrite it via another drive().
    // Phase 7: backed by Wire<RedirectRequest>. reset() de-asserts (default-
    // constructed empty RedirectRequest); drive() asserts the override or the
    // resolved redirect.
    next_redirect_.reset();
    if (redirect_override_) {
        next_redirect_.drive(*redirect_override_);
    }

    // Phase 7 of current_mut() elimination: processed_this_cycle_ is
    // driven unconditionally from has_pending_.next() (so reset would be
    // redundant). pending_cycle_ was driven by accept() (this cycle or
    // a prior one if has_pending_ carries).
    processed_this_cycle_.drive(has_pending_.next());

    // Read has_pending_.next() here: this is the input written THIS cycle by
    // accept() (combinational forward inside the unit). Cross-cycle state
    // would be read from current(); ALU has no committed pending input
    // because it always drains its accept in a single cycle.
    if (has_pending_.next()) {
        const DispatchInput& in = pending_input_.next();
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
        const bool writes_back = in.decoded.has_rd && in.decoded.rd != 0;
        // 1-cycle latency: result is available after this evaluate. We write
        // into the staged slot; commit() will publish it as the committed
        // result_buffer_, which is what the writeback arbiter reads (via
        // current_has_result()) on the next cycle.
        WritebackEntry& rb = result_buffer_.next_mut();
        rb.valid = writes_back;
        rb.warp_id = in.warp_id;
        rb.dest_reg = in.decoded.rd;
        rb.values = in.trace.results;
        rb.source_unit = ExecUnit::ALU;
        rb.pc = in.pc;
        rb.raw_instruction = in.decoded.raw;
        rb.issue_cycle = pending_cycle_.value();
        // Phase 10B.0.5: alu_stats.busy_cycles / .instructions are incremented
        // in commit() (gated on processed_this_cycle_), not here.

        // Phase 10A: branch resolution. Phase 10B.3 / 10E: branch resolution
        // is combinational — it re-runs each stalled cycle — but every
        // observable effect must fire EXACTLY ONCE. The branch_resolved_
        // control bit (category-4: not gated, not seeded) guards them: fire
        // only when clear, then set it. commit() clears it whenever the
        // resolve-stage register advances (a non-stalled cycle).
        //
        // Phase 10E: the redirect itself is now a COMBINATIONAL-backward
        // transient (no gated commit() to dedup it), so its assertion is
        // gated by branch_resolved_ alongside the predictor/tracker/Stats
        // side-effects. A branch held at the resolve stage by a multi-cycle
        // writeback stall therefore asserts next_redirect() exactly once —
        // on the first (non-resolved) evaluate of that branch — and the
        // frontend, which is not gated by the stall, applies the flush that
        // same cycle (see fetch_stage.cpp / decode_stage.cpp).
        // Phase 1 (reg-family closeout): branch_resolved_ is now Reg<bool>.
        // Read .next() (intra-stage self-read of the staged slot) so a
        // stalled commit() that left next_=true blocks the next evaluate
        // from re-firing — Reg::current() would still be false in that
        // case because commit_all() did not flip. set_next(true) below
        // arms the same persistence for any future stalled re-evaluations
        // of THIS branch; a non-stalled commit() will set_next(false) and
        // flip both slots, resetting for the next branch.
        if (in.trace.is_branch && !branch_resolved_.next()) {
            const bool mispredicted = branch_mispredicted(in);
            const uint32_t actual_target = in.trace.branch_taken
                ? in.trace.branch_target
                : (in.pc + 4);
            stats_.branch_predictions++;
            if (branch_predictor_) {
                branch_predictor_->update(in.pc, in.decoded, in.prediction,
                                          in.trace.branch_taken,
                                          in.trace.branch_target);
            }
            if (mispredicted) {
                // Assert the combinational-backward redirect this cycle.
                // Phase 7: full-value drive() into the Wire (the prior
                // field-by-field mutation on a plain struct is equivalent —
                // .valid started false from the reset above).
                RedirectRequest req;
                req.valid = true;
                req.warp_id = in.warp_id;
                req.target_pc = actual_target;
                next_redirect_.drive(req);
                stats_.branch_mispredictions++;
                stats_.branch_flushes++;
            } else if (branch_tracker_) {
                // Correct prediction: clear the in-flight bit (writes the
                // tracker's next_ slot).
                branch_tracker_->note_resolved_correctly(in.warp_id);
            }
            branch_resolved_.set_next(true);
        }

        has_pending_.set_next(false);
    }
}

void ALUUnit::commit() {
    // Phase 10B.3: writeback-stall self-gate. The arbiter (sequenced first in
    // the evaluate sweep) asserts next_writeback_stall() when a load preempted
    // a fixed-latency writeback. On a stalled cycle this stage holds: no
    // commit_all() flip, no Stats increment, branch_resolved_ untouched —
    // so the cycle re-evaluates identically next tick. evaluate() ran
    // unconditionally and is re-runnable (10B.0.5). flush() (panic) is NOT
    // gated by the stall and is handled separately by the panic cascade.
    if (wb_arbiter_ != nullptr && wb_arbiter_->next_writeback_stall()) {
        return;
    }

    // Phase 10B.0.5: Stats increments relocated here from evaluate(). Counting
    // them at commit() (which a stalled cycle skips) means a re-evaluated
    // cycle is not double-counted.
    if (processed_this_cycle_.value()) {
        stats_.alu_stats.busy_cycles++;
        stats_.alu_stats.instructions++;
    }
    processed_this_cycle_.reset();

    // Phase 10E: the redirect is a COMBINATIONAL-backward transient — no
    // committed slot, no flip. evaluate() resets and re-asserts it each
    // cycle; the frontend reads it the same cycle (back-to-front sweep).
    // Phase 10B.3: a non-stalled commit() advances the resolve-stage register,
    // so the branch (if any) leaves the ALU — stage false into
    // branch_resolved_ before commit_all() so the next branch to occupy the
    // stage resolves its side-effects afresh (both current_ and next_ end up
    // false after the flip). Phase 1: Reg<bool> set_next(false) replaces the
    // prior plain-field write.
    branch_resolved_.set_next(false);
    commit_all();
}

void ALUUnit::reset() {
    // Phase 1 (reg-family closeout): branch_resolved_ is now Reg<bool>;
    // reset_all() clears both current_ and next_ to false.
    reset_all();
    // Phase 7: Wire<RedirectRequest> de-asserts via reset() (default empty
    // RedirectRequest) — equivalent to the prior `= RedirectRequest{}` clear.
    next_redirect_.reset();
    redirect_override_.reset();
    pending_cycle_.reset();
    processed_this_cycle_.reset();
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
    // cycle after the unit produced the result — committed state.
    return result_buffer_.current().valid;
}

WritebackEntry ALUUnit::consume_result() {
    // Phase 10B.3: pure read. Returns the committed entry and mutates nothing.
    // A consumed result clears naturally — the non-stalled commit() lets
    // evaluate() overwrite the staged slot fresh; a preempted result is held
    // by the stalled (gated) commit().
    return result_buffer_.current();
}

} // namespace gpu_sim
