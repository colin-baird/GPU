#include "gpu_sim/timing/operand_collector.h"

namespace gpu_sim {

void OperandCollector::accept(const IssueOutput& issue) {
    // Phase 2 discipline: writes only into next_* slots. The scheduler's
    // pre-evaluate ready_out() check (which reads current_busy_) gates calling
    // accept() so we never overwrite an in-flight instruction.
    next_busy_ = true;
    next_instr_ = issue;
    // 2 cycles for 3-operand (VDOT8), 1 cycle for everything else.
    next_cycles_remaining_ = (issue.decoded.num_src_regs == 3) ? 2 : 1;
}

void OperandCollector::evaluate() {
    next_output_ = std::nullopt;

    // Operates on next_*. After the prior tick's commit(), next_* equals
    // current_* — so for an in-flight instruction the live values describe
    // what was committed at end-of-last-cycle. If accept() ran earlier this
    // tick (only valid when current_busy_ was false, i.e. fresh arrival),
    // next_* now holds the freshly-issued payload.
    if (!next_busy_) return;

    stats_.operand_collector_busy_cycles++;
    next_cycles_remaining_--;

    if (next_cycles_remaining_ == 0) {
        DispatchInput out;
        out.decoded = next_instr_.decoded;
        out.trace = next_instr_.trace;
        out.warp_id = next_instr_.warp_id;
        out.pc = next_instr_.pc;
        out.prediction = next_instr_.prediction;
        next_output_ = out;
        next_busy_ = false;
    }
}

void OperandCollector::commit() {
    // Flip next_* -> current_* for every double-buffered field. After commit
    // next_* still holds the same value, so the next tick's evaluate() can
    // read it directly (matching the in-flight carry-forward case).
    current_busy_ = next_busy_;
    current_cycles_remaining_ = next_cycles_remaining_;
    current_instr_ = next_instr_;
    current_output_ = next_output_;
    // Phase 5: flip the redirect-request slot. Then clear next_ so that a
    // single mispredict doesn't repeat-fire on subsequent cycles. Fetch and
    // decode read current_redirect_request_ during their own commit() this
    // tick (commits run in tick-order in TimingModel::tick(); opcoll.commit()
    // runs after fetch.commit() / decode.commit() — the redirect they see is
    // last cycle's, latched by THIS commit a tick ago).
    current_redirect_request_ = next_redirect_request_;
    next_redirect_request_.valid = false;
}

void OperandCollector::resolve_branch(uint32_t warp_id, bool mispredicted,
                                      uint32_t target_pc) {
    if (mispredicted) {
        // Mispredict: defer the branch-shadow clear until fetch.commit()
        // actually applies the redirect (cycle N+1 in REGISTERED terms).
        // Clearing here would unblock the scheduler in cycle N+1 to issue
        // a shadow instruction from a buffer that has not yet been flushed,
        // committing a wrong-path instruction. The redirect-applying side
        // (FetchStage::commit) clears the tracker as part of its flush.
        next_redirect_request_.valid = true;
        next_redirect_request_.warp_id = warp_id;
        next_redirect_request_.target_pc = target_pc;
    } else if (branch_tracker_) {
        // Correct prediction: no shadow path — fetch was already speculating
        // down the right path — so clear immediately. This matches the
        // pre-Phase-5 behavior for not-taken / correctly-predicted branches.
        branch_tracker_->note_resolved_correctly(warp_id);
    }
}

void OperandCollector::reset() {
    current_busy_ = false;
    next_busy_ = false;
    current_cycles_remaining_ = 0;
    next_cycles_remaining_ = 0;
    current_output_ = std::nullopt;
    next_output_ = std::nullopt;
    current_redirect_request_ = RedirectRequest{};
    next_redirect_request_ = RedirectRequest{};
}

void OperandCollector::flush() {
    reset();
}

} // namespace gpu_sim
