#include "gpu_sim/timing/operand_collector.h"

namespace gpu_sim {

void OperandCollector::accept(const IssueOutput& issue) {
    // Phase 2 discipline: writes only into next_* slots. The scheduler's
    // pre-evaluate is_free() check (which reads current_busy_) gates calling
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
}

void OperandCollector::reset() {
    current_busy_ = false;
    next_busy_ = false;
    current_cycles_remaining_ = 0;
    next_cycles_remaining_ = 0;
    current_output_ = std::nullopt;
    next_output_ = std::nullopt;
}

} // namespace gpu_sim
