#include "gpu_sim/timing/operand_collector.h"

namespace gpu_sim {

void OperandCollector::accept(const IssueOutput& issue) {
    // Phase 2 discipline: writes only into next_* slots. The scheduler's
    // pre-evaluate current_busy() check (which reads current_busy_) gates
    // calling accept() so we never overwrite an in-flight instruction.
    next_busy_ = true;
    next_instr_ = issue;
    // 2 cycles for 3-operand (VDOT8), 1 cycle for everything else.
    next_cycles_remaining_ = (issue.decoded.num_src_regs == 3) ? 2 : 1;
}

void OperandCollector::seed_next() {
    // Phase 10B.0.5: re-establish the carry-forward fields in next_* at the
    // top of the tick. next_output_ is NOT seeded — evaluate() resets it.
    next_busy_ = current_busy_;
    next_cycles_remaining_ = current_cycles_remaining_;
    next_instr_ = current_instr_;
}

void OperandCollector::evaluate() {
    next_output_ = std::nullopt;

    // Operates on next_*. After seed_next() at the top of the tick, next_*
    // equals current_* — so for an in-flight instruction the live values
    // describe what was committed at end-of-last-cycle. If accept() ran
    // earlier this tick (only valid when current_busy_ was false, i.e. fresh
    // arrival), next_* now holds the freshly-issued payload.
    // Phase 10B.0.5: assign the per-cycle busy flag fresh;
    // operand_collector_busy_cycles is incremented at commit() gated on it.
    busy_this_cycle_ = next_busy_;
    if (!next_busy_) return;

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
    // Phase 10B.0.5: operand_collector_busy_cycles relocated here from
    // evaluate() — counting at commit() (skipped on a stalled cycle) means a
    // re-evaluated cycle is not double-counted. Byte-identical while no stall
    // exists.
    if (busy_this_cycle_) {
        stats_.operand_collector_busy_cycles++;
        busy_this_cycle_ = false;
    }

    // Flip next_* -> current_* for every double-buffered field. seed_next()
    // re-establishes next_* == current_* at the top of the next tick.
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
    busy_this_cycle_ = false;
}

void OperandCollector::flush() {
    reset();
}

} // namespace gpu_sim
