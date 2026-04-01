#include "gpu_sim/timing/operand_collector.h"

namespace gpu_sim {

void OperandCollector::accept(const IssueOutput& issue) {
    current_instr_ = issue;
    busy_ = true;
    // 2 cycles for 3-operand (VDOT8), 1 cycle for everything else
    cycles_remaining_ = (issue.decoded.num_src_regs == 3) ? 2 : 1;
}

void OperandCollector::evaluate() {
    next_output_ = std::nullopt;

    if (!busy_) return;

    stats_.operand_collector_busy_cycles++;
    cycles_remaining_--;

    if (cycles_remaining_ == 0) {
        DispatchInput out;
        out.decoded = current_instr_.decoded;
        out.trace = current_instr_.trace;
        out.warp_id = current_instr_.warp_id;
        out.pc = current_instr_.pc;
        out.prediction = current_instr_.prediction;
        next_output_ = out;
        busy_ = false;
    }
}

void OperandCollector::commit() {
    current_output_ = next_output_;
}

void OperandCollector::reset() {
    busy_ = false;
    cycles_remaining_ = 0;
    current_output_ = std::nullopt;
    next_output_ = std::nullopt;
}

} // namespace gpu_sim
