#include "gpu_sim/timing/alu_unit.h"

namespace gpu_sim {

void ALUUnit::compute_ready() {
    // Phase 4 READY/STALL: read only committed (current_*) state. Mirrors
    // is_ready() exactly; the latter is retained for post-commit drain
    // checks (pipeline_drained / execution_units_drained / trace_cycle).
    ready_out_ = !current_result_buffer_.valid && !current_has_pending_;
}

void ALUUnit::accept(const DispatchInput& input, uint64_t cycle) {
    // Phase 1 discipline: accept() writes only the next_* slot. The unit's
    // own evaluate() runs after accept() in the same tick and treats the
    // freshly-accepted input as combinational ("in flight" through the unit
    // this cycle, latched at commit).
    next_has_pending_ = true;
    next_pending_input_ = input;
    next_pending_cycle_ = cycle;
}

void ALUUnit::evaluate() {
    // Read next_has_pending_ here: this is the input written THIS cycle by
    // accept() (combinational forward inside the unit). Cross-cycle state
    // would be read from current_*; ALU has no committed pending input
    // because it always drains its accept in a single cycle.
    if (next_has_pending_) {
        // 1-cycle latency: result is available after this evaluate. We write
        // into next_result_buffer_; commit() will publish it as
        // current_result_buffer_, which is what writeback_arbiter.has_result()
        // reads next cycle.
        next_result_buffer_.valid = true;
        next_result_buffer_.warp_id = next_pending_input_.warp_id;
        next_result_buffer_.dest_reg = next_pending_input_.decoded.rd;
        next_result_buffer_.values = next_pending_input_.trace.results;
        next_result_buffer_.source_unit = ExecUnit::ALU;
        next_result_buffer_.pc = next_pending_input_.pc;
        next_result_buffer_.raw_instruction = next_pending_input_.decoded.raw;
        next_result_buffer_.issue_cycle = next_pending_cycle_;
        next_has_pending_ = false;
        stats_.alu_stats.busy_cycles++;
        stats_.alu_stats.instructions++;
    }
}

void ALUUnit::commit() {
    // Flip next_* -> current_* for every double-buffered field.
    current_result_buffer_ = next_result_buffer_;
    current_has_pending_ = next_has_pending_;
    current_pending_input_ = next_pending_input_;
    current_pending_cycle_ = next_pending_cycle_;
}

void ALUUnit::reset() {
    current_result_buffer_.valid = false;
    next_result_buffer_.valid = false;
    current_has_pending_ = false;
    next_has_pending_ = false;
    ready_out_ = true;
}

bool ALUUnit::is_ready() const {
    // Queried by the scheduler BEFORE the unit's evaluate() in the same tick
    // to decide whether a new instruction may be dispatched. Reads committed
    // (current_*) state: end-of-last-cycle status, matching the scheduler's
    // contract on the unit_ready_fn_ callback.
    return !current_result_buffer_.valid && !current_has_pending_;
}

bool ALUUnit::has_result() const {
    // COMBINATIONAL edge with the writeback arbiter: it queries has_result()
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
