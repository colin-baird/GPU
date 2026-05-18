#include "gpu_sim/timing/multiply_unit.h"

namespace gpu_sim {

void MultiplyUnit::accept(const DispatchInput& input, uint64_t cycle) {
    // Phase 1 discipline: writes only into next_* slots. The newly-pushed
    // entry is visible to this same tick's evaluate() through next_pipeline_.
    PipelineEntry entry;
    entry.wb.valid = true;
    entry.wb.warp_id = input.warp_id;
    entry.wb.dest_reg = input.decoded.rd;
    entry.wb.values = input.trace.results;
    entry.wb.source_unit = ExecUnit::MULTIPLY;
    entry.wb.pc = input.pc;
    entry.wb.raw_instruction = input.decoded.raw;
    entry.wb.issue_cycle = cycle;
    entry.cycles_remaining = pipeline_stages_;
    next_pipeline_.push_back(entry);
    // Phase 10B.0.5: mul_stats.instructions is incremented in commit()
    // (gated on accepted_this_cycle_), not here.
    accepted_this_cycle_ = true;
}

void MultiplyUnit::seed_next() {
    // Phase 10B.0.5: re-establish the carry-forward pipeline state in next_*
    // at the top of the tick. A deque copy — depth is kMulPipelineStages.
    next_pipeline_ = current_pipeline_;
}

void MultiplyUnit::evaluate() {
    // At entry, next_pipeline_ already contains all prior committed entries
    // (seeded by seed_next() via next_=current_) plus any entry just pushed by
    // accept() this tick. We mutate next_pipeline_ in place: decrement
    // cycles_remaining, and pop the head into next_result_buffer_ when ready.
    // Phase 10B.0.5: assign the per-cycle busy flag fresh; mul_stats.busy_cycles
    // is incremented at commit() gated on it.
    busy_this_cycle_ = !next_pipeline_.empty();

    // result_buffer_ is double-buffered with mutations routed through next_*;
    // read next_ so we observe the live value if anything earlier this tick
    // mutated it.
    bool head_blocked = next_result_buffer_.valid && !next_pipeline_.empty() &&
                        next_pipeline_.front().cycles_remaining == 0;

    for (auto& entry : next_pipeline_) {
        if (head_blocked && &entry == &next_pipeline_.front()) {
            continue;
        }
        if (entry.cycles_remaining > 0) {
            entry.cycles_remaining--;
        }
    }

    // Check if head of pipeline is done
    if (!next_pipeline_.empty() && next_pipeline_.front().cycles_remaining == 0) {
        if (!next_result_buffer_.valid) {
            next_result_buffer_ = next_pipeline_.front().wb;
            next_pipeline_.pop_front();
        }
        // If result buffer is occupied, pipeline stalls (head entry stays)
    }
}

void MultiplyUnit::commit() {
    // Phase 10B.0.5: Stats increments relocated here from evaluate()/accept().
    // Counting at commit() (skipped on a stalled cycle) means a re-evaluated
    // cycle is not double-counted. Byte-identical while no stall exists.
    // Both per-cycle flags are consumed and cleared at commit() (matching
    // the accepted_this_cycle_ pattern), so a commit() not preceded by an
    // evaluate() in the same cycle — which the canonical lifecycle never
    // does, but isolated unit tests can — never re-counts a stale flag.
    if (busy_this_cycle_) {
        stats_.mul_stats.busy_cycles++;
        busy_this_cycle_ = false;
    }
    if (accepted_this_cycle_) {
        stats_.mul_stats.instructions++;
        accepted_this_cycle_ = false;
    }

    // Flip next_* -> current_*.
    current_pipeline_ = next_pipeline_;
    current_result_buffer_ = next_result_buffer_;
}

void MultiplyUnit::reset() {
    current_pipeline_.clear();
    next_pipeline_.clear();
    current_result_buffer_.valid = false;
    next_result_buffer_.valid = false;
    busy_this_cycle_ = false;
    accepted_this_cycle_ = false;
}

bool MultiplyUnit::next_has_result() const {
    // COMBINATIONAL edge with the writeback arbiter: it queries next_has_result()
    // AFTER this unit's evaluate in the same tick, and must see the
    // freshly-popped result. Reading next_* preserves zero cycle-count delta.
    return next_result_buffer_.valid;
}

WritebackEntry MultiplyUnit::consume_result() {
    // Return the live entry; invalidate only the next_* slot. commit() at
    // tick-end latches the empty buffer into current_*.
    WritebackEntry entry = next_result_buffer_;
    next_result_buffer_.valid = false;
    return entry;
}

std::vector<uint32_t> MultiplyUnit::active_warps() const {
    std::vector<uint32_t> warps;
    warps.reserve(current_pipeline_.size());
    for (const auto& entry : current_pipeline_) {
        warps.push_back(entry.wb.warp_id);
    }
    return warps;
}

std::vector<MultiplyUnit::PipelineSnapshot> MultiplyUnit::pipeline_snapshot() const {
    std::vector<PipelineSnapshot> snapshot;
    snapshot.reserve(current_pipeline_.size());
    for (const auto& entry : current_pipeline_) {
        PipelineSnapshot item;
        item.warp_id = entry.wb.warp_id;
        item.pc = entry.wb.pc;
        item.raw_instruction = entry.wb.raw_instruction;
        item.dest_reg = entry.wb.dest_reg;
        snapshot.push_back(item);
    }
    return snapshot;
}

} // namespace gpu_sim
