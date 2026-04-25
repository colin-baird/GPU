#include "gpu_sim/timing/multiply_unit.h"

namespace gpu_sim {

void MultiplyUnit::compute_ready() {
    // Phase 4 READY/STALL: read only committed (current_*) state. Mirrors
    // is_ready(): pipeline is "stalled" when last cycle's head reached zero
    // but the result buffer was already occupied.
    bool head_blocked = current_result_buffer_.valid &&
                        !current_pipeline_.empty() &&
                        current_pipeline_.front().cycles_remaining == 0;
    ready_out_ = !head_blocked;
}

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
    stats_.mul_stats.instructions++;
}

void MultiplyUnit::evaluate() {
    // At entry, next_pipeline_ already contains all prior committed entries
    // (seeded by commit() via current_=next_) plus any entry just pushed by
    // accept() this tick. We mutate next_pipeline_ in place: decrement
    // cycles_remaining, and pop the head into next_result_buffer_ when ready.
    stats_.mul_stats.busy_cycles += next_pipeline_.empty() ? 0 : 1;

    // Intra-stage same-cycle read: head_blocked needs the live, this-cycle
    // value of next_result_buffer_.valid. The hint from Phase 0 explicitly
    // forbids reading current_result_buffer_.valid here -- if a prior
    // mid-evaluate write had set next_result_buffer_, current_ would still
    // be stale and we'd issue an extra entry into a blocked head. (Today the
    // single set point is the pop below, which has not yet run, so current_
    // and next_ happen to be equal at this point; reading next_ keeps the
    // invariant correct under future intra-evaluate mutations as well.)
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
    // Flip next_* -> current_*. After this, next_ and current_ are equal,
    // which "seeds" next_ for the upcoming tick's accept()/evaluate() to
    // mutate without losing committed state.
    current_pipeline_ = next_pipeline_;
    current_result_buffer_ = next_result_buffer_;
}

void MultiplyUnit::reset() {
    current_pipeline_.clear();
    next_pipeline_.clear();
    current_result_buffer_.valid = false;
    next_result_buffer_.valid = false;
    ready_out_ = true;
}

bool MultiplyUnit::is_ready() const {
    // Queried by scheduler BEFORE this tick's evaluate; reads committed
    // end-of-last-cycle state. Pipeline is "stalled" if last cycle the head
    // was ready but the result buffer was occupied.
    if (current_result_buffer_.valid && !current_pipeline_.empty() &&
        current_pipeline_.front().cycles_remaining == 0) {
        return false;
    }
    return true;
}

bool MultiplyUnit::has_result() const {
    // COMBINATIONAL edge with the writeback arbiter: it queries has_result()
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
