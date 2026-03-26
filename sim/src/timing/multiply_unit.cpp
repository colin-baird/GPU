#include "gpu_sim/timing/multiply_unit.h"

namespace gpu_sim {

void MultiplyUnit::accept(const DispatchInput& input, uint64_t cycle) {
    PipelineEntry entry;
    entry.wb.valid = true;
    entry.wb.warp_id = input.warp_id;
    entry.wb.dest_reg = input.decoded.rd;
    entry.wb.values = input.trace.results;
    entry.wb.source_unit = ExecUnit::MULTIPLY;
    entry.wb.issue_cycle = cycle;
    entry.cycles_remaining = pipeline_stages_;
    pipeline_.push_back(entry);
    stats_.mul_stats.instructions++;
}

void MultiplyUnit::evaluate() {
    stats_.mul_stats.busy_cycles += pipeline_.empty() ? 0 : 1;

    for (auto& entry : pipeline_) {
        entry.cycles_remaining--;
    }

    // Check if head of pipeline is done
    if (!pipeline_.empty() && pipeline_.front().cycles_remaining == 0) {
        if (!result_buffer_.valid) {
            result_buffer_ = pipeline_.front().wb;
            pipeline_.pop_front();
        }
        // If result buffer is occupied, pipeline stalls (head entry stays)
    }
}

void MultiplyUnit::commit() {}

void MultiplyUnit::reset() {
    pipeline_.clear();
    result_buffer_.valid = false;
}

bool MultiplyUnit::is_ready() const {
    // Can accept if pipeline isn't stalling (result buffer not blocking)
    // and pipeline has room. For simplicity: always ready unless result buffer
    // is occupied and pipeline head is ready to exit
    if (result_buffer_.valid && !pipeline_.empty() && pipeline_.front().cycles_remaining == 0) {
        return false;  // Pipeline is stalled
    }
    return true;
}

bool MultiplyUnit::has_result() const {
    return result_buffer_.valid;
}

WritebackEntry MultiplyUnit::consume_result() {
    WritebackEntry entry = result_buffer_;
    result_buffer_.valid = false;
    return entry;
}

} // namespace gpu_sim
