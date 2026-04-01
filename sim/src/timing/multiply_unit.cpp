#include "gpu_sim/timing/multiply_unit.h"

namespace gpu_sim {

void MultiplyUnit::accept(const DispatchInput& input, uint64_t cycle) {
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
    pipeline_.push_back(entry);
    stats_.mul_stats.instructions++;
}

void MultiplyUnit::evaluate() {
    stats_.mul_stats.busy_cycles += pipeline_.empty() ? 0 : 1;

    bool head_blocked = result_buffer_.valid && !pipeline_.empty() &&
                        pipeline_.front().cycles_remaining == 0;

    for (auto& entry : pipeline_) {
        if (head_blocked && &entry == &pipeline_.front()) {
            continue;
        }
        if (entry.cycles_remaining > 0) {
            entry.cycles_remaining--;
        }
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

std::vector<uint32_t> MultiplyUnit::active_warps() const {
    std::vector<uint32_t> warps;
    warps.reserve(pipeline_.size());
    for (const auto& entry : pipeline_) {
        warps.push_back(entry.wb.warp_id);
    }
    return warps;
}

std::vector<MultiplyUnit::PipelineSnapshot> MultiplyUnit::pipeline_snapshot() const {
    std::vector<PipelineSnapshot> snapshot;
    snapshot.reserve(pipeline_.size());
    for (const auto& entry : pipeline_) {
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
