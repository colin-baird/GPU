#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/stats.h"
#include <deque>

namespace gpu_sim {

class MultiplyUnit : public ExecutionUnit {
public:
    struct PipelineSnapshot {
        uint32_t warp_id = 0;
        uint32_t pc = 0;
        uint32_t raw_instruction = 0;
        uint8_t dest_reg = 0;
    };

    MultiplyUnit(uint32_t pipeline_stages, Stats& stats)
        : pipeline_stages_(pipeline_stages), stats_(stats) {}

    void evaluate() override;
    void commit() override;
    void reset() override;
    bool is_ready() const override;
    bool has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::MULTIPLY; }

    void accept(const DispatchInput& input, uint64_t cycle);
    bool busy() const { return !current_pipeline_.empty(); }
    uint32_t pipeline_occupancy() const { return static_cast<uint32_t>(current_pipeline_.size()); }
    std::vector<uint32_t> active_warps() const;
    std::vector<PipelineSnapshot> pipeline_snapshot() const;
    const WritebackEntry* result_entry() const {
        // Result-buffer accessor used by snapshots (tracing) which run after
        // the unit's evaluate but also after commit. Read next_* so that
        // same-tick popped results are visible alongside has_result().
        return next_result_buffer_.valid ? &next_result_buffer_ : nullptr;
    }

private:
    struct PipelineEntry {
        WritebackEntry wb;
        uint32_t cycles_remaining;
    };

    uint32_t pipeline_stages_;
    Stats& stats_;
    // Phase 1 discipline: pipeline_ and result_buffer_ are double-buffered.
    // accept() / evaluate() / consume_result() write only next_*; commit()
    // flips next_* -> current_*. has_result() and result_entry() read next_*
    // (COMBINATIONAL same-tick edge with the writeback arbiter to preserve
    // zero cycle delta); is_ready() reads current_* (queried by scheduler
    // before unit evaluate, sees committed end-of-last-cycle state).
    std::deque<PipelineEntry> current_pipeline_;
    std::deque<PipelineEntry> next_pipeline_;
    WritebackEntry current_result_buffer_;
    WritebackEntry next_result_buffer_;
};

} // namespace gpu_sim
