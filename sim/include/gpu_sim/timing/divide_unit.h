#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/stats.h"

namespace gpu_sim {

class DivideUnit : public ExecutionUnit {
public:
    explicit DivideUnit(Stats& stats) : stats_(stats) {}

    void evaluate() override;
    void commit() override;
    void reset() override;
    bool is_ready() const override;
    bool has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::DIVIDE; }

    void accept(const DispatchInput& input, uint64_t cycle);
    bool busy() const { return current_busy_; }
    uint32_t cycles_remaining() const { return current_cycles_remaining_; }
    std::optional<uint32_t> active_warp() const {
        if (!current_busy_) return std::nullopt;
        return current_pending_result_.warp_id;
    }
    const WritebackEntry* pending_entry() const {
        return current_busy_ ? &current_pending_result_ : nullptr;
    }
    const WritebackEntry* result_entry() const {
        // Matches has_result(): read next_* so same-tick produced results
        // are visible to the writeback arbiter and the post-evaluate trace.
        return next_result_buffer_.valid ? &next_result_buffer_ : nullptr;
    }

private:
    static constexpr uint32_t DIVIDE_LATENCY = 32;

    Stats& stats_;
    // Phase 1 discipline: every cross-cycle field is double-buffered.
    bool current_busy_ = false;
    bool next_busy_ = false;
    uint32_t current_cycles_remaining_ = 0;
    uint32_t next_cycles_remaining_ = 0;
    WritebackEntry current_pending_result_;
    WritebackEntry next_pending_result_;
    WritebackEntry current_result_buffer_;
    WritebackEntry next_result_buffer_;
};

} // namespace gpu_sim
