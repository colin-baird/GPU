#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/stats.h"
#include <optional>

namespace gpu_sim {

class ALUUnit : public ExecutionUnit {
public:
    explicit ALUUnit(Stats& stats) : stats_(stats) {}

    bool ready_out() const override {
        return !current_result_buffer_.valid && !current_has_pending_;
    }
    void evaluate() override;
    void commit() override;
    void reset() override;
    bool has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::ALU; }

    void accept(const DispatchInput& input, uint64_t cycle);
    bool busy() const { return current_has_pending_; }
    std::optional<uint32_t> active_warp() const {
        if (!current_has_pending_) return std::nullopt;
        return current_pending_input_.warp_id;
    }
    const DispatchInput* pending_input() const {
        return current_has_pending_ ? &current_pending_input_ : nullptr;
    }
    const WritebackEntry* result_entry() const {
        // Matches has_result(): read next_* so same-tick popped results are
        // visible to the writeback arbiter and the post-evaluate trace path.
        return next_result_buffer_.valid ? &next_result_buffer_ : nullptr;
    }

private:
    Stats& stats_;
    // Double-buffered cross-cycle state. accept() / evaluate() / consume_result()
    // write only next_*; commit() flips next_* -> current_*. External readers
    // (writeback arbiter, scheduler, panic drain, snapshot) see current_*.
    WritebackEntry current_result_buffer_;
    WritebackEntry next_result_buffer_;
    bool current_has_pending_ = false;
    bool next_has_pending_ = false;
    DispatchInput current_pending_input_;
    DispatchInput next_pending_input_;
    uint64_t current_pending_cycle_ = 0;
    uint64_t next_pending_cycle_ = 0;
};

} // namespace gpu_sim
