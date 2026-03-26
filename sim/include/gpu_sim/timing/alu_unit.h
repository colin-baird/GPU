#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/stats.h"
#include <optional>

namespace gpu_sim {

class ALUUnit : public ExecutionUnit {
public:
    explicit ALUUnit(Stats& stats) : stats_(stats) {}

    void evaluate() override;
    void commit() override;
    void reset() override;
    bool is_ready() const override;
    bool has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::ALU; }

    void accept(const DispatchInput& input, uint64_t cycle);

private:
    Stats& stats_;
    WritebackEntry result_buffer_;
    bool has_pending_ = false;
    DispatchInput pending_input_;
    uint64_t pending_cycle_ = 0;
};

} // namespace gpu_sim
