#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/stats.h"

namespace gpu_sim {

class TLookupUnit : public ExecutionUnit {
public:
    explicit TLookupUnit(Stats& stats) : stats_(stats) {}

    void evaluate() override;
    void commit() override;
    void reset() override;
    bool is_ready() const override;
    bool has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::TLOOKUP; }

    void accept(const DispatchInput& input, uint64_t cycle);

private:
    // 2 cycles per lane * 32 lanes = 64 cycles
    static constexpr uint32_t TLOOKUP_LATENCY = 64;

    Stats& stats_;
    bool busy_ = false;
    uint32_t cycles_remaining_ = 0;
    WritebackEntry pending_result_;
    WritebackEntry result_buffer_;
};

} // namespace gpu_sim
