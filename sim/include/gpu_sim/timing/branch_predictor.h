#pragma once

#include "gpu_sim/trace_event.h"
#include <cstdint>

namespace gpu_sim {

struct BranchPrediction {
    bool is_control_flow = false;
    bool predicted_taken = false;
    uint32_t predicted_target = 0;
};

class BranchPredictor {
public:
    virtual ~BranchPredictor() = default;

    virtual BranchPrediction predict(uint32_t pc, uint32_t raw_instruction) const = 0;

    virtual void update(uint32_t pc, const DecodedInstruction& decoded,
                        const BranchPrediction& prediction, bool actual_taken,
                        uint32_t actual_target) = 0;
};

class StaticDirectionalBranchPredictor : public BranchPredictor {
public:
    BranchPrediction predict(uint32_t pc, uint32_t raw_instruction) const override;

    void update(uint32_t pc, const DecodedInstruction& decoded,
                const BranchPrediction& prediction, bool actual_taken,
                uint32_t actual_target) override;
};

} // namespace gpu_sim
