#pragma once

namespace gpu_sim {

class PipelineStage {
public:
    virtual ~PipelineStage() = default;
    // Stages with READY/STALL outputs expose them as `const` accessors that
    // read only their own committed (current_*) state — there is no separate
    // backward-sweep phase. See resources/timing_discipline.md.
    virtual void evaluate() = 0;
    virtual void commit() = 0;
    virtual void reset() = 0;
};

} // namespace gpu_sim
