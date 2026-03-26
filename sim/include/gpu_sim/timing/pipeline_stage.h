#pragma once

namespace gpu_sim {

class PipelineStage {
public:
    virtual ~PipelineStage() = default;
    virtual void evaluate() = 0;
    virtual void commit() = 0;
    virtual void reset() = 0;
};

} // namespace gpu_sim
