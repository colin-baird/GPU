#pragma once

namespace gpu_sim {

class PipelineStage {
public:
    virtual ~PipelineStage() = default;
    // Phase 8: compute_ready() is the backward-sweep phase of a tick. Stages
    // with READY/STALL outputs override this to compute their ready_out from
    // committed (current_*) state only. The default no-op is correct for
    // stages that have no cross-stage ready output (e.g., WritebackArbiter,
    // CoalescingUnit). See resources/timing_discipline.md.
    virtual void compute_ready() {}
    virtual void evaluate() = 0;
    virtual void commit() = 0;
    virtual void reset() = 0;
};

} // namespace gpu_sim
