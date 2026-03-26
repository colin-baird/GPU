#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/scoreboard.h"
#include "gpu_sim/stats.h"
#include <vector>
#include <optional>

namespace gpu_sim {

class WritebackArbiter : public PipelineStage {
public:
    WritebackArbiter(Scoreboard& scoreboard, Stats& stats);

    void evaluate() override;
    void commit() override;
    void reset() override;

    // Register writeback sources (execution units + MSHR fill buffer)
    void add_source(ExecutionUnit* unit);

    // Add a direct writeback entry (for cache hits / MSHR fills)
    void submit_fill(const WritebackEntry& entry);

    // The writeback that happened this cycle (for stats/trace)
    const std::optional<WritebackEntry>& committed_entry() const { return committed_; }

private:
    Scoreboard& scoreboard_;
    Stats& stats_;
    std::vector<ExecutionUnit*> sources_;

    // Direct fill buffer for cache/MSHR results
    WritebackEntry fill_buffer_;

    uint32_t rr_pointer_ = 0;
    std::optional<WritebackEntry> committed_;
    std::optional<WritebackEntry> pending_commit_;
};

} // namespace gpu_sim
