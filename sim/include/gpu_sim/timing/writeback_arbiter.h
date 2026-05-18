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

    // Panic flush hook. Called at the commit-phase boundary when the panic
    // signal becomes active. Delegates to reset().
    void flush();

    // Register writeback sources. Phase 10B.3: sources are classified by the
    // arbiter at evaluate() time — the LoadGatherBufferFile (get_type()==LDST)
    // is the variable-latency source and wins the port over the four
    // fixed-latency units.
    void add_source(ExecutionUnit* unit);

    // The writeback that happened this cycle (for stats/trace)
    const std::optional<WritebackEntry>& current_committed_entry() const { return committed_; }
    bool current_busy() const;
    uint32_t ready_source_count() const;

    // Phase 10B.3: combinational-backward writeback stall. A single-slot
    // signal, reset at the top of every evaluate(), asserted-blocking
    // polarity. It is true for the remainder of the cycle when a
    // variable-latency load took the writeback port while a fixed-latency
    // unit also had a result — that fixed-latency unit is preempted. Every
    // issue/execute stage (the five units, OperandCollector) reads this at
    // the top of commit() and self-gates; the WarpScheduler reads it at the
    // top of evaluate() (early-return) and also gates commit(). The arbiter
    // is sequenced FIRST in the evaluate sweep so the signal is readable
    // same-cycle by every consumer. COMBINATIONAL backward, `next_` prefix
    // per the cross-stage accessor naming discipline.
    bool next_writeback_stall() const { return writeback_stall_; }

private:
    // Counts fixed-latency sources presenting a result this cycle. The
    // scheduler's binding writeback bitmap (10B.0) guarantees this is <= 1;
    // an assert in evaluate() is the live check that kIssueToWritebackOffset[]
    // is exact.
    uint32_t count_fixed_with_result() const;
    // The single fixed-latency source with a result this cycle, or nullptr.
    ExecutionUnit* first_fixed_with_result() const;

    Scoreboard& scoreboard_;
    Stats& stats_;
    std::vector<ExecutionUnit*> sources_;

    std::optional<WritebackEntry> committed_;
    std::optional<WritebackEntry> pending_commit_;
    bool writeback_stall_ = false;
};

} // namespace gpu_sim
