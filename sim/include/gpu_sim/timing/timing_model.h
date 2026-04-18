#pragma once

#include "gpu_sim/config.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/timing/branch_predictor.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/timing/scoreboard.h"
#include "gpu_sim/timing/fetch_stage.h"
#include "gpu_sim/timing/decode_stage.h"
#include "gpu_sim/timing/warp_scheduler.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/timing/alu_unit.h"
#include "gpu_sim/timing/multiply_unit.h"
#include "gpu_sim/timing/divide_unit.h"
#include "gpu_sim/timing/tlookup_unit.h"
#include "gpu_sim/timing/ldst_unit.h"
#include "gpu_sim/timing/writeback_arbiter.h"
#include "gpu_sim/timing/memory_interface.h"
#include "gpu_sim/timing/cache.h"
#include "gpu_sim/timing/coalescing_unit.h"
#include "gpu_sim/timing/load_gather_buffer.h"
#include "gpu_sim/timing/panic_controller.h"
#include "gpu_sim/timing/timing_trace.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gpu_sim {

class TimingModel {
public:
    TimingModel(const SimConfig& config, FunctionalModel& func_model, Stats& stats,
                TimingTraceOptions trace_options = {});
    ~TimingModel();

    // Advance one cycle. Returns true if simulation should continue.
    bool tick();

    // Run to completion or max_cycles
    void run(uint64_t max_cycles = 0);

    uint64_t cycle_count() const { return cycle_; }
    const std::optional<CycleTraceSnapshot>& last_cycle_snapshot() const {
        return last_cycle_snapshot_;
    }
    const std::optional<WritebackEntry>& last_committed_writeback() const {
        return wb_arbiter_->committed_entry();
    }

private:
    struct ActiveTraceSlice {
        bool valid = false;
        uint64_t start_cycle = 0;
        std::string key;
        std::string name;
        TraceArgs args;
    };

    void dispatch_to_unit(const DispatchInput& input);
    bool branch_mispredicted(const DispatchInput& input) const;
    bool pipeline_drained() const;
    bool execution_units_drained() const;
    void discard_writeback_results();
    bool all_warps_done() const;
    void initialize_trace_writer();
    void record_cycle_trace(bool panic_triggered);
    CycleTraceSnapshot build_cycle_snapshot() const;
    void emit_cycle_events(const CycleTraceSnapshot& snapshot, bool panic_triggered);
    void update_track_slice(ActiveTraceSlice& slice, int pid, int tid,
                            const std::string& key, const std::string& name,
                            const TraceArgs& args, uint64_t cycle);
    void flush_track_slice(ActiveTraceSlice& slice, int pid, int tid, uint64_t cycle);
    void finalize_trace();

    SimConfig config_;
    FunctionalModel& func_model_;
    Stats& stats_;
    TimingTraceOptions trace_options_;

    // Per-warp state
    std::vector<WarpState> warps_;
    std::unique_ptr<BranchPredictor> branch_predictor_;

    // Pipeline components
    Scoreboard scoreboard_;
    std::unique_ptr<FetchStage> fetch_;
    std::unique_ptr<DecodeStage> decode_;
    std::unique_ptr<WarpScheduler> scheduler_;
    std::unique_ptr<OperandCollector> opcoll_;

    // Execution units
    std::unique_ptr<ALUUnit> alu_;
    std::unique_ptr<MultiplyUnit> mul_;
    std::unique_ptr<DivideUnit> div_;
    std::unique_ptr<TLookupUnit> tlookup_;
    std::unique_ptr<LdStUnit> ldst_;

    // Memory system
    std::unique_ptr<ExternalMemoryInterface> mem_if_;
    std::unique_ptr<LoadGatherBufferFile> gather_file_;
    std::unique_ptr<L1Cache> cache_;
    std::unique_ptr<CoalescingUnit> coalescing_;

    // Writeback
    std::unique_ptr<WritebackArbiter> wb_arbiter_;

    // Panic
    std::unique_ptr<PanicController> panic_;
    std::unique_ptr<ChromeTraceWriter> structured_trace_;

    uint64_t cycle_ = 0;
    bool trace_enabled_ = false;
    bool trace_metadata_written_ = false;
    std::optional<CycleTraceSnapshot> last_cycle_snapshot_;
    std::vector<ActiveTraceSlice> warp_trace_slices_;
    std::vector<ActiveTraceSlice> hardware_trace_slices_;

    void trace_cycle() const;
};

} // namespace gpu_sim
