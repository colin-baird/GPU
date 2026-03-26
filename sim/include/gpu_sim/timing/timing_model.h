#pragma once

#include "gpu_sim/config.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/functional/functional_model.h"
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
#include "gpu_sim/timing/panic_controller.h"
#include <memory>

namespace gpu_sim {

class TimingModel {
public:
    TimingModel(const SimConfig& config, FunctionalModel& func_model, Stats& stats);

    // Advance one cycle. Returns true if simulation should continue.
    bool tick();

    // Run to completion or max_cycles
    void run(uint64_t max_cycles = 0);

    uint64_t cycle_count() const { return cycle_; }

private:
    void dispatch_to_unit(const DispatchInput& input);
    bool all_units_idle() const;
    bool all_warps_done() const;

    SimConfig config_;
    FunctionalModel& func_model_;
    Stats& stats_;

    // Per-warp state
    std::vector<WarpState> warps_;

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
    std::unique_ptr<L1Cache> cache_;
    std::unique_ptr<CoalescingUnit> coalescing_;

    // Writeback
    std::unique_ptr<WritebackArbiter> wb_arbiter_;

    // Panic
    std::unique_ptr<PanicController> panic_;

    uint64_t cycle_ = 0;
    bool trace_enabled_ = false;

    void trace_cycle() const;
};

} // namespace gpu_sim
