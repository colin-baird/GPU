#pragma once

#include "gpu_sim/config.h"
#include "gpu_sim/stats.h"
#include "gpu_sim/functional/functional_model.h"
#include "gpu_sim/timing/branch_predictor.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/timing/scoreboard.h"
#include "gpu_sim/timing/branch_shadow_tracker.h"
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
    const std::optional<CycleTraceSnapshot>& current_cycle_snapshot() const {
        return last_cycle_snapshot_;
    }
    const std::optional<WritebackEntry>& current_committed_writeback() const {
        return wb_arbiter_->current_committed_entry();
    }

private:
    struct ActiveTraceSlice {
        bool valid = false;
        uint64_t start_cycle = 0;
        std::string key;
        std::string name;
        TraceArgs args;
    };

    bool pipeline_drained() const;
    bool execution_units_drained() const;
    void discard_writeback_results();
    bool all_warps_done() const;
    // Phase 3 (close-the-Reg-family-migration): dedicated ungated commit
    // pass for cross-stage FIFOs. Sequenced in tick() after the per-stage
    // gated commit_all() calls. By construction a cross-stage FIFO never
    // sits inside a stage's gated commit, so the producer's writeback-stall
    // gating cannot over-freeze the FIFO (which would otherwise block the
    // consumer's pop). Phases 4-5 add additional cross-stage FIFOs (memory
    // responses, DRAMSim3 CDC FIFOs) here.
    void commit_cross_stage_fifos();
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
    BranchShadowTracker branch_tracker_;
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

    // The simulator's notion of elapsed time, not a hardware register. In a
    // real GPU there would be a clocked cycle counter; this field is the
    // simulator's meta-observation of its own progress (the simulated cycle
    // number), incremented at the top of every tick and read by trace
    // timestamps, stats, and snapshot builders. Stays plain per the
    // sim-instrumentation taxonomy (Phase 6 audit classification corrected
    // from the initial Reg<uint64_t> suggestion).
    uint64_t cycle_ = 0;  // sim-instrumentation
    // Phase 7 of current_mut() elimination: evaluate→commit handoff within
    // a single tick. Driven at top-of-tick when the committed ebreak
    // request is observed; read at the commit-phase panic-flush cascade.
    // Wire<bool> encodes the transient semantics — the reset() in the
    // commit cascade replaces the prior `= false`.
    Wire<bool> pending_panic_flush_;
    // Phase 2 (close-the-Reg-family-migration): per-warp ECALL deactivation
    // request. The ECALL-retirement path (off opcoll's committed output)
    // drives a per-warp bit mid-tick BEFORE fetch.evaluate() reads warp
    // eligibility; fetch combinationally masks
    // `warps_[w].active_.current() && !deactivation_request_[w]`. On the
    // next tick the staged `active_.set_next(false)` has committed, so the
    // wire is no longer needed for the same-cycle visibility — reset at the
    // top of every tick (the `next_redirect_` convention; the wire's role
    // is one tick: producer is ECALL-retire, consumer is fetch.evaluate,
    // both within the same tick). The panic path does not drive this wire:
    // panic_->evaluate() drives `active_.set_next(false)` directly and
    // fetch.evaluate() does not run in the panic branch, so no same-cycle
    // mask is required.
    Wire<std::array<bool, MAX_WARPS>> deactivation_request_;
    // Phase 3 (close-the-Reg-family-migration): cross-stage addr-gen FIFO.
    // The FIFO is a peer of LdStUnit (producer) and CoalescingUnit
    // (consumer), not a member of either stage. LdStUnit::evaluate() stages
    // the push gated on !wb_arbiter_->next_writeback_stall() (the literal
    // simulator translation of the RTL wr_en && !stall AND-gate);
    // CoalescingUnit::evaluate() stages the pop unconditionally on its pop
    // decision. The FIFO is committed in a dedicated ungated cross-stage
    // FIFO commit pass at the bottom of tick() — NEVER enrolled in either
    // stage's gated commit_all(). On a stalled cycle: producer's pipeline
    // registers freeze (LdStUnit's commit_all() is gated), the cross-stage
    // commit pass still runs and applies any staged pop; the held push lands
    // on the resumed cycle. Borrowed pattern: WarpState[] is owned by
    // TimingModel and passed by reference. The cross-stage commit pass is
    // designed to grow in Phases 4-5 (memory cross-stage FIFOs, DRAMSim3 CDC
    // FIFOs).
    RegFifo<AddrGenFIFOEntry> addr_gen_fifo_;  // timing-naming-allow: cross-stage FIFO committed by commit_cross_stage_fifos() (dedicated ungated pass, distinct from any per-stage commit_all). The cross-stage role precludes enrollment in a stage's RegisteredStage; the dedicated pass is the lifecycle owner.
    // Construction-time flag pair; set by enable_*_trace() helpers called
    // once at config time and read throughout the tick. Effectively config
    // after construction.
    bool trace_enabled_ = false;            // config (write-once at init)
    bool trace_metadata_written_ = false;   // sim-instrumentation (per-run latch)
    std::optional<CycleTraceSnapshot> last_cycle_snapshot_;  // sim-instrumentation
    std::vector<ActiveTraceSlice> warp_trace_slices_;        // sim-instrumentation
    std::vector<ActiveTraceSlice> hardware_trace_slices_;    // sim-instrumentation

    void trace_cycle() const;
};

} // namespace gpu_sim
