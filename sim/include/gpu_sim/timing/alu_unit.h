#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/timing/branch_predictor.h"
#include "gpu_sim/timing/branch_shadow_tracker.h"
#include "gpu_sim/stats.h"
#include <optional>

namespace gpu_sim {

class WritebackArbiter;

class ALUUnit : public ExecutionUnit {
public:
    explicit ALUUnit(Stats& stats) : stats_(stats) {}

    bool current_busy() const override {
        return current_result_buffer_.valid || current_has_pending_;
    }

    // Phase 10B.1: nullptr-tolerant back-pointers. The ALU pulls opcoll's
    // committed output in evaluate() (REGISTERED opcoll->unit edge) and reads
    // the arbiter's writeback stall in commit(). Wired by TimingModel; null in
    // unit tests that exercise the ALU in isolation (then accept() is driven
    // directly and commit() never stalls).
    void set_operand_collector(class OperandCollector* opcoll) {
        opcoll_ = opcoll;
    }
    void set_writeback_arbiter(class WritebackArbiter* arbiter) {
        wb_arbiter_ = arbiter;
    }
    // Phase 10B.1: the pull model's accept() needs the current simulation
    // cycle (formerly passed by TimingModel::dispatch_to_unit). The unit reads
    // it through this pointer into TimingModel::cycle_. Null in unit tests,
    // which call accept(input, cycle) directly with an explicit cycle.
    void set_sim_cycle(const uint64_t* cycle) { sim_cycle_ = cycle; }

    // Phase 10B.0.5: the ALU has 1-cycle latency, so its execution slot
    // (has_pending_ / pending_input_) is a per-cycle latch of opcoll's
    // output — not multi-cycle carry-forward state. evaluate() consumes the
    // value written by accept() THIS tick, never a prior committed value, so
    // there is nothing to re-establish in next_*. seed_next() is therefore
    // empty (the ExecutionUnit interface still carries it for the iterative
    // units). See the classification criterion in the phase-10 plan.
    void seed_next() override {}
    void evaluate() override;
    void commit() override;
    void reset() override;
    bool current_has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::ALU; }

    // Phase 10A: branch resolution moved here from TimingModel::tick(). The
    // ALU resolves a branch in its own evaluate() — it updates the branch
    // predictor, and on misprediction publishes a REGISTERED redirect into
    // next_redirect_request_; on a correct prediction it clears the
    // branch-shadow tracker's in-flight bit. Wired by TimingModel after
    // construction (nullptr-tolerant for unit tests in isolation).
    void set_branch_tracker(BranchShadowTracker* tracker) {
        branch_tracker_ = tracker;
    }
    void set_branch_predictor(BranchPredictor* predictor) {
        branch_predictor_ = predictor;
    }

    // Phase 10A REGISTERED redirect-request signal. evaluate() writes
    // next_redirect_request_ on a mispredicted branch; commit() flips it to
    // current_redirect_request_. FetchStage and DecodeStage read
    // current_redirect_request() during their own commit() to apply the
    // flush. Kept REGISTERED as an interim staging step (see RedirectRequest
    // comment in execution_unit.h); converted to combinational backward in
    // Phase 10E.
    const RedirectRequest& current_redirect_request() const {
        return current_redirect_request_;
    }

    // Folds the override-vs-current_redirect_request_ choice into a member
    // function so the libclang AST extractor sees a statically resolvable
    // receiver. FetchStage::commit() and DecodeStage::commit() call this
    // through their wired `alu_` pointer; the caller still gates on
    // `alu_ != nullptr` for the unit-test default.
    RedirectRequest current_redirect_request_or_override(
        const std::optional<RedirectRequest>& override_request) const {
        if (override_request) return *override_request;
        return current_redirect_request_;
    }

    // Test hook: explicit override of the redirect-request signal, used by
    // tests that drive ALUUnit in isolation. Mirrors the prior opcoll hook.
    void set_redirect_request_override(bool valid, uint32_t warp_id,
                                       uint32_t target_pc) {
        RedirectRequest req;
        req.valid = valid;
        req.warp_id = warp_id;
        req.target_pc = target_pc;
        next_redirect_request_ = req;
    }

    // Phase 10A: shared misprediction check, moved from TimingModel. A branch
    // is mispredicted when the predicted next-PC differs from the resolved
    // actual next-PC. Public + static so the trace path (branch_redirect
    // event emission) can reuse it without an ALU instance.
    static bool branch_mispredicted(const DispatchInput& input);

    void accept(const DispatchInput& input, uint64_t cycle);
    bool busy() const { return current_has_pending_; }
    std::optional<uint32_t> active_warp() const {
        if (!current_has_pending_) return std::nullopt;
        return current_pending_input_.warp_id;
    }
    const DispatchInput* pending_input() const {
        return current_has_pending_ ? &current_pending_input_ : nullptr;
    }
    const WritebackEntry* result_entry() const {
        // Phase 10B.3: the result buffer is a plain double-buffered pipeline
        // register. The trace path runs after commit(), so it reads the
        // committed current_* slot.
        return current_result_buffer_.valid ? &current_result_buffer_ : nullptr;
    }

private:
    Stats& stats_;
    // Double-buffered cross-cycle state. accept() / evaluate() / consume_result()
    // write only next_*; commit() flips next_* -> current_*. External readers
    // (writeback arbiter, scheduler, panic drain, snapshot) see current_*.
    WritebackEntry current_result_buffer_;
    WritebackEntry next_result_buffer_;
    bool current_has_pending_ = false;
    bool next_has_pending_ = false;
    DispatchInput current_pending_input_;
    DispatchInput next_pending_input_;
    // Phase 10B.0.5: not double-buffered. accept() writes it and evaluate()
    // reads it within the same tick (into next_result_buffer_.issue_cycle);
    // no consumer ever reads a prior committed value, so it carries no
    // cross-cycle information for a 1-cycle ALU and needs no current_/next_
    // pair.
    uint64_t pending_cycle_ = 0;

    // Phase 10B.0.5: per-cycle scratch flag. evaluate() sets it to whether it
    // processed an instruction this cycle; commit() consumes it to relocate
    // the alu_stats.busy_cycles / .instructions increments out of evaluate()
    // (a re-evaluated stalled cycle must not double-count Stats artifacts).
    // Not a double-buffered field — evaluate() assigns it fresh each cycle.
    bool processed_this_cycle_ = false;

    // Phase 10A REGISTERED redirect-request signal. evaluate() writes
    // next_redirect_request_ on a mispredicted branch; commit() flips it to
    // current_redirect_request_. FetchStage / DecodeStage read
    // current_redirect_request() during their own commit().
    RedirectRequest current_redirect_request_{};
    RedirectRequest next_redirect_request_{};

    // Phase 10B.3: category-4 control state — deliberately NOT gated by the
    // writeback stall and NOT seed_next'd. Branch resolution is combinational
    // (the clock-enable gates state latches, not the resolution datapath), so
    // a stalled ALU re-runs evaluate() on the SAME branch each frozen cycle.
    // The branch-predictor update and the branch-tracker write are writes to
    // shared structures, not pipeline registers, so the gated commit() does
    // not protect them. branch_resolved_ records that resolution side-effects
    // have already fired for the branch currently in the resolve-stage
    // register: side-effects fire only when it is clear, firing them sets it,
    // and commit() clears it whenever the resolve-stage register advances (a
    // non-stalled commit()). "Branch still at the resolve stage" iff "ALU was
    // stalled" — the ALU is fully pipelined and the writeback stall is the
    // only thing that holds it — so no instruction tag is needed.
    bool branch_resolved_ = false;

    // Wired post-construction; nullptr-tolerant for unit tests in isolation.
    BranchShadowTracker* branch_tracker_ = nullptr;
    BranchPredictor* branch_predictor_ = nullptr;
    // Phase 10B.1/10B.3 back-pointers. nullptr-tolerant for unit tests.
    OperandCollector* opcoll_ = nullptr;
    WritebackArbiter* wb_arbiter_ = nullptr;
    const uint64_t* sim_cycle_ = nullptr;
};

} // namespace gpu_sim
