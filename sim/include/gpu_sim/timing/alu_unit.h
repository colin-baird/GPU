#pragma once

#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/timing/branch_predictor.h"
#include "gpu_sim/timing/branch_shadow_tracker.h"
#include "gpu_sim/timing/reg.h"
#include "gpu_sim/stats.h"
#include <optional>

namespace gpu_sim {

class WritebackArbiter;

class ALUUnit : public ExecutionUnit, public RegisteredStage {
public:
    explicit ALUUnit(Stats& stats) : stats_(stats) {
        register_state(&result_buffer_, &has_pending_, &pending_input_);
    }

    bool current_busy() const override {
        return result_buffer_.current().valid || has_pending_.current();
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
    // there is nothing to re-establish in the staged slot. seed_next() is
    // therefore empty (the ExecutionUnit interface still carries it for the
    // iterative units). Phase 3: the Reg<T>s here are committed and reset but
    // never seeded; this is faithful because evaluate() re-stages every cycle
    // (result_buffer_ unconditionally at the top; has_pending_/pending_input_
    // via accept() then cleared at line that ends `next_has_pending_=false`).
    // See the classification criterion in the phase-10 plan.
    void seed_next() override {}
    void evaluate() override;
    void commit() override;
    void reset() override;
    bool current_has_result() const override;
    WritebackEntry consume_result() override;
    ExecUnit get_type() const override { return ExecUnit::ALU; }

    // Phase 10A: branch resolution moved here from TimingModel::tick(). The
    // ALU resolves a branch in its own evaluate() — it updates the branch
    // predictor, and on misprediction asserts the combinational-backward
    // redirect (next_redirect_); on a correct prediction it clears the
    // branch-shadow tracker's in-flight bit. Wired by TimingModel after
    // construction (nullptr-tolerant for unit tests in isolation).
    void set_branch_tracker(BranchShadowTracker* tracker) {
        branch_tracker_ = tracker;
    }
    void set_branch_predictor(BranchPredictor* predictor) {
        branch_predictor_ = predictor;
    }

    // Phase 10E COMBINATIONAL-backward redirect signal. evaluate() resets
    // next_redirect_ at its top, then asserts it on a mispredicted branch
    // (gated by branch_resolved_ so it fires exactly once across a
    // multi-cycle writeback stall). FetchStage and DecodeStage read
    // next_redirect() at the top of their own evaluate() — the back-to-front
    // sweep (Phase 10D) runs the ALU before the frontend, so the read sees
    // this tick's fresh transient. The redirect is a backward control signal
    // (Principle 6): there is no current_* slot and no commit() flip. Will
    // become a Wire<T> in a later phase of the reg.h migration.
    const RedirectRequest& next_redirect() const {
        return next_redirect_;
    }

    // Test hook: explicit override of the redirect signal, used by tests that
    // drive ALUUnit in isolation. evaluate() reads this directly into the
    // transient next_redirect_ at its top (before branch resolution); a real
    // resolved branch in the same evaluate() may overwrite it.
    void set_redirect_request_override(bool valid, uint32_t warp_id,
                                       uint32_t target_pc) {
        RedirectRequest req;
        req.valid = valid;
        req.warp_id = warp_id;
        req.target_pc = target_pc;
        redirect_override_ = req;
    }
    void clear_redirect_request_override() {
        redirect_override_.reset();
    }

    // Phase 10A: shared misprediction check, moved from TimingModel. A branch
    // is mispredicted when the predicted next-PC differs from the resolved
    // actual next-PC. Public + static so the trace path (branch_redirect
    // event emission) can reuse it without an ALU instance.
    static bool branch_mispredicted(const DispatchInput& input);

    void accept(const DispatchInput& input, uint64_t cycle);
    bool busy() const { return has_pending_.current(); }
    std::optional<uint32_t> active_warp() const {
        if (!has_pending_.current()) return std::nullopt;
        return pending_input_.current().warp_id;
    }
    const DispatchInput* pending_input() const {
        return has_pending_.current() ? &pending_input_.current() : nullptr;
    }
    const WritebackEntry* result_entry() const {
        // Phase 10B.3: the result buffer is a plain double-buffered pipeline
        // register. The trace path runs after commit(), so it reads the
        // committed slot.
        return result_buffer_.current().valid ? &result_buffer_.current() : nullptr;
    }

private:
    Stats& stats_;
    // Phase 3 (reg.h migration): the ALU's per-cycle pipeline registers.
    // accept() / evaluate() / consume_result() write only the staged slot via
    // set_next() / next_mut(); commit() latches. External readers (writeback
    // arbiter, scheduler, panic drain, snapshot) see the committed slot.
    Reg<WritebackEntry> result_buffer_;
    Reg<bool> has_pending_;
    Reg<DispatchInput> pending_input_;
    // Phase 10B.0.5: not double-buffered. accept() writes it and evaluate()
    // reads it within the same tick (into next_result_buffer_.issue_cycle);
    // no consumer ever reads a prior committed value, so it carries no
    // cross-cycle information for a 1-cycle ALU and needs no Reg wrapper.
    uint64_t pending_cycle_ = 0;         // scratch (single-tick latch)

    // Phase 10B.0.5: per-cycle scratch flag. evaluate() sets it to whether it
    // processed an instruction this cycle; commit() consumes it to relocate
    // the alu_stats.busy_cycles / .instructions increments out of evaluate()
    // (a re-evaluated stalled cycle must not double-count Stats artifacts).
    // Not a double-buffered field — evaluate() assigns it fresh each cycle.
    bool processed_this_cycle_ = false;  // scratch

    // Phase 10E COMBINATIONAL-backward redirect signal. Single transient
    // slot, reset at the top of evaluate() and asserted on a mispredicted
    // branch the same cycle; there is no current_* twin and no commit()
    // flip. FetchStage / DecodeStage read it via next_redirect() at the top
    // of their own evaluate() (back-to-front sweep: ALU runs first). Will
    // become a Wire<T> in a later phase of the reg.h migration; Phase 3
    // leaves it as a plain transient.
    RedirectRequest next_redirect_{};
    // Test-only override; when set, evaluate() reads it into next_redirect_.
    std::optional<RedirectRequest> redirect_override_;

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
    bool branch_resolved_ = false;  // sim-instrumentation: deliberate non-register, see timing_discipline.md

    // Wired post-construction; nullptr-tolerant for unit tests in isolation.
    BranchShadowTracker* branch_tracker_ = nullptr;
    BranchPredictor* branch_predictor_ = nullptr;
    // Phase 10B.1/10B.3 back-pointers. nullptr-tolerant for unit tests.
    OperandCollector* opcoll_ = nullptr;
    WritebackArbiter* wb_arbiter_ = nullptr;
    const uint64_t* sim_cycle_ = nullptr;
};

} // namespace gpu_sim
