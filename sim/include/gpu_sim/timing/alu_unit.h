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
        register_state(&result_buffer_, &has_pending_, &pending_input_,
                       &branch_resolved_);
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
    // via accept() then cleared by has_pending_.set_next(false) at the bottom
    // of the dispatch branch).
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
    // (Principle 6): there is no current_* slot and no commit() flip. Phase 7
    // (reg.h migration): the underlying storage is a Wire<RedirectRequest>;
    // this accessor is a one-line forwarder to wire_.value().
    const RedirectRequest& next_redirect() const {
        return next_redirect_.value();
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
    // Phase 7 of current_mut() elimination: single-tick latches as Wire<T>.
    // pending_cycle_ is written by accept() and read by evaluate() in the
    // same tick into result_buffer_.next_mut().issue_cycle. Wire<uint64_t>
    // encodes the intra-tick semantics.
    Wire<uint64_t> pending_cycle_;

    // Phase 7 of current_mut() elimination: per-cycle scratch flag as
    // Wire<bool>. evaluate() drives it; commit() reads .value() to relocate
    // alu_stats increments out of evaluate (so a stalled re-evaluated cycle
    // does not double-count).
    Wire<bool> processed_this_cycle_;

    // Phase 10E COMBINATIONAL-backward redirect signal. Single transient
    // slot, reset at the top of evaluate() and asserted on a mispredicted
    // branch the same cycle; there is no current_* twin and no commit()
    // flip. FetchStage / DecodeStage read it via next_redirect() at the top
    // of their own evaluate() (back-to-front sweep: ALU runs first). Phase 7
    // (reg.h migration): wrapped as Wire<RedirectRequest> — drive()/reset()
    // in evaluate(); reset() also fires from reset(). Not enrolled via
    // register_state (Wire is not a RegBase).
    Wire<RedirectRequest> next_redirect_;
    // Test-only override; when set, evaluate() reads it into next_redirect_.
    std::optional<RedirectRequest> redirect_override_;  // test-only-override

    // Phase 10B.3: control state guarding branch-resolution side-effects.
    // Branch resolution is combinational (the clock-enable gates state
    // latches, not the resolution datapath), so a stalled ALU re-runs
    // evaluate() on the SAME branch each frozen cycle. The branch-predictor
    // update and the branch-tracker write are writes to shared structures,
    // not pipeline registers, so the gated commit() does not protect them.
    // branch_resolved_ records that resolution side-effects have already
    // fired for the branch currently in the resolve-stage register:
    // side-effects fire only when it is clear, firing them sets it (via
    // set_next(true)), and commit() clears it (set_next(false)) whenever the
    // resolve-stage register advances (a non-stalled commit()). "Branch
    // still at the resolve stage" iff "ALU was stalled" — the ALU is fully
    // pipelined and the writeback stall is the only thing that holds it —
    // so no instruction tag is needed.
    //
    // Phase 1 (reg-family closeout): wrapped as Reg<bool>. The encoding
    // reads branch_resolved_.next() inside evaluate() (intra-stage self-
    // read of the staged slot, allowed per reg.h discipline) so that on a
    // stalled cycle the persisted next_=true blocks re-firing even though
    // the auto-seed semantics would otherwise overwrite it. ALUUnit's
    // seed_next() is empty (Phase 3 of the original migration — no
    // automatic next_=current_ copy at top of tick), which is what makes
    // the "next_ holds true across a stalled commit() into the next
    // evaluate" pattern faithful: a stalled commit() returns early without
    // flipping, the next tick does not reseed, so the next evaluate reads
    // the same next_=true the prior fire-cycle staged. A non-stalled
    // commit() explicitly stages set_next(false) before commit_all(), so
    // the next branch lands on a fresh resolve-stage register with both
    // slots cleared.
    Reg<bool> branch_resolved_;

    // Wired post-construction; nullptr-tolerant for unit tests in isolation.
    BranchShadowTracker* branch_tracker_ = nullptr;  // back-pointer
    BranchPredictor* branch_predictor_ = nullptr;    // back-pointer
    // Phase 10B.1/10B.3 back-pointers. nullptr-tolerant for unit tests.
    OperandCollector* opcoll_ = nullptr;     // back-pointer
    WritebackArbiter* wb_arbiter_ = nullptr; // back-pointer
    const uint64_t* sim_cycle_ = nullptr;    // back-pointer
};

} // namespace gpu_sim
