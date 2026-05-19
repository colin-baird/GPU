#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/branch_predictor.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/reg.h"
#include "gpu_sim/functional/memory.h"
#include "gpu_sim/stats.h"
#include <optional>

namespace gpu_sim {

class DecodeStage;          // forward decl: fetch reads decode.current_busy()
class BranchShadowTracker;  // forward decl: fetch clears in-flight on redirect apply
class ALUUnit;              // forward decl: fetch reads alu.next_redirect()

struct FetchOutput {
    uint32_t raw_instruction;
    uint32_t warp_id;
    uint32_t pc;
    BranchPrediction prediction;
};

class FetchStage : public PipelineStage, public RegisteredStage {
public:
    FetchStage(uint32_t num_warps, WarpState* warps,
               const InstructionMemory& imem, BranchPredictor& predictor, Stats& stats);

    void evaluate() override;
    void commit() override;
    void reset() override;
    std::optional<FetchOutput>& output() { return output_.next_mut(); }
    const std::optional<FetchOutput>& current_output() const { return output_.current(); }

    // Wire decode after both stages are constructed. Fetch reads
    // decode->current_busy() and decode->current_pending_warp() during
    // evaluate() as READY/STALL signals (Phase 3 discipline).
    void set_decode(const DecodeStage* decode) { decode_ = decode; }

    // Phase 10A/10E: wire the ALU so fetch.evaluate() can read its
    // COMBINATIONAL-backward next_redirect() and apply the flush the same
    // cycle the branch resolves. Branch resolution moved from OperandCollector
    // to ALUUnit in Phase 10A; this setter replaced the former set_opcoll(...).
    void set_alu(const ALUUnit* alu) { alu_ = alu; }

    // Phase 5: wire branch-shadow tracker so fetch.commit() can clear the
    // in-flight bit (write into tracker.next_) at the same moment it
    // applies a mispredict-redirect. The clear is deferred relative to
    // opcoll.resolve_branch() exactly because the scheduler must keep
    // seeing branch_in_flight==true through the cycle where the redirect
    // is applied — otherwise it could issue a shadow instruction from a
    // not-yet-flushed buffer.
    void set_branch_tracker(BranchShadowTracker* tracker) {
        branch_tracker_ = tracker;
    }

    // Test hook: explicit override of the redirect signal for unit tests that
    // drive FetchStage in isolation. When set, evaluate() uses it in place of
    // alu_->next_redirect().
    void set_redirect_request_override(bool valid, uint32_t warp_id, uint32_t target_pc) {
        RedirectRequest req;
        req.valid = valid;
        req.warp_id = warp_id;
        req.target_pc = target_pc;
        redirect_override_ = req;
    }
    void clear_redirect_request_override() {
        redirect_override_.reset();
    }

    // Test hook: explicit override of the decode-pending-warp signal for unit
    // tests that drive FetchStage without a real DecodeStage. When set,
    // takes precedence over decode_->current_pending_warp().
    void set_decode_pending_warp_override(std::optional<uint32_t> warp) {
        decode_pending_warp_override_ = warp;
        has_pending_override_ = true;
    }

    // Test hook: explicit override of decode.current_busy() for
    // unit tests. When set, takes precedence over decode_->current_busy().
    void set_decode_ready_override(bool ready) {
        decode_ready_override_ = ready;
        has_ready_override_ = true;
    }

private:
    bool query_decode_ready() const;
    std::optional<uint32_t> query_decode_pending_warp() const;
    // Phase 10E: applied from evaluate() when the ALU's COMBINATIONAL-backward
    // redirect (or the test override) is asserted. Mutates committed state
    // (warp PC, instr_buffer, the output register's committed and staged
    // slots) — the redirect is a backward control signal and the flush is its
    // same-cycle effect.
    void apply_redirect(uint32_t warp_id, uint32_t target_pc);

    uint32_t num_warps_;
    WarpState* warps_;
    const InstructionMemory& imem_;
    BranchPredictor& predictor_;
    Stats& stats_;
    const DecodeStage* decode_ = nullptr;
    const ALUUnit* alu_ = nullptr;
    BranchShadowTracker* branch_tracker_ = nullptr;

    // Round-robin warp pointer. A clock-edge register: evaluate() reads the
    // committed (pre-advance) value and stages the advanced value; commit()
    // latches it. FetchStage is not seeded (it has no seed_next() — see
    // TimingModel::tick()), which is faithful because evaluate() stages both
    // rr_pointer_ and the output slot on every code path each cycle.
    Reg<uint32_t> rr_pointer_;

    // Fetch->decode REGISTERED output slot. evaluate() stages next via
    // set_next(); commit() latches it into the committed slot that
    // current_output() exposes.
    Reg<std::optional<FetchOutput>> output_;

    // Test-only overrides; default state is "ready, no pending warp" so a
    // FetchStage exercised in isolation behaves like the previous default.
    bool has_pending_override_ = false;
    std::optional<uint32_t> decode_pending_warp_override_;
    bool has_ready_override_ = false;
    bool decode_ready_override_ = true;

    // Phase 5 test hook: redirect-request override for unit tests.
    std::optional<RedirectRequest> redirect_override_;
};

} // namespace gpu_sim
