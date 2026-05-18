#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/fetch_stage.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/decoder.h"
#include <optional>

namespace gpu_sim {

class ALUUnit;  // forward decl: decode reads alu.next_redirect()

// Phase 6 REGISTERED EBREAK side-channel. evaluate() writes
// next_ebreak_request_ when it sees an EBREAK at decode; commit() flips
// next_ -> current_; TimingModel::tick() observes current_ebreak_request()
// at the top of the *next* tick (one cycle later) and triggers the panic
// controller. This replaces the prior plain-bool ebreak_detected_ that was
// mutated and read in the same evaluate() phase. The +1 cycle delay
// matches Option A from the orchestrator design choice.
struct EBreakRequest {
    bool valid = false;
    uint32_t warp_id = 0;
    uint32_t pc = 0;
};

class DecodeStage : public PipelineStage {
public:
    DecodeStage(WarpState* warps, FetchStage& fetch);

    // Phase 10D.0: explicit double-buffering for the pending decode result.
    // seed_next() copies current_pending_ -> next_pending_ at the top of the
    // tick, making the "next_pending_ == current_pending_ on entry to
    // evaluate()" precondition explicit and sweep-order-independent. While the
    // current evaluate sweep order is preserved this is a redundant copy (the
    // prior commit() already left next_pending_ == current_pending_), so it is
    // byte-identical. Called at the top of TimingModel::tick() alongside the
    // other stages' seed_next().
    void seed_next();
    void evaluate() override;
    void commit() override;
    void reset() override;

    // Phase 6 REGISTERED ebreak signal: TimingModel reads
    // current_ebreak_request() at the top of the next cycle (after this
    // cycle's commit() flipped next_ -> current_) to decide whether to
    // call panic_->trigger(). Replaces the prior same-cycle ebreak_detected
    // accessors (ebreak_detected/ebreak_warp/ebreak_pc).
    const EBreakRequest& current_ebreak_request() const {
        return current_ebreak_request_;
    }
    std::optional<uint32_t> current_pending_warp() const {
        if (!current_pending_.valid) return std::nullopt;
        return current_pending_.target_warp;
    }
    const BufferEntry* pending_entry() const {
        return current_pending_.valid ? &current_pending_.entry : nullptr;
    }

    // Back-pressure (REGISTERED + back-pressure direction): true when
    // decode cannot accept a new fetch output this cycle. Phase 10D.0: reads
    // committed state (current_pending_, flipped only by commit()). evaluate()
    // mutates next_pending_, so this accessor returns genuinely committed
    // state regardless of evaluate-sweep order — the prerequisite for Phase
    // 10D's back-to-front sweep reversal.
    bool current_busy() const { return current_pending_.valid; }

    // Phase 10A/10E: wire the ALU so decode.evaluate() can read its
    // COMBINATIONAL-backward next_redirect() and invalidate any matching
    // pending entry the same cycle the branch resolves. Branch resolution
    // moved from OperandCollector to ALUUnit in Phase 10A; this setter
    // replaced the former set_opcoll(...).
    void set_alu(const ALUUnit* alu) { alu_ = alu; }

    // Test hook: explicit override of the redirect signal for unit tests that
    // drive DecodeStage in isolation. When set, evaluate() uses it in place of
    // alu_->next_redirect().
    void set_redirect_request_override(bool valid, uint32_t warp_id) {
        RedirectRequest req;
        req.valid = valid;
        req.warp_id = warp_id;
        // target_pc is unused by DecodeStage's redirect handling.
        redirect_override_ = req;
    }
    void clear_redirect_request_override() {
        redirect_override_.reset();
    }

private:
    // Phase 10E: applied from the top of evaluate() when the ALU's
    // COMBINATIONAL-backward redirect (or the test override) is asserted.
    // Drops the next_pending_ entry if it belongs to the redirected warp.
    void apply_redirect_invalidate(uint32_t warp_id);

    WarpState* warps_;
    FetchStage& fetch_;
    const ALUUnit* alu_ = nullptr;

    // Phase 6 REGISTERED ebreak side-channel: evaluate() writes next_;
    // commit() flips next_ -> current_; TimingModel reads current_ at the
    // top of the next tick.
    EBreakRequest current_ebreak_request_{};
    EBreakRequest next_ebreak_request_{};

    // Pending decode result. Phase 10D.0: explicitly double-buffered.
    // evaluate() consumes its prior-cycle committed value (the `if
    // (next_pending_.valid) return;` guard reads last cycle's state to decide
    // whether to pull a new fetch output) — by the 10B.0.5 per-field
    // criterion this is genuine carry-forward, so it IS seed_next'd.
    // evaluate() mutates next_pending_; commit() flips next_pending_ ->
    // current_pending_ and then applies the redirect-invalidate and the
    // pending->buffer push to current_pending_ (committed-state mutations
    // belong in commit()). Cross-stage accessors (current_busy /
    // current_pending_warp / pending_entry) read current_pending_, so they
    // return committed state independent of evaluate-sweep order.
    struct PendingDecode {
        BufferEntry entry;
        uint32_t target_warp;
        bool valid = false;
    };
    PendingDecode current_pending_;
    PendingDecode next_pending_;

    // Phase 5 test hook field.
    std::optional<RedirectRequest> redirect_override_;
};

} // namespace gpu_sim
