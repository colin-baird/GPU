#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/fetch_stage.h"
#include "gpu_sim/timing/execution_unit.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/timing/reg.h"
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

class DecodeStage : public PipelineStage, public RegisteredStage {
public:
    DecodeStage(WarpState* warps, FetchStage& fetch);

    // Phase 10D.0: explicit double-buffering for the pending decode result.
    // seed_next() seeds the registered state (the staged slot from the
    // committed slot) at the top of the tick, making the "staged == committed
    // on entry to evaluate()" precondition explicit and sweep-order-
    // independent. While the current evaluate sweep order is preserved this is
    // a redundant copy (the prior commit() already left the slots equal), so
    // it is byte-identical. Called at the top of TimingModel::tick() alongside
    // the other stages' seed_next().
    void seed_next();
    void evaluate() override;
    void commit() override;
    void reset() override;

    // Phase 6 REGISTERED ebreak signal: TimingModel reads
    // current_ebreak_request() at the top of the next cycle (after this
    // cycle's commit() latched the staged value) to decide whether to
    // call panic_->trigger(). Replaces the prior same-cycle ebreak_detected
    // accessors (ebreak_detected/ebreak_warp/ebreak_pc).
    const EBreakRequest& current_ebreak_request() const {
        return ebreak_request_.current();
    }
    std::optional<uint32_t> current_pending_warp() const {
        if (!pending_.current().valid) return std::nullopt;
        return pending_.current().target_warp;
    }
    const BufferEntry* pending_entry() const {
        return pending_.current().valid ? &pending_.current().entry : nullptr;
    }

    // Back-pressure (REGISTERED + back-pressure direction): true when
    // decode cannot accept a new fetch output this cycle. Phase 10D.0: reads
    // committed state (pending_.current(), latched only by commit()).
    // evaluate() mutates the staged slot, so this accessor returns genuinely
    // committed state regardless of evaluate-sweep order — the prerequisite
    // for Phase 10D's back-to-front sweep reversal.
    bool current_busy() const { return pending_.current().valid; }

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
    // Drops the staged pending entry if it belongs to the redirected warp.
    void apply_redirect_invalidate(uint32_t warp_id);

    WarpState* warps_;
    FetchStage& fetch_;
    const ALUUnit* alu_ = nullptr;

    // Pending decode result. Phase 10D.0: explicitly double-buffered.
    // evaluate() consumes its prior-cycle committed value (the `if
    // (pending_.next().valid) return;` guard reads the staged slot — seeded
    // equal to the committed slot by seed_next() — to decide whether to pull
    // a new fetch output) — by the 10B.0.5 per-field criterion this is
    // genuine carry-forward, so it IS seeded. evaluate() mutates the staged
    // slot; commit() latches it and then applies the pending->buffer push to
    // the committed slot (a committed-state mutation that belongs in
    // commit()). Cross-stage accessors (current_busy / current_pending_warp /
    // pending_entry) read the committed slot, so they return committed state
    // independent of evaluate-sweep order.
    struct PendingDecode {
        BufferEntry entry;
        uint32_t target_warp;
        bool valid = false;
    };

    // Phase 6 REGISTERED ebreak side-channel: evaluate() stages the request;
    // commit() latches it; TimingModel reads current_ebreak_request() at the
    // top of the next tick. seed_next() seeds it alongside pending_, which is
    // byte-identical because evaluate() unconditionally restages the value
    // (EBreakRequest{}) at its top before any conditional write.
    Reg<EBreakRequest> ebreak_request_;

    // Pending decode register (see PendingDecode above).
    Reg<PendingDecode> pending_;

    // Phase 5 test hook field.
    std::optional<RedirectRequest> redirect_override_;
};

} // namespace gpu_sim
