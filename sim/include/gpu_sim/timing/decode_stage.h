#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/fetch_stage.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/decoder.h"
#include <optional>

namespace gpu_sim {

class OperandCollector;  // forward decl: decode reads opcoll.current_redirect_request()

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

    // Phase 3 READY/STALL discipline: compute_ready() reads only committed
    // state (pending_) and exposes ready_to_consume_fetch() for
    // FetchStage::evaluate() to query as a backpressure gate. Called by
    // TimingModel::tick() before fetch_->evaluate() so fetch sees a stable,
    // committed-state-derived signal. Phase 8: overrides PipelineStage's
    // virtual default no-op so the backward sweep can dispatch via base.
    void compute_ready() override;
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
    bool has_pending() const { return pending_.valid; }
    std::optional<uint32_t> pending_warp() const {
        if (!pending_.valid) return std::nullopt;
        return pending_.target_warp;
    }
    const BufferEntry* pending_entry() const {
        return pending_.valid ? &pending_.entry : nullptr;
    }

    // True if decode can accept a new fetch output this cycle.
    // Equivalent to !pending_.valid: decode.evaluate() consumes a new
    // fetch output only when its pending slot is empty at evaluate time
    // (which equals committed state, since compute_ready() runs first).
    // Computed by compute_ready() from committed state only.
    bool ready_to_consume_fetch() const { return ready_to_consume_fetch_; }

    // Phase 5: wire opcoll so decode.commit() can read its REGISTERED
    // current_redirect_request() and invalidate any matching pending entry.
    // Replaces the prior mid-tick decode_->invalidate_warp(...) call from
    // timing_model.cpp.
    void set_opcoll(const OperandCollector* opcoll) { opcoll_ = opcoll; }

    // Phase 5 test hook: explicit override of the redirect-request signal
    // for unit tests that drive DecodeStage in isolation.
    void set_redirect_request_override(bool valid, uint32_t warp_id) {
        redirect_override_valid_ = valid;
        redirect_override_warp_ = warp_id;
        has_redirect_override_ = true;
    }
    void clear_redirect_request_override() {
        has_redirect_override_ = false;
        redirect_override_valid_ = false;
    }

private:
    // Phase 5: applied from commit() when the upstream REGISTERED redirect
    // signal is valid. Drops the pending entry if it belongs to the
    // redirected warp.
    void apply_redirect_invalidate(uint32_t warp_id);

    WarpState* warps_;
    FetchStage& fetch_;
    const OperandCollector* opcoll_ = nullptr;

    // Phase 6 REGISTERED ebreak side-channel: evaluate() writes next_;
    // commit() flips next_ -> current_; TimingModel reads current_ at the
    // top of the next tick.
    EBreakRequest current_ebreak_request_{};
    EBreakRequest next_ebreak_request_{};
    bool ready_to_consume_fetch_ = true;

    // Pending decode result (staged for commit). pending_ is committed
    // state at the top of tick: it was last mutated by the previous
    // cycle's commit() (push attempt) or evaluate() (pull). The READY/STALL
    // contract requires compute_ready() to read pending_ before this
    // cycle's evaluate() mutates it.
    struct PendingDecode {
        BufferEntry entry;
        uint32_t target_warp;
        bool valid = false;
    };
    PendingDecode pending_;

    // Phase 5 test hook fields.
    bool has_redirect_override_ = false;
    bool redirect_override_valid_ = false;
    uint32_t redirect_override_warp_ = 0;
};

} // namespace gpu_sim
