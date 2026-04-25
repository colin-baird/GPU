#pragma once

#include "gpu_sim/timing/pipeline_stage.h"
#include "gpu_sim/timing/fetch_stage.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/decoder.h"
#include <optional>

namespace gpu_sim {

class DecodeStage : public PipelineStage {
public:
    DecodeStage(WarpState* warps, FetchStage& fetch);

    // Phase 3 READY/STALL discipline: compute_ready() reads only committed
    // state (pending_) and exposes ready_to_consume_fetch() for
    // FetchStage::evaluate() to query as a backpressure gate. Called by
    // TimingModel::tick() before fetch_->evaluate() so fetch sees a stable,
    // committed-state-derived signal.
    void compute_ready();
    void evaluate() override;
    void commit() override;
    void reset() override;

    // Returns true if EBREAK was detected this cycle
    bool ebreak_detected() const { return ebreak_detected_; }
    uint32_t ebreak_warp() const { return ebreak_warp_id_; }
    uint32_t ebreak_pc() const { return ebreak_pc_; }
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

    // Invalidate any pending decode for a given warp (branch redirect)
    void invalidate_warp(uint32_t warp_id);

private:
    WarpState* warps_;
    FetchStage& fetch_;

    bool ebreak_detected_ = false;
    uint32_t ebreak_warp_id_ = 0;
    uint32_t ebreak_pc_ = 0;
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
};

} // namespace gpu_sim
