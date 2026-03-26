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

    void evaluate() override;
    void commit() override;
    void reset() override;

    // Returns true if EBREAK was detected this cycle
    bool ebreak_detected() const { return ebreak_detected_; }
    uint32_t ebreak_warp() const { return ebreak_warp_id_; }
    uint32_t ebreak_pc() const { return ebreak_pc_; }

    // Invalidate any pending decode for a given warp (branch redirect)
    void invalidate_warp(uint32_t warp_id);

private:
    WarpState* warps_;
    FetchStage& fetch_;

    bool ebreak_detected_ = false;
    uint32_t ebreak_warp_id_ = 0;
    uint32_t ebreak_pc_ = 0;

    // Pending decode result (staged for commit)
    struct PendingDecode {
        BufferEntry entry;
        uint32_t target_warp;
        bool valid = false;
    };
    PendingDecode pending_;
};

} // namespace gpu_sim
