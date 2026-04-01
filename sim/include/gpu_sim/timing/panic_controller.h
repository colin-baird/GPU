#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/functional/functional_model.h"

namespace gpu_sim {

class PanicController {
public:
    PanicController(uint32_t num_warps, WarpState* warps, FunctionalModel& func_model);

    void trigger(uint32_t warp_id, uint32_t pc);
    void evaluate();
    void reset();

    bool is_active() const { return active_; }
    bool is_done() const { return done_; }
    uint32_t step() const { return step_; }
    uint32_t panic_warp() const { return panic_warp_; }
    uint32_t panic_pc() const { return panic_pc_; }
    uint32_t panic_cause() const { return panic_cause_; }

    // Check if all execution units are drained (caller sets this)
    void set_units_drained(bool drained) { units_drained_ = drained; }

private:
    uint32_t num_warps_;
    WarpState* warps_;
    FunctionalModel& func_model_;

    bool active_ = false;
    bool done_ = false;
    uint32_t step_ = 0;
    uint32_t drain_cycles_ = 0;
    uint32_t panic_warp_ = 0;
    uint32_t panic_pc_ = 0;
    uint32_t panic_cause_ = 0;
    bool units_drained_ = false;

    static constexpr uint32_t MAX_DRAIN_CYCLES = 64;
};

} // namespace gpu_sim
