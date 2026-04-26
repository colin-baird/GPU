#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/timing/warp_state.h"
#include "gpu_sim/functional/functional_model.h"
#include <functional>

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

    // Phase 6: replace the prior pre-evaluate set_units_drained() setter
    // (which latched live state from another stage in violation of the
    // discipline) with a wired callable. evaluate() invokes the callable
    // when it needs to query whether all execution units / opcoll / ldst
    // FIFO / writeback arbiter are drained. The callable is expected to
    // read only committed (current_*) state — the caller (TimingModel)
    // composes it from each unit's is_ready()/has_result() etc., which
    // are already REGISTERED accessors. Wired once at construction by
    // TimingModel; tests may also drive it directly to override the
    // drained query.
    void set_drained_query(std::function<bool()> drained_fn) {
        drained_query_ = std::move(drained_fn);
    }

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

    // Phase 6 wiring-based drained query. nullptr-tolerant: when unset (e.g.
    // tests that never wire it), the controller treats units as drained
    // immediately so the state machine progresses without external help.
    std::function<bool()> drained_query_;

    static constexpr uint32_t MAX_DRAIN_CYCLES = 32;
};

} // namespace gpu_sim
