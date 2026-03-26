#include "gpu_sim/timing/panic_controller.h"

namespace gpu_sim {

PanicController::PanicController(uint32_t num_warps, WarpState* warps,
                                 FunctionalModel& func_model)
    : num_warps_(num_warps), warps_(warps), func_model_(func_model) {}

void PanicController::trigger(uint32_t warp_id, uint32_t pc) {
    active_ = true;
    done_ = false;
    step_ = 0;
    drain_cycles_ = 0;
    panic_warp_ = warp_id;
    panic_pc_ = pc;
}

void PanicController::evaluate() {
    if (!active_ || done_) return;

    switch (step_) {
    case 0:
        // Step 0: Assert panic-pending, inhibit scheduler (done externally)
        step_ = 1;
        break;
    case 1:
        // Step 1: Read r31 from panicking warp (functional model already has it)
        step_ = 2;
        break;
    case 2:
        // Step 2: Latch diagnostics (functional model state already captured)
        step_ = 3;
        break;
    case 3:
        // Step 3: Drain in-flight instructions
        drain_cycles_++;
        if (units_drained_ || drain_cycles_ >= MAX_DRAIN_CYCLES) {
            step_ = 4;
        }
        break;
    case 4:
        // Step 4: Mark all warps inactive, set DONE+PANIC
        for (uint32_t w = 0; w < num_warps_; ++w) {
            warps_[w].active = false;
        }
        done_ = true;
        break;
    }
}

void PanicController::reset() {
    active_ = false;
    done_ = false;
    step_ = 0;
    drain_cycles_ = 0;
    units_drained_ = false;
}

} // namespace gpu_sim
