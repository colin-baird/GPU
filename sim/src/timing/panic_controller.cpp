#include "gpu_sim/timing/panic_controller.h"

namespace gpu_sim {

PanicController::PanicController(uint32_t num_warps, WarpState* warps,
                                 FunctionalModel& func_model)
    : num_warps_(num_warps), warps_(warps), func_model_(func_model) {}

void PanicController::trigger(uint32_t warp_id, uint32_t pc) {
    active_ = true;
    done_ = false;
    // Decode already consumed cycle 1 by asserting panic-pending.
    // The first panic-controller tick performs the cycle-2 latch work.
    step_ = 1;
    drain_cycles_ = 0;
    panic_warp_ = warp_id;
    panic_pc_ = pc;
    panic_cause_ = 0;
}

void PanicController::evaluate() {
    if (!active_ || done_) return;

    switch (step_) {
    case 1:
        // Cycle 2: read r31 lane 0 and latch host-visible panic diagnostics.
        panic_cause_ = func_model_.register_file().read(panic_warp_, 0, 31);
        func_model_.latch_panic(panic_warp_, panic_pc_, panic_cause_);
        step_ = 2;
        break;
    case 2: {
        // Drain already-dispatched work until the machine is quiescent.
        // Phase 6: query the wired callable instead of a pre-evaluate setter.
        // When unwired (test fixtures that don't supply one), treat as
        // drained so the state machine progresses.
        drain_cycles_++;
        const bool drained = drained_query_ ? drained_query_() : true;
        if (drained || drain_cycles_ >= MAX_DRAIN_CYCLES) {
            step_ = 3;
        }
        break;
    }
    case 3:
        // Halt the full SM once the pipeline has drained.
        // Phase 2 (close-the-Reg-family-migration): active_ is Reg<bool>;
        // stage next-cycle's deactivation for every warp. In the panic flow
        // fetch.evaluate() does not run, so the deactivation_request_ Wire is
        // not needed here — TimingModel::tick()'s panic branch commits
        // warps_[w].active_ in its commit phase and the staged-false becomes
        // current_=false in time for the post-commit snapshot to observe an
        // inactive warp. Subsequent panic ticks (which re-run panic_->
        // evaluate at step 3 if not yet done) repeat the staged write
        // idempotently.
        for (uint32_t w = 0; w < num_warps_; ++w) {
            warps_[w].active_.set_next(false);
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
    panic_cause_ = 0;
    // drained_query_ is not cleared; it represents wiring, not transient
    // state, and is owned by the construction-time setup.
}

} // namespace gpu_sim
