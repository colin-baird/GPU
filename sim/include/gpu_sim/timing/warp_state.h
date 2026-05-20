#pragma once

#include "gpu_sim/types.h"
#include "gpu_sim/timing/instruction_buffer.h"
#include "gpu_sim/timing/reg.h"

namespace gpu_sim {

// Per-warp hardware state.
//
// Phase 2 (close-the-Reg-family-migration): `pc_` and `active_` are wrapped
// as `Reg<T>`. Fetch is the sole writer of `pc_` (and always writes
// next-cycle's PC). `active_` is written by the ECALL-retirement path
// mid-tick before fetch.evaluate() reads it (handled by the per-warp
// `deactivation_request_` Wire on `TimingModel`) and by the panic
// controller in step 3 of the panic drain (fetch does not run in the panic
// branch, so the staged write is sufficient there).
//
// Helpers `seed_next()` / `commit()` drive both per-warp Regs together.
// `TimingModel::tick()` calls these for every warp on every tick;
// fine-grained test fixtures that exercise FetchStage or WarpScheduler
// directly (bypassing tick()) call them in the same pattern they already
// use for `Scoreboard` / `BranchShadowTracker`.
//
// `WarpState` itself does not derive `RegisteredStage`: the per-warp Regs
// are simple `Reg<T>` fields and the struct remains movable so the
// existing `std::vector<WarpState>` + `warps_.data()` ownership pattern
// (used by `TimingModel` and every test fixture) is preserved unchanged.
struct WarpState {
    Reg<uint32_t> pc_;      // timing-naming-allow: driven by WarpState::seed_next()/commit() helpers below; see struct docstring above.
    Reg<bool> active_;      // timing-naming-allow: driven by WarpState::seed_next()/commit() helpers below; see struct docstring above.
    InstructionBuffer instr_buffer;

    explicit WarpState(uint32_t buffer_depth = 3) : instr_buffer(buffer_depth) {}

    void reset(uint32_t start_pc) {
        pc_.initialize(start_pc);
        active_.initialize(true);
        instr_buffer.reset();
    }

    void seed_next() {
        pc_.seed();
        active_.seed();
    }

    void commit() {
        pc_.commit();
        active_.commit();
    }
};

} // namespace gpu_sim
