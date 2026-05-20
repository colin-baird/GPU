#include "gpu_sim/timing/operand_collector.h"
#include "gpu_sim/timing/writeback_arbiter.h"

namespace gpu_sim {

void OperandCollector::accept(const IssueOutput& issue) {
    // Phase 2 discipline: writes only into next_* slots. The scheduler's
    // pre-evaluate current_busy() check (which reads current_busy_) gates
    // calling accept() so we never overwrite an in-flight instruction.
    busy_.set_next(true);
    instr_.set_next(issue);
    // 2 cycles for 3-operand (VDOT8), 1 cycle for everything else.
    cycles_remaining_.set_next((issue.decoded.num_src_regs == 3) ? 2 : 1);
}

void OperandCollector::evaluate() {
    // Phase 4 (reg.h migration): output_ is unconditionally re-staged here
    // every cycle, so seeding it via seed_all() is byte-identical (the seeded
    // value is immediately overwritten before any conditional read).
    output_.set_next(std::nullopt);

    // Phase 10B.2: REGISTERED scheduler->opcoll edge via the pull model. The
    // opcoll reads the scheduler's committed output at the top of evaluate()
    // and, if it is free this cycle, latches it via its own accept(). At this
    // point next_busy_ has been seeded equal to current_busy_ and accept() has
    // not yet run, so !next_busy_ is exactly "opcoll free". The scheduler's
    // opcoll_cooldown_cycles_ gate (10B.0) already paces issue so the opcoll
    // is free whenever the scheduler presents an op; the !next_busy_ guard is
    // a belt-and-braces invariant — a busy opcoll never overwrites an in-flight
    // instruction. scheduler_ is null in isolated unit tests, which drive
    // accept() directly.
    if (scheduler_ != nullptr && !busy_.next()) {
        const auto& issued = scheduler_->current_output();
        if (issued) {
            accept(*issued);
        }
    }

    // Operates on next_*. After seed_next() at the top of the tick, next_*
    // equals current_* — so for an in-flight instruction the live values
    // describe what was committed at end-of-last-cycle. If accept() ran
    // earlier this tick (only valid when current_busy_ was false, i.e. fresh
    // arrival), next_* now holds the freshly-issued payload.
    // Phase 10B.0.5: assign the per-cycle busy flag fresh;
    // operand_collector_busy_cycles is incremented at commit() gated on it.
    busy_this_cycle_.drive(busy_.next());
    if (!busy_.next()) return;

    cycles_remaining_.next_mut()--;

    if (cycles_remaining_.next() == 0) {
        DispatchInput out;
        out.decoded = instr_.next().decoded;
        out.trace = instr_.next().trace;
        out.warp_id = instr_.next().warp_id;
        out.pc = instr_.next().pc;
        out.prediction = instr_.next().prediction;
        output_.set_next(out);
        busy_.set_next(false);
    }
}

void OperandCollector::commit() {
    // Phase 10B.3: writeback-stall self-gate. On a stalled cycle the opcoll
    // holds — no next_->current_ flip — so current_output_, busy_,
    // cycles_remaining_, instr_ are all preserved and the cycle re-evaluates
    // identically next tick. The frozen current_output_ is exactly the frozen
    // input the (re-runnable) downstream units pull idempotently.
    if (wb_arbiter_ != nullptr && wb_arbiter_->next_writeback_stall()) {
        return;
    }

    // Phase 10B.0.5: operand_collector_busy_cycles relocated here from
    // evaluate() — counting at commit() (skipped on a stalled cycle) means a
    // re-evaluated cycle is not double-counted. Byte-identical while no stall
    // exists.
    if (busy_this_cycle_.value()) {
        stats_.operand_collector_busy_cycles++;
    }

    commit_all();
}

void OperandCollector::reset() {
    reset_all();
    busy_this_cycle_.reset();
}

void OperandCollector::flush() {
    reset();
}

} // namespace gpu_sim
