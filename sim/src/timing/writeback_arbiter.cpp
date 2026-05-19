#include "gpu_sim/timing/writeback_arbiter.h"

#include <cassert>

namespace gpu_sim {

WritebackArbiter::WritebackArbiter(Scoreboard& scoreboard, Stats& stats)
    : scoreboard_(scoreboard), stats_(stats) {}

void WritebackArbiter::add_source(ExecutionUnit* unit) {
    sources_.push_back(unit);
}

namespace {
// The variable-latency source — the LoadGatherBufferFile — registers with
// get_type()==LDST. Every other source (ALU/MULTIPLY/DIVIDE/TLOOKUP) is a
// fixed-latency unit reserved on the scheduler's writeback bitmap.
bool is_variable_latency(const ExecutionUnit* src) {
    return src->get_type() == ExecUnit::LDST;
}
}  // namespace

uint32_t WritebackArbiter::count_fixed_with_result() const {
    uint32_t count = 0;
    for (const auto* source : sources_) {
        if (!is_variable_latency(source) && source->current_has_result()) {
            count++;
        }
    }
    return count;
}

ExecutionUnit* WritebackArbiter::first_fixed_with_result() const {
    for (auto* source : sources_) {
        if (!is_variable_latency(source) && source->current_has_result()) {
            return source;
        }
    }
    return nullptr;
}

void WritebackArbiter::evaluate() {
    // Phase 10B.3: REGISTERED unit->arbiter edge + fixed-priority arbitration +
    // the combinational-backward writeback stall. Sources are read through
    // current_has_result() (committed end-of-last-cycle state); consume_result()
    // is a pure read. The arbiter is sequenced FIRST in the evaluate sweep so
    // next_writeback_stall() is readable same-cycle by every consumer.
    pending_commit_ = std::nullopt;
    // Phase 7: Wire<bool> de-asserts via reset() at the top of every evaluate.
    writeback_stall_.reset();

    // The scheduler's binding writeback bitmap (10B.0) guarantees at most one
    // fixed-latency source presents a result on any cycle. Assert it — a trip
    // means the scheduler's runtime issue->writeback offset is wrong for some
    // unit.
    assert(count_fixed_with_result() <= 1 &&
           "scheduler bitmap must prevent fixed-vs-fixed writeback contention");

    // Locate the (at most one) ready variable-latency source and the (at most
    // one, asserted above) ready fixed-latency unit.
    ExecutionUnit* variable = nullptr;
    for (auto* source : sources_) {
        if (is_variable_latency(source) && source->current_has_result()) {
            variable = source;
            break;
        }
    }
    ExecutionUnit* fixed = first_fixed_with_result();

    // Fixed-priority arbitration: variable-latency sources (loads) first,
    // fixed-latency units second. A load on the critical path of every
    // dependent instruction must retire as early as possible.
    ExecutionUnit* winner = nullptr;
    if (variable != nullptr) {
        winner = variable;
        // A fixed-latency unit with a result this cycle lost the port — it is
        // preempted. Assert the combinational-backward stall so it (and every
        // other issue/execute stage) freezes for the cycle.
        if (fixed != nullptr) {
            writeback_stall_.drive(true);
            stats_.fixed_writeback_preempted_cycles++;
        }
    } else if (fixed != nullptr) {
        winner = fixed;
    }
    // else: idle writeback cycle.

    if (winner != nullptr) {
        WritebackEntry entry = winner->consume_result();
        pending_commit_ = entry;
        if (entry.dest_reg != 0) {
            scoreboard_.clear_pending(entry.warp_id, entry.dest_reg);
        }
    }
}

void WritebackArbiter::commit() {
    committed_ = pending_commit_;
}

bool WritebackArbiter::current_busy() const {
    if (pending_commit_.has_value()) {
        return true;
    }

    for (const auto* source : sources_) {
        if (source->current_has_result()) {
            return true;
        }
    }

    return false;
}

uint32_t WritebackArbiter::ready_source_count() const {
    uint32_t count = 0;
    for (const auto* source : sources_) {
        if (source->current_has_result()) {
            count++;
        }
    }
    return count;
}

void WritebackArbiter::reset() {
    committed_ = std::nullopt;
    pending_commit_ = std::nullopt;
    // Phase 7: Wire<bool> de-asserts via reset() (default false) — equivalent
    // to the prior `= false` clear.
    writeback_stall_.reset();
}

void WritebackArbiter::flush() {
    reset();
}

} // namespace gpu_sim
