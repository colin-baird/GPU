#include "gpu_sim/timing/multiply_unit.h"
#include "gpu_sim/timing/writeback_arbiter.h"

namespace gpu_sim {

void MultiplyUnit::accept(const DispatchInput& input, uint64_t cycle) {
    // Phase 1 discipline: writes only into next_* slots. The newly-pushed
    // entry is visible to this same tick's evaluate() through next_pipeline_.
    PipelineEntry entry;
    entry.wb.valid = true;
    entry.wb.warp_id = input.warp_id;
    entry.wb.dest_reg = input.decoded.rd;
    entry.wb.values = input.trace.results;
    entry.wb.source_unit = ExecUnit::MULTIPLY;
    entry.wb.pc = input.pc;
    entry.wb.raw_instruction = input.decoded.raw;
    entry.wb.issue_cycle = cycle;
    entry.cycles_remaining = pipeline_stages_;
    next_pipeline_.push_back(entry);
    // Phase 10B.0.5: mul_stats.instructions is incremented in commit()
    // (gated on accepted_this_cycle_), not here.
    accepted_this_cycle_ = true;
}

void MultiplyUnit::seed_next() {
    // Phase 10B.0.5: re-establish the carry-forward pipeline state in next_*
    // at the top of the tick. A deque copy — depth is kMulPipelineStages.
    next_pipeline_ = current_pipeline_;
}

void MultiplyUnit::evaluate() {
    // Phase 10B.1: REGISTERED opcoll->unit edge via the pull model. Read
    // opcoll's committed output and self-select by target_unit.
    if (opcoll_ != nullptr) {
        const auto& dispatched = opcoll_->current_output();
        if (dispatched && dispatched->decoded.target_unit == ExecUnit::MULTIPLY) {
            accept(*dispatched, sim_cycle_ != nullptr ? *sim_cycle_ : 0);
        }
    }

    // At entry, next_pipeline_ already contains all prior committed entries
    // (seeded by seed_next() via next_=current_) plus any entry just pushed by
    // accept() this tick. We mutate next_pipeline_ in place: decrement
    // cycles_remaining, and pop the head into next_result_buffer_ when ready.
    // Phase 10B.0.5: assign the per-cycle busy flag fresh; mul_stats.busy_cycles
    // is incremented at commit() gated on it.
    busy_this_cycle_ = !next_pipeline_.empty();

    // Phase 10B.3: the result buffer is a plain double-buffered pipeline
    // register — assign next_result_buffer_ fresh every cycle. There is no
    // head-blocked hold: the writeback bitmap (10B.0) guarantees the prior
    // fixed-latency result was consumed before a fresh one is produced, and a
    // load-preempted result is held by the stalled (gated) commit() — so a
    // non-stalled evaluate() can always pop a ready head unconditionally.
    next_result_buffer_ = WritebackEntry{};

    for (auto& entry : next_pipeline_) {
        if (entry.cycles_remaining > 0) {
            entry.cycles_remaining--;
        }
    }

    // Check if head of pipeline is done; pop it into the result buffer.
    if (!next_pipeline_.empty() && next_pipeline_.front().cycles_remaining == 0) {
        next_result_buffer_ = next_pipeline_.front().wb;
        next_pipeline_.pop_front();
    }
}

void MultiplyUnit::commit() {
    // Phase 10B.3: writeback-stall self-gate. On a stalled cycle this stage
    // holds — no next_->current_ flip, no Stats increment — so the pipeline
    // and result buffer re-evaluate identically next tick.
    if (wb_arbiter_ != nullptr && wb_arbiter_->next_writeback_stall()) {
        return;
    }

    // Phase 10B.0.5: Stats increments relocated here from evaluate()/accept().
    // Counting at commit() (skipped on a stalled cycle) means a re-evaluated
    // cycle is not double-counted. Byte-identical while no stall exists.
    // Both per-cycle flags are consumed and cleared at commit() (matching
    // the accepted_this_cycle_ pattern), so a commit() not preceded by an
    // evaluate() in the same cycle — which the canonical lifecycle never
    // does, but isolated unit tests can — never re-counts a stale flag.
    if (busy_this_cycle_) {
        stats_.mul_stats.busy_cycles++;
        busy_this_cycle_ = false;
    }
    if (accepted_this_cycle_) {
        stats_.mul_stats.instructions++;
        accepted_this_cycle_ = false;
    }

    // Flip next_* -> current_*.
    current_pipeline_ = next_pipeline_;
    current_result_buffer_ = next_result_buffer_;
}

void MultiplyUnit::reset() {
    current_pipeline_.clear();
    next_pipeline_.clear();
    current_result_buffer_.valid = false;
    next_result_buffer_.valid = false;
    busy_this_cycle_ = false;
    accepted_this_cycle_ = false;
}

bool MultiplyUnit::current_has_result() const {
    // Phase 10B.3: REGISTERED unit->arbiter edge — committed (current_*) state.
    return current_result_buffer_.valid;
}

WritebackEntry MultiplyUnit::consume_result() {
    // Phase 10B.3: pure read. A consumed result clears naturally next cycle
    // (the non-stalled commit() lets evaluate() overwrite next_result_buffer_);
    // a preempted result is held by the stalled (gated) commit().
    return current_result_buffer_;
}

std::vector<uint32_t> MultiplyUnit::active_warps() const {
    std::vector<uint32_t> warps;
    warps.reserve(current_pipeline_.size());
    for (const auto& entry : current_pipeline_) {
        warps.push_back(entry.wb.warp_id);
    }
    return warps;
}

std::vector<MultiplyUnit::PipelineSnapshot> MultiplyUnit::pipeline_snapshot() const {
    std::vector<PipelineSnapshot> snapshot;
    snapshot.reserve(current_pipeline_.size());
    for (const auto& entry : current_pipeline_) {
        PipelineSnapshot item;
        item.warp_id = entry.wb.warp_id;
        item.pc = entry.wb.pc;
        item.raw_instruction = entry.wb.raw_instruction;
        item.dest_reg = entry.wb.dest_reg;
        snapshot.push_back(item);
    }
    return snapshot;
}

} // namespace gpu_sim
