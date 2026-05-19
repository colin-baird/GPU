#include "gpu_sim/timing/mshr.h"

namespace gpu_sim {

MSHRFile::MSHRFile(uint32_t num_entries)
    : num_entries_(num_entries) {
    // Phase 5b: enroll the entries_ register so reset_all() / commit_all() /
    // seed_all() drive it via the RegisteredStage mixin. Then size the
    // entries vector to num_entries_ — Reg's default value-initialization
    // leaves the inner vector empty, so the constructor must explicitly
    // initialize the sizing before any allocate() / free() / current_at() /
    // next_at() can index it. Reg::initialize() sizes both slots in one step.
    register_state(&entries_);
    entries_.initialize(std::vector<MSHREntry>(num_entries));
}

int MSHRFile::allocate(const MSHREntry& entry) {
    // Scan committed state for a free slot: a slot freed this cycle (cleared
    // in the staged entries) still reads valid in the committed entries, so
    // it is not reused until the next cycle — the registered-file semantics.
    //
    // INVARIANT: at most one allocate() per tick. The scan reads committed
    // state but the write lands in staged state, so two allocate() calls in
    // one tick would both select the same slot and the second would clobber
    // the first. This holds because L1Cache processes at most one command
    // (one process_load / process_store) per evaluate(); a future
    // multi-command path would have to arbitrate allocations.
    const auto& current = entries_.current();
    auto& next = entries_.next_mut();
    for (uint32_t i = 0; i < num_entries_; ++i) {
        if (!current[i].valid) {
            next[i] = entry;
            next[i].valid = true;
            return static_cast<int>(i);
        }
    }
    return -1;
}

void MSHRFile::free(uint32_t index) {
    if (index < num_entries_) {
        entries_.next_mut()[index].valid = false;
    }
}

bool MSHRFile::has_free() const {
    const auto& current = entries_.current();
    for (uint32_t i = 0; i < num_entries_; ++i) {
        if (!current[i].valid) return true;
    }
    return false;
}

bool MSHRFile::has_active() const {
    const auto& current = entries_.current();
    for (uint32_t i = 0; i < num_entries_; ++i) {
        if (current[i].valid) return true;
    }
    return false;
}

int MSHRFile::find_chain_tail(uint32_t line_addr) const {
    const auto& current = entries_.current();
    for (uint32_t i = 0; i < num_entries_; ++i) {
        const auto& e = current[i];
        if (!e.valid) continue;
        if (e.cache_line_addr != line_addr) continue;
        if (e.next_in_chain == MSHREntry::INVALID_MSHR) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void MSHRFile::seed_next() {
    // Phase 5b: delegate to seed_all() — entries_.seed() does next_ = current_,
    // identical to the pre-Phase-5b `next_entries_ = current_entries_`.
    seed_all();
}

void MSHRFile::commit() {
    // Phase 5b: delegate to commit_all() — entries_.commit() does
    // current_ = next_, identical to the pre-Phase-5b
    // `current_entries_ = next_entries_`.
    commit_all();
}

void MSHRFile::reset() {
    // Phase 5b: reset_all() value-initializes the Reg's inner vector to empty.
    // Re-establish the num_entries_ sizing via Reg::initialize.
    reset_all();
    entries_.initialize(std::vector<MSHREntry>(num_entries_));
}

} // namespace gpu_sim
