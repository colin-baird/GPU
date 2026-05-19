#include "gpu_sim/timing/mshr.h"

namespace gpu_sim {

MSHRFile::MSHRFile(uint32_t num_entries)
    : num_entries_(num_entries),
      current_entries_(num_entries),
      next_entries_(num_entries) {}

int MSHRFile::allocate(const MSHREntry& entry) {
    // Scan committed state for a free slot: a slot freed this cycle (cleared
    // in next_entries_) still reads valid in current_entries_, so it is not
    // reused until the next cycle — the registered-file semantics.
    //
    // INVARIANT: at most one allocate() per tick. The scan reads
    // current_entries_ but the write lands in next_entries_, so two
    // allocate() calls in one tick would both select the same slot and the
    // second would clobber the first. This holds because L1Cache processes
    // at most one command (one process_load / process_store) per evaluate();
    // a future multi-command path would have to arbitrate allocations.
    for (uint32_t i = 0; i < num_entries_; ++i) {
        if (!current_entries_[i].valid) {
            next_entries_[i] = entry;
            next_entries_[i].valid = true;
            return static_cast<int>(i);
        }
    }
    return -1;
}

void MSHRFile::free(uint32_t index) {
    if (index < num_entries_) {
        next_entries_[index].valid = false;
    }
}

bool MSHRFile::has_free() const {
    for (uint32_t i = 0; i < num_entries_; ++i) {
        if (!current_entries_[i].valid) return true;
    }
    return false;
}

bool MSHRFile::has_active() const {
    for (uint32_t i = 0; i < num_entries_; ++i) {
        if (current_entries_[i].valid) return true;
    }
    return false;
}

int MSHRFile::find_chain_tail(uint32_t line_addr) const {
    for (uint32_t i = 0; i < num_entries_; ++i) {
        const auto& e = current_entries_[i];
        if (!e.valid) continue;
        if (e.cache_line_addr != line_addr) continue;
        if (e.next_in_chain == MSHREntry::INVALID_MSHR) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void MSHRFile::seed_next() {
    next_entries_ = current_entries_;
}

void MSHRFile::commit() {
    current_entries_ = next_entries_;
}

void MSHRFile::reset() {
    for (auto& e : current_entries_) {
        e = MSHREntry{};
    }
    for (auto& e : next_entries_) {
        e = MSHREntry{};
    }
}

} // namespace gpu_sim
