#include "gpu_sim/timing/mshr.h"

namespace gpu_sim {

MSHRFile::MSHRFile(uint32_t num_entries)
    : num_entries_(num_entries), entries_(num_entries) {}

int MSHRFile::allocate(const MSHREntry& entry) {
    for (uint32_t i = 0; i < num_entries_; ++i) {
        if (!entries_[i].valid) {
            entries_[i] = entry;
            entries_[i].valid = true;
            return static_cast<int>(i);
        }
    }
    return -1;
}

void MSHRFile::free(uint32_t index) {
    if (index < num_entries_) {
        entries_[index].valid = false;
    }
}

bool MSHRFile::has_free() const {
    for (uint32_t i = 0; i < num_entries_; ++i) {
        if (!entries_[i].valid) return true;
    }
    return false;
}

bool MSHRFile::has_active() const {
    for (uint32_t i = 0; i < num_entries_; ++i) {
        if (entries_[i].valid) return true;
    }
    return false;
}

int MSHRFile::find_chain_tail(uint32_t line_addr) const {
    for (uint32_t i = 0; i < num_entries_; ++i) {
        const auto& e = entries_[i];
        if (!e.valid) continue;
        if (e.cache_line_addr != line_addr) continue;
        if (e.next_in_chain == MSHREntry::INVALID_MSHR) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void MSHRFile::reset() {
    for (auto& e : entries_) {
        e = MSHREntry{};
    }
}

} // namespace gpu_sim
