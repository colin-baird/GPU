#include "gpu_sim/timing/cache.h"
#include <cassert>

namespace gpu_sim {

L1Cache::L1Cache(uint32_t cache_size, uint32_t line_size, uint32_t num_mshrs,
                 uint32_t write_buffer_depth, ExternalMemoryInterface& mem_if,
                 LoadGatherBufferFile& gather_file, Stats& stats)
    : cache_size_(cache_size), line_size_(line_size),
      num_sets_(cache_size / line_size),
      tags_(cache_size / line_size),
      mshrs_(num_mshrs),
      write_buffer_depth_(write_buffer_depth),
      mem_if_(mem_if), gather_file_(gather_file), stats_(stats) {}

uint32_t L1Cache::get_set(uint32_t addr) const {
    return (addr / line_size_) % num_sets_;
}

uint32_t L1Cache::get_tag(uint32_t addr) const {
    return addr / line_size_ / num_sets_;
}

uint32_t L1Cache::get_line_addr(uint32_t addr) const {
    return addr / line_size_;
}

bool L1Cache::process_load(uint32_t addr, uint32_t warp_id, uint32_t lane_mask,
                           const std::array<uint32_t, WARP_SIZE>& results,
                           uint64_t issue_cycle, uint32_t pc, uint32_t raw_instruction) {
    uint32_t set = get_set(addr);
    uint32_t tag = get_tag(addr);
    uint32_t line_addr = get_line_addr(addr);

    if (tags_[set].valid && tags_[set].tag == tag) {
        // Cache hit: attempt to deposit lane values into the gather buffer.
        // Per §5.3 port model (FILL > secondary drain > HIT), the HIT path
        // must lose to any in-cycle FILL or secondary-drain extraction on the
        // single cache gather-extract port. The gather-file try_write already
        // enforces the per-buffer FILL-vs-HIT edge; the cache-level flag here
        // covers the secondary-drain-vs-HIT edge when the drain and hit
        // target different gather buffers.
        if (gather_extract_port_used_) {
            return false;
        }
        if (!gather_file_.try_write(warp_id, lane_mask, results,
                                    LoadGatherBufferFile::GatherWriteSource::HIT)) {
            return false;
        }
        stats_.cache_hits++;
        stats_.load_hits++;
        gather_extract_port_used_ = true;
        return true;
    }

    // Cache miss: check for line-pin stall before accepting the request. A
    // pin stall means this request has not been accepted, so the miss/load-miss
    // counters must not be incremented here.
    if (tags_[set].valid && tags_[set].tag != tag && tags_[set].pinned) {
        stats_.line_pin_stall_cycles++;
        stalled_ = true;
        stall_reason_ = CacheStallReason::LINE_PINNED;
        last_pin_stall_event_.valid = true;
        last_pin_stall_event_.warp_id = warp_id;
        last_pin_stall_event_.requested_line_addr = line_addr;
        last_pin_stall_event_.pinned_line_addr = tags_[set].tag * num_sets_ + set;
        last_pin_stall_event_.is_store = false;
        return false;
    }

    if (!mshrs_.has_free()) {
        stats_.mshr_stall_cycles++;
        stalled_ = true;
        stall_reason_ = CacheStallReason::MSHR_FULL;
        return false;
    }

    stats_.cache_misses++;
    stats_.load_misses++;

    int tail_idx = mshrs_.find_chain_tail(line_addr);

    MSHREntry entry;
    entry.cache_line_addr = line_addr;
    entry.is_store = false;
    entry.warp_id = warp_id;
    entry.dest_reg = 0;
    entry.pc = pc;
    entry.raw_instruction = raw_instruction;
    entry.issue_cycle = issue_cycle;
    entry.results = results;
    entry.lane_mask = lane_mask;
    entry.is_secondary = (tail_idx >= 0);

    int mshr_idx = mshrs_.allocate(entry);
    assert(mshr_idx >= 0);

    last_miss_event_.valid = true;
    last_miss_event_.warp_id = warp_id;
    last_miss_event_.line_addr = line_addr;
    last_miss_event_.is_store = false;
    last_miss_event_.pc = pc;
    last_miss_event_.raw_instruction = raw_instruction;
    last_miss_event_.merged_secondary = (tail_idx >= 0);

    if (tail_idx >= 0) {
        // Secondary: inherit the primary's external fetch. Link to the tail.
        mshrs_.at(static_cast<uint32_t>(tail_idx)).next_in_chain =
            static_cast<uint32_t>(mshr_idx);
        stats_.mshr_merged_loads++;
    } else {
        mem_if_.submit_read(line_addr, static_cast<uint32_t>(mshr_idx));
    }

    return true;  // Request accepted (but result will come later)
}

bool L1Cache::process_store(uint32_t line_addr, uint32_t warp_id, uint64_t issue_cycle,
                            uint32_t pc, uint32_t raw_instruction) {
    uint32_t addr = line_addr * line_size_;
    uint32_t set = get_set(addr);
    uint32_t tag = get_tag(addr);

    if (tags_[set].valid && tags_[set].tag == tag) {
        // Store hit: write-through to write buffer (timing model tracks tags only, not data)
        if (write_buffer_.size() >= write_buffer_depth_) {
            stats_.write_buffer_stall_cycles++;
            stalled_ = true;
            stall_reason_ = CacheStallReason::WRITE_BUFFER_FULL;
            return false;
        }
        stats_.cache_hits++;
        stats_.store_hits++;
        write_buffer_.push_back(line_addr);
        return true;
    }

    // Store miss (write-allocate): pin check first. Non-acceptance path must
    // not bump the miss counters.
    if (tags_[set].valid && tags_[set].tag != tag && tags_[set].pinned) {
        stats_.line_pin_stall_cycles++;
        stalled_ = true;
        stall_reason_ = CacheStallReason::LINE_PINNED;
        last_pin_stall_event_.valid = true;
        last_pin_stall_event_.warp_id = warp_id;
        last_pin_stall_event_.requested_line_addr = line_addr;
        last_pin_stall_event_.pinned_line_addr = tags_[set].tag * num_sets_ + set;
        last_pin_stall_event_.is_store = true;
        return false;
    }

    if (!mshrs_.has_free()) {
        stats_.mshr_stall_cycles++;
        stalled_ = true;
        stall_reason_ = CacheStallReason::MSHR_FULL;
        return false;
    }

    stats_.cache_misses++;
    stats_.store_misses++;

    int tail_idx = mshrs_.find_chain_tail(line_addr);

    MSHREntry entry;
    entry.cache_line_addr = line_addr;
    entry.is_store = true;
    entry.warp_id = warp_id;
    entry.dest_reg = 0;
    entry.pc = pc;
    entry.raw_instruction = raw_instruction;
    entry.issue_cycle = issue_cycle;
    entry.is_secondary = (tail_idx >= 0);

    int mshr_idx = mshrs_.allocate(entry);
    assert(mshr_idx >= 0);

    last_miss_event_.valid = true;
    last_miss_event_.warp_id = warp_id;
    last_miss_event_.line_addr = line_addr;
    last_miss_event_.is_store = true;
    last_miss_event_.pc = pc;
    last_miss_event_.raw_instruction = raw_instruction;
    last_miss_event_.merged_secondary = (tail_idx >= 0);

    if (tail_idx >= 0) {
        mshrs_.at(static_cast<uint32_t>(tail_idx)).next_in_chain =
            static_cast<uint32_t>(mshr_idx);
        stats_.mshr_merged_stores++;
    } else {
        mem_if_.submit_read(line_addr, static_cast<uint32_t>(mshr_idx));
    }

    return true;
}

bool L1Cache::complete_fill(const MemoryResponse& resp) {
    auto& mshr = mshrs_.at(resp.mshr_id);
    assert(!mshr.is_secondary && "fill response must complete a primary MSHR");

    // Install line in cache
    uint32_t addr = mshr.cache_line_addr * line_size_;
    uint32_t set = get_set(addr);
    uint32_t new_tag = get_tag(addr);

    // If the target set is currently pinned for a different tag (another
    // line's secondaries are still draining), we cannot evict it. Defer this
    // fill until the pin is released. The fill stays in `pending_fill_` and
    // will be retried next cycle.
    if (tags_[set].valid && tags_[set].pinned && tags_[set].tag != new_tag) {
        stats_.line_pin_stall_cycles++;
        last_fill_event_.valid = true;
        last_fill_event_.warp_id = mshr.warp_id;
        last_fill_event_.line_addr = mshr.cache_line_addr;
        last_fill_event_.is_store = mshr.is_store;
        last_fill_event_.pc = mshr.pc;
        last_fill_event_.raw_instruction = mshr.raw_instruction;
        last_fill_event_.chain_length_at_fill = 0;
        last_fill_event_.deferred = true;
        return false;
    }

    // Count the full chain length at fill time for tracing.
    uint32_t chain_length = 1;
    for (uint32_t nxt = mshr.next_in_chain; nxt != MSHREntry::INVALID_MSHR;
         nxt = mshrs_.at(nxt).next_in_chain) {
        chain_length++;
    }

    if (mshr.is_store) {
        if (write_buffer_.size() >= write_buffer_depth_) {
            stats_.write_buffer_stall_cycles++;
            stalled_ = true;
            stall_reason_ = CacheStallReason::WRITE_BUFFER_FULL;
            return false;
        }

        tags_[set].valid = true;
        tags_[set].tag = get_tag(addr);
        tags_[set].pinned = (mshr.next_in_chain != MSHREntry::INVALID_MSHR);
        write_buffer_.push_back(mshr.cache_line_addr);
    } else {
        // Load miss fill: deposit lane values into the owning warp's gather
        // buffer. FILL wins its per-buffer port over HIT; the cache-level
        // extraction port is also marked used so any secondary-drain this
        // cycle defers.
        tags_[set].valid = true;
        tags_[set].tag = get_tag(addr);
        tags_[set].pinned = (mshr.next_in_chain != MSHREntry::INVALID_MSHR);
        bool ok = gather_file_.try_write(mshr.warp_id, mshr.lane_mask, mshr.results,
                                         LoadGatherBufferFile::GatherWriteSource::FILL);
        (void)ok;
        gather_extract_port_used_ = true;
    }

    last_fill_event_.valid = true;
    last_fill_event_.warp_id = mshr.warp_id;
    last_fill_event_.line_addr = mshr.cache_line_addr;
    last_fill_event_.is_store = mshr.is_store;
    last_fill_event_.pc = mshr.pc;
    last_fill_event_.raw_instruction = mshr.raw_instruction;
    last_fill_event_.chain_length_at_fill = chain_length;
    mshrs_.free(resp.mshr_id);
    return true;
}

void L1Cache::handle_responses() {
    if (pending_fill_.valid) {
        if (complete_fill(pending_fill_.response)) {
            pending_fill_.valid = false;
        }
        return;
    }

    while (mem_if_.has_response()) {
        auto resp = mem_if_.get_response();
        if (resp.is_write) {
            // Write ack from write buffer drain -- nothing to do
            continue;
        }

        pending_fill_.valid = true;
        pending_fill_.response = resp;

        if (complete_fill(resp)) {
            pending_fill_.valid = false;
        }
        return;
    }
}

void L1Cache::drain_secondary_chain_head() {
    // Find the first valid secondary MSHR that is the chain head for its line
    // (i.e., no other valid same-line MSHR points to it via `next_in_chain`).
    // The head resides on a pinned, matching tag (the primary installed it).
    const uint32_t n = mshrs_.num_entries();
    for (uint32_t i = 0; i < n; ++i) {
        auto& cand = mshrs_.at(i);
        if (!cand.valid || !cand.is_secondary) continue;

        uint32_t line_addr = cand.cache_line_addr;
        uint32_t addr = line_addr * line_size_;
        uint32_t set = get_set(addr);
        uint32_t tag = get_tag(addr);
        if (!tags_[set].valid || tags_[set].tag != tag || !tags_[set].pinned) {
            continue;
        }

        // Confirm this secondary is the chain head: no other valid same-line
        // MSHR references `i` in `next_in_chain`.
        bool is_head = true;
        for (uint32_t j = 0; j < n; ++j) {
            if (j == i) continue;
            const auto& other = mshrs_.at(j);
            if (!other.valid) continue;
            if (other.cache_line_addr != line_addr) continue;
            if (other.next_in_chain == i) {
                is_head = false;
                break;
            }
        }
        if (!is_head) continue;

        // Drain exactly one per cycle.
        if (cand.is_store) {
            if (write_buffer_.size() >= write_buffer_depth_) {
                // Normal write-buffer backpressure; leave pinned, retry next
                // cycle. No stall-counter bump.
                return;
            }
            write_buffer_.push_back(line_addr);
            stats_.secondary_drain_cycles++;
            last_drain_event_.valid = true;
            last_drain_event_.warp_id = cand.warp_id;
            last_drain_event_.line_addr = line_addr;
            last_drain_event_.is_store = true;
            last_drain_event_.pc = cand.pc;
            last_drain_event_.raw_instruction = cand.raw_instruction;
            uint32_t next = cand.next_in_chain;
            mshrs_.free(i);
            if (next == MSHREntry::INVALID_MSHR) {
                tags_[set].pinned = false;
            }
            return;
        } else {
            // Load secondary: share the cache extraction port with FILL/HIT.
            if (gather_extract_port_used_) {
                return;
            }
            bool ok = gather_file_.try_write(cand.warp_id, cand.lane_mask, cand.results,
                                             LoadGatherBufferFile::GatherWriteSource::FILL);
            if (!ok) {
                // Per-buffer port already used; defer.
                return;
            }
            gather_extract_port_used_ = true;
            stats_.secondary_drain_cycles++;
            last_drain_event_.valid = true;
            last_drain_event_.warp_id = cand.warp_id;
            last_drain_event_.line_addr = line_addr;
            last_drain_event_.is_store = false;
            last_drain_event_.pc = cand.pc;
            last_drain_event_.raw_instruction = cand.raw_instruction;
            uint32_t next = cand.next_in_chain;
            mshrs_.free(i);
            if (next == MSHREntry::INVALID_MSHR) {
                tags_[set].pinned = false;
            }
            return;
        }
    }
}

void L1Cache::drain_write_buffer() {
    // Drain one entry per cycle if external memory can accept. The bool
    // return must be respected: backends with a bounded request FIFO
    // (DRAMSim3) reject the submit when the FIFO is full, and silently
    // popping the entry would lose the write entirely (the timing model
    // tracks tags only — functional data is unaffected, but cycle counts
    // and external_memory_writes go astray and the line is never marked
    // observed by the memory model).
    if (!write_buffer_.empty()) {
        if (mem_if_.submit_write(write_buffer_.front())) {
            write_buffer_.pop_front();
        }
    }
}

void L1Cache::evaluate() {
    stalled_ = false;
    stall_reason_ = CacheStallReason::NONE;
    last_miss_event_.valid = false;
    last_fill_event_.valid = false;
    last_fill_event_.deferred = false;
    last_drain_event_.valid = false;
    last_pin_stall_event_.valid = false;

    // FILL takes priority over HIT for the gather-buffer write port: run the
    // fill path first so a same-cycle hit-path write sees port_used_this_cycle
    // set and stalls one cycle.
    handle_responses();

    // Secondary drain runs after FILL (priority: FILL > secondary > HIT). A
    // same-cycle HIT from process_load() called later will see HIT still
    // possible only if neither FILL nor secondary drain used the port.
    drain_secondary_chain_head();
}

void L1Cache::commit() {
    // Per-cycle reset: the gather-extract port is a one-extraction-per-cycle
    // resource shared by FILL, secondary drain, and HIT. It is set inside
    // evaluate() and consulted by process_load() within the same cycle; we
    // clear it in commit() so the next cycle starts with the port free.
    gather_extract_port_used_ = false;
}

bool L1Cache::is_idle() const {
    return !pending_fill_.valid && !mshrs_.has_active() && write_buffer_.empty();
}

uint32_t L1Cache::active_mshr_count() const {
    uint32_t count = 0;
    for (uint32_t i = 0; i < mshrs_.num_entries(); ++i) {
        if (mshrs_.at(i).valid) {
            count++;
        }
    }
    return count;
}

std::vector<uint32_t> L1Cache::active_mshr_warps() const {
    std::vector<uint32_t> warps;
    for (uint32_t i = 0; i < mshrs_.num_entries(); ++i) {
        if (mshrs_.at(i).valid) {
            warps.push_back(mshrs_.at(i).warp_id);
        }
    }
    return warps;
}

void L1Cache::reset() {
    for (auto& t : tags_) {
        t.valid = false;
        t.tag = 0;
        t.pinned = false;
    }
    mshrs_.reset();
    write_buffer_.clear();
    stalled_ = false;
    stall_reason_ = CacheStallReason::NONE;
    pending_fill_.valid = false;
    last_miss_event_.valid = false;
    last_fill_event_.valid = false;
    last_fill_event_.deferred = false;
    last_drain_event_.valid = false;
    last_pin_stall_event_.valid = false;
    gather_extract_port_used_ = false;
}

uint32_t L1Cache::pinned_line_count() const {
    uint32_t count = 0;
    for (const auto& t : tags_) {
        if (t.valid && t.pinned) count++;
    }
    return count;
}

uint32_t L1Cache::secondary_mshr_count() const {
    uint32_t count = 0;
    for (uint32_t i = 0; i < mshrs_.num_entries(); ++i) {
        const auto& m = mshrs_.at(i);
        if (m.valid && m.is_secondary) count++;
    }
    return count;
}

} // namespace gpu_sim
