#include "gpu_sim/timing/cache.h"

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
        // FILL path runs first each cycle; if it won the port we must retry.
        if (!gather_file_.try_write(warp_id, lane_mask, results,
                                    LoadGatherBufferFile::GatherWriteSource::HIT)) {
            return false;
        }
        stats_.cache_hits++;
        stats_.load_hits++;
        return true;
    }

    // Cache miss
    stats_.cache_misses++;
    stats_.load_misses++;

    if (!mshrs_.has_free()) {
        stats_.mshr_stall_cycles++;
        stalled_ = true;
        stall_reason_ = CacheStallReason::MSHR_FULL;
        return false;
    }

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

    int mshr_idx = mshrs_.allocate(entry);
    mem_if_.submit_read(line_addr, static_cast<uint32_t>(mshr_idx));
    last_miss_event_.valid = true;
    last_miss_event_.warp_id = warp_id;
    last_miss_event_.line_addr = line_addr;
    last_miss_event_.is_store = false;
    last_miss_event_.pc = pc;
    last_miss_event_.raw_instruction = raw_instruction;

    return true;  // Request accepted (but result will come later)
}

bool L1Cache::process_store(uint32_t line_addr, uint32_t warp_id, uint64_t issue_cycle,
                            uint32_t pc, uint32_t raw_instruction) {
    uint32_t addr = line_addr * line_size_;
    uint32_t set = get_set(addr);
    uint32_t tag = get_tag(addr);

    if (tags_[set].valid && tags_[set].tag == tag) {
        // Store hit: write-through to write buffer (timing model tracks tags only, not data)
        stats_.cache_hits++;
        stats_.store_hits++;

        if (write_buffer_.size() >= write_buffer_depth_) {
            stats_.write_buffer_stall_cycles++;
            stalled_ = true;
            stall_reason_ = CacheStallReason::WRITE_BUFFER_FULL;
            return false;
        }
        write_buffer_.push_back(line_addr);
        return true;
    }

    // Store miss (write-allocate): need to fetch line first
    stats_.cache_misses++;
    stats_.store_misses++;

    if (!mshrs_.has_free()) {
        stats_.mshr_stall_cycles++;
        stalled_ = true;
        stall_reason_ = CacheStallReason::MSHR_FULL;
        return false;
    }

    MSHREntry entry;
    entry.cache_line_addr = line_addr;
    entry.is_store = true;
    entry.warp_id = warp_id;
    entry.dest_reg = 0;
    entry.pc = pc;
    entry.raw_instruction = raw_instruction;
    entry.issue_cycle = issue_cycle;

    int mshr_idx = mshrs_.allocate(entry);
    mem_if_.submit_read(line_addr, static_cast<uint32_t>(mshr_idx));
    last_miss_event_.valid = true;
    last_miss_event_.warp_id = warp_id;
    last_miss_event_.line_addr = line_addr;
    last_miss_event_.is_store = true;
    last_miss_event_.pc = pc;
    last_miss_event_.raw_instruction = raw_instruction;

    return true;
}

bool L1Cache::complete_fill(const MemoryResponse& resp) {
    auto& mshr = mshrs_.at(resp.mshr_id);

    // Install line in cache
    uint32_t addr = mshr.cache_line_addr * line_size_;
    uint32_t set = get_set(addr);

    if (mshr.is_store) {
        if (write_buffer_.size() >= write_buffer_depth_) {
            stats_.write_buffer_stall_cycles++;
            stalled_ = true;
            stall_reason_ = CacheStallReason::WRITE_BUFFER_FULL;
            return false;
        }

        tags_[set].valid = true;
        tags_[set].tag = get_tag(addr);
        write_buffer_.push_back(mshr.cache_line_addr);
    } else {
        // Load miss fill: deposit lane values into the owning warp's gather
        // buffer. FILL has priority over a same-cycle HIT, so a busy port here
        // would indicate two FILLs to the same buffer in one cycle, which
        // cannot happen (one outstanding load per warp).
        tags_[set].valid = true;
        tags_[set].tag = get_tag(addr);
        bool ok = gather_file_.try_write(mshr.warp_id, mshr.lane_mask, mshr.results,
                                         LoadGatherBufferFile::GatherWriteSource::FILL);
        (void)ok;
    }

    last_fill_event_.valid = true;
    last_fill_event_.warp_id = mshr.warp_id;
    last_fill_event_.line_addr = mshr.cache_line_addr;
    last_fill_event_.is_store = mshr.is_store;
    last_fill_event_.pc = mshr.pc;
    last_fill_event_.raw_instruction = mshr.raw_instruction;
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

void L1Cache::drain_write_buffer() {
    // Drain one entry per cycle if external memory can accept
    if (!write_buffer_.empty()) {
        mem_if_.submit_write(write_buffer_.front());
        write_buffer_.pop_front();
    }
}

void L1Cache::evaluate() {
    stalled_ = false;
    stall_reason_ = CacheStallReason::NONE;
    last_miss_event_.valid = false;
    last_fill_event_.valid = false;

    // FILL takes priority over HIT for the gather-buffer write port: run the
    // fill path first so a same-cycle hit-path write sees port_used_this_cycle
    // set and stalls one cycle.
    handle_responses();
}

void L1Cache::commit() {}

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
    }
    mshrs_.reset();
    write_buffer_.clear();
    stalled_ = false;
    stall_reason_ = CacheStallReason::NONE;
    pending_fill_.valid = false;
    last_miss_event_.valid = false;
    last_fill_event_.valid = false;
}

} // namespace gpu_sim
