#include "gpu_sim/timing/cache.h"
#include <algorithm>
#include <cassert>

namespace gpu_sim {

L1Cache::L1Cache(uint32_t cache_size, uint32_t line_size, uint32_t num_mshrs,
                 uint32_t write_buffer_depth, uint32_t max_outstanding_writes,
                 ExternalMemoryInterface& mem_if,
                 LoadGatherBufferFile& gather_file, Stats& stats)
    : cache_size_(cache_size), line_size_(line_size),
      num_sets_(cache_size / line_size),
      current_tags_(cache_size / line_size),
      next_tags_(cache_size / line_size),
      mshrs_(num_mshrs),
      write_buffer_depth_(write_buffer_depth),
      max_outstanding_writes_(max_outstanding_writes),
      mem_if_(mem_if), gather_file_(gather_file), stats_(stats),
      current_outstanding_writes_(cache_size / line_size),
      next_outstanding_writes_(cache_size / line_size) {
    // A cap of 0 deadlocks every store (no write-through could ever be
    // enqueued). SimConfig::validate() also enforces this.
    assert(max_outstanding_writes_ >= 1);
}

uint32_t L1Cache::get_set(uint32_t addr) const {
    return (addr / line_size_) % num_sets_;
}

uint32_t L1Cache::get_tag(uint32_t addr) const {
    return addr / line_size_ / num_sets_;
}

uint32_t L1Cache::get_line_addr(uint32_t addr) const {
    return addr / line_size_;
}

bool L1Cache::is_pinned(uint32_t set) const {
    // Effective pin = chain pin OR write-ack pin. Both terms are REGISTERED:
    // current_tags_[set].pinned and the per-set outstanding-write counter.
    return current_tags_[set].pinned || current_outstanding_writes_[set] > 0;
}

bool L1Cache::outstanding_writes_at_cap() const {
    return current_outstanding_writes_total_ >= max_outstanding_writes_;
}

bool L1Cache::queue_write_through(uint32_t line_addr) {
    // Single, fallible enqueue wrapper. The write buffer has one enqueue port
    // per cycle: if it is already claimed, refuse without staging anything.
    if (next_write_buffer_port_claimed_) {
        stats_.write_buffer_port_conflict_cycles++;
        return false;
    }
    // Claim the port and stage the enqueue (applied to write_buffer_ at
    // commit). Both outstanding-write counters are bumped into next_ — a
    // queued write-through pins its set and consumes one outstanding-write
    // credit until its ack returns.
    next_write_buffer_port_claimed_ = true;
    next_write_buffer_push_ = line_addr;
    uint32_t set = get_set(line_addr * line_size_);
    ++next_outstanding_writes_[set];
    ++next_outstanding_writes_total_;
    return true;
}

bool L1Cache::process_load(uint32_t addr, uint32_t warp_id, uint32_t lane_mask,
                           const std::array<uint32_t, WARP_SIZE>& results,
                           uint64_t issue_cycle, uint32_t pc, uint32_t raw_instruction) {
    uint32_t set = get_set(addr);
    uint32_t tag = get_tag(addr);
    uint32_t line_addr = get_line_addr(addr);

    // Fill-conflict retry: a load racing a fill that installs into this set
    // this cycle is rejected and re-staged by coalescing's valid/ready
    // handshake. This precedes every tag read, MSHR access, side effect, and
    // trace event so a retried command leaves no partial state. Retrying
    // loads (not just stores) closes the orphaned-secondary hang — a load
    // miss in the fill cycle would chain onto the line's just-freed lone
    // primary. See registered-tag-array.md Step 4.
    if (fill_installed_set_ == static_cast<int32_t>(set)) {
        stats_.fill_conflict_retry_cycles++;
        return false;
    }

    if (current_tags_[set].valid && current_tags_[set].tag == tag) {
        // Cache hit: deposit lane values into the gather buffer. Phase 7:
        // arbitration is owned by LoadGatherBufferFile via a single shared
        // REGISTERED next_port_claimed flag (models spec §5.3 — one
        // line-to-gather-buffer extraction per cycle). A same-tick FILL or
        // secondary drain (which ran earlier from cache.handle_responses /
        // cache.drain_secondary_chain_head) will have claimed the port, so
        // try_write() returns false and the caller retries next cycle.
        // Tick ordering preserves FILL > secondary > HIT priority without a
        // cache-side scratch flag.
        if (!gather_file_.try_write(warp_id, lane_mask, results,
                                    LoadGatherBufferFile::GatherWriteSource::HIT)) {
            return false;
        }
        stats_.cache_hits++;
        stats_.load_hits++;
        return true;
    }

    // Cache miss: check for line-pin stall before accepting the request. A
    // pin stall means this request has not been accepted, so the miss/load-miss
    // counters must not be incremented here.
    if (current_tags_[set].valid && current_tags_[set].tag != tag && is_pinned(set)) {
        // Attribute the stall: chain pin takes precedence over the write-ack
        // pin (precedence: chain-pin > write-ack-pin).
        if (current_tags_[set].pinned) {
            stats_.line_pin_stall_cycles++;
        } else {
            stats_.write_ack_pin_stall_cycles++;
        }
        stalled_ = true;
        stall_reason_ = CacheStallReason::LINE_PINNED;
        next_last_pin_stall_event_.valid = true;
        next_last_pin_stall_event_.warp_id = warp_id;
        next_last_pin_stall_event_.requested_line_addr = line_addr;
        next_last_pin_stall_event_.pinned_line_addr = current_tags_[set].tag * num_sets_ + set;
        next_last_pin_stall_event_.is_store = false;
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

    next_last_miss_event_.valid = true;
    next_last_miss_event_.warp_id = warp_id;
    next_last_miss_event_.line_addr = line_addr;
    next_last_miss_event_.is_store = false;
    next_last_miss_event_.pc = pc;
    next_last_miss_event_.raw_instruction = raw_instruction;
    next_last_miss_event_.merged_secondary = (tail_idx >= 0);

    if (tail_idx >= 0) {
        // Secondary: inherit the primary's external fetch. Link to the tail.
        mshrs_.next_at(static_cast<uint32_t>(tail_idx)).next_in_chain =
            static_cast<uint32_t>(mshr_idx);
        stats_.mshr_merged_loads++;
    } else {
        mem_if_.set_next_read_request(line_addr, static_cast<uint32_t>(mshr_idx));
    }

    return true;  // Request accepted (but result will come later)
}

bool L1Cache::process_store(uint32_t line_addr, uint32_t warp_id, uint64_t issue_cycle,
                            uint32_t pc, uint32_t raw_instruction) {
    uint32_t addr = line_addr * line_size_;
    uint32_t set = get_set(addr);
    uint32_t tag = get_tag(addr);

    // Fill-conflict retry: a store racing a fill that installs into this set
    // this cycle is rejected and re-staged. Precedes every tag read, MSHR
    // access, side effect, and trace event. See registered-tag-array.md
    // Step 4.
    if (fill_installed_set_ == static_cast<int32_t>(set)) {
        stats_.fill_conflict_retry_cycles++;
        return false;
    }

    if (current_tags_[set].valid && current_tags_[set].tag == tag) {
        // Store hit: write-through to write buffer (timing model tracks tags only, not data).
        // Admission requires both a free write-buffer slot and an outstanding-
        // write credit; precedence for stall attribution is buffer-full > cap.
        if (write_buffer_.size() >= write_buffer_depth_ || outstanding_writes_at_cap()) {
            if (write_buffer_.size() >= write_buffer_depth_) {
                stats_.write_buffer_stall_cycles++;
            } else {
                stats_.write_throttle_stall_cycles++;
            }
            stalled_ = true;
            stall_reason_ = CacheStallReason::WRITE_BUFFER_FULL;
            return false;
        }
        if (!queue_write_through(line_addr)) {
            // Write-buffer enqueue port already claimed this cycle (by an
            // earlier FILL or secondary drain). Retry next cycle — no
            // stalled_/stall_reason_, consistent with the fill-conflict retry.
            return false;
        }
        stats_.cache_hits++;
        stats_.store_hits++;
        return true;
    }

    // Store miss (write-allocate): pin check first. Non-acceptance path must
    // not bump the miss counters.
    if (current_tags_[set].valid && current_tags_[set].tag != tag && is_pinned(set)) {
        // Attribute the stall: chain-pin > write-ack-pin.
        if (current_tags_[set].pinned) {
            stats_.line_pin_stall_cycles++;
        } else {
            stats_.write_ack_pin_stall_cycles++;
        }
        stalled_ = true;
        stall_reason_ = CacheStallReason::LINE_PINNED;
        next_last_pin_stall_event_.valid = true;
        next_last_pin_stall_event_.warp_id = warp_id;
        next_last_pin_stall_event_.requested_line_addr = line_addr;
        next_last_pin_stall_event_.pinned_line_addr = current_tags_[set].tag * num_sets_ + set;
        next_last_pin_stall_event_.is_store = true;
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

    next_last_miss_event_.valid = true;
    next_last_miss_event_.warp_id = warp_id;
    next_last_miss_event_.line_addr = line_addr;
    next_last_miss_event_.is_store = true;
    next_last_miss_event_.pc = pc;
    next_last_miss_event_.raw_instruction = raw_instruction;
    next_last_miss_event_.merged_secondary = (tail_idx >= 0);

    if (tail_idx >= 0) {
        mshrs_.next_at(static_cast<uint32_t>(tail_idx)).next_in_chain =
            static_cast<uint32_t>(mshr_idx);
        stats_.mshr_merged_stores++;
    } else {
        // Store miss is write-allocate: fetch the line first.
        mem_if_.set_next_read_request(line_addr, static_cast<uint32_t>(mshr_idx));
    }

    return true;
}

bool L1Cache::complete_fill(const MemoryResponse& resp) {
    const auto& mshr = mshrs_.current_at(resp.mshr_id);
    assert(!mshr.is_secondary && "fill response must complete a primary MSHR");

    // Install line in cache
    uint32_t addr = mshr.cache_line_addr * line_size_;
    uint32_t set = get_set(addr);
    uint32_t new_tag = get_tag(addr);

    // If the target set is currently pinned for a different tag (another
    // line's secondaries are still draining), we cannot evict it. Defer this
    // fill until the pin is released. The fill stays in `pending_fill_` and
    // will be retried next cycle.
    if (current_tags_[set].valid && is_pinned(set) && current_tags_[set].tag != new_tag) {
        // Attribute the stall: chain-pin > write-ack-pin.
        if (current_tags_[set].pinned) {
            stats_.line_pin_stall_cycles++;
        } else {
            stats_.write_ack_pin_stall_cycles++;
        }
        next_last_fill_event_.valid = true;
        next_last_fill_event_.warp_id = mshr.warp_id;
        next_last_fill_event_.line_addr = mshr.cache_line_addr;
        next_last_fill_event_.is_store = mshr.is_store;
        next_last_fill_event_.pc = mshr.pc;
        next_last_fill_event_.raw_instruction = mshr.raw_instruction;
        next_last_fill_event_.chain_length_at_fill = 0;
        next_last_fill_event_.deferred = true;
        return false;
    }

    // Count the full chain length at fill time for tracing.
    uint32_t chain_length = 1;
    for (uint32_t nxt = mshr.next_in_chain; nxt != MSHREntry::INVALID_MSHR;
         nxt = mshrs_.current_at(nxt).next_in_chain) {
        chain_length++;
    }

    if (mshr.is_store) {
        // Admission gated on write-buffer depth AND the outstanding-write
        // cap, tested before any tag install so a refusal leaves no partial
        // state. Precedence for stall attribution is buffer-full > cap.
        if (write_buffer_.size() >= write_buffer_depth_ || outstanding_writes_at_cap()) {
            if (write_buffer_.size() >= write_buffer_depth_) {
                stats_.write_buffer_stall_cycles++;
            } else {
                stats_.write_throttle_stall_cycles++;
            }
            stalled_ = true;
            stall_reason_ = CacheStallReason::WRITE_BUFFER_FULL;
            return false;
        }

        // Probe the fallible write-buffer enqueue BEFORE installing the tag,
        // so a port-busy refusal leaves no partial state — the fill simply
        // stays in pending_fill_ and is retried next cycle. (In tick order
        // FILL runs first, so the port is normally free here.)
        if (!queue_write_through(mshr.cache_line_addr)) {
            return false;
        }
        next_tags_[set].valid = true;
        next_tags_[set].tag = get_tag(addr);
        next_tags_[set].pinned = (mshr.next_in_chain != MSHREntry::INVALID_MSHR);
    } else {
        // Load miss fill: deposit lane values into the owning warp's gather
        // buffer. FILL runs first in the tick (cache.handle_responses is
        // invoked at the top of cache.evaluate()), so try_write() lands and
        // sets the buffer's `next_port_claimed`. Any same-tick HIT to the
        // same buffer (later via coalescing_->evaluate -> process_load) will
        // see the live `next_port_claimed` and bail.
        next_tags_[set].valid = true;
        next_tags_[set].tag = get_tag(addr);
        next_tags_[set].pinned = (mshr.next_in_chain != MSHREntry::INVALID_MSHR);
        bool ok = gather_file_.try_write(mshr.warp_id, mshr.lane_mask, mshr.results,
                                         LoadGatherBufferFile::GatherWriteSource::FILL);
        (void)ok;
    }

    // Successful (non-deferred) install: record the set so a command racing
    // this fill to the same set is rejected and retried (fill-conflict
    // retry). The write-buffer-full early return above does not reach here.
    fill_installed_set_ = static_cast<int32_t>(set);

    next_last_fill_event_.valid = true;
    next_last_fill_event_.warp_id = mshr.warp_id;
    next_last_fill_event_.line_addr = mshr.cache_line_addr;
    next_last_fill_event_.is_store = mshr.is_store;
    next_last_fill_event_.pc = mshr.pc;
    next_last_fill_event_.raw_instruction = mshr.raw_instruction;
    next_last_fill_event_.chain_length_at_fill = chain_length;
    mshrs_.free(resp.mshr_id);
    return true;
}

void L1Cache::handle_responses() {
    // Step 1: consume at most one write ack per cycle, UNCONDITIONALLY —
    // before the deferred-fill early return below. This is the deadlock fix:
    // a fill deferred on a write-ack pin can only make progress once the
    // blocking ack is consumed and the pin clears, so the ack drain must run
    // even while a fill is deferred. The ack returns one outstanding-write
    // credit (per-set pin + global cap). next_outstanding_writes_ has just
    // been seeded from current_ in evaluate(), so the asserts against
    // current_ and the decrements of next_ are consistent.
    if (mem_if_.current_has_write_ack()) {
        MemoryResponse ack = mem_if_.get_write_ack();
        uint32_t set = get_set(ack.line_addr * line_size_);
        assert(current_tags_[set].valid &&
               "write ack: acked line must still be resident (self-enforcing)");
        assert(current_outstanding_writes_[set] > 0 &&
               "write ack: per-set outstanding-write underflow");
        assert(current_outstanding_writes_total_ > 0 &&
               "write ack: global outstanding-write underflow");
        --next_outstanding_writes_[set];
        --next_outstanding_writes_total_;
    }

    // pending_fill_ is REGISTERED: read from current_, write next_. seed_next
    // at the top of evaluate() copies current_ -> next_ so a deferred fill
    // carries forward unless this cycle's complete_fill succeeds.
    if (current_pending_fill_.valid) {
        if (complete_fill(current_pending_fill_.response)) {
            next_pending_fill_.valid = false;
        }
        return;
    }

    // Read fills only — write completions now arrive on the separate
    // write-ack channel drained above, never on responses_.
    while (mem_if_.current_has_response()) {
        auto resp = mem_if_.get_response();

        next_pending_fill_.valid = true;
        next_pending_fill_.response = resp;

        if (complete_fill(resp)) {
            next_pending_fill_.valid = false;
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
        const auto& cand = mshrs_.current_at(i);
        if (!cand.valid || !cand.is_secondary) continue;

        uint32_t line_addr = cand.cache_line_addr;
        uint32_t addr = line_addr * line_size_;
        uint32_t set = get_set(addr);
        uint32_t tag = get_tag(addr);
        // drain_secondary_chain_head is part of the in-evaluate fill/drain
        // machinery: it must see complete_fill's same-cycle install, so it
        // reads (and writes) next_tags_. Reading current_tags_ here would
        // delay every chain drain by a cycle — a shift plan 1 does not
        // predict. (registered-tag-array.md Step 2 omits this read site;
        // Step 3 routes its pin-clear write to next_tags_.)
        if (!next_tags_[set].valid || next_tags_[set].tag != tag ||
            !next_tags_[set].pinned) {
            continue;
        }

        // Confirm this secondary is the chain head: no other valid same-line
        // MSHR references `i` in `next_in_chain`.
        bool is_head = true;
        for (uint32_t j = 0; j < n; ++j) {
            if (j == i) continue;
            const auto& other = mshrs_.current_at(j);
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
            if (write_buffer_.size() >= write_buffer_depth_ || outstanding_writes_at_cap()) {
                // Normal write-buffer / outstanding-write-cap backpressure;
                // leave pinned, retry next cycle. No stall-counter bump.
                return;
            }
            // Probe the fallible enqueue before any side effect (MSHR free,
            // pin clear). If the write-buffer port was claimed this cycle by
            // the FILL, leave the secondary pinned and retry next cycle.
            if (!queue_write_through(line_addr)) {
                return;
            }
            stats_.secondary_drain_cycles++;
            next_last_drain_event_.valid = true;
            next_last_drain_event_.warp_id = cand.warp_id;
            next_last_drain_event_.line_addr = line_addr;
            next_last_drain_event_.is_store = true;
            next_last_drain_event_.pc = cand.pc;
            next_last_drain_event_.raw_instruction = cand.raw_instruction;
            uint32_t next = cand.next_in_chain;
            mshrs_.free(i);
            if (next == MSHREntry::INVALID_MSHR) {
                next_tags_[set].pinned = false;
            }
            return;
        } else {
            // Load secondary: arbitration via the gather buffer's per-buffer
            // next_port_claimed flag. If a same-tick FILL to this same warp's
            // buffer already ran (handle_responses ran earlier in
            // cache.evaluate), try_write() returns false and we defer.
            // Secondary drains tagged as FILL-source so a HIT-vs-secondary
            // collision counts as a HIT loss in stats — consistent with the
            // pre-Phase-7 priority order (HIT loses to FILL and secondary).
            bool ok = gather_file_.try_write(cand.warp_id, cand.lane_mask, cand.results,
                                             LoadGatherBufferFile::GatherWriteSource::FILL);
            if (!ok) {
                // Per-buffer port already used; defer.
                return;
            }
            stats_.secondary_drain_cycles++;
            next_last_drain_event_.valid = true;
            next_last_drain_event_.warp_id = cand.warp_id;
            next_last_drain_event_.line_addr = line_addr;
            next_last_drain_event_.is_store = false;
            next_last_drain_event_.pc = cand.pc;
            next_last_drain_event_.raw_instruction = cand.raw_instruction;
            uint32_t next = cand.next_in_chain;
            mshrs_.free(i);
            if (next == MSHREntry::INVALID_MSHR) {
                next_tags_[set].pinned = false;
            }
            return;
        }
    }
}

void L1Cache::drain_write_buffer() {
    // Drain one entry per cycle if external memory can accept. Backends with
    // a bounded request FIFO (DRAMSim3) raise next_request_stall() when the
    // write region is full; the entry stays in write_buffer_ and is retried
    // next cycle. The timing model tracks tags only — functional data is
    // unaffected — but skipping the stall check would lose the write,
    // mis-count external_memory_writes, and leave the line never observed by
    // the memory model. write_buffer_ is a registered FIFO: the pop is staged
    // here and applied by commit(), so the entry submitted this cycle is not
    // removed from write_buffer_ until the cycle boundary.
    if (!write_buffer_.empty() && !mem_if_.next_request_stall()) {
        mem_if_.set_next_write_request(write_buffer_.front());
        next_write_buffer_pop_ = true;
    }
}

void L1Cache::evaluate() {
    // seed_next: scratch fields (stall flags, trace events) clear to false
    // so an unwritten cycle commits a quiescent observable state. The
    // pending_fill_ slot is a multi-cycle deferred-fill carrier, so it
    // copies forward; complete_fill clears next_pending_fill_.valid on
    // success, leaving current_pending_fill_ set across the cycle.
    stalled_ = false;
    stall_reason_ = CacheStallReason::NONE;
    fill_installed_set_ = -1;
    next_last_miss_event_.valid = false;
    next_last_fill_event_.valid = false;
    next_last_fill_event_.deferred = false;
    next_last_drain_event_.valid = false;
    next_last_pin_stall_event_.valid = false;
    next_pending_fill_ = current_pending_fill_;
    // REGISTERED tag array: seed next from current so an un-touched set
    // carries forward; complete_fill / drain_secondary_chain_head write
    // next_tags_, commit() flips it into current_tags_.
    next_tags_ = current_tags_;
    // REGISTERED write-ack pin counters: seed next from current. Enqueues
    // increment next_; the write-ack consumer decrements next_.
    next_outstanding_writes_ = current_outstanding_writes_;
    next_outstanding_writes_total_ = current_outstanding_writes_total_;
    // REGISTERED MSHR file: seed its next-state from committed state.
    mshrs_.seed_next();

    // Phase 7 + M3: FILL > secondary > HIT priority encoded by tick order.
    // handle_responses() (FILL) runs first; drain_secondary_chain_head()
    // (secondary) runs second; the HIT slot processes the REGISTERED
    // current_load_cmd_ / current_store_cmd_ that coalescing submitted at
    // the prior cycle. The shared gather-extract port is arbitrated by
    // LoadGatherBufferFile's REGISTERED next_port_claimed_ flag.
    handle_responses();
    drain_secondary_chain_head();

    // Phase M3 (valid/ready): process the cmd staged by coalescing at the
    // previous cycle. Cache is memoryless — the slot is unconditionally
    // cleared after the attempt. Whether the attempt succeeded is exposed
    // to coalescing combinationally via next_cmd_ready_ (which coalescing
    // reads later in the same tick to decide advance vs re-stage). On
    // failure, the cmd is dropped; coalescing's processing_/current_entry_
    // state holds the source data and re-stages next cycle.
    next_cmd_ready_ = false;
    int processed_count = 0;
    if (current_load_cmd_.valid) {
        bool ok = process_load(current_load_cmd_.addr, current_load_cmd_.warp_id,
                               current_load_cmd_.lane_mask, current_load_cmd_.results,
                               current_load_cmd_.issue_cycle, current_load_cmd_.pc,
                               current_load_cmd_.raw_instruction);
        current_load_cmd_.valid = false;
        if (ok) {
            next_cmd_ready_ = true;
            ++processed_count;
        }
    }
    if (current_store_cmd_.valid) {
        bool ok = process_store(current_store_cmd_.line_addr, current_store_cmd_.warp_id,
                                current_store_cmd_.issue_cycle, current_store_cmd_.pc,
                                current_store_cmd_.raw_instruction);
        current_store_cmd_.valid = false;
        if (ok) {
            next_cmd_ready_ = true;
            ++processed_count;
        }
    }
    // Throughput invariant: at most one cmd per cycle (coalescing only
    // stages one type at a time). When ready is asserted, exactly one cmd
    // was processed.
    assert(processed_count <= 1 &&
           "valid/ready handshake: cache processed >1 cmd in a single cycle");
}

void L1Cache::set_next_load_cmd(uint32_t addr, uint32_t warp_id, uint32_t lane_mask,
                                const std::array<uint32_t, WARP_SIZE>& results,
                                uint64_t issue_cycle, uint32_t pc,
                                uint32_t raw_instruction) {
    // Phase M3 (valid/ready): the slot is memoryless — cleared every
    // evaluate. Coalescing may stage every cycle (re-stage on hold) and
    // overwriting an empty slot is harmless. Asserting that the prior
    // value was empty is unnecessary now.
    next_load_cmd_.valid = true;
    next_load_cmd_.addr = addr;
    next_load_cmd_.warp_id = warp_id;
    next_load_cmd_.lane_mask = lane_mask;
    next_load_cmd_.results = results;
    next_load_cmd_.issue_cycle = issue_cycle;
    next_load_cmd_.pc = pc;
    next_load_cmd_.raw_instruction = raw_instruction;
}

void L1Cache::set_next_store_cmd(uint32_t line_addr, uint32_t warp_id,
                                 uint64_t issue_cycle, uint32_t pc,
                                 uint32_t raw_instruction) {
    next_store_cmd_.valid = true;
    next_store_cmd_.line_addr = line_addr;
    next_store_cmd_.warp_id = warp_id;
    next_store_cmd_.issue_cycle = issue_cycle;
    next_store_cmd_.pc = pc;
    next_store_cmd_.raw_instruction = raw_instruction;
}

bool L1Cache::any_pinned_tag() const {
    // Effective pin (chain pin OR write-ack pin) over all sets — used by
    // next_cmd_stall_reason() for LINE_PINNED trace classification.
    for (uint32_t set = 0; set < num_sets_; ++set) {
        if (is_pinned(set)) return true;
    }
    return false;
}

CacheStallReason L1Cache::next_cmd_stall_reason() const {
    // Trace classification: pure cache-state resource-exhaustion accessor.
    // Independent of any in-flight cmd. Order: MSHR_FULL > WB_FULL > LINE_PINNED.
    if (!mshrs_.has_free()) return CacheStallReason::MSHR_FULL;
    if (write_buffer_.size() >= write_buffer_depth_) return CacheStallReason::WRITE_BUFFER_FULL;
    if (any_pinned_tag()) return CacheStallReason::LINE_PINNED;
    return CacheStallReason::NONE;
}

void L1Cache::commit() {
    // Phase 9: flip the REGISTERED observable state (pending_fill_,
    // trace events). stalled_ / stall_reason_ / fill_installed_set_ are
    // COMBINATIONAL same-tick scratch (single slot, reset at top of
    // evaluate, observed mid-tick), so they are not flipped here. The tag
    // array, the MSHR file, and the write buffer are all REGISTERED and are
    // flipped / applied below — see resources/timing_discipline.md row 10.
    // The gather-extract port-claim flag is owned by LoadGatherBufferFile
    // (separate REGISTERED pair).
    current_tags_ = next_tags_;
    mshrs_.commit();
    // Registered write-buffer FIFO: apply the staged ops. Pop first, then
    // push, so a same-cycle drain+enqueue nets to no depth change. Then
    // clear the staging slots and release the enqueue port for next cycle.
    if (next_write_buffer_pop_) {
        write_buffer_.pop_front();
    }
    if (next_write_buffer_push_.has_value()) {
        write_buffer_.push_back(*next_write_buffer_push_);
    }
    next_write_buffer_pop_ = false;
    next_write_buffer_push_.reset();
    next_write_buffer_port_claimed_ = false;
    current_pending_fill_ = next_pending_fill_;
    current_outstanding_writes_ = next_outstanding_writes_;
    current_outstanding_writes_total_ = next_outstanding_writes_total_;
    current_last_miss_event_ = next_last_miss_event_;
    current_last_fill_event_ = next_last_fill_event_;
    current_last_drain_event_ = next_last_drain_event_;
    current_last_pin_stall_event_ = next_last_pin_stall_event_;
    // Phase M3 (valid/ready): unconditional flip. Cache is memoryless —
    // evaluate clears current_*_cmd_ every cycle regardless of success,
    // so there's no retry slot to preserve. Coalescing's processing_/
    // current_entry_ holds the source data and re-stages on a hold.
    current_load_cmd_ = next_load_cmd_;
    next_load_cmd_ = LoadCommand{};
    current_store_cmd_ = next_store_cmd_;
    next_store_cmd_ = StoreCommand{};
    // fill_installed_set_ is COMBINATIONAL same-tick scratch — clear it at
    // the tick boundary so it cannot leak into the next tick. evaluate()
    // also clears it at the top of the tick (the production path).
    fill_installed_set_ = -1;
}

bool L1Cache::is_idle() const {
    // Idle reflects committed observable state. current_pending_fill_ is
    // set when a deferred fill is carrying across cycles. Phase M3:
    // current_/next_ cmd slots also count — an in-flight cmd is not idle.
    // A write that left write_buffer_ but is not yet acked keeps the cache
    // non-idle (O(1) via the total scalar — no per-set scan).
    return !current_pending_fill_.valid && !mshrs_.has_active() && write_buffer_.empty()
           && current_outstanding_writes_total_ == 0
           && !current_load_cmd_.valid && !next_load_cmd_.valid
           && !current_store_cmd_.valid && !next_store_cmd_.valid;
}

uint32_t L1Cache::active_mshr_count() const {
    uint32_t count = 0;
    for (uint32_t i = 0; i < mshrs_.num_entries(); ++i) {
        if (mshrs_.current_at(i).valid) {
            count++;
        }
    }
    return count;
}

std::vector<uint32_t> L1Cache::active_mshr_warps() const {
    std::vector<uint32_t> warps;
    for (uint32_t i = 0; i < mshrs_.num_entries(); ++i) {
        if (mshrs_.current_at(i).valid) {
            warps.push_back(mshrs_.current_at(i).warp_id);
        }
    }
    return warps;
}

void L1Cache::reset() {
    for (auto& t : current_tags_) {
        t.valid = false;
        t.tag = 0;
        t.pinned = false;
    }
    next_tags_ = current_tags_;
    mshrs_.reset();
    write_buffer_.clear();
    next_write_buffer_port_claimed_ = false;
    next_write_buffer_push_.reset();
    next_write_buffer_pop_ = false;
    std::fill(current_outstanding_writes_.begin(), current_outstanding_writes_.end(), 0u);
    std::fill(next_outstanding_writes_.begin(), next_outstanding_writes_.end(), 0u);
    current_outstanding_writes_total_ = 0;
    next_outstanding_writes_total_ = 0;
    stalled_ = false;
    stall_reason_ = CacheStallReason::NONE;
    fill_installed_set_ = -1;
    current_pending_fill_ = PendingCacheFill{};
    next_pending_fill_ = PendingCacheFill{};
    current_last_miss_event_ = CacheMissTraceEvent{};
    next_last_miss_event_ = CacheMissTraceEvent{};
    current_last_fill_event_ = CacheFillTraceEvent{};
    next_last_fill_event_ = CacheFillTraceEvent{};
    current_last_drain_event_ = CacheSecondaryDrainTraceEvent{};
    next_last_drain_event_ = CacheSecondaryDrainTraceEvent{};
    current_last_pin_stall_event_ = CachePinStallTraceEvent{};
    next_last_pin_stall_event_ = CachePinStallTraceEvent{};
    current_load_cmd_ = LoadCommand{};
    next_load_cmd_ = LoadCommand{};
    current_store_cmd_ = StoreCommand{};
    next_store_cmd_ = StoreCommand{};
}

uint32_t L1Cache::pinned_line_count() const {
    uint32_t count = 0;
    for (const auto& t : current_tags_) {
        if (t.valid && t.pinned) count++;
    }
    return count;
}

uint32_t L1Cache::secondary_mshr_count() const {
    uint32_t count = 0;
    for (uint32_t i = 0; i < mshrs_.num_entries(); ++i) {
        const auto& m = mshrs_.current_at(i);
        if (m.valid && m.is_secondary) count++;
    }
    return count;
}

} // namespace gpu_sim
