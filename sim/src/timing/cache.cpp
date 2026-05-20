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
      mshrs_(num_mshrs),
      write_buffer_depth_(write_buffer_depth),
      max_outstanding_writes_(max_outstanding_writes),
      mem_if_(mem_if), gather_file_(gather_file), stats_(stats) {
    // A cap of 0 deadlocks every store (no write-through could ever be
    // enqueued). SimConfig::validate() also enforces this.
    assert(max_outstanding_writes_ >= 1);
    // Enroll every Reg / RegFifo / PulseReg so reset_all() / commit_all()
    // drive them uniformly. The cache has no seed_next() — it self-seeds
    // the carry-forward Regs in-place at the top of evaluate() (see the
    // seed block). The load_cmd_ / store_cmd_ slots are PulseReg<T>, which
    // is self-clearing at commit (Phase 4 of current_mut() elimination), so
    // they need no entry in the in-place seed block. See cache.h above
    // register declarations and resources/timing_discipline.md.
    register_state(&tags_, &write_buffer_, &pending_fill_,
                   &outstanding_writes_, &outstanding_writes_total_,
                   &last_miss_event_, &last_fill_event_,
                   &last_drain_event_, &last_pin_stall_event_,
                   &load_cmd_, &store_cmd_);
    // Initial tag/outstanding-writes vector sizing. reset() applies the
    // same sizing through reset_all() + explicit re-sizing, but the
    // constructor needs the vectors initialized to num_sets_ elements before
    // any evaluate()/process_load can index them.
    tags_.initialize(std::vector<CacheTag>(num_sets_));
    outstanding_writes_.initialize(std::vector<uint32_t>(num_sets_));
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
    // tags_.current()[set].pinned and the per-set outstanding-write counter.
    return tags_.current()[set].pinned || outstanding_writes_.current()[set] > 0;
}

bool L1Cache::outstanding_writes_at_cap() const {
    return outstanding_writes_total_.current() >= max_outstanding_writes_;
}

bool L1Cache::queue_write_through(uint32_t line_addr) {
    // Single, fallible enqueue wrapper. The write buffer has one enqueue port
    // per cycle: if it is already claimed, refuse without staging anything.
    if (write_buffer_.port_claimed()) {
        stats_.write_buffer_port_conflict_cycles++;
        return false;
    }
    // Claim the port and stage the enqueue (applied to write_buffer_ at
    // commit). Both outstanding-write counters are bumped into next_ — a
    // queued write-through pins its set and consumes one outstanding-write
    // credit until its ack returns.
    write_buffer_.claim_port();
    write_buffer_.stage_push(line_addr);
    uint32_t set = get_set(line_addr * line_size_);
    ++outstanding_writes_.next_mut()[set];
    ++outstanding_writes_total_.next_mut();
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
    if (fill_installed_set_.value() == static_cast<int32_t>(set)) {
        stats_.fill_conflict_retry_cycles++;
        return false;
    }

    const auto& tags = tags_.current();
    if (tags[set].valid && tags[set].tag == tag) {
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
    if (tags[set].valid && tags[set].tag != tag && is_pinned(set)) {
        // Attribute the stall: chain pin takes precedence over the write-ack
        // pin (precedence: chain-pin > write-ack-pin).
        if (tags[set].pinned) {
            stats_.line_pin_stall_cycles++;
        } else {
            stats_.write_ack_pin_stall_cycles++;
        }
        stalled_.drive(true);
        stall_reason_.drive(CacheStallReason::LINE_PINNED);
        auto& pin_ev = last_pin_stall_event_.next_mut();
        pin_ev.valid = true;
        pin_ev.warp_id = warp_id;
        pin_ev.requested_line_addr = line_addr;
        pin_ev.pinned_line_addr = tags[set].tag * num_sets_ + set;
        pin_ev.is_store = false;
        return false;
    }

    if (!mshrs_.has_free()) {
        stats_.mshr_stall_cycles++;
        stalled_.drive(true);
        stall_reason_.drive(CacheStallReason::MSHR_FULL);
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

    auto& miss_ev = last_miss_event_.next_mut();
    miss_ev.valid = true;
    miss_ev.warp_id = warp_id;
    miss_ev.line_addr = line_addr;
    miss_ev.is_store = false;
    miss_ev.pc = pc;
    miss_ev.raw_instruction = raw_instruction;
    miss_ev.merged_secondary = (tail_idx >= 0);

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
    if (fill_installed_set_.value() == static_cast<int32_t>(set)) {
        stats_.fill_conflict_retry_cycles++;
        return false;
    }

    const auto& tags = tags_.current();
    if (tags[set].valid && tags[set].tag == tag) {
        // Store hit: write-through to write buffer (timing model tracks tags only, not data).
        // Admission requires both a free write-buffer slot and an outstanding-
        // write credit; precedence for stall attribution is buffer-full > cap.
        if (write_buffer_.current_size() >= write_buffer_depth_ || outstanding_writes_at_cap()) {
            if (write_buffer_.current_size() >= write_buffer_depth_) {
                stats_.write_buffer_stall_cycles++;
            } else {
                stats_.write_throttle_stall_cycles++;
            }
            stalled_.drive(true);
            stall_reason_.drive(CacheStallReason::WRITE_BUFFER_FULL);
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
    if (tags[set].valid && tags[set].tag != tag && is_pinned(set)) {
        // Attribute the stall: chain-pin > write-ack-pin.
        if (tags[set].pinned) {
            stats_.line_pin_stall_cycles++;
        } else {
            stats_.write_ack_pin_stall_cycles++;
        }
        stalled_.drive(true);
        stall_reason_.drive(CacheStallReason::LINE_PINNED);
        auto& pin_ev = last_pin_stall_event_.next_mut();
        pin_ev.valid = true;
        pin_ev.warp_id = warp_id;
        pin_ev.requested_line_addr = line_addr;
        pin_ev.pinned_line_addr = tags[set].tag * num_sets_ + set;
        pin_ev.is_store = true;
        return false;
    }

    if (!mshrs_.has_free()) {
        stats_.mshr_stall_cycles++;
        stalled_.drive(true);
        stall_reason_.drive(CacheStallReason::MSHR_FULL);
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

    auto& miss_ev = last_miss_event_.next_mut();
    miss_ev.valid = true;
    miss_ev.warp_id = warp_id;
    miss_ev.line_addr = line_addr;
    miss_ev.is_store = true;
    miss_ev.pc = pc;
    miss_ev.raw_instruction = raw_instruction;
    miss_ev.merged_secondary = (tail_idx >= 0);

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
    const auto& tags = tags_.current();
    if (tags[set].valid && is_pinned(set) && tags[set].tag != new_tag) {
        // Attribute the stall: chain-pin > write-ack-pin.
        if (tags[set].pinned) {
            stats_.line_pin_stall_cycles++;
        } else {
            stats_.write_ack_pin_stall_cycles++;
        }
        auto& fill_ev = last_fill_event_.next_mut();
        fill_ev.valid = true;
        fill_ev.warp_id = mshr.warp_id;
        fill_ev.line_addr = mshr.cache_line_addr;
        fill_ev.is_store = mshr.is_store;
        fill_ev.pc = mshr.pc;
        fill_ev.raw_instruction = mshr.raw_instruction;
        fill_ev.chain_length_at_fill = 0;
        fill_ev.deferred = true;
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
        if (write_buffer_.current_size() >= write_buffer_depth_ || outstanding_writes_at_cap()) {
            if (write_buffer_.current_size() >= write_buffer_depth_) {
                stats_.write_buffer_stall_cycles++;
            } else {
                stats_.write_throttle_stall_cycles++;
            }
            stalled_.drive(true);
            stall_reason_.drive(CacheStallReason::WRITE_BUFFER_FULL);
            return false;
        }

        // Probe the fallible write-buffer enqueue BEFORE installing the tag,
        // so a port-busy refusal leaves no partial state — the fill simply
        // stays in pending_fill_ and is retried next cycle. (In tick order
        // FILL runs first, so the port is normally free here.)
        if (!queue_write_through(mshr.cache_line_addr)) {
            return false;
        }
        auto& tag_slot = tags_.next_mut()[set];
        tag_slot.valid = true;
        tag_slot.tag = get_tag(addr);
        tag_slot.pinned = (mshr.next_in_chain != MSHREntry::INVALID_MSHR);
    } else {
        // Load miss fill: deposit lane values into the owning warp's gather
        // buffer. FILL runs first in the tick (cache.handle_responses is
        // invoked at the top of cache.evaluate()), so try_write() lands and
        // sets the buffer's `next_port_claimed`. Any same-tick HIT to the
        // same buffer (later via coalescing_->evaluate -> process_load) will
        // see the live `next_port_claimed` and bail.
        auto& tag_slot = tags_.next_mut()[set];
        tag_slot.valid = true;
        tag_slot.tag = get_tag(addr);
        tag_slot.pinned = (mshr.next_in_chain != MSHREntry::INVALID_MSHR);
        bool ok = gather_file_.try_write(mshr.warp_id, mshr.lane_mask, mshr.results,
                                         LoadGatherBufferFile::GatherWriteSource::FILL);
        (void)ok;
    }

    // Successful (non-deferred) install: record the set so a command racing
    // this fill to the same set is rejected and retried (fill-conflict
    // retry). The write-buffer-full early return above does not reach here.
    fill_installed_set_.drive(static_cast<int32_t>(set));

    auto& fill_ev = last_fill_event_.next_mut();
    fill_ev.valid = true;
    fill_ev.warp_id = mshr.warp_id;
    fill_ev.line_addr = mshr.cache_line_addr;
    fill_ev.is_store = mshr.is_store;
    fill_ev.pc = mshr.pc;
    fill_ev.raw_instruction = mshr.raw_instruction;
    fill_ev.chain_length_at_fill = chain_length;
    mshrs_.free(resp.mshr_id);
    return true;
}

void L1Cache::handle_responses() {
    // Step 1: consume at most one write ack per cycle, UNCONDITIONALLY —
    // before the deferred-fill early return below. This is the deadlock fix:
    // a fill deferred on a write-ack pin can only make progress once the
    // blocking ack is consumed and the pin clears, so the ack drain must run
    // even while a fill is deferred. The ack returns one outstanding-write
    // credit (per-set pin + global cap). outstanding_writes_.next_mut() has
    // just been seeded from current_ in evaluate(), so the asserts against
    // .current() and the decrements of .next_mut() are consistent.
    if (mem_if_.current_has_write_ack()) {
        MemoryResponse ack = mem_if_.current_write_ack_front();
        mem_if_.stage_write_ack_pop();
        uint32_t set = get_set(ack.line_addr * line_size_);
        assert(tags_.current()[set].valid &&
               "write ack: acked line must still be resident (self-enforcing)");
        assert(outstanding_writes_.current()[set] > 0 &&
               "write ack: per-set outstanding-write underflow");
        assert(outstanding_writes_total_.current() > 0 &&
               "write ack: global outstanding-write underflow");
        --outstanding_writes_.next_mut()[set];
        --outstanding_writes_total_.next_mut();
    }

    // pending_fill_ is REGISTERED: read from current_, write next_. The
    // top-of-evaluate seed (pending_fill_.set_next(pending_fill_.current()))
    // carries a deferred fill forward unless this cycle's complete_fill
    // succeeds.
    if (pending_fill_.current().valid) {
        if (complete_fill(pending_fill_.current().response)) {
            pending_fill_.next_mut().valid = false;
        }
        return;
    }

    // Read fills only — write completions now arrive on the separate
    // write-ack channel drained above, never on responses_.
    //
    // Phase 4 (close-the-Reg-family-migration): at most one response is
    // consumed per cycle (the while-then-return is a hardware single-read-
    // port pattern). The peek reads the cross-stage RegFifo's current_front
    // — last cycle's committed completion; the stage_pop applies at the
    // cross-stage commit pass, atomic with mem_if's same-cycle stage_push.
    if (mem_if_.current_has_response()) {
        auto resp = mem_if_.current_response_front();
        mem_if_.stage_response_pop();

        auto& pf = pending_fill_.next_mut();
        pf.valid = true;
        pf.response = resp;

        if (complete_fill(resp)) {
            pending_fill_.next_mut().valid = false;
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
        // Registered tag array: read tags_.current(), like every other tag
        // reader (registered-tag-array.md Step 2). The pin-clear further
        // down is the write side and targets tags_.next_mut() (Step 3).
        //
        // A secondary of the line whose primary fills THIS cycle cannot
        // drain in the fill cycle regardless of which slot is read: the
        // registered MSHR file keeps the primary valid in current_entries_
        // until commit, so the head-detection scan below never sees a
        // secondary become the chain head until the primary is
        // committed-free (T+1). next_tags here would therefore observe
        // complete_fill's same-cycle install but the drain is gated anyway —
        // reading current is observationally identical and is the
        // discipline-correct choice (no same-cycle read of a registered
        // next_ write).
        const auto& tags = tags_.current();
        if (!tags[set].valid || tags[set].tag != tag || !tags[set].pinned) {
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
            if (write_buffer_.current_size() >= write_buffer_depth_ || outstanding_writes_at_cap()) {
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
            auto& drain_ev = last_drain_event_.next_mut();
            drain_ev.valid = true;
            drain_ev.warp_id = cand.warp_id;
            drain_ev.line_addr = line_addr;
            drain_ev.is_store = true;
            drain_ev.pc = cand.pc;
            drain_ev.raw_instruction = cand.raw_instruction;
            uint32_t next = cand.next_in_chain;
            mshrs_.free(i);
            if (next == MSHREntry::INVALID_MSHR) {
                tags_.next_mut()[set].pinned = false;
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
            auto& drain_ev = last_drain_event_.next_mut();
            drain_ev.valid = true;
            drain_ev.warp_id = cand.warp_id;
            drain_ev.line_addr = line_addr;
            drain_ev.is_store = false;
            drain_ev.pc = cand.pc;
            drain_ev.raw_instruction = cand.raw_instruction;
            uint32_t next = cand.next_in_chain;
            mshrs_.free(i);
            if (next == MSHREntry::INVALID_MSHR) {
                tags_.next_mut()[set].pinned = false;
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
    if (!write_buffer_.current_empty() && !mem_if_.next_request_stall()) {
        mem_if_.set_next_write_request(write_buffer_.current_front());
        write_buffer_.stage_pop();
    }
}

void L1Cache::evaluate() {
    // seed_next: scratch fields (stall flags, trace events) clear to false
    // so an unwritten cycle commits a quiescent observable state. The
    // pending_fill_ slot is a multi-cycle deferred-fill carrier, so it
    // copies forward; complete_fill clears pending_fill_.next_mut().valid on
    // success, leaving pending_fill_.current() set across the cycle.
    //
    // In-place seeding is used here rather than seed_all() because the
    // trace-event Regs need both a seed AND an immediate clear of .valid
    // (an un-driven cycle commits valid=false) — a behavior the generic
    // seed_all() cannot express in one call. load_cmd_ / store_cmd_ are
    // PulseReg<T>, which is self-clearing at commit (Phase 4 of
    // current_mut() elimination), so they need no entry below.
    //
    // Phase 7: stalled_ / stall_reason_ / fill_installed_set_ are Wire<T>;
    // reset() at the top of evaluate de-asserts them (default false /
    // CacheStallReason::NONE / -1) — equivalent to today's `= false` /
    // `= NONE` / `= -1` clear.
    stalled_.reset();
    stall_reason_.reset();
    fill_installed_set_.reset();
    // Trace events: only the .valid flag matters when un-asserted. Seed the
    // whole struct from current (faithful to a Reg<T>'s auto-seed semantics)
    // then clear .valid. Faithful to pre-migration: only .valid was assigned;
    // the other fields are observationally dead when valid is false.
    last_miss_event_.set_next(last_miss_event_.current());
    last_miss_event_.next_mut().valid = false;
    last_fill_event_.set_next(last_fill_event_.current());
    last_fill_event_.next_mut().valid = false;
    last_fill_event_.next_mut().deferred = false;
    last_drain_event_.set_next(last_drain_event_.current());
    last_drain_event_.next_mut().valid = false;
    last_pin_stall_event_.set_next(last_pin_stall_event_.current());
    last_pin_stall_event_.next_mut().valid = false;
    pending_fill_.set_next(pending_fill_.current());
    // REGISTERED tag array: seed next from current so an un-touched set
    // carries forward; complete_fill / drain_secondary_chain_head write
    // tags_.next_mut(), commit() flips into tags_.current().
    tags_.set_next(tags_.current());
    // REGISTERED write-ack pin counters: seed next from current. Enqueues
    // increment next_; the write-ack consumer decrements next_.
    outstanding_writes_.set_next(outstanding_writes_.current());
    outstanding_writes_total_.set_next(outstanding_writes_total_.current());
    // REGISTERED MSHR file: seed its next-state from committed state.
    mshrs_.seed_next();

    // Phase 7 + M3: FILL > secondary > HIT priority encoded by tick order.
    // handle_responses() (FILL) runs first; drain_secondary_chain_head()
    // (secondary) runs second; the HIT slot processes the REGISTERED
    // committed load/store cmd that coalescing submitted at the prior cycle.
    // The shared gather-extract port is arbitrated by
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
    // Phase 7: next_cmd_ready_ is Wire<bool> — reset to default false; later
    // drive(true) on an accepted cmd. Coalescing reads via next_cmd_ready()
    // later in the same tick.
    next_cmd_ready_.reset();
    int processed_count = 0;
    if (load_cmd_.current().valid) {
        const auto& lc = load_cmd_.current();
        bool ok = process_load(lc.addr, lc.warp_id, lc.lane_mask, lc.results,
                               lc.issue_cycle, lc.pc, lc.raw_instruction);
        // No mid-cycle Q write — the PulseReg<LoadCommand> slot's seed-to-T{}
        // at the top of the next tick defaults the slot to invalid, so
        // coalescing must explicitly re-stage to keep the slot live.
        if (ok) {
            next_cmd_ready_.drive(true);
            ++processed_count;
        }
    }
    if (store_cmd_.current().valid) {
        const auto& sc = store_cmd_.current();
        bool ok = process_store(sc.line_addr, sc.warp_id, sc.issue_cycle, sc.pc,
                                sc.raw_instruction);
        if (ok) {
            next_cmd_ready_.drive(true);
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
    auto& lc = load_cmd_.next_mut();
    lc.valid = true;
    lc.addr = addr;
    lc.warp_id = warp_id;
    lc.lane_mask = lane_mask;
    lc.results = results;
    lc.issue_cycle = issue_cycle;
    lc.pc = pc;
    lc.raw_instruction = raw_instruction;
}

void L1Cache::set_next_store_cmd(uint32_t line_addr, uint32_t warp_id,
                                 uint64_t issue_cycle, uint32_t pc,
                                 uint32_t raw_instruction) {
    auto& sc = store_cmd_.next_mut();
    sc.valid = true;
    sc.line_addr = line_addr;
    sc.warp_id = warp_id;
    sc.issue_cycle = issue_cycle;
    sc.pc = pc;
    sc.raw_instruction = raw_instruction;
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
    if (write_buffer_.current_size() >= write_buffer_depth_) return CacheStallReason::WRITE_BUFFER_FULL;
    if (any_pinned_tag()) return CacheStallReason::LINE_PINNED;
    return CacheStallReason::NONE;
}

void L1Cache::commit() {
    // Phase 9: flip the REGISTERED observable state. stalled_ /
    // stall_reason_ / next_cmd_ready_ / fill_installed_set_ are Wire<T>
    // (Phase 7) — all COMBINATIONAL same-tick (no current/next pair, no
    // commit() flip; reset at top of evaluate(), observed mid-tick). The tag
    // array, the MSHR file, and the write-buffer FIFO are all REGISTERED —
    // see resources/timing_discipline.md row 10. The gather-extract
    // port-claim flag is owned by LoadGatherBufferFile (now a Wire<bool>).
    //
    // Phase 5a: commit_all() drives every enrolled Reg (tags_, pending_fill_,
    // outstanding_writes_, outstanding_writes_total_, the four last_*_event_,
    // load_cmd_, store_cmd_) and the write_buffer_ RegFifo (pop-then-push
    // applied, staging slots and port-claim flag cleared) in one sweep. The
    // MSHR file is committed separately (Phase 5b will fold it into the
    // RegisteredStage list).
    commit_all();
    mshrs_.commit();
    // fill_installed_set_ is COMBINATIONAL same-tick scratch (Wire<int32_t>,
    // default -1) — reset it at the tick boundary so it cannot leak into the
    // next tick. evaluate() also resets it at the top of the tick (the
    // production path); the dual reset is preserved exactly.
    fill_installed_set_.reset();
}

bool L1Cache::is_idle() const {
    // Idle reflects committed observable state. pending_fill_.current() is
    // set when a deferred fill is carrying across cycles. Phase M3:
    // current/next cmd slots also count — an in-flight cmd is not idle.
    // A write that left write_buffer_ but is not yet acked keeps the cache
    // non-idle (O(1) via the total scalar — no per-set scan).
    //
    // load_cmd_.next() / store_cmd_.next() are intra-class staged reads of
    // the same Reg; they observe a cmd that has been set_next'd by
    // coalescing this cycle but not yet committed.
    return !pending_fill_.current().valid && !mshrs_.has_active() && write_buffer_.current_empty()
           && outstanding_writes_total_.current() == 0
           && !load_cmd_.current().valid && !load_cmd_.next().valid
           && !store_cmd_.current().valid && !store_cmd_.next().valid;
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
    // Phase 5a: reset_all() clears every enrolled Reg's current_ AND next_
    // to T{} and clears the RegFifo's deque + staging. Then re-size the
    // tag and outstanding-writes vectors back to num_sets_ (Reg::reset()
    // value-initializes the inner vector to empty), and clear the
    // combinational scratch + the MSHR file by hand.
    reset_all();
    tags_.initialize(std::vector<CacheTag>(num_sets_));
    outstanding_writes_.initialize(std::vector<uint32_t>(num_sets_));
    mshrs_.reset();
    // Phase 7: Wire<T>::reset() de-asserts each combinational-backward signal
    // (default false / CacheStallReason::NONE / false / -1) — equivalent to
    // the prior `= false` / `= NONE` / `= -1` clears.
    stalled_.reset();
    stall_reason_.reset();
    next_cmd_ready_.reset();
    fill_installed_set_.reset();
}

uint32_t L1Cache::pinned_line_count() const {
    uint32_t count = 0;
    for (const auto& t : tags_.current()) {
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
