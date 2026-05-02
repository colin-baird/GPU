#include "gpu_sim/timing/coalescing_unit.h"

namespace gpu_sim {

CoalescingUnit::CoalescingUnit(LdStUnit& ldst, L1Cache& cache,
                               LoadGatherBufferFile& gather_file,
                               uint32_t line_size, Stats& stats)
    : ldst_(ldst), cache_(cache), gather_file_(gather_file),
      line_size_(line_size), stats_(stats) {}

void CoalescingUnit::evaluate() {
    // Phase M3: replace the COMBINATIONAL same-tick stall read +
    // synchronous process_load/process_store call with a COMBINATIONAL
    // backward stall + REGISTERED forward cmd path. cache.evaluate runs
    // earlier in the tick; cache.next_cmd_stall() reflects end-of-cache-
    // evaluate state. When the stall is clear, we stage a cmd via
    // cache.set_next_load_cmd / set_next_store_cmd; cache.commit flips
    // the slot and cache.evaluate at the next cycle processes it. The
    // stall guarantees acceptability — no rejection-retry handshake.
    //
    // The legacy cache.next_stalled() / next_stall_reason() path is left
    // intact for the FIFO-front gating below (gather buffer busy check
    // remains REGISTERED back-pressure on the per-warp slot).
    if (cache_.next_cmd_stall()) return;

    if (!processing_) {
        if (ldst_.current_fifo_empty()) return;
        const auto& fifo_front = ldst_.current_fifo_front();

        // For loads, the target warp's gather buffer must be free. If busy,
        // stall without popping — another load for this warp is outstanding.
        if (fifo_front.is_load && gather_file_.current_busy(fifo_front.warp_id)) {
            stats_.gather_buffer_stall_cycles++;
            return;
        }

        current_entry_ = fifo_front;
        // Phase M1: REGISTERED FIFO. Stage the pop intent; commit() will
        // apply it via ldst_.pop_front(). Producer's commit only writes the
        // back of the deque, so the front observed here is the same one
        // pop_front() will remove next phase.
        next_pop_ = true;
        processing_ = true;

        // All-or-nothing coalescing check: do all 32 addresses fall in one cache line?
        uint32_t first_line = current_entry_.trace.mem_addresses[0] / line_size_;
        is_coalesced_ = true;
        for (uint32_t i = 1; i < WARP_SIZE; ++i) {
            if (current_entry_.trace.mem_addresses[i] / line_size_ != first_line) {
                is_coalesced_ = false;
                break;
            }
        }

        if (is_coalesced_) {
            stats_.coalesced_requests++;
        } else {
            stats_.serialized_requests++;
        }

        serial_index_ = 0;

        if (current_entry_.is_load) {
            gather_file_.claim(current_entry_.warp_id, current_entry_.dest_reg,
                               current_entry_.trace.pc, current_entry_.issue_cycle,
                               current_entry_.trace.decoded.raw);
        }
    }

    if (processing_) {
        if (is_coalesced_) {
            // Single REGISTERED cmd for all 32 threads; fires next cycle.
            uint32_t line_addr = current_entry_.trace.mem_addresses[0] / line_size_;
            if (current_entry_.is_load) {
                cache_.set_next_load_cmd(
                    current_entry_.trace.mem_addresses[0],
                    current_entry_.warp_id,
                    0xFFFFFFFFu,
                    current_entry_.trace.results,
                    current_entry_.issue_cycle,
                    current_entry_.trace.pc,
                    current_entry_.trace.decoded.raw);
            } else {
                cache_.set_next_store_cmd(line_addr, current_entry_.warp_id,
                                          current_entry_.issue_cycle,
                                          current_entry_.trace.pc,
                                          current_entry_.trace.decoded.raw);
            }
            processing_ = false;
        } else {
            // Serialized: one REGISTERED cmd per cycle (cache processes at N+1).
            if (serial_index_ < WARP_SIZE) {
                uint32_t addr = current_entry_.trace.mem_addresses[serial_index_];
                uint32_t line_addr = addr / line_size_;

                if (current_entry_.is_load) {
                    uint32_t lane_mask = 1u << serial_index_;
                    cache_.set_next_load_cmd(
                        addr, current_entry_.warp_id, lane_mask,
                        current_entry_.trace.results, current_entry_.issue_cycle,
                        current_entry_.trace.pc, current_entry_.trace.decoded.raw);
                } else {
                    cache_.set_next_store_cmd(line_addr, current_entry_.warp_id,
                                              current_entry_.issue_cycle,
                                              current_entry_.trace.pc,
                                              current_entry_.trace.decoded.raw);
                }
                serial_index_++;
            }

            if (serial_index_ >= WARP_SIZE) {
                processing_ = false;
            }
        }
    }
}

void CoalescingUnit::commit() {
    // Phase M1: apply the staged FIFO pop. The defensive empty check below
    // should never fire — between the evaluate that set next_pop_ and this
    // commit, the only mutation to addr_gen_fifo_ is LdStUnit::commit()
    // pushing to the back; the front is stable.
    if (next_pop_ && !ldst_.current_fifo_empty()) {
        ldst_.pop_front();
    }
    next_pop_ = false;
}

void CoalescingUnit::reset() {
    processing_ = false;
    serial_index_ = 0;
}

} // namespace gpu_sim
