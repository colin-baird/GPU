#include "gpu_sim/timing/coalescing_unit.h"

namespace gpu_sim {

CoalescingUnit::CoalescingUnit(LdStUnit& ldst, L1Cache& cache,
                               LoadGatherBufferFile& gather_file,
                               uint32_t line_size, Stats& stats)
    : ldst_(ldst), cache_(cache), gather_file_(gather_file),
      line_size_(line_size), stats_(stats) {}

void CoalescingUnit::evaluate() {
    // Phase M3 (valid/ready handshake):
    //
    //   producer (this)        consumer (cache)
    //   ─────────────          ────────────────
    //   stage cmd N → next_*_cmd_ (commit buffer)
    //                          cache.commit flips next → current
    //                          cache.evaluate processes current_*_cmd_
    //                          and asserts next_cmd_ready iff accepted
    //   read next_cmd_ready ← (this cycle's cache.evaluate, run earlier
    //                          in tick order)
    //   advance | re-stage
    //
    // Cache is memoryless: it clears current_*_cmd_ every evaluate
    // regardless of success. Producer-side retry lives entirely in
    // (current_entry_, serial_index_, processing_) — on !ready we leave
    // those untouched and re-stage the same cmd at the bottom.

    // Step 1: read this cycle's ack from cache. Only meaningful if we
    // staged a cmd last cycle (cmd_in_flight_); otherwise the ready
    // signal reflects an unrelated or stale cycle and must not advance us.
    if (cmd_in_flight_) {
        if (cache_.next_cmd_ready()) {
            if (is_coalesced_) {
                processing_ = false;
            } else {
                serial_index_++;
                if (serial_index_ >= WARP_SIZE) {
                    processing_ = false;
                }
            }
        }
        cmd_in_flight_ = false;
    }

    // Step 2: pop a new entry from the LDST FIFO if we're idle.
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
        // Phase M1: defer the pop to commit. Read remains a stable
        // committed-state read.
        next_pop_ = true;
        processing_ = true;

        // All-or-nothing coalescing check.
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

    // Step 3: stage the current lane's cmd into the commit buffer. We
    // stage every cycle we have work — on a hold (!ready last cycle), the
    // re-stage repeats the same cmd so cache attempts it again next cycle.
    // The slot's overwrite is harmless because cache cleared it at evaluate
    // and our prior commit, if any, has already been consumed.
    if (processing_) {
        if (is_coalesced_) {
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
        } else {
            // Serialized: one lane per cycle. serial_index_ advances only
            // when cache asserted ready (Step 1).
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
        }
        cmd_in_flight_ = true;
    }
}

void CoalescingUnit::commit() {
    // Phase M1: apply the deferred pop on the ldst addr-gen FIFO. This commit
    // is ungated — coalescing is not part of the writeback-stall freeze, so a
    // pop may still drain the FIFO even when LdStUnit::commit is held; the
    // held push lands on the resumed cycle.
    if (next_pop_) {
        if (!ldst_.current_fifo_empty()) {
            ldst_.pop_front();
        }
        next_pop_ = false;
    }
}

void CoalescingUnit::reset() {
    processing_ = false;
    serial_index_ = 0;
    next_pop_ = false;
    cmd_in_flight_ = false;
}

} // namespace gpu_sim
