#include "gpu_sim/timing/coalescing_unit.h"

namespace gpu_sim {

CoalescingUnit::CoalescingUnit(LdStUnit& ldst, L1Cache& cache,
                               LoadGatherBufferFile& gather_file,
                               uint32_t line_size, Stats& stats)
    : ldst_(ldst), cache_(cache), gather_file_(gather_file),
      line_size_(line_size), stats_(stats) {}

void CoalescingUnit::evaluate() {
    if (cache_.is_stalled()) return;

    if (!processing_) {
        if (ldst_.fifo_empty()) return;
        const auto& fifo_front = ldst_.fifo_front();

        // For loads, the target warp's gather buffer must be free. If busy,
        // stall without popping — another load for this warp is outstanding.
        if (fifo_front.is_load && gather_file_.is_busy(fifo_front.warp_id)) {
            stats_.gather_buffer_stall_cycles++;
            return;
        }

        current_entry_ = fifo_front;
        ldst_.fifo_pop();
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
            // Single request for all 32 threads
            uint32_t line_addr = current_entry_.trace.mem_addresses[0] / line_size_;
            bool accepted;
            if (current_entry_.is_load) {
                accepted = cache_.process_load(
                    current_entry_.trace.mem_addresses[0],
                    current_entry_.warp_id,
                    0xFFFFFFFFu,
                    current_entry_.trace.results,
                    current_entry_.issue_cycle,
                    current_entry_.trace.pc,
                    current_entry_.trace.decoded.raw);
            } else {
                accepted = cache_.process_store(line_addr, current_entry_.warp_id,
                                               current_entry_.issue_cycle,
                                               current_entry_.trace.pc,
                                               current_entry_.trace.decoded.raw);
            }

            if (accepted) {
                processing_ = false;
            }
        } else {
            // Serialized: one request per thread per cycle. Each load lane
            // deposits its single slot into the warp's gather buffer; the
            // writeback fires when all 32 slots are valid.
            if (serial_index_ < WARP_SIZE) {
                uint32_t addr = current_entry_.trace.mem_addresses[serial_index_];
                uint32_t line_addr = addr / line_size_;

                bool accepted;
                if (current_entry_.is_load) {
                    uint32_t lane_mask = 1u << serial_index_;
                    accepted = cache_.process_load(
                        addr, current_entry_.warp_id, lane_mask,
                        current_entry_.trace.results, current_entry_.issue_cycle,
                        current_entry_.trace.pc, current_entry_.trace.decoded.raw);
                } else {
                    accepted = cache_.process_store(line_addr, current_entry_.warp_id,
                                                   current_entry_.issue_cycle,
                                                   current_entry_.trace.pc,
                                                   current_entry_.trace.decoded.raw);
                }

                if (accepted) {
                    serial_index_++;
                }
            }

            if (serial_index_ >= WARP_SIZE) {
                processing_ = false;
            }
        }
    }
}

void CoalescingUnit::commit() {}

void CoalescingUnit::reset() {
    processing_ = false;
    serial_index_ = 0;
}

} // namespace gpu_sim
