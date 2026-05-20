#include "gpu_sim/timing/coalescing_unit.h"

namespace gpu_sim {

CoalescingUnit::CoalescingUnit(LdStUnit& ldst, L1Cache& cache,
                               LoadGatherBufferFile& gather_file,
                               uint32_t line_size, Stats& stats)
    : ldst_(ldst), cache_(cache), gather_file_(gather_file),
      line_size_(line_size), stats_(stats) {
    register_state(&processing_, &current_entry_, &is_coalesced_,
                   &serial_index_, &cmd_in_flight_);
}

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
    // Phase 6: all reads .next() (the seeded staged value, equal to
    // current_ at the top of the tick); mutations via set_next / next_mut.
    if (cmd_in_flight_.next()) {
        if (cache_.next_cmd_ready()) {
            if (is_coalesced_.next()) {
                processing_.set_next(false);
            } else {
                serial_index_.next_mut()++;
                if (serial_index_.next() >= WARP_SIZE) {
                    processing_.set_next(false);
                }
            }
        }
        cmd_in_flight_.set_next(false);
    }

    // Step 2: pop a new entry from the LDST FIFO if we're idle.
    if (!processing_.next()) {
        if (ldst_.current_fifo_empty()) return;
        const auto& fifo_front = ldst_.current_fifo_front();

        // For loads, the target warp's gather buffer must be free. If busy,
        // stall without popping — another load for this warp is outstanding.
        if (fifo_front.is_load && gather_file_.current_busy(fifo_front.warp_id)) {
            stats_.gather_buffer_stall_cycles++;
            return;
        }

        current_entry_.set_next(fifo_front);
        // Phase 3 (close-the-Reg-family-migration): stage the pop directly on
        // the cross-stage addr-gen FIFO. The pop applies at TimingModel's
        // dedicated ungated cross-stage commit pass (NOT at CoalescingUnit's
        // own commit), so it drains even on a writeback-stall cycle when
        // LdStUnit's commit is gated and the producer holds its push.
        if (addr_gen_fifo_ != nullptr) {
            addr_gen_fifo_->stage_pop();
        }
        processing_.set_next(true);

        // All-or-nothing coalescing check.
        const auto& entry = current_entry_.next();
        uint32_t first_line = entry.trace.mem_addresses[0] / line_size_;
        bool coalesced = true;
        for (uint32_t i = 1; i < WARP_SIZE; ++i) {
            if (entry.trace.mem_addresses[i] / line_size_ != first_line) {
                coalesced = false;
                break;
            }
        }
        is_coalesced_.set_next(coalesced);

        if (coalesced) {
            stats_.coalesced_requests++;
        } else {
            stats_.serialized_requests++;
        }

        serial_index_.set_next(0);

        if (entry.is_load) {
            gather_file_.claim(entry.warp_id, entry.dest_reg,
                               entry.trace.pc, entry.issue_cycle,
                               entry.trace.decoded.raw);
        }
    }

    // Step 3: stage the current lane's cmd into the commit buffer. We
    // stage every cycle we have work — on a hold (!ready last cycle), the
    // re-stage repeats the same cmd so cache attempts it again next cycle.
    // The slot's overwrite is harmless because cache cleared it at evaluate
    // and our prior commit, if any, has already been consumed.
    if (processing_.next()) {
        const auto& entry = current_entry_.next();
        if (is_coalesced_.next()) {
            uint32_t line_addr = entry.trace.mem_addresses[0] / line_size_;
            if (entry.is_load) {
                cache_.set_next_load_cmd(
                    entry.trace.mem_addresses[0],
                    entry.warp_id,
                    0xFFFFFFFFu,
                    entry.trace.results,
                    entry.issue_cycle,
                    entry.trace.pc,
                    entry.trace.decoded.raw);
            } else {
                cache_.set_next_store_cmd(line_addr, entry.warp_id,
                                          entry.issue_cycle,
                                          entry.trace.pc,
                                          entry.trace.decoded.raw);
            }
        } else {
            // Serialized: one lane per cycle. serial_index_ advances only
            // when cache asserted ready (Step 1).
            uint32_t idx = serial_index_.next();
            uint32_t addr = entry.trace.mem_addresses[idx];
            uint32_t line_addr = addr / line_size_;

            if (entry.is_load) {
                uint32_t lane_mask = 1u << idx;
                cache_.set_next_load_cmd(
                    addr, entry.warp_id, lane_mask,
                    entry.trace.results, entry.issue_cycle,
                    entry.trace.pc, entry.trace.decoded.raw);
            } else {
                cache_.set_next_store_cmd(line_addr, entry.warp_id,
                                          entry.issue_cycle,
                                          entry.trace.pc,
                                          entry.trace.decoded.raw);
            }
        }
        cmd_in_flight_.set_next(true);
    }
}

void CoalescingUnit::commit() {
    // Phase 3 (close-the-Reg-family-migration): the addr-gen FIFO pop is
    // applied by TimingModel's dedicated ungated cross-stage commit pass,
    // not here. CoalescingUnit::commit no longer mutates the FIFO directly;
    // the pop intent was already staged on the RegFifo in evaluate().
    // Phase 6 of current_mut() elimination: flip the wrapped Regs
    // (processing_, current_entry_, is_coalesced_, serial_index_,
    // cmd_in_flight_) in one sweep.
    commit_all();
}

void CoalescingUnit::reset() {
    reset_all();
    // The cross-stage addr-gen FIFO is owned by TimingModel and reset by
    // TimingModel; do not clear it here.
}

} // namespace gpu_sim
