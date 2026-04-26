// Adversarial tests for same-line MSHR merging in the L1 cache.
//
// Covers spec §5.3.1 (Same-line merging, Secondary MSHR wake path, Line
// pinning), §5.3.2 (write-buffer cross-ref), and §5.4 (RAW correctness).
//
// These tests target the post-fix implementation where `complete_fill` defers
// when the target set is pinned with a different tag.

#include "catch.hpp"
#include "gpu_sim/timing/cache.h"
#include "gpu_sim/timing/load_gather_buffer.h"
#include "gpu_sim/timing/memory_interface.h"
#include "gpu_sim/stats.h"

using namespace gpu_sim;

namespace {

constexpr uint32_t CACHE_SIZE = 4096;
constexpr uint32_t LINE_SIZE = 128;
constexpr uint32_t NUM_MSHRS = 4;
constexpr uint32_t WB_DEPTH = 4;
constexpr uint32_t NUM_WARPS = 4;
constexpr uint32_t MEM_LATENCY = 10;
constexpr uint32_t FULL_MASK = 0xFFFFFFFFu;
constexpr uint32_t NUM_SETS = CACHE_SIZE / LINE_SIZE;

struct MergeFixture {
    Stats stats;
    FixedLatencyMemory mem_if{MEM_LATENCY, stats};
    LoadGatherBufferFile gather_file{NUM_WARPS, stats};
    L1Cache cache{CACHE_SIZE, LINE_SIZE, NUM_MSHRS, WB_DEPTH, mem_if, gather_file, stats};

    void tick_mem(uint32_t cycles) {
        for (uint32_t i = 0; i < cycles; ++i) {
            mem_if.evaluate();
        }
    }

    void end_cycle() {
        cache.commit();
        gather_file.commit();
    }

    void claim(uint32_t warp_id, uint8_t dest_reg = 1) {
        gather_file.claim(warp_id, dest_reg, /*pc=*/0, /*issue_cycle=*/0,
                          /*raw_instruction=*/0);
    }
};

std::array<uint32_t, WARP_SIZE> make_results(uint32_t base = 0) {
    std::array<uint32_t, WARP_SIZE> r{};
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        r[i] = base + i;
    }
    return r;
}

} // namespace

// ----------------------------------------------------------------------------
// Case 1: RAW across outstanding store miss -- spec §5.3.1, §5.4
// ----------------------------------------------------------------------------
TEST_CASE("MSHR merging: load secondary behind store-miss primary serializes (RAW)",
          "[cache][mshr][merging]") {
    MergeFixture f;
    auto results = make_results(7000);

    // Store miss to line L=0: allocates a primary, submits exactly one read.
    REQUIRE(f.cache.process_store(/*line_addr=*/0, /*warp=*/0, 1, 0, 0));
    REQUIRE(f.stats.store_misses == 1);
    REQUIRE(f.stats.external_memory_reads == 1);
    REQUIRE(f.cache.active_mshr_count() == 1);

    // Load to the same line before the fill: allocates as secondary.
    f.claim(/*warp=*/0, /*dest_reg=*/5);
    REQUIRE(f.cache.process_load(/*addr=*/0, /*warp=*/0, FULL_MASK, results, 2, 0, 0));
    REQUIRE(f.cache.active_mshr_count() == 2);
    REQUIRE(f.stats.external_memory_reads == 1); // no second external read
    REQUIRE(f.stats.mshr_merged_loads == 1);
    REQUIRE(f.stats.load_misses == 1);
    REQUIRE(f.cache.last_miss_event().valid);
    REQUIRE(f.cache.last_miss_event().merged_secondary);

    // Deliver the fill: primary retires, tag installed and pinned.
    f.tick_mem(MEM_LATENCY);
    f.cache.evaluate(); // handle_responses + drain_secondary_chain_head
    REQUIRE(f.cache.last_fill_event().valid);
    REQUIRE(f.cache.last_fill_event().is_store == true);
    REQUIRE(f.cache.last_fill_event().chain_length_at_fill == 2);
    // Within one evaluate(): handle_responses retires the store primary (which
    // does not consume the gather extraction port), then
    // drain_secondary_chain_head drains the load secondary in the same cycle.
    // Both complete atomically from the observer's view.
    REQUIRE(f.gather_file.has_result());
    REQUIRE(f.stats.secondary_drain_cycles == 1);
    // After both retired, MSHR count is 0 and pin is cleared.
    REQUIRE(f.cache.active_mshr_count() == 0);

    // Consume writeback, then a subsequent load should HIT (pin is clear).
    WritebackEntry wb = f.gather_file.consume_result();
    REQUIRE(wb.dest_reg == 5);
    REQUIRE(wb.values[0] == 7000);
    f.end_cycle();

    f.claim(0, 6);
    auto r2 = make_results(9000);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, r2, 3, 0, 0));
    REQUIRE(f.stats.load_hits == 1); // pin cleared, hit proceeds
}

// ----------------------------------------------------------------------------
// Case 2: FIFO chain of three (store A -> load B -> store C) same line/warp.
// ----------------------------------------------------------------------------
TEST_CASE("MSHR merging: FIFO chain store-load-store drains in program order",
          "[cache][mshr][merging]") {
    MergeFixture f;
    auto results = make_results(1234);

    // Store A (primary).
    REQUIRE(f.cache.process_store(0, 0, 1, 0, 0));
    // Load B (secondary).
    f.claim(0, 3);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 2, 0, 0));
    // Store C (secondary).
    REQUIRE(f.cache.process_store(0, 0, 3, 0, 0));

    REQUIRE(f.stats.external_memory_reads == 1);
    REQUIRE(f.stats.mshr_merged_loads == 1);
    REQUIRE(f.stats.mshr_merged_stores == 1);
    REQUIRE(f.cache.active_mshr_count() == 3);

    // Deliver fill.
    f.tick_mem(MEM_LATENCY);

    // Cycle N: primary store fill retires, pushes to write buffer, pin set.
    size_t wb_before = f.cache.write_buffer_size();
    f.cache.evaluate();
    REQUIRE(f.cache.last_fill_event().is_store);
    REQUIRE(f.cache.last_fill_event().chain_length_at_fill == 3);
    // Store primary pushes one wb entry; store-fill does NOT use gather
    // extract port, so load-secondary B can drain in the same evaluate().
    REQUIRE(f.cache.write_buffer_size() == wb_before + 1);
    // Load secondary drained in the same cycle (store fill did not claim the
    // extraction port), so B is written into warp 0's gather buffer.
    REQUIRE(f.gather_file.has_result());
    REQUIRE(f.stats.secondary_drain_cycles == 1);
    // Chain still has store C waiting, so pin remains.
    REQUIRE(f.cache.active_mshr_count() == 1);

    // Consume writeback for B (dest_reg 3).
    WritebackEntry wbB = f.gather_file.consume_result();
    REQUIRE(wbB.dest_reg == 3);
    f.end_cycle();

    // Next cycle: drain store C.
    f.cache.evaluate();
    REQUIRE(f.stats.secondary_drain_cycles == 2);
    REQUIRE(f.cache.active_mshr_count() == 0);
    // Pin must now be cleared: a miss to a different tag mapping to set 0
    // should not see LINE_PINNED stall.
    f.end_cycle();
    auto rX = make_results();
    uint32_t conflict_line = NUM_SETS; // different tag, same set 0
    REQUIRE(f.cache.process_store(conflict_line, 0, 10, 0, 0));
    REQUIRE(f.cache.stall_reason() != CacheStallReason::LINE_PINNED);
}

// ----------------------------------------------------------------------------
// Case 3: Cross-warp same-line merging.
// ----------------------------------------------------------------------------
TEST_CASE("MSHR merging: cross-warp same-line loads share one external read",
          "[cache][mshr][merging]") {
    MergeFixture f;
    auto r0 = make_results(100);
    auto r1 = make_results(200);

    f.claim(0, 4);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, r0, 1, 0, 0));
    f.claim(1, 9);
    REQUIRE(f.cache.process_load(0, 1, FULL_MASK, r1, 2, 0, 0));

    REQUIRE(f.stats.external_memory_reads == 1);
    REQUIRE(f.stats.mshr_merged_loads == 1);
    REQUIRE(f.cache.active_mshr_count() == 2);

    f.tick_mem(MEM_LATENCY);
    // Cycle: primary load fill -> warp 0 gets values; FILL marks extraction
    // port used, so the warp-1 secondary defers.
    f.cache.evaluate();
    REQUIRE(f.gather_file.has_result()); // warp 0 ready
    REQUIRE(f.cache.active_mshr_count() == 1);
    // Consume warp 0 writeback.
    WritebackEntry wb0 = f.gather_file.consume_result();
    REQUIRE(wb0.warp_id == 0);
    REQUIRE(wb0.dest_reg == 4);
    REQUIRE(wb0.values[5] == 105);
    f.end_cycle();

    // Next cycle: secondary drains into warp 1.
    f.cache.evaluate();
    REQUIRE(f.gather_file.has_result());
    WritebackEntry wb1 = f.gather_file.consume_result();
    REQUIRE(wb1.warp_id == 1);
    REQUIRE(wb1.dest_reg == 9);
    REQUIRE(wb1.values[5] == 205);
    REQUIRE(f.cache.active_mshr_count() == 0);
}

// ----------------------------------------------------------------------------
// Case 4: Same-set different-tag pinning -- spec §5.3.1 Line pinning.
// ----------------------------------------------------------------------------
TEST_CASE("MSHR merging: different-tag miss into pinned set stalls LINE_PINNED",
          "[cache][mshr][merging][pinning]") {
    MergeFixture f;
    auto r = make_results();

    // Build a chain on line L=0: primary load + secondary load.
    f.claim(0, 1);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, r, 1, 0, 0));
    f.claim(1, 2);
    REQUIRE(f.cache.process_load(0, 1, FULL_MASK, r, 2, 0, 0));

    // Deliver primary. Pin is established on set 0.
    f.tick_mem(MEM_LATENCY);
    f.cache.evaluate();
    // Warp 0 got its fill; warp 1 secondary deferred (port conflict).
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Before draining warp 1, inject a different-tag miss to set 0.
    uint32_t conflict_line = NUM_SETS; // same set, different tag
    auto misses_before = f.stats.load_misses;
    auto cache_misses_before = f.stats.cache_misses;
    auto pin_stalls_before = f.stats.line_pin_stall_cycles;

    f.claim(2, 3);
    REQUIRE_FALSE(f.cache.process_load(conflict_line * LINE_SIZE, 2, FULL_MASK,
                                       r, 3, 0, 0));
    REQUIRE(f.cache.stall_reason() == CacheStallReason::LINE_PINNED);
    REQUIRE(f.stats.line_pin_stall_cycles == pin_stalls_before + 1);
    // Rejection path: load_misses and cache_misses must NOT increment.
    REQUIRE(f.stats.load_misses == misses_before);
    REQUIRE(f.stats.cache_misses == cache_misses_before);
    // The rejected request's claim on warp 2's gather buffer is still held,
    // but no MSHR was allocated for it.
    // Active MSHRs = just the warp-1 secondary.
    REQUIRE(f.cache.active_mshr_count() == 1);

    // Drain the secondary -> pin clears.
    f.cache.evaluate();
    // Warp 1 gets its values.
    REQUIRE(f.gather_file.has_result());
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Now the conflict request can proceed. Reclaim first (warp 2's buffer is
    // still busy with its earlier claim, so we just reissue -- but that buffer
    // is busy. Use a fresh warp to keep the test simple.)
    f.claim(3, 4);
    REQUIRE(f.cache.process_load(conflict_line * LINE_SIZE, 3, FULL_MASK, r, 4, 0, 0));
    REQUIRE(f.stats.load_misses == misses_before + 1);
    REQUIRE(f.cache.stall_reason() != CacheStallReason::LINE_PINNED);
}

// ----------------------------------------------------------------------------
// Case 5: In-flight primary fill vs. pinned line on same set.
//         Post-fix regression test (complete_fill must DEFER).
// ----------------------------------------------------------------------------
TEST_CASE("MSHR merging: store-fill defers when target set pinned by other line",
          "[cache][mshr][merging][pinning][regression]") {
    MergeFixture f;
    auto r = make_results(5000);

    // P1: load miss on line L=0 (set 0).
    f.claim(0, 1);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, r, 1, 0, 0));
    // S1: load secondary on same line L=0, different warp.
    f.claim(1, 2);
    REQUIRE(f.cache.process_load(0, 1, FULL_MASK, r, 2, 0, 0));
    // P2: load miss on line L' (different tag, same set 0). Allocated BEFORE
    // P1's fill -> at this point set 0 is not yet valid/pinned, so P2 is
    // accepted and becomes a primary (not in L's chain).
    uint32_t Lprime_line = NUM_SETS;
    f.claim(2, 3);
    REQUIRE(f.cache.process_load(Lprime_line * LINE_SIZE, 2, FULL_MASK, r, 3, 0, 0));

    REQUIRE(f.stats.external_memory_reads == 2); // P1 and P2 only
    REQUIRE(f.cache.active_mshr_count() == 3);

    // Deliver fills. P1 response arrives first (FIFO in mem_if).
    f.tick_mem(MEM_LATENCY);

    // Cycle A: P1 fill retires, installs L on set 0, pins it. S1 secondary
    // defers (FILL took extraction port). P2 fill response is queued but not
    // yet dequeued (handle_responses processes only one per cycle).
    f.cache.evaluate();
    REQUIRE(f.gather_file.has_result()); // warp 0 got its data
    (void)f.gather_file.consume_result();
    f.end_cycle();
    REQUIRE(f.cache.active_mshr_count() == 2); // S1 and P2 remain

    // Cycle B: handle_responses dequeues P2's response. complete_fill must
    // DEFER because set 0 is pinned with tag(L). The fill stays in
    // pending_fill_. Meanwhile drain_secondary_chain_head drains S1 -> warp 1
    // gets its data, pin clears.
    auto pin_stall_before = f.stats.line_pin_stall_cycles;
    f.cache.evaluate();
    REQUIRE(f.stats.line_pin_stall_cycles == pin_stall_before + 1);
    REQUIRE(f.cache.pending_fill().valid); // P2 still deferred
    REQUIRE(f.gather_file.has_result()); // warp 1 drained
    WritebackEntry wb1 = f.gather_file.consume_result();
    REQUIRE(wb1.warp_id == 1);
    f.end_cycle();
    // S1 freed; P2 still pending.
    REQUIRE(f.cache.active_mshr_count() == 1);

    // Cycle C: pin is now clear; pending P2 fill can complete.
    f.cache.evaluate();
    REQUIRE_FALSE(f.cache.pending_fill().valid);
    REQUIRE(f.cache.active_mshr_count() == 0);
    REQUIRE(f.gather_file.has_result()); // warp 2 gets its data
    WritebackEntry wb2 = f.gather_file.consume_result();
    REQUIRE(wb2.warp_id == 2);
    REQUIRE(wb2.dest_reg == 3);
    f.end_cycle();

    // L' is now installed on set 0. Re-reading it should hit.
    f.claim(0, 5);
    auto r2 = make_results();
    REQUIRE(f.cache.process_load(Lprime_line * LINE_SIZE, 0, FULL_MASK, r2, 10, 0, 0));
    REQUIRE(f.stats.load_hits >= 1);

    // No duplication of external reads.
    REQUIRE(f.stats.external_memory_reads == 2);
}

// ----------------------------------------------------------------------------
// Case 6: Write-buffer full backpressure on secondary store drain.
//         Spec §5.3.2: write-buffer-full stalls apply to secondary drains; a
//         stalled secondary drain keeps its line pinned.
// ----------------------------------------------------------------------------
TEST_CASE("MSHR merging: secondary-store drain stalls on write-buffer full",
          "[cache][mshr][merging][writebuffer]") {
    MergeFixture f;
    auto r = make_results();

    // Install line 0 so we can pre-fill the write buffer with hit stores.
    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, r, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Fill write buffer to WB_DEPTH-1 with hit stores (leave one slot free so
    // the primary store-miss fill will succeed but its secondary cannot).
    for (uint32_t i = 0; i < WB_DEPTH - 1; ++i) {
        REQUIRE(f.cache.process_store(0, 0, 2 + i, 0, 0));
    }
    REQUIRE(f.cache.write_buffer_size() == WB_DEPTH - 1);

    // Primary store miss to a new line (set 1).
    REQUIRE(f.cache.process_store(1, 0, 20, 0, 0));
    // Secondary store to same new line -> becomes secondary.
    REQUIRE(f.cache.process_store(1, 0, 21, 0, 0));
    REQUIRE(f.stats.mshr_merged_stores == 1);

    f.tick_mem(MEM_LATENCY);
    // Cycle: primary retires -> pushes to WB (now full), pin set. Secondary
    // wants to drain but WB is full -> stalls, pin remains.
    f.cache.evaluate();
    REQUIRE(f.cache.write_buffer_size() == WB_DEPTH);
    REQUIRE(f.cache.active_mshr_count() == 1); // secondary still there
    auto drains_before = f.stats.secondary_drain_cycles;

    f.end_cycle();
    // Try again with WB still full -> still stalls.
    f.cache.evaluate();
    REQUIRE(f.stats.secondary_drain_cycles == drains_before);
    REQUIRE(f.cache.active_mshr_count() == 1);
    f.end_cycle();

    // Drain one WB entry.
    f.cache.drain_write_buffer();
    REQUIRE(f.cache.write_buffer_size() == WB_DEPTH - 1);

    // Next evaluate: secondary drains.
    f.cache.evaluate();
    REQUIRE(f.stats.secondary_drain_cycles == drains_before + 1);
    REQUIRE(f.cache.active_mshr_count() == 0);
    REQUIRE(f.cache.write_buffer_size() == WB_DEPTH);
}

// ----------------------------------------------------------------------------
// Case 7: Gather-port contention -- FILL beats secondary load drain.
// ----------------------------------------------------------------------------
TEST_CASE("MSHR merging: FILL wins gather-extract port over secondary drain",
          "[cache][mshr][merging][port]") {
    MergeFixture f;
    auto rA = make_results(0);
    auto rB = make_results(1000);

    // Chain for line 0: primary load (warp 0) + secondary load (warp 1).
    f.claim(0, 1);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, rA, 1, 0, 0));
    f.claim(1, 2);
    REQUIRE(f.cache.process_load(0, 1, FULL_MASK, rA, 2, 0, 0));

    // Independent primary load on a different line (set 1).
    f.claim(2, 3);
    REQUIRE(f.cache.process_load(LINE_SIZE, 2, FULL_MASK, rB, 3, 0, 0));

    REQUIRE(f.stats.external_memory_reads == 2);
    REQUIRE(f.cache.active_mshr_count() == 3);

    f.tick_mem(MEM_LATENCY);

    // Cycle A: P1 (line 0) fill retires into warp 0's gather. The FILL
    // claims the gather-extract port (LoadGatherBufferFile.next_port_claimed_
    // = true), so the warp-1 secondary defers.
    auto drains_before = f.stats.secondary_drain_cycles;
    f.cache.evaluate();
    REQUIRE(f.stats.secondary_drain_cycles == drains_before); // no drain
    REQUIRE(f.gather_file.has_result()); // warp 0 filled
    WritebackEntry wb0 = f.gather_file.consume_result();
    REQUIRE(wb0.warp_id == 0);
    f.end_cycle();

    // Cycle B: next handle_responses dequeues the line-1 P2 fill. That fill
    // goes to warp 2's gather buffer and consumes the extraction port again.
    // Warp-1 secondary is still waiting; must defer another cycle.
    f.cache.evaluate();
    REQUIRE(f.stats.secondary_drain_cycles == drains_before); // still 0
    REQUIRE(f.gather_file.has_result()); // warp 2 got line 1
    WritebackEntry wb2 = f.gather_file.consume_result();
    REQUIRE(wb2.warp_id == 2);
    f.end_cycle();

    // Cycle C: no more FILLs pending -> secondary finally drains.
    f.cache.evaluate();
    REQUIRE(f.stats.secondary_drain_cycles == drains_before + 1);
    REQUIRE(f.gather_file.has_result());
    WritebackEntry wb1 = f.gather_file.consume_result();
    REQUIRE(wb1.warp_id == 1);
    REQUIRE(f.cache.active_mshr_count() == 0);
}

// ----------------------------------------------------------------------------
// Case 8: MSHR exhaustion via secondaries -- one external read, 3 merged.
// ----------------------------------------------------------------------------
TEST_CASE("MSHR merging: exhaustion via 4 same-line misses stalls a new line",
          "[cache][mshr][merging][exhaustion]") {
    MergeFixture f;
    auto r = make_results();

    // 4 same-line loads across 4 warps: 1 primary + 3 secondaries.
    for (uint32_t w = 0; w < NUM_MSHRS; ++w) {
        f.claim(w, static_cast<uint8_t>(w + 1));
        REQUIRE(f.cache.process_load(0, w, FULL_MASK, r, 1 + w, 0, 0));
    }
    REQUIRE(f.stats.external_memory_reads == 1);
    REQUIRE(f.stats.mshr_merged_loads == 3);
    REQUIRE(f.cache.active_mshr_count() == NUM_MSHRS);

    // A 5th miss to ANY line stalls MSHR_FULL. Don't claim (MSHRs full -> the
    // request is rejected before gather-buffer touch).
    auto stalls_before = f.stats.mshr_stall_cycles;
    auto misses_before = f.stats.load_misses;
    REQUIRE_FALSE(f.cache.process_load(2 * LINE_SIZE, 0, FULL_MASK, r, 10, 0, 0));
    REQUIRE(f.cache.stall_reason() == CacheStallReason::MSHR_FULL);
    REQUIRE(f.stats.mshr_stall_cycles == stalls_before + 1);
    // Rejection must NOT bump load_misses: the miss counter only advances on
    // accepted requests, otherwise a single miss waiting N cycles for a free
    // MSHR would be counted N+1 times.
    REQUIRE(f.stats.load_misses == misses_before);

    // Deliver fill. Primary retires + one secondary drains in same cycle
    // (FILL + secondary on different buffers -> but the gather-extract port
    // is shared at the LoadGatherBufferFile level; secondary defers). Either
    // way, at least the primary retires.
    f.tick_mem(MEM_LATENCY);
    f.cache.evaluate();
    REQUIRE(f.cache.active_mshr_count() <= 3);
    // Drain all pending results until gather file empty.
    while (f.gather_file.has_result()) {
        (void)f.gather_file.consume_result();
    }
    f.end_cycle();

    // Now a new primary on a different line should allocate (there is at
    // least one free MSHR slot).
    REQUIRE(f.cache.mshrs().num_entries() > f.cache.active_mshr_count());
    f.claim(0, 10);
    // Wait -- warp 0's gather buffer was just consumed, so we can claim again.
    auto reads_before = f.stats.external_memory_reads;
    REQUIRE(f.cache.process_load(2 * LINE_SIZE, 0, FULL_MASK, r, 20, 0, 0));
    REQUIRE(f.stats.external_memory_reads == reads_before + 1);
}

// ----------------------------------------------------------------------------
// Case 9: Reset clears chain state.
// ----------------------------------------------------------------------------
TEST_CASE("MSHR merging: reset() clears pinned tags, MSHRs, and port state",
          "[cache][mshr][merging][reset]") {
    MergeFixture f;
    auto r = make_results();

    // Build a chain: primary + 2 secondaries on line 0.
    f.claim(0, 1);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, r, 1, 0, 0));
    f.claim(1, 2);
    REQUIRE(f.cache.process_load(0, 1, FULL_MASK, r, 2, 0, 0));
    f.claim(2, 3);
    REQUIRE(f.cache.process_load(0, 2, FULL_MASK, r, 3, 0, 0));
    REQUIRE(f.cache.active_mshr_count() == 3);
    REQUIRE(f.cache.write_buffer_size() == 0);

    // Deliver fill to partially advance state (primary fills, pin set).
    f.tick_mem(MEM_LATENCY);
    f.cache.evaluate();
    // Reset.
    f.cache.reset();
    f.mem_if.reset();
    f.gather_file.reset();

    // Post-reset: no MSHRs valid, no write buffer entries, no pending fill.
    REQUIRE(f.cache.active_mshr_count() == 0);
    REQUIRE(f.cache.write_buffer_size() == 0);
    REQUIRE_FALSE(f.cache.pending_fill().valid);

    // A fresh miss to the previously-pinned line should allocate as a new
    // primary (not see a stale pin / not be a secondary).
    f.claim(0, 1);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, r, 1, 0, 0));
    REQUIRE(f.stats.external_memory_reads >= 1);
    REQUIRE_FALSE(f.cache.last_miss_event().merged_secondary);
    REQUIRE(f.cache.active_mshr_count() == 1);
}

// ----------------------------------------------------------------------------
// Case 10: Secondary-load scoreboard/writeback end-to-end.
//          Two different warps load the same line with different dest regs;
//          both must produce distinct writebacks carrying the correct rd.
// ----------------------------------------------------------------------------
TEST_CASE("MSHR merging: primary and secondary loads produce distinct writebacks",
          "[cache][mshr][merging][writeback]") {
    MergeFixture f;
    auto r = make_results(4242);

    // Warp A (primary) dest_reg 11, warp B (secondary) dest_reg 22.
    f.claim(0, 11);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, r, 1, 0, 0));
    f.claim(1, 22);
    REQUIRE(f.cache.process_load(0, 1, FULL_MASK, r, 2, 0, 0));

    f.tick_mem(MEM_LATENCY);

    // Cycle 1: primary fill -> warp 0 writeback ready with dest_reg 11.
    f.cache.evaluate();
    REQUIRE(f.gather_file.has_result());
    WritebackEntry wbA = f.gather_file.consume_result();
    REQUIRE(wbA.valid);
    REQUIRE(wbA.warp_id == 0);
    REQUIRE(wbA.dest_reg == 11);
    REQUIRE(wbA.values[0] == 4242);
    REQUIRE(wbA.values[31] == 4242 + 31);
    REQUIRE(wbA.source_unit == ExecUnit::LDST);
    f.end_cycle();

    // Cycle 2: secondary drains -> warp 1 writeback ready with dest_reg 22.
    f.cache.evaluate();
    REQUIRE(f.gather_file.has_result());
    WritebackEntry wbB = f.gather_file.consume_result();
    REQUIRE(wbB.valid);
    REQUIRE(wbB.warp_id == 1);
    REQUIRE(wbB.dest_reg == 22);
    REQUIRE(wbB.values[0] == 4242);
    REQUIRE(wbB.source_unit == ExecUnit::LDST);

    // Both writebacks carry correct rd -> consumer (scoreboard) will clear
    // the per-warp pending bit on its rd correctly.
    REQUIRE(wbA.dest_reg != wbB.dest_reg);
    REQUIRE(wbA.warp_id != wbB.warp_id);
}

// ----------------------------------------------------------------------------
// Case 11: Same-warp RAW timing -- the secondary load's writeback must not be
//          observable before the primary's external fill completes, even
//          though the functional model already has the value in `results` at
//          issue time. This probes the property that an in-order, same-warp
//          store-then-load to the same line incurs the full miss latency of
//          the primary, rather than being shortcut by the pre-populated
//          functional result leaking into the gather buffer early.
// ----------------------------------------------------------------------------
TEST_CASE("MSHR merging: same-warp RAW load timing respects primary fill latency",
          "[cache][mshr][merging][timing][raw]") {
    MergeFixture f;
    auto load_results = make_results(5000);

    // Cycle 0: store miss (primary) and subsequent load (secondary), same warp.
    REQUIRE(f.cache.process_store(/*line_addr=*/0, /*warp=*/0, /*issue_cycle=*/1, 0, 0));
    REQUIRE(f.cache.active_mshr_count() == 1);
    REQUIRE(f.stats.external_memory_reads == 1);

    f.claim(/*warp=*/0, /*dest_reg=*/7);
    REQUIRE(f.cache.process_load(/*addr=*/0, /*warp=*/0, FULL_MASK, load_results,
                                 /*issue_cycle=*/2, 0, 0));
    REQUIRE(f.stats.mshr_merged_loads == 1);
    REQUIRE(f.stats.external_memory_reads == 1); // no second external fetch
    REQUIRE(f.cache.active_mshr_count() == 2);
    REQUIRE_FALSE(f.gather_file.has_result());

    // Cycles 1 .. MEM_LATENCY-1: fill still in flight. Secondary must not
    // drain, no writeback must appear, tag must not become pinned, and no
    // fill event must be observed. If the functional value leaked through
    // the gather buffer early, one of these would fail.
    for (uint32_t k = 1; k < MEM_LATENCY; ++k) {
        f.mem_if.evaluate();
        f.cache.evaluate();
        REQUIRE_FALSE(f.cache.last_fill_event().valid);
        REQUIRE_FALSE(f.gather_file.has_result());
        REQUIRE(f.cache.active_mshr_count() == 2);
        REQUIRE(f.stats.secondary_drain_cycles == 0);
        f.end_cycle();
    }

    // Cycle MEM_LATENCY: fill response now ready. One cache.evaluate() retires
    // the store primary (WB push) and drains the load secondary in the same
    // call (store fill does not claim the gather-extract port). The writeback
    // appears this cycle -- not before.
    f.mem_if.evaluate();
    f.cache.evaluate();
    REQUIRE(f.cache.last_fill_event().valid);
    REQUIRE(f.cache.last_fill_event().is_store);
    REQUIRE(f.cache.last_fill_event().chain_length_at_fill == 2);
    REQUIRE(f.stats.secondary_drain_cycles == 1);
    REQUIRE(f.cache.active_mshr_count() == 0);

    REQUIRE(f.gather_file.has_result());
    WritebackEntry wb = f.gather_file.consume_result();
    REQUIRE(wb.warp_id == 0);
    REQUIRE(wb.dest_reg == 7);
    REQUIRE(wb.values[0] == 5000);
    REQUIRE(wb.values[31] == 5000 + 31);
    f.end_cycle();

    REQUIRE_FALSE(f.gather_file.has_result());
    REQUIRE(f.stats.external_memory_reads == 1);
}

// ----------------------------------------------------------------------------
// Case 12: Arbitration edge -- secondary drain beats HIT on the cache
//          gather-extract port when no FILL is present.
//          Spec §5.3 "Port model": FILL > secondary drain > HIT. A HIT issued
//          in the same cycle a secondary drain uses the single extraction port
//          must be rejected and retry next cycle, REGARDLESS of whether the
//          HIT targets the same warp's gather buffer as the drain or a
//          different one. This is the "secondary > HIT" edge (the dual of
//          Case 7, which covers FILL > secondary).
//
//          Self-check: if the cache-level gather_extract_port gate were
//          removed, the HIT would succeed this cycle -> load_hits would
//          increment and process_load would return true. The assertions below
//          fail in that case.
// ----------------------------------------------------------------------------
TEST_CASE("MSHR merging: secondary drain wins gather-extract port over HIT",
          "[cache][mshr][merging][port]") {
    MergeFixture f;
    auto r = make_results(3000);
    auto r_hit = make_results(8000);

    // Step 1: Pre-install a DIFFERENT line L1 (set 1) as resident. Bring it
    // in via a plain load miss+fill, consume the writeback so the warp-0
    // gather buffer is free for reuse later. After this, line L1 is resident
    // and its chain is empty (no pin).
    f.claim(/*warp=*/0, /*dest_reg=*/9);
    REQUIRE(f.cache.process_load(LINE_SIZE, /*warp=*/0, FULL_MASK, r_hit, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.evaluate();
    REQUIRE(f.gather_file.has_result());
    (void)f.gather_file.consume_result();
    f.end_cycle();
    REQUIRE(f.cache.active_mshr_count() == 0);

    // Step 2: Build a primary+secondary load chain on line L0 (set 0) between
    // two DIFFERENT warps, so warp-0's gather buffer stays free for the hit.
    f.claim(/*warp=*/1, /*dest_reg=*/4);
    REQUIRE(f.cache.process_load(/*addr=*/0, /*warp=*/1, FULL_MASK, r, 2, 0, 0));
    f.claim(/*warp=*/2, /*dest_reg=*/5);
    REQUIRE(f.cache.process_load(/*addr=*/0, /*warp=*/2, FULL_MASK, r, 3, 0, 0));
    REQUIRE(f.cache.active_mshr_count() == 2);
    REQUIRE(f.stats.external_memory_reads == 2); // L1 + L0

    // Step 3: Deliver the L0 fill. Cycle A retires the primary into warp 1's
    // buffer (FILL consumes the port), warp-2 secondary defers. Consume warp 1.
    f.tick_mem(MEM_LATENCY);
    f.cache.evaluate();
    REQUIRE(f.gather_file.has_result());
    WritebackEntry wb1 = f.gather_file.consume_result();
    REQUIRE(wb1.warp_id == 1);
    f.end_cycle();
    // Warp-2 secondary still pending; set 0 still pinned.
    REQUIRE(f.cache.active_mshr_count() == 1);
    REQUIRE(f.cache.pinned_line_count() == 1);

    // Step 4: Cycle B. No pending fill (handle_responses is empty). The
    // drain_secondary_chain_head path will drain warp-2's secondary and claim
    // the single cache gather-extract port. In the SAME cycle, issue a hit to
    // the pre-installed line L1 on warp 0 (a different buffer). Per spec,
    // secondary beats HIT on the shared extraction port -> HIT must be
    // rejected this cycle.
    auto drains_before = f.stats.secondary_drain_cycles;
    auto hits_before = f.stats.load_hits;
    f.cache.evaluate();
    // The drain took the port this cycle.
    REQUIRE(f.stats.secondary_drain_cycles == drains_before + 1);
    // Warp 2 is about to get its writeback. Do NOT consume yet.
    REQUIRE(f.gather_file.has_result());

    // Now issue the HIT on a DIFFERENT warp (warp 0, whose buffer is free).
    // The hit targets line L1 which IS resident, so tag matches -> normally
    // a hit. But the cache gather-extract port is already used by the
    // secondary drain, so process_load must return false and NOT bump
    // load_hits.
    f.claim(/*warp=*/0, /*dest_reg=*/7);
    auto r_hit2 = make_results(9000);
    bool hit_accepted = f.cache.process_load(LINE_SIZE, /*warp=*/0, FULL_MASK,
                                             r_hit2, /*issue_cycle=*/99, 0, 0);
    REQUIRE_FALSE(hit_accepted);
    REQUIRE(f.stats.load_hits == hits_before);
    // The loser's side effects must be absent: no writeback should be
    // produced into warp 0's buffer this cycle. has_result() may still be
    // true from warp 2's drained secondary, but warp 0 must not be ready.
    // Consume warp 2's drained result first; then assert warp 0 is NOT ready.
    WritebackEntry wb2 = f.gather_file.consume_result();
    REQUIRE(wb2.warp_id == 2);
    REQUIRE_FALSE(f.gather_file.has_result());
    f.end_cycle();

    // Step 5: Next cycle, the HIT retries. Port reset, no drain pending, no
    // fill pending -> hit succeeds.
    f.cache.evaluate();
    bool hit_accepted2 = f.cache.process_load(LINE_SIZE, /*warp=*/0, FULL_MASK,
                                              r_hit2, /*issue_cycle=*/100, 0, 0);
    REQUIRE(hit_accepted2);
    REQUIRE(f.stats.load_hits == hits_before + 1);
    REQUIRE(f.gather_file.has_result());
    WritebackEntry wb0 = f.gather_file.consume_result();
    REQUIRE(wb0.warp_id == 0);
    REQUIRE(wb0.dest_reg == 7);
    REQUIRE(wb0.values[0] == 9000);
    f.end_cycle();
}

// ----------------------------------------------------------------------------
// Case 13: Companion to Case 6 -- secondary-store drain stalls on write-
//          buffer full, but unblock via the NATURAL drain path (WB entries
//          draining to external memory through mem_if cycles) rather than
//          via the direct drain_write_buffer() sidechannel being called at
//          test scope alone. This asserts the same §5.3.2 invariants end-
//          to-end through the public cache+mem_if surface: during WB-full
//          stall cycles `secondary_drain_cycles` does not increment, the
//          pin stays set, the MSHR stays allocated; once the WB naturally
//          drains, the secondary drain proceeds.
//
//          Self-check: if the WB-full gate in drain_secondary_chain_head
//          were removed, the secondary would retire in the first cycle
//          after primary fill even while the WB is full -- the assertions
//          on `secondary_drain_cycles` and `active_mshr_count` during the
//          stall window would fail.
// ----------------------------------------------------------------------------
TEST_CASE("MSHR merging: secondary-store drain WB-full stall unblocks via natural drain",
          "[cache][mshr][merging][writebuffer]") {
    MergeFixture f;
    auto r = make_results();

    // Install line 0 so we can pre-fill the write buffer with hit stores.
    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, r, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Fill write buffer to WB_DEPTH-1 with hit stores.
    for (uint32_t i = 0; i < WB_DEPTH - 1; ++i) {
        REQUIRE(f.cache.process_store(0, 0, 2 + i, 0, 0));
    }
    REQUIRE(f.cache.write_buffer_size() == WB_DEPTH - 1);

    // Primary store miss to a new line on set 1 + secondary store to it.
    REQUIRE(f.cache.process_store(1, 0, 20, 0, 0));
    REQUIRE(f.cache.process_store(1, 0, 21, 0, 0));
    REQUIRE(f.stats.mshr_merged_stores == 1);

    f.tick_mem(MEM_LATENCY);

    // Cycle: primary retires -> WB now full; secondary stalls, pin remains.
    f.cache.evaluate();
    REQUIRE(f.cache.write_buffer_size() == WB_DEPTH);
    REQUIRE(f.cache.active_mshr_count() == 1);
    REQUIRE(f.cache.pinned_line_count() == 1);
    auto drains_before = f.stats.secondary_drain_cycles;
    f.end_cycle();

    // Natural drain path: each timing-model tick calls drain_write_buffer()
    // after mem_if.evaluate(). Here we simulate that cadence: tick mem_if
    // (advances in-flight writes toward completion) and call
    // drain_write_buffer() once per "cycle" -- this is NOT a test-only
    // sidechannel, it's exactly what TimingModel::evaluate does in
    // timing_model.cpp L328. Each iteration submits one WB entry to mem_if
    // and frees a WB slot.
    //
    // During this process, call cache.evaluate() each cycle to let the
    // secondary drain retry. Assert that while WB is full at cycle start,
    // secondary_drain_cycles does NOT increment and the MSHR stays live.
    // The secondary must only retire on a cycle where WB started with a
    // free slot.

    // Cycle 1: at start, WB size == WB_DEPTH (full). evaluate() sees full
    // WB -> drain stalls. Then drain_write_buffer() submits one entry and
    // frees a slot.
    REQUIRE(f.cache.write_buffer_size() == WB_DEPTH);
    f.cache.evaluate();
    REQUIRE(f.stats.secondary_drain_cycles == drains_before); // stalled
    REQUIRE(f.cache.active_mshr_count() == 1);                // still pending
    REQUIRE(f.cache.pinned_line_count() == 1);                // pin intact
    f.mem_if.evaluate();
    f.cache.drain_write_buffer();
    REQUIRE(f.cache.write_buffer_size() == WB_DEPTH - 1);     // one freed
    f.end_cycle();

    // Cycle 2: WB started with a free slot; secondary drain succeeds.
    f.cache.evaluate();
    REQUIRE(f.stats.secondary_drain_cycles == drains_before + 1);
    REQUIRE(f.cache.active_mshr_count() == 0); // secondary retired
    REQUIRE(f.cache.pinned_line_count() == 0); // pin cleared
    // WB grew by 1 (secondary pushed its line) but also loses any mem_if
    // completion -- we just assert it's within depth.
    REQUIRE(f.cache.write_buffer_size() <= WB_DEPTH);
    f.end_cycle();
}
