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
constexpr uint32_t MAX_OUTSTANDING_WRITES = 32;
constexpr uint32_t NUM_WARPS = 4;
constexpr uint32_t MEM_LATENCY = 10;
constexpr uint32_t FULL_MASK = 0xFFFFFFFFu;
constexpr uint32_t NUM_SETS = CACHE_SIZE / LINE_SIZE;

struct MergeFixture {
    Stats stats;
    FixedLatencyMemory mem_if{MEM_LATENCY, stats};
    LoadGatherBufferFile gather_file{NUM_WARPS, stats};
    L1Cache cache{CACHE_SIZE, LINE_SIZE, NUM_MSHRS, WB_DEPTH, MAX_OUTSTANDING_WRITES,
                  mem_if, gather_file, stats};

    // Phase M5: commit() flips next_*_request_ → current_*_request_; the
    // first evaluate() drains current_ into in_flight_. Without this, a
    // request staged by process_load/process_store stays in next_ and
    // never reaches in_flight_, so no fill ever returns.
    //
    // registered-mshr-write-buffer: an MSHR allocated by a preceding
    // process_load/process_store lives in next_entries_ until cache.commit().
    // tick_mem is the cycle boundary after the issuing access, so commit the
    // cache here too — this latches the MSHR/write-buffer staging into the
    // committed state that complete_fill reads. Idempotent after an explicit
    // cache.commit() at the same boundary.
    void tick_mem(uint32_t cycles) {
        cache.commit();
        mem_if.commit();
        for (uint32_t i = 0; i < cycles; ++i) {
            mem_if.evaluate();
        }
    }

    void end_cycle() {
        cache.commit();
        gather_file.commit();
    }

    // Phase M2: claim is REGISTERED. Drive commit + evaluate so the
    // claim is visible to subsequent operations in the same test step.
    void claim(uint32_t warp_id, uint8_t dest_reg = 1) {
        gather_file.claim(warp_id, dest_reg, /*pc=*/0, /*issue_cycle=*/0,
                          /*raw_instruction=*/0);
        gather_file.commit();
        gather_file.evaluate();
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
    // Commit so the primary MSHR lands in current_entries_: the secondary's
    // find_chain_tail and active_mshr_count() both scan committed state.
    f.cache.commit();
    REQUIRE(f.cache.active_mshr_count() == 1);

    // Load to the same line before the fill: allocates as secondary.
    f.claim(/*warp=*/0, /*dest_reg=*/5);
    REQUIRE(f.cache.process_load(/*addr=*/0, /*warp=*/0, FULL_MASK, results, 2, 0, 0));
    f.cache.commit();
    REQUIRE(f.cache.active_mshr_count() == 2);
    REQUIRE(f.stats.external_memory_reads == 1); // no second external read
    REQUIRE(f.stats.mshr_merged_loads == 1);
    REQUIRE(f.stats.load_misses == 1);
    f.cache.commit(); // observe REGISTERED scratch state
    REQUIRE(f.cache.current_last_miss_event().valid);
    REQUIRE(f.cache.current_last_miss_event().merged_secondary);

    // Deliver the fill: primary retires, tag installed and pinned.
    f.tick_mem(MEM_LATENCY);
    f.cache.evaluate(); // handle_responses retires the store primary
    f.cache.commit();
    REQUIRE(f.cache.current_last_fill_event().valid);
    REQUIRE(f.cache.current_last_fill_event().is_store == true);
    REQUIRE(f.cache.current_last_fill_event().chain_length_at_fill == 2);
    // registered-mshr-write-buffer: the primary's MSHR free lands in
    // next_entries_, so within the fill cycle drain_secondary_chain_head
    // still sees the primary valid in current_entries_ and treats the
    // secondary as non-head. The secondary therefore drains the FOLLOWING
    // cycle, once the primary's free has been committed.
    f.cache.evaluate(); // drain_secondary_chain_head drains the load secondary
    f.cache.commit();
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result());
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

    // Store A (primary). commit between chain accesses so each later
    // allocation's find_chain_tail scans the committed current_entries_.
    REQUIRE(f.cache.process_store(0, 0, 1, 0, 0));
    f.cache.commit();
    // Load B (secondary).
    f.claim(0, 3);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 2, 0, 0));
    f.cache.commit();
    // Store C (secondary).
    REQUIRE(f.cache.process_store(0, 0, 3, 0, 0));
    f.cache.commit();

    REQUIRE(f.stats.external_memory_reads == 1);
    REQUIRE(f.stats.mshr_merged_loads == 1);
    REQUIRE(f.stats.mshr_merged_stores == 1);
    REQUIRE(f.cache.active_mshr_count() == 3);

    // Deliver fill.
    f.tick_mem(MEM_LATENCY);

    // Cycle N: primary store fill retires, pushes to write buffer, pin set.
    size_t wb_before = f.cache.write_buffer_size();
    f.cache.evaluate();
    f.cache.commit();
    REQUIRE(f.cache.current_last_fill_event().is_store);
    REQUIRE(f.cache.current_last_fill_event().chain_length_at_fill == 3);
    // Store primary pushes one wb entry.
    REQUIRE(f.cache.write_buffer_size() == wb_before + 1);

    // Cycle N+1: registered-mshr-write-buffer — drain_secondary_chain_head
    // treats B as the chain head only once the primary A's free is committed,
    // so the load secondary B drains the cycle AFTER the primary fill.
    f.cache.evaluate();
    f.cache.commit();
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result());
    REQUIRE(f.stats.secondary_drain_cycles == 1);
    // Chain still has store C waiting, so pin remains.
    REQUIRE(f.cache.active_mshr_count() == 1);

    // Consume writeback for B (dest_reg 3).
    WritebackEntry wbB = f.gather_file.consume_result();
    REQUIRE(wbB.dest_reg == 3);
    f.end_cycle();

    // Cycle N+2: drain store C — head only after B's free committed. The
    // drain frees C's MSHR into next_entries_; commit before
    // active_mshr_count() reads current_entries_.
    f.cache.evaluate();
    f.cache.commit();
    REQUIRE(f.stats.secondary_drain_cycles == 2);
    REQUIRE(f.cache.active_mshr_count() == 0);
    f.gather_file.commit();

    // Plan 2 (write-ack pinning): store A's fill and store C's drain each
    // queued a write-through for line 0, so set 0 stays write-ack-pinned
    // even after the chain pin clears. Drain those write-throughs and let
    // their write acks be consumed so the pin fully releases.
    for (uint32_t i = 0; i < 200 && !f.cache.is_idle(); ++i) {
        f.cache.evaluate();
        f.mem_if.evaluate();
        f.cache.drain_write_buffer();
        f.cache.commit();
        f.mem_if.commit();
    }
    REQUIRE(f.cache.is_idle());

    // Both pins now clear: a miss to a different tag mapping to set 0 must
    // not see a LINE_PINNED stall.
    uint32_t conflict_line = NUM_SETS; // different tag, same set 0
    REQUIRE(f.cache.process_store(conflict_line, 0, 10, 0, 0));
    f.cache.commit();
    REQUIRE(f.cache.next_stall_reason() != CacheStallReason::LINE_PINNED);
}

// ----------------------------------------------------------------------------
// Case 3: Cross-warp same-line merging.
// ----------------------------------------------------------------------------
TEST_CASE("MSHR merging: cross-warp same-line loads share one external read",
          "[cache][mshr][merging]") {
    MergeFixture f;
    auto r0 = make_results(100);
    auto r1 = make_results(200);

    // commit between the two accesses so the secondary's find_chain_tail
    // scans the committed primary in current_entries_.
    f.claim(0, 4);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, r0, 1, 0, 0));
    f.cache.commit();
    f.claim(1, 9);
    REQUIRE(f.cache.process_load(0, 1, FULL_MASK, r1, 2, 0, 0));
    f.cache.commit();

    REQUIRE(f.stats.external_memory_reads == 1);
    REQUIRE(f.stats.mshr_merged_loads == 1);
    REQUIRE(f.cache.active_mshr_count() == 2);

    f.tick_mem(MEM_LATENCY);
    // Cycle: primary load fill -> warp 0 gets values; FILL marks extraction
    // port used, so the warp-1 secondary defers.
    f.cache.evaluate();
    f.cache.commit();   // latch the fill's MSHR free into current_entries_
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result()); // warp 0 ready
    REQUIRE(f.cache.active_mshr_count() == 1);
    // Consume warp 0 writeback.
    WritebackEntry wb0 = f.gather_file.consume_result();
    REQUIRE(wb0.warp_id == 0);
    REQUIRE(wb0.dest_reg == 4);
    REQUIRE(wb0.values[5] == 105);
    f.end_cycle();

    // Next cycle: secondary drains into warp 1.
    f.cache.evaluate();
    f.cache.commit();   // latch the secondary drain's MSHR free
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result());
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

    // Build a chain on line L=0: primary load + secondary load. commit
    // between so the secondary's find_chain_tail sees the committed primary.
    f.claim(0, 1);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, r, 1, 0, 0));
    f.cache.commit();
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
    f.cache.commit();
    REQUIRE(f.cache.next_stall_reason() == CacheStallReason::LINE_PINNED);
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
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result());
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Now the conflict request can proceed. Reclaim first (warp 2's buffer is
    // still busy with its earlier claim, so we just reissue -- but that buffer
    // is busy. Use a fresh warp to keep the test simple.)
    f.claim(3, 4);
    REQUIRE(f.cache.process_load(conflict_line * LINE_SIZE, 3, FULL_MASK, r, 4, 0, 0));
    REQUIRE(f.stats.load_misses == misses_before + 1);
    f.cache.commit();
    REQUIRE(f.cache.next_stall_reason() != CacheStallReason::LINE_PINNED);
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
    // Phase M5: drain P1's staged read into in_flight_ before issuing
    // a second primary on a different line. Without this commit +
    // evaluate, P2's set_next_read_request would overwrite P1's
    // staging slot and only P2 would reach memory.
    // registered-mshr-write-buffer: also commit the cache so P1's MSHR is in
    // current_entries_ — S1's find_chain_tail and P2's allocate both scan it.
    f.cache.commit();
    f.mem_if.commit();
    f.mem_if.evaluate();
    // S1: load secondary on same line L=0, different warp.
    f.claim(1, 2);
    REQUIRE(f.cache.process_load(0, 1, FULL_MASK, r, 2, 0, 0));
    // Commit so S1's MSHR is committed before P2's allocate picks a slot
    // (otherwise P2 would reuse S1's not-yet-committed next_entries_ slot).
    f.cache.commit();
    // P2: load miss on line L' (different tag, same set 0). Allocated BEFORE
    // P1's fill -> at this point set 0 is not yet valid/pinned, so P2 is
    // accepted and becomes a primary (not in L's chain).
    uint32_t Lprime_line = NUM_SETS;
    f.claim(2, 3);
    REQUIRE(f.cache.process_load(Lprime_line * LINE_SIZE, 2, FULL_MASK, r, 3, 0, 0));
    f.cache.commit();

    REQUIRE(f.stats.external_memory_reads == 2); // P1 and P2 only
    REQUIRE(f.cache.active_mshr_count() == 3);

    // Deliver fills. P1 response arrives first (FIFO in mem_if). P1 was
    // drained one cycle ahead of P2 (the inline commit+evaluate above);
    // tick_mem(MEM_LATENCY) covers both: P1 completes at evaluate #9,
    // P2 at evaluate #10. Both responses are queued in arrival order.
    f.tick_mem(MEM_LATENCY);

    // Cycle A: P1 fill retires, installs L on set 0, pins it. S1 secondary
    // defers (FILL took extraction port). P2 fill response is queued but not
    // yet dequeued (handle_responses processes only one per cycle).
    f.cache.evaluate();
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result()); // warp 0 got its data
    (void)f.gather_file.consume_result();
    f.end_cycle();
    REQUIRE(f.cache.active_mshr_count() == 2); // S1 and P2 remain

    // Cycle B: handle_responses dequeues P2's response. complete_fill must
    // DEFER because set 0 is pinned with tag(L). The fill stays in
    // pending_fill_. Meanwhile drain_secondary_chain_head drains S1 -> warp 1
    // gets its data, pin clears.
    auto pin_stall_before = f.stats.line_pin_stall_cycles;
    f.cache.evaluate();
    f.cache.commit();
    REQUIRE(f.stats.line_pin_stall_cycles == pin_stall_before + 1);
    REQUIRE(f.cache.current_pending_fill().valid); // P2 still deferred
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result()); // warp 1 drained
    WritebackEntry wb1 = f.gather_file.consume_result();
    REQUIRE(wb1.warp_id == 1);
    f.end_cycle();
    // S1 freed; P2 still pending.
    REQUIRE(f.cache.active_mshr_count() == 1);

    // Cycle C: pin is now clear; pending P2 fill can complete.
    f.cache.evaluate();
    f.cache.commit();
    REQUIRE_FALSE(f.cache.current_pending_fill().valid);
    REQUIRE(f.cache.active_mshr_count() == 0);
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result()); // warp 2 gets its data
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
    // Single enqueue port per cycle: commit between stores so each staged
    // write-through enqueue lands before the next store.
    for (uint32_t i = 0; i < WB_DEPTH - 1; ++i) {
        REQUIRE(f.cache.process_store(0, 0, 2 + i, 0, 0));
        f.cache.commit();
    }
    REQUIRE(f.cache.write_buffer_size() == WB_DEPTH - 1);

    // Primary store miss to a new line (set 1). commit between primary and
    // secondary so the secondary's find_chain_tail sees the committed primary.
    REQUIRE(f.cache.process_store(1, 0, 20, 0, 0));
    f.cache.commit();
    // Secondary store to same new line -> becomes secondary.
    REQUIRE(f.cache.process_store(1, 0, 21, 0, 0));
    f.cache.commit();
    REQUIRE(f.stats.mshr_merged_stores == 1);

    f.tick_mem(MEM_LATENCY);
    // Cycle: primary retires -> pushes to WB (now full), pin set. Secondary
    // wants to drain but WB is full -> stalls, pin remains.
    f.cache.evaluate();
    f.cache.commit();   // apply the fill's staged WB push + MSHR free
    REQUIRE(f.cache.write_buffer_size() == WB_DEPTH);
    REQUIRE(f.cache.active_mshr_count() == 1); // secondary still there
    auto drains_before = f.stats.secondary_drain_cycles;

    f.gather_file.commit();
    // Try again with WB still full -> still stalls.
    f.cache.evaluate();
    f.cache.commit();
    REQUIRE(f.stats.secondary_drain_cycles == drains_before);
    REQUIRE(f.cache.active_mshr_count() == 1);
    f.gather_file.commit();

    // Drain one WB entry. drain_write_buffer() only STAGES the pop; commit
    // applies it before write_buffer_size() observes the freed slot and
    // before the secondary drain retry sees room in the buffer.
    f.cache.drain_write_buffer();
    f.cache.commit();
    REQUIRE(f.cache.write_buffer_size() == WB_DEPTH - 1);

    // Next evaluate: secondary drains. commit applies its staged WB push and
    // MSHR free before they are observed.
    f.cache.evaluate();
    f.cache.commit();
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
    // Phase M5: drain P1 (line 0) into in_flight_ before issuing P3 on
    // a different line below; otherwise P3's set_next_read_request would
    // overwrite P1's staging slot.
    // registered-mshr-write-buffer: also commit the cache so P1's MSHR is in
    // current_entries_ for the secondary's find_chain_tail / P3's allocate.
    f.cache.commit();
    f.mem_if.commit();
    f.mem_if.evaluate();
    f.claim(1, 2);
    REQUIRE(f.cache.process_load(0, 1, FULL_MASK, rA, 2, 0, 0));
    f.cache.commit();

    // Independent primary load on a different line (set 1).
    f.claim(2, 3);
    REQUIRE(f.cache.process_load(LINE_SIZE, 2, FULL_MASK, rB, 3, 0, 0));
    f.cache.commit();

    REQUIRE(f.stats.external_memory_reads == 2);
    REQUIRE(f.cache.active_mshr_count() == 3);

    // tick_mem(MEM_LATENCY) covers both fills: P1 already advanced one
    // cycle, so P1 completes at evaluate #9 and P3 at evaluate #10.
    f.tick_mem(MEM_LATENCY);

    // Cycle A: P1 (line 0) fill retires into warp 0's gather. The FILL
    // claims the gather-extract port (LoadGatherBufferFile.next_port_claimed_
    // = true), so the warp-1 secondary defers.
    auto drains_before = f.stats.secondary_drain_cycles;
    f.cache.evaluate();
    REQUIRE(f.stats.secondary_drain_cycles == drains_before); // no drain
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result()); // warp 0 filled
    WritebackEntry wb0 = f.gather_file.consume_result();
    REQUIRE(wb0.warp_id == 0);
    f.end_cycle();

    // Cycle B: next handle_responses dequeues the line-1 P2 fill. That fill
    // goes to warp 2's gather buffer and consumes the extraction port again.
    // Warp-1 secondary is still waiting; must defer another cycle.
    f.cache.evaluate();
    REQUIRE(f.stats.secondary_drain_cycles == drains_before); // still 0
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result()); // warp 2 got line 1
    WritebackEntry wb2 = f.gather_file.consume_result();
    REQUIRE(wb2.warp_id == 2);
    f.end_cycle();

    // Cycle C: no more FILLs pending -> secondary finally drains.
    f.cache.evaluate();
    f.cache.commit();   // latch the secondary drain's MSHR free
    REQUIRE(f.stats.secondary_drain_cycles == drains_before + 1);
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result());
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

    // 4 same-line loads across 4 warps: 1 primary + 3 secondaries. commit
    // between accesses so each later allocation's find_chain_tail / allocate
    // scans the committed current_entries_ (else each becomes a fresh
    // primary, allocating a duplicate external read).
    for (uint32_t w = 0; w < NUM_MSHRS; ++w) {
        f.claim(w, static_cast<uint8_t>(w + 1));
        REQUIRE(f.cache.process_load(0, w, FULL_MASK, r, 1 + w, 0, 0));
        f.cache.commit();
    }
    REQUIRE(f.stats.external_memory_reads == 1);
    REQUIRE(f.stats.mshr_merged_loads == 3);
    REQUIRE(f.cache.active_mshr_count() == NUM_MSHRS);

    // A 5th miss to ANY line stalls MSHR_FULL. Don't claim (MSHRs full -> the
    // request is rejected before gather-buffer touch).
    auto stalls_before = f.stats.mshr_stall_cycles;
    auto misses_before = f.stats.load_misses;
    REQUIRE_FALSE(f.cache.process_load(2 * LINE_SIZE, 0, FULL_MASK, r, 10, 0, 0));
    f.cache.commit();
    REQUIRE(f.cache.next_stall_reason() == CacheStallReason::MSHR_FULL);
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
    f.cache.commit();   // latch the fill's primary MSHR free
    REQUIRE(f.cache.active_mshr_count() <= 3);
    // Drain all pending results until gather file empty.
    f.gather_file.commit();
    while (f.gather_file.current_has_result()) {
        (void)f.gather_file.consume_result();
        f.gather_file.commit();  // Phase M4: re-latch after consume
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

    // Build a chain: primary + 2 secondaries on line 0. commit between
    // accesses so each secondary's find_chain_tail sees the committed chain.
    f.claim(0, 1);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, r, 1, 0, 0));
    f.cache.commit();
    f.claim(1, 2);
    REQUIRE(f.cache.process_load(0, 1, FULL_MASK, r, 2, 0, 0));
    f.cache.commit();
    f.claim(2, 3);
    REQUIRE(f.cache.process_load(0, 2, FULL_MASK, r, 3, 0, 0));
    f.cache.commit();
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
    REQUIRE_FALSE(f.cache.current_pending_fill().valid);

    // A fresh miss to the previously-pinned line should allocate as a new
    // primary (not see a stale pin / not be a secondary).
    f.claim(0, 1);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, r, 1, 0, 0));
    REQUIRE(f.stats.external_memory_reads >= 1);
    f.cache.commit();
    REQUIRE_FALSE(f.cache.current_last_miss_event().merged_secondary);
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

    // Warp A (primary) dest_reg 11, warp B (secondary) dest_reg 22. commit
    // between so the secondary's find_chain_tail sees the committed primary.
    f.claim(0, 11);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, r, 1, 0, 0));
    f.cache.commit();
    f.claim(1, 22);
    REQUIRE(f.cache.process_load(0, 1, FULL_MASK, r, 2, 0, 0));

    f.tick_mem(MEM_LATENCY);

    // Cycle 1: primary fill -> warp 0 writeback ready with dest_reg 11.
    f.cache.evaluate();
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result());
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
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result());
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
    // commit between so the primary is committed for the secondary's
    // find_chain_tail and for active_mshr_count() reads.
    REQUIRE(f.cache.process_store(/*line_addr=*/0, /*warp=*/0, /*issue_cycle=*/1, 0, 0));
    f.cache.commit();
    REQUIRE(f.cache.active_mshr_count() == 1);
    REQUIRE(f.stats.external_memory_reads == 1);

    f.claim(/*warp=*/0, /*dest_reg=*/7);
    REQUIRE(f.cache.process_load(/*addr=*/0, /*warp=*/0, FULL_MASK, load_results,
                                 /*issue_cycle=*/2, 0, 0));
    f.cache.commit();
    REQUIRE(f.stats.mshr_merged_loads == 1);
    REQUIRE(f.stats.external_memory_reads == 1); // no second external fetch
    REQUIRE(f.cache.active_mshr_count() == 2);
    f.gather_file.commit();
    REQUIRE_FALSE(f.gather_file.current_has_result());
    // Phase M5: flip the staged store-miss read into mem_if's
    // current_read_request_ so the per-cycle evaluate loop below
    // drains it into in_flight_ on the first iteration.
    f.mem_if.commit();

    // Cycles 1 .. MEM_LATENCY-1: fill still in flight. Secondary must not
    // drain, no writeback must appear, tag must not become pinned, and no
    // fill event must be observed. If the functional value leaked through
    // the gather buffer early, one of these would fail.
    for (uint32_t k = 1; k < MEM_LATENCY; ++k) {
        f.mem_if.evaluate();
        f.cache.evaluate();
        REQUIRE_FALSE(f.cache.current_last_fill_event().valid);
        f.gather_file.commit();
    REQUIRE_FALSE(f.gather_file.current_has_result());
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
    f.cache.commit();
    REQUIRE(f.cache.current_last_fill_event().valid);
    REQUIRE(f.cache.current_last_fill_event().is_store);
    REQUIRE(f.cache.current_last_fill_event().chain_length_at_fill == 2);

    // registered-mshr-write-buffer: the store primary's MSHR free lands in
    // next_entries_, so the load secondary is treated as the chain head and
    // drains only the FOLLOWING cycle, once the free is committed.
    f.cache.evaluate();
    f.cache.commit();
    REQUIRE(f.stats.secondary_drain_cycles == 1);
    REQUIRE(f.cache.active_mshr_count() == 0);

    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result());
    WritebackEntry wb = f.gather_file.consume_result();
    REQUIRE(wb.warp_id == 0);
    REQUIRE(wb.dest_reg == 7);
    REQUIRE(wb.values[0] == 5000);
    REQUIRE(wb.values[31] == 5000 + 31);
    f.end_cycle();

    f.gather_file.commit();
    REQUIRE_FALSE(f.gather_file.current_has_result());
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
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result());
    (void)f.gather_file.consume_result();
    f.end_cycle();
    REQUIRE(f.cache.active_mshr_count() == 0);

    // Step 2: Build a primary+secondary load chain on line L0 (set 0) between
    // two DIFFERENT warps, so warp-0's gather buffer stays free for the hit.
    // commit between so the secondary's find_chain_tail sees the primary.
    f.claim(/*warp=*/1, /*dest_reg=*/4);
    REQUIRE(f.cache.process_load(/*addr=*/0, /*warp=*/1, FULL_MASK, r, 2, 0, 0));
    f.cache.commit();
    f.claim(/*warp=*/2, /*dest_reg=*/5);
    REQUIRE(f.cache.process_load(/*addr=*/0, /*warp=*/2, FULL_MASK, r, 3, 0, 0));
    f.cache.commit();
    REQUIRE(f.cache.active_mshr_count() == 2);
    REQUIRE(f.stats.external_memory_reads == 2); // L1 + L0

    // Step 3: Deliver the L0 fill. Cycle A retires the primary into warp 1's
    // buffer (FILL consumes the port), warp-2 secondary defers. Consume warp 1.
    f.tick_mem(MEM_LATENCY);
    f.cache.evaluate();
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result());
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
    // Phase M2: pre-claim warp 0's buffer for the upcoming HIT before
    // cache.evaluate runs. The helper's commit + evaluate would otherwise
    // clear the cache's port-claim flag mid-cycle, masking the arbitration
    // we want to observe. Pre-claiming sets buf[0].busy ahead of the
    // secondary drain so the HIT's process_load is what tests the port.
    f.claim(/*warp=*/0, /*dest_reg=*/7);
    f.cache.evaluate();
    // The drain took the port this cycle.
    REQUIRE(f.stats.secondary_drain_cycles == drains_before + 1);

    // Now issue the HIT on a DIFFERENT warp (warp 0, whose buffer is free).
    // The hit targets line L1 which IS resident, so tag matches -> normally
    // a hit. But the cache gather-extract port is already used by the
    // secondary drain, so process_load must return false and NOT bump
    // load_hits. Phase M4: avoid gather_file.commit() before the HIT —
    // commit() clears next_port_claimed_ mid-cycle and would mask the
    // arbitration we want to observe. Defer the has_result check until
    // after the HIT.
    auto r_hit2 = make_results(9000);
    bool hit_accepted = f.cache.process_load(LINE_SIZE, /*warp=*/0, FULL_MASK,
                                             r_hit2, /*issue_cycle=*/99, 0, 0);
    REQUIRE_FALSE(hit_accepted);
    REQUIRE(f.stats.load_hits == hits_before);
    // After end_cycle, gather_file.commit latches has_result and the
    // drained warp-2 writeback is observable.
    f.end_cycle();
    REQUIRE(f.gather_file.current_has_result()); // warp 2 was drained
    WritebackEntry wb2 = f.gather_file.consume_result();
    REQUIRE(wb2.warp_id == 2);
    f.gather_file.commit();
    REQUIRE_FALSE(f.gather_file.current_has_result());

    // Step 5: Next cycle, the HIT retries. Port reset, no drain pending, no
    // fill pending -> hit succeeds.
    f.cache.evaluate();
    bool hit_accepted2 = f.cache.process_load(LINE_SIZE, /*warp=*/0, FULL_MASK,
                                              r_hit2, /*issue_cycle=*/100, 0, 0);
    REQUIRE(hit_accepted2);
    REQUIRE(f.stats.load_hits == hits_before + 1);
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result());
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

    // Fill write buffer to WB_DEPTH-1 with hit stores. Single enqueue port
    // per cycle: commit between stores so each staged enqueue lands.
    for (uint32_t i = 0; i < WB_DEPTH - 1; ++i) {
        REQUIRE(f.cache.process_store(0, 0, 2 + i, 0, 0));
        f.cache.commit();
    }
    REQUIRE(f.cache.write_buffer_size() == WB_DEPTH - 1);

    // Primary store miss to a new line on set 1 + secondary store to it.
    // commit between so the secondary's find_chain_tail sees the primary.
    REQUIRE(f.cache.process_store(1, 0, 20, 0, 0));
    f.cache.commit();
    REQUIRE(f.cache.process_store(1, 0, 21, 0, 0));
    f.cache.commit();
    REQUIRE(f.stats.mshr_merged_stores == 1);

    f.tick_mem(MEM_LATENCY);

    // Cycle: primary retires -> WB now full; secondary stalls, pin remains.
    f.cache.evaluate();
    f.cache.commit();   // apply the fill's staged WB push + MSHR free
    REQUIRE(f.cache.write_buffer_size() == WB_DEPTH);
    REQUIRE(f.cache.active_mshr_count() == 1);
    auto drains_before = f.stats.secondary_drain_cycles;
    f.gather_file.commit();
    // Registered tag array: the pin installed by complete_fill this cycle
    // is observable via pinned_line_count() (a current_tags_ reader) only
    // after commit() — end_cycle() flips next_tags_ → current_tags_.
    REQUIRE(f.cache.pinned_line_count() == 1);

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
    // End of cycle 1: commit applies the secondary-drain stall (no-op), the
    // staged write-buffer pop from drain_write_buffer(), and the mem request.
    f.cache.commit();
    f.mem_if.commit();
    f.gather_file.commit();
    // registered-mshr-write-buffer: the pop is applied now, so the freed slot
    // is observable.
    REQUIRE(f.cache.write_buffer_size() == WB_DEPTH - 1);     // one freed

    // Cycle 2: WB started with a free slot; secondary drain succeeds.
    f.cache.evaluate();
    f.cache.commit();   // apply the secondary drain's MSHR free + WB push
    REQUIRE(f.stats.secondary_drain_cycles == drains_before + 1);
    REQUIRE(f.cache.active_mshr_count() == 0); // secondary retired
    // WB grew by 1 (secondary pushed its line) but also loses any mem_if
    // completion -- we just assert it's within depth.
    REQUIRE(f.cache.write_buffer_size() <= WB_DEPTH);
    f.end_cycle();
    // Registered tag array: drain_secondary_chain_head clears the pin into
    // next_tags_; pinned_line_count() (current_tags_) sees it after commit.
    REQUIRE(f.cache.pinned_line_count() == 0); // pin cleared
}

// ----------------------------------------------------------------------------
// Case 14: registered-mshr-write-buffer Part A — direct MSHRFile tests.
//          The MSHR file is double-buffered: allocate/free mutate
//          next_entries_, readers scan current_entries_, commit() flips.
// ----------------------------------------------------------------------------
TEST_CASE("MSHRFile: a slot freed this cycle is not reusable until next cycle",
          "[cache][mshr]") {
    MSHRFile mshrs(2);
    MSHREntry e;
    e.cache_line_addr = 100;

    int i0 = mshrs.allocate(e);
    REQUIRE(i0 == 0);
    mshrs.commit();
    REQUIRE(mshrs.has_active());

    // Free slot 0 this cycle (clears next_entries_[0]).
    mshrs.seed_next();
    mshrs.free(0);
    // allocate() scans current_entries_, where slot 0 still reads valid — so
    // a same-cycle allocation must NOT reuse slot 0.
    int i1 = mshrs.allocate(e);
    REQUIRE(i1 == 1);
    mshrs.commit();

    // Next cycle: slot 0's free is committed, so it is now reusable.
    mshrs.seed_next();
    REQUIRE(mshrs.has_free());
    int i2 = mshrs.allocate(e);
    REQUIRE(i2 == 0);
}

TEST_CASE("MSHRFile: a chain tail freed this cycle stays visible until commit",
          "[cache][mshr]") {
    MSHRFile mshrs(3);
    MSHREntry p;
    p.cache_line_addr = 50;

    int pi = mshrs.allocate(p);
    REQUIRE(pi == 0);
    mshrs.commit();
    REQUIRE(mshrs.find_chain_tail(50) == 0);

    // Free the primary this cycle.
    mshrs.seed_next();
    mshrs.free(0);
    // find_chain_tail scans current_entries_ — the just-freed primary is
    // still a valid chain tail this cycle. (registered-tag-array.md Step 4's
    // universal fill-conflict retry is what keeps this from being a hazard:
    // no command reaches find_chain_tail in a cycle a primary is freed.)
    REQUIRE(mshrs.find_chain_tail(50) == 0);
    mshrs.commit();

    // Next cycle: the free is committed; the line has no chain tail.
    mshrs.seed_next();
    REQUIRE(mshrs.find_chain_tail(50) == -1);
    REQUIRE(mshrs.has_free());
}
