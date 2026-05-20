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

// Common fixture: wire a cache, memory interface, and gather-buffer file
// against a shared Stats object. Each test constructs one of these.
struct CacheFixture {
    Stats stats;
    FixedLatencyMemory mem_if{MEM_LATENCY, stats};
    LoadGatherBufferFile gather_file{NUM_WARPS, stats};
    L1Cache cache{CACHE_SIZE, LINE_SIZE, NUM_MSHRS, WB_DEPTH, MAX_OUTSTANDING_WRITES,
                  mem_if, gather_file, stats};

    // Advance the external memory interface by N cycles. Phase M5: the
    // request staged via set_next_*_request needs commit() to flip
    // next_ → current_ before the first evaluate() drains it into
    // in_flight_. Insert that commit at the head so callers can keep
    // the pre-Phase-M5 process_load/process_store + tick_mem(MEM_LATENCY)
    // pattern.
    //
    // registered-mshr-write-buffer: an MSHR allocated by a preceding
    // process_load/process_store lives in next_entries_ until cache.commit().
    // tick_mem is always the cycle boundary after the issuing access, so
    // commit the cache here too — this latches the MSHR into current_entries_
    // before the fill's complete_fill reads it via current_at(). The commit
    // is idempotent after an explicit cache.commit() at the same boundary.
    void tick_mem(uint32_t cycles) {
        cache.commit();
        mem_if.commit();
        for (uint32_t i = 0; i < cycles; ++i) {
            mem_if.evaluate();
            // Phase 4 of current_mut() elimination: each loop iteration is
            // one tick, so commit() must follow evaluate() to advance
            // PulseReg<PendingMemoryRequest> slots. Otherwise current_.valid
            // stays true across iterations and evaluate re-submits the same
            // request every cycle (the old shape relied on a mid-cycle
            // current_mut() clear).
            mem_if.commit();
        }
    }

    // Tick the gather buffer's per-cycle commit (clears the shared
    // next_port_claimed_ flag at end-of-cycle).
    void end_cycle() {
        cache.commit();
        gather_file.commit();
    }

    // One full cache+memory tick: cache.evaluate() (incl. handle_responses,
    // which consumes one write ack), memory advances, one write-buffer entry
    // drains to memory, then all stages commit. Used to drain write-throughs
    // and let their write acks release the write-ack pin.
    void pump() {
        cache.evaluate();
        mem_if.evaluate();
        cache.drain_write_buffer();
        cache.commit();
        mem_if.commit();
        gather_file.commit();
    }

    // Claim the gather buffer for a load on behalf of `warp_id`. The
    // coalescing unit normally does this before the first cache access.
    // Phase M2: the claim is REGISTERED. To keep test ergonomics close to
    // pre-M2 (claim takes effect by the time this helper returns), drive
    // the commit + evaluate phases that move the staged request from
    // next_ → current_ and apply the buffer mutation in-place.
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

TEST_CASE("Cache: coalesced load hit populates all 32 gather slots", "[cache]") {
    CacheFixture f;
    auto results = make_results(1000);

    // Install a line via a coalesced load miss + fill.
    f.claim(/*warp_id=*/0);
    REQUIRE(f.cache.process_load(256, 0, FULL_MASK, results, 1, 0, 0));
    REQUIRE(f.stats.load_misses == 1);
    REQUIRE_FALSE(f.gather_file.current_has_result()); // still waiting on fill

    // The MSHR allocated by process_load lands in next_entries_; commit it so
    // complete_fill (via handle_responses) reads it from current_entries_.
    f.cache.commit();
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    // Phase M4: has_result is REGISTERED; the just-completed FILL staged
    // next_has_result_, but a commit is required to latch it into
    // current_has_result_ where the arbiter (and tests) observe it.
    f.gather_file.commit();
    // Fill should have populated all 32 slots via the gather buffer.
    REQUIRE(f.gather_file.current_has_result());
    WritebackEntry wb = f.gather_file.consume_result();
    REQUIRE(wb.valid);
    REQUIRE(wb.warp_id == 0);
    REQUIRE(wb.values[0] == 1000);
    REQUIRE(wb.values[31] == 1031);
    f.end_cycle();

    // Second coalesced load to the same line: hit path writes all 32 slots
    // into the gather buffer in one cycle.
    f.claim(/*warp_id=*/0, /*dest_reg=*/7);
    auto results2 = make_results(5000);
    REQUIRE(f.cache.process_load(256, 0, FULL_MASK, results2, 2, 0, 0));
    REQUIRE(f.stats.load_hits == 1);
    // Phase M4: latch the staged has_result via commit before observing.
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result()); // all 32 slots filled in one cycle
    REQUIRE(f.gather_file.buffer(0).filled_count == WARP_SIZE);
    WritebackEntry wb2 = f.gather_file.consume_result();
    REQUIRE(wb2.dest_reg == 7);
    REQUIRE(wb2.values[0] == 5000);
    REQUIRE(wb2.values[31] == 5031);
}

TEST_CASE("Cache: load miss allocates single MSHR and fill lands in gather buffer",
          "[cache]") {
    CacheFixture f;
    auto results = make_results(42);

    f.claim(/*warp_id=*/0, /*dest_reg=*/5);
    REQUIRE(f.cache.process_load(256, 0, FULL_MASK, results, 1, 0, 0));
    REQUIRE(f.stats.load_misses == 1);
    REQUIRE(f.stats.external_memory_reads == 1);
    REQUIRE_FALSE(f.gather_file.current_has_result());

    // Commit so the just-allocated MSHR is visible in current_entries_ to
    // active_mshr_count() and to complete_fill below.
    f.cache.commit();
    REQUIRE(f.cache.active_mshr_count() == 1);

    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    // Phase M4: latch staged has_result via commit before observing.
    f.gather_file.commit();
    // The fill staged the MSHR free into next_entries_; commit applies it
    // before active_mshr_count() reads current_entries_.
    f.cache.commit();

    // Fill releases the MSHR and deposits the full line into the gather buffer.
    REQUIRE(f.cache.active_mshr_count() == 0);
    REQUIRE(f.gather_file.current_has_result());
    WritebackEntry wb = f.gather_file.consume_result();
    REQUIRE(wb.dest_reg == 5);
    REQUIRE(wb.values[0] == 42);
}

TEST_CASE("Cache: store hit pushes line into write buffer", "[cache]") {
    CacheFixture f;
    auto results = make_results();

    // Install a line first.
    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    // Phase 10D: commit() latches the fill into committed buffer state so
    // consume_result() (a pure committed read) sees it and stages the
    // release; end_cycle() then applies the staged release.
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Store to the same line hits. Stores never touch the gather buffer.
    REQUIRE(f.cache.process_store(0, 0, 2, 0, 0));
    REQUIRE(f.stats.store_hits == 1);
    // The write-through is staged into the registered write buffer; commit
    // applies the enqueue before write_buffer_size() observes it.
    f.cache.commit();
    REQUIRE(f.cache.write_buffer_size() == 1);
    REQUIRE_FALSE(f.gather_file.current_busy(0));
}

TEST_CASE("Cache: store miss allocates MSHR (write-allocate)", "[cache]") {
    CacheFixture f;

    REQUIRE(f.cache.process_store(1, 0, 1, 0, 0));
    REQUIRE(f.stats.store_misses == 1);
    REQUIRE(f.stats.external_memory_reads == 1);
    // Commit so the store-miss MSHR is visible in current_entries_ before the
    // count is observed and before complete_fill reads it.
    f.cache.commit();
    REQUIRE(f.cache.active_mshr_count() == 1);

    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    // The fill's write-through is staged; commit applies the MSHR free and
    // the write-buffer enqueue before they are observed.
    f.cache.commit();

    // After fill, the allocated line should be in the write buffer; no
    // writeback should appear in the gather file.
    REQUIRE(f.cache.active_mshr_count() == 0);
    REQUIRE(f.cache.write_buffer_size() == 1);
    REQUIRE_FALSE(f.gather_file.current_has_result());
}

TEST_CASE("Cache: MSHR exhaustion returns false and signals stall", "[cache]") {
    CacheFixture f;
    auto results = make_results();

    // Fill all MSHRs with load misses on distinct lines. Each allocation
    // lands in next_entries_; commit between iterations so the following
    // allocate() scans current_entries_ and sees the slot already taken.
    for (uint32_t i = 0; i < NUM_MSHRS; ++i) {
        uint32_t addr = i * LINE_SIZE * 100;
        f.claim(i % NUM_WARPS, static_cast<uint8_t>(i + 1));
        REQUIRE(f.cache.process_load(addr, i % NUM_WARPS, FULL_MASK, results,
                                     1, 0, 0));
        f.cache.commit();
    }

    // Next miss should fail because all MSHRs are occupied. The process_load
    // path rejects at MSHR allocation before touching the gather buffer, so
    // no additional claim is needed here.
    uint32_t addr = NUM_MSHRS * LINE_SIZE * 100;
    REQUIRE_FALSE(f.cache.process_load(addr, 0, FULL_MASK, results, 2, 0, 0));
    REQUIRE(f.stats.mshr_stall_cycles == 1);
    // Phase 9: cache observable state is REGISTERED — commit to observe.
    f.cache.commit();
    REQUIRE(f.cache.next_stalled());
    REQUIRE(f.cache.next_stall_reason() == CacheStallReason::MSHR_FULL);
}

TEST_CASE("Cache: MSHR stall clears once a fill completes", "[cache]") {
    CacheFixture f;
    auto results = make_results();

    // Commit between allocations so each scans the committed current_entries_.
    for (uint32_t i = 0; i < NUM_MSHRS; ++i) {
        uint32_t addr = i * LINE_SIZE * 100;
        f.claim(i % NUM_WARPS, static_cast<uint8_t>(i + 1));
        REQUIRE(f.cache.process_load(addr, i % NUM_WARPS, FULL_MASK, results,
                                     1, 0, 0));
        f.cache.commit();
    }

    // Retry load sees the stall. No claim needed: process_load rejects on
    // MSHR exhaustion before touching the gather buffer.
    REQUIRE_FALSE(f.cache.process_load(NUM_MSHRS * LINE_SIZE * 100, 0, FULL_MASK,
                                       results, 2, 0, 0));
    f.cache.commit();
    REQUIRE(f.cache.next_stalled());

    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    f.gather_file.commit();  // Phase M4: latch staged has_result
    // Drain the writeback so the buffer is free.
    while (f.gather_file.current_has_result()) {
        (void)f.gather_file.consume_result();
        f.gather_file.commit();  // re-latch after each consume
    }
    f.end_cycle();

    f.cache.evaluate();
    // Now that an MSHR is free, the retry is accepted. Warp 0 is idle (its
    // gather buffer was just drained), so claim before issuing the new load.
    f.claim(0);
    REQUIRE(f.cache.process_load(NUM_MSHRS * LINE_SIZE * 100, 0, FULL_MASK,
                                 results, 3, 0, 0));
}

TEST_CASE("Cache: direct-mapped conflict eviction", "[cache]") {
    CacheFixture f;
    auto results = make_results();

    const uint32_t num_sets = CACHE_SIZE / LINE_SIZE;
    const uint32_t addr_a = 0;
    const uint32_t addr_b = num_sets * LINE_SIZE; // same set, different tag

    // Install addr_a.
    f.claim(0);
    REQUIRE(f.cache.process_load(addr_a, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    // Phase 10D: commit() latches the fill into committed buffer state so
    // consume_result() (a pure committed read) sees it and stages the
    // release; end_cycle() then applies the staged release.
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // addr_a hits.
    f.claim(0);
    REQUIRE(f.cache.process_load(addr_a, 0, FULL_MASK, results, 2, 0, 0));
    REQUIRE(f.stats.load_hits == 1);
    // Phase 10D: commit() latches the fill into committed buffer state so
    // consume_result() (a pure committed read) sees it and stages the
    // release; end_cycle() then applies the staged release.
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // addr_b maps to same set but different tag — evicts addr_a.
    f.claim(0);
    REQUIRE(f.cache.process_load(addr_b, 0, FULL_MASK, results, 3, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    // Phase 10D: commit() latches the fill into committed buffer state so
    // consume_result() (a pure committed read) sees it and stages the
    // release; end_cycle() then applies the staged release.
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // addr_a should now miss (evicted).
    f.claim(0);
    REQUIRE(f.cache.process_load(addr_a, 0, FULL_MASK, results, 4, 0, 0));
    REQUIRE(f.stats.load_misses == 3);
    REQUIRE(f.stats.load_hits == 1);
}

TEST_CASE("Cache: write buffer full stalls store hits", "[cache]") {
    CacheFixture f;
    auto results = make_results();

    // Install a line.
    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    // Phase 10D: commit() latches the fill into committed buffer state so
    // consume_result() (a pure committed read) sees it and stages the
    // release; end_cycle() then applies the staged release.
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Fill the write buffer with hits to line 0. The write buffer has a
    // single enqueue port per cycle, so commit between stores to apply each
    // staged enqueue before the next store claims the port.
    for (uint32_t i = 0; i < WB_DEPTH; ++i) {
        REQUIRE(f.cache.process_store(0, 0, 1, 0, 0));
        f.cache.commit();
    }

    REQUIRE_FALSE(f.cache.process_store(0, 0, 1, 0, 0));
    REQUIRE(f.stats.write_buffer_stall_cycles == 1);
    f.cache.commit();
    REQUIRE(f.cache.next_stall_reason() == CacheStallReason::WRITE_BUFFER_FULL);
}

TEST_CASE("Cache: store-miss fill stalls when write buffer is full", "[cache]") {
    CacheFixture f;
    auto results = make_results();

    // Install a line so we can fill the write buffer with store hits.
    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    // Phase 10D: commit() latches the fill into committed buffer state so
    // consume_result() (a pure committed read) sees it and stages the
    // release; end_cycle() then applies the staged release.
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Single enqueue port per cycle: commit between stores so each staged
    // write-through enqueue lands before the next store.
    for (uint32_t i = 0; i < WB_DEPTH; ++i) {
        REQUIRE(f.cache.process_store(0, 0, 1, 0, 0));
        f.cache.commit();
    }

    // Issue a store miss; the line fetch will complete, but the fill cannot
    // push to the write buffer because it is full.
    REQUIRE(f.cache.process_store(1, 0, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    f.cache.commit();
    REQUIRE(f.cache.next_stalled());
    REQUIRE_FALSE(f.cache.is_idle());

    // Drain one write-buffer entry and retry the pending fill. evaluate()
    // calls handle_responses() internally; the explicit redundant call from
    // the pre-Phase-9 test is dropped since handle_responses now writes
    // next_pending_fill_ and a second call in the same cycle would re-run
    // complete_fill against the not-yet-flipped current_pending_fill_.
    //
    // registered-mshr-write-buffer: drain_write_buffer() only STAGES the pop;
    // commit() applies it. The pop must land before the fill retry runs, else
    // evaluate() still sees the buffer full. So commit the staged pop first,
    // then evaluate the fill retry against the now-drained buffer.
    f.cache.drain_write_buffer();
    f.cache.commit();
    f.cache.evaluate();
    f.cache.commit();
    REQUIRE_FALSE(f.cache.next_stalled());
}

TEST_CASE("Cache: reset clears all state", "[cache]") {
    CacheFixture f;
    auto results = make_results();

    // Install a line.
    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    // Phase 10D: commit() latches the fill into committed buffer state so
    // consume_result() (a pure committed read) sees it and stages the
    // release; end_cycle() then applies the staged release.
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Verify it hits.
    f.claim(0);
    auto before = f.stats.load_hits;
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 2, 0, 0));
    REQUIRE(f.stats.load_hits == before + 1);
    // Phase 10D: commit() latches the fill into committed buffer state so
    // consume_result() (a pure committed read) sees it and stages the
    // release; end_cycle() then applies the staged release.
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    f.cache.reset();
    f.mem_if.reset();
    f.gather_file.reset();

    // After reset, the same address should miss again.
    f.claim(0);
    auto hits = f.stats.load_hits;
    auto misses = f.stats.load_misses;
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 3, 0, 0));
    REQUIRE(f.stats.load_hits == hits);
    REQUIRE(f.stats.load_misses == misses + 1);
}

// ----------------------------------------------------------------------------
// registered-tag-array.md Step 4: fill-conflict command retry.
//
// The tag array is a REGISTERED current_/next_ pair. A fill writes next_tags_
// and "wins"; a load or store command racing that fill to the same set is
// rejected this cycle (next_cmd_ready() false, fill_conflict_retry_cycles
// bumped) and re-staged by coalescing. On the retry it reads the committed
// post-fill current_tags_.
// ----------------------------------------------------------------------------

TEST_CASE("Cache: store racing a same-set fill is retried then hits", "[cache]") {
    CacheFixture f;
    auto results = make_results();

    // Bring line 0 (set 0) toward residency via a load miss on warp 0.
    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);

    // Stage a store command to set 0 for the cycle the fill lands.
    f.cache.set_next_store_cmd(/*line_addr=*/0, 0, 2, 0, 0);
    f.cache.commit();   // flip the staged cmd into current_store_cmd_
    // evaluate(): handle_responses installs line 0 (fill_installed_set_ = 0);
    // the staged store to set 0 is rejected for the fill conflict.
    f.cache.evaluate();
    REQUIRE_FALSE(f.cache.next_cmd_ready());
    REQUIRE(f.stats.fill_conflict_retry_cycles == 1);
    // The rejected store left no partial state: nothing entered the WB.
    REQUIRE(f.cache.write_buffer_size() == 0);
    // The fill itself completed regardless of the racing command.
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result());
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Retry: re-stage the store. No fill this cycle -> it reads the committed
    // post-fill tags and hits.
    f.cache.set_next_store_cmd(/*line_addr=*/0, 0, 3, 0, 0);
    f.cache.commit();
    f.cache.evaluate();
    REQUIRE(f.cache.next_cmd_ready());
    REQUIRE(f.stats.store_hits == 1);
    // The retried store hit stages its write-through; commit applies the
    // enqueue before write_buffer_size() observes it.
    f.cache.commit();
    REQUIRE(f.cache.write_buffer_size() == 1);
    REQUIRE(f.stats.fill_conflict_retry_cycles == 1);  // no spurious retry
}

TEST_CASE("Cache: load racing a same-set fill is retried then hits", "[cache]") {
    CacheFixture f;
    auto results = make_results(2000);

    // Load miss to line 0 (set 0) on warp 0.
    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);

    // Stage a fresh load to the SAME line on warp 1 for the fill cycle.
    f.claim(1, /*dest_reg=*/3);
    auto results2 = make_results(9000);
    f.cache.set_next_load_cmd(0, 1, FULL_MASK, results2, 2, 0, 0);
    f.cache.commit();
    f.cache.evaluate();
    // The load racing the fill is retried exactly as a store would be.
    REQUIRE_FALSE(f.cache.next_cmd_ready());
    REQUIRE(f.stats.fill_conflict_retry_cycles == 1);
    REQUIRE(f.stats.load_hits == 0);
    REQUIRE(f.stats.load_misses == 1);   // no second miss allocated
    // Fill delivered warp 0's data.
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result());
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Retry: the load now reads the committed post-fill tags and hits.
    f.cache.set_next_load_cmd(0, 1, FULL_MASK, results2, 3, 0, 0);
    f.cache.commit();
    f.cache.evaluate();
    REQUIRE(f.cache.next_cmd_ready());
    REQUIRE(f.stats.load_hits == 1);
    REQUIRE(f.stats.load_misses == 1);
}

TEST_CASE("Cache: load racing an evicting fill misses cleanly on retry", "[cache]") {
    CacheFixture f;
    auto results = make_results();
    const uint32_t num_sets = CACHE_SIZE / LINE_SIZE;
    const uint32_t line_a = 0;                 // set 0, tag 0
    const uint32_t line_b = num_sets;          // set 0, tag 1 (conflicts with A)

    // Install line A in set 0.
    f.claim(0);
    REQUIRE(f.cache.process_load(line_a * LINE_SIZE, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Load miss to line B (same set, different tag) — its fill will evict A.
    f.claim(1);
    REQUIRE(f.cache.process_load(line_b * LINE_SIZE, 1, FULL_MASK, results, 2, 0, 0));
    f.tick_mem(MEM_LATENCY);

    // Stage a load to line A racing B's evicting fill.
    f.claim(2, /*dest_reg=*/4);
    f.cache.set_next_load_cmd(line_a * LINE_SIZE, 2, FULL_MASK, results, 3, 0, 0);
    f.cache.commit();
    f.cache.evaluate();
    REQUIRE_FALSE(f.cache.next_cmd_ready());
    REQUIRE(f.stats.fill_conflict_retry_cycles == 1);
    auto misses_before = f.stats.load_misses;
    f.gather_file.commit();
    (void)f.gather_file.consume_result();   // B's fill data
    f.end_cycle();

    // Retry: set 0 now holds B; the load to A misses cleanly (no stale hit).
    f.cache.set_next_load_cmd(line_a * LINE_SIZE, 2, FULL_MASK, results, 4, 0, 0);
    f.cache.commit();
    f.cache.evaluate();
    REQUIRE(f.cache.next_cmd_ready());
    REQUIRE(f.stats.load_misses == misses_before + 1);
    REQUIRE(f.stats.load_hits == 0);   // never a stale hit on the evicted line
}

TEST_CASE("Cache: orphaned-secondary regression — load racing a lone-primary fill",
          "[cache]") {
    CacheFixture f;
    auto results = make_results(7000);

    // Lone primary: a single load miss to line 0, no dependent chain.
    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 1, 0, 0));
    REQUIRE(f.stats.external_memory_reads == 1);
    f.tick_mem(MEM_LATENCY);

    // A fresh load to the same line, racing the cycle complete_fill frees the
    // lone primary. Step 4's retry must reject it BEFORE it reaches
    // find_chain_tail — otherwise (retry restricted to stores) it would chain
    // a secondary onto the just-freed primary, or allocate a duplicate fetch.
    f.claim(1, /*dest_reg=*/6);
    auto results2 = make_results(123);
    f.cache.set_next_load_cmd(0, 1, FULL_MASK, results2, 2, 0, 0);
    f.cache.commit();
    f.cache.evaluate();
    REQUIRE_FALSE(f.cache.next_cmd_ready());
    REQUIRE(f.stats.fill_conflict_retry_cycles == 1);
    // No second external fetch, no secondary MSHR allocated.
    REQUIRE(f.stats.external_memory_reads == 1);
    REQUIRE(f.stats.mshr_merged_loads == 0);
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();
    // The lone primary was freed by the fill; nothing chained to it.
    REQUIRE(f.cache.active_mshr_count() == 0);

    // Retry: the load hits the now-resident line; no undrained secondary.
    f.cache.set_next_load_cmd(0, 1, FULL_MASK, results2, 3, 0, 0);
    f.cache.commit();
    f.cache.evaluate();
    REQUIRE(f.cache.next_cmd_ready());
    REQUIRE(f.stats.load_hits == 1);
    REQUIRE(f.stats.external_memory_reads == 1);
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();
    // Cache fully drained: no leaked MSHR, sim can terminate.
    REQUIRE(f.cache.is_idle());
}

TEST_CASE("Cache: store hit racing an evicting fill retries into a miss", "[cache]") {
    CacheFixture f;
    auto results = make_results();
    const uint32_t num_sets = CACHE_SIZE / LINE_SIZE;
    const uint32_t line_a = 0;            // set 0, tag 0
    const uint32_t line_b = num_sets;     // set 0, tag 1

    // Install line A in set 0.
    f.claim(0);
    REQUIRE(f.cache.process_load(line_a * LINE_SIZE, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Load-miss to line B — its fill evicts A from set 0. (A load fill is
    // used so the evicting line is installed unpinned: a store-miss fill
    // would write-ack-pin set 0 and the racing store would then stall
    // LINE_PINNED rather than retry into a clean miss.)
    f.claim(1);
    REQUIRE(f.cache.process_load(line_b * LINE_SIZE, 1, FULL_MASK, results, 2, 0, 0));
    f.tick_mem(MEM_LATENCY);

    // Stage a store to line A racing B's evicting fill. Pre-change this would
    // "hit" the doomed line and queue a write-through for it.
    auto store_hits_before = f.stats.store_hits;
    auto store_misses_before = f.stats.store_misses;
    f.cache.set_next_store_cmd(line_a, 0, 3, 0, 0);
    f.cache.commit();
    f.cache.evaluate();
    REQUIRE_FALSE(f.cache.next_cmd_ready());
    REQUIRE(f.stats.fill_conflict_retry_cycles == 1);
    REQUIRE(f.stats.store_hits == store_hits_before);   // did not hit doomed A
    f.gather_file.commit();
    (void)f.gather_file.consume_result();               // B's load fill data
    f.end_cycle();

    // Retry: set 0 holds B (unpinned); the store to A is correctly a miss
    // (write-allocate).
    f.cache.set_next_store_cmd(line_a, 0, 4, 0, 0);
    f.cache.commit();
    f.cache.evaluate();
    REQUIRE(f.cache.next_cmd_ready());
    REQUIRE(f.stats.store_misses == store_misses_before + 1);
    REQUIRE(f.stats.store_hits == store_hits_before);
}

TEST_CASE("Cache: a fill is never blocked by a same-set command", "[cache]") {
    CacheFixture f;
    auto results = make_results(4242);

    // Load miss to line 0; the fill is pending.
    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);

    // A store command to the same set is staged for the fill cycle.
    f.cache.set_next_store_cmd(0, 0, 2, 0, 0);
    f.cache.commit();
    f.cache.evaluate();
    // The command was rejected, but the fill completed unconditionally.
    REQUIRE_FALSE(f.cache.next_cmd_ready());
    f.gather_file.commit();
    REQUIRE(f.gather_file.current_has_result());
    (void)f.gather_file.consume_result();
    f.end_cycle();
    REQUIRE(f.cache.active_mshr_count() == 0);   // fill freed the MSHR
}

TEST_CASE("Cache: command to a set with no fill is not spuriously retried",
          "[cache]") {
    CacheFixture f;
    auto results = make_results();

    // Install line 0 in set 0.
    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // A store to set 0 with no fill landing this cycle proceeds normally.
    f.cache.set_next_store_cmd(0, 0, 2, 0, 0);
    f.cache.commit();
    f.cache.evaluate();
    REQUIRE(f.cache.next_cmd_ready());
    REQUIRE(f.stats.fill_conflict_retry_cycles == 0);
    REQUIRE(f.stats.store_hits == 1);
}

// ----------------------------------------------------------------------------
// eager-wobbling-pizza.md: write-ack line pinning. A cache line is kept pinned
// (un-evictable) from the moment a write-through is queued for it until the
// external write ack for that store is received.
// ----------------------------------------------------------------------------

TEST_CASE("Cache: a store hit pins its set until the write ack", "[cache]") {
    CacheFixture f;
    auto results = make_results();
    const uint32_t num_sets = CACHE_SIZE / LINE_SIZE;
    const uint32_t line_a = 0;            // set 0, tag 0
    const uint32_t line_b = num_sets;     // set 0, tag 1

    // Install line A in set 0.
    f.claim(0);
    REQUIRE(f.cache.process_load(line_a * LINE_SIZE, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Store hit to A queues a write-through, which pins set 0 (write-ack pin).
    REQUIRE(f.cache.process_store(line_a, 0, 2, 0, 0));
    REQUIRE(f.stats.store_hits == 1);
    f.end_cycle();   // commit: current_outstanding_writes_[0] becomes 1

    // A conflicting different-tag miss to set 0 stalls LINE_PINNED on the
    // write-ack pin — attributed to write_ack_pin_stall_cycles, not the
    // chain-pin counter.
    auto wap_before = f.stats.write_ack_pin_stall_cycles;
    REQUIRE_FALSE(f.cache.process_store(line_b, 0, 3, 0, 0));
    f.cache.commit();
    REQUIRE(f.cache.next_stall_reason() == CacheStallReason::LINE_PINNED);
    REQUIRE(f.stats.write_ack_pin_stall_cycles == wap_before + 1);
    REQUIRE(f.stats.line_pin_stall_cycles == 0);   // no chain pin involved

    // Drain A's write-through; once its ack is consumed the pin releases.
    for (uint32_t i = 0; i < 200 && !f.cache.is_idle(); ++i) f.pump();
    REQUIRE(f.cache.is_idle());

    // The conflicting store is now accepted (write-allocate miss).
    REQUIRE(f.cache.process_store(line_b, 0, 4, 0, 0));
    REQUIRE(f.stats.store_misses == 1);
}

TEST_CASE("Cache: write-ack pin releases the cycle after the ack is consumed",
          "[cache]") {
    CacheFixture f;
    auto results = make_results();
    const uint32_t num_sets = CACHE_SIZE / LINE_SIZE;
    const uint32_t line_a = 0;
    const uint32_t line_b = num_sets;

    // Install line A, then a store hit pins set 0.
    f.claim(0);
    REQUIRE(f.cache.process_load(line_a * LINE_SIZE, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();
    REQUIRE(f.cache.process_store(line_a, 0, 2, 0, 0));
    f.end_cycle();

    // Drain A's write-through to memory and advance memory until the write
    // ack is sitting in the interface — but do NOT let the cache consume it.
    f.cache.drain_write_buffer();
    f.cache.commit();
    f.mem_if.commit();
    for (uint32_t i = 0; i < MEM_LATENCY + 4 && !f.mem_if.current_has_write_ack(); ++i) {
        f.mem_if.evaluate();
    }
    REQUIRE(f.mem_if.current_has_write_ack());

    // The pin is still in force: a conflicting miss stalls LINE_PINNED.
    REQUIRE_FALSE(f.cache.process_store(line_b, 0, 3, 0, 0));

    // This evaluate() consumes the write ack (decrements next_), but the
    // registered counter has not flipped yet — the pin is still observable.
    f.cache.evaluate();
    REQUIRE_FALSE(f.cache.process_store(line_b, 0, 4, 0, 0));   // still pinned
    f.cache.commit();   // flip: current_outstanding_writes_[0] becomes 0

    // The cycle AFTER the ack was consumed, the pin is gone.
    REQUIRE(f.cache.process_store(line_b, 0, 5, 0, 0));
    REQUIRE(f.stats.store_misses == 1);
}

TEST_CASE("Cache: a conflicting fill is deferred while a write-ack pin holds",
          "[cache]") {
    CacheFixture f;
    auto results = make_results();
    const uint32_t num_sets = CACHE_SIZE / LINE_SIZE;
    const uint32_t line_a = 0;
    const uint32_t line_b = num_sets;

    // Install line A in set 0.
    f.claim(0);
    REQUIRE(f.cache.process_load(line_a * LINE_SIZE, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Load miss to line B (set 0, different tag) — allocate its MSHR and
    // start its fetch BEFORE set 0 is pinned.
    f.claim(1);
    REQUIRE(f.cache.process_load(line_b * LINE_SIZE, 1, FULL_MASK, results, 2, 0, 0));
    // Store hit to A pins set 0 before B's fill returns.
    REQUIRE(f.cache.process_store(line_a, 0, 3, 0, 0));
    f.end_cycle();

    // B's fill returns. complete_fill must DEFER (set 0 is write-ack-pinned)
    // and A must NOT be evicted — the core memory-ordering guarantee.
    f.tick_mem(MEM_LATENCY);
    f.cache.evaluate();
    f.cache.commit();
    REQUIRE(f.cache.current_last_fill_event().deferred);
    REQUIRE(f.cache.current_pending_fill().valid);   // B still pending
    REQUIRE(f.stats.write_ack_pin_stall_cycles > 0);
    // A is still resident — a load to A still hits.
    f.claim(2);
    REQUIRE(f.cache.process_load(line_a * LINE_SIZE, 2, FULL_MASK, results, 4, 0, 0));
    REQUIRE(f.stats.load_hits == 1);
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Drain A's write-through; once the pin clears B's deferred fill lands.
    for (uint32_t i = 0; i < 400 && !f.cache.is_idle(); ++i) f.pump();
    REQUIRE(f.cache.is_idle());
    // B is now resident.
    f.claim(3);
    REQUIRE(f.cache.process_load(line_b * LINE_SIZE, 3, FULL_MASK, results, 5, 0, 0));
    REQUIRE(f.stats.load_hits == 2);
}

TEST_CASE("Cache: deadlock regression — read fill deferred by a write-ack pin "
          "still makes progress", "[cache]") {
    // If write acks shared the read-fill response queue, a fill deferred on a
    // write-ack pin could never be cleared (the ack would sit behind the
    // deferred fill). handle_responses consumes the write ack on a separate
    // channel UNCONDITIONALLY, before the deferred-fill early return — so the
    // pin clears, the fill lands, and the sim terminates.
    CacheFixture f;
    auto results = make_results();
    const uint32_t num_sets = CACHE_SIZE / LINE_SIZE;
    const uint32_t line_a = 0;
    const uint32_t line_b = num_sets;

    // Install A; start B's fetch; store-hit A to pin set 0.
    f.claim(0);
    REQUIRE(f.cache.process_load(line_a * LINE_SIZE, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    f.claim(1);
    REQUIRE(f.cache.process_load(line_b * LINE_SIZE, 1, FULL_MASK, results, 2, 0, 0));
    REQUIRE(f.cache.process_store(line_a, 0, 3, 0, 0));
    f.end_cycle();

    // B's fill is now deferred behind the write-ack pin. Pump: the write ack
    // is consumed unconditionally even while the fill is deferred, so the
    // cache makes forward progress and reaches is_idle().
    f.tick_mem(MEM_LATENCY);
    bool was_deferred = false;
    for (uint32_t i = 0; i < 400 && !f.cache.is_idle(); ++i) {
        f.pump();
        if (f.cache.current_pending_fill().valid) was_deferred = true;
    }
    REQUIRE(was_deferred);            // the fill genuinely was deferred
    REQUIRE(f.cache.is_idle());       // ... yet the cache still drained
}

TEST_CASE("Cache: N outstanding writes keep a set pinned until every ack",
          "[cache]") {
    CacheFixture f;
    auto results = make_results();
    const uint32_t num_sets = CACHE_SIZE / LINE_SIZE;
    const uint32_t line_a = 0;
    const uint32_t line_b = num_sets;
    constexpr uint32_t kN = 3;        // N <= WB_DEPTH so all N hits enqueue

    // Install line A.
    f.claim(0);
    REQUIRE(f.cache.process_load(line_a * LINE_SIZE, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // N store hits to A: N outstanding write-throughs on set 0.
    for (uint32_t i = 0; i < kN; ++i) {
        REQUIRE(f.cache.process_store(line_a, 0, 2 + i, 0, 0));
        f.end_cycle();
    }

    // Set 0 stays pinned while any of the N writes is unacked. Pump until the
    // cache is idle (all N acks consumed, one per cycle); a conflicting store
    // is rejected on every cycle before that and accepted only after.
    uint32_t rejections = 0;
    for (uint32_t i = 0; i < 400 && !f.cache.is_idle(); ++i) {
        if (!f.cache.process_store(line_b, 0, 100, 0, 0)) ++rejections;
        f.pump();
    }
    REQUIRE(f.cache.is_idle());
    REQUIRE(rejections >= kN);        // pinned across at least N drain cycles
    REQUIRE(f.cache.process_store(line_b, 0, 200, 0, 0));   // accepted now
}

TEST_CASE("Cache: chain pin hands off to the write-ack pin", "[cache]") {
    CacheFixture f;
    auto results = make_results();
    const uint32_t num_sets = CACHE_SIZE / LINE_SIZE;
    const uint32_t line_a = 0;
    const uint32_t line_b = num_sets;

    // Store-miss to A (primary) + store to A (secondary): a dependent chain.
    // commit between them so the secondary's find_chain_tail scans the
    // committed current_entries_ and sees the primary (in production the two
    // accesses are always >= 1 cycle apart with a commit between).
    REQUIRE(f.cache.process_store(line_a, 0, 1, 0, 0));
    f.cache.commit();
    REQUIRE(f.cache.process_store(line_a, 0, 2, 0, 0));
    f.cache.commit();
    REQUIRE(f.stats.mshr_merged_stores == 1);

    // Deliver the fill. complete_fill installs A, sets the chain pin, and
    // queues A's write-through (write-ack pin). The secondary then drains,
    // queuing a second write-through and clearing the chain pin.
    f.tick_mem(MEM_LATENCY);
    f.cache.evaluate();   // primary fill
    f.cache.commit();
    f.cache.evaluate();   // secondary drain
    f.cache.commit();
    REQUIRE(f.cache.active_mshr_count() == 0);
    // Chain fully drained — chain pin is clear (pinned_line_count counts only
    // the chain pin) — but the set stays effectively pinned via the write-ack
    // counter: a conflicting miss still stalls LINE_PINNED.
    REQUIRE(f.cache.pinned_line_count() == 0);
    auto wap_before = f.stats.write_ack_pin_stall_cycles;
    REQUIRE_FALSE(f.cache.process_store(line_b, 0, 3, 0, 0));
    REQUIRE(f.stats.write_ack_pin_stall_cycles == wap_before + 1);

    // Once the write-throughs are acked, the set fully unpins.
    for (uint32_t i = 0; i < 400 && !f.cache.is_idle(); ++i) f.pump();
    REQUIRE(f.cache.is_idle());
    REQUIRE(f.cache.process_store(line_b, 0, 4, 0, 0));
}

TEST_CASE("Cache: the outstanding-write cap refuses enqueue and throttles",
          "[cache]") {
    Stats stats;
    FixedLatencyMemory mem_if{MEM_LATENCY, stats};
    LoadGatherBufferFile gather_file{NUM_WARPS, stats};
    // A small cap, below WB_DEPTH, so the cap — not write-buffer depth — is
    // the binding constraint.
    constexpr uint32_t kCap = 2;
    L1Cache cache{CACHE_SIZE, LINE_SIZE, NUM_MSHRS, WB_DEPTH, kCap,
                  mem_if, gather_file, stats};
    auto results = make_results();

    // Install line 0.
    gather_file.claim(0, 1, 0, 0, 0);
    gather_file.commit();
    gather_file.evaluate();
    REQUIRE(cache.process_load(0, 0, FULL_MASK, results, 1, 0, 0));
    mem_if.commit();
    for (uint32_t i = 0; i < MEM_LATENCY; ++i) mem_if.evaluate();
    cache.handle_responses();
    gather_file.commit();
    (void)gather_file.consume_result();
    cache.commit();
    gather_file.commit();

    // kCap store hits — kCap outstanding write-throughs — reach the cap.
    for (uint32_t i = 0; i < kCap; ++i) {
        REQUIRE(cache.process_store(0, 0, 2 + i, 0, 0));
        cache.commit();
    }
    REQUIRE(cache.write_buffer_size() < WB_DEPTH);   // buffer still has room

    // The next write-through is refused by the CAP, not by buffer depth:
    // WRITE_BUFFER_FULL stall reason, write_throttle_stall_cycles bumped,
    // write_buffer_stall_cycles NOT.
    auto throttle_before = stats.write_throttle_stall_cycles;
    auto wbfull_before = stats.write_buffer_stall_cycles;
    REQUIRE_FALSE(cache.process_store(0, 0, 99, 0, 0));
    cache.commit();
    REQUIRE(cache.next_stall_reason() == CacheStallReason::WRITE_BUFFER_FULL);
    REQUIRE(stats.write_throttle_stall_cycles == throttle_before + 1);
    REQUIRE(stats.write_buffer_stall_cycles == wbfull_before);

    // Each write ack returns a credit; drain to idle, then the store succeeds.
    for (uint32_t i = 0; i < 400 && !cache.is_idle(); ++i) {
        cache.evaluate();
        mem_if.evaluate();
        cache.drain_write_buffer();
        cache.commit();
        mem_if.commit();
        gather_file.commit();
    }
    REQUIRE(cache.is_idle());
    REQUIRE(cache.process_store(0, 0, 100, 0, 0));
}

// ----------------------------------------------------------------------------
// registered-mshr-write-buffer.md Part B: the write buffer is a REGISTERED,
// single-enqueue-port FIFO mutated only at commit().
// ----------------------------------------------------------------------------

TEST_CASE("Cache: the write buffer is not fall-through", "[cache]") {
    CacheFixture f;
    auto results = make_results();

    // Install line 0.
    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Cycle T: a store hit stages a write-through. The registered FIFO is not
    // mutated until commit, so write_buffer_size() still reads 0, and a
    // same-cycle drain_write_buffer sees the committed (empty) buffer and
    // submits nothing.
    REQUIRE(f.cache.process_store(0, 0, 2, 0, 0));
    REQUIRE(f.cache.write_buffer_size() == 0);
    f.cache.drain_write_buffer();
    REQUIRE(f.stats.external_memory_writes == 0);   // nothing submitted at T
    f.end_cycle();   // commit: the staged enqueue lands now

    // Cycle T+1: the entry is in the buffer and is submittable — not before.
    REQUIRE(f.cache.write_buffer_size() == 1);
    f.cache.drain_write_buffer();
    REQUIRE(f.stats.external_memory_writes == 1);
}

TEST_CASE("Cache: write buffer has a single enqueue port — FILL beats HIT",
          "[cache]") {
    CacheFixture f;
    auto results = make_results();
    const uint32_t num_sets = CACHE_SIZE / LINE_SIZE;
    const uint32_t line_a = 0;            // set 0
    const uint32_t line_b = 1;            // set 1 — distinct set, no conflict
    (void)num_sets;

    // Install line A (set 0) so a later store to it is a hit.
    f.claim(0);
    REQUIRE(f.cache.process_load(line_a * LINE_SIZE, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    f.gather_file.commit();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Store-miss to line B (set 1): allocates a write-allocate MSHR.
    REQUIRE(f.cache.process_store(line_b, 0, 2, 0, 0));
    f.tick_mem(MEM_LATENCY);

    // One evaluate where B's store-miss fill (FILL) and a staged store hit to
    // A (HIT) both want to enqueue a write-through. The single enqueue port
    // goes to the FILL by tick order; the HIT is refused and retried.
    auto port_conflicts_before = f.stats.write_buffer_port_conflict_cycles;
    f.cache.set_next_store_cmd(line_a, 0, 3, 0, 0);
    f.cache.commit();
    f.cache.evaluate();
    REQUIRE_FALSE(f.cache.next_cmd_ready());   // HIT lost the enqueue port
    REQUIRE(f.stats.write_buffer_port_conflict_cycles == port_conflicts_before + 1);
    f.end_cycle();
    // The FILL's write-through (line B) is the one that entered the buffer.
    REQUIRE(f.cache.write_buffer_size() == 1);

    // Retry: next cycle the port is free, so the store hit succeeds.
    f.cache.set_next_store_cmd(line_a, 0, 4, 0, 0);
    f.cache.commit();
    f.cache.evaluate();
    REQUIRE(f.cache.next_cmd_ready());
    REQUIRE(f.stats.store_hits == 1);
}

TEST_CASE("Cache: terminates with the registered MSHR file", "[cache]") {
    CacheFixture f;
    auto results = make_results();

    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 1, 0, 0));
    f.cache.commit();   // latch the allocated MSHR into committed state
    REQUIRE_FALSE(f.cache.is_idle());   // an active MSHR keeps the cache busy

    f.tick_mem(MEM_LATENCY);
    for (uint32_t i = 0; i < 200 && !f.cache.is_idle(); ++i) {
        f.pump();
        if (f.gather_file.current_has_result()) {
            (void)f.gather_file.consume_result();
        }
    }
    REQUIRE(f.cache.is_idle());   // registered MSHR file drains cleanly
}
