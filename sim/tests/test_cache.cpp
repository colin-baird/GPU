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

// Common fixture: wire a cache, memory interface, and gather-buffer file
// against a shared Stats object. Each test constructs one of these.
struct CacheFixture {
    Stats stats;
    FixedLatencyMemory mem_if{MEM_LATENCY, stats};
    LoadGatherBufferFile gather_file{NUM_WARPS, stats};
    L1Cache cache{CACHE_SIZE, LINE_SIZE, NUM_MSHRS, WB_DEPTH, mem_if, gather_file, stats};

    // Advance the external memory interface by N cycles. Phase M5: the
    // request staged via set_next_*_request needs commit() to flip
    // next_ → current_ before the first evaluate() drains it into
    // in_flight_. Insert that commit at the head so callers can keep
    // the pre-Phase-M5 process_load/process_store + tick_mem(MEM_LATENCY)
    // pattern.
    void tick_mem(uint32_t cycles) {
        mem_if.commit();
        for (uint32_t i = 0; i < cycles; ++i) {
            mem_if.evaluate();
        }
    }

    // Tick the gather buffer's per-cycle commit (clears the shared
    // next_port_claimed_ flag at end-of-cycle).
    void end_cycle() {
        cache.commit();
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
    REQUIRE(f.cache.active_mshr_count() == 1);
    REQUIRE(f.stats.load_misses == 1);
    REQUIRE(f.stats.external_memory_reads == 1);
    REQUIRE_FALSE(f.gather_file.current_has_result());

    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    // Phase M4: latch staged has_result via commit before observing.
    f.gather_file.commit();

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
    REQUIRE(f.cache.write_buffer_size() == 1);
    REQUIRE_FALSE(f.gather_file.current_busy(0));
}

TEST_CASE("Cache: store miss allocates MSHR (write-allocate)", "[cache]") {
    CacheFixture f;

    REQUIRE(f.cache.process_store(1, 0, 1, 0, 0));
    REQUIRE(f.stats.store_misses == 1);
    REQUIRE(f.stats.external_memory_reads == 1);
    REQUIRE(f.cache.active_mshr_count() == 1);

    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();

    // After fill, the allocated line should be in the write buffer; no
    // writeback should appear in the gather file.
    REQUIRE(f.cache.active_mshr_count() == 0);
    REQUIRE(f.cache.write_buffer_size() == 1);
    REQUIRE_FALSE(f.gather_file.current_has_result());
}

TEST_CASE("Cache: MSHR exhaustion returns false and signals stall", "[cache]") {
    CacheFixture f;
    auto results = make_results();

    // Fill all MSHRs with load misses on distinct lines.
    for (uint32_t i = 0; i < NUM_MSHRS; ++i) {
        uint32_t addr = i * LINE_SIZE * 100;
        f.claim(i % NUM_WARPS, static_cast<uint8_t>(i + 1));
        REQUIRE(f.cache.process_load(addr, i % NUM_WARPS, FULL_MASK, results,
                                     1, 0, 0));
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

    for (uint32_t i = 0; i < NUM_MSHRS; ++i) {
        uint32_t addr = i * LINE_SIZE * 100;
        f.claim(i % NUM_WARPS, static_cast<uint8_t>(i + 1));
        REQUIRE(f.cache.process_load(addr, i % NUM_WARPS, FULL_MASK, results,
                                     1, 0, 0));
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

    // Fill the write buffer with hits to line 0.
    for (uint32_t i = 0; i < WB_DEPTH; ++i) {
        REQUIRE(f.cache.process_store(0, 0, 1, 0, 0));
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

    for (uint32_t i = 0; i < WB_DEPTH; ++i) {
        REQUIRE(f.cache.process_store(0, 0, 1, 0, 0));
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
    f.cache.drain_write_buffer();
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

    // Store-miss (write-allocate) to line B — its fill evicts A from set 0.
    REQUIRE(f.cache.process_store(line_b, 0, 2, 0, 0));
    REQUIRE(f.stats.store_misses == 1);
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
    f.end_cycle();

    // Retry: set 0 holds B; the store to A is correctly a miss (write-allocate).
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
