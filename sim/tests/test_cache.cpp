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

    // Advance the external memory interface by N cycles.
    void tick_mem(uint32_t cycles) {
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

TEST_CASE("Cache: coalesced load hit populates all 32 gather slots", "[cache]") {
    CacheFixture f;
    auto results = make_results(1000);

    // Install a line via a coalesced load miss + fill.
    f.claim(/*warp_id=*/0);
    REQUIRE(f.cache.process_load(256, 0, FULL_MASK, results, 1, 0, 0));
    REQUIRE(f.stats.load_misses == 1);
    REQUIRE_FALSE(f.gather_file.has_result()); // still waiting on fill

    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    // Fill should have populated all 32 slots via the gather buffer.
    REQUIRE(f.gather_file.has_result());
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
    REQUIRE(f.gather_file.has_result()); // all 32 slots filled in one cycle
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
    REQUIRE_FALSE(f.gather_file.has_result());

    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();

    // Fill releases the MSHR and deposits the full line into the gather buffer.
    REQUIRE(f.cache.active_mshr_count() == 0);
    REQUIRE(f.gather_file.has_result());
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
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Store to the same line hits. Stores never touch the gather buffer.
    REQUIRE(f.cache.process_store(0, 0, 2, 0, 0));
    REQUIRE(f.stats.store_hits == 1);
    REQUIRE(f.cache.write_buffer_size() == 1);
    REQUIRE_FALSE(f.gather_file.is_busy(0));
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
    REQUIRE_FALSE(f.gather_file.has_result());
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
    REQUIRE(f.cache.is_stalled());
    REQUIRE(f.cache.stall_reason() == CacheStallReason::MSHR_FULL);
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
    REQUIRE(f.cache.is_stalled());

    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    // Drain the writeback so the buffer is free.
    while (f.gather_file.has_result()) {
        (void)f.gather_file.consume_result();
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
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // addr_a hits.
    f.claim(0);
    REQUIRE(f.cache.process_load(addr_a, 0, FULL_MASK, results, 2, 0, 0));
    REQUIRE(f.stats.load_hits == 1);
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // addr_b maps to same set but different tag — evicts addr_a.
    f.claim(0);
    REQUIRE(f.cache.process_load(addr_b, 0, FULL_MASK, results, 3, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
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
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Fill the write buffer with hits to line 0.
    for (uint32_t i = 0; i < WB_DEPTH; ++i) {
        REQUIRE(f.cache.process_store(0, 0, 1, 0, 0));
    }

    REQUIRE_FALSE(f.cache.process_store(0, 0, 1, 0, 0));
    REQUIRE(f.stats.write_buffer_stall_cycles == 1);
    REQUIRE(f.cache.stall_reason() == CacheStallReason::WRITE_BUFFER_FULL);
}

TEST_CASE("Cache: store-miss fill stalls when write buffer is full", "[cache]") {
    CacheFixture f;
    auto results = make_results();

    // Install a line so we can fill the write buffer with store hits.
    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
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
    REQUIRE(f.cache.is_stalled());
    REQUIRE_FALSE(f.cache.is_idle());

    // Drain one write-buffer entry and retry the pending fill.
    f.cache.drain_write_buffer();
    f.cache.evaluate();
    f.cache.handle_responses();
    REQUIRE_FALSE(f.cache.is_stalled());
}

TEST_CASE("Cache: reset clears all state", "[cache]") {
    CacheFixture f;
    auto results = make_results();

    // Install a line.
    f.claim(0);
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 1, 0, 0));
    f.tick_mem(MEM_LATENCY);
    f.cache.handle_responses();
    (void)f.gather_file.consume_result();
    f.end_cycle();

    // Verify it hits.
    f.claim(0);
    auto before = f.stats.load_hits;
    REQUIRE(f.cache.process_load(0, 0, FULL_MASK, results, 2, 0, 0));
    REQUIRE(f.stats.load_hits == before + 1);
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
