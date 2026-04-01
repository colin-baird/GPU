#include "catch.hpp"
#include "gpu_sim/timing/cache.h"
#include "gpu_sim/timing/memory_interface.h"
#include "gpu_sim/stats.h"

using namespace gpu_sim;

static const uint32_t CACHE_SIZE = 4096;
static const uint32_t LINE_SIZE = 128;
static const uint32_t NUM_MSHRS = 4;
static const uint32_t WB_DEPTH = 4;

struct CacheFixture {
    Stats stats;
    ExternalMemoryInterface mem_if{10, stats};  // 10 cycle latency for fast tests
    L1Cache cache{CACHE_SIZE, LINE_SIZE, NUM_MSHRS, WB_DEPTH, mem_if, stats};

    // Helper: advance memory interface N cycles
    void tick_mem(uint32_t cycles) {
        for (uint32_t i = 0; i < cycles; ++i) {
            mem_if.evaluate();
        }
    }
};

TEST_CASE("Cache: load miss then hit", "[cache]") {
    CacheFixture f;
    WritebackEntry wb;
    std::array<uint32_t, WARP_SIZE> results{};
    results[0] = 42;

    // First load: should miss (no line installed yet)
    bool accepted = f.cache.process_load(256, 0, 5, results, 1, 0, 0, wb);
    REQUIRE(accepted);
    REQUIRE_FALSE(wb.valid); // Miss, no immediate writeback
    REQUIRE(f.stats.load_misses == 1);

    // Advance memory to deliver response
    f.tick_mem(10);

    // Handle response: should produce writeback
    WritebackEntry fill_wb;
    bool fill_valid = false;
    f.cache.handle_responses(fill_wb, fill_valid);
    REQUIRE(fill_valid);
    REQUIRE(fill_wb.warp_id == 0);
    REQUIRE(fill_wb.dest_reg == 5);
    REQUIRE(fill_wb.values[0] == 42);

    // Second load to same line: should hit
    WritebackEntry wb2;
    wb2.valid = false;
    bool accepted2 = f.cache.process_load(256, 0, 6, results, 2, 0, 0, wb2);
    REQUIRE(accepted2);
    REQUIRE(wb2.valid); // Hit!
    REQUIRE(f.stats.load_hits == 1);
}

TEST_CASE("Cache: store hit writes to write buffer", "[cache]") {
    CacheFixture f;
    WritebackEntry wb;
    std::array<uint32_t, WARP_SIZE> results{};

    // Install a cache line via load miss + fill
    f.cache.process_load(0, 0, 1, results, 1, 0, 0, wb);
    f.tick_mem(10);
    WritebackEntry fill_wb;
    bool fill_valid;
    f.cache.handle_responses(fill_wb, fill_valid);

    // Now store to same line: should hit
    uint32_t line_addr = 0 / LINE_SIZE; // line_addr = 0
    bool accepted = f.cache.process_store(line_addr, 0, 1, 0, 0);
    REQUIRE(accepted);
    REQUIRE(f.stats.store_hits == 1);
}

TEST_CASE("Cache: store miss allocates MSHR (write-allocate)", "[cache]") {
    CacheFixture f;

    // Store to line not in cache
    uint32_t line_addr = 1;
    bool accepted = f.cache.process_store(line_addr, 0, 1, 0, 0);
    REQUIRE(accepted);
    REQUIRE(f.stats.store_misses == 1);
    REQUIRE(f.stats.external_memory_reads == 1); // Write-allocate fetches line
}

TEST_CASE("Cache: MSHR stall when all entries used", "[cache]") {
    CacheFixture f;
    WritebackEntry wb;
    std::array<uint32_t, WARP_SIZE> results{};

    // Fill all MSHRs (4 entries) with load misses to different lines
    for (uint32_t i = 0; i < NUM_MSHRS; ++i) {
        uint32_t addr = i * LINE_SIZE * 100; // Different sets + tags
        wb.valid = false;
        bool accepted = f.cache.process_load(addr, 0, static_cast<uint8_t>(i + 1),
                                             results, 1, 0, 0, wb);
        REQUIRE(accepted);
    }
    REQUIRE(f.stats.load_misses == NUM_MSHRS);

    // Next load miss should fail (no free MSHR)
    wb.valid = false;
    uint32_t addr = NUM_MSHRS * LINE_SIZE * 100;
    bool accepted = f.cache.process_load(addr, 0, 10, results, 1, 0, 0, wb);
    REQUIRE_FALSE(accepted);
    REQUIRE(f.stats.mshr_stall_cycles == 1);
}

TEST_CASE("Cache: transient MSHR stall retries once a slot is free", "[cache]") {
    CacheFixture f;
    WritebackEntry wb;
    std::array<uint32_t, WARP_SIZE> results{};

    for (uint32_t i = 0; i < NUM_MSHRS; ++i) {
        uint32_t addr = i * LINE_SIZE * 100;
        wb.valid = false;
        REQUIRE(f.cache.process_load(addr, 0, static_cast<uint8_t>(i + 1), results, 1, 0, 0, wb));
    }

    wb.valid = false;
    REQUIRE_FALSE(f.cache.process_load(NUM_MSHRS * LINE_SIZE * 100, 0, 9, results, 2, 0, 0, wb));
    REQUIRE(f.cache.is_stalled());

    f.tick_mem(10);
    WritebackEntry fill_wb;
    bool fill_valid = false;
    f.cache.handle_responses(fill_wb, fill_valid);
    REQUIRE(fill_valid);

    f.cache.evaluate();
    wb.valid = false;
    REQUIRE(f.cache.process_load(NUM_MSHRS * LINE_SIZE * 100, 0, 9, results, 3, 0, 0, wb));
}

TEST_CASE("Cache: direct-mapped conflict eviction", "[cache]") {
    CacheFixture f;
    WritebackEntry wb;
    std::array<uint32_t, WARP_SIZE> results{};

    uint32_t num_sets = CACHE_SIZE / LINE_SIZE; // 32 sets

    // Load addr A that maps to set 0
    uint32_t addr_a = 0;
    f.cache.process_load(addr_a, 0, 1, results, 1, 0, 0, wb);
    f.tick_mem(10);
    WritebackEntry fill;
    bool fill_valid;
    f.cache.handle_responses(fill, fill_valid);

    // Verify addr_a hits
    wb.valid = false;
    f.cache.process_load(addr_a, 0, 1, results, 2, 0, 0, wb);
    REQUIRE(wb.valid); // Hit

    // Load addr B that maps to same set but different tag
    uint32_t addr_b = num_sets * LINE_SIZE; // Same set, different tag
    wb.valid = false;
    f.cache.process_load(addr_b, 0, 2, results, 3, 0, 0, wb);
    REQUIRE_FALSE(wb.valid); // Miss -- evicts addr_a's line

    f.tick_mem(10);
    f.cache.handle_responses(fill, fill_valid);

    // Now addr_a should miss (evicted)
    wb.valid = false;
    f.cache.process_load(addr_a, 0, 3, results, 4, 0, 0, wb);
    REQUIRE_FALSE(wb.valid); // Miss!
    // load_misses = miss(addr_a) + miss(addr_b) + miss(addr_a again) = 3
    // load_hits = hit(addr_a first recheck) = 1
    REQUIRE(f.stats.load_misses == 3);
    REQUIRE(f.stats.load_hits == 1);
}

TEST_CASE("Cache: write buffer full stalls store", "[cache]") {
    CacheFixture f;
    WritebackEntry wb;
    std::array<uint32_t, WARP_SIZE> results{};

    // Install a cache line
    f.cache.process_load(0, 0, 1, results, 1, 0, 0, wb);
    f.tick_mem(10);
    WritebackEntry fill;
    bool fill_valid;
    f.cache.handle_responses(fill, fill_valid);

    // Fill write buffer with stores (all to same line, which is in cache)
    for (uint32_t i = 0; i < WB_DEPTH; ++i) {
        bool accepted = f.cache.process_store(0, 0, 1, 0, 0); // line_addr 0
        REQUIRE(accepted);
    }

    // Write buffer full: next store should stall
    bool accepted = f.cache.process_store(0, 0, 1, 0, 0);
    REQUIRE_FALSE(accepted);
    REQUIRE(f.stats.write_buffer_stall_cycles == 1);
}

TEST_CASE("Cache: multiple load fills are replayed across cycles", "[cache]") {
    CacheFixture f;
    WritebackEntry wb;
    std::array<uint32_t, WARP_SIZE> results_a{};
    std::array<uint32_t, WARP_SIZE> results_b{};
    results_a[0] = 11;
    results_b[0] = 22;

    REQUIRE(f.cache.process_load(0, 0, 5, results_a, 1, 0, 0, wb));
    REQUIRE(f.cache.process_load(LINE_SIZE * 100, 0, 6, results_b, 2, 0, 0, wb));

    f.tick_mem(10);

    WritebackEntry fill_wb;
    bool fill_valid = false;
    f.cache.handle_responses(fill_wb, fill_valid);
    REQUIRE(fill_valid);
    REQUIRE(fill_wb.dest_reg == 5);
    REQUIRE(fill_wb.values[0] == 11);

    f.cache.evaluate();
    fill_valid = false;
    f.cache.handle_responses(fill_wb, fill_valid);
    REQUIRE(fill_valid);
    REQUIRE(fill_wb.dest_reg == 6);
    REQUIRE(fill_wb.values[0] == 22);
}

TEST_CASE("Cache: store miss fill waits for write buffer space", "[cache]") {
    CacheFixture f;
    WritebackEntry wb;
    std::array<uint32_t, WARP_SIZE> results{};

    // Install a line so stores can fill the write buffer.
    REQUIRE(f.cache.process_load(0, 0, 1, results, 1, 0, 0, wb));
    f.tick_mem(10);
    WritebackEntry fill_wb;
    bool fill_valid = false;
    f.cache.handle_responses(fill_wb, fill_valid);
    REQUIRE(fill_valid);

    for (uint32_t i = 0; i < WB_DEPTH; ++i) {
        REQUIRE(f.cache.process_store(0, 0, 1, 0, 0));
    }

    REQUIRE(f.cache.process_store(1, 0, 1, 0, 0));
    f.tick_mem(10);

    fill_valid = false;
    f.cache.handle_responses(fill_wb, fill_valid);
    REQUIRE_FALSE(fill_valid);
    REQUIRE(f.cache.is_stalled());
    REQUIRE_FALSE(f.cache.is_idle());

    f.cache.drain_write_buffer();
    f.cache.evaluate();
    fill_valid = false;
    f.cache.handle_responses(fill_wb, fill_valid);
    REQUIRE_FALSE(fill_valid);
    REQUIRE_FALSE(f.cache.is_stalled());
}

TEST_CASE("Cache: reset clears all state", "[cache]") {
    CacheFixture f;
    WritebackEntry wb;
    std::array<uint32_t, WARP_SIZE> results{};

    // Install a line
    f.cache.process_load(0, 0, 1, results, 1, 0, 0, wb);
    f.tick_mem(10);
    WritebackEntry fill;
    bool fill_valid;
    f.cache.handle_responses(fill, fill_valid);

    // Verify it hits
    wb.valid = false;
    f.cache.process_load(0, 0, 1, results, 2, 0, 0, wb);
    REQUIRE(wb.valid);

    // Reset
    f.cache.reset();
    f.mem_if.reset();

    // Now it should miss
    wb.valid = false;
    f.cache.process_load(0, 0, 1, results, 3, 0, 0, wb);
    REQUIRE_FALSE(wb.valid);
}
