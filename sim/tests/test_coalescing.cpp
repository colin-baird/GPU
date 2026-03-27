#include "catch.hpp"
#include "gpu_sim/timing/coalescing_unit.h"
#include "gpu_sim/timing/ldst_unit.h"
#include "gpu_sim/timing/cache.h"
#include "gpu_sim/timing/memory_interface.h"
#include "gpu_sim/stats.h"

using namespace gpu_sim;

static const uint32_t LINE_SIZE = 128;

// Build a DispatchInput for a load instruction with given addresses per lane
static DispatchInput make_load_dispatch(uint32_t warp_id, uint8_t dest_reg,
                                         const std::array<uint32_t, WARP_SIZE>& addrs) {
    DispatchInput input;
    input.warp_id = warp_id;
    input.pc = 0;
    input.decoded.type = InstructionType::LOAD;
    input.decoded.target_unit = ExecUnit::LDST;
    input.decoded.has_rd = true;
    input.decoded.rd = dest_reg;
    input.decoded.mem_op = MemOp::LW;
    input.trace.warp_id = warp_id;
    input.trace.is_load = true;
    input.trace.is_store = false;
    input.trace.mem_addresses = addrs;
    input.trace.decoded.rd = dest_reg;
    input.trace.decoded.has_rd = true;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        input.trace.results[i] = 100 + i;
        input.trace.mem_size[i] = 4;
    }
    return input;
}

TEST_CASE("Coalescing: contiguous addresses coalesce to 1 request", "[coalescing]") {
    Stats stats;
    ExternalMemoryInterface mem_if(10, stats);
    L1Cache cache(4096, LINE_SIZE, 4, 4, mem_if, stats);
    LdStUnit ldst(8, 4, stats);
    CoalescingUnit coal(ldst, cache, LINE_SIZE, stats);

    // All 32 threads load from within the same 128-byte cache line
    std::array<uint32_t, WARP_SIZE> addrs;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        addrs[i] = 256 + (i * 4); // 256..256+124, all in one 128B line (256/128=2)
    }

    auto input = make_load_dispatch(0, 5, addrs);
    ldst.accept(input, 1);

    // Process the ldst unit until entry appears in FIFO
    for (int i = 0; i < 10; ++i) {
        ldst.evaluate();
        ldst.commit();
        if (!ldst.fifo_empty()) break;
    }
    REQUIRE_FALSE(ldst.fifo_empty());

    // Run coalescing
    WritebackEntry wb;
    bool wb_valid;
    coal.evaluate(wb, wb_valid);

    REQUIRE(stats.coalesced_requests == 1);
    REQUIRE(stats.serialized_requests == 0);
}

TEST_CASE("Coalescing: scattered addresses serialize to 32 requests", "[coalescing]") {
    Stats stats;
    ExternalMemoryInterface mem_if(10, stats);
    L1Cache cache(4096, LINE_SIZE, 4, 4, mem_if, stats);
    LdStUnit ldst(8, 4, stats);
    CoalescingUnit coal(ldst, cache, LINE_SIZE, stats);

    // Each thread loads from a different cache line
    std::array<uint32_t, WARP_SIZE> addrs;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        addrs[i] = i * LINE_SIZE * 100; // Far apart, different lines
    }

    auto input = make_load_dispatch(0, 5, addrs);
    ldst.accept(input, 1);

    // Process ldst unit
    for (int i = 0; i < 10; ++i) {
        ldst.evaluate();
        ldst.commit();
        if (!ldst.fifo_empty()) break;
    }
    REQUIRE_FALSE(ldst.fifo_empty());

    // Run coalescing - first evaluate starts processing
    WritebackEntry wb;
    bool wb_valid;
    coal.evaluate(wb, wb_valid);

    REQUIRE(stats.serialized_requests == 1);
    REQUIRE(stats.coalesced_requests == 0);
}

TEST_CASE("Coalescing: boundary check - last byte in different line", "[coalescing]") {
    Stats stats;
    ExternalMemoryInterface mem_if(10, stats);
    L1Cache cache(4096, LINE_SIZE, 4, 4, mem_if, stats);
    LdStUnit ldst(8, 4, stats);
    CoalescingUnit coal(ldst, cache, LINE_SIZE, stats);

    // 31 threads in same line, 1 thread in different line -> serialized
    std::array<uint32_t, WARP_SIZE> addrs;
    for (uint32_t i = 0; i < WARP_SIZE - 1; ++i) {
        addrs[i] = 0 + (i * 4);
    }
    addrs[WARP_SIZE - 1] = LINE_SIZE; // Different line

    auto input = make_load_dispatch(0, 5, addrs);
    ldst.accept(input, 1);

    for (int i = 0; i < 10; ++i) {
        ldst.evaluate();
        ldst.commit();
        if (!ldst.fifo_empty()) break;
    }

    WritebackEntry wb;
    bool wb_valid;
    coal.evaluate(wb, wb_valid);

    REQUIRE(stats.serialized_requests == 1);
    REQUIRE(stats.coalesced_requests == 0);
}
