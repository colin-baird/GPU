#include "catch.hpp"
#include "gpu_sim/timing/load_gather_buffer.h"
#include "gpu_sim/timing/cache.h"
#include "gpu_sim/timing/coalescing_unit.h"
#include "gpu_sim/timing/ldst_unit.h"
#include "gpu_sim/timing/memory_interface.h"
#include "gpu_sim/stats.h"

using namespace gpu_sim;

namespace {

constexpr uint32_t NUM_WARPS = 4;
constexpr uint32_t LINE_SIZE = 128;
constexpr uint32_t CACHE_SIZE = 4096;
constexpr uint32_t NUM_MSHRS = 32;
constexpr uint32_t WB_DEPTH = 16;
constexpr uint32_t MAX_OUTSTANDING_WRITES = 64;
constexpr uint32_t MEM_LATENCY = 4;
constexpr uint32_t FULL_MASK = 0xFFFFFFFFu;

std::array<uint32_t, WARP_SIZE> make_values(uint32_t base) {
    std::array<uint32_t, WARP_SIZE> v{};
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        v[i] = base + i;
    }
    return v;
}

DispatchInput make_load_dispatch(uint32_t warp_id, uint8_t dest_reg,
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
    input.trace.mem_addresses = addrs;
    input.trace.decoded.rd = dest_reg;
    input.trace.decoded.has_rd = true;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        input.trace.results[i] = 100 + i;
        input.trace.mem_size[i] = 4;
    }
    return input;
}

} // namespace

TEST_CASE("LoadGatherBuffer: coalesced write fills all 32 slots in one cycle",
          "[gather_buffer]") {
    Stats stats;
    LoadGatherBufferFile file(NUM_WARPS, stats);

    file.claim(0, /*dest_reg=*/7, /*pc=*/0, /*issue_cycle=*/0, /*raw=*/0);
    // Phase M2: claim is REGISTERED. Drive commit + evaluate to apply.
    file.commit();
    file.evaluate();
    REQUIRE(file.current_busy(0));
    REQUIRE(file.buffer(0).filled_count == 0);

    auto values = make_values(42);
    REQUIRE(file.try_write(0, FULL_MASK, values,
                           LoadGatherBufferFile::GatherWriteSource::FILL));

    // Phase 10D: the fill lands in next_buffers_; commit() flips it into the
    // committed buffer state that buffer() and current_has_result() expose.
    file.commit();
    REQUIRE(file.buffer(0).filled_count == WARP_SIZE);
    REQUIRE(file.current_has_result());

    WritebackEntry wb = file.consume_result();
    REQUIRE(wb.valid);
    REQUIRE(wb.warp_id == 0);
    REQUIRE(wb.dest_reg == 7);
    REQUIRE(wb.values[0] == 42);
    REQUIRE(wb.values[31] == 73);
}

TEST_CASE("LoadGatherBuffer: partial write populates only selected slots",
          "[gather_buffer]") {
    Stats stats;
    LoadGatherBufferFile file(NUM_WARPS, stats);
    file.claim(0, 1, 0, 0, 0);
    // Phase M2: apply REGISTERED claim before exercising try_write.
    file.commit();
    file.evaluate();

    // Mask with 8 set bits (lanes 0, 4, 7, 12, 17, 20, 25, 31).
    const uint32_t mask = (1u << 0) | (1u << 4) | (1u << 7) | (1u << 12) |
                          (1u << 17) | (1u << 20) | (1u << 25) | (1u << 31);
    auto values = make_values(1000);
    REQUIRE(file.try_write(0, mask, values,
                           LoadGatherBufferFile::GatherWriteSource::FILL));

    // Phase 10D: commit() flips the partial fill into the committed buffer.
    file.commit();
    const auto& buf = file.buffer(0);
    REQUIRE(buf.filled_count == 8);
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        bool expected = (mask >> i) & 1u;
        REQUIRE(buf.slot_valid[i] == expected);
        if (expected) {
            REQUIRE(buf.values[i] == 1000 + i);
        }
    }

    // Writeback must not fire yet.
    REQUIRE_FALSE(file.current_has_result());
}

TEST_CASE("LoadGatherBuffer: writeback withheld until all 32 slots valid",
          "[gather_buffer]") {
    // Serialized load where lane 0 hits an installed line and lanes 1-31
    // each miss into distinct lines. The writeback must not appear until
    // the last MSHR fill lands — this is the primary correctness property
    // of the gather-buffer refactor.
    Stats stats;
    FixedLatencyMemory mem_if(MEM_LATENCY, stats);
    LoadGatherBufferFile gather_file(NUM_WARPS, stats);
    // Phase 4 (close-the-Reg-family-migration): cross-stage response /
    // write-ack RegFifos (TimingModel-owned in production); local here and
    // committed in lockstep with mem_if.
    RegFifo<MemoryResponse> mem_responses;
    RegFifo<MemoryResponse> mem_write_acks;
    mem_if.set_response_queues(&mem_responses, &mem_write_acks);
    auto commit_mem_fifos = [&]() {
        mem_responses.commit();
        mem_write_acks.commit();
    };
    L1Cache cache(CACHE_SIZE, LINE_SIZE, NUM_MSHRS, WB_DEPTH, MAX_OUTSTANDING_WRITES,
                  mem_if, gather_file, stats);

    // Pre-install the line that lane 0's address will map to, via a separate
    // warp so we don't perturb warp 0's gather buffer.
    auto results = make_values(0);
    gather_file.claim(/*warp_id=*/1, 3, 0, 0, 0);
    // Phase M2: apply REGISTERED claim before driving the cache request.
    gather_file.commit();
    gather_file.evaluate();
    REQUIRE(cache.process_load(0, 1, FULL_MASK, results, 1, 0, 0));
    // Phase M5: drain the staged read request into in_flight_ before ticking.
    mem_if.commit();
    for (uint32_t i = 0; i < MEM_LATENCY; ++i) mem_if.evaluate();
    // Phase 4 (close-the-Reg-family-migration): commit the cross-stage
    // response FIFO so mem_if's stage_push lands in current_ and the
    // cache's handle_responses can read it.
    commit_mem_fifos();
    cache.handle_responses();
    // Phase 10D: the FILL landed in next_buffers_; commit() flips it into the
    // committed buffer so consume_result() (a pure committed read) sees the
    // full buffer and stages its release.
    gather_file.commit();
    (void)gather_file.consume_result();
    cache.commit();
    commit_mem_fifos();
    // Apply the staged release so warp 1's buffer is free for the rest of
    // the test.
    gather_file.commit();

    // Now issue a serialized load on warp 0: lane 0 hits line 0, lanes 1..31
    // miss into distinct lines.
    gather_file.claim(/*warp_id=*/0, /*dest_reg=*/9, 0, 0, 0);
    // Phase M2: apply REGISTERED claim.
    gather_file.commit();
    gather_file.evaluate();
    auto values = make_values(500);

    // Lane 0 — hit.
    REQUIRE(cache.process_load(/*addr=*/0, 0, /*mask=*/1u, values, 1, 0, 0));
    // Phase 10D: the HIT fill lands in next_buffers_; commit() to observe it.
    gather_file.commit();
    REQUIRE(gather_file.buffer(0).filled_count == 1);
    REQUIRE_FALSE(gather_file.current_has_result());

    // Lanes 1..31 — each a miss on a distinct line. Phase M5: each
    // miss stages a request in mem_if's next_read_request_; commit +
    // evaluate between issues drains the previous request into
    // in_flight_ before the next set_next_read_request would overwrite
    // the staging slot.
    // registered-mshr-write-buffer: each miss allocates an MSHR into
    // next_entries_; cache.commit() between issues flips it into
    // current_entries_ so the next allocate() scans a consistent file and
    // picks a fresh free slot (without it every allocate would reuse slot 0).
    for (uint32_t lane = 1; lane < WARP_SIZE; ++lane) {
        uint32_t addr = static_cast<uint32_t>(lane) * LINE_SIZE * 100;
        uint32_t mask = 1u << lane;
        REQUIRE(cache.process_load(addr, 0, mask, values, 1, 0, 0));
        cache.commit();
        mem_if.commit();
        mem_if.evaluate();
        // Phase 4 (close-the-Reg-family-migration): each loop iteration is
        // one cycle — commit the cross-stage response FIFO so the prior
        // iteration's stage_push lands before the next iteration's
        // stage_push (RegFifo holds at most one staged push per cycle).
        commit_mem_fifos();
        gather_file.commit();
    }

    // At this point 31 MSHRs are outstanding; only slot 0 is filled.
    REQUIRE(gather_file.buffer(0).filled_count == 1);
    REQUIRE_FALSE(gather_file.current_has_result());

    // Drain the fills one cycle at a time, verifying the writeback does not
    // fire early.
    // cache.evaluate() internally runs handle_responses(); calling both in
    // the same cycle would try two FILL writes per cycle and the second
    // would collide on the shared write port and silently drop. One call
    // per cycle only.
    uint32_t fills_completed = 0;
    for (uint32_t cycle = 0; cycle < 200 && fills_completed < WARP_SIZE - 1;
         ++cycle) {
        mem_if.evaluate();
        // Phase 4 (close-the-Reg-family-migration): commit the cross-stage
        // response FIFO between mem_if.evaluate's stage_push and the
        // following cache.evaluate's read — production runs this in the
        // commit_cross_stage_fifos pass at end of tick. Also commit again
        // after cache.evaluate so the stage_response_pop is applied before
        // the next iteration's mem_if.evaluate.
        commit_mem_fifos();
        cache.evaluate();
        commit_mem_fifos();
        gather_file.commit();
        uint32_t filled_now = gather_file.buffer(0).filled_count;
        if (filled_now < WARP_SIZE) {
            REQUIRE_FALSE(gather_file.current_has_result());
        }
        fills_completed = filled_now - 1;
    }

    REQUIRE(gather_file.buffer(0).filled_count == WARP_SIZE);
    REQUIRE(gather_file.current_has_result());
}

TEST_CASE("LoadGatherBuffer: gather_buffer_stall_cycles increments when warp is busy",
          "[gather_buffer]") {
    // Directly force the coalescing unit to encounter a busy gather buffer
    // on the target warp. Under normal operation the scoreboard prevents
    // this — this test verifies the structural safety net.
    Stats stats;
    FixedLatencyMemory mem_if(MEM_LATENCY, stats);
    LoadGatherBufferFile gather_file(NUM_WARPS, stats);
    L1Cache cache(CACHE_SIZE, LINE_SIZE, NUM_MSHRS, WB_DEPTH, MAX_OUTSTANDING_WRITES,
                  mem_if, gather_file, stats);
    LdStUnit ldst(8, 4, stats);
    CoalescingUnit coal(ldst, cache, gather_file, LINE_SIZE, stats);
    // Phase 3 (close-the-Reg-family-migration): wire the cross-stage addr-
    // gen FIFO (normally owned by TimingModel) and commit it explicitly in
    // the fixture's tick sequence.
    RegFifo<AddrGenFIFOEntry> addr_gen_fifo;
    ldst.set_addr_gen_fifo(&addr_gen_fifo);
    coal.set_addr_gen_fifo(&addr_gen_fifo);

    // Mark warp 0's gather buffer busy without issuing a load.
    gather_file.claim(0, 1, 0, 0, 0);
    // Phase M2: apply REGISTERED claim.
    gather_file.commit();
    gather_file.evaluate();
    REQUIRE(gather_file.current_busy(0));

    std::array<uint32_t, WARP_SIZE> addrs;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) addrs[i] = 256 + i * 4;
    auto input = make_load_dispatch(0, 5, addrs);
    ldst.accept(input, 1);
    for (int i = 0; i < 16 && ldst.current_fifo_empty(); ++i) {
        ldst.evaluate();
        ldst.commit();
        addr_gen_fifo.commit();
    }
    REQUIRE_FALSE(ldst.current_fifo_empty());

    coal.evaluate();
    coal.commit();
    addr_gen_fifo.commit();

    REQUIRE(stats.gather_buffer_stall_cycles == 1);
    // FIFO entry must remain (not popped) because the buffer was busy.
    REQUIRE_FALSE(ldst.current_fifo_empty());
    REQUIRE(stats.coalesced_requests == 0);
    REQUIRE(stats.serialized_requests == 0);
}

TEST_CASE("LoadGatherBuffer: consume_result releases buffer and re-claim succeeds",
          "[gather_buffer]") {
    Stats stats;
    LoadGatherBufferFile file(NUM_WARPS, stats);

    file.claim(2, 11, 0, 0, 0);
    // Phase M2: apply REGISTERED claim.
    file.commit();
    file.evaluate();
    auto values = make_values(1);
    REQUIRE(file.try_write(2, FULL_MASK, values,
                           LoadGatherBufferFile::GatherWriteSource::FILL));
    // Phase M4: latch staged has_result via commit before observing.
    file.commit();
    REQUIRE(file.current_has_result());

    WritebackEntry wb = file.consume_result();
    REQUIRE(wb.valid);

    // Phase 10D: consume_result() is a pure read — it stages the buffer
    // release into next_release_; commit() applies it.
    file.commit();
    const auto& buf = file.buffer(2);
    REQUIRE_FALSE(buf.busy);
    REQUIRE(buf.filled_count == 0);
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        REQUIRE_FALSE(buf.slot_valid[i]);
    }
    REQUIRE_FALSE(file.current_busy(2));

    // Re-claim must succeed (no assert, buffer clean).
    file.claim(2, 13, 0, 0, 0);
    // Phase M2: apply REGISTERED claim.
    file.commit();
    file.evaluate();
    REQUIRE(file.current_busy(2));
}

TEST_CASE("LoadGatherBuffer: round-robin emission across two completed buffers",
          "[gather_buffer]") {
    Stats stats;
    LoadGatherBufferFile file(NUM_WARPS, stats);

    // Phase 7: the gather-buffer write port is a single shared resource
    // (spec §5.3 Port model — one line-to-gather-buffer extraction per
    // cycle). Two buffers completing therefore requires two cycles of
    // try_write with a commit between them. rr_pointer_ starts at 0, so
    // consume_result() should return warp 1 first, then warp 3.
    // Phase M2: each REGISTERED claim takes a commit + evaluate to apply.
    // The single-slot claim_request can hold only one pending claim at a
    // time, so we drive the cycle boundary between consecutive claims.
    file.claim(1, 5, 0, 0, 0);
    file.commit();
    file.evaluate();
    file.claim(3, 7, 0, 0, 0);
    file.commit();
    file.evaluate();
    auto v1 = make_values(100);
    auto v3 = make_values(200);
    REQUIRE(file.try_write(1, FULL_MASK, v1,
                           LoadGatherBufferFile::GatherWriteSource::FILL));
    file.commit();
    REQUIRE(file.try_write(3, FULL_MASK, v3,
                           LoadGatherBufferFile::GatherWriteSource::FILL));
    file.commit();

    REQUIRE(file.current_has_result());
    WritebackEntry first = file.consume_result();
    REQUIRE(first.warp_id == 1);
    REQUIRE(first.dest_reg == 5);
    // Phase M4: recompute REGISTERED has_result via commit between consumes.
    file.commit();

    REQUIRE(file.current_has_result());
    WritebackEntry second = file.consume_result();
    REQUIRE(second.warp_id == 3);
    REQUIRE(second.dest_reg == 7);
    file.commit();

    REQUIRE_FALSE(file.current_has_result());

    // Next completed buffer after warp 3 should wrap back and prefer the
    // lowest index first: complete warp 0 and warp 2 simultaneously. After
    // the previous emission rr_pointer_ is (3 + 1) % 4 = 0, so warp 0 wins.
    // Phase M2: drive cycle boundary between consecutive claims.
    file.claim(0, 9, 0, 0, 0);
    file.commit();
    file.evaluate();
    file.claim(2, 11, 0, 0, 0);
    file.commit();
    file.evaluate();
    auto v0 = make_values(300);
    auto v2 = make_values(400);
    REQUIRE(file.try_write(0, FULL_MASK, v0,
                           LoadGatherBufferFile::GatherWriteSource::FILL));
    file.commit();
    REQUIRE(file.try_write(2, FULL_MASK, v2,
                           LoadGatherBufferFile::GatherWriteSource::FILL));
    file.commit();

    WritebackEntry third = file.consume_result();
    REQUIRE(third.warp_id == 0);
    // Phase 10D: consume_result() stages the release (and the round-robin
    // pointer advance) for commit(). commit() between the two consumes
    // applies warp 0's release and advances rr_pointer_ so the next scan
    // selects warp 2.
    file.commit();
    WritebackEntry fourth = file.consume_result();
    REQUIRE(fourth.warp_id == 2);
}

// Hit-vs-fill port conflict: setting up a same-cycle HIT and FILL to the
// same buffer through the full cache/coalescing stack is too
// implementation-coupled to arrange reliably from a test. Exercise the
// primitive directly at the LoadGatherBufferFile interface instead — this
// verifies the port arbitration semantics that the cache relies on.
TEST_CASE("LoadGatherBuffer: FILL wins port over same-cycle HIT", "[gather_buffer]") {
    Stats stats;
    LoadGatherBufferFile file(NUM_WARPS, stats);
    file.claim(0, 3, 0, 0, 0);
    // Phase M2: apply REGISTERED claim.
    file.commit();
    file.evaluate();

    // FILL runs first (cache.cpp schedules handle_responses before hit-path
    // writes), so the port is consumed by the FILL. A same-cycle HIT must
    // return false and bump the port-conflict counter.
    auto fill_vals = make_values(10);
    REQUIRE(file.try_write(0, 0x1u, fill_vals,
                           LoadGatherBufferFile::GatherWriteSource::FILL));

    auto hit_vals = make_values(999);
    REQUIRE_FALSE(file.try_write(0, 0x2u, hit_vals,
                                 LoadGatherBufferFile::GatherWriteSource::HIT));
    REQUIRE(stats.gather_buffer_port_conflict_cycles == 1);

    // FILL's write should be the one that landed; HIT's must not have
    // overwritten or added to slot 1. Phase 10D: the fill landed in
    // next_buffers_; commit() flips it into the committed buffer.
    file.commit();
    const auto& buf = file.buffer(0);
    REQUIRE(buf.slot_valid[0]);
    REQUIRE(buf.values[0] == 10);
    REQUIRE_FALSE(buf.slot_valid[1]);
    REQUIRE(buf.filled_count == 1);

    // Next cycle the HIT should succeed.
    REQUIRE(file.try_write(0, 0x2u, hit_vals,
                           LoadGatherBufferFile::GatherWriteSource::HIT));
    // Phase 10D: commit() to observe the HIT fill.
    file.commit();
    REQUIRE(buf.slot_valid[1]);
    REQUIRE(buf.values[1] == 1000);
    REQUIRE(buf.filled_count == 2);
}
