#include "catch.hpp"
#include "gpu_sim/timing/cache.h"
#include "gpu_sim/timing/dramsim3_memory.h"
#include "gpu_sim/timing/load_gather_buffer.h"
#include "gpu_sim/config.h"
#include "gpu_sim/stats.h"

#include <array>
#include <cstdint>
#include <deque>
#include <set>
#include <vector>

#ifndef DRAMSIM3_TEST_INI_PATH
#error "DRAMSIM3_TEST_INI_PATH must be defined by the build system"
#endif

using namespace gpu_sim;

namespace {

constexpr uint32_t MAX_EVALUATES = 200000;

// Build a SimConfig that selects the DE-10 Nano DDR3 model and slots a
// fresh DRAMSim3 output dir per test so multiple cases don't collide.
SimConfig make_dramsim3_config(uint32_t num_mshrs = 4,
                               const char* tag = "default") {
    SimConfig c;
    c.memory_backend       = "dramsim3";
    c.dramsim3_config_path = DRAMSIM3_TEST_INI_PATH;
    c.dramsim3_output_dir  = std::string("/tmp/gpu_sim_dramsim3_test_") + tag;
    c.num_mshrs            = num_mshrs;
    c.fpga_clock_mhz       = 150.0;
    c.dram_clock_mhz       = 400.0;
    c.cache_line_size_bytes      = 128;
    c.dramsim3_bytes_per_burst   = 32;
    c.dramsim3_request_fifo_depth = 64;  // generous so step 6 never backpressures
    c.validate();
    return c;
}

// Pump evaluate() until either `mem.has_response()` or a hard cap. Returns
// the number of fabric cycles elapsed.
uint32_t pump_until_response(DRAMSim3Memory& mem) {
    uint32_t n = 0;
    while (!mem.has_response() && n < MAX_EVALUATES) {
        mem.evaluate();
        ++n;
    }
    return n;
}

// Pump evaluate() until every in-flight operation has produced a response
// (or a hard cap). Does NOT drain the response queue — that's the caller's
// job, since `is_idle()` requires the response queue to be empty too.
uint32_t pump_until_drained(DRAMSim3Memory& mem) {
    uint32_t n = 0;
    while (mem.in_flight_count() > 0 && n < MAX_EVALUATES) {
        mem.evaluate();
        ++n;
    }
    return n;
}

} // namespace

TEST_CASE("DRAMSim3Memory: constructs with the DE-10 Nano .ini",
          "[dramsim3_memory]") {
    Stats stats;
    SimConfig cfg = make_dramsim3_config(4, "construct");
    DRAMSim3Memory mem(cfg, stats);

    CHECK(mem.chunks_per_line() == 4u);     // 128 / 32
    CHECK(mem.is_idle());
    CHECK(mem.in_flight_count() == 0u);
    CHECK(mem.response_count() == 0u);
}

TEST_CASE("DRAMSim3Memory: chunk reassembly emits one response per line",
          "[dramsim3_memory]") {
    Stats stats;
    SimConfig cfg = make_dramsim3_config(4, "chunkreassembly");
    DRAMSim3Memory mem(cfg, stats);

    constexpr uint32_t LINE_ADDR = 0x10;
    constexpr uint32_t MSHR_ID   = 2;
    REQUIRE(mem.submit_read(LINE_ADDR, MSHR_ID));
    CHECK(mem.in_flight_count() == 1u);

    const uint32_t cycles = pump_until_response(mem);
    REQUIRE(mem.has_response());

    // A 128 B line moves as 4 BL8 transactions over a 32-bit DDR3 bus.
    // Each BL8 occupies 4 tCK of data-bus time (8 beats / 2 beats-per-tCK).
    // So the data bus alone consumes 4 x 4 = 16 tCK, regardless of CAS or
    // ACT scheduling — the response cannot arrive sooner than that.
    CHECK(mem.dram_ticks() >= 16u);
    CHECK(cycles > 0u);

    auto resp = mem.get_response();
    CHECK(resp.line_addr == LINE_ADDR);
    CHECK(resp.mshr_id   == MSHR_ID);
    CHECK_FALSE(resp.is_write);

    // Exactly one response — no duplicates from chunk reassembly.
    CHECK_FALSE(mem.has_response());
    CHECK(stats.external_memory_reads == 1u);
}

TEST_CASE("DRAMSim3Memory: 8 reads with distinct mshr_ids each return once",
          "[dramsim3_memory]") {
    Stats stats;
    SimConfig cfg = make_dramsim3_config(8, "mshrmux");
    DRAMSim3Memory mem(cfg, stats);

    constexpr uint32_t N = 8;
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t line_addr = 0x100 + i * 4;  // disjoint lines
        REQUIRE(mem.submit_read(line_addr, i));
    }
    CHECK(mem.in_flight_count() == N);

    pump_until_drained(mem);

    std::set<uint32_t> seen_mshrs;
    std::set<uint32_t> seen_lines;
    while (mem.has_response()) {
        auto r = mem.get_response();
        CHECK_FALSE(r.is_write);
        CHECK(seen_mshrs.insert(r.mshr_id).second);
        seen_lines.insert(r.line_addr);
    }
    CHECK(seen_mshrs.size() == N);
    CHECK(seen_lines.size() == N);
    CHECK(mem.is_idle());
}

TEST_CASE("DRAMSim3Memory: sequential reads complete faster than thrashing strided reads",
          "[dramsim3_memory]") {
    auto run_pattern = [](const std::vector<uint32_t>& line_addrs,
                          const char* tag) {
        Stats stats;
        SimConfig cfg = make_dramsim3_config(
            static_cast<uint32_t>(line_addrs.size()), tag);
        DRAMSim3Memory mem(cfg, stats);

        for (uint32_t i = 0; i < line_addrs.size(); ++i) {
            REQUIRE(mem.submit_read(line_addrs[i], i));
        }
        return pump_until_drained(mem);
    };

    constexpr uint32_t N = 8;

    // Sequential: consecutive cache lines. With OPEN_PAGE policy these
    // typically share row buffers and a small handful of banks, so the
    // controller can stream through them with few row-activate cycles.
    std::vector<uint32_t> sequential;
    for (uint32_t i = 0; i < N; ++i) sequential.push_back(i);

    // Strided: huge line stride forces each access into a fresh row in a
    // bank that's likely just been used by a previous access — every
    // access pays the precharge + activate penalty.
    std::vector<uint32_t> strided;
    for (uint32_t i = 0; i < N; ++i) strided.push_back(i * 8192);

    const uint32_t seq_cycles = run_pattern(sequential, "seqlat");
    const uint32_t str_cycles = run_pattern(strided, "strlat");

    INFO("sequential cycles=" << seq_cycles
         << " strided cycles=" << str_cycles);
    CHECK(seq_cycles < str_cycles);
}

TEST_CASE("DRAMSim3Memory: phase accumulator tracks the fabric/DRAM clock ratio",
          "[dramsim3_memory]") {
    Stats stats;
    SimConfig cfg = make_dramsim3_config(4, "clockratio");
    cfg.fpga_clock_mhz = 150.0;
    cfg.dram_clock_mhz = 400.0;
    DRAMSim3Memory mem(cfg, stats);

    constexpr uint32_t FABRIC_CYCLES = 1000;
    for (uint32_t i = 0; i < FABRIC_CYCLES; ++i) mem.evaluate();

    const double expected = FABRIC_CYCLES * (400.0 / 150.0);  // ~2666.667
    const long actual = static_cast<long>(mem.dram_ticks());
    CHECK(std::abs(actual - static_cast<long>(expected)) <= 1);
}

TEST_CASE("DRAMSim3Memory: idle goes true after drain; reset clears state",
          "[dramsim3_memory]") {
    Stats stats;
    SimConfig cfg = make_dramsim3_config(4, "idlereset");
    DRAMSim3Memory mem(cfg, stats);

    REQUIRE(mem.submit_read(0x40, 0));
    REQUIRE(mem.submit_read(0x80, 1));
    CHECK_FALSE(mem.is_idle());

    pump_until_drained(mem);
    while (mem.has_response()) (void)mem.get_response();
    CHECK(mem.is_idle());
    CHECK(mem.in_flight_count() == 0u);
    CHECK(mem.response_count() == 0u);
    CHECK(mem.request_fifo_size() == 0u);

    // Re-submit and reset mid-flight. After reset everything must be clear.
    REQUIRE(mem.submit_read(0x100, 0));
    REQUIRE(mem.submit_read(0x140, 1));
    REQUIRE(mem.submit_write(0x200));
    CHECK(mem.in_flight_count() > 0u);

    mem.reset();
    CHECK(mem.is_idle());
    CHECK(mem.in_flight_count() == 0u);
    CHECK(mem.response_count() == 0u);
    CHECK(mem.request_fifo_size() == 0u);
    CHECK(mem.dram_ticks() == 0u);

    // After reset, the backend still functions: a fresh read drains.
    REQUIRE(mem.submit_read(0x300, 0));
    pump_until_drained(mem);
    REQUIRE(mem.has_response());
    auto r = mem.get_response();
    CHECK(r.line_addr == 0x300u);
    CHECK(r.mshr_id   == 0u);
    CHECK_FALSE(r.is_write);
}

TEST_CASE("DRAMSim3Memory: worst-case cache traffic never drops requests",
          "[dramsim3_memory]") {
    // Architectural requirements under test:
    //   - The request FIFO is sized exactly to (num_mshrs + write_buffer_depth)
    //     * chunks_per_line — the worst-case in-flight chunk count the cache
    //     can hold against the backend at one instant. With the cache
    //     respecting submit_*'s bool return (drain_write_buffer pops only on
    //     success), the FIFO never overflows and no request is dropped.
    //   - The response queue stays within `response_queue_capacity_` =
    //     num_mshrs + write_buffer_depth + chunks_per_line, asserted at every
    //     push site in DRAMSim3Memory.
    //
    // Drives the backend at peak production: every cycle, allocate any free
    // MSHR for a fresh read AND attempt to drain one entry from a wb_depth-
    // deep "cache write buffer" of distinct lines. Distinct lines defeat
    // write_assembly collapsing so the bound is exercised, not bypassed.
    // Cache-side drain models L1Cache::handle_responses (all leading writes
    // + at most one read per cycle) and L1Cache::drain_write_buffer (pop only
    // when submit succeeds — the architectural contract this test certifies).
    Stats stats;
    SimConfig cfg = make_dramsim3_config(4, "stress");
    cfg.write_buffer_depth = 4;
    const uint32_t chunks_per_line =
        cfg.cache_line_size_bytes / cfg.dramsim3_bytes_per_burst;
    cfg.dramsim3_request_fifo_depth =
        (cfg.num_mshrs + cfg.write_buffer_depth) * chunks_per_line;
    cfg.validate();
    DRAMSim3Memory mem(cfg, stats);

    constexpr uint32_t kReadsToIssue  = 256;
    constexpr uint32_t kWritesToIssue = 256;
    constexpr uint32_t kCycleCap      = 200000;

    std::vector<bool> mshr_busy(cfg.num_mshrs, false);
    uint32_t reads_submitted  = 0;
    uint32_t writes_submitted = 0;
    uint32_t reads_completed  = 0;
    uint32_t writes_completed = 0;
    uint32_t reads_dropped    = 0;
    uint32_t writes_dropped   = 0;
    uint32_t write_submit_rejected = 0;
    uint32_t read_submit_rejected  = 0;
    std::deque<uint32_t> write_buffer;  // models L1Cache::write_buffer_
    std::set<uint32_t> writes_in_flight;  // line addrs popped from buffer

    auto next_free_mshr = [&]() -> int {
        for (uint32_t i = 0; i < cfg.num_mshrs; ++i) {
            if (!mshr_busy[i]) return static_cast<int>(i);
        }
        return -1;
    };

    for (uint32_t cycle = 0; cycle < kCycleCap; ++cycle) {
        // Cache-side drain (equivalent of L1Cache::handle_responses): consume
        // all leading writes, then at most one read per cycle.
        while (mem.has_response()) {
            const auto resp = mem.get_response();
            if (resp.is_write) {
                REQUIRE(writes_in_flight.erase(resp.line_addr) == 1u);
                ++writes_completed;
                continue;
            }
            REQUIRE(resp.mshr_id < cfg.num_mshrs);
            REQUIRE(mshr_busy[resp.mshr_id]);
            mshr_busy[resp.mshr_id] = false;
            ++reads_completed;
            break;
        }

        // Issue one read into any free MSHR. The architecture guarantees the
        // request FIFO has room for a fresh read whenever an MSHR is free,
        // because the cache only ever has num_mshrs reads in flight; the
        // remaining (FIFO - num_mshrs * chunks_per_line) chunks are reserved
        // for writes which are bounded by the wb-depth credit.
        if (reads_submitted < kReadsToIssue) {
            const int slot = next_free_mshr();
            if (slot >= 0) {
                const uint32_t line_addr = 0x10000u + reads_submitted;
                const bool ok = mem.submit_read(line_addr,
                                                static_cast<uint32_t>(slot));
                if (ok) {
                    mshr_busy[slot] = true;
                    ++reads_submitted;
                } else {
                    ++read_submit_rejected;  // must remain 0
                }
            }
        }

        // Refill the model write buffer (cache stores), then drain at most one
        // entry to mem_if. Pop only on successful submit (architectural
        // contract — without this, writes leak into a silently-overflowing
        // FIFO).
        while (write_buffer.size() < cfg.write_buffer_depth &&
               (writes_submitted + write_buffer.size()) < kWritesToIssue) {
            write_buffer.push_back(0x40000u + writes_submitted +
                                   static_cast<uint32_t>(write_buffer.size()));
        }
        if (!write_buffer.empty()) {
            const uint32_t addr = write_buffer.front();
            if (mem.submit_write(addr)) {
                write_buffer.pop_front();
                writes_in_flight.insert(addr);
                ++writes_submitted;
            } else {
                ++write_submit_rejected;
            }
        }

        mem.evaluate();

        if (reads_completed == kReadsToIssue &&
            writes_completed == kWritesToIssue &&
            reads_submitted == kReadsToIssue &&
            writes_submitted == kWritesToIssue) {
            break;
        }
    }

    // Architectural promise: every submitted request reached completion. No
    // silent drops. Reads and writes balance: nothing leaked.
    CHECK(reads_completed  == kReadsToIssue);
    CHECK(writes_completed == kWritesToIssue);
    CHECK(reads_dropped    == 0u);
    CHECK(writes_dropped   == 0u);

    // submit_read must NEVER be rejected — at most num_mshrs reads can be
    // outstanding, and the FIFO reserves space for them by construction.
    CHECK(read_submit_rejected == 0u);

    // submit_write may transiently reject when the cache pumps faster than
    // DRAM retires; the cache MUST then leave the entry in write_buffer and
    // retry next cycle (modeled above). Sanity: at peak load, this should
    // happen at least sometimes — otherwise we're not actually exercising
    // the bound, only the easy path.
    CHECK(write_submit_rejected > 0u);

    // The response queue stayed within the architectural bound throughout.
    CHECK(mem.max_observed_response_queue() <= mem.response_queue_capacity());
    CHECK(mem.max_observed_response_queue() > 0u);  // sanity: the test ran

    // The backend is fully drained at the end.
    CHECK(mem.is_idle());
    CHECK(mem.in_flight_count() == 0u);
    CHECK(mem.request_fifo_size() == 0u);
}

TEST_CASE("DRAMSim3Memory + L1Cache: write-region saturation propagates to cache stall",
          "[dramsim3_memory]") {
    // End-to-end backpressure chain under DRAMSim3:
    //   DRAMSim3 write region full
    //     -> submit_write returns false
    //     -> L1Cache::drain_write_buffer pops only on success (write_buffer_
    //        stays at write_buffer_depth)
    //     -> L1Cache::process_store sees write_buffer full, sets stall_reason
    //        = WRITE_BUFFER_FULL, increments write_buffer_stall_cycles
    //
    // The remaining propagation (coalescing -> ldst -> scheduler -> warp trace
    // WAIT_L1_WRITE_BUFFER) is structural: each stage forwards `is_stalled()`
    // or its own busy bit unchanged. CoalescingUnit::evaluate early-returns on
    // cache.is_stalled(); LdStUnit::ready_out() returns !busy_ which stays high
    // when addr_gen_fifo is full; the warp scheduler's unit-ready check stalls
    // any LDST issuer; build_cycle_snapshot maps stall_reason ==
    // WRITE_BUFFER_FULL to WarpRestReason::WAIT_L1_WRITE_BUFFER (timing_model.cpp
    // ~line 597). Those mappings are 1-to-1 conditional forwards verified by
    // code review and by the existing cache-level test_cache.cpp:246; this
    // case exercises the part that is unique to step 8 — that DRAMSim3's
    // bool-return is properly observed by drain_write_buffer.
    Stats stats;
    SimConfig cfg = make_dramsim3_config(4, "wbsaturation");
    // Tighten the write region to one line so saturation is reachable from a
    // small number of stores even though DDR3 is fast.
    cfg.write_buffer_depth = 1;
    cfg.dramsim3_request_fifo_depth =
        (cfg.num_mshrs + cfg.write_buffer_depth) *
        (cfg.cache_line_size_bytes / cfg.dramsim3_bytes_per_burst);
    cfg.validate();

    constexpr uint32_t kNumWarps = 4;
    DRAMSim3Memory mem(cfg, stats);
    LoadGatherBufferFile gather_file(kNumWarps, stats);
    L1Cache cache(/*cache_size=*/4096, cfg.cache_line_size_bytes,
                  cfg.num_mshrs, cfg.write_buffer_depth, mem, gather_file,
                  stats);

    auto pump_one_cycle = [&]() {
        cache.evaluate();              // clears prior-cycle stall flag
        cache.handle_responses();      // drain memory responses
        mem.evaluate();                // tick DRAMSim3
        cache.drain_write_buffer();    // submits one entry if mem accepts
        cache.commit();
        gather_file.commit();
    };

    // Pre-install a line so subsequent stores hit the write buffer (the
    // path that exercises WRITE_BUFFER_FULL — store misses go through MSHRs
    // which would hit MSHR_FULL first under this config).
    constexpr uint32_t kHitLine = 0x100;
    constexpr uint32_t kHitAddr = kHitLine * 128;
    std::array<uint32_t, WARP_SIZE> dummy_results{};
    gather_file.claim(/*warp_id=*/0, /*dest_reg=*/1, 0, 0, 0);
    REQUIRE(cache.process_load(kHitAddr, 0, 0xFFFFFFFFu, dummy_results,
                               /*issue_cycle=*/1, 0, 0));
    cache.commit();
    gather_file.commit();
    // Pump until the line is filled. Bound the wait so an unrelated
    // regression doesn't deadlock this test.
    for (uint32_t i = 0; i < 1000 && !gather_file.has_result(); ++i) {
        pump_one_cycle();
    }
    REQUIRE(gather_file.has_result());
    (void)gather_file.consume_result();
    cache.commit();
    gather_file.commit();

    // Drive store hits at peak rate. With write_buffer_depth=1 and DRAMSim3
    // ingesting at one chunk per DRAM tick (~2.67 chunks/fabric cycle), each
    // submit_write needs chunks_per_line=4 free FIFO slots in the write
    // region — exactly what a wb_depth=1 region holds, so successive submits
    // are gated on completion. Saturation is reachable in a handful of cycles.
    constexpr uint32_t kStoresToIssue = 64;
    constexpr uint32_t kCycleCap = 100000;
    uint32_t stores_accepted = 0;
    uint32_t stores_rejected_by_cache = 0;
    uint64_t observed_wb_full_cycles = 0;

    for (uint32_t cycle = 0; cycle < kCycleCap && stores_accepted < kStoresToIssue; ++cycle) {
        cache.evaluate();
        cache.handle_responses();

        // Try one store hit per cycle (cache hits since kHitLine is resident).
        if (stores_accepted < kStoresToIssue) {
            const bool ok = cache.process_store(kHitLine, /*warp_id=*/0,
                                                /*issue_cycle=*/cycle, 0, 0);
            if (ok) {
                ++stores_accepted;
            } else {
                ++stores_rejected_by_cache;
                CHECK(cache.is_stalled());
                CHECK(cache.stall_reason() == CacheStallReason::WRITE_BUFFER_FULL);
            }
        }

        if (cache.stall_reason() == CacheStallReason::WRITE_BUFFER_FULL) {
            ++observed_wb_full_cycles;
        }

        mem.evaluate();
        cache.drain_write_buffer();
        cache.commit();
        gather_file.commit();
    }

    // Drain remaining writes to confirm nothing was lost.
    for (uint32_t i = 0; i < kCycleCap && (!mem.is_idle() ||
                                            cache.write_buffer_size() > 0); ++i) {
        pump_one_cycle();
    }

    // The whole point of the chain: the cache observed WRITE_BUFFER_FULL at
    // least once, the stall counter incremented, and process_store rejected
    // at least one store while the write buffer was full.
    CHECK(stores_accepted == kStoresToIssue);
    CHECK(stores_rejected_by_cache > 0u);
    CHECK(stats.write_buffer_stall_cycles > 0u);
    CHECK(stats.write_buffer_stall_cycles >= stores_rejected_by_cache);
    CHECK(observed_wb_full_cycles > 0u);

    // External writes match accepted stores — none were silently dropped
    // by the (now bool-respecting) drain_write_buffer.
    CHECK(stats.external_memory_writes == kStoresToIssue);

    // After full drain, the system is idle: no leaks anywhere in the chain.
    CHECK(mem.is_idle());
    CHECK(cache.write_buffer_size() == 0u);
}
