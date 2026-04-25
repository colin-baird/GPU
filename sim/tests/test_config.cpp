#include "catch.hpp"
#include "gpu_sim/config.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace gpu_sim;

namespace {

// Write `body` into a uniquely-named temporary file and return the path.
// Caller is responsible for the file's lifetime; we just create + return.
std::string write_temp_json(const std::string& body) {
    char tmpl[] = "/tmp/gpu_sim_test_config_XXXXXX";
    int fd = mkstemp(tmpl);
    REQUIRE(fd >= 0);
    std::string path(tmpl);
    // Append a .json extension on disk via rename so future readers see it.
    std::ofstream out(path);
    out << body;
    out.close();
    ::close(fd);
    return path;
}

// argv-style helper for CLI tests.
struct Argv {
    std::vector<std::string> storage;
    std::vector<char*>       ptrs;

    explicit Argv(std::initializer_list<std::string> args) : storage(args) {
        ptrs.reserve(storage.size());
        for (auto& s : storage) ptrs.push_back(s.data());
    }

    int    argc() const { return static_cast<int>(ptrs.size()); }
    char** argv()       { return ptrs.data(); }
};

} // namespace

TEST_CASE("SimConfig: default values for memory-backend knobs", "[config]") {
    SimConfig c;
    CHECK(c.memory_backend == "fixed");
    CHECK(c.dramsim3_config_path.empty());
    CHECK(c.dramsim3_output_dir == "/tmp/dramsim3");
    CHECK(c.fpga_clock_mhz == Approx(150.0));
    CHECK(c.dram_clock_mhz == Approx(400.0));
    CHECK(c.dramsim3_request_fifo_depth == 32u);
    CHECK(c.dramsim3_bytes_per_burst == 32u);
    CHECK_NOTHROW(c.validate());
}

TEST_CASE("SimConfig: default DRAMSim3 request FIFO depth equals worst case",
          "[config]") {
    // The default sizing is exactly (num_mshrs + write_buffer_depth) *
    // (cache_line_size_bytes / dramsim3_bytes_per_burst). Anything larger is
    // wasteful; anything smaller can drop requests under worst-case cache
    // traffic. Reads off SimConfig defaults to make the relationship explicit.
    SimConfig c;
    const uint32_t chunks_per_line =
        c.cache_line_size_bytes / c.dramsim3_bytes_per_burst;
    const uint32_t worst_case =
        (c.num_mshrs + c.write_buffer_depth) * chunks_per_line;
    CHECK(c.dramsim3_request_fifo_depth == worst_case);
}

TEST_CASE("SimConfig::from_json parses all DRAMSim3 knobs", "[config]") {
    const std::string json = R"({
        "num_warps": 4,
        "memory_backend": "dramsim3",
        "dramsim3_config_path": "sim/configs/dram/DDR3_4Gb_x16_800.ini",
        "dramsim3_output_dir": "/tmp/dramsim3-test",
        "fpga_clock_mhz": 200.0,
        "dram_clock_mhz": 533.3,
        "dramsim3_request_fifo_depth": 32,
        "dramsim3_bytes_per_burst": 32
    })";
    const std::string path = write_temp_json(json);

    SimConfig c = SimConfig::from_json(path);
    std::remove(path.c_str());

    CHECK(c.memory_backend == "dramsim3");
    CHECK(c.dramsim3_config_path == "sim/configs/dram/DDR3_4Gb_x16_800.ini");
    CHECK(c.dramsim3_output_dir == "/tmp/dramsim3-test");
    CHECK(c.fpga_clock_mhz == Approx(200.0));
    CHECK(c.dram_clock_mhz == Approx(533.3));
    CHECK(c.dramsim3_request_fifo_depth == 32u);
    CHECK(c.dramsim3_bytes_per_burst == 32u);
    CHECK_NOTHROW(c.validate());
}

TEST_CASE("SimConfig::validate rejects unknown memory_backend", "[config]") {
    SimConfig c;
    c.memory_backend = "rambus";
    CHECK_THROWS_AS(c.validate(), std::invalid_argument);
}

TEST_CASE("SimConfig::validate rejects non-positive clocks", "[config]") {
    SimConfig c;
    c.fpga_clock_mhz = 0.0;
    CHECK_THROWS_AS(c.validate(), std::invalid_argument);

    SimConfig d;
    d.dram_clock_mhz = -1.0;
    CHECK_THROWS_AS(d.validate(), std::invalid_argument);
}

TEST_CASE("SimConfig::validate requires line size to be multiple of burst",
          "[config]") {
    SimConfig c;
    // Default line=128 with burst=48 isn't a multiple — invalid.
    c.dramsim3_bytes_per_burst = 48;
    CHECK_THROWS_AS(c.validate(), std::invalid_argument);

    // 128 % 32 == 0 is fine.
    c.dramsim3_bytes_per_burst = 32;
    CHECK_NOTHROW(c.validate());
}

TEST_CASE("SimConfig::validate rejects undersized DRAMSim3 request FIFO",
          "[config]") {
    SimConfig c;
    c.memory_backend = "dramsim3";
    // (num_mshrs + write_buffer_depth) * chunks_per_line = (4+4)*4 = 32.
    c.dramsim3_request_fifo_depth = 16;
    CHECK_THROWS_AS(c.validate(), std::invalid_argument);

    // Exactly the minimum is acceptable.
    c.dramsim3_request_fifo_depth = 32;
    CHECK_NOTHROW(c.validate());

    // The check is gated on memory_backend; under "fixed" it doesn't fire.
    SimConfig f;
    f.memory_backend = "fixed";
    f.dramsim3_request_fifo_depth = 1;
    CHECK_NOTHROW(f.validate());
}

TEST_CASE("SimConfig::apply_cli_overrides sets DRAMSim3 knobs", "[config]") {
    Argv av({
        "prog",
        "--memory-backend=dramsim3",
        "--dramsim3-config-path=/etc/dram.ini",
        "--dramsim3-output-dir=/tmp/out",
        "--fpga-clock-mhz=120.5",
        "--dram-clock-mhz=800.0",
        "--dramsim3-request-fifo-depth=64",
        "--dramsim3-bytes-per-burst=32",
    });

    SimConfig c;
    c.apply_cli_overrides(av.argc(), av.argv());

    CHECK(c.memory_backend == "dramsim3");
    CHECK(c.dramsim3_config_path == "/etc/dram.ini");
    CHECK(c.dramsim3_output_dir == "/tmp/out");
    CHECK(c.fpga_clock_mhz == Approx(120.5));
    CHECK(c.dram_clock_mhz == Approx(800.0));
    CHECK(c.dramsim3_request_fifo_depth == 64u);
    CHECK(c.dramsim3_bytes_per_burst == 32u);
    CHECK_NOTHROW(c.validate());
}
