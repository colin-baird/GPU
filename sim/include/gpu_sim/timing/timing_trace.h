#pragma once

#include "gpu_sim/trace_event.h"
#include "gpu_sim/types.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace gpu_sim {

struct TimingTraceOptions {
    std::string output_path;

    bool enabled() const { return !output_path.empty(); }
};

enum class WarpTraceState {
    FETCH,
    DECODE_PENDING,
    OPERAND_COLLECT,
    EXECUTE_ALU,
    EXECUTE_MUL,
    EXECUTE_DIV,
    EXECUTE_TLOOKUP,
    ADDR_GEN,
    LDST_FIFO,
    COALESCING,
    MEMORY_WAIT,
    WRITEBACK_WAIT,
    AT_REST,
    PANIC_DRAIN,
    RETIRED
};

enum class WarpRestReason {
    NONE,
    WAIT_FRONTEND,
    WAIT_BRANCH_SHADOW,
    WAIT_SCOREBOARD,
    WAIT_OPCOLL,
    WAIT_UNIT_ALU,
    WAIT_UNIT_MULTIPLY,
    WAIT_UNIT_DIVIDE,
    WAIT_UNIT_TLOOKUP,
    WAIT_UNIT_LDST,
    WAIT_ROUND_ROBIN,
    WAIT_L1_MSHR,
    WAIT_L1_WRITE_BUFFER,
    WAIT_MEMORY_RESPONSE,
    WAIT_WRITEBACK
};

struct WarpTraceSnapshot {
    uint32_t warp_id = 0;
    bool active = false;
    WarpTraceState state = WarpTraceState::RETIRED;
    WarpRestReason rest_reason = WarpRestReason::NONE;
    uint32_t pc = 0;
    uint32_t raw_instruction = 0;
    ExecUnit target_unit = ExecUnit::NONE;
    uint8_t dest_reg = 0;
    bool branch_taken = false;
    uint32_t branch_target = 0;
    bool has_memory_address = false;
    uint32_t first_memory_address = 0;
    bool coalesced_memory = false;
};

struct CycleTraceSnapshot {
    uint64_t cycle = 0;
    uint32_t num_warps = 0;
    std::array<WarpTraceSnapshot, MAX_WARPS> warps{};

    uint32_t active_warps = 0;
    bool opcoll_busy = false;
    uint32_t opcoll_warp = 0;
    uint32_t opcoll_cycles_remaining = 0;
    bool alu_busy = false;
    bool mul_busy = false;
    uint32_t mul_pipeline_occupancy = 0;
    bool div_busy = false;
    bool tlookup_busy = false;
    bool ldst_busy = false;
    uint32_t ldst_fifo_depth = 0;
    uint32_t active_mshrs = 0;
    uint32_t write_buffer_depth = 0;
    bool panic_active = false;
};

const char* to_string(WarpTraceState state);
const char* to_string(WarpRestReason reason);

using TraceArgValue = std::variant<int64_t, uint64_t, double, std::string, bool>;
using TraceArgs = std::vector<std::pair<std::string, TraceArgValue>>;

class ChromeTraceWriter {
public:
    explicit ChromeTraceWriter(const std::string& path);
    ~ChromeTraceWriter();

    ChromeTraceWriter(const ChromeTraceWriter&) = delete;
    ChromeTraceWriter& operator=(const ChromeTraceWriter&) = delete;

    bool is_open() const { return out_.is_open(); }
    const std::string& path() const { return path_; }

    void write_process_metadata(int pid, const std::string& name);
    void write_thread_metadata(int pid, int tid, const std::string& name, int sort_index = 0);
    void write_complete(const std::string& name, uint64_t ts, uint64_t dur,
                        int pid, int tid, const TraceArgs& args = {});
    void write_instant(const std::string& name, uint64_t ts, int pid, int tid,
                       const TraceArgs& args = {});
    void write_counter(const std::string& name, uint64_t ts, int pid, int tid,
                       const TraceArgs& args);
    void finalize();

private:
    std::ofstream out_;
    std::string path_;
    bool first_event_ = true;
    bool finalized_ = false;

    void begin_event();
    static void write_escaped(std::ostream& out, const std::string& value);
    static void write_args(std::ostream& out, const TraceArgs& args);
};

} // namespace gpu_sim
