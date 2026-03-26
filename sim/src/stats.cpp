#include "gpu_sim/stats.h"
#include <iomanip>

namespace gpu_sim {

void Stats::report(std::ostream& out, uint32_t num_warps) const {
    out << "=== GPU Simulator Statistics ===\n\n";

    out << "--- Global ---\n";
    out << "Total cycles:              " << total_cycles << "\n";
    out << "Total instructions issued: " << total_instructions_issued << "\n";
    if (total_cycles > 0) {
        out << "IPC:                       " << std::fixed << std::setprecision(3)
            << static_cast<double>(total_instructions_issued) / static_cast<double>(total_cycles) << "\n";
    }
    out << "\n";

    out << "--- Per-Warp ---\n";
    for (uint32_t w = 0; w < num_warps; ++w) {
        out << "Warp " << w << ": issued=" << warp_instructions[w]
            << " active_cycles=" << warp_cycles_active[w]
            << " stall_scoreboard=" << warp_stall_scoreboard[w]
            << " stall_buffer_empty=" << warp_stall_buffer_empty[w]
            << " stall_unit_busy=" << warp_stall_unit_busy[w] << "\n";
    }
    out << "\n";

    out << "--- Pipeline ---\n";
    out << "Fetch skips:               " << fetch_skip_count << "\n";
    out << "Scheduler idle cycles:     " << scheduler_idle_cycles << "\n";
    out << "Operand collector busy:    " << operand_collector_busy_cycles << "\n";
    out << "Branch flushes:            " << branch_flushes << "\n";
    out << "\n";

    out << "--- Execution Units ---\n";
    auto print_unit = [&](const char* name, const UnitStats& s) {
        out << name << ": busy=" << s.busy_cycles << " instructions=" << s.instructions;
        if (total_cycles > 0) {
            out << " utilization=" << std::fixed << std::setprecision(1)
                << (100.0 * s.busy_cycles / total_cycles) << "%";
        }
        out << "\n";
    };
    print_unit("ALU    ", alu_stats);
    print_unit("MUL    ", mul_stats);
    print_unit("DIV    ", div_stats);
    print_unit("LD/ST  ", ldst_stats);
    print_unit("TLOOKUP", tlookup_stats);
    out << "\n";

    out << "--- Memory System ---\n";
    uint64_t total_accesses = cache_hits + cache_misses;
    out << "Cache hits:                " << cache_hits << "\n";
    out << "Cache misses:              " << cache_misses << "\n";
    if (total_accesses > 0) {
        out << "Hit rate:                  " << std::fixed << std::setprecision(1)
            << (100.0 * cache_hits / total_accesses) << "%\n";
    }
    out << "Load hits/misses:          " << load_hits << "/" << load_misses << "\n";
    out << "Store hits/misses:         " << store_hits << "/" << store_misses << "\n";
    out << "MSHR stall cycles:         " << mshr_stall_cycles << "\n";
    out << "Write buffer stall cycles: " << write_buffer_stall_cycles << "\n";
    out << "Coalesced requests:        " << coalesced_requests << "\n";
    out << "Serialized requests:       " << serialized_requests << "\n";
    out << "External memory reads:     " << external_memory_reads << "\n";
    out << "External memory writes:    " << external_memory_writes << "\n";
    if (total_loads_completed > 0) {
        out << "Avg load latency:          " << std::fixed << std::setprecision(1)
            << static_cast<double>(total_load_latency) / static_cast<double>(total_loads_completed)
            << " cycles\n";
    }
    out << "\n";

    out << "--- Writeback ---\n";
    out << "Writeback conflicts:       " << writeback_conflicts << "\n";
}

void Stats::report_json(std::ostream& out, uint32_t num_warps) const {
    out << "{\n";
    out << "  \"total_cycles\": " << total_cycles << ",\n";
    out << "  \"total_instructions_issued\": " << total_instructions_issued << ",\n";
    if (total_cycles > 0) {
        out << "  \"ipc\": " << std::fixed << std::setprecision(6)
            << static_cast<double>(total_instructions_issued) / static_cast<double>(total_cycles) << ",\n";
    }
    out << "  \"scheduler_idle_cycles\": " << scheduler_idle_cycles << ",\n";
    out << "  \"branch_flushes\": " << branch_flushes << ",\n";
    out << "  \"cache_hits\": " << cache_hits << ",\n";
    out << "  \"cache_misses\": " << cache_misses << ",\n";
    out << "  \"coalesced_requests\": " << coalesced_requests << ",\n";
    out << "  \"serialized_requests\": " << serialized_requests << ",\n";
    out << "  \"external_memory_reads\": " << external_memory_reads << ",\n";
    out << "  \"external_memory_writes\": " << external_memory_writes << ",\n";

    out << "  \"warp_instructions\": [";
    for (uint32_t w = 0; w < num_warps; ++w) {
        if (w > 0) out << ", ";
        out << warp_instructions[w];
    }
    out << "]\n";
    out << "}\n";
}

} // namespace gpu_sim
