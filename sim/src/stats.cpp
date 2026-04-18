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
            << " stall_branch_shadow=" << warp_stall_branch_shadow[w]
            << " stall_unit_busy=" << warp_stall_unit_busy[w] << "\n";
    }
    out << "\n";

    out << "--- Pipeline ---\n";
    out << "Fetch skips:               " << fetch_skip_count << "\n";
    out << "  Backpressure:            " << fetch_skip_backpressure << "\n";
    out << "  All buffers full:        " << fetch_skip_all_full << "\n";
    out << "Scheduler idle cycles:     " << scheduler_idle_cycles << "\n";
    out << "  Frontend stall:          " << scheduler_frontend_stall_cycles << "\n";
    out << "  Backend stall:           " << scheduler_stall_backend_cycles << "\n";
    out << "Operand collector busy:    " << operand_collector_busy_cycles << "\n";
    out << "Branch predictions:        " << branch_predictions << "\n";
    out << "Branch mispredictions:     " << branch_mispredictions << "\n";
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
    out << "Gather buffer stall:       " << gather_buffer_stall_cycles << "\n";
    out << "Gather buffer port conf.:  " << gather_buffer_port_conflict_cycles << "\n";
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

    // Global
    out << "  \"total_cycles\": " << total_cycles << ",\n";
    out << "  \"total_instructions_issued\": " << total_instructions_issued << ",\n";
    if (total_cycles > 0) {
        out << "  \"ipc\": " << std::fixed << std::setprecision(6)
            << static_cast<double>(total_instructions_issued) / static_cast<double>(total_cycles) << ",\n";
    }

    // Pipeline
    out << "  \"fetch_skip_count\": " << fetch_skip_count << ",\n";
    out << "  \"fetch_skip_backpressure\": " << fetch_skip_backpressure << ",\n";
    out << "  \"fetch_skip_all_full\": " << fetch_skip_all_full << ",\n";
    out << "  \"scheduler_idle_cycles\": " << scheduler_idle_cycles << ",\n";
    out << "  \"scheduler_frontend_stall_cycles\": " << scheduler_frontend_stall_cycles << ",\n";
    out << "  \"scheduler_stall_backend_cycles\": " << scheduler_stall_backend_cycles << ",\n";
    out << "  \"operand_collector_busy_cycles\": " << operand_collector_busy_cycles << ",\n";
    out << "  \"branch_predictions\": " << branch_predictions << ",\n";
    out << "  \"branch_mispredictions\": " << branch_mispredictions << ",\n";
    out << "  \"branch_flushes\": " << branch_flushes << ",\n";

    // Execution units
    auto emit_unit = [&](const char* prefix, const UnitStats& s) {
        out << "  \"" << prefix << "_busy_cycles\": " << s.busy_cycles << ",\n";
        out << "  \"" << prefix << "_instructions\": " << s.instructions << ",\n";
    };
    emit_unit("alu", alu_stats);
    emit_unit("mul", mul_stats);
    emit_unit("div", div_stats);
    emit_unit("ldst", ldst_stats);
    emit_unit("tlookup", tlookup_stats);

    // Memory system
    out << "  \"cache_hits\": " << cache_hits << ",\n";
    out << "  \"cache_misses\": " << cache_misses << ",\n";
    out << "  \"load_hits\": " << load_hits << ",\n";
    out << "  \"load_misses\": " << load_misses << ",\n";
    out << "  \"store_hits\": " << store_hits << ",\n";
    out << "  \"store_misses\": " << store_misses << ",\n";
    out << "  \"mshr_stall_cycles\": " << mshr_stall_cycles << ",\n";
    out << "  \"write_buffer_stall_cycles\": " << write_buffer_stall_cycles << ",\n";
    out << "  \"coalesced_requests\": " << coalesced_requests << ",\n";
    out << "  \"serialized_requests\": " << serialized_requests << ",\n";
    out << "  \"external_memory_reads\": " << external_memory_reads << ",\n";
    out << "  \"external_memory_writes\": " << external_memory_writes << ",\n";
    out << "  \"total_load_latency\": " << total_load_latency << ",\n";
    out << "  \"total_loads_completed\": " << total_loads_completed << ",\n";
    out << "  \"gather_buffer_stall_cycles\": " << gather_buffer_stall_cycles << ",\n";
    out << "  \"gather_buffer_port_conflict_cycles\": " << gather_buffer_port_conflict_cycles << ",\n";

    // Writeback
    out << "  \"writeback_conflicts\": " << writeback_conflicts << ",\n";

    // Per-warp arrays
    auto emit_warp_array = [&](const char* name,
                               const std::array<uint64_t, MAX_WARPS>& arr) {
        out << "  \"" << name << "\": [";
        for (uint32_t w = 0; w < num_warps; ++w) {
            if (w > 0) out << ", ";
            out << arr[w];
        }
        out << "],\n";
    };
    emit_warp_array("warp_instructions", warp_instructions);
    emit_warp_array("warp_cycles_active", warp_cycles_active);
    emit_warp_array("warp_stall_scoreboard", warp_stall_scoreboard);
    emit_warp_array("warp_stall_buffer_empty", warp_stall_buffer_empty);
    emit_warp_array("warp_stall_branch_shadow", warp_stall_branch_shadow);

    // Last field — no trailing comma
    out << "  \"warp_stall_unit_busy\": [";
    for (uint32_t w = 0; w < num_warps; ++w) {
        if (w > 0) out << ", ";
        out << warp_stall_unit_busy[w];
    }
    out << "]\n";

    out << "}\n";
}

} // namespace gpu_sim
