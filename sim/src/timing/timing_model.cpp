#include "gpu_sim/timing/timing_model.h"
#include <iostream>
#include <iomanip>

namespace gpu_sim {

TimingModel::TimingModel(const SimConfig& config, FunctionalModel& func_model, Stats& stats)
    : config_(config), func_model_(func_model), stats_(stats),
      trace_enabled_(config.trace_enabled) {

    // Initialize per-warp state
    for (uint32_t w = 0; w < config.num_warps; ++w) {
        warps_.emplace_back(config.instruction_buffer_depth);
        warps_.back().reset(config.start_pc);
    }

    // Pipeline stages
    fetch_ = std::make_unique<FetchStage>(config.num_warps, warps_.data(),
                                          func_model.instruction_memory(), stats);
    decode_ = std::make_unique<DecodeStage>(warps_.data(), *fetch_);
    scheduler_ = std::make_unique<WarpScheduler>(config.num_warps, warps_.data(),
                                                  scoreboard_, func_model, stats);
    opcoll_ = std::make_unique<OperandCollector>(stats);

    // Execution units
    alu_ = std::make_unique<ALUUnit>(stats);
    mul_ = std::make_unique<MultiplyUnit>(config.multiply_pipeline_stages, stats);
    div_ = std::make_unique<DivideUnit>(stats);
    tlookup_ = std::make_unique<TLookupUnit>(stats);
    ldst_ = std::make_unique<LdStUnit>(config.num_ldst_units, config.addr_gen_fifo_depth, stats);

    // Memory system
    mem_if_ = std::make_unique<ExternalMemoryInterface>(config.external_memory_latency_cycles, stats);
    cache_ = std::make_unique<L1Cache>(config.l1_cache_size_bytes, config.cache_line_size_bytes,
                                        config.num_mshrs, config.write_buffer_depth,
                                        *mem_if_, stats);
    coalescing_ = std::make_unique<CoalescingUnit>(*ldst_, *cache_, config.cache_line_size_bytes, stats);

    // Writeback arbiter
    wb_arbiter_ = std::make_unique<WritebackArbiter>(scoreboard_, stats);
    wb_arbiter_->add_source(alu_.get());
    wb_arbiter_->add_source(mul_.get());
    wb_arbiter_->add_source(div_.get());
    wb_arbiter_->add_source(tlookup_.get());
    // LD/ST results come through fill buffer, not as a direct source
    // (ldst->has_result() returns false)

    // Panic controller
    panic_ = std::make_unique<PanicController>(config.num_warps, warps_.data(), func_model);

    // Wire scheduler's unit readiness check
    scheduler_->set_unit_ready_fn([this](ExecUnit unit) -> bool {
        switch (unit) {
            case ExecUnit::ALU:      return alu_->is_ready();
            case ExecUnit::MULTIPLY: return mul_->is_ready();
            case ExecUnit::DIVIDE:   return div_->is_ready();
            case ExecUnit::TLOOKUP:  return tlookup_->is_ready();
            case ExecUnit::LDST:     return ldst_->is_ready();
            case ExecUnit::SYSTEM:   return true;  // ECALL handled inline
            default:                 return false;
        }
    });
}

void TimingModel::dispatch_to_unit(const DispatchInput& input) {
    switch (input.decoded.target_unit) {
        case ExecUnit::ALU:
            alu_->accept(input, cycle_);
            break;
        case ExecUnit::MULTIPLY:
            mul_->accept(input, cycle_);
            break;
        case ExecUnit::DIVIDE:
            div_->accept(input, cycle_);
            break;
        case ExecUnit::TLOOKUP:
            tlookup_->accept(input, cycle_);
            break;
        case ExecUnit::LDST:
            ldst_->accept(input, cycle_);
            break;
        case ExecUnit::SYSTEM:
            // ECALL/CSR: handled at issue time by the functional model
            // For ECALL: warp is already marked inactive in functional model
            // We need to sync the timing model's warp state
            if (input.trace.is_ecall) {
                warps_[input.warp_id].active = false;
            }
            // CSR reads produce a result that needs writeback
            if (input.decoded.type == InstructionType::CSR) {
                // Route CSR results through ALU (1-cycle result)
                alu_->accept(input, cycle_);
            }
            break;
        default:
            break;
    }
}

bool TimingModel::all_units_idle() const {
    return alu_->is_ready() && !alu_->has_result() &&
           mul_->is_ready() && !mul_->has_result() &&
           div_->is_ready() && !div_->has_result() &&
           tlookup_->is_ready() && !tlookup_->has_result() &&
           ldst_->is_ready() && ldst_->fifo_empty();
}

bool TimingModel::all_warps_done() const {
    for (uint32_t w = 0; w < config_.num_warps; ++w) {
        if (warps_[w].active) return false;
    }
    return true;
}

bool TimingModel::tick() {
    cycle_++;
    stats_.total_cycles = cycle_;

    // Count active warp cycles
    for (uint32_t w = 0; w < config_.num_warps; ++w) {
        if (warps_[w].active) stats_.warp_cycles_active[w]++;
    }

    // If panic controller is active, run it instead of normal pipeline
    if (panic_->is_active()) {
        panic_->set_units_drained(all_units_idle());
        panic_->evaluate();

        // Still need to drain execution units
        alu_->evaluate();
        mul_->evaluate();
        div_->evaluate();
        tlookup_->evaluate();
        ldst_->evaluate();
        mem_if_->evaluate();

        // Handle memory responses and writeback during drain
        WritebackEntry fill_wb;
        bool fill_valid = false;
        cache_->handle_responses(fill_wb, fill_valid);
        if (fill_valid) {
            wb_arbiter_->submit_fill(fill_wb);
        }
        wb_arbiter_->evaluate();
        wb_arbiter_->commit();

        // Commit execution units
        alu_->commit();
        mul_->commit();
        div_->commit();
        tlookup_->commit();
        ldst_->commit();
        mem_if_->commit();
        scoreboard_.commit();

        if (panic_->is_done()) {
            return false;
        }
        return true;
    }

    // Seed scoreboard next from current
    scoreboard_.seed_next();

    // === Forward-order pipeline evaluation ===

    // 1. Fetch
    fetch_->evaluate();

    // 2. Decode
    decode_->evaluate();

    // Check for EBREAK
    if (decode_->ebreak_detected()) {
        panic_->trigger(decode_->ebreak_warp(), decode_->ebreak_pc());
        // Don't proceed with normal pipeline this cycle
        fetch_->commit();
        decode_->commit();
        scoreboard_.commit();
        return true;
    }

    // 3. Warp Scheduler
    scheduler_->set_opcoll_free(opcoll_->is_free());
    scheduler_->evaluate();

    // 4. Operand Collector
    // If scheduler issued, feed to operand collector
    if (scheduler_->output()) {
        opcoll_->accept(*scheduler_->output());
    }
    opcoll_->evaluate();

    // 5. Dispatch to execution units
    if (opcoll_->output()) {
        dispatch_to_unit(*opcoll_->output());

        // Handle branch resolution
        const auto& out = *opcoll_->output();
        if (out.trace.is_branch && out.trace.branch_taken) {
            fetch_->redirect_warp(out.warp_id, out.trace.branch_target);
            decode_->invalidate_warp(out.warp_id);
            stats_.branch_flushes++;
        }
    }

    // 6. Execute
    alu_->evaluate();
    mul_->evaluate();
    div_->evaluate();
    tlookup_->evaluate();
    ldst_->evaluate();

    // 7. Coalescing unit
    WritebackEntry coal_wb;
    bool coal_wb_valid = false;
    coalescing_->evaluate(coal_wb, coal_wb_valid);

    // 8. Memory interface
    mem_if_->evaluate();

    // 9. Handle MSHR fills
    WritebackEntry fill_wb;
    bool fill_valid = false;
    cache_->handle_responses(fill_wb, fill_valid);

    // 10. Write buffer drain (reads prioritized, so drain after reads)
    cache_->drain_write_buffer();

    // Submit fill results to writeback arbiter
    // Priority: MSHR fill > cache hit from coalescing
    if (fill_valid) {
        wb_arbiter_->submit_fill(fill_wb);
    } else if (coal_wb_valid) {
        wb_arbiter_->submit_fill(coal_wb);
    }

    // 11. Writeback arbiter
    wb_arbiter_->evaluate();

    // === Commit all stages ===
    fetch_->commit();
    decode_->commit();
    scheduler_->commit();
    opcoll_->commit();
    alu_->commit();
    mul_->commit();
    div_->commit();
    tlookup_->commit();
    ldst_->commit();
    coalescing_->commit();
    cache_->commit();
    mem_if_->commit();
    wb_arbiter_->commit();
    scoreboard_.commit();

    if (trace_enabled_) {
        trace_cycle();
    }

    // Check termination
    if (all_warps_done() && all_units_idle()) {
        return false;
    }

    return true;
}

void TimingModel::run(uint64_t max_cycles) {
    while (tick()) {
        if (max_cycles > 0 && cycle_ >= max_cycles) {
            std::cerr << "Timing model: max cycles reached (" << max_cycles << ")\n";
            break;
        }
    }
}

void TimingModel::trace_cycle() const {
    std::cerr << "cycle=" << std::setw(6) << cycle_;

    // Fetch info
    if (fetch_->current_output()) {
        std::cerr << " fetch=W" << fetch_->current_output()->warp_id;
    } else {
        std::cerr << " fetch=--";
    }

    // Scheduler info
    if (scheduler_->current_output()) {
        std::cerr << " issue=W" << scheduler_->current_output()->warp_id;
    } else {
        std::cerr << " issue=--";
    }

    // OpColl
    std::cerr << " opcoll=" << (opcoll_->is_free() ? "free" : "busy");

    // Execution units
    std::cerr << " alu=" << (alu_->is_ready() ? "rdy" : "bsy");
    std::cerr << " mul=" << (mul_->is_ready() ? "rdy" : "bsy");
    std::cerr << " div=" << (div_->is_ready() ? "rdy" : "bsy");
    std::cerr << " tlk=" << (tlookup_->is_ready() ? "rdy" : "bsy");
    std::cerr << " ldst=" << (ldst_->is_ready() ? "rdy" : "bsy");

    // Writeback
    if (wb_arbiter_->committed_entry()) {
        const auto& wb = *wb_arbiter_->committed_entry();
        std::cerr << " wb=W" << wb.warp_id << ":x" << static_cast<int>(wb.dest_reg);
    } else {
        std::cerr << " wb=--";
    }

    std::cerr << "\n";
}

} // namespace gpu_sim
