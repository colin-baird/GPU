#include "gpu_sim/timing/timing_model.h"
#ifdef GPU_SIM_USE_DRAMSIM3
#include "gpu_sim/timing/dramsim3_memory.h"
#endif
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace gpu_sim {

namespace {

constexpr int kWarpPid = 100;
constexpr int kHardwarePid = 200;
constexpr int kCounterPid = 300;

constexpr int kOpcollTid = 1;
constexpr int kAluTid = 2;
constexpr int kMulTid = 3;
constexpr int kDivTid = 4;
constexpr int kTlookupTid = 5;
constexpr int kLdstTid = 6;
constexpr int kCoalescerTid = 7;
constexpr int kCacheTid = 8;
constexpr int kWritebackTid = 9;
constexpr int kPanicTid = 10;
constexpr size_t kHardwareTrackCount = 10;

const char* exec_unit_name(ExecUnit unit) {
    switch (unit) {
        case ExecUnit::ALU:      return "alu";
        case ExecUnit::MULTIPLY: return "multiply";
        case ExecUnit::DIVIDE:   return "divide";
        case ExecUnit::TLOOKUP:  return "tlookup";
        case ExecUnit::LDST:     return "ldst";
        case ExecUnit::SYSTEM:   return "system";
        default:                 return "none";
    }
}

WarpRestReason scheduler_rest_reason(SchedulerIssueOutcome outcome) {
    switch (outcome) {
        case SchedulerIssueOutcome::BUFFER_EMPTY:      return WarpRestReason::WAIT_FRONTEND;
        case SchedulerIssueOutcome::SCOREBOARD:        return WarpRestReason::WAIT_SCOREBOARD;
        case SchedulerIssueOutcome::OPCOLL_BUSY:       return WarpRestReason::WAIT_OPCOLL;
        case SchedulerIssueOutcome::UNIT_BUSY_ALU:     return WarpRestReason::WAIT_UNIT_ALU;
        case SchedulerIssueOutcome::UNIT_BUSY_MULTIPLY:return WarpRestReason::WAIT_UNIT_MULTIPLY;
        case SchedulerIssueOutcome::UNIT_BUSY_DIVIDE:  return WarpRestReason::WAIT_UNIT_DIVIDE;
        case SchedulerIssueOutcome::UNIT_BUSY_TLOOKUP: return WarpRestReason::WAIT_UNIT_TLOOKUP;
        case SchedulerIssueOutcome::UNIT_BUSY_LDST:    return WarpRestReason::WAIT_UNIT_LDST;
        case SchedulerIssueOutcome::BRANCH_SHADOW:     return WarpRestReason::WAIT_BRANCH_SHADOW;
        case SchedulerIssueOutcome::READY_NOT_SELECTED:return WarpRestReason::WAIT_ROUND_ROBIN;
        default:                                       return WarpRestReason::NONE;
    }
}

void fill_from_decoded(WarpTraceSnapshot& warp, uint32_t pc,
                       const DecodedInstruction& decoded) {
    warp.pc = pc;
    warp.raw_instruction = decoded.raw;
    warp.target_unit = decoded.target_unit;
    warp.dest_reg = decoded.rd;
}

void fill_from_writeback(WarpTraceSnapshot& warp, const WritebackEntry& wb) {
    warp.pc = wb.pc;
    warp.raw_instruction = wb.raw_instruction;
    warp.target_unit = wb.source_unit;
    warp.dest_reg = wb.dest_reg;
}

std::string make_warp_key(const WarpTraceSnapshot& warp) {
    std::ostringstream out;
    out << static_cast<int>(warp.state) << ':' << static_cast<int>(warp.rest_reason)
        << ':' << warp.pc << ':' << warp.raw_instruction << ':'
        << static_cast<int>(warp.target_unit) << ':' << static_cast<int>(warp.dest_reg)
        << ':' << warp.branch_taken << ':' << warp.branch_target << ':'
        << warp.has_memory_address << ':' << warp.first_memory_address << ':'
        << warp.coalesced_memory;
    return out.str();
}

TraceArgs make_warp_args(const WarpTraceSnapshot& warp, uint64_t cycle) {
    TraceArgs args;
    args.emplace_back("cycle", static_cast<uint64_t>(cycle));
    args.emplace_back("state", std::string(to_string(warp.state)));
    if (warp.rest_reason != WarpRestReason::NONE) {
        args.emplace_back("rest_reason", std::string(to_string(warp.rest_reason)));
    }
    args.emplace_back("pc", static_cast<uint64_t>(warp.pc));
    args.emplace_back("raw_instruction", static_cast<uint64_t>(warp.raw_instruction));
    args.emplace_back("target_unit", std::string(exec_unit_name(warp.target_unit)));
    args.emplace_back("dest_reg", static_cast<uint64_t>(warp.dest_reg));
    if (warp.branch_taken) {
        args.emplace_back("branch_target", static_cast<uint64_t>(warp.branch_target));
    }
    if (warp.has_memory_address) {
        args.emplace_back("first_memory_address",
                          static_cast<uint64_t>(warp.first_memory_address));
        args.emplace_back("coalesced_memory", warp.coalesced_memory);
    }
    return args;
}

std::string warp_slice_name(const WarpTraceSnapshot& warp) {
    if (warp.state == WarpTraceState::AT_REST && warp.rest_reason != WarpRestReason::NONE) {
        return to_string(warp.rest_reason);
    }
    return to_string(warp.state);
}

TraceArgs instruction_args(uint64_t cycle, uint32_t warp_id, uint32_t pc,
                           uint32_t raw_instruction, ExecUnit unit, uint8_t dest_reg) {
    TraceArgs args;
    args.emplace_back("cycle", static_cast<uint64_t>(cycle));
    args.emplace_back("warp", static_cast<uint64_t>(warp_id));
    args.emplace_back("pc", static_cast<uint64_t>(pc));
    args.emplace_back("raw_instruction", static_cast<uint64_t>(raw_instruction));
    args.emplace_back("target_unit", std::string(exec_unit_name(unit)));
    args.emplace_back("dest_reg", static_cast<uint64_t>(dest_reg));
    return args;
}

} // namespace

TimingModel::TimingModel(const SimConfig& config, FunctionalModel& func_model, Stats& stats,
                         TimingTraceOptions trace_options)
    : config_(config), func_model_(func_model), stats_(stats),
      trace_options_(std::move(trace_options)),
      trace_enabled_(config.trace_enabled) {

    for (uint32_t w = 0; w < config.num_warps; ++w) {
        warps_.emplace_back(config.instruction_buffer_depth);
        warps_.back().reset(config.start_pc);
    }

    branch_predictor_ = std::make_unique<StaticDirectionalBranchPredictor>();
    fetch_ = std::make_unique<FetchStage>(config.num_warps, warps_.data(),
                                          func_model.instruction_memory(),
                                          *branch_predictor_, stats);
    decode_ = std::make_unique<DecodeStage>(warps_.data(), *fetch_);
    // Phase 3: wire decode into fetch so fetch.evaluate() can query
    // decode->ready_to_consume_fetch() and decode->pending_warp() directly,
    // replacing the pre-evaluate set_decode_pending_warp setter and the
    // output_consumed_ round-trip.
    fetch_->set_decode(decode_.get());
    scheduler_ = std::make_unique<WarpScheduler>(config.num_warps, warps_.data(),
                                                 scoreboard_, func_model, stats);
    opcoll_ = std::make_unique<OperandCollector>(stats);

    alu_ = std::make_unique<ALUUnit>(stats);
    mul_ = std::make_unique<MultiplyUnit>(config.multiply_pipeline_stages, stats);
    div_ = std::make_unique<DivideUnit>(stats);
    tlookup_ = std::make_unique<TLookupUnit>(stats);
    ldst_ = std::make_unique<LdStUnit>(config.num_ldst_units, config.addr_gen_fifo_depth, stats);

    if (config.memory_backend == "dramsim3") {
#ifdef GPU_SIM_USE_DRAMSIM3
        mem_if_ = std::make_unique<DRAMSim3Memory>(config, stats);
#else
        throw std::invalid_argument(
            "memory_backend=\"dramsim3\" requires the simulator to be built "
            "with -DGPU_SIM_USE_DRAMSIM3=ON (currently OFF)");
#endif
    } else {
        mem_if_ = std::make_unique<FixedLatencyMemory>(
            config.external_memory_latency_cycles, stats);
    }
    gather_file_ = std::make_unique<LoadGatherBufferFile>(config.num_warps, stats);
    cache_ = std::make_unique<L1Cache>(config.l1_cache_size_bytes, config.cache_line_size_bytes,
                                       config.num_mshrs, config.write_buffer_depth,
                                       *mem_if_, *gather_file_, stats);
    coalescing_ = std::make_unique<CoalescingUnit>(*ldst_, *cache_, *gather_file_,
                                                   config.cache_line_size_bytes, stats);

    wb_arbiter_ = std::make_unique<WritebackArbiter>(scoreboard_, stats);
    wb_arbiter_->add_source(alu_.get());
    wb_arbiter_->add_source(mul_.get());
    wb_arbiter_->add_source(div_.get());
    wb_arbiter_->add_source(tlookup_.get());
    wb_arbiter_->add_source(gather_file_.get());

    panic_ = std::make_unique<PanicController>(config.num_warps, warps_.data(), func_model);

    // Phase 4 wiring: scheduler reads each consumer's ready_out() directly.
    // Each consumer's compute_ready() runs at the top of tick() to populate
    // ready_out_ from committed state, replacing the prior pre-evaluate
    // setter pair (set_opcoll_free / set_unit_ready_fn).
    scheduler_->set_consumers(opcoll_.get(), alu_.get(), mul_.get(), div_.get(),
                              tlookup_.get(), ldst_.get());

    warp_trace_slices_.resize(config.num_warps);
    hardware_trace_slices_.resize(kHardwareTrackCount);
    initialize_trace_writer();
}

TimingModel::~TimingModel() {
    finalize_trace();
}

void TimingModel::initialize_trace_writer() {
    if (!trace_options_.enabled()) {
        return;
    }

    structured_trace_ = std::make_unique<ChromeTraceWriter>(trace_options_.output_path);
    structured_trace_->write_process_metadata(kWarpPid, "Warp States");
    for (uint32_t w = 0; w < config_.num_warps; ++w) {
        structured_trace_->write_thread_metadata(kWarpPid, static_cast<int>(w + 1),
                                                 "Warp " + std::to_string(w),
                                                 static_cast<int>(w));
    }

    structured_trace_->write_process_metadata(kHardwarePid, "Hardware Blocks");
    structured_trace_->write_thread_metadata(kHardwarePid, kOpcollTid, "Operand Collector", 0);
    structured_trace_->write_thread_metadata(kHardwarePid, kAluTid, "ALU", 1);
    structured_trace_->write_thread_metadata(kHardwarePid, kMulTid, "Multiply", 2);
    structured_trace_->write_thread_metadata(kHardwarePid, kDivTid, "Divide", 3);
    structured_trace_->write_thread_metadata(kHardwarePid, kTlookupTid, "TLookup", 4);
    structured_trace_->write_thread_metadata(kHardwarePid, kLdstTid, "LD/ST", 5);
    structured_trace_->write_thread_metadata(kHardwarePid, kCoalescerTid, "Coalescer", 6);
    structured_trace_->write_thread_metadata(kHardwarePid, kCacheTid, "Cache", 7);
    structured_trace_->write_thread_metadata(kHardwarePid, kWritebackTid, "Writeback", 8);
    structured_trace_->write_thread_metadata(kHardwarePid, kPanicTid, "Panic", 9);

    structured_trace_->write_process_metadata(kCounterPid, "Counters");
    structured_trace_->write_thread_metadata(kCounterPid, 1, "Timing Counters", 0);
    trace_metadata_written_ = true;
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
            if (input.trace.is_ecall) {
                warps_[input.warp_id].active = false;
            }
            break;
        default:
            break;
    }
}

bool TimingModel::pipeline_drained() const {
    return opcoll_->is_free() &&
           alu_->is_ready() && !alu_->has_result() &&
           mul_->is_ready() && !mul_->has_result() &&
           div_->is_ready() && !div_->has_result() &&
           tlookup_->is_ready() && !tlookup_->has_result() &&
           ldst_->is_ready() && ldst_->fifo_empty() &&
           coalescing_->is_idle() &&
           cache_->is_idle() &&
           mem_if_->is_idle() &&
           !wb_arbiter_->has_pending_work();
}

bool TimingModel::execution_units_drained() const {
    return opcoll_->is_free() &&
           alu_->is_ready() && !alu_->has_result() &&
           mul_->is_ready() && !mul_->has_result() &&
           div_->is_ready() && !div_->has_result() &&
           tlookup_->is_ready() && !tlookup_->has_result() &&
           ldst_->is_ready() && ldst_->fifo_empty() &&
           !wb_arbiter_->has_pending_work();
}

void TimingModel::discard_writeback_results() {
    auto discard = [](ExecutionUnit& unit) {
        if (unit.has_result()) {
            unit.consume_result();
        }
    };

    discard(*alu_);
    discard(*mul_);
    discard(*div_);
    discard(*tlookup_);
    discard(*gather_file_);
}

bool TimingModel::all_warps_done() const {
    for (uint32_t w = 0; w < config_.num_warps; ++w) {
        if (warps_[w].active) {
            return false;
        }
    }
    return true;
}

bool TimingModel::branch_mispredicted(const DispatchInput& input) const {
    if (!input.trace.is_branch) {
        return false;
    }

    const uint32_t predicted_next_pc = input.prediction.predicted_taken
        ? input.prediction.predicted_target
        : (input.pc + 4);
    const uint32_t actual_next_pc = input.trace.branch_taken
        ? input.trace.branch_target
        : (input.pc + 4);

    return predicted_next_pc != actual_next_pc;
}

bool TimingModel::tick() {
    cycle_++;
    stats_.total_cycles = cycle_;

    for (uint32_t w = 0; w < config_.num_warps; ++w) {
        if (warps_[w].active) {
            stats_.warp_cycles_active[w]++;
        }
    }

    if (panic_->is_active()) {
        cache_->evaluate();
        panic_->set_units_drained(execution_units_drained());
        panic_->evaluate();

        alu_->evaluate();
        mul_->evaluate();
        div_->evaluate();
        tlookup_->evaluate();
        ldst_->evaluate();
        coalescing_->evaluate();
        mem_if_->evaluate();
        cache_->drain_write_buffer();

        discard_writeback_results();

        alu_->commit();
        mul_->commit();
        div_->commit();
        tlookup_->commit();
        ldst_->commit();
        coalescing_->commit();
        cache_->commit();
        mem_if_->commit();
        gather_file_->commit();

        record_cycle_trace(false);

        if (panic_->is_done()) {
            return false;
        }
        return true;
    }

    scoreboard_.seed_next();
    cache_->evaluate();

    // Phase 4 READY/STALL discipline: each consumer reads only its own
    // committed (current_*) state and writes its ready_out_ slot. Producers
    // (scheduler / fetch) then read those signals during their evaluate()
    // this same cycle. Within this group, no compute_ready() reads another
    // stage's ready_out_, so the order does not matter for correctness;
    // scheduler-consumed signals (opcoll, units) come first for readability,
    // followed by decode (consumed by fetch) per Phase 3.
    opcoll_->compute_ready();
    alu_->compute_ready();
    mul_->compute_ready();
    div_->compute_ready();
    tlookup_->compute_ready();
    ldst_->compute_ready();
    decode_->compute_ready();

    fetch_->evaluate();
    decode_->evaluate();

    if (decode_->ebreak_detected()) {
        panic_->trigger(decode_->ebreak_warp(), decode_->ebreak_pc());
        fetch_->commit();
        decode_->commit();
        scheduler_->reset();
        opcoll_->reset();
        gather_file_->reset();
        wb_arbiter_->reset();
        record_cycle_trace(true);
        return true;
    }

    // Phase 4: scheduler reads opcoll_->ready_out() and each unit's ready_out()
    // directly inside evaluate(); the prior pre-evaluate set_opcoll_free
    // setter is gone.
    scheduler_->evaluate();

    if (scheduler_->output()) {
        opcoll_->accept(*scheduler_->output());
    }
    opcoll_->evaluate();

    if (opcoll_->output()) {
        dispatch_to_unit(*opcoll_->output());

        const auto& out = *opcoll_->output();
        if (out.trace.is_branch) {
            warps_[out.warp_id].branch_in_flight = false;
            stats_.branch_predictions++;
            branch_predictor_->update(out.pc, out.decoded, out.prediction,
                                      out.trace.branch_taken, out.trace.branch_target);
            if (branch_mispredicted(out)) {
                const uint32_t actual_target = out.trace.branch_taken
                    ? out.trace.branch_target
                    : (out.pc + 4);
                fetch_->redirect_warp(out.warp_id, actual_target);
                decode_->invalidate_warp(out.warp_id);
                stats_.branch_mispredictions++;
                stats_.branch_flushes++;
            }
        }
    }

    alu_->evaluate();
    mul_->evaluate();
    div_->evaluate();
    tlookup_->evaluate();
    ldst_->evaluate();

    // Coalescing drives HIT writes into the gather buffer. cache_->evaluate()
    // at the top of tick already deposited any FILL into the gather buffer
    // (FILL has port priority); HIT writes here stall one cycle if FILL won
    // the port this cycle.
    coalescing_->evaluate();

    mem_if_->evaluate();

    cache_->drain_write_buffer();

    wb_arbiter_->evaluate();

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
    // The gather buffer's write-port scratch must be cleared after the
    // arbiter observes has_result() for this cycle.
    gather_file_->commit();
    scoreboard_.commit();

    if (trace_enabled_) {
        trace_cycle();
    }
    record_cycle_trace(false);

    if (all_warps_done() && pipeline_drained()) {
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
    finalize_trace();
}

CycleTraceSnapshot TimingModel::build_cycle_snapshot() const {
    CycleTraceSnapshot snapshot;
    snapshot.cycle = cycle_;
    snapshot.num_warps = config_.num_warps;
    snapshot.opcoll_busy = opcoll_->busy();
    if (auto warp = opcoll_->resident_warp()) {
        snapshot.opcoll_warp = *warp;
    }
    snapshot.opcoll_cycles_remaining = opcoll_->cycles_remaining();
    snapshot.alu_busy = alu_->busy();
    snapshot.mul_busy = mul_->busy();
    snapshot.mul_pipeline_occupancy = mul_->pipeline_occupancy();
    snapshot.div_busy = div_->busy();
    snapshot.tlookup_busy = tlookup_->busy();
    snapshot.ldst_busy = ldst_->busy();
    snapshot.ldst_fifo_depth = static_cast<uint32_t>(ldst_->fifo_entries().size());
    snapshot.active_mshrs = cache_->active_mshr_count();
    snapshot.secondary_mshrs = cache_->secondary_mshr_count();
    snapshot.pinned_lines = cache_->pinned_line_count();
    snapshot.write_buffer_depth = static_cast<uint32_t>(cache_->write_buffer_size());
    snapshot.panic_active = panic_->is_active();

    const auto& scheduler_diag = scheduler_->current_diagnostics();

    for (uint32_t w = 0; w < config_.num_warps; ++w) {
        auto& warp = snapshot.warps[w];
        warp.warp_id = w;
        warp.active = warps_[w].active;
        warp.pc = warps_[w].pc;

        if (!warps_[w].active) {
            warp.state = WarpTraceState::RETIRED;
            continue;
        }

        snapshot.active_warps++;
        warp.state = WarpTraceState::AT_REST;
        warp.rest_reason = scheduler_rest_reason(scheduler_diag[w]);
        if (warp.rest_reason == WarpRestReason::NONE) {
            warp.rest_reason = WarpRestReason::WAIT_FRONTEND;
        }
        if (!warps_[w].instr_buffer.is_empty()) {
            fill_from_decoded(warp, warps_[w].instr_buffer.front().pc,
                              warps_[w].instr_buffer.front().decoded);
        }
    }

    auto set_warp = [&](uint32_t warp_id, WarpTraceState state, WarpRestReason reason,
                        uint32_t pc, uint32_t raw_instruction, ExecUnit unit,
                        uint8_t dest_reg, bool has_mem = false, uint32_t mem_addr = 0,
                        bool coalesced = false, bool branch_taken = false,
                        uint32_t branch_target = 0) {
        if (warp_id >= config_.num_warps) {
            return;
        }
        auto& warp = snapshot.warps[warp_id];
        warp.active = warps_[warp_id].active;
        warp.state = state;
        warp.rest_reason = reason;
        warp.pc = pc;
        warp.raw_instruction = raw_instruction;
        warp.target_unit = unit;
        warp.dest_reg = dest_reg;
        warp.has_memory_address = has_mem;
        warp.first_memory_address = mem_addr;
        warp.coalesced_memory = coalesced;
        warp.branch_taken = branch_taken;
        warp.branch_target = branch_target;
    };

    if (fetch_->current_output()) {
        const auto& fetch = *fetch_->current_output();
        set_warp(fetch.warp_id, WarpTraceState::FETCH, WarpRestReason::NONE,
                 fetch.pc, fetch.raw_instruction, ExecUnit::NONE, 0);
    }

    if (const auto* pending = decode_->pending_entry()) {
        set_warp(pending->warp_id, WarpTraceState::DECODE_PENDING, WarpRestReason::NONE,
                 pending->pc, pending->decoded.raw, pending->decoded.target_unit,
                 pending->decoded.rd);
    }

    if (const auto* issue = opcoll_->current_instruction()) {
        set_warp(issue->warp_id, WarpTraceState::OPERAND_COLLECT, WarpRestReason::NONE,
                 issue->pc, issue->decoded.raw, issue->decoded.target_unit, issue->decoded.rd,
                 issue->trace.is_load || issue->trace.is_store,
                 issue->trace.is_load || issue->trace.is_store ? issue->trace.mem_addresses[0] : 0,
                 false, issue->trace.branch_taken, issue->trace.branch_target);
    }

    if (const auto* pending = alu_->pending_input()) {
        set_warp(pending->warp_id, WarpTraceState::EXECUTE_ALU, WarpRestReason::NONE,
                 pending->pc, pending->decoded.raw, ExecUnit::ALU, pending->decoded.rd,
                 pending->trace.is_load || pending->trace.is_store,
                 pending->trace.is_load || pending->trace.is_store ? pending->trace.mem_addresses[0] : 0,
                 false, pending->trace.branch_taken, pending->trace.branch_target);
    }

    for (const auto& mul_entry : mul_->pipeline_snapshot()) {
        set_warp(mul_entry.warp_id, WarpTraceState::EXECUTE_MUL, WarpRestReason::NONE,
                 mul_entry.pc, mul_entry.raw_instruction, ExecUnit::MULTIPLY,
                 mul_entry.dest_reg);
    }

    if (const auto* pending = div_->pending_entry()) {
        set_warp(pending->warp_id, WarpTraceState::EXECUTE_DIV, WarpRestReason::NONE,
                 pending->pc, pending->raw_instruction, ExecUnit::DIVIDE, pending->dest_reg);
    }

    if (const auto* pending = tlookup_->pending_entry()) {
        set_warp(pending->warp_id, WarpTraceState::EXECUTE_TLOOKUP, WarpRestReason::NONE,
                 pending->pc, pending->raw_instruction, ExecUnit::TLOOKUP, pending->dest_reg);
    }

    if (const auto* pending = ldst_->pending_entry()) {
        set_warp(pending->warp_id, WarpTraceState::ADDR_GEN, WarpRestReason::NONE,
                 pending->trace.pc, pending->trace.decoded.raw, ExecUnit::LDST,
                 pending->dest_reg, pending->is_load || pending->is_store,
                 pending->is_load || pending->is_store ? pending->trace.mem_addresses[0] : 0);
    }

    for (const auto& entry : ldst_->fifo_entries()) {
        set_warp(entry.warp_id, WarpTraceState::LDST_FIFO, WarpRestReason::NONE,
                 entry.trace.pc, entry.trace.decoded.raw, ExecUnit::LDST, entry.dest_reg,
                 entry.is_load || entry.is_store,
                 entry.is_load || entry.is_store ? entry.trace.mem_addresses[0] : 0);
    }

    if (const auto* entry = coalescing_->current_entry()) {
        WarpTraceState state = WarpTraceState::COALESCING;
        WarpRestReason reason = WarpRestReason::NONE;
        if (cache_->stall_reason() == CacheStallReason::MSHR_FULL) {
            state = WarpTraceState::AT_REST;
            reason = WarpRestReason::WAIT_L1_MSHR;
        } else if (cache_->stall_reason() == CacheStallReason::WRITE_BUFFER_FULL) {
            state = WarpTraceState::AT_REST;
            reason = WarpRestReason::WAIT_L1_WRITE_BUFFER;
        }
        set_warp(entry->warp_id, state, reason, entry->trace.pc, entry->trace.decoded.raw,
                 ExecUnit::LDST, entry->dest_reg, entry->is_load || entry->is_store,
                 entry->is_load || entry->is_store ? entry->trace.mem_addresses[0] : 0,
                 coalescing_->is_coalesced());
    }

    for (uint32_t i = 0; i < cache_->mshrs().num_entries(); ++i) {
        const auto& entry = cache_->mshrs().at(i);
        if (!entry.valid) {
            continue;
        }
        set_warp(entry.warp_id, WarpTraceState::MEMORY_WAIT, WarpRestReason::WAIT_MEMORY_RESPONSE,
                 entry.pc, entry.raw_instruction, ExecUnit::LDST, entry.dest_reg,
                 true, entry.cache_line_addr * config_.cache_line_size_bytes, true);
    }

    if (cache_->pending_fill().valid) {
        const auto& entry = cache_->mshrs().at(cache_->pending_fill().response.mshr_id);
        WarpRestReason reason = entry.is_store ? WarpRestReason::WAIT_L1_WRITE_BUFFER
                                               : WarpRestReason::WAIT_MEMORY_RESPONSE;
        WarpTraceState state = entry.is_store ? WarpTraceState::AT_REST
                                              : WarpTraceState::MEMORY_WAIT;
        set_warp(entry.warp_id, state, reason, entry.pc, entry.raw_instruction,
                 ExecUnit::LDST, entry.dest_reg, true,
                 entry.cache_line_addr * config_.cache_line_size_bytes, true);
    }

    if (const auto* wb = alu_->result_entry()) {
        set_warp(wb->warp_id, WarpTraceState::WRITEBACK_WAIT, WarpRestReason::WAIT_WRITEBACK,
                 wb->pc, wb->raw_instruction, wb->source_unit, wb->dest_reg);
    }
    if (const auto* wb = mul_->result_entry()) {
        set_warp(wb->warp_id, WarpTraceState::WRITEBACK_WAIT, WarpRestReason::WAIT_WRITEBACK,
                 wb->pc, wb->raw_instruction, wb->source_unit, wb->dest_reg);
    }
    if (const auto* wb = div_->result_entry()) {
        set_warp(wb->warp_id, WarpTraceState::WRITEBACK_WAIT, WarpRestReason::WAIT_WRITEBACK,
                 wb->pc, wb->raw_instruction, wb->source_unit, wb->dest_reg);
    }
    if (const auto* wb = tlookup_->result_entry()) {
        set_warp(wb->warp_id, WarpTraceState::WRITEBACK_WAIT, WarpRestReason::WAIT_WRITEBACK,
                 wb->pc, wb->raw_instruction, wb->source_unit, wb->dest_reg);
    }
    for (uint32_t w = 0; w < gather_file_->num_buffers(); ++w) {
        const auto& buf = gather_file_->buffer(w);
        if (buf.busy && buf.filled_count == WARP_SIZE) {
            set_warp(w, WarpTraceState::WRITEBACK_WAIT, WarpRestReason::WAIT_WRITEBACK,
                     buf.pc, buf.raw_instruction, ExecUnit::LDST, buf.dest_reg);
        }
    }

    if (panic_->is_active() && !panic_->is_done()) {
        for (uint32_t w = 0; w < config_.num_warps; ++w) {
            if (snapshot.warps[w].active) {
                snapshot.warps[w].state = WarpTraceState::PANIC_DRAIN;
                snapshot.warps[w].rest_reason = WarpRestReason::NONE;
            }
        }
    }

    return snapshot;
}

void TimingModel::record_cycle_trace(bool panic_triggered) {
    last_cycle_snapshot_ = build_cycle_snapshot();
    if (structured_trace_) {
        emit_cycle_events(*last_cycle_snapshot_, panic_triggered);
    }
}

void TimingModel::emit_cycle_events(const CycleTraceSnapshot& snapshot, bool panic_triggered) {
    for (uint32_t w = 0; w < snapshot.num_warps; ++w) {
        const auto& warp = snapshot.warps[w];
        update_track_slice(warp_trace_slices_[w], kWarpPid, static_cast<int>(w + 1),
                           make_warp_key(warp), warp_slice_name(warp),
                           make_warp_args(warp, snapshot.cycle), snapshot.cycle);
    }

    if (snapshot.opcoll_busy) {
        TraceArgs args{{"cycle", static_cast<uint64_t>(snapshot.cycle)},
                       {"warp", static_cast<uint64_t>(snapshot.opcoll_warp)},
                       {"cycles_remaining", static_cast<uint64_t>(snapshot.opcoll_cycles_remaining)}};
        update_track_slice(hardware_trace_slices_[0], kHardwarePid, kOpcollTid,
                           "warp:" + std::to_string(snapshot.opcoll_warp),
                           "busy", args, snapshot.cycle);
    } else {
        update_track_slice(hardware_trace_slices_[0], kHardwarePid, kOpcollTid, "", "", {},
                           snapshot.cycle);
    }

    auto emit_busy_track = [&](size_t idx, int tid, const char* name, bool busy,
                               const WritebackEntry* wb, std::optional<uint32_t> active_warp,
                               uint32_t cycles_remaining = 0) {
        if (!busy && wb == nullptr) {
            update_track_slice(hardware_trace_slices_[idx], kHardwarePid, tid, "", "", {},
                               snapshot.cycle);
            return;
        }
        TraceArgs args{{"cycle", static_cast<uint64_t>(snapshot.cycle)}};
        std::string key = std::string(name) + ":";
        if (active_warp) {
            args.emplace_back("warp", static_cast<uint64_t>(*active_warp));
            key += "warp:" + std::to_string(*active_warp);
        }
        if (cycles_remaining > 0) {
            args.emplace_back("cycles_remaining", static_cast<uint64_t>(cycles_remaining));
            key += ":cr:" + std::to_string(cycles_remaining);
        }
        if (wb != nullptr) {
            args.emplace_back("writeback_warp", static_cast<uint64_t>(wb->warp_id));
            key += ":wb:" + std::to_string(wb->warp_id);
        }
        update_track_slice(hardware_trace_slices_[idx], kHardwarePid, tid, key, "busy", args,
                           snapshot.cycle);
    };

    emit_busy_track(1, kAluTid, "alu", snapshot.alu_busy, alu_->result_entry(),
                    alu_->active_warp(), alu_->busy() ? 1u : 0u);

    if (snapshot.mul_busy || mul_->result_entry() != nullptr) {
        TraceArgs args{{"cycle", static_cast<uint64_t>(snapshot.cycle)},
                       {"occupancy", static_cast<uint64_t>(snapshot.mul_pipeline_occupancy)}};
        if (const auto* wb = mul_->result_entry()) {
            args.emplace_back("writeback_warp", static_cast<uint64_t>(wb->warp_id));
        }
        update_track_slice(hardware_trace_slices_[2], kHardwarePid, kMulTid,
                           "mul:" + std::to_string(snapshot.mul_pipeline_occupancy) +
                               ":wb:" + std::to_string(mul_->result_entry()
                                                        ? mul_->result_entry()->warp_id : 0),
                           "busy", args, snapshot.cycle);
    } else {
        update_track_slice(hardware_trace_slices_[2], kHardwarePid, kMulTid, "", "", {},
                           snapshot.cycle);
    }

    emit_busy_track(3, kDivTid, "div", snapshot.div_busy, div_->result_entry(),
                    div_->active_warp(), div_->cycles_remaining());
    emit_busy_track(4, kTlookupTid, "tlookup", snapshot.tlookup_busy,
                    tlookup_->result_entry(), tlookup_->active_warp(),
                    tlookup_->cycles_remaining());
    emit_busy_track(5, kLdstTid, "ldst", snapshot.ldst_busy, nullptr, ldst_->active_warp(),
                    ldst_->cycles_remaining());

    if (const auto* entry = coalescing_->current_entry()) {
        TraceArgs args{{"cycle", static_cast<uint64_t>(snapshot.cycle)},
                       {"warp", static_cast<uint64_t>(entry->warp_id)},
                       {"mode", std::string(coalescing_->is_coalesced()
                                                ? "coalesced" : "serialized")},
                       {"serial_index", static_cast<uint64_t>(coalescing_->serial_index())}};
        if (cache_->stall_reason() == CacheStallReason::MSHR_FULL) {
            args.emplace_back("stall_reason", std::string("mshr_full"));
        } else if (cache_->stall_reason() == CacheStallReason::WRITE_BUFFER_FULL) {
            args.emplace_back("stall_reason", std::string("write_buffer_full"));
        }
        update_track_slice(hardware_trace_slices_[6], kHardwarePid, kCoalescerTid,
                           "coalescer:" + std::to_string(entry->warp_id) + ":" +
                               std::to_string(coalescing_->serial_index()) + ":" +
                               std::to_string(static_cast<int>(cache_->stall_reason())),
                           coalescing_->is_coalesced() ? "coalesced" : "serialized",
                           args, snapshot.cycle);
    } else {
        update_track_slice(hardware_trace_slices_[6], kHardwarePid, kCoalescerTid, "", "", {},
                           snapshot.cycle);
    }

    if (snapshot.active_mshrs > 0 || snapshot.write_buffer_depth > 0 || cache_->pending_fill().valid) {
        TraceArgs args{{"cycle", static_cast<uint64_t>(snapshot.cycle)},
                       {"active_mshrs", static_cast<uint64_t>(snapshot.active_mshrs)},
                       {"secondary_mshrs", static_cast<uint64_t>(snapshot.secondary_mshrs)},
                       {"pinned_lines", static_cast<uint64_t>(snapshot.pinned_lines)},
                       {"write_buffer_depth", static_cast<uint64_t>(snapshot.write_buffer_depth)}};
        if (cache_->pending_fill().valid) {
            args.emplace_back("pending_fill_mshr",
                              static_cast<uint64_t>(cache_->pending_fill().response.mshr_id));
        }
        update_track_slice(hardware_trace_slices_[7], kHardwarePid, kCacheTid,
                           "cache:" + std::to_string(snapshot.active_mshrs) + ":" +
                               std::to_string(snapshot.write_buffer_depth) + ":" +
                               std::to_string(cache_->pending_fill().valid),
                           "active", args, snapshot.cycle);
    } else {
        update_track_slice(hardware_trace_slices_[7], kHardwarePid, kCacheTid, "", "", {},
                           snapshot.cycle);
    }

    if (wb_arbiter_->committed_entry()) {
        const auto& wb = *wb_arbiter_->committed_entry();
        TraceArgs args{{"cycle", static_cast<uint64_t>(snapshot.cycle)},
                       {"warp", static_cast<uint64_t>(wb.warp_id)},
                       {"dest_reg", static_cast<uint64_t>(wb.dest_reg)}};
        update_track_slice(hardware_trace_slices_[8], kHardwarePid, kWritebackTid,
                           "wb:" + std::to_string(wb.warp_id) + ":" +
                               std::to_string(wb.dest_reg),
                           "commit", args, snapshot.cycle);
    } else if (wb_arbiter_->ready_source_count() > 0) {
        TraceArgs args{{"cycle", static_cast<uint64_t>(snapshot.cycle)},
                       {"ready_sources", static_cast<uint64_t>(wb_arbiter_->ready_source_count())}};
        update_track_slice(hardware_trace_slices_[8], kHardwarePid, kWritebackTid,
                           "pending:" + std::to_string(wb_arbiter_->ready_source_count()),
                           "pending", args, snapshot.cycle);
    } else {
        update_track_slice(hardware_trace_slices_[8], kHardwarePid, kWritebackTid, "", "", {},
                           snapshot.cycle);
    }

    if (snapshot.panic_active) {
        TraceArgs args{{"cycle", static_cast<uint64_t>(snapshot.cycle)},
                       {"panic_warp", static_cast<uint64_t>(panic_->panic_warp())},
                       {"step", static_cast<uint64_t>(panic_->step())}};
        update_track_slice(hardware_trace_slices_[9], kHardwarePid, kPanicTid,
                           "panic:" + std::to_string(panic_->step()), "panic_drain",
                           args, snapshot.cycle);
    } else {
        update_track_slice(hardware_trace_slices_[9], kHardwarePid, kPanicTid, "", "", {},
                           snapshot.cycle);
    }

    structured_trace_->write_counter("active_warps", snapshot.cycle, kCounterPid, 1,
                                     {{"value", static_cast<uint64_t>(snapshot.active_warps)}});
    structured_trace_->write_counter("opcoll_busy", snapshot.cycle, kCounterPid, 1,
                                     {{"value", snapshot.opcoll_busy ? 1ULL : 0ULL}});
    structured_trace_->write_counter("alu_busy", snapshot.cycle, kCounterPid, 1,
                                     {{"value", snapshot.alu_busy ? 1ULL : 0ULL}});
    structured_trace_->write_counter("mul_occupancy", snapshot.cycle, kCounterPid, 1,
                                     {{"value", static_cast<uint64_t>(snapshot.mul_pipeline_occupancy)}});
    structured_trace_->write_counter("div_busy", snapshot.cycle, kCounterPid, 1,
                                     {{"value", snapshot.div_busy ? 1ULL : 0ULL}});
    structured_trace_->write_counter("tlookup_busy", snapshot.cycle, kCounterPid, 1,
                                     {{"value", snapshot.tlookup_busy ? 1ULL : 0ULL}});
    structured_trace_->write_counter("ldst_busy", snapshot.cycle, kCounterPid, 1,
                                     {{"value", snapshot.ldst_busy ? 1ULL : 0ULL}});
    structured_trace_->write_counter("ldst_fifo_depth", snapshot.cycle, kCounterPid, 1,
                                     {{"value", static_cast<uint64_t>(snapshot.ldst_fifo_depth)}});
    structured_trace_->write_counter("active_mshrs", snapshot.cycle, kCounterPid, 1,
                                     {{"value", static_cast<uint64_t>(snapshot.active_mshrs)}});
    structured_trace_->write_counter("secondary_mshrs", snapshot.cycle, kCounterPid, 1,
                                     {{"value", static_cast<uint64_t>(snapshot.secondary_mshrs)}});
    structured_trace_->write_counter("pinned_lines", snapshot.cycle, kCounterPid, 1,
                                     {{"value", static_cast<uint64_t>(snapshot.pinned_lines)}});
    structured_trace_->write_counter("write_buffer_depth", snapshot.cycle, kCounterPid, 1,
                                     {{"value", static_cast<uint64_t>(snapshot.write_buffer_depth)}});

    if (scheduler_->current_output()) {
        const auto& issue = *scheduler_->current_output();
        structured_trace_->write_instant("issue", snapshot.cycle, kWarpPid,
                                         static_cast<int>(issue.warp_id + 1),
                                         instruction_args(snapshot.cycle, issue.warp_id,
                                                          issue.pc, issue.decoded.raw,
                                                          issue.decoded.target_unit,
                                                          issue.decoded.rd));
    }

    if (opcoll_->current_output() &&
        branch_mispredicted(*opcoll_->current_output())) {
        const auto& branch = *opcoll_->current_output();
        TraceArgs args = instruction_args(snapshot.cycle, branch.warp_id, branch.pc,
                                          branch.decoded.raw, branch.decoded.target_unit,
                                          branch.decoded.rd);
        args.emplace_back("branch_target",
                          static_cast<uint64_t>(branch.trace.branch_target));
        structured_trace_->write_instant("branch_redirect", snapshot.cycle, kWarpPid,
                                         static_cast<int>(branch.warp_id + 1), args);
    }

    if (wb_arbiter_->committed_entry()) {
        const auto& wb = *wb_arbiter_->committed_entry();
        structured_trace_->write_instant("writeback", snapshot.cycle, kWarpPid,
                                         static_cast<int>(wb.warp_id + 1),
                                         instruction_args(snapshot.cycle, wb.warp_id, wb.pc,
                                                          wb.raw_instruction, wb.source_unit,
                                                          wb.dest_reg));
    }

    if (panic_triggered) {
        structured_trace_->write_instant("panic_trigger", snapshot.cycle, kHardwarePid,
                                         kPanicTid,
                                         {{"cycle", static_cast<uint64_t>(snapshot.cycle)},
                                          {"panic_warp", static_cast<uint64_t>(panic_->panic_warp())},
                                          {"panic_pc", static_cast<uint64_t>(panic_->panic_pc())}});
    }

    if (cache_->last_miss_event().valid) {
        const auto& miss = cache_->last_miss_event();
        TraceArgs args = instruction_args(snapshot.cycle, miss.warp_id, miss.pc,
                                          miss.raw_instruction, ExecUnit::LDST, 0);
        args.emplace_back("line_addr", static_cast<uint64_t>(miss.line_addr));
        args.emplace_back("is_store", miss.is_store);
        args.emplace_back("merged_secondary", miss.merged_secondary);
        structured_trace_->write_instant("cache_miss_alloc", snapshot.cycle, kHardwarePid,
                                         kCacheTid, args);
    }

    if (cache_->last_fill_event().valid) {
        const auto& fill = cache_->last_fill_event();
        TraceArgs args = instruction_args(snapshot.cycle, fill.warp_id, fill.pc,
                                          fill.raw_instruction, ExecUnit::LDST, 0);
        args.emplace_back("line_addr", static_cast<uint64_t>(fill.line_addr));
        args.emplace_back("is_store", fill.is_store);
        args.emplace_back("chain_length_at_fill",
                          static_cast<uint64_t>(fill.chain_length_at_fill));
        args.emplace_back("deferred", fill.deferred);
        structured_trace_->write_instant("memory_response_complete", snapshot.cycle,
                                         kHardwarePid, kCacheTid, args);
    }

    if (cache_->last_drain_event().valid) {
        const auto& drain = cache_->last_drain_event();
        TraceArgs args = instruction_args(snapshot.cycle, drain.warp_id, drain.pc,
                                          drain.raw_instruction, ExecUnit::LDST, 0);
        args.emplace_back("line_addr", static_cast<uint64_t>(drain.line_addr));
        args.emplace_back("is_store", drain.is_store);
        structured_trace_->write_instant("secondary_drain", snapshot.cycle, kHardwarePid,
                                         kCacheTid, args);
    }

    if (cache_->last_pin_stall_event().valid) {
        const auto& pin = cache_->last_pin_stall_event();
        TraceArgs args{{"cycle", static_cast<uint64_t>(snapshot.cycle)},
                       {"warp_id", static_cast<uint64_t>(pin.warp_id)},
                       {"requested_line_addr",
                        static_cast<uint64_t>(pin.requested_line_addr)},
                       {"pinned_line_addr",
                        static_cast<uint64_t>(pin.pinned_line_addr)},
                       {"is_store", pin.is_store}};
        structured_trace_->write_instant("line_pin_stall", snapshot.cycle, kHardwarePid,
                                         kCacheTid, args);
    }
}

void TimingModel::update_track_slice(ActiveTraceSlice& slice, int pid, int tid,
                                     const std::string& key, const std::string& name,
                                     const TraceArgs& args, uint64_t cycle) {
    if (!structured_trace_) {
        return;
    }

    if (key.empty()) {
        flush_track_slice(slice, pid, tid, cycle);
        return;
    }

    if (slice.valid && slice.key == key) {
        return;
    }

    flush_track_slice(slice, pid, tid, cycle);
    slice.valid = true;
    slice.start_cycle = cycle;
    slice.key = key;
    slice.name = name;
    slice.args = args;
}

void TimingModel::flush_track_slice(ActiveTraceSlice& slice, int pid, int tid, uint64_t cycle) {
    if (!structured_trace_ || !slice.valid) {
        slice.valid = false;
        return;
    }

    if (cycle > slice.start_cycle) {
        structured_trace_->write_complete(slice.name, slice.start_cycle,
                                          cycle - slice.start_cycle, pid, tid,
                                          slice.args);
    }

    slice.valid = false;
    slice.key.clear();
    slice.name.clear();
    slice.args.clear();
}

void TimingModel::finalize_trace() {
    if (!structured_trace_) {
        return;
    }

    uint64_t end_cycle = cycle_ + 1;
    for (uint32_t w = 0; w < warp_trace_slices_.size(); ++w) {
        flush_track_slice(warp_trace_slices_[w], kWarpPid, static_cast<int>(w + 1), end_cycle);
    }
    const int tids[kHardwareTrackCount] = {
        kOpcollTid, kAluTid, kMulTid, kDivTid, kTlookupTid,
        kLdstTid, kCoalescerTid, kCacheTid, kWritebackTid, kPanicTid
    };
    for (size_t i = 0; i < hardware_trace_slices_.size(); ++i) {
        flush_track_slice(hardware_trace_slices_[i], kHardwarePid, tids[i], end_cycle);
    }
    structured_trace_->finalize();
}

void TimingModel::trace_cycle() const {
    std::cerr << "cycle=" << std::setw(6) << cycle_;

    if (fetch_->current_output()) {
        std::cerr << " fetch=W" << fetch_->current_output()->warp_id;
    } else {
        std::cerr << " fetch=--";
    }

    if (scheduler_->current_output()) {
        std::cerr << " issue=W" << scheduler_->current_output()->warp_id;
    } else {
        std::cerr << " issue=--";
    }

    std::cerr << " opcoll=" << (opcoll_->is_free() ? "free" : "busy");
    std::cerr << " alu=" << (alu_->is_ready() ? "rdy" : "bsy");
    std::cerr << " mul=" << (mul_->is_ready() ? "rdy" : "bsy");
    std::cerr << " div=" << (div_->is_ready() ? "rdy" : "bsy");
    std::cerr << " tlk=" << (tlookup_->is_ready() ? "rdy" : "bsy");
    std::cerr << " ldst=" << (ldst_->is_ready() ? "rdy" : "bsy");

    if (wb_arbiter_->committed_entry()) {
        const auto& wb = *wb_arbiter_->committed_entry();
        std::cerr << " wb=W" << wb.warp_id << ":x" << static_cast<int>(wb.dest_reg);
    } else {
        std::cerr << " wb=--";
    }

    std::cerr << "\n";
}

} // namespace gpu_sim
