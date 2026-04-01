#include "gpu_sim/timing/timing_trace.h"
#include <stdexcept>

namespace gpu_sim {

const char* to_string(WarpTraceState state) {
    switch (state) {
        case WarpTraceState::FETCH:            return "fetch";
        case WarpTraceState::DECODE_PENDING:   return "decode_pending";
        case WarpTraceState::OPERAND_COLLECT:  return "operand_collect";
        case WarpTraceState::EXECUTE_ALU:      return "execute_alu";
        case WarpTraceState::EXECUTE_MUL:      return "execute_mul";
        case WarpTraceState::EXECUTE_DIV:      return "execute_div";
        case WarpTraceState::EXECUTE_TLOOKUP:  return "execute_tlookup";
        case WarpTraceState::ADDR_GEN:         return "addr_gen";
        case WarpTraceState::LDST_FIFO:        return "ldst_fifo";
        case WarpTraceState::COALESCING:       return "coalescing";
        case WarpTraceState::MEMORY_WAIT:      return "memory_wait";
        case WarpTraceState::WRITEBACK_WAIT:   return "writeback_wait";
        case WarpTraceState::AT_REST:          return "at_rest";
        case WarpTraceState::PANIC_DRAIN:      return "panic_drain";
        case WarpTraceState::RETIRED:          return "retired";
        default:                               return "unknown";
    }
}

const char* to_string(WarpRestReason reason) {
    switch (reason) {
        case WarpRestReason::NONE:                 return "none";
        case WarpRestReason::WAIT_FRONTEND:        return "wait_frontend";
        case WarpRestReason::WAIT_SCOREBOARD:      return "wait_scoreboard";
        case WarpRestReason::WAIT_OPCOLL:          return "wait_opcoll";
        case WarpRestReason::WAIT_UNIT_ALU:        return "wait_unit_alu";
        case WarpRestReason::WAIT_UNIT_MULTIPLY:   return "wait_unit_multiply";
        case WarpRestReason::WAIT_UNIT_DIVIDE:     return "wait_unit_divide";
        case WarpRestReason::WAIT_UNIT_TLOOKUP:    return "wait_unit_tlookup";
        case WarpRestReason::WAIT_UNIT_LDST:       return "wait_unit_ldst";
        case WarpRestReason::WAIT_ROUND_ROBIN:     return "wait_round_robin";
        case WarpRestReason::WAIT_L1_MSHR:         return "wait_l1_mshr";
        case WarpRestReason::WAIT_L1_WRITE_BUFFER: return "wait_l1_write_buffer";
        case WarpRestReason::WAIT_MEMORY_RESPONSE: return "wait_memory_response";
        case WarpRestReason::WAIT_WRITEBACK:       return "wait_writeback";
        default:                                   return "unknown";
    }
}

ChromeTraceWriter::ChromeTraceWriter(const std::string& path) : path_(path) {
    out_.open(path_);
    if (!out_.is_open()) {
        throw std::runtime_error("failed to open trace file: " + path_);
    }
    out_ << "{\"traceEvents\":[\n";
}

ChromeTraceWriter::~ChromeTraceWriter() {
    finalize();
}

void ChromeTraceWriter::finalize() {
    if (finalized_ || !out_.is_open()) {
        return;
    }
    out_ << "\n]}\n";
    out_.close();
    finalized_ = true;
}

void ChromeTraceWriter::begin_event() {
    if (!first_event_) {
        out_ << ",\n";
    }
    first_event_ = false;
}

void ChromeTraceWriter::write_escaped(std::ostream& out, const std::string& value) {
    out << '"';
    for (char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"':  out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:   out << c; break;
        }
    }
    out << '"';
}

void ChromeTraceWriter::write_args(std::ostream& out, const TraceArgs& args) {
    out << "\"args\":{";
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        write_escaped(out, args[i].first);
        out << ':';
        std::visit([&out](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, std::string>) {
                write_escaped(out, value);
            } else if constexpr (std::is_same_v<T, bool>) {
                out << (value ? "true" : "false");
            } else {
                out << value;
            }
        }, args[i].second);
    }
    out << '}';
}

void ChromeTraceWriter::write_process_metadata(int pid, const std::string& name) {
    begin_event();
    out_ << "{\"ph\":\"M\",\"pid\":" << pid << ",\"tid\":0,\"name\":\"process_name\",";
    out_ << "\"args\":{\"name\":";
    write_escaped(out_, name);
    out_ << "}}";
}

void ChromeTraceWriter::write_thread_metadata(int pid, int tid, const std::string& name,
                                              int sort_index) {
    begin_event();
    out_ << "{\"ph\":\"M\",\"pid\":" << pid << ",\"tid\":" << tid
         << ",\"name\":\"thread_name\",\"args\":{\"name\":";
    write_escaped(out_, name);
    out_ << "}}";

    begin_event();
    out_ << "{\"ph\":\"M\",\"pid\":" << pid << ",\"tid\":" << tid
         << ",\"name\":\"thread_sort_index\",\"args\":{\"sort_index\":" << sort_index
         << "}}";
}

void ChromeTraceWriter::write_complete(const std::string& name, uint64_t ts, uint64_t dur,
                                       int pid, int tid, const TraceArgs& args) {
    begin_event();
    out_ << "{\"ph\":\"X\",\"name\":";
    write_escaped(out_, name);
    out_ << ",\"ts\":" << ts << ",\"dur\":" << dur
         << ",\"pid\":" << pid << ",\"tid\":" << tid << ',';
    write_args(out_, args);
    out_ << '}';
}

void ChromeTraceWriter::write_instant(const std::string& name, uint64_t ts, int pid, int tid,
                                      const TraceArgs& args) {
    begin_event();
    out_ << "{\"ph\":\"i\",\"s\":\"t\",\"name\":";
    write_escaped(out_, name);
    out_ << ",\"ts\":" << ts << ",\"pid\":" << pid << ",\"tid\":" << tid << ',';
    write_args(out_, args);
    out_ << '}';
}

void ChromeTraceWriter::write_counter(const std::string& name, uint64_t ts, int pid, int tid,
                                      const TraceArgs& args) {
    begin_event();
    out_ << "{\"ph\":\"C\",\"name\":";
    write_escaped(out_, name);
    out_ << ",\"ts\":" << ts << ",\"pid\":" << pid << ",\"tid\":" << tid << ',';
    write_args(out_, args);
    out_ << '}';
}

} // namespace gpu_sim
