#include "runner/backend.h"
#include "runner/backends/perf_sim_backend.h"

namespace gpu_sim {

std::unique_ptr<Backend> create_backend(const std::string& name) {
    if (name == "perf_sim") {
        return std::make_unique<PerfSimBackend>();
    }
    return nullptr;
}

} // namespace gpu_sim
