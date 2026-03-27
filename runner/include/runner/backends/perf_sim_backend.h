#pragma once

#include "runner/backend.h"
#include "gpu_sim/functional/functional_model.h"

namespace gpu_sim {

// Performance simulator backend: functional and cycle-accurate timing models.
class PerfSimBackend : public Backend {
public:
    int run(const ProgramImage& image, SimConfig& config,
            int argc, char* argv[]) override;
    std::string name() const override { return "perf_sim"; }

private:
    void load_image(FunctionalModel& model, const ProgramImage& image);
    int run_functional_only(FunctionalModel& model, const SimConfig& config,
                            uint64_t max_cycles);
};

} // namespace gpu_sim
