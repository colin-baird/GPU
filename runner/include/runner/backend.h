#pragma once

#include "gpu_sim/elf_loader.h"
#include "gpu_sim/config.h"
#include <memory>
#include <string>

namespace gpu_sim {

// Abstract base class for execution backends.
// Each backend consumes a parsed ProgramImage and runs it to completion.
class Backend {
public:
    virtual ~Backend() = default;

    // Run the program to completion. Returns an exit code (0 = success).
    virtual int run(const ProgramImage& image, SimConfig& config,
                    int argc, char* argv[]) = 0;

    virtual std::string name() const = 0;
};

// Create a backend by name. Returns nullptr if the name is unrecognized.
std::unique_ptr<Backend> create_backend(const std::string& name);

} // namespace gpu_sim
