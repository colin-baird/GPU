cmake_minimum_required(VERSION 3.16)
project(gpu_sim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Main library
add_library(gpu_sim_lib
    src/config.cpp
    src/decoder.cpp
    src/elf_loader.cpp
    src/stats.cpp
    src/backend.cpp
    src/backends/perf_sim_backend.cpp
    src/functional/alu.cpp
    src/functional/functional_model.cpp
    src/timing/fetch_stage.cpp
    src/timing/decode_stage.cpp
    src/timing/warp_scheduler.cpp
    src/timing/operand_collector.cpp
    src/timing/alu_unit.cpp
    src/timing/multiply_unit.cpp
    src/timing/divide_unit.cpp
    src/timing/tlookup_unit.cpp
    src/timing/ldst_unit.cpp
    src/timing/writeback_arbiter.cpp
    src/timing/mshr.cpp
    src/timing/cache.cpp
    src/timing/coalescing_unit.cpp
    src/timing/memory_interface.cpp
    src/timing/panic_controller.cpp
    src/timing/timing_model.cpp
)

target_include_directories(gpu_sim_lib PUBLIC include)

# Main executable
add_executable(gpu_sim src/main.cpp)
target_link_libraries(gpu_sim PRIVATE gpu_sim_lib)

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
