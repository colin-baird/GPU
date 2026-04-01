cmake_minimum_required(VERSION 3.16)

# When built standalone (not as a subdirectory), set up the project.
if(NOT CMAKE_PROJECT_NAME OR CMAKE_PROJECT_NAME STREQUAL "gpu_sim")
    project(gpu_sim VERSION 0.1.0 LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# Simulator library
add_library(gpu_sim_lib
    src/config.cpp
    src/decoder.cpp
    src/elf_loader.cpp
    src/stats.cpp
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
    src/timing/timing_trace.cpp
    src/timing/timing_model.cpp
)

target_include_directories(gpu_sim_lib PUBLIC include)

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
