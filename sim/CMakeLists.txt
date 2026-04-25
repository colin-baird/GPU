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
    src/timing/branch_predictor.cpp
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
    src/timing/load_gather_buffer.cpp
    src/timing/memory_interface.cpp
    src/timing/panic_controller.cpp
    src/timing/timing_trace.cpp
    src/timing/timing_model.cpp
)

target_include_directories(gpu_sim_lib PUBLIC include)
target_compile_definitions(gpu_sim_lib PUBLIC GPU_MAX_WARPS=${GPU_MAX_WARPS})

# DRAMSim3 (vendored via FetchContent). When the option is OFF the source file
# that depends on DRAMSim3 is not compiled and the GPU_SIM_USE_DRAMSIM3 macro
# is not defined; timing_model.cpp / config.cpp gate the dramsim3 backend on
# the macro and reject memory_backend=dramsim3 with a clear error.
option(GPU_SIM_USE_DRAMSIM3 "Build and link DRAMSim3 as the external DRAM model" ON)
if(GPU_SIM_USE_DRAMSIM3)
    include(FetchContent)
    FetchContent_Declare(
        dramsim3
        GIT_REPOSITORY https://github.com/umd-memsys/DRAMsim3.git
        GIT_TAG        29817593b3389f1337235d63cac515024ab8fd6e
    )
    FetchContent_GetProperties(dramsim3)
    if(NOT dramsim3_POPULATED)
        FetchContent_Populate(dramsim3)
        add_subdirectory(${dramsim3_SOURCE_DIR} ${dramsim3_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
    target_sources(gpu_sim_lib PRIVATE src/timing/dramsim3_memory.cpp)
    target_link_libraries(gpu_sim_lib PUBLIC dramsim3)
    target_compile_definitions(gpu_sim_lib PUBLIC GPU_SIM_USE_DRAMSIM3=1)
endif()

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
