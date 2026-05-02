cmake_minimum_required(VERSION 3.16)
project(gpu VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
endif()

set(GPU_MAX_WARPS "8" CACHE STRING "Maximum resident warps per SM (compile-time array bound)")

add_subdirectory(sim)
add_subdirectory(runner)

if(BUILD_TESTS)
    add_subdirectory(tests/matmul)
    add_subdirectory(tests/gemv)
    add_subdirectory(tests/fused_linear_activation)
    add_subdirectory(tests/softmax_row)
    add_subdirectory(tests/embedding_gather)
    add_subdirectory(tests/layernorm_lite)
    add_subdirectory(tests/synthetic)

    # Cross-stage accessor naming lint. Phase 6 of the
    # naming-and-access-discipline refactor: enforcement mode (any
    # finding fails the test). Use `tools/lint_timing_naming.py
    # --report-only` locally to inspect findings without failing.
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME timing_naming_lint
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_SOURCE_DIR}/tools/lint_timing_naming.py)
    endif()
endif()
