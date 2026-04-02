cmake_minimum_required(VERSION 3.16)
project(gpu VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(sim)
add_subdirectory(runner)

if(BUILD_TESTS)
    add_subdirectory(tests/matmul)
    add_subdirectory(tests/gemv)
    add_subdirectory(tests/fused_linear_activation)
    add_subdirectory(tests/softmax_row)
    add_subdirectory(tests/embedding_gather)
    add_subdirectory(tests/layernorm_lite)
endif()
