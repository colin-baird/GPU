cmake_minimum_required(VERSION 3.16)

# Runner library (backend abstraction + concrete backends)
add_library(runner_lib
    src/backend.cpp
    src/backends/perf_sim_backend.cpp
)

target_include_directories(runner_lib PUBLIC include)
target_link_libraries(runner_lib PUBLIC gpu_sim_lib)

# Main executable
add_executable(gpu_sim src/main.cpp)
target_link_libraries(gpu_sim PRIVATE runner_lib)
