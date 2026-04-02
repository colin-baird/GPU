if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(gemv_bench VERSION 0.1.0 LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(BUILD_TESTS OFF CACHE BOOL "Build simulator tests" FORCE)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../sim ${CMAKE_BINARY_DIR}/gpu_sim_build)
    enable_testing()
endif()

find_program(RISCV_CC
    NAMES riscv64-unknown-elf-gcc riscv32-unknown-elf-gcc riscv64-elf-gcc)

if(NOT RISCV_CC)
    message(STATUS "Skipping gemv benchmark target: RISC-V cross-compiler not found")
    return()
endif()

set(GEMV_KERNEL_ELF ${CMAKE_CURRENT_BINARY_DIR}/gemv_kernel.elf)

add_custom_command(
    OUTPUT ${GEMV_KERNEL_ELF}
    COMMAND ${RISCV_CC}
            -march=rv32im_zicsr
            -mabi=ilp32
            -nostdlib
            -nostartfiles
            -T ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
            -o ${GEMV_KERNEL_ELF}
            ${CMAKE_CURRENT_SOURCE_DIR}/gemv_kernel.S
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
        ${CMAKE_CURRENT_SOURCE_DIR}/gemv_kernel.S
    VERBATIM)

add_custom_target(gemv_kernel DEPENDS ${GEMV_KERNEL_ELF})

add_executable(gemv_bench gemv_bench.cpp)
add_dependencies(gemv_bench gemv_kernel)
target_link_libraries(gemv_bench PRIVATE gpu_sim_lib)
target_compile_definitions(gemv_bench PRIVATE GEMV_KERNEL_ELF="${GEMV_KERNEL_ELF}")

add_test(NAME gemv_bench COMMAND gemv_bench)
