find_program(RISCV_CC
    NAMES riscv64-unknown-elf-gcc riscv32-unknown-elf-gcc riscv64-elf-gcc)

if(NOT RISCV_CC)
    message(STATUS "Skipping matmul benchmark target: RISC-V cross-compiler not found")
    return()
endif()

set(MATMUL_KERNEL_ELF ${CMAKE_CURRENT_BINARY_DIR}/matmul_kernel.elf)

add_custom_command(
    OUTPUT ${MATMUL_KERNEL_ELF}
    COMMAND ${RISCV_CC}
            -march=rv32im_zicsr
            -mabi=ilp32
            -nostdlib
            -nostartfiles
            -T ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
            -o ${MATMUL_KERNEL_ELF}
            ${CMAKE_CURRENT_SOURCE_DIR}/matmul_kernel.S
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_kernel.S
    VERBATIM)

add_custom_target(matmul_kernel DEPENDS ${MATMUL_KERNEL_ELF})

add_executable(matmul_bench matmul_bench.cpp)
add_dependencies(matmul_bench matmul_kernel)
target_link_libraries(matmul_bench PRIVATE gpu_sim_lib)
target_compile_definitions(matmul_bench PRIVATE MATMUL_KERNEL_ELF="${MATMUL_KERNEL_ELF}")

add_test(NAME matmul_bench COMMAND matmul_bench)
