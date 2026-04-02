find_program(RISCV_CC
    NAMES riscv64-unknown-elf-gcc riscv32-unknown-elf-gcc riscv64-elf-gcc)

if(NOT RISCV_CC)
    message(STATUS "Skipping fused_linear_activation benchmark target: RISC-V cross-compiler not found")
    return()
endif()

set(FUSED_LINEAR_ACTIVATION_KERNEL_ELF ${CMAKE_CURRENT_BINARY_DIR}/fused_linear_activation_kernel.elf)

add_custom_command(
    OUTPUT ${FUSED_LINEAR_ACTIVATION_KERNEL_ELF}
    COMMAND ${RISCV_CC}
            -march=rv32im_zicsr
            -mabi=ilp32
            -nostdlib
            -nostartfiles
            -T ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
            -o ${FUSED_LINEAR_ACTIVATION_KERNEL_ELF}
            ${CMAKE_CURRENT_SOURCE_DIR}/fused_linear_activation_kernel.S
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
        ${CMAKE_CURRENT_SOURCE_DIR}/fused_linear_activation_kernel.S
    VERBATIM)

add_custom_target(fused_linear_activation_kernel DEPENDS ${FUSED_LINEAR_ACTIVATION_KERNEL_ELF})

add_executable(fused_linear_activation_bench fused_linear_activation_bench.cpp)
add_dependencies(fused_linear_activation_bench fused_linear_activation_kernel)
target_link_libraries(fused_linear_activation_bench PRIVATE gpu_sim_lib)
target_compile_definitions(fused_linear_activation_bench PRIVATE
    FUSED_LINEAR_ACTIVATION_KERNEL_ELF="${FUSED_LINEAR_ACTIVATION_KERNEL_ELF}")

add_test(NAME fused_linear_activation_bench COMMAND fused_linear_activation_bench)
