find_program(RISCV_CC
    NAMES riscv64-unknown-elf-gcc riscv32-unknown-elf-gcc riscv64-elf-gcc)

if(NOT RISCV_CC)
    message(STATUS "Skipping layernorm_lite benchmark target: RISC-V cross-compiler not found")
    return()
endif()

set(LAYERNORM_LITE_KERNEL_ELF ${CMAKE_CURRENT_BINARY_DIR}/layernorm_lite_kernel.elf)

add_custom_command(
    OUTPUT ${LAYERNORM_LITE_KERNEL_ELF}
    COMMAND ${RISCV_CC}
            -march=rv32im_zicsr
            -mabi=ilp32
            -nostdlib
            -nostartfiles
            -T ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
            -o ${LAYERNORM_LITE_KERNEL_ELF}
            ${CMAKE_CURRENT_SOURCE_DIR}/layernorm_lite_kernel.S
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
        ${CMAKE_CURRENT_SOURCE_DIR}/layernorm_lite_kernel.S
    VERBATIM)

add_custom_target(layernorm_lite_kernel DEPENDS ${LAYERNORM_LITE_KERNEL_ELF})

add_executable(layernorm_lite_bench layernorm_lite_bench.cpp)
add_dependencies(layernorm_lite_bench layernorm_lite_kernel)
target_link_libraries(layernorm_lite_bench PRIVATE gpu_sim_lib)
target_compile_definitions(layernorm_lite_bench PRIVATE
    LAYERNORM_LITE_KERNEL_ELF="${LAYERNORM_LITE_KERNEL_ELF}")

add_test(NAME layernorm_lite_bench COMMAND layernorm_lite_bench)
