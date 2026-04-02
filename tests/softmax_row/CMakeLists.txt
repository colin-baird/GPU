find_program(RISCV_CC
    NAMES riscv64-unknown-elf-gcc riscv32-unknown-elf-gcc riscv64-elf-gcc)

if(NOT RISCV_CC)
    message(STATUS "Skipping softmax_row benchmark target: RISC-V cross-compiler not found")
    return()
endif()

set(SOFTMAX_ROW_KERNEL_ELF ${CMAKE_CURRENT_BINARY_DIR}/softmax_row_kernel.elf)

add_custom_command(
    OUTPUT ${SOFTMAX_ROW_KERNEL_ELF}
    COMMAND ${RISCV_CC}
            -march=rv32im_zicsr
            -mabi=ilp32
            -nostdlib
            -nostartfiles
            -T ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
            -o ${SOFTMAX_ROW_KERNEL_ELF}
            ${CMAKE_CURRENT_SOURCE_DIR}/softmax_row_kernel.S
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
        ${CMAKE_CURRENT_SOURCE_DIR}/softmax_row_kernel.S
    VERBATIM)

add_custom_target(softmax_row_kernel DEPENDS ${SOFTMAX_ROW_KERNEL_ELF})

add_executable(softmax_row_bench softmax_row_bench.cpp)
add_dependencies(softmax_row_bench softmax_row_kernel)
target_link_libraries(softmax_row_bench PRIVATE gpu_sim_lib)
target_compile_definitions(softmax_row_bench PRIVATE SOFTMAX_ROW_KERNEL_ELF="${SOFTMAX_ROW_KERNEL_ELF}")

add_test(NAME softmax_row_bench COMMAND softmax_row_bench)
