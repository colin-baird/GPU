find_program(RISCV_CC
    NAMES riscv64-unknown-elf-gcc riscv32-unknown-elf-gcc riscv64-elf-gcc)

if(NOT RISCV_CC)
    message(STATUS "Skipping rr_tiebreak benchmark target: RISC-V cross-compiler not found")
    return()
endif()

set(RR_TIEBREAK_KERNEL_ELF ${CMAKE_CURRENT_BINARY_DIR}/rr_tiebreak_kernel.elf)

add_custom_command(
    OUTPUT ${RR_TIEBREAK_KERNEL_ELF}
    COMMAND ${RISCV_CC}
            -march=rv32im_zicsr
            -mabi=ilp32
            -nostdlib
            -nostartfiles
            -T ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
            -o ${RR_TIEBREAK_KERNEL_ELF}
            ${CMAKE_CURRENT_SOURCE_DIR}/rr_tiebreak_kernel.S
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
        ${CMAKE_CURRENT_SOURCE_DIR}/rr_tiebreak_kernel.S
    VERBATIM)

add_custom_target(rr_tiebreak_kernel DEPENDS ${RR_TIEBREAK_KERNEL_ELF})

add_executable(rr_tiebreak_bench rr_tiebreak_bench.cpp)
add_dependencies(rr_tiebreak_bench rr_tiebreak_kernel)
target_link_libraries(rr_tiebreak_bench PRIVATE gpu_sim_lib)
target_compile_definitions(rr_tiebreak_bench PRIVATE RR_TIEBREAK_KERNEL_ELF="${RR_TIEBREAK_KERNEL_ELF}")

add_test(NAME rr_tiebreak_bench COMMAND rr_tiebreak_bench)
