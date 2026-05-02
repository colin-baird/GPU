find_program(RISCV_CC
    NAMES riscv64-unknown-elf-gcc riscv32-unknown-elf-gcc riscv64-elf-gcc)

if(NOT RISCV_CC)
    message(STATUS "Skipping jalr_storm benchmark target: RISC-V cross-compiler not found")
    return()
endif()

set(JALR_KERNEL_ELF ${CMAKE_CURRENT_BINARY_DIR}/jalr_storm_kernel.elf)

add_custom_command(
    OUTPUT ${JALR_KERNEL_ELF}
    COMMAND ${RISCV_CC}
            -march=rv32im_zicsr
            -mabi=ilp32
            -nostdlib
            -nostartfiles
            -T ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
            -o ${JALR_KERNEL_ELF}
            ${CMAKE_CURRENT_SOURCE_DIR}/jalr_storm_kernel.S
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
        ${CMAKE_CURRENT_SOURCE_DIR}/jalr_storm_kernel.S
    VERBATIM)

add_custom_target(jalr_storm_kernel DEPENDS ${JALR_KERNEL_ELF})

add_executable(jalr_storm_bench jalr_storm_bench.cpp)
add_dependencies(jalr_storm_bench jalr_storm_kernel)
target_link_libraries(jalr_storm_bench PRIVATE gpu_sim_lib)
target_compile_definitions(jalr_storm_bench PRIVATE JALR_KERNEL_ELF="${JALR_KERNEL_ELF}")

add_test(NAME jalr_storm_bench COMMAND jalr_storm_bench)
