find_program(RISCV_CC
    NAMES riscv64-unknown-elf-gcc riscv32-unknown-elf-gcc riscv64-elf-gcc)

if(NOT RISCV_CC)
    message(STATUS "Skipping mshr_same_line_race benchmark target: RISC-V cross-compiler not found")
    return()
endif()

set(MSLR_KERNEL_ELF ${CMAKE_CURRENT_BINARY_DIR}/mshr_same_line_race_kernel.elf)

add_custom_command(
    OUTPUT ${MSLR_KERNEL_ELF}
    COMMAND ${RISCV_CC}
            -march=rv32im_zicsr
            -mabi=ilp32
            -nostdlib
            -nostartfiles
            -T ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
            -o ${MSLR_KERNEL_ELF}
            ${CMAKE_CURRENT_SOURCE_DIR}/mshr_same_line_race_kernel.S
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
        ${CMAKE_CURRENT_SOURCE_DIR}/mshr_same_line_race_kernel.S
    VERBATIM)

add_custom_target(mshr_same_line_race_kernel DEPENDS ${MSLR_KERNEL_ELF})

add_executable(mshr_same_line_race_bench mshr_same_line_race_bench.cpp)
add_dependencies(mshr_same_line_race_bench mshr_same_line_race_kernel)
target_link_libraries(mshr_same_line_race_bench PRIVATE gpu_sim_lib)
target_compile_definitions(mshr_same_line_race_bench PRIVATE MSLR_KERNEL_ELF="${MSLR_KERNEL_ELF}")

add_test(NAME mshr_same_line_race_bench COMMAND mshr_same_line_race_bench)
