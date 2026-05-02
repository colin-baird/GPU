find_program(RISCV_CC
    NAMES riscv64-unknown-elf-gcc riscv32-unknown-elf-gcc riscv64-elf-gcc)

if(NOT RISCV_CC)
    message(STATUS "Skipping line_boundary_load benchmark target: RISC-V cross-compiler not found")
    return()
endif()

set(LBL_KERNEL_ELF ${CMAKE_CURRENT_BINARY_DIR}/line_boundary_load_kernel.elf)

add_custom_command(
    OUTPUT ${LBL_KERNEL_ELF}
    COMMAND ${RISCV_CC}
            -march=rv32im_zicsr
            -mabi=ilp32
            -nostdlib
            -nostartfiles
            -T ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
            -o ${LBL_KERNEL_ELF}
            ${CMAKE_CURRENT_SOURCE_DIR}/line_boundary_load_kernel.S
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
        ${CMAKE_CURRENT_SOURCE_DIR}/line_boundary_load_kernel.S
    VERBATIM)

add_custom_target(line_boundary_load_kernel DEPENDS ${LBL_KERNEL_ELF})

add_executable(line_boundary_load_bench line_boundary_load_bench.cpp)
add_dependencies(line_boundary_load_bench line_boundary_load_kernel)
target_link_libraries(line_boundary_load_bench PRIVATE gpu_sim_lib)
target_compile_definitions(line_boundary_load_bench PRIVATE LBL_KERNEL_ELF="${LBL_KERNEL_ELF}")

add_test(NAME line_boundary_load_bench COMMAND line_boundary_load_bench)
