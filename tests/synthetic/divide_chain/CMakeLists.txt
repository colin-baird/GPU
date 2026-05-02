find_program(RISCV_CC
    NAMES riscv64-unknown-elf-gcc riscv32-unknown-elf-gcc riscv64-elf-gcc)

if(NOT RISCV_CC)
    message(STATUS "Skipping divide_chain benchmark target: RISC-V cross-compiler not found")
    return()
endif()

set(DIVCHAIN_KERNEL_ELF ${CMAKE_CURRENT_BINARY_DIR}/divide_chain_kernel.elf)

add_custom_command(
    OUTPUT ${DIVCHAIN_KERNEL_ELF}
    COMMAND ${RISCV_CC}
            -march=rv32im_zicsr
            -mabi=ilp32
            -nostdlib
            -nostartfiles
            -T ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
            -o ${DIVCHAIN_KERNEL_ELF}
            ${CMAKE_CURRENT_SOURCE_DIR}/divide_chain_kernel.S
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/link.ld
        ${CMAKE_CURRENT_SOURCE_DIR}/divide_chain_kernel.S
    VERBATIM)

add_custom_target(divide_chain_kernel DEPENDS ${DIVCHAIN_KERNEL_ELF})

add_executable(divide_chain_bench divide_chain_bench.cpp)
add_dependencies(divide_chain_bench divide_chain_kernel)
target_link_libraries(divide_chain_bench PRIVATE gpu_sim_lib)
target_compile_definitions(divide_chain_bench PRIVATE DIVCHAIN_KERNEL_ELF="${DIVCHAIN_KERNEL_ELF}")

add_test(NAME divide_chain_bench COMMAND divide_chain_bench)
