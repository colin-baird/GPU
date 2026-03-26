add_library(catch_main OBJECT catch_main.cpp)
target_include_directories(catch_main PUBLIC vendor)

function(add_gpu_test TEST_NAME TEST_SRC)
    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} PRIVATE gpu_sim_lib catch_main)
    target_include_directories(${TEST_NAME} PRIVATE vendor)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

add_gpu_test(test_decoder test_decoder.cpp)
add_gpu_test(test_alu test_alu.cpp)
add_gpu_test(test_functional test_functional.cpp)
