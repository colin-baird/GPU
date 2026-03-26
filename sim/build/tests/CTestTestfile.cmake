# CMake generated Testfile for 
# Source directory: /Users/colinbaird/Projects/GPU/sim/tests
# Build directory: /Users/colinbaird/Projects/GPU/sim/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_decoder "/Users/colinbaird/Projects/GPU/sim/build/tests/test_decoder")
set_tests_properties(test_decoder PROPERTIES  _BACKTRACE_TRIPLES "/Users/colinbaird/Projects/GPU/sim/tests/CMakeLists.txt;8;add_test;/Users/colinbaird/Projects/GPU/sim/tests/CMakeLists.txt;11;add_gpu_test;/Users/colinbaird/Projects/GPU/sim/tests/CMakeLists.txt;0;")
add_test(test_alu "/Users/colinbaird/Projects/GPU/sim/build/tests/test_alu")
set_tests_properties(test_alu PROPERTIES  _BACKTRACE_TRIPLES "/Users/colinbaird/Projects/GPU/sim/tests/CMakeLists.txt;8;add_test;/Users/colinbaird/Projects/GPU/sim/tests/CMakeLists.txt;12;add_gpu_test;/Users/colinbaird/Projects/GPU/sim/tests/CMakeLists.txt;0;")
add_test(test_functional "/Users/colinbaird/Projects/GPU/sim/build/tests/test_functional")
set_tests_properties(test_functional PROPERTIES  _BACKTRACE_TRIPLES "/Users/colinbaird/Projects/GPU/sim/tests/CMakeLists.txt;8;add_test;/Users/colinbaird/Projects/GPU/sim/tests/CMakeLists.txt;13;add_gpu_test;/Users/colinbaird/Projects/GPU/sim/tests/CMakeLists.txt;0;")
