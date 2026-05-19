# CMake generated Testfile for 
# Source directory: /workspace
# Build directory: /workspace/build_fixed
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(timing_naming_lint "/usr/bin/python3" "/workspace/tools/lint_timing_naming.py" "--compile-db" "/workspace/build_fixed/compile_commands.json" "--sim-root" "/workspace/sim")
set_tests_properties(timing_naming_lint PROPERTIES  _BACKTRACE_TRIPLES "/workspace/CMakeLists.txt;33;add_test;/workspace/CMakeLists.txt;0;")
add_test(signal_diagram_ast_snapshot "/usr/bin/python3" "/workspace/tests/test_signal_diagram.py")
set_tests_properties(signal_diagram_ast_snapshot PROPERTIES  ENVIRONMENT "GPU_SIGNAL_COMPILE_DB=/workspace/build_fixed/compile_commands.json" _BACKTRACE_TRIPLES "/workspace/CMakeLists.txt;38;add_test;/workspace/CMakeLists.txt;0;")
subdirs("sim")
subdirs("runner")
subdirs("tests/matmul")
subdirs("tests/gemv")
subdirs("tests/fused_linear_activation")
subdirs("tests/softmax_row")
subdirs("tests/embedding_gather")
subdirs("tests/layernorm_lite")
subdirs("tests/synthetic")
