cmake_minimum_required(VERSION 3.18)
project(ufs_emulator_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use Torch from an external path passed via -DTORCH_ROOT=...</>
# Example: cmake -DTORCH_ROOT=/path/to/torch -B build -S .
if(NOT DEFINED TORCH_ROOT)
  message(FATAL_ERROR "TORCH_ROOT not set. Pass -DTORCH_ROOT=/path/to/python/site-packages/torch")
endif()
message(STATUS "Using TORCH_ROOT: ${TORCH_ROOT}")

# Include and library directories
include_directories(
    ${TORCH_ROOT}/include
    ${TORCH_ROOT}/include/torch/csrc/api/include
)
link_directories(${TORCH_ROOT}/lib)

add_executable(ufs_emulator ufs_emulator.cpp)

# Link Torch CPU libs (adjust if using CUDA wheel)
target_link_libraries(ufs_emulator torch_cpu c10)

# Ensure C++ standard
set_property(TARGET ufs_emulator PROPERTY CXX_STANDARD 17)

# RPATH so the binary can find libs at runtime
set_target_properties(ufs_emulator PROPERTIES
  BUILD_RPATH "${TORCH_ROOT}/lib"
  INSTALL_RPATH "${TORCH_ROOT}/lib"
)

# Simple demo executable for jac_physical method
add_executable(ufs_emulator_ts ufs_emulator_ts.cpp)
target_link_libraries(ufs_emulator_ts torch_cpu c10)
set_property(TARGET ufs_emulator_ts PROPERTY CXX_STANDARD 17)
set_target_properties(ufs_emulator_ts PROPERTIES
  BUILD_RPATH "${TORCH_ROOT}/lib"
  INSTALL_RPATH "${TORCH_ROOT}/lib"
)

# Test executable for validating C++ against Python Jacobian
add_executable(ufs_emulator_ts_test ufs_emulator_ts_test.cpp)
target_link_libraries(ufs_emulator_ts_test torch_cpu c10)
set_property(TARGET ufs_emulator_ts_test PROPERTY CXX_STANDARD 17)
set_target_properties(ufs_emulator_ts_test PROPERTIES
  BUILD_RPATH "${TORCH_ROOT}/lib"
  INSTALL_RPATH "${TORCH_ROOT}/lib"
)

# Test executable for comparing Python and C++ implementations
add_executable(test_compare_python_cpp test_compare_python_cpp.cpp)
target_link_libraries(test_compare_python_cpp torch_cpu c10)
set_property(TARGET test_compare_python_cpp PROPERTY CXX_STANDARD 17)
set_target_properties(test_compare_python_cpp PROPERTIES
  BUILD_RPATH "${TORCH_ROOT}/lib"
  INSTALL_RPATH "${TORCH_ROOT}/lib"
)


# Enable testing
include(CTest)
enable_testing()

# Find Python for test data generation
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Test 1: Generate test data using Python model
# This test computes the reference Jacobian using Python's jac_physical() method
# and saves input and Jacobian for C++ validation
# Requires activating the virtual environment first
add_test(
  NAME generate_test_data_ts
  COMMAND bash -c "source /scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/mlvb/bin/activate && \
          /scratch3/NCEPDEV/da/Guillaume.Vernieres/venvs/mlvb/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_data_for_ts.py \
          /scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/aibalance/run_tair/tair.ts \
          ${CMAKE_CURRENT_BINARY_DIR}/test_ts_input.pt \
          ${CMAKE_CURRENT_BINARY_DIR}/test_ts_jacobian_ref.pt && \
          deactivate"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set_tests_properties(generate_test_data_ts PROPERTIES
  TIMEOUT 60
  LABELS "python;data_generation"
)

# Test 2: Validate C++ Jacobian against Python reference
# This test runs the C++ implementation and compares against Python results
# Tolerance set to 1e-5 for strict validation
add_test(
  NAME validate_cpp_jacobian
  COMMAND ufs_emulator_ts_test
          /scratch3/NCEPDEV/da/Guillaume.Vernieres/runs/aibalance/run_tair/tair.ts
          ${CMAKE_CURRENT_BINARY_DIR}/test_ts_input.pt
          ${CMAKE_CURRENT_BINARY_DIR}/test_ts_jacobian_ref.pt
          1e-5
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set_tests_properties(validate_cpp_jacobian PROPERTIES
  DEPENDS generate_test_data_ts
  TIMEOUT 60
  LABELS "cpp;validation"
)

# Test 3: Simple usage test (no args). Program prints Usage and exits non-zero.
add_test(NAME ufs_emulator_usage COMMAND ufs_emulator)
set_tests_properties(ufs_emulator_usage PROPERTIES
  WILL_FAIL TRUE
  LABELS "usage"
)

# Test 4: Original Python/C++ comparison test
# Assumes test data has been generated in the source directory
# Run generate_data_only.sh first if files don't exist
add_test(
  NAME compare_python_cpp
  COMMAND test_compare_python_cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/test_model.ts
          ${CMAKE_CURRENT_SOURCE_DIR}/test_input.pt
          ${CMAKE_CURRENT_SOURCE_DIR}/test_output.pt
          ${CMAKE_CURRENT_SOURCE_DIR}/test_jacobian.pt
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set_tests_properties(compare_python_cpp PROPERTIES
  TIMEOUT 300
  LABELS "comprehensive"
)

message(STATUS "Torch include dirs: ${TORCH_ROOT}/include;${TORCH_ROOT}/include/torch/csrc/api/include")
message(STATUS "Torch libraries dir: ${TORCH_ROOT}/lib")
