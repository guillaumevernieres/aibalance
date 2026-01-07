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

# Simple usage test (no args). Program prints Usage and exits non-zero.
add_test(NAME ufs_emulator_usage COMMAND ufs_emulator)
set_tests_properties(ufs_emulator_usage PROPERTIES
  WILL_FAIL TRUE
)

# Python/C++ comparison test
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
)

message(STATUS "Torch include dirs: ${TORCH_ROOT}/include;${TORCH_ROOT}/include/torch/csrc/api/include")
message(STATUS "Torch libraries dir: ${TORCH_ROOT}/lib")
