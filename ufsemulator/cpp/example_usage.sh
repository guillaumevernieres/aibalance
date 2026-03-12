#!/bin/bash
# Example: How to set up and run the Python/C++ comparison test

echo "==================================================================="
echo "Python/C++ Inference and Jacobian Comparison Test Setup"
echo "==================================================================="
echo ""

# Step 1: Configure the build
echo "Step 1: Configuring CMake build..."
echo "----------------------------------------"
TORCH_ROOT=$(python3 -c 'import torch; print(torch.__path__[0])')
echo "Detected TORCH_ROOT: $TORCH_ROOT"

cmake -DTORCH_ROOT=$TORCH_ROOT -B build -S .

if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed"
    exit 1
fi

# Step 2: Build the test executable
echo ""
echo "Step 2: Building test executable..."
echo "----------------------------------------"
cmake --build build --target test_compare_python_cpp

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

# Step 3: Run using CTest
echo ""
echo "Step 3: Running tests with CTest..."
echo "----------------------------------------"
echo "This will:"
echo "  1. Generate test data using Python (exports model + computes reference)"
echo "  2. Run C++ comparison test"
echo ""

cd build
ctest -V -R compare_python_cpp

if [ $? -eq 0 ]; then
    echo ""
    echo "==================================================================="
    echo "✅ SUCCESS: All tests passed!"
    echo "==================================================================="
    echo ""
    echo "The C++ implementation produces identical results to Python for:"
    echo "  ✓ Inference (forward pass)"
    echo "  ✓ Jacobian (gradient computation)"
    echo ""
else
    echo ""
    echo "==================================================================="
    echo "❌ FAILURE: Some tests failed"
    echo "==================================================================="
    echo ""
    echo "Check the output above for details on the differences."
    echo ""
fi

# Alternative: Run tests individually
echo ""
echo "Alternative: You can also run tests individually:"
echo "  cd build && ctest -N                    # List all tests"
echo "  cd build && ctest -R generate_test_data # Run data generation only"
echo "  cd build && ctest -R compare_python_cpp # Run comparison only"
echo ""
echo "Or use the helper script:"
echo "  ./run_comparison_test.sh ../run_sst/models_sst/best_model.pt"
