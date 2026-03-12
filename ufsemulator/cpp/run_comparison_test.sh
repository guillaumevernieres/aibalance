#!/bin/bash
# Run the Python/C++ comparison test
#
# This script:
# 1. Generates test data using Python (exports model and computes reference values)
# 2. Builds the C++ test program
# 3. Runs the C++ comparison test
#
# Usage:
#   ./run_comparison_test.sh [checkpoint_path]
#   ./run_comparison_test.sh --skip-generation  # Skip Python data generation
#   ./run_comparison_test.sh --generate-only [checkpoint_path]  # Only generate data
#
# Example:
#   ./run_comparison_test.sh ../run_sst/models_sst/best_model.pt
#   ./run_comparison_test.sh --generate-only ../run_sst/models_sst/best_model.pt
#   ./run_comparison_test.sh --skip-generation

set -e  # Exit on error

# Parse arguments
SKIP_GENERATION=false
GENERATE_ONLY=false
CHECKPOINT=""

if [ "$1" = "--skip-generation" ]; then
    SKIP_GENERATION=true
elif [ "$1" = "--generate-only" ]; then
    GENERATE_ONLY=true
    CHECKPOINT="${2:-../run_sst/models_sst/best_model.pt}"
else
    CHECKPOINT="${1:-../run_sst/models_sst/best_model.pt}"
fi

echo "================================"
echo "Python/C++ Comparison Test"
echo "================================"

# Step 1: Generate test data (unless skipped)
if [ "$SKIP_GENERATION" = false ]; then
    if [ ! -f "$CHECKPOINT" ]; then
        echo "❌ Error: Checkpoint file not found: $CHECKPOINT"
        echo "Usage: $0 [checkpoint_path]"
        echo "       $0 --skip-generation"
        echo "       $0 --generate-only [checkpoint_path]"
        exit 1
    fi
    
    echo "Checkpoint: $CHECKPOINT"
    echo ""
    echo "Step 1: Generating test data with Python..."
    python3 generate_test_data.py \
        --checkpoint "$CHECKPOINT" \
        --output test_model.ts \
        --test-input test_input.pt \
        --test-output test_output.pt \
        --test-jacobian test_jacobian.pt \
        --num-samples 5 \
        --seed 42

    if [ $? -ne 0 ]; then
        echo "❌ Failed to generate test data"
        exit 1
    fi
    
    if [ "$GENERATE_ONLY" = true ]; then
        echo ""
        echo "================================"
        echo "✅ Test data generation complete!"
        echo "================================"
        echo ""
        echo "Generated files:"
        echo "  - test_model.ts"
        echo "  - test_input.pt"
        echo "  - test_output.pt"
        echo "  - test_jacobian.pt"
        echo ""
        echo "To run the C++ test in a different environment, use:"
        echo "  ./run_comparison_test.sh --skip-generation"
        exit 0
    fi
else
    echo "Skipping Python data generation (using existing files)"
    echo ""
    
    # Check if required files exist
    if [ ! -f "test_model.ts" ] || [ ! -f "test_input.pt" ] || \
       [ ! -f "test_output.pt" ] || [ ! -f "test_jacobian.pt" ]; then
        echo "❌ Error: Required test files not found!"
        echo "Please run data generation first:"
        echo "  ./run_comparison_test.sh --generate-only [checkpoint_path]"
        exit 1
    fi
fi

echo ""
echo "Step 2: Building C++ test program..."

# Check if build directory exists and has been configured
if [ ! -f "build/Makefile" ] && [ ! -f "build/build.ninja" ]; then
    echo "⚠️  Build directory not configured. Please run cmake first:"
    echo "  cmake -DTORCH_ROOT=\$(python3 -c 'import torch; print(torch.__path__[0])') -B build -S ."
    exit 1
fi

# Build the test
cmake --build build --target test_compare_python_cpp

if [ $? -ne 0 ]; then
    echo "❌ Failed to build test program"
    exit 1
fi

echo ""
echo "Step 3: Running C++ comparison test..."
./build/test_compare_python_cpp test_model.ts test_input.pt test_output.pt test_jacobian.pt

TEST_RESULT=$?

echo ""
echo "================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Tests failed with exit code: $TEST_RESULT"
fi
echo "================================"

exit $TEST_RESULT
