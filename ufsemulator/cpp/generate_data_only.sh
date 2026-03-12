#!/bin/bash
# Generate test data for Python/C++ comparison
# Run this in your Python environment with PyTorch installed
#
# This script will:
# 1. Export the model to TorchScript
# 2. Find and load actual training data (.npz files near the checkpoint)
# 3. Extract a few samples for testing
# 4. Compute reference outputs and Jacobians
#
# Usage:
#   ./generate_data_only.sh [checkpoint_path] [output_dir]
#
# Example:
#   ./generate_data_only.sh ../run_sst/models_sst/best_model.pt .
#   ./generate_data_only.sh ../run_sst/models_sst/best_model.pt /path/to/output

set -e

CHECKPOINT="${1:-../run_sst/models_sst/best_model.pt}"
OUTPUT_DIR="${2:-.}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint file not found: $CHECKPOINT"
    echo "Usage: $0 [checkpoint_path] [output_dir]"
    exit 1
fi

echo "=========================================="
echo "Generating Test Data (Python Environment)"
echo "=========================================="
echo "Checkpoint:  $CHECKPOINT"
echo "Output dir:  $OUTPUT_DIR"
echo ""
echo "The script will automatically search for training data (.npz files)"
echo "near the checkpoint directory."
echo ""

cd "$OUTPUT_DIR"

python3 "$(dirname "$0")/generate_test_data.py" \
    --checkpoint "$CHECKPOINT" \
    --output test_model.ts \
    --test-input test_input.pt \
    --test-output test_output.pt \
    --test-jacobian test_jacobian.pt \
    --num-samples 5 \
    --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Test data generation complete!"
    echo "=========================================="
    echo ""
    echo "Generated files in $OUTPUT_DIR:"
    ls -lh test_model.ts test_input.pt test_output.pt test_jacobian.pt 2>/dev/null || echo "Files generated"
    echo ""
    echo "Next steps:"
    echo "1. Switch to your C++ environment (load appropriate modules)"
    echo "2. Run: cd $(dirname "$0")/build && ctest -V -R compare_python_cpp"
else
    echo "❌ Failed to generate test data"
    exit 1
fi
