#!/usr/bin/env python3
"""
Generate test data for comparing Python and C++ inference and Jacobian computation.

This script:
1. Loads a trained model checkpoint
2. Exports it to TorchScript format
3. Loads actual training data and thins it for testing
4. Computes reference outputs (inference and Jacobian) using Python
5. Saves everything to files for C++ test comparison

Usage:
    python generate_test_data.py --checkpoint ../runs/models_aice/best_model.pt \
        --data-path ../run_sst \
        --output test_model.ts
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ufsemulator.export_to_torchscript import export_model


def load_training_data(checkpoint_path: str, num_samples: int = 5, seed: int = 42):
    """
    Load and thin actual training data from the checkpoint or data directory.

    Args:
        checkpoint_path: Path to model checkpoint (used to find data directory)
        num_samples: Number of samples to extract
        seed: Random seed for reproducible sampling

    Returns:
        torch.Tensor of shape [num_samples, input_size]
    """
    print(f"\nLoading training data...")

    # Try to find training data near the checkpoint
    checkpoint_dir = Path(checkpoint_path).parent.parent

    # Look for .npz files (preprocessed data)
    npz_files = list(checkpoint_dir.glob("*.npz"))

    if not npz_files:
        # Try looking in run_* directories
        for run_dir in checkpoint_dir.glob("run_*"):
            npz_files = list(run_dir.glob("*.npz"))
            if npz_files:
                break

    if not npz_files:
        print("Warning: No .npz training data found.")
        print("Falling back to random data generation.")
        return None

    # Use the first npz file found
    data_file = npz_files[0]
    print(f"Loading data from: {data_file}")

    data = np.load(data_file)

    # Try to find input arrays (commonly named 'inputs', 'X', 'features', etc.)
    possible_keys = ['inputs', 'X', 'features', 'input', 'x', 'train_inputs']
    input_key = None

    for key in possible_keys:
        if key in data:
            input_key = key
            break

    if input_key is None:
        # Just take the first large array
        for key in data.keys():
            if data[key].ndim >= 2:
                input_key = key
                print(f"Using array '{key}' as input data")
                break

    if input_key is None:
        print("Warning: Could not find suitable input data in .npz file")
        print(f"Available keys: {list(data.keys())}")
        return None

    input_data = data[input_key]

    # Ensure 2D shape [samples, features]
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    elif input_data.ndim > 2:
        # Flatten all but first dimension
        input_data = input_data.reshape(input_data.shape[0], -1)

    total_samples = input_data.shape[0]
    print(f"Found {total_samples} samples with {input_data.shape[1]} features")

    # Randomly select samples
    np.random.seed(seed)
    if num_samples > total_samples:
        print(f"Warning: Requested {num_samples} samples but only {total_samples} available")
        num_samples = total_samples
        selected_indices = np.arange(total_samples)
    else:
        selected_indices = np.random.choice(total_samples, num_samples, replace=False)
        selected_indices = np.sort(selected_indices)

    print(f"Selected {num_samples} samples at indices: {selected_indices.tolist()}")

    selected_data = input_data[selected_indices]

    # Convert to tensor
    test_input = torch.from_numpy(selected_data).float()

    print(f"Input data shape: {test_input.shape}")
    print(f"Input statistics:")
    print(f"  min: {test_input.min().item():.6f}")
    print(f"  max: {test_input.max().item():.6f}")
    print(f"  mean: {test_input.mean().item():.6f}")
    print(f"  std: {test_input.std().item():.6f}")

    return test_input


def generate_test_data(
    torchscript_path: str,
    checkpoint_path: str,
    test_input_path: str = "test_input.pt",
    test_output_path: str = "test_output.pt",
    test_jacobian_path: str = "test_jacobian.pt",
    num_samples: int = 5,
    seed: int = 42,
):
    """
    Generate test data for C++ comparison using actual training data.

    Args:
        torchscript_path: Path to exported TorchScript model
        checkpoint_path: Path to model checkpoint (used to find training data)
        test_input_path: Path to save test input tensor
        test_output_path: Path to save reference output tensor
        test_jacobian_path: Path to save reference Jacobian tensor
        num_samples: Number of test samples to extract
        seed: Random seed for reproducibility
    """
    print("\n" + "=" * 60)
    print("Generating test data for Python/C++ comparison")
    print("=" * 60)

    # Load the exported TorchScript model
    print(f"\nLoading TorchScript model from: {torchscript_path}")
    model = torch.jit.load(torchscript_path)
    model.eval()

    # Determine input size from model
    try:
        input_names = model.attr("input_names")
        input_size = len(input_names.tolist())
        print(f"Input size from metadata: {input_size}")
    except Exception:
        # Fallback: try to infer from first linear layer
        for name, param in model.named_parameters():
            if "0.weight" in name and len(param.shape) == 2:
                input_size = param.shape[1]
                print(f"Input size inferred from model parameters: {input_size}")
                break
        else:
            raise ValueError("Could not determine input size from model")

    # Try to load actual training data
    test_input = load_training_data(checkpoint_path, num_samples, seed)

    # Fall back to random data if loading failed
    if test_input is None:
        print(f"\nGenerating {num_samples} random test samples as fallback...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        test_input = torch.randn(num_samples, input_size, dtype=torch.float32)

    # Verify input size matches model
    if test_input.shape[1] != input_size:
        print(f"Warning: Input size mismatch. Data has {test_input.shape[1]} features but model expects {input_size}")
        print("Adjusting data to match model...")
        if test_input.shape[1] > input_size:
            # Truncate
            test_input = test_input[:, :input_size]
        else:
            # Pad with zeros
            padding = torch.zeros(test_input.shape[0], input_size - test_input.shape[1])
            test_input = torch.cat([test_input, padding], dim=1)

    # Compute reference output (inference)
    print("Computing reference inference output...")
    with torch.no_grad():
        test_output = model(test_input)

    output_size = test_output.shape[1]
    print(f"Output size: {output_size}")

    # Compute reference Jacobian using jac_physical() method
    print("Computing reference Jacobian using jac_physical()...")
    # Jacobian shape: [num_samples, output_size, input_size]
    test_jacobian = model.jac_physical(test_input)

    print(f"Jacobian shape: {test_jacobian.shape}")
    print(f"First Jacobian element [0,0,0]: {test_jacobian[0, 0, 0].item():.8f}")

    # Save test data
    print(f"\nSaving test input to: {test_input_path}")
    torch.save(test_input, test_input_path)

    print(f"Saving reference output to: {test_output_path}")
    torch.save(test_output, test_output_path)

    print(f"Saving reference Jacobian to: {test_jacobian_path}")
    torch.save(test_jacobian, test_jacobian_path)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Test Data Summary:")
    print("=" * 60)
    print(f"Input shape:    {test_input.shape}")
    print(f"Output shape:   {test_output.shape}")
    print(f"Jacobian shape: {test_jacobian.shape}")
    print(f"\nInput statistics:")
    print(f"  min: {test_input.min().item():.6f}")
    print(f"  max: {test_input.max().item():.6f}")
    print(f"  mean: {test_input.mean().item():.6f}")
    print(f"\nOutput statistics:")
    print(f"  min: {test_output.min().item():.6f}")
    print(f"  max: {test_output.max().item():.6f}")
    print(f"  mean: {test_output.mean().item():.6f}")
    print(f"\nJacobian statistics:")
    print(f"  min: {test_jacobian.min().item():.6f}")
    print(f"  max: {test_jacobian.max().item():.6f}")
    print(f"  mean: {test_jacobian.mean().item():.6f}")
    print("=" * 60)
    print("\n✅ Test data generation complete!")

    return test_input, test_output, test_jacobian


def main():
    parser = argparse.ArgumentParser(
        description="Generate test data for Python/C++ comparison using real training data"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        default="test_model.ts",
        help="Output TorchScript model path (default: test_model.ts)",
    )
    parser.add_argument(
        "--test-input",
        default="test_input.pt",
        help="Output test input path (default: test_input.pt)",
    )
    parser.add_argument(
        "--test-output",
        default="test_output.pt",
        help="Output reference output path (default: test_output.pt)",
    )
    parser.add_argument(
        "--test-jacobian",
        default="test_jacobian.pt",
        help="Output reference Jacobian path (default: test_jacobian.pt)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of test samples to extract from training data (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )

    args = parser.parse_args()

    # First export the model to TorchScript
    print("Step 1: Exporting model to TorchScript...")
    export_model(args.checkpoint, args.output)

    # Then generate test data from training data
    print("\nStep 2: Generating test data from training data...")
    generate_test_data(
        torchscript_path=args.output,
        checkpoint_path=args.checkpoint,
        test_input_path=args.test_input,
        test_output_path=args.test_output,
        test_jacobian_path=args.test_jacobian,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("All files generated successfully!")
    print("=" * 60)
    print(f"Model:    {args.output}")
    print(f"Input:    {args.test_input}")
    print(f"Output:   {args.test_output}")
    print(f"Jacobian: {args.test_jacobian}")
    print("=" * 60)


if __name__ == "__main__":
    main()
