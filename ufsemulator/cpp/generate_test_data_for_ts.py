#!/usr/bin/env python3
"""
Generate test data for ufs_emulator_ts C++ executable.
Computes physical Jacobian using Python and saves input/output for C++ validation.
"""

import torch
import sys
import os

def main():
    if len(sys.argv) < 4:
        print("Usage: generate_test_data_for_ts.py <model.ts> <output_input.pt> <output_jacobian.pt>")
        sys.exit(1)

    model_path = sys.argv[1]
    output_input_path = sys.argv[2]
    output_jacobian_path = sys.argv[3]

    # Load TorchScript model
    print(f"Loading model from {model_path}...")
    model = torch.jit.load(model_path)
    model.eval()

    # Determine input size from model's normalization parameters
    # Get the input_mean attribute to determine expected input size
    input_mean = model.input_mean
    input_size = input_mean.shape[0]
    print(f"Model expects input size: {input_size}")

    # Create test input (batch_size=1, input_size determined from model)
    # Use realistic values from actual model run (Node 0)
    # Order: sea_water_potential_temperature, sea_ice_area_fraction,
    #        skin_temperature_at_surface, water_vapor_mixing_ratio_wrt_moist_air
    print("Creating test input with realistic values...")
    test_input = torch.tensor([[
        11.6968,      # sea_water_potential_temperature
        0.0,          # sea_ice_area_fraction
        284.341,      # skin_temperature_at_surface
        2.50374e-06   # water_vapor_mixing_ratio_wrt_moist_air
    ]], dtype=torch.float32)

    print(f"Test input values: SST={test_input[0,0].item():.4f}, "
          f"AICE={test_input[0,1].item():.4f}, "
          f"Tskin={test_input[0,2].item():.3f}, "
          f"Q={test_input[0,3].item():.6e}")
    print(f"Expected d(Tair)/d(SST) ≈ 0.188939 (from reference run)")

    # Compute physical Jacobian using Python
    # Note: Gradients must be enabled for Jacobian computation
    print("Computing physical Jacobian using Python model.jac_physical()...")
    jacobian_python = model.jac_physical(test_input)

    # Save test data
    print(f"Saving test input to {output_input_path}...")
    torch.save(test_input, output_input_path)

    print(f"Saving Python Jacobian to {output_jacobian_path}...")
    torch.save(jacobian_python, output_jacobian_path)

    # Print summary for verification
    print(f"\nTest data generated successfully!")
    print(f"Input shape: {test_input.shape}")
    print(f"Jacobian shape: {jacobian_python.shape}")
    print(f"Jacobian mean: {jacobian_python.mean().item():.6f}")
    print(f"Jacobian std: {jacobian_python.std().item():.6f}")
    print(f"Jacobian min: {jacobian_python.min().item():.6f}")
    print(f"Jacobian max: {jacobian_python.max().item():.6f}")

    # Print first few Jacobian values for debugging
    print(f"\nPython Jacobian values (first 5x5 block):")
    jac_flat = jacobian_python.flatten()
    for i in range(min(25, jac_flat.numel())):
        if i % 5 == 0 and i > 0:
            print()
        print(f"{jac_flat[i].item():12.6e}", end=" ")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
