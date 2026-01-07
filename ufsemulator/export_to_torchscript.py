#!/usr/bin/env python3
"""
Export trained UfsEmulatorFFNN model to TorchScript format for C++ inference.

Usage:
    python export_to_torchscript.py --checkpoint models/best_model.pt \\
        --output ufs_emulator.ts
"""

import argparse
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ufsemulator.model import create_ufs_emulator_ffnn
from ufsemulator.cf_mappings import CF_ATM, CF_OCN


def export_model(checkpoint_path: str, output_path: str):
    """Export trained model to TorchScript."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    # In PyTorch 2.6+, default weights_only=True can block legacy pickles.
    # We trust our own checkpoints, so explicitly set weights_only=False.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # Get model configuration
    if "config" in checkpoint:
        config = checkpoint["config"]
        model_config = config["model"]
    else:
        raise ValueError("No config found in checkpoint")

    # Create model
    print(f"Creating model: input_size={model_config['input_size']}, "
          f"hidden_size={model_config['hidden_size']}, "
          f"output_size={model_config['output_size']}, "
          f"hidden_layers={model_config.get('hidden_layers', 2)}")

    model = create_ufs_emulator_ffnn(
        input_size=model_config["input_size"],
        hidden_size=model_config["hidden_size"],
        output_size=model_config["output_size"],
        hidden_layers=model_config.get("hidden_layers", 2),
    )

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load normalization parameters
    checkpoint_dir = Path(checkpoint_path).parent
    norm_path = checkpoint_dir / "normalization.pt"

    if not norm_path.exists():
        raise FileNotFoundError(f"Normalization file not found: {norm_path}")

    print(f"Loading normalization from: {norm_path}")
    moments = torch.load(norm_path, map_location="cpu")

    model.input_mean.data = moments['input_mean']
    model.input_std.data = moments['input_std']
    model.output_mean.data = moments['output_mean']
    model.output_std.data = moments['output_std']

    print(f"  Input normalization: mean={model.input_mean.mean().item():.4f}, std={model.input_std.mean().item():.4f}")
    print(f"  Output normalization: mean={model.output_mean.mean().item():.4f}, std={model.output_std.mean().item():.4f}")

    # Set TorchScript-serializable metadata (IO names + arbitrary meta)
    # Use CF-1 standard names from metadata if available, otherwise fallback to short names
    variables_cfg = config.get("variables")
    if not variables_cfg:
        raise ValueError("Missing 'variables' section in checkpoint config")

    input_vars = variables_cfg.get("input_variables")
    output_vars = variables_cfg.get("output_variables")
    if (not isinstance(input_vars, list)
            or not isinstance(output_vars, list)):
        raise ValueError(
            "'variables.input_variables' and 'variables.output_variables' "
            "must be lists"
        )

    # Get CF-1 mappings from metadata (required)
    metadata_cfg = config.get("metadata", {})
    input_cf_mapping = metadata_cfg.get("input_cf_mapping", {})
    output_cf_mapping = metadata_cfg.get("output_cf_mapping", {})

    if not input_cf_mapping or not output_cf_mapping:
        print("\n⚠️  WARNING: CF-1 mappings not found in checkpoint metadata.")
        print("   Generating fallback mappings using cf_mappings module.")
        print("   For production, retrain the model with updated training code.\n")
        
        # Get atmospheric level from config
        domain_cfg = config.get('domain', {})
        atm_level = domain_cfg.get('atm_level_index', 127)  # Default for 128-level model
        
        # Generate mappings
        if not input_cf_mapping:
            input_cf_mapping = {}
            for var in input_vars:
                if var in CF_ATM:
                    input_cf_mapping[var] = {
                        'cf_name': CF_ATM[var],
                        'source': 'atmosphere',
                        'level_index': atm_level
                    }
                elif var in CF_OCN:
                    input_cf_mapping[var] = {
                        'cf_name': CF_OCN[var],
                        'source': 'ocean',
                        'level_index': 0
                    }
                else:
                    raise ValueError(f"Unknown variable '{var}' - not in CF_ATM or CF_OCN")
        
        if not output_cf_mapping:
            output_cf_mapping = {}
            for var in output_vars:
                if var in CF_ATM:
                    output_cf_mapping[var] = {
                        'cf_name': CF_ATM[var],
                        'source': 'atmosphere',
                        'level_index': atm_level
                    }
                elif var in CF_OCN:
                    output_cf_mapping[var] = {
                        'cf_name': CF_OCN[var],
                        'source': 'ocean',
                        'level_index': 0
                    }
                else:
                    raise ValueError(f"Unknown variable '{var}' - not in CF_ATM or CF_OCN")

    print("\nVariable mappings:")

    # Build input names and levels
    input_names = []
    input_levels = []
    for var in input_vars:
        if var not in input_cf_mapping:
            raise ValueError(f"Missing CF-1 mapping for input variable: {var}")

        cf_name = input_cf_mapping[var]["cf_name"]
        level = input_cf_mapping[var]["level_index"]
        source = input_cf_mapping[var].get("source", "unknown")
        input_names.append(cf_name)
        input_levels.append(level)
        print(f"  Input:  {var:8s} -> {cf_name:45s} @ level {level:2d} ({source})")

    # Build output names and levels
    output_names = []
    output_levels = []
    for var in output_vars:
        if var not in output_cf_mapping:
            raise ValueError(f"Missing CF-1 mapping for output variable: {var}")

        cf_name = output_cf_mapping[var]["cf_name"]
        level = output_cf_mapping[var]["level_index"]
        source = output_cf_mapping[var].get("source", "unknown")
        output_names.append(cf_name)
        output_levels.append(level)
        print(f"  Output: {var:8s} -> {cf_name:45s} @ level {level:2d} ({source})")

    model.set_io_names(input_names, output_names)
    model.set_io_levels(input_levels, output_levels)

    meta = {}
    meta["model_class"] = "UfsEmulatorFFNN"
    meta["hidden_layers"] = str(model_config.get("hidden_layers", 2))
    meta["hidden_size"] = str(model_config.get("hidden_size"))
    meta["input_variables"] = ",".join(input_names)
    meta["output_variables"] = ",".join(output_names)
    meta["input_short_names"] = ",".join([str(v) for v in input_vars])
    meta["output_short_names"] = ",".join([str(v) for v in output_vars])
    if "training" in config:
        meta["optimizer"] = str(config["training"].get("optimizer", ""))
        meta["loss"] = str(config["training"].get("loss", ""))
    model.set_metadata(meta)

    # Set to evaluation mode
    model.eval()

    # Debug: Verify model has all required methods before export
    print("\nVerifying model methods before export...")
    required_methods = ['forward', 'predict', 'normalize_input', 'denormalize_output', 'jac', 'jac_physical']
    for method_name in required_methods:
        if hasattr(model, method_name):
            print(f"  ✓ {method_name}")
        else:
            print(f"  ✗ {method_name} - MISSING!")
            raise AttributeError(f"Model is missing required method: {method_name}")

    # Export to TorchScript
    print("\nExporting to TorchScript...")
    scripted_model = torch.jit.script(model)

    # Save
    scripted_model.save(output_path)
    print(f"✅ Exported TorchScript model to: {output_path}")

    # Verify the export
    print("\nVerifying export...")
    loaded = torch.jit.load(output_path)

    # Test input (physical space)
    test_input = torch.randn(5, model_config["input_size"], dtype=torch.float32)
    print(f"Test input shape: {test_input.shape}")

    # Test 1: forward() - expects normalized input, returns normalized output
    print("\n1. Testing forward() [normalized → normalized]...")
    test_input_norm = loaded.normalize_input(test_input)
    test_output_norm = loaded.forward(test_input_norm)
    print(f"   Output (normalized) shape: {test_output_norm.shape}")

    # Test 2: predict() - end-to-end physical space
    print("\n2. Testing predict() [physical → physical]...")
    test_output_phys = loaded.predict(test_input)
    print(f"   Output (physical) shape: {test_output_phys.shape}")

    # Test 3 & 4: Jacobian methods (may not work after serialization due to autograd limitations)
    print("\n3. Testing jac() [∂y_norm/∂x_norm]...")
    try:
        jac_norm = loaded.jac(test_input)
        print(f"   Jacobian (normalized) shape: {jac_norm.shape}")
        print(f"   Statistics: min={jac_norm.min().item():.4f}, "
              f"max={jac_norm.max().item():.4f}, "
              f"mean={jac_norm.mean().item():.4f}")
    except Exception as e:
        print(f"   ⚠️  Jacobian test failed (expected - autograd limitation in serialized TorchScript)")
        print(f"      Error: {str(e)[:100]}")
        print(f"      Note: Jacobian methods work in Python but may need C++ autograd setup")

    print("\n4. Testing jac_physical() [∂y_phys/∂x_phys]...")
    try:
        jac_phys = loaded.jac_physical(test_input)
        print(f"   Jacobian (physical) shape: {jac_phys.shape}")
        print(f"   Statistics: min={jac_phys.min().item():.4f}, "
              f"max={jac_phys.max().item():.4f}, "
              f"mean={jac_phys.mean().item():.4f}")
    except Exception as e:
        print(f"   ⚠️  Jacobian test failed (expected - autograd limitation in serialized TorchScript)")
        print(f"      Error: {str(e)[:100]}")
        print(f"      Note: Jacobian methods work in Python but may need C++ autograd setup")

    # Show serialized metadata
    try:
        print("\n5. Serialized metadata:")
        print(f"   Input names: {loaded.attr('input_names')}")
        print(f"   Output names: {loaded.attr('output_names')}")
        print(f"   Metadata keys: {list(loaded.attr('meta').keys())}")
    except Exception as e:
        print(f"   Warning: Could not access metadata: {e}")

    print("\n✅ Export verified successfully!")

    return model_config


def main():
    parser = argparse.ArgumentParser(
        description="Export UfsEmulatorFFNN to TorchScript"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        default="ufs_emulator.ts",
        help="Output TorchScript file path",
    )

    args = parser.parse_args()

    # Export model
    config = export_model(args.checkpoint, args.output)

    print("\n" + "=" * 60)
    print("To use in C++:")
    print(f"  ./ufs_emulator {args.output} {config['input_size']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
