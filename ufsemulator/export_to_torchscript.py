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
from ufsemulator.cf_mappings import CF_ATM, CF_OCN, DEFAULT_ATM_LEVEL


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

    if norm_path.exists():
        print(f"Loading normalization from: {norm_path}")
        moments = torch.load(norm_path, map_location="cpu")
        model.input_mean.data = moments[0]
        model.input_std.data = moments[1]
    else:
        print("Warning: No normalization file found, using defaults")

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

    # Try to get CF-1 mappings from metadata
    metadata_cfg = config.get("metadata", {})
    input_cf_mapping = metadata_cfg.get("input_cf_mapping", {})
    output_cf_mapping = metadata_cfg.get("output_cf_mapping", {})

    # Fallback: Create CF-1 mappings if missing (for old checkpoints)
    if not input_cf_mapping or not output_cf_mapping:
        print("\nWarning: CF-1 mappings not found in checkpoint metadata.")
        print("Creating fallback mappings from cf_mappings module...")

        # Get atmospheric level from domain config
        # This should be nlevs-1 where nlevs is the number of vertical levels in the atmospheric data
        # For 64-level model: atm_level_index = 63
        # For 128-level model: atm_level_index = 127
        domain_cfg = config.get('domain', {})
        atm_level = domain_cfg.get('atm_level_index', DEFAULT_ATM_LEVEL)

        print(f"Note: Using atmospheric level index {atm_level} (should be nlevs-1)")
        print(f"      If this is incorrect, set 'domain.atm_level_index' in your config file.")

        # Create mappings for all variables
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

        print(f"Created fallback mappings using atmospheric level: {atm_level}\n")

    # Use CF-1 standard names (long form) for input/output names
    # Store level indices separately
    input_names = []
    input_levels = []
    for var in input_vars:
        if var in input_cf_mapping:
            cf_name = input_cf_mapping[var]["cf_name"]
            level = input_cf_mapping[var]["level_index"]
            source = input_cf_mapping[var].get("source", "unknown")
            input_names.append(cf_name)
            input_levels.append(level)
            print(f"  Input: {var:8s} -> {cf_name:45s} @ level {level:2d} ({source})")
        else:
            # Fallback to short name if no mapping found
            print(f"Warning: No CF-1 mapping found for input '{var}', using short name")
            input_names.append(str(var))
            input_levels.append(-1)  # -1 indicates no level info

    output_names = []
    output_levels = []
    for var in output_vars:
        if var in output_cf_mapping:
            cf_name = output_cf_mapping[var]["cf_name"]
            level = output_cf_mapping[var]["level_index"]
            source = output_cf_mapping[var].get("source", "unknown")
            output_names.append(cf_name)
            output_levels.append(level)
            print(f"  Output: {var:8s} -> {cf_name:45s} @ level {level:2d} ({source})")
        else:
            print(f"Warning: No CF-1 mapping found for output '{var}', using short name")
            output_names.append(str(var))
            output_levels.append(-1)

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

    # Export to TorchScript
    print("Exporting to TorchScript...")
    scripted_model = torch.jit.script(model)

    # Save
    scripted_model.save(output_path)
    print(f"✅ Exported TorchScript model to: {output_path}")

    # Verify the export
    print("\nVerifying export...")
    loaded = torch.jit.load(output_path)
    test_input = torch.randn(1, model_config["input_size"])\
        .to(torch.float32)
    test_output = loaded(test_input)
    print(f"Test output shape: {test_output.shape}")

    # Show serialized IO names (optional)
    try:
        print("Serialized input names:", loaded.attr("input_names"))
        print("Serialized output names:", loaded.attr("output_names"))
        print("Serialized meta:", loaded.attr("meta"))
    except Exception:
        print("Note: IO names/meta not found on scripted module.")

    print("✅ Export verified successfully!")

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
