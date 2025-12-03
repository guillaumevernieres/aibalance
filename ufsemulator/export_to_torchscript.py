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


def export_model(checkpoint_path: str, output_path: str):
    """Export trained model to TorchScript."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

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
    test_input = torch.randn(1, model_config["input_size"])
    test_output = loaded(test_input)
    print(f"Test output shape: {test_output.shape}")
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
