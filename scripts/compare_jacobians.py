#!/usr/bin/env python3
"""
Compare Jacobian ∂tair/∂sst from PyTorch (.pt), TorchScript (.ts in Python), and C++ (.ts in C++).

Usage:
    python compare_jacobians.py \
        --atm-file <atm.nc> \
        --ocean-file <ocean.nc> \
        --model <model.pt> \
        --torchscript <model.ts> \
        --cpp-jacobian <jacobian_*.nc> \
        --config <config.yaml>
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import netCDF4 as nc
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ufsemulator.model import UfsEmulatorFFNN
from ufsemulator.data import IceDataPreparer
import yaml


def create_diverging_colormap():
    """
    Create a custom diverging colormap:
    - Cool blues/purples for negative values
    - White at zero
    - Jet colors for positive values
    """
    # Get the jet colormap for positive values
    jet = plt.cm.get_cmap('jet', 256)
    jet_colors = jet(np.linspace(0, 1, 128))

    # Get cool colormap for negative values (reversed so darkest is most negative)
    cool = plt.cm.get_cmap('cool_r', 256)
    cool_colors = cool(np.linspace(1, 0, 128))  # Reversed

    # Combine: cool for negative, jet for positive
    all_colors = np.vstack((cool_colors, jet_colors))

    # Create the colormap
    diverging_cmap = mcolors.LinearSegmentedColormap.from_list('diverging_jet', all_colors)

    return diverging_cmap


def load_model(model_path: str):
    """Load PyTorch model from checkpoint."""
    print(f"Loading PyTorch model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Get configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_config = config['model']
    else:
        raise ValueError("No config found in checkpoint")

    # Create model
    model = UfsEmulatorFFNN(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        output_size=model_config['output_size'],
        hidden_layers=model_config.get('hidden_layers', 2)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load normalization
    norm_path = Path(model_path).parent / "normalization.pt"
    if norm_path.exists():
        moments = torch.load(norm_path, map_location='cpu', weights_only=False)
        model.input_mean.data = moments['input_mean']
        model.input_std.data = moments['input_std']
        model.output_mean.data = moments['output_mean']
        model.output_std.data = moments['output_std']
        print(f"  Loaded normalization from {norm_path}")
    else:
        raise FileNotFoundError(f"Normalization file not found: {norm_path}")

    model.eval()
    print(f"  Model loaded successfully")

    return model, config


def load_cpp_jacobian(jacobian_file: str):
    """Load Jacobian from C++ output NetCDF file."""
    print(f"\nLoading C++ Jacobian from: {jacobian_file}")

    with nc.Dataset(jacobian_file, 'r') as ds:
        lons = ds.variables['lon'][:]
        lats = ds.variables['lat'][:]
        dtair_dsst = ds.variables['dtair_div_dsst'][:, 0]  # Extract first (only) vertical level

        # Handle fill values
        fill_value = ds.variables['dtair_div_dsst']._FillValue
        dtair_dsst = np.ma.masked_equal(dtair_dsst, fill_value).filled(np.nan)

    print(f"  C++ Jacobian shape: {dtair_dsst.shape}")
    print(f"  C++ Jacobian range: [{np.nanmin(dtair_dsst):.6f}, {np.nanmax(dtair_dsst):.6f}]")
    print(f"  C++ Jacobian valid points: {np.count_nonzero(~np.isnan(dtair_dsst))}")

    return lons, lats, dtair_dsst


def compute_torchscript_jacobian(atm_file: str, ocean_file: str, torchscript_path: str, config_path: str):
    """Compute Jacobian using TorchScript model (.ts file)."""
    print(f"\nComputing TorchScript Jacobian...")
    print(f"  TorchScript model: {torchscript_path}")
    print(f"  Atm file: {atm_file}")
    print(f"  Ocean file: {ocean_file}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load TorchScript model
    print(f"  Loading TorchScript model from {torchscript_path}")
    model = torch.jit.load(torchscript_path, map_location='cpu')
    model.eval()
    print(f"  TorchScript model loaded successfully")

    # Load and prepare data
    preparer = IceDataPreparer(config)
    data = preparer.read_netcdf_data_pair(atm_file, ocean_file)
    patterns, targets, lons, lats = preparer.filter_data(data, max_patterns=None)

    print(f"  Data shape: {patterns.shape}")

    # Get input variables
    input_vars = config.get('variables', {}).get('input_variables', [])
    output_vars = config.get('variables', {}).get('output_variables', ['aice'])
    print(f"  Input variables: {input_vars}")
    print(f"  Output variables: {output_vars}")

    # Find sst index
    try:
        sst_idx = input_vars.index('sst')
    except ValueError:
        raise ValueError(f"'sst' not found in input variables: {input_vars}")

    print(f"  SST is input index: {sst_idx}")
    print(f"  Computing d({output_vars[0]})/d(sst) using TorchScript model.jac_physical()")

    # Convert to torch tensor
    features_tensor = torch.tensor(patterns, dtype=torch.float32)

    # Compute Jacobian in physical space using model.jac_physical() method
    # Returns shape: [batch_size, output_size, input_size]
    print(f"  Running TorchScript model.jac_physical()...")
    jacobian_full = model.jac_physical(features_tensor)

    print(f"  Full Jacobian shape: {jacobian_full.shape}")

    # Extract ∂output[0]/∂input[sst_idx]
    jacobian_np = jacobian_full[:, 0, sst_idx].detach().numpy()

    print(f"  TorchScript Jacobian range: [{jacobian_np.min():.6f}, {jacobian_np.max():.6f}]")
    print(f"  TorchScript Jacobian mean: {jacobian_np.mean():.6f}")
    print(f"  TorchScript Jacobian std: {jacobian_np.std():.6f}")

    # Debug: Show sample
    print(f"\n  Debug: Sample Jacobian matrix for first 3 points:")
    for i in range(min(3, jacobian_full.shape[0])):
        print(f"    Point {i}:")
        print(f"      Input (physical): {features_tensor[i].numpy()}")
        print(f"      Full Jacobian [output=0, all inputs]: {jacobian_full[i, 0, :].numpy()}")
        print(f"      Extracted d(output[0])/d(sst[{sst_idx}]): {jacobian_np[i]:.6f}")

    return lons, lats, jacobian_np


def compute_pytorch_jacobian(atm_file: str, ocean_file: str, model_path: str, config_path: str):
    """Compute Jacobian using PyTorch model.jac() method."""
    print(f"\nComputing PyTorch Jacobian...")
    print(f"  Model: {model_path}")
    print(f"  Atm file: {atm_file}")
    print(f"  Ocean file: {ocean_file}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    model, model_config = load_model(model_path)

    # Load and prepare data
    preparer = IceDataPreparer(config)
    data = preparer.read_netcdf_data_pair(atm_file, ocean_file)
    patterns, targets, lons, lats = preparer.filter_data(data, max_patterns=None)

    print(f"  Data shape: {patterns.shape}")

    # Get input variables
    input_vars = config.get('variables', {}).get('input_variables', [])
    output_vars = config.get('variables', {}).get('output_variables', ['aice'])
    print(f"  Input variables: {input_vars}")
    print(f"  Output variables: {output_vars}")

    # Find sst index
    try:
        sst_idx = input_vars.index('sst')
    except ValueError:
        raise ValueError(f"'sst' not found in input variables: {input_vars}")

    print(f"  SST is input index: {sst_idx}")
    print(f"  Computing d({output_vars[0]})/d(sst) using model.jac_physical()")

    # Convert to torch tensor
    features_tensor = torch.tensor(patterns, dtype=torch.float32)

    # Compute Jacobian in physical space using model.jac_physical() method
    # Returns shape: [batch_size, output_size, input_size]
    print(f"  Running model.jac_physical()...")
    jacobian_full = model.jac_physical(features_tensor)

    print(f"  Full Jacobian shape: {jacobian_full.shape}")

    # Extract ∂output[0]/∂input[sst_idx]
    # jacobian_full[:, 0, sst_idx] gives d(output_0)/d(sst) for all points
    jacobian_np = jacobian_full[:, 0, sst_idx].detach().numpy()

    print(f"  PyTorch Jacobian range: [{jacobian_np.min():.6f}, {jacobian_np.max():.6f}]")
    print(f"  PyTorch Jacobian mean: {jacobian_np.mean():.6f}")
    print(f"  PyTorch Jacobian std: {jacobian_np.std():.6f}")

    # Debug: Show sample of full Jacobian matrix for first few points
    print(f"\n  Debug: Sample Jacobian matrix for first 3 points:")
    for i in range(min(3, jacobian_full.shape[0])):
        print(f"    Point {i}:")
        print(f"      Input (physical): {features_tensor[i].numpy()}")
        print(f"      Full Jacobian [output=0, all inputs]: {jacobian_full[i, 0, :].numpy()}")
        print(f"      Extracted d(output[0])/d(sst[{sst_idx}]): {jacobian_np[i]:.6f}")

    return lons, lats, jacobian_np


def plot_comparison(py_lons, py_lats, pytorch_jac,
                   ts_lons, ts_lats, torchscript_jac,
                   cpp_lons, cpp_lats, cpp_jac,
                   output_file='jacobian_comparison.png'):
    """Create 3-panel visual comparison: PyTorch (.pt), TorchScript (.ts), and C++ Jacobians."""

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Filter out NaN values for PyTorch
    py_valid_mask = ~np.isnan(pytorch_jac)
    py_lons_valid = py_lons[py_valid_mask]
    py_lats_valid = py_lats[py_valid_mask]
    py_jac_valid = pytorch_jac[py_valid_mask]

    # Filter out NaN values for TorchScript
    ts_valid_mask = ~np.isnan(torchscript_jac)
    ts_lons_valid = ts_lons[ts_valid_mask]
    ts_lats_valid = ts_lats[ts_valid_mask]
    ts_jac_valid = torchscript_jac[ts_valid_mask]

    # Filter out NaN values for C++
    cpp_valid_mask = ~np.isnan(cpp_jac)
    cpp_lons_valid = cpp_lons[cpp_valid_mask]
    cpp_lats_valid = cpp_lats[cpp_valid_mask]
    cpp_jac_valid = cpp_jac[cpp_valid_mask]

    print(f"\nVisual Comparison Info:")
    print(f"  PyTorch (.pt):     {len(py_jac_valid)} valid points")
    print(f"  TorchScript (.ts): {len(ts_jac_valid)} valid points")
    print(f"  C++:               {len(cpp_jac_valid)} valid points")
    print(f"  PyTorch Jacobian range:     [{np.nanmin(py_jac_valid):.6f}, {np.nanmax(py_jac_valid):.6f}]")
    print(f"  TorchScript Jacobian range: [{np.nanmin(ts_jac_valid):.6f}, {np.nanmax(ts_jac_valid):.6f}]")
    print(f"  C++ Jacobian range:         [{np.nanmin(cpp_jac_valid):.6f}, {np.nanmax(cpp_jac_valid):.6f}]")
    print(f"  Note: Grids are different - this is a visual comparison only")

    # Use asymmetric colorbar range: limit negative to -0.1, positive based on data
    vmin = -0.02
    vmax = 0.2  #max(np.nanmax(py_jac_valid), np.nanmax(ts_jac_valid), np.nanmax(cpp_jac_valid))

    # Get custom diverging colormap
    cmap = create_diverging_colormap()

    # Use TwoSlopeNorm to ensure zero maps to the center of the colormap
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # Panel 1: PyTorch Jacobian (.pt checkpoint)
    ax1 = axes[0]
    scatter1 = ax1.scatter(py_lons_valid, py_lats_valid, c=py_jac_valid,
                          s=1, cmap=cmap, norm=norm)
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.set_title(f'PyTorch .pt (∂tair/∂tref)\n{len(py_jac_valid)} points',
                  fontsize=14, fontweight='bold')
    ax1.set_aspect('equal', adjustable='box')
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    cbar1.set_ticks(np.linspace(vmin, vmax, 9))
    ax1.grid(True, alpha=0.3)

    # Panel 2: TorchScript Jacobian (.ts file, Python)
    ax2 = axes[1]
    scatter2 = ax2.scatter(ts_lons_valid, ts_lats_valid, c=ts_jac_valid,
                          s=1, cmap=cmap, norm=norm)
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)
    ax2.set_title(f'TorchScript .ts Python (∂tair/∂tref)\n{len(ts_jac_valid)} points',
                  fontsize=14, fontweight='bold')
    ax2.set_aspect('equal', adjustable='box')
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.5)
    cbar2.set_ticks(np.linspace(vmin, vmax, 9))
    ax2.grid(True, alpha=0.3)

    # Panel 3: C++ Jacobian (.ts file, C++)
    ax3 = axes[2]
    scatter3 = ax3.scatter(cpp_lons_valid, cpp_lats_valid, c=cpp_jac_valid,
                          s=1, cmap=cmap, norm=norm)
    ax3.set_xlabel('Longitude', fontsize=12)
    ax3.set_ylabel('Latitude', fontsize=12)
    ax3.set_title(f'TorchScript .ts C++ (∂tair/∂tref)\n{len(cpp_jac_valid)} points',
                  fontsize=14, fontweight='bold')
    ax3.set_aspect('equal', adjustable='box')
    cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.5)
    cbar3.set_ticks(np.linspace(vmin, vmax, 9))
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def plot_histograms(py_jac, ts_jac, cpp_jac, output_file='jacobian_histograms.png'):
    """Create 3-panel histogram comparison of Jacobian distributions."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Filter out NaN values
    py_valid = py_jac[~np.isnan(py_jac)]
    ts_valid = ts_jac[~np.isnan(ts_jac)]
    cpp_valid = cpp_jac[~np.isnan(cpp_jac)]

    # Compute common bin range for all histograms
    all_data = np.concatenate([py_valid, ts_valid, cpp_valid])
    vmin, vmax = np.percentile(all_data, [0.1, 99.9])
    bins = np.linspace(vmin, vmax, 50)

    # Panel 1: PyTorch histogram
    ax1 = axes[0]
    ax1.hist(py_valid, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(py_valid), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(py_valid):.4f}')
    ax1.axvline(np.median(py_valid), color='orange', linestyle='--', linewidth=2, label=f'Median = {np.median(py_valid):.4f}')
    ax1.set_xlabel('Jacobian Value', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'PyTorch .pt\nStd = {np.std(py_valid):.4f}', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: TorchScript histogram
    ax2 = axes[1]
    ax2.hist(ts_valid, bins=bins, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(ts_valid), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(ts_valid):.4f}')
    ax2.axvline(np.median(ts_valid), color='orange', linestyle='--', linewidth=2, label=f'Median = {np.median(ts_valid):.4f}')
    ax2.set_xlabel('Jacobian Value', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'TorchScript .ts (Python)\nStd = {np.std(ts_valid):.4f}', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: C++ histogram
    ax3 = axes[2]
    ax3.hist(cpp_valid, bins=bins, color='coral', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(cpp_valid), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(cpp_valid):.4f}')
    ax3.axvline(np.median(cpp_valid), color='orange', linestyle='--', linewidth=2, label=f'Median = {np.median(cpp_valid):.4f}')
    ax3.set_xlabel('Jacobian Value', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title(f'TorchScript .ts (C++)\nStd = {np.std(cpp_valid):.4f}', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nHistogram plot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare PyTorch and C++ TorchScript Jacobians'
    )
    parser.add_argument('--atm-file', required=True,
                        help='Atmosphere NetCDF file')
    parser.add_argument('--ocean-file', required=True,
                        help='Ocean/ice NetCDF file')
    parser.add_argument('--model', required=True,
                        help='PyTorch model checkpoint (.pt)')
    parser.add_argument('--torchscript', required=True,
                        help='TorchScript model (.ts)')
    parser.add_argument('--cpp-jacobian', required=True,
                        help='C++ Jacobian NetCDF file')
    parser.add_argument('--config', required=True,
                        help='Configuration YAML file')
    parser.add_argument('--output', default='jacobian_comparison.png',
                        help='Output plot filename')
    parser.add_argument('--output-hist', default='jacobian_histograms.png',
                        help='Output histogram filename')

    args = parser.parse_args()

    print("=" * 80)
    print("Jacobian Comparison: PyTorch (.pt) vs TorchScript (.ts) vs C++")
    print("=" * 80)

    # Load C++ Jacobian
    cpp_lons, cpp_lats, cpp_jac = load_cpp_jacobian(args.cpp_jacobian)

    # Compute PyTorch Jacobian from .pt checkpoint
    py_lons, py_lats, py_jac = compute_pytorch_jacobian(
        args.atm_file, args.ocean_file, args.model, args.config
    )

    # Compute TorchScript Jacobian from .ts file (in Python)
    ts_lons, ts_lats, ts_jac = compute_torchscript_jacobian(
        args.atm_file, args.ocean_file, args.torchscript, args.config
    )

    # Plot 3-panel spatial comparison
    plot_comparison(py_lons, py_lats, py_jac,
                    ts_lons, ts_lats, ts_jac,
                    cpp_lons, cpp_lats, cpp_jac,
                    args.output)

    # Plot 3-panel histogram comparison
    plot_histograms(py_jac, ts_jac, cpp_jac, args.output_hist)

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
