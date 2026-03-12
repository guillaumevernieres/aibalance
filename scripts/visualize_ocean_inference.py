#!/usr/bin/env python3
"""
Visualize ocean temperature emulator inference.

Shows input salt profile and depth, predicted temperature, and validation temperature.
Creates multiple visualization types:
1. Individual profile comparisons
2. Multi-sample overview plot
3. Scatter plots of predicted vs true temperature
4. Depth-binned error analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from pathlib import Path
import sys
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ufsemulator.model import UfsEmulatorFFNN


def load_model_and_data(model_file, data_file):
    """Load trained model and test data."""
    model_path = Path(model_file)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load normalization from same directory as model
    model_dir = model_path.parent
    norm_path = model_dir / "normalization.pt"
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalization file not found: {norm_path}")

    norm = torch.load(norm_path, weights_only=False)
    input_mean = norm['input_mean'].numpy()
    input_std = norm['input_std'].numpy()
    output_mean = norm['output_mean'].numpy()
    output_std = norm['output_std'].numpy()

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Get state dict
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # Detect if model uses conv1d
    has_conv1d = 'conv1d.weight' in state_dict
    use_conv1d = has_conv1d
    conv_channels = 32  # default
    conv_kernel_size = 3  # default

    if has_conv1d:
        # Infer conv parameters from state dict
        conv_channels = state_dict['conv1d.weight'].shape[0]
        conv_kernel_size = state_dict['conv1d.weight'].shape[2]
        print(f"Detected conv1d layer: {conv_channels} channels, kernel size {conv_kernel_size}")

    # Extract model config
    if 'config' in checkpoint:
        config = checkpoint['config']
        # Override with detected values if conv1d is present in state_dict
        model_config = {
            'input_size': config['model']['input_size'],
            'hidden_size': config['model']['hidden_size'],
            'output_size': config['model']['output_size'],
            'hidden_layers': config['model']['hidden_layers'],
            'activation': config['model'].get('activation', 'gelu'),
            'use_conv1d': use_conv1d,
            'conv_channels': conv_channels,
            'conv_kernel_size': conv_kernel_size,
        }
        model = UfsEmulatorFFNN(**model_config)
    else:
        # Try to infer from state dict
        if has_conv1d:
            # With conv1d, first FFNN layer input size = conv_channels * original_input_size
            # We need to infer original input size
            first_layer_weight = state_dict['network.0.weight']
            ffnn_input_size = first_layer_weight.shape[1]
            hidden_size = first_layer_weight.shape[0]
            # original_input_size = ffnn_input_size / conv_channels
            input_size = ffnn_input_size // conv_channels
        else:
            first_layer_weight = state_dict['network.0.weight']
            input_size = first_layer_weight.shape[1]
            hidden_size = first_layer_weight.shape[0]

        # Find output size
        last_layer_key = [k for k in state_dict.keys() if 'weight' in k][-1]
        output_size = state_dict[last_layer_key].shape[0]

        # Count layers
        hidden_layers = len([k for k in state_dict.keys() if '.weight' in k and 'network' in k]) - 1

        print(f"Inferred architecture: {input_size} → {hidden_size}×{hidden_layers} → {output_size}")
        if has_conv1d:
            print(f"  with conv1d: {conv_channels} channels, kernel {conv_kernel_size}")

        model = UfsEmulatorFFNN(
            input_size, hidden_size, output_size, hidden_layers,
            activation='gelu',
            use_conv1d=use_conv1d,
            conv_channels=conv_channels,
            conv_kernel_size=conv_kernel_size
        )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Load data
    print(f"Loading data from: {data_file}")
    data = np.load(data_file)
    X = data['inputs'].astype(np.float32)  # (N, 150) - Salt (75) + depth (75)
    y = data['targets'].astype(np.float32)  # (N, 75) - Temp

    print(f"Data shapes: X={X.shape}, y={y.shape}")
    return model, X, y, input_mean, input_std, output_mean, output_std


def load_num_levels_from_yaml(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    vcfg = cfg.get('variables', {})
    return int(vcfg.get('num_levels', 75))


def load_variable_names_from_yaml(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    vcfg = cfg.get('variables', {})
    output_variables = vcfg.get('output_variables')
    input_variables = vcfg.get('input_variables')
    missing = [k for k, v in [('output_variables', output_variables), ('input_variables', input_variables)] if v is None]
    if missing:
        raise ValueError(f"Missing required key(s) under 'variables' in config '{config_path}': {', '.join(missing)}")
    return output_variables[0], input_variables[0]


def plot_single_profile(input_profile, depth, output_pred, output_true, point_idx, output_file, input_variable, output_variable, units, jacobian=None, mean_jacobian=None):
    """Plot input and output profiles side by side for a single point."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    n_levels = len(input_profile)

    # Plot 1: Input profile
    ax = axes[0]
    ax.plot(input_profile, depth, 'b-o', linewidth=2, markersize=4, label=f'{input_variable} (input)')
    ax.set_xlabel(f'{input_variable} ({units[input_variable]})', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'Input: {input_variable} Profile\n(Point {point_idx})', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Depth increases downward
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 2: Output comparison
    ax = axes[1]
    ax.plot(output_true, depth, 'k-o', linewidth=2, markersize=4, label=f'True {output_variable}', alpha=0.7)
    ax.plot(output_pred, depth, 'r--s', linewidth=2, markersize=4, label=f'Predicted {output_variable}', alpha=0.7)
    ax.set_xlabel(f'{output_variable} ({units[output_variable]})', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'Output: {output_variable} Profile', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 3: Error profile
    ax = axes[2]
    error = output_pred - output_true
    ax.plot(error, depth, 'g-o', linewidth=2, markersize=4)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(f'{output_variable} Error ({units[output_variable]})', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title('Prediction Error', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Add statistics
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))

    stats_text = f'RMSE: {rmse:.3f}{units[output_variable]}\nMAE: {mae:.3f}{units[output_variable]}\nMax: {max_error:.3f}{units[output_variable]}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()


def plot_single_profile_jacobian(jacobian, mean_jacobian, depth, point_idx, output_file, input_variable, output_variable, units):
    """Plot Jacobian vertical structure for a single sample across multiple output levels."""
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    axes = axes.flatten()

    n_levels = len(depth)
    output_levels = [0, 10, 20, 25, 30, 35, 40, 45, 49]

    # Compute truncated SVD of this sample's Jacobian
    SVD_RTOL = 1e-6
    U_j, s_j, Vt_j = np.linalg.svd(jacobian, full_matrices=False)
    thresh_j = SVD_RTOL * s_j[0]
    rank_j = int(np.sum(s_j > thresh_j))
    s_trunc_j = np.where(s_j > thresh_j, s_j, 0.0)
    jacobian_trunc = U_j @ np.diag(s_trunc_j) @ Vt_j

    for i, level in enumerate(output_levels):
        if i >= len(axes):
            break
        if level >= n_levels:
            continue

        ax = axes[i]

        # Plot this sample's full sensitivity
        sensitivity = jacobian[level, :]  # Response of output[level] to all input
        ax.plot(sensitivity, depth, 'b-', linewidth=2.5, label=f'Full (Sample {point_idx})')

        # Plot truncated SVD sensitivity
        sensitivity_trunc = jacobian_trunc[level, :]
        ax.plot(sensitivity_trunc, depth, 'r--', linewidth=2, label=f'Truncated SVD (rank {rank_j})')

        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel(f'd{output_variable}/d{input_variable} ({units[output_variable]}/{units[input_variable]})', fontsize=11)
        ax.set_ylabel(f'{input_variable} Depth (m)', fontsize=11)
        ax.set_title(f'Sensitivity of {output_variable} at Level {level} (Depth ≈ {depth[level]:.0f}m)\n' +
                    f'to {input_variable} at all depths',
                    fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=8, loc='best')

        # Highlight diagonal (same level) with marker
        ax.plot(sensitivity[level], depth[level], 'bo', markersize=8, zorder=11)
        ax.plot(sensitivity_trunc[level], depth[level], 'rs', markersize=8, zorder=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved Jacobian plot: {output_file}")
    plt.close()


def plot_multi_sample_overview(samples_data, output_file, input_variable, output_variable, units):
    """Plot multiple samples on one figure for comparison."""
    n_samples = len(samples_data)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4*n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, (input_profile, depth, output_pred, output_true, idx) in enumerate(samples_data):
        # Salt profile
        ax = axes[i, 0]
        ax.plot(input_profile, depth, 'b-o', linewidth=1.5, markersize=3)
        ax.set_ylabel('Depth (m)', fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(f'Input: {input_variable}', fontsize=12, fontweight='bold')
        if i == n_samples - 1:
            ax.set_xlabel(f'{input_variable} ({units[input_variable]})', fontsize=10)
        ax.text(0.05, 0.95, f'Point {idx}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Temperature comparison
        ax = axes[i, 1]
        ax.plot(output_true, depth, 'k-o', linewidth=1.5, markersize=3, label='True', alpha=0.7)
        ax.plot(output_pred, depth, 'r--s', linewidth=1.5, markersize=3, label='Pred', alpha=0.7)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(f'Output: {output_variable}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
        if i == n_samples - 1:
            ax.set_xlabel(f'{output_variable} ({units[output_variable]})', fontsize=10)

        # Error profile
        ax = axes[i, 2]
        error = output_pred - output_true
        ax.plot(error, depth, 'g-o', linewidth=1.5, markersize=3)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title('Error', fontsize=12, fontweight='bold')
        if i == n_samples - 1:
            ax.set_xlabel(f'Error ({units[output_variable]})', fontsize=10)

        # Add RMSE
        rmse = np.sqrt(np.mean(error**2))
        ax.text(0.95, 0.95, f'RMSE: {rmse:.3f}{units[output_variable]}', transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved multi-sample plot: {output_file}")
    plt.close()


def plot_scatter_analysis(y_true, y_pred, output_file, output_variable, units):
    """Create scatter plots of predicted vs true temperature."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Flatten arrays for scatter plot
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Plot 1: Full scatter plot
    ax = axes[0]

    # Create hexbin for better visualization with many points
    hexbin = ax.hexbin(y_true_flat, y_pred_flat, gridsize=50, cmap='viridis',
                       mincnt=1, bins='log')

    # Add 1:1 line
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')

    ax.set_xlabel(f'True {output_variable} ({units[output_variable]})', fontsize=12)
    ax.set_ylabel(f'Predicted {output_variable} ({units[output_variable]})', fontsize=12)
    ax.set_title(f'Predicted vs True {output_variable}\n(All Depths)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Add colorbar
    cbar = plt.colorbar(hexbin, ax=ax)
    cbar.set_label('Count (log scale)', fontsize=10)

    # Calculate statistics
    errors = y_pred_flat - y_true_flat
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    r2 = 1 - np.sum(errors**2) / np.sum((y_true_flat - np.mean(y_true_flat))**2)
    bias = np.mean(errors)

    stats_text = (f'RMSE: {rmse:.4f}{units[output_variable]}\n'
                  f'MAE: {mae:.4f}{units[output_variable]}\n'
                  f'Bias: {bias:.4f}{units[output_variable]}\n'
                  f'R²: {r2:.4f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 2: Error distribution
    ax = axes[1]
    ax.hist(errors, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax.axvline(bias, color='g', linestyle='--', linewidth=2, label=f'Bias: {bias:.4f}{units[output_variable]}')
    ax.set_xlabel(f'Prediction Error ({units[output_variable]})', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentiles
    p5, p95 = np.percentile(errors, [5, 95])
    ax.axvline(p5, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(p95, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(0.98, 0.95, f'5-95%: [{p5:.3f}, {p95:.3f}]{units[output_variable]}',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved scatter analysis: {output_file}")
    plt.close()


def plot_depth_binned_errors(y_true, y_pred, depth_values, output_file, n_bins, output_variable, units):
    """Analyze errors binned by depth."""
    n_samples, n_levels = y_true.shape

    # Use first sample's depth values (assuming all samples have same depth grid)
    depth_grid = depth_values[0]  # (75,)

    # Create depth bins
    depth_edges = np.percentile(depth_grid, np.linspace(0, 100, n_bins+1))
    depth_centers = (depth_edges[:-1] + depth_edges[1:]) / 2

    # Initialize arrays for statistics
    rmse_by_depth = np.zeros(n_bins)
    mae_by_depth = np.zeros(n_bins)
    bias_by_depth = np.zeros(n_bins)
    std_by_depth = np.zeros(n_bins)
    n_points_by_depth = np.zeros(n_bins)

    # Compute errors
    errors = y_pred - y_true  # (n_samples, n_levels)

    # Bin errors by depth
    for i in range(n_bins):
        # Find levels in this depth bin
        mask = (depth_grid >= depth_edges[i]) & (depth_grid < depth_edges[i+1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (depth_grid >= depth_edges[i]) & (depth_grid <= depth_edges[i+1])

        if np.any(mask):
            # Extract errors for this depth bin (all samples, selected levels)
            errors_bin = errors[:, mask].flatten()

            rmse_by_depth[i] = np.sqrt(np.mean(errors_bin**2))
            mae_by_depth[i] = np.mean(np.abs(errors_bin))
            bias_by_depth[i] = np.mean(errors_bin)
            std_by_depth[i] = np.std(errors_bin)
            n_points_by_depth[i] = len(errors_bin)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: RMSE by depth
    ax = axes[0, 0]
    ax.plot(rmse_by_depth, depth_centers, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel(f'RMSE ({units[output_variable]})', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title('RMSE by Depth', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Plot 2: MAE by depth
    ax = axes[0, 1]
    ax.plot(mae_by_depth, depth_centers, 'g-o', linewidth=2, markersize=6)
    ax.set_xlabel(f'MAE ({units[output_variable]})', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title('MAE by Depth', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Plot 3: Bias by depth
    ax = axes[1, 0]
    ax.plot(bias_by_depth, depth_centers, 'r-o', linewidth=2, markersize=6)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(f'Bias ({units[output_variable]})', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title('Bias by Depth', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Add shading for bias direction
    ax.fill_betweenx(depth_centers, 0, bias_by_depth,
                     where=(bias_by_depth > 0), alpha=0.2, color='red', label='Positive bias')
    ax.fill_betweenx(depth_centers, 0, bias_by_depth,
                     where=(bias_by_depth < 0), alpha=0.2, color='blue', label='Negative bias')
    ax.legend(fontsize=9)

    # Plot 4: Error std by depth
    ax = axes[1, 1]
    ax.plot(std_by_depth, depth_centers, 'm-o', linewidth=2, markersize=6)
    ax.set_xlabel(f'Error Std Dev ({units[output_variable]})', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title('Error Variability by Depth', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved depth-binned analysis: {output_file}")
    plt.close()

    # Print summary
    print("\nDepth-Binned Error Summary:")
    print(f"{'Depth Range (m)':<20} {'RMSE':<12} {'MAE':<12} {'Bias':<12} {'N Points':<10}")
    print("-" * 66)
    for i in range(n_bins):
        depth_range = f"{depth_edges[i]:.0f}-{depth_edges[i+1]:.0f}"
        print(f"{depth_range:<20} {rmse_by_depth[i]:<12.4f} {mae_by_depth[i]:<12.4f} "
              f"{bias_by_depth[i]:<12.4f} {int(n_points_by_depth[i]):<10}")


def compute_jacobian(model, X, input_mean, input_std, output_mean, output_std, n_levels=75, output_variable='Temp', input_variable='Salt'):
    """
    Compute Jacobian for a batch of samples.
    The input to the model is always [variable_profile (n_levels), depth (n_levels)].
    The variable_profile is the input_variable (e.g., Salt for Temp emulator, or Temp for Salt emulator).

    output_variable: 'Temp' or 'Salt' (what the emulator predicts)
    input_variable: 'Salt' or 'Temp' (the first n_levels of input, what we're analyzing sensitivity to)
    Returns:
        jacobian: (n_samples, n_levels, n_levels) - dOutput/dInput
    """
    X_norm = (X - input_mean) / input_std
    X_tensor = torch.from_numpy(X_norm).float()
    X_tensor.requires_grad_(True)
    y_norm = model(X_tensor)
    n_samples = X_tensor.shape[0]
    jacobians = []
    print(f"Computing Jacobian for {n_samples} samples...")
    for i in range(n_samples):
        if i % 100 == 0:
            print(f"  Sample {i}/{n_samples}")
        jac = torch.autograd.functional.jacobian(
            lambda x: model(x.unsqueeze(0)).squeeze(0),
            X_tensor[i]
        )  # Shape: (n_outputs, n_inputs)

        # The Jacobian of output w.r.t. the input variable profile (first n_levels of input)
        # Output is always n_levels, input variable profile is first n_levels
        jac_block = jac[:n_levels, :n_levels].detach().numpy()  # (n_levels, n_levels)

        # Account for normalization: convert from normalized to physical units
        # d(output_physical)/d(input_physical) = (output_std / input_std) * d(output_norm)/d(input_norm)
        out_std = output_std  # shape (n_levels,)
        in_std = input_std[:n_levels]  # The variable profile std, shape (n_levels,)
        jac_physical = (out_std[:, None] / in_std[None, :]) * jac_block
        jacobians.append(jac_physical)
    return np.array(jacobians)


def plot_jacobian_analysis(jacobians, depth_values, output_dir, output_variable='Temp', input_variable='Salt', units=None):
    """
    Create multiple plots analyzing the Jacobian dOutput/dInput.
    Shows both mean behavior and individual location examples.

    Args:
        jacobians: (n_samples, 75, 75) - dOutput_i/dInput_j for each sample
        depth_values: (n_samples, 75) - depth at each level
        output_dir: Path object for output directory
        output_variable: str, e.g. 'Temp' or 'Salt'
        input_variable: str, e.g. 'Salt' or 'Temp'
        units: dict, e.g. {'Temp': '°C', 'Salt': 'psu'}
    """
    if units is None:
        units = {'Temp': '°C', 'Salt': 'psu'}
    n_samples = jacobians.shape[0]
    depth_grid = depth_values[0]  # Assuming same depth grid for all
    n_levels = len(depth_grid)

    # Select random examples for individual plots
    num_examples = min(8, n_samples)
    example_indices = np.random.choice(n_samples, num_examples, replace=False)

    # Compute mean Jacobian
    mean_jac = np.mean(jacobians, axis=0)  # (n_levels, n_levels)

    # ── Truncated SVD of mean Jacobian ───────────────────────────────
    SVD_RTOL = 1e-1  # singular values < rtol * σ_max are discarded
    U, s, Vt = np.linalg.svd(mean_jac, full_matrices=False)
    thresh = SVD_RTOL * s[0]
    eff_rank = 2  #int(np.sum(s > thresh))
    s_trunc = np.where(s > thresh, s, 0.0)
    mean_jac_trunc = U @ np.diag(s_trunc) @ Vt
    cond_full = s[0] / s[-1] if s[-1] > 1e-15 else float("inf")
    cond_trunc = s[0] / s[eff_rank - 1] if eff_rank > 0 else float("inf")

    print(f"  Selected {num_examples} example locations: {example_indices}")
    print(f"  Mean Jacobian SVD: rank={eff_rank}/{n_levels}, "
          f"cond(full)={cond_full:.2e}, cond(trunc)={cond_trunc:.2f}, rtol={SVD_RTOL}")

    # 1. Mean Jacobian heatmap: full, truncated SVD, diagonals, singular values
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))

    # Use shared colorscale for both heatmaps
    vmax_mean = max(np.abs(mean_jac).max(), np.abs(mean_jac_trunc).max())

    # Plot 1a: Full Jacobian matrix
    ax = axes[0, 0]
    extent = [depth_grid[0], depth_grid[-1], depth_grid[0], depth_grid[-1]]
    im = ax.imshow(mean_jac, aspect='auto', cmap='RdBu_r', origin='lower',
                   extent=extent, vmin=-vmax_mean, vmax=vmax_mean)
    ax.set_xlabel(f'{input_variable} Depth (m)', fontsize=12)
    ax.set_ylabel(f'{output_variable} Depth (m)', fontsize=12)
    ax.set_title(f'Mean Jacobian (Full)\nd{output_variable}/d{input_variable}, rank={n_levels}',
                 fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'd{output_variable}/d{input_variable} ({units[output_variable]}/{units[input_variable]})', fontsize=10)
    depth_interval = 500
    for d in np.arange(0, depth_grid[-1], depth_interval):
        ax.axhline(d, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvline(d, color='gray', linewidth=0.5, alpha=0.3)

    # Plot 1b: Truncated SVD Jacobian
    ax = axes[0, 1]
    im = ax.imshow(mean_jac_trunc, aspect='auto', cmap='RdBu_r', origin='lower',
                   extent=extent, vmin=-vmax_mean, vmax=vmax_mean)
    ax.set_xlabel(f'{input_variable} Depth (m)', fontsize=12)
    ax.set_ylabel(f'{output_variable} Depth (m)', fontsize=12)
    ax.set_title(f'Mean Jacobian (Truncated SVD)\nEffective rank={eff_rank}/{n_levels}, rtol={SVD_RTOL}',
                 fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'd{output_variable}/d{input_variable} ({units[output_variable]}/{units[input_variable]})', fontsize=10)
    for d in np.arange(0, depth_grid[-1], depth_interval):
        ax.axhline(d, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvline(d, color='gray', linewidth=0.5, alpha=0.3)

    # Plot 1c: Diagonal comparison (full vs truncated)
    ax = axes[1, 0]
    diag_full = np.diagonal(mean_jac)
    diag_trunc = np.diagonal(mean_jac_trunc)
    ax.plot(diag_full, depth_grid, 'b-o', linewidth=2, markersize=4, label='Full')
    ax.plot(diag_trunc, depth_grid, 'r--s', linewidth=2, markersize=4, label=f'Truncated (rank {eff_rank})')
    ax.set_xlabel(f'd{output_variable}/d{input_variable} ({units[output_variable]}/{units[input_variable]})', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'Local Sensitivity (Diagonal)\nFull vs Truncated SVD', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 1d: Singular value spectrum
    ax = axes[1, 1]
    ax.semilogy(range(1, len(s) + 1), s, 'ko-', lw=2, ms=5)
    ax.axhline(thresh, color='r', ls='--', lw=1.5, alpha=0.7, label=f'Threshold = {thresh:.2e}')
    retained = np.arange(1, len(s) + 1)[s > thresh]
    discarded = np.arange(1, len(s) + 1)[s <= thresh]
    if len(retained) > 0:
        ax.semilogy(retained, s[s > thresh], 'go', ms=7, zorder=5, label=f'Retained ({eff_rank})')
    if len(discarded) > 0:
        ax.semilogy(discarded, s[s <= thresh], 'rx', ms=7, zorder=5, label=f'Discarded ({n_levels - eff_rank})')
    ax.set_xlabel('Singular Value Index', fontsize=12)
    ax.set_ylabel('σ', fontsize=12)
    ax.set_title(f'Singular Value Spectrum\ncond(full)={cond_full:.2e}, cond(trunc)={cond_trunc:.2f}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'jacobian_mean_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'jacobian_mean_heatmap.png'}")
    plt.close()

    # 2. Sample-specific Jacobians (show individual examples)
    #    Top 2 rows: full Jacobian, bottom 2 rows: truncated SVD
    fig, axes = plt.subplots(4, 4, figsize=(28, 24))

    vmax = np.percentile(np.abs(jacobians), 99)
    extent = [depth_grid[0], depth_grid[-1], depth_grid[0], depth_grid[-1]]

    for i, idx in enumerate(example_indices):
        # ── Full Jacobian (top 2 rows) ──
        row_full = i // 4
        col = i % 4
        ax = axes[row_full, col]
        im = ax.imshow(jacobians[idx], aspect='auto', cmap='RdBu_r',
                      origin='lower', vmin=-vmax, vmax=vmax, extent=extent)
        ax.set_xlabel(f'{input_variable} Depth (m)', fontsize=10)
        ax.set_ylabel(f'{output_variable} Depth (m)', fontsize=10)
        ax.set_title(f'Sample {idx}: Full Jacobian', fontsize=11, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'd{output_variable}/d{input_variable}', fontsize=8)

        # ── Truncated SVD Jacobian (bottom 2 rows) ──
        jac_i = jacobians[idx]
        U_i, s_i, Vt_i = np.linalg.svd(jac_i, full_matrices=False)
        thresh_i = SVD_RTOL * s_i[0]
        rank_i = int(np.sum(s_i > thresh_i))
        s_trunc_i = np.where(s_i > thresh_i, s_i, 0.0)
        jac_trunc_i = U_i @ np.diag(s_trunc_i) @ Vt_i

        row_trunc = 2 + i // 4
        ax = axes[row_trunc, col]
        im = ax.imshow(jac_trunc_i, aspect='auto', cmap='RdBu_r',
                      origin='lower', vmin=-vmax, vmax=vmax, extent=extent)
        ax.set_xlabel(f'{input_variable} Depth (m)', fontsize=10)
        ax.set_ylabel(f'{output_variable} Depth (m)', fontsize=10)
        ax.set_title(f'Sample {idx}: Truncated SVD (rank {rank_i})', fontsize=11, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'd{output_variable}/d{input_variable}', fontsize=8)

    # Add row labels
    fig.text(0.02, 0.75, 'Full Jacobian', va='center', ha='center',
             fontsize=16, fontweight='bold', rotation=90)
    fig.text(0.02, 0.25, f'Truncated SVD (rtol={SVD_RTOL})', va='center', ha='center',
             fontsize=16, fontweight='bold', rotation=90)

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.savefig(output_dir / 'jacobian_sample_examples.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'jacobian_sample_examples.png'}")
    plt.close()

    # 3. Statistics: Frobenius norm, spectral properties
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Define colors for individual examples
    colors = plt.cm.tab10(np.linspace(0, 1, num_examples))

    frobenius_norms = np.array([np.linalg.norm(jac, 'fro') for jac in jacobians])
    spectral_norms = np.array([np.linalg.norm(jac, 2) for jac in jacobians])

    ax = axes[0, 0]
    ax.hist(frobenius_norms, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Frobenius Norm', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Jacobian Frobenius Norm Distribution', fontsize=12, fontweight='bold')
    ax.axvline(np.mean(frobenius_norms), color='r', linestyle='--',
              linewidth=2, label=f'Mean: {np.mean(frobenius_norms):.2f}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[0, 1]
    ax.hist(spectral_norms, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Spectral Norm (σ_max)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Jacobian Spectral Norm Distribution', fontsize=12, fontweight='bold')
    ax.axvline(np.mean(spectral_norms), color='r', linestyle='--',
              linewidth=2, label=f'Mean: {np.mean(spectral_norms):.2f}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 0]
    offdiag_strength = []
    for jac in jacobians:
        diag_vals = np.abs(np.diagonal(jac))
        offdiag_vals = np.abs(jac - np.diag(np.diagonal(jac)))
        offdiag_strength.append(np.mean(offdiag_vals) / (np.mean(diag_vals) + 1e-10))

    ax.hist(offdiag_strength, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel('Off-diagonal / Diagonal Ratio', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Vertical Coupling Strength\n(Higher = more non-local effects)',
                fontsize=12, fontweight='bold')
    ax.axvline(np.mean(offdiag_strength), color='r', linestyle='--',
              linewidth=2, label=f'Mean: {np.mean(offdiag_strength):.3f}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    for j, idx in enumerate(example_indices):
        diag_individual = np.abs(np.diagonal(jacobians[idx]))
        ax.plot(diag_individual, depth_grid, color=colors[j], alpha=0.5,
               linewidth=1.5, label=f'Sample {idx}')

    diag_mean = np.abs(np.diagonal(mean_jac))
    diag_std = np.std([np.abs(np.diagonal(jac)) for jac in jacobians], axis=0)
    ax.plot(diag_mean, depth_grid, 'k-', linewidth=3, label='Mean', zorder=10)
    ax.fill_betweenx(depth_grid, diag_mean - diag_std, diag_mean + diag_std,
                     alpha=0.2, color='gray', label='±1 std')

    ax.set_xlabel(f'|d{output_variable}/d{input_variable}| ({units[output_variable]}/{units[input_variable]})', fontsize=11)
    ax.set_ylabel('Depth (m)', fontsize=11)
    ax.set_title(f'Local Sensitivity of {output_variable} to {input_variable}\n(Diagonal elements)',
                fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig(output_dir / 'jacobian_statistics.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'jacobian_statistics.png'}")
    plt.close()

    print("\nJacobian Statistics:")
    print(f"  Mean Frobenius norm: {np.mean(frobenius_norms):.4f} ± {np.std(frobenius_norms):.4f}")
    print(f"  Mean Spectral norm:  {np.mean(spectral_norms):.4f} ± {np.std(spectral_norms):.4f}")
    print(f"  Mean off-diag/diag ratio: {np.mean(offdiag_strength):.4f} ± {np.std(offdiag_strength):.4f}")
    print(f"  Mean diagonal sensitivity: {np.mean(diag_mean):.4f} ± {np.std(diag_mean):.4f} {units[output_variable]}/{units[input_variable]}")


def main():
    units = {'Temp': '°C', 'Salt': 'psu'}
    parser = argparse.ArgumentParser(description='Visualize ocean emulator inference')
    parser.add_argument('--model-file', type=str,
                        help='Path to trained model file (.pt)')
    parser.add_argument('--data-file', type=str,
                        help='NPZ data file')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of random samples to plot')
    parser.add_argument('--output-dir', type=str, default='inference_plots',
                        help='Output directory for plots')
    parser.add_argument('--max-points', type=int, default=50000,
                        help='Maximum number of points to use for analysis (for speed)')
    parser.add_argument('--depth-bins', type=int, default=15,
                        help='Number of depth bins for error analysis')
    parser.add_argument('--num-jacobian-samples', type=int, default=100,
                        help='Number of samples to use for Jacobian analysis')
    parser.add_argument('--skip-jacobian', action='store_true',
                        help='Skip Jacobian computation (can be slow)')
    parser.add_argument('--config-file', type=str, default='config_ocntemp.yaml',
                        help='YAML config file to read num_levels from')
    args = parser.parse_args()

    # Read num_levels and variable names from config
    num_levels = load_num_levels_from_yaml(args.config_file)
    output_variable, input_variable = load_variable_names_from_yaml(args.config_file)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model and data
    model, X, y, input_mean, input_std, output_mean, output_std = load_model_and_data(
        args.model_file, args.data_file
    )

    # Subsample if dataset is very large
    if len(X) > args.max_points:
        print(f"Subsampling {args.max_points} points from {len(X)} for faster analysis...")
        indices = np.random.choice(len(X), args.max_points, replace=False)
        X = X[indices]
        y = y[indices]

    n_levels = num_levels

    # Run inference on all data
    print("\nRunning inference on all data...")
    X_tensor = torch.from_numpy((X - input_mean) / input_std).float()

    with torch.no_grad():
        y_pred_norm = model(X_tensor).numpy()

    y_pred = y_pred_norm * output_std + output_mean

    # Extract depth values (second half of input features)
    depth_values = X[:, n_levels:2*n_levels]

    # Compute overall statistics
    errors = y_pred - y
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    r2 = 1 - np.sum(errors**2) / np.sum((y - np.mean(y))**2)
    bias = np.mean(errors)

    print(f"\nOverall Validation Statistics ({len(X)} points, {n_levels} levels):")
    print(f"  RMSE: {rmse:.4f}°C")
    print(f"  MAE:  {mae:.4f}°C")
    print(f"  Bias: {bias:.4f}°C")
    print(f"  R²:   {r2:.4f}")

    # 1. Individual profile plots
    n_samples = min(args.num_samples, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)

    print(f"\nGenerating {n_samples} individual profile plots...")
    samples_data = []

    # Compute Jacobians for profile samples if not skipping
    profile_jacobians = None
    mean_jacobian = None
    if not args.skip_jacobian:
        print(f"\nComputing Jacobians for {n_samples} profile samples...")
        X_profile = X[indices]
        profile_jacobians = compute_jacobian(
            model, X_profile, input_mean, input_std, output_mean, output_std, num_levels,
            output_variable=output_variable, input_variable=input_variable
        )
        # Also compute mean Jacobian from larger sample for reference
        print(f"\nComputing mean Jacobian from {args.num_jacobian_samples} samples for reference...")
        n_jac_samples = min(args.num_jacobian_samples, len(X))
        jac_indices = np.random.choice(len(X), n_jac_samples, replace=False)
        X_jac = X[jac_indices]
        jacobians_all = compute_jacobian(
            model, X_jac, input_mean, input_std, output_mean, output_std, num_levels,
            output_variable=output_variable, input_variable=input_variable
        )
        mean_jacobian = np.mean(jacobians_all, axis=0)

    for i, idx in enumerate(indices):
        # Get input and true output
        x_raw = X[idx]
        y_true_i = y[idx]
        y_pred_i = y_pred[idx]

        # Input structure is always [variable_profile (n_levels), depth (n_levels)]
        # The variable_profile is the input_variable (Salt for Temp emulator, Temp for Salt emulator)
        input_profile = x_raw[:n_levels]
        depth = x_raw[n_levels:2*n_levels]

        samples_data.append((input_profile, depth, y_pred_i, y_true_i, idx))

        # Plot individual profile
        output_file = output_dir / f"profile_sample_{i+1:02d}_idx_{idx}.png"
        plot_single_profile(input_profile, depth, y_pred_i, y_true_i, idx, output_file, input_variable, output_variable, units)

        # Plot Jacobian vertical structure for this sample
        if profile_jacobians is not None:
            output_file_jac = output_dir / f"jacobian_sample_{i+1:02d}_idx_{idx}.png"
            plot_single_profile_jacobian(profile_jacobians[i], mean_jacobian, depth, idx, output_file_jac,
                                        input_variable, output_variable, units)

        print(f"  Sample {i+1}/{n_samples}: Point {idx}")

    # 2. Multi-sample overview plot
    print("\nGenerating multi-sample overview plot...")
    output_file = output_dir / "profile_overview_multi_samples.png"
    plot_multi_sample_overview(samples_data, output_file, input_variable, output_variable, units)

    # 3. Scatter plots
    print("\nGenerating scatter analysis...")
    output_file = output_dir / "scatter_predicted_vs_true.png"
    plot_scatter_analysis(y, y_pred, output_file, output_variable, units)

    # 4. Depth-binned error analysis
    print("\nGenerating depth-binned error analysis...")
    output_file = output_dir / "errors_by_depth.png"
    plot_depth_binned_errors(y, y_pred, depth_values, output_file, n_bins=args.depth_bins, output_variable=output_variable, units=units)

    # 5. Jacobian analysis (summary plots)
    if not args.skip_jacobian:
        print("\n" + "="*60)
        print(f"Jacobian Analysis: Summary Plots")
        print("="*60)
        # Use the already computed jacobians_all if available
        if 'jacobians_all' not in locals():
            n_jac_samples = min(args.num_jacobian_samples, len(X))
            jac_indices = np.random.choice(len(X), n_jac_samples, replace=False)
            X_jac = X[jac_indices]
            depth_jac = depth_values[jac_indices]
            print(f"Computing Jacobian for {n_jac_samples} samples (this may take a few minutes)...")
            jacobians_all = compute_jacobian(
                model, X_jac, input_mean, input_std, output_mean, output_std, num_levels,
                output_variable=output_variable, input_variable=input_variable
            )
        else:
            depth_jac = depth_values[jac_indices]

        print("\nGenerating Jacobian summary plots...")
        plot_jacobian_analysis(jacobians_all, depth_jac, output_dir,
            output_variable=output_variable, input_variable=input_variable, units=units)
    else:
        print("\nSkipping Jacobian analysis (use --skip-jacobian to disable)")

    print(f"\n{'='*60}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
