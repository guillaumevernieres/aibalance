#!/usr/bin/env python3
"""
UFS Emulator Inference and Jacobian Visualization Script

This script:
1. Reads NetCDF data from separate atmosphere and ocean/ice CF-1 files
2. Runs inference with trained UfsEmulatorFFNN model
3. Computes Jacobian matrices
4. Creates separate Arctic/Antarctic field plots
5. Shows input features, predictions, and Jacobian sensitivities

Usage:
    python plot_inference_jacobian.py --atm-file atm.nc --ocean-file ocean.nc \
        --model models/best_model.pt
    python plot_inference_jacobian.py --atm-file atm.nc --ocean-file ocean.nc \
        --model models/best_model.pt --arctic-only
    python plot_inference_jacobian.py --atm-file atm.nc --ocean-file ocean.nc \
        --model models/best_model.pt --antarctic-only
    python plot_inference_jacobian.py --atm-file atm.nc --ocean-file ocean.nc \
        --model models/best_model.pt --global-only --global-plot
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path to import ufsemulator modules
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# Cartopy for polar projections
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from ufsemulator.model import UfsEmulatorFFNN
from ufsemulator.data import IceDataPreparer


class UfsEmulatorInferencePlotter:
    """Handles NetCDF data processing, inference, and visualization."""

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """Initialize with model and configuration."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Initialize variables (will be set during model loading)
        self.input_variables = []
        self.output_variables = []
        self.input_units = {}
        self.output_units = {}

        # Variable mappings (same as in data.py)
        self.var_names = {
            "lat": "ULAT",
            "lon": "ULON",
            "aice": "aice_h",
            "tsfc": "Tsfc_h",
            "sst": "sst_h",
            "sss": "sss_h",
            "sice": "sice_h",
            "hi": "hi_h",
            "hs": "hs_h",
            "mask": "umask",
            "tair": "Tair_h",
            "frzmlt": "frzmlt_h",
            # Ice temperatures
            "sitempbot": "sitempbot_h",
            "sitempsnic": "sitempsnic_h",
            "sitemptop": "sitemptop_h",
            # Ocean/ice stresses
            "strocnx": "strocnx_h",
            "strocny": "strocny_h",
            # Atmosphere/ice stresses
            "strairx": "strairx_h",
            "strairy": "strairy_h",
            # Heat and flux variables
            "fhocn": "fhocn_h",
            "qref": "Qref_h",
            "flwup": "flwup_h",
            "fsens": "fsens_h",
            "flat": "flat_h",
            "flwdn": "flwdn_h",
            "fswdn": "fswdn_h",
        }

        # Alternative variable names for GDAS compatibility
        self.alt_var_names = {
            "aice_h": ["aice", "ice_concentration", "aicen"],
            "hi_h": ["hi", "ice_thickness", "hicen"],
            "hs_h": ["hs", "snow_thickness", "hsnon"],
            "Tair_h": ["tair", "air_temperature"],
            "Tsfc_h": ["tsfc", "surface_temperature"],
            "sst_h": ["sst", "sea_surface_temperature"],
            "sss_h": ["sss", "sea_surface_salinity"],
            "sice_h": ["sice", "ice_salinity"],
            "frzmlt_h": ["frzmlt", "frazil_melt", "frazil_ice_melt"],
            "ULAT": ["lat", "latitude"],
            "ULON": ["lon", "longitude"],
            "umask": ["mask", "land_mask"],
        }

        # Load model and extract configuration from checkpoint
        self.model, self.config = self._load_model_and_config(
            model_path, config_path
        )

        # Debug: Check if variables were properly set
        print(f"Input variables: {self.input_variables}")
        print(f"Output variables: {self.output_variables}")

    def _load_model_and_config(
        self, model_path: str, config_path: Optional[str] = None
    ) -> Tuple[UfsEmulatorFFNN, Dict]:
        """Load trained UfsEmulatorFFNN model and configuration."""
        print(f"Loading model from: {model_path}")

        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device,
                               weights_only=False)

        # Get configuration from checkpoint
        if "config" in checkpoint:
            config = checkpoint["config"]
            model_config = config["model"]
        else:
            print("No config found in checkpoint.")
            raise RuntimeError(
                f"Checkpoint '{model_path}' does not contain a 'config' entry. "
                "Please provide a checkpoint with configuration or supply a separate config."
            )
            model_config = config["model"]

        # Extract variable configuration
        self._setup_variables_from_config(config)

        # Create model with saved configuration
        model = UfsEmulatorFFNN(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            output_size=model_config["output_size"],
            hidden_layers=model_config.get("hidden_layers", 2)
        )

        # Load model weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        # Load normalization parameters
        norm_path = Path(model_path).parent / "normalization.pt"
        if not norm_path.exists():
            raise FileNotFoundError(
                f"Normalization file not found: {norm_path}\n"
                f"Please ensure normalization.pt exists in the model directory."
            )

        print(f"Loading normalization from: {norm_path}")
        moments = torch.load(norm_path, map_location=self.device, weights_only=False)

        # New dict format with all four normalization parameters
        model.input_mean.data = moments['input_mean']
        model.input_std.data = moments['input_std']
        model.output_mean.data = moments['output_mean']
        model.output_std.data = moments['output_std']

        print(f"✅ Loaded normalization (input_mean: {model.input_mean.mean().item():.4f}, "
              f"output_mean: {model.output_mean.mean().item():.4f})")

        print("✅ Model loaded successfully")
        return model, config

    def _setup_variables_from_config(self, config: Dict):
        """Set up input/output variables from model configuration."""
        variables_config = config.get('variables', {})

        # Get input variables from config
        self.input_variables = variables_config.get('input_variables', [
            'sst', 'sss', 'tair', 'tsfc', 'hi', 'hs', 'sice',
            'strocnx', 'strocny', 'strairx', 'strairy', 'qref',
            'flwdn', 'fswdn'
        ])

        # Get output variables from config
        self.output_variables = variables_config.get('output_variables',
                                                     ['aice'])

        # Define units for each variable
        self.input_units = {
            'sst': '°C', 'sss': 'psu', 'tair': '°C', 'tsfc': '°C',
            'hi': 'm', 'hs': 'm', 'sice': 'psu',
            'strocnx': 'N/m²', 'strocny': 'N/m²',
            'strairx': 'N/m²', 'strairy': 'N/m²',
            'qref': 'kg/kg', 'flwdn': 'W/m²', 'fswdn': 'W/m²',
            'fhocn': 'W/m²', 'fsens': 'W/m²', 'flat': 'W/m²'
        }

        self.output_units = {
            'aice': 'fraction', 'hi': 'm', 'hs': 'm', 'sice': 'psu'
        }

        print(f"Input variables ({len(self.input_variables)}): "
              f"{self.input_variables}")
        print(f"Output variables ({len(self.output_variables)}): "
              f"{self.output_variables}")

    def read_netcdf_data(self, filename: str) -> Dict[str, np.ndarray]:
        """Read NetCDF data with variable name fallback."""
        print(f"Reading NetCDF file: {filename}")

        with nc.Dataset(filename, "r") as dataset:
            data = {}

            for key, var_name in self.var_names.items():
                found_var = None

                # Try primary variable name
                if var_name in dataset.variables:
                    found_var = var_name
                else:
                    # Try alternative names
                    alt_names = self.alt_var_names.get(var_name, [])
                    for alt_name in alt_names:
                        if alt_name in dataset.variables:
                            found_var = alt_name
                            print(
                                f"Using alternative: "
                                f"{var_name} -> {alt_name}"
                            )
                            break

                if found_var:
                    var_data = dataset.variables[found_var][:]
                    # Handle time dimension if present
                    if var_data.ndim == 3:  # (time, lat, lon)
                        var_data = var_data[0]  # Take first time step
                    data[key] = var_data
                    print(f"Read {key}: shape {data[key].shape}")
                else:
                    # Handle missing variables gracefully for optional features
                    optional_vars = ["frzmlt", "sice", "hs", "sitempbot", "sitempsnic",
                                   "sitemptop", "strocnx", "strocny", "strairx", "strairy",
                                   "fhocn", "qref", "flwup", "fsens", "flat", "flwdn", "fswdn"]
                    if key in optional_vars:  # Optional features
                        print(f"⚠️ Optional variable {var_name} not found - will use zeros if needed")
                        # Don't add to data dict - will be handled in feature extraction
                    else:
                        # Required variables (coordinates, basic fields)
                        available_vars = list(dataset.variables.keys())[:10]
                        raise KeyError(
                            f"Required variable {var_name} not found. "
                            f"Available: {available_vars}"
                        )

        return data

    def filter_domain(
        self, data: Dict[str, np.ndarray], domain: str = "arctic",
        min_ice: Optional[float] = None, mask_mode: Optional[str] = None
    ) -> Tuple[np.ndarray, ...]:
        """Filter data for Arctic or Antarctic domain using training criteria."""
        # Get domain configuration from model config if not provided
        if min_ice is None:
            min_ice = self.config.get('domain', {}).get(
                'min_ice_concentration', 0.0)
        if mask_mode is None:
            mask_mode = self.config.get('domain', {}).get('mask_mode', 'both')
        lats = data["lat"]
        lons = data["lon"]
        mask = data["mask"]

        # Flatten arrays
        lats_flat = lats.flatten()
        lons_flat = lons.flatten()
        mask_flat = mask.flatten()

        # Get other variables flattened
        aice_flat = data["aice"].flatten()
        # Apply the same filtering used in training
        valid_mask = mask_flat == 1

        if mask_mode == "sea_ice":
            valid_mask = valid_mask & (aice_flat > min_ice)
        elif mask_mode == "ocean":
            valid_mask = valid_mask & (aice_flat < min_ice)
        # else "both" - just use the mask as-is

        selected_indices = np.where(valid_mask)[0]

        print(f"Selected {len(selected_indices)} points for {domain} domain "
              f"(min_ice={min_ice}, mask_mode={mask_mode})")

        if len(selected_indices) == 0:
            raise ValueError(f"No valid data points found for {domain} domain")

        # Create domain mask
        domain_mask = np.zeros(len(lats_flat), dtype=bool)
        domain_mask[selected_indices] = True

        # Extract features for valid points based on model requirements
        print("Extracting features for inference...")
        print(f"Model expects inputs: {self.input_variables}")
        print(f"Available data keys: {list(data.keys())}")

        features = []
        for var_name in self.input_variables:
            if var_name in data:
                var_data = data[var_name].flatten()[domain_mask]
                features.append(var_data)
                print(f"✅ Added input {var_name}, shape: {var_data.shape}")
            else:
                # Handle missing features by filling with zeros
                print(f"⚠️ Input {var_name} not found in data - "
                      f"filling with zeros")
                n_points = np.sum(domain_mask)
                zero_data = np.zeros(n_points)
                features.append(zero_data)
                print(f"✅ Added zero-filled input {var_name}, "
                      f"shape: {zero_data.shape}")

        if len(features) == 0:
            raise ValueError("No features were successfully extracted")

        features = np.column_stack(features)
        print(f"Final features shape: {features.shape}")

        # Extract target variables
        targets = []
        for var_name in self.output_variables:
            if var_name in data:
                target_data = data[var_name].flatten()[domain_mask]
                targets.append(target_data)
                print(f"✅ Added target {var_name}, shape: {target_data.shape}")
            else:
                print(f"⚠️ Target {var_name} not found - using zeros")
                n_points = np.sum(domain_mask)
                targets.append(np.zeros(n_points))

        if len(targets) == 1:
            targets = targets[0]  # Single output - keep as 1D
        else:
            targets = np.column_stack(targets)  # Multi-output - stack as 2D

        return (
            features,
            targets,
            lons_flat[domain_mask],
            lats_flat[domain_mask],
            domain_mask,
            lats.shape,
        )

    def run_inference(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run model inference and compute Jacobians."""
        print("Running inference...")

        # Convert to torch tensors
        features_tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            # Forward pass using predict() for physical-space output
            predictions = self.model.predict(features_tensor)
            predictions = predictions.cpu().numpy().flatten()

        print("Computing Jacobians...")
        jacobians = []

        # Compute Jacobian for each sample using jac_physical()
        for i in range(len(features)):
            sample = features_tensor[i:i + 1]  # Single sample

            # Compute physical-space Jacobian
            jac_phys = self.model.jac_physical(sample)  # [1, output_size, input_size]
            jac = jac_phys.cpu().numpy().flatten()
            jacobians.append(jac)

        jacobians = np.array(jacobians)

        print(f"✅ Inference complete: {len(predictions)} predictions")
        return predictions, jacobians

    def create_colormap(self, name: str):
        """Create custom colormaps for different variables."""
        if name in ["aice", "predictions"]:
            # Custom ice concentration colormap:
            # Green for open water (0-0.01), then white to blue for ice (0.01-1.0)
            from matplotlib.colors import ListedColormap
            import numpy as np

            # Create color segments
            n_total = 256
            n_open_water = int(0.01 * n_total)  # ~3 colors for 0-0.01 range
            n_ice = n_total - n_open_water      # Rest for 0.01-1.0 range

            # Open water colors (green shades)
            open_water_colors = plt.cm.Greens(np.linspace(0.3, 0.7, n_open_water))

            # Ice colors (white to blue)
            ice_colors = np.array([
                [1.0, 1.0, 1.0, 1.0],      # white
                [0.8, 0.9, 1.0, 1.0],      # very light blue
                [0.6, 0.8, 1.0, 1.0],      # light blue
                [0.4, 0.7, 1.0, 1.0],      # medium blue
                [0.2, 0.5, 0.9, 1.0],      # blue
                [0.0, 0.3, 0.8, 1.0],      # dark blue
                [0.0, 0.1, 0.6, 1.0],      # very dark blue
            ])
            ice_interp = np.zeros((n_ice, 4))
            for i in range(4):  # RGBA channels
                ice_interp[:, i] = np.interp(
                    np.linspace(0, len(ice_colors)-1, n_ice),
                    np.arange(len(ice_colors)),
                    ice_colors[:, i]
                )

            # Combine colors
            all_colors = np.vstack([open_water_colors, ice_interp])
            return ListedColormap(all_colors, name='ice_concentration')

        elif (
            "temp" in name.lower()
            or "sst" in name.lower()
            or "tsfc" in name.lower()
        ):
            # Temperature: blue to red
            return plt.cm.RdBu_r
        elif "jacobian" in name.lower():
            # Jacobian: centered around zero, red for positive, blue for negative
            return plt.cm.RdBu_r
        else:
            # Default - using NCAR colormap for generic fields
            return plt.cm.gist_ncar

    def _plot_ice_concentration(
        self,
        ax,
        lons: np.ndarray,
        lats: np.ndarray,
        ice_data: np.ndarray,
        targets: np.ndarray,
        title: str,
        extent: list,
        transform,
        colorbar_label: str = "Ice fraction",
        show_ice_edge: bool = True,
        use_data_range: bool = False,
        vmin: float = None,
        vmax: float = None
    ):
        """Helper function to plot ice concentration data with consistent styling."""
        # Set up map projection
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, alpha=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.3, color="lightgray")
        ax.gridlines(draw_labels=True, alpha=0.3)

        # Set color scale range
        if vmin is None or vmax is None:
            if use_data_range:
                vmin, vmax = np.min(ice_data), np.max(ice_data)
            else:
                vmin, vmax = 0, 1

        # Main scatter plot
        scatter = ax.scatter(
            lons,
            lats,
            c=ice_data,
            s=0.5,
            cmap=self.create_colormap("aice"),
            vmin=vmin,
            vmax=vmax,
            transform=transform,
        )

        # Add ice edge overlay if requested
        if show_ice_edge:
            ice_edge_mask = (targets >= 0.14) & (targets <= 0.16)
            if np.any(ice_edge_mask):
                ax.scatter(
                    lons[ice_edge_mask],
                    lats[ice_edge_mask],
                    c='red',
                    s=1.5,
                    alpha=0.7,
                    transform=transform,
                    label='Ice Edge (~0.15)'
                )
                ax.legend(loc='upper right', fontsize=8)

        # Add title as text box inside the plot instead of as axis title
        ax.text(0.02, 0.98, title, transform=ax.transAxes,
                fontsize=10, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.colorbar(scatter, ax=ax, shrink=0.3, label=colorbar_label)
        return scatter

    def _plot_generic_field(
        self,
        ax,
        lons: np.ndarray,
        lats: np.ndarray,
        field_data: np.ndarray,
        title: str,
        extent: list,
        transform,
        unit: str = "",
        vmin: float = None,
        vmax: float = None
    ):
        """Helper function to plot generic field data."""
        # Set up map projection
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, alpha=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.3, color="lightgray")
        ax.gridlines(draw_labels=True, alpha=0.3)

        # Choose appropriate colormap and scaling
        if vmin is None or vmax is None:
            vmin, vmax = - 0.5*np.std(field_data), 0.5*np.std(field_data)

        # Main scatter plot
        scatter = ax.scatter(
            lons, lats, c=field_data, s=0.5,
            cmap=self.create_colormap("generic"),
            vmin=vmin, vmax=vmax,
            transform=transform,
        )

        # Add title as text box
        ax.text(0.02, 0.98, title, transform=ax.transAxes,
                fontsize=10, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='white', alpha=0.8))

        colorbar_label = unit if unit else "Value"
        plt.colorbar(scatter, ax=ax, shrink=0.3, label=colorbar_label)
        return scatter

    def plot_domain_fields(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        predictions: np.ndarray,
        jacobians: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
        domain: str,
        output_dir: str = "plots",
        use_global_projection: bool = False,
        jacobian_vmin: Optional[float] = None,
        jacobian_vmax: Optional[float] = None,
    ):
        """Create separate figures: one for model outputs and one per Jacobian component."""

        Path(output_dir).mkdir(exist_ok=True)

        # Choose projection based on domain and plotting option
        if domain.lower() == "global" or use_global_projection:
            projection = ccrs.PlateCarree()
            extent = [-180, 180, -90, 90]  # Global extent
        elif domain.lower() == "arctic":
            projection = ccrs.NorthPolarStereo()
            extent = [-180, 180, 50, 90]
        else:  # antarctic
            projection = ccrs.SouthPolarStereo()
            extent = [-180, 180, -90, -50]
        transform = ccrs.PlateCarree()

        # Calculate RMSE and correlation for titles
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        corr = np.corrcoef(predictions, targets)[0, 1]

        # ===== FIGURE 1: MODEL OUTPUTS =====
        fig1 = plt.figure(figsize=(16, 8))

        # Determine what to plot based on output variables
        output_var = self.output_variables[0]  # Use first output for main plot
        output_unit = self.output_units.get(output_var, "")

        # For multi-output models, use the first output
        if predictions.ndim > 1:
            plot_predictions = predictions[:, 0]
            plot_targets = targets[:, 0]
        else:
            plot_predictions = predictions
            plot_targets = targets

        # Keep a copy of raw target stats before masking
        raw_tmin = float(np.nanmin(plot_targets))
        raw_tmax = float(np.nanmax(plot_targets))

        # Apply physical bounds for specific outputs (e.g., tair in Kelvin)
        if output_var == 'tair':
            tmin, tmax = 183.15, 333.15  # Kelvin
            plot_targets = np.where((plot_targets < tmin) | (plot_targets > tmax), np.nan, plot_targets)
            # Do NOT mask predictions; we want to see out-of-range predictions

        # Calculate common color scale based on observed data (ignore NaNs)
        obs_min, obs_max = np.nanmin(plot_targets), np.nanmax(plot_targets)
        # Debug print for target bounds
        n_valid = int(np.count_nonzero(~np.isnan(plot_targets)))
        n_nan = int(np.isnan(plot_targets).sum())
        print(f"[{domain}] Target (raw) range: [{raw_tmin:.3f}, {raw_tmax:.3f}] | Target (masked) range: [{obs_min:.3f}, {obs_max:.3f}] | valid={n_valid}, NaNs={n_nan}")

        # Extend range slightly to include predictions if they go beyond observed
        pred_min, pred_max = np.nanmin(plot_predictions), np.nanmax(plot_predictions)
        common_min = min(obs_min, pred_min)
        common_max = max(obs_max, pred_max)

        # Define target-based color scale (what the user wants to match)
        if output_var == 'aice':
            target_vmin, target_vmax = 0.0, 1.0
        else:
            target_vmin, target_vmax = obs_min, obs_max

        # For ice concentration, always use 0-1 range for consistency
        if output_var == 'aice':
            common_min, common_max = 0.0, 1.0

        # 1. Target values (left)
        ax1 = plt.subplot(1, 2, 1, projection=projection)

        if output_var == 'aice':
            self._plot_ice_concentration(
                ax1, lons, lats, plot_targets, plot_targets,
                (f"UFS {output_var.upper()}\n"
                 f"Range: [{obs_min:.3f}, {obs_max:.3f}]"),
                extent, transform, show_ice_edge=True,
                vmin=target_vmin, vmax=target_vmax
            )
        else:
            self._plot_generic_field(
                ax1, lons, lats, plot_targets,
                (f"UFS {output_var.upper()}\n"
                 f"Range: [{obs_min:.3f}, {obs_max:.3f}]"),
                extent, transform, output_unit,
                vmin=target_vmin, vmax=target_vmax
            )

        # 2. Predicted values (right)
        ax2 = plt.subplot(1, 2, 2, projection=projection)

        title = (f"Predicted {output_var.upper()}\n"
                 f"Range: [{pred_min:.3f}, {pred_max:.3f}]")

        # Force predicted color scale to match target color scale
        vmin_pred, vmax_pred = target_vmin, target_vmax

        if output_var == 'aice':
            self._plot_ice_concentration(
                ax2, lons, lats, plot_predictions, plot_targets,
                title, extent, transform, show_ice_edge=True,
                vmin=vmin_pred, vmax=vmax_pred
            )
        else:
            self._plot_generic_field(
                ax2, lons, lats, plot_predictions,
                title, extent, transform, output_unit,
                vmin=vmin_pred, vmax=vmax_pred
            )

        # Add title for output figure
        output_title = (
            f"UfsEmulator {output_var.upper()} - {domain.title()} Domain\n"
            f"RMSE: {rmse:.3f} | Pearson r: {corr:.3f}"
        )
        plt.figtext(0.5, 0.95, output_title,
                   fontsize=14, fontweight="bold",
                   horizontalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save output plot
        output_file = f"{output_dir}/ufs_emulator_{output_var}_{domain.lower()}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved {output_var} plot: {output_file}")
        plt.close(fig1)

        # ===== SEPARATE FIGURES FOR EACH JACOBIAN COMPONENT =====
        output_var = self.output_variables[0] if self.output_variables else "output"

        jacobian_figures = []
        for i, feature_name in enumerate(self.input_variables):
            # Skip self-sensitivity (output variable in inputs)
            if feature_name == output_var:
                continue

            if i >= jacobians.shape[1]:
                continue

            # Create individual figure for this Jacobian component
            fig_jac = plt.figure(figsize=(12, 10))
            ax = plt.subplot(1, 1, 1, projection=projection)

            sensitivity = jacobians[:, i]

            # Determine color scale range
            if jacobian_vmin is not None and jacobian_vmax is not None:
                # Use user-provided bounds
                vmin, vmax = jacobian_vmin, jacobian_vmax
                print(f"Using user-specified Jacobian bounds: [{vmin}, {vmax}]")
            else:
                # Auto-scale based on data
                std_val = np.std(sensitivity)
                min_val, max_val = np.min(sensitivity), np.max(sensitivity)

                # Use 95th percentile range for better visualization
                p5, p95 = np.percentile(sensitivity, [5, 95])

                # Choose appropriate range
                if std_val > 1e-8:  # If std is reasonable, use 2*std
                    vmin, vmax = 0, 2  #-0.5*std_val, 0.5*std_val
                elif max_val - min_val > 1e-8:  # If range is reasonable, use percentiles
                    vmin, vmax = p5, p95
                else:  # For very small values, use actual range
                    vmin, vmax = min_val, max_val

            # Set up map
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, alpha=0.5)
            ax.add_feature(cfeature.LAND, alpha=0.3, color="lightgray")
            ax.gridlines(draw_labels=True, alpha=0.3)

            # Debug info
            min_val, max_val = np.min(sensitivity), np.max(sensitivity)
            print(f"Jacobian d{output_var}/d{feature_name}: "
                  f"range=[{min_val:.6f}, {max_val:.6f}], "
                  f"colorbar=[{vmin:.6f}, {vmax:.6f}]")

            # Plot scatter
            #scatter = ax.scatter(
            #    lons, lats, c=sensitivity, s=1.5,
            #    cmap=self.create_colormap("jacobian"),
            #    transform=transform, vmin=vmin, vmax=vmax, alpha=0.7
            scatter = ax.scatter(
                lons, lats, c=sensitivity, s=1.5,
                cmap='RdBu_r',
                transform=transform, vmin=vmin, vmax=vmax, alpha=1
            )

            # Add title
            title_text = (f"d{output_var}/d{feature_name}\n"
                         f"Data range: [{min_val:.2e}, {max_val:.2e}]\n"
                         f"Colorbar: [{vmin:.2e}, {vmax:.2e}]")

            plt.title(title_text, fontsize=14, fontweight='bold', pad=20)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.05)
            cbar.set_label(f"d{output_var}/d{feature_name}", fontsize=12)

            plt.tight_layout()

            # Save individual Jacobian plot
            jac_output_file = (f"{output_dir}/jacobian_"
                              f"d{output_var}_d{feature_name}_{domain.lower()}.png")
            plt.savefig(jac_output_file, dpi=150, bbox_inches="tight")
            print(f"Saved Jacobian plot: {jac_output_file}")

            jacobian_figures.append(fig_jac)
            plt.close(fig_jac)

        return fig1, jacobian_figures

    def create_summary_statistics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        jacobians: np.ndarray,
        features: np.ndarray,
        domain: str,
    ):
        """Print simplified summary statistics."""
        print(f"\n📊 {domain.title()} Domain Statistics:")
        print("=" * 40)

        # Prediction metrics
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mae = np.mean(np.abs(predictions - targets))
        corr = np.corrcoef(predictions, targets)[0, 1]

        print(f"Prediction RMSE: {rmse:.4f}")
        print(f"Prediction MAE:  {mae:.4f}")
        print(f"Correlation:     {corr:.4f}")

        # Ice edge statistics
        ice_edge_mask = (targets >= 0.10) & (targets <= 0.16)
        n_ice_edge = np.sum(ice_edge_mask)
        total_points = len(targets)
        ice_edge_percentage = n_ice_edge / total_points * 100

        print(f"\nIce Edge Statistics (0.10-0.16 concentration):")
        print(f"  Points: {n_ice_edge}/{total_points} ({ice_edge_percentage:.1f}%)")

        if n_ice_edge > 0:
            ice_edge_pred = predictions[ice_edge_mask]
            ice_edge_targets = targets[ice_edge_mask]
            ice_edge_rmse = np.sqrt(np.mean((ice_edge_pred - ice_edge_targets) ** 2))
            ice_edge_bias = np.mean(ice_edge_pred - ice_edge_targets)

            print(f"  Ice Edge RMSE: {ice_edge_rmse:.4f}")
            print(f"  Ice Edge Bias: {ice_edge_bias:.4f} (pred - obs)")
            print(f"  Predicted range at ice edge: [{np.min(ice_edge_pred):.3f}, {np.max(ice_edge_pred):.3f}]")

        # Open water statistics
        open_water_mask = targets < 0.05
        n_open_water = np.sum(open_water_mask)
        open_water_percentage = n_open_water / total_points * 100

        print(f"\nOpen Water Statistics (<0.05 concentration):")
        print(f"  Points: {n_open_water}/{total_points} ({open_water_percentage:.1f}%)")

        if n_open_water > 0:
            open_water_pred = predictions[open_water_mask]
            open_water_targets = targets[open_water_mask]
            open_water_rmse = np.sqrt(np.mean((open_water_pred - open_water_targets) ** 2))
            open_water_bias = np.mean(open_water_pred - open_water_targets)

            print(f"  Open Water RMSE: {open_water_rmse:.4f}")
            print(f"  Open Water Bias: {open_water_bias:.4f} (pred - obs)")
            print(f"  Predicted range in open water: [{np.min(open_water_pred):.3f}, {np.max(open_water_pred):.3f}]")

        # Jacobian statistics for each input variable
        for i, var_name in enumerate(self.input_variables):
            jacobian_vals = jacobians[:, i]
            print(f"\n{var_name} Jacobian d(output)/d{var_name}:")
            print(f"  Range: [{np.min(jacobian_vals):.6f}, "
                  f"{np.max(jacobian_vals):.6f}]")
            print(f"  Mean: {np.mean(jacobian_vals):.6f}")
            print(f"  Std:  {np.std(jacobian_vals):.6f}")

            # Input statistics for this variable
            feature_vals = features[:, i]
            unit = self.input_units.get(var_name, "")
            print(f"\n{var_name} Input Statistics:")
            print(f"  Range: [{np.min(feature_vals):.2f}, "
                  f"{np.max(feature_vals):.2f}] {unit}")
            print(f"  Mean: {np.mean(feature_vals):.2f} {unit}")
            print(f"  Std:  {np.std(feature_vals):.2f} {unit}")

        # Variable sensitivity ranking
        jac_magnitudes = np.abs(jacobians)
        most_sensitive = np.argmax(jac_magnitudes, axis=1)

        print("\nMost sensitive input variables:")
        for i, var_name in enumerate(self.input_variables):
            count = np.sum(most_sensitive == i)
            percentage = count / len(most_sensitive) * 100
            if percentage > 1.0:  # Only show variables with >1% sensitivity
                print(f"  {var_name}: {percentage:.1f}% of points")


def thin_data(inputs, targets, lons, lats, fraction):
    if fraction < 1.0:
        n = len(targets)
        n_thin = int(n * fraction)
        idx = np.random.choice(n, n_thin, replace=False)
        return inputs[idx], targets[idx], lons[idx], lats[idx]
    return inputs, targets, lons, lats


def main():
    parser = argparse.ArgumentParser(
        description="UFS Emulator Inference and Jacobian Visualization"
    )
    parser.add_argument(
        "--atm-file", required=True, help="Atmosphere NetCDF file (CF-1)"
    )
    parser.add_argument(
        "--ocean-file", required=True, help="Ocean/Ice NetCDF file (CF-1)"
    )
    parser.add_argument(
        "--model", required=True, help="Trained model file (.pt)"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Config file"
    )
    parser.add_argument(
        "--output-dir", default="plots", help="Output directory"
    )
    parser.add_argument(
        "--arctic-only", action="store_true", help="Plot Arctic only"
    )
    parser.add_argument(
        "--antarctic-only", action="store_true", help="Plot Antarctic only"
    )
    parser.add_argument(
        "--global-only", action="store_true", help="Plot global domain only"
    )
    parser.add_argument('--thin-fraction', type=float, default=1.0,
                        help='Fraction of data to plot (e.g. 0.5 for 50 percent)')
    parser.add_argument(
        "--global-plot", action="store_true",
        help="Use global projection instead of polar stereographic"
    )
    parser.add_argument(
        "--jacobian-vmin", type=float, default=None,
        help="Minimum value for Jacobian colorbar (default: auto-scale)"
    )
    parser.add_argument(
        "--jacobian-vmax", type=float, default=None,
        help="Maximum value for Jacobian colorbar (default: auto-scale)"
    )

    args = parser.parse_args()

    # Initialize plotter
    plotter = UfsEmulatorInferencePlotter(args.model, args.config)

    # Read NetCDF data pair using the same logic as training
    preparer = IceDataPreparer(plotter.config)
    data = preparer.read_netcdf_data_pair(args.atm_file, args.ocean_file)

    # Determine domains to process
    domains = []
    if args.arctic_only:
        domains = ["arctic"]
    elif args.antarctic_only:
        domains = ["antarctic"]
    elif args.global_only or args.global_plot:
        domains = ["global"]
    else:
        domains = ["arctic", "antarctic"]

    # Process each domain
    for domain in domains:
        print(f"\n🌍 Processing {domain.title()} domain...")

        try:
            # Filter data for domain using same criteria as training
            # Parameters will be read from model configuration
            features, targets, lons, lats, mask, shape = plotter.filter_domain(
                data, domain
            )

            if len(features) == 0:
                print(f"⚠️ No valid data points found for {domain} domain")
                continue

            # Thin data BEFORE inference for efficiency
            if args.thin_fraction < 1.0:
                print(f"Thinning data to {args.thin_fraction:.1%} "
                      f"({int(len(features) * args.thin_fraction)} of "
                      f"{len(features)} points) before inference...")
                features, targets, lons, lats = thin_data(
                    features, targets, lons, lats, args.thin_fraction)

            # Run inference and compute Jacobians on thinned data
            predictions, jacobians = plotter.run_inference(features)

            # Create plots with configurable Jacobian bounds
            fig1, jac_figs = plotter.plot_domain_fields(
                features,
                targets,
                predictions,
                jacobians,
                lons,
                lats,
                domain,
                args.output_dir,
                args.global_plot,
                args.jacobian_vmin,
                args.jacobian_vmax,
            )

            # Print statistics
            plotter.create_summary_statistics(
                predictions, targets, jacobians, features, domain
            )

            # Figures already closed in plot_domain_fields

        except Exception as e:
            print(f"❌ Error processing {domain} domain: {e}")
            continue

    print(f"\n✅ Analysis complete! Check {args.output_dir}/ for plots")


if __name__ == "__main__":
    main()
