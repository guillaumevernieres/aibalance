#!/usr/bin/env python3
"""
Prepare ocean training dataset from NetCDF files using config YAML.
Creates a .npz file for use in training and validation.
"""
import sys
from pathlib import Path
# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
import yaml
from ufsemulator.data import IceDataPreparer

def main():
    parser = argparse.ArgumentParser(description="Prepare ocean training data (.npz) from NetCDF and config YAML")
    parser.add_argument('--config', type=str, required=True, help='YAML config file (e.g., config_ocntemp.yaml)')
    parser.add_argument('--output', type=str, required=False, help='Output .npz file (overrides config)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Determine output file
    output_file = args.output
    if not output_file:
        # Use config-specified output or default
        model_dir = config.get('output', {}).get('model_dir', 'models_ocntemp/')
        output_file = model_dir.rstrip('/') + '/training_data.npz'

    # Prepare data
    preparer = IceDataPreparer(config)
    data_config = config.get('data', {})
    atm_file = data_config.get('atm_file')
    ocn_file = data_config.get('ocean_file')
    max_patterns = data_config.get('max_patterns', 400000)
    thin_fraction = data_config.get('thin_fraction', 1.0)

    print(f"Preparing training data from: {ocn_file}")
    if atm_file:
        print(f"  (Atmosphere file: {atm_file})")
    print(f"Saving to: {output_file}")

    preparer.prepare_training_data(atm_file, ocn_file, max_patterns, output_file, thin_fraction)
    print(f"Done. Training data written to: {output_file}")

if __name__ == "__main__":
    main()
