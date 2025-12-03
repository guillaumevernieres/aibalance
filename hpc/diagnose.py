#!/usr/bin/env python
"""Diagnostic script to check threading and performance."""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

print("=" * 60)
print("DIAGNOSTIC INFORMATION")
print("=" * 60)

# Environment variables
print(f"\nEnvironment Variables:")
print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'NOT SET')}")
print(f"  NUMEXPR_NUM_THREADS: {os.environ.get('NUMEXPR_NUM_THREADS', 'NOT SET')}")

# CPU info
print(f"\nCPU Information:")
print(f"  os.cpu_count(): {os.cpu_count()}")
print(f"  PyTorch threads: {torch.get_num_threads()}")

# CUDA info
print(f"\nCUDA Information:")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

# NumPy/MKL info
print(f"\nNumPy Configuration:")
try:
    import numpy as np
    np.show_config()
except:
    print("  Could not get NumPy config")

# Quick performance test
print(f"\n" + "=" * 60)
print("PERFORMANCE TEST")
print("=" * 60)

# Test matrix multiplication
size = 2000
print(f"\nMatrix multiplication test ({size}x{size}):")

# NumPy test
A = np.random.randn(size, size).astype(np.float32)
B = np.random.randn(size, size).astype(np.float32)

start = time.time()
C = np.dot(A, B)
numpy_time = time.time() - start
print(f"  NumPy: {numpy_time:.3f} seconds")

# PyTorch CPU test
A_torch = torch.from_numpy(A)
B_torch = torch.from_numpy(B)

start = time.time()
C_torch = torch.mm(A_torch, B_torch)
torch_time = time.time() - start
print(f"  PyTorch CPU: {torch_time:.3f} seconds")

# Test data loading
print(f"\nData loading test:")
sys.path.insert(0, str(Path('..').resolve()))

nc_file = 'gdas.ice.t00z.inst.f009.nc'
if Path(nc_file).exists():
    from ufsemulator.data import IceDataPreparer
    from ufsemulator.training import load_config
    
    config = load_config('config_aice.yaml')
    
    preparer = IceDataPreparer(config)
    
    start = time.time()
    data = preparer.read_netcdf_data(nc_file)
    nc_time = time.time() - start
    print(f"  NetCDF read: {nc_time:.3f} seconds")
    
    start = time.time()
    patterns, targets, lons, lats = preparer.filter_data(data, max_patterns=10000)
    filter_time = time.time() - start
    print(f"  Filter data (10k patterns): {filter_time:.3f} seconds")
else:
    print(f"  NetCDF file not found: {nc_file}")

print(f"\n" + "=" * 60)
