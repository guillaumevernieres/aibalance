#!/usr/bin/env python3
"""
CLI script for training UFS Emulator models.
This provides a simple command-line interface to the ufsemulator package.
"""

import sys
from pathlib import Path

# Add the parent directory to Python path to import ufsemulator package
sys.path.insert(0, str(Path(__file__).parent.parent))

from ufsemulator.training_simple import main  # noqa: E402


if __name__ == "__main__":
    main()
