#!/usr/bin/env python3
"""
GFACS Installation Verification Script (Wrapper)

This wrapper script calls the main verification function from gfacs.verify_install.
This maintains compatibility with setup scripts that expect verify_install.py
to be in the scripts/ directory.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path so we can import gfacs
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from gfacs.verify_install import main
except ImportError as e:
    print(f"ERROR: Could not import gfacs.verify_install: {e}")
    print("Make sure GFACS is properly installed and the Python path is correct.")
    sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())