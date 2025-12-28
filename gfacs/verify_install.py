#!/usr/bin/env python3
"""
GFACS Installation Verification Script

This script verifies that all dependencies are correctly installed and
external solvers are working properly.
"""

import sys
import os
import subprocess
import importlib
import argparse
from pathlib import Path

# Determine project root based on script location
# gfacs/verify_install.py -> gfacs/ -> project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Add the project root to Python path for imports
sys.path.insert(0, str(PROJECT_ROOT))
# Also add the gfacs directory for direct module imports
sys.path.insert(0, str(PROJECT_ROOT / "gfacs"))

class VerificationError(Exception):
    """Custom exception for verification failures."""
    pass

def print_status(message: str, status: str = "INFO"):
    """Print status message with color coding."""
    colors = {
        "INFO": "\033[0;34m",  # Blue
        "SUCCESS": "\033[0;32m",  # Green
        "WARNING": "\033[1;33m",  # Yellow
        "ERROR": "\033[0;31m",  # Red
        "RESET": "\033[0m"  # Reset
    }

    color = colors.get(status, colors["RESET"])
    reset = colors["RESET"]
    print(f"{color}[{status}] {message}{reset}")

def check_python_version():
    """Check Python version requirement."""
    required = (3, 11)
    current = sys.version_info

    if current < required:
        raise VerificationError(
            f"Python {required[0]}.{required[1]}+ required, found {current[0]}.{current[1]}\n"
            "Please upgrade Python to version 3.11 or higher."
        )

    print_status(f"Python version: {current[0]}.{current[1]}.{current[2]}", "SUCCESS")

def check_core_dependencies():
    """Check core Python dependencies."""
    required_deps = [
        "numpy",
        "numba",
        "pandas",
        "scipy",
        "tqdm",
        "wandb"
    ]

    failed_deps = []

    for dep in required_deps:
        try:
            importlib.import_module(dep)
            print_status(f"✓ {dep}", "SUCCESS")
        except ImportError:
            failed_deps.append(dep)
            print_status(f"✗ {dep}", "ERROR")

    if failed_deps:
        error_msg = f"Missing core dependencies: {', '.join(failed_deps)}\n"
        error_msg += "Try installing with: pip install " + " ".join(failed_deps)
        raise VerificationError(error_msg)

def check_optional_dependencies():
    """Check optional dependencies."""
    optional_deps = [
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("pyvrp", "PyVRP")
    ]

    for module, name in optional_deps:
        try:
            importlib.import_module(module)
            print_status(f"✓ {name} available", "SUCCESS")
        except ImportError:
            print_status(f"⚠ {name} not available (optional)", "WARNING")

def check_external_solvers():
    """Check external solvers."""
    print_status("Checking external solvers...", "INFO")

    # Check HGS-CVRP (platform-specific extensions)
    hgs_found = False
    hgs_path = None

    # Check for all possible library extensions
    for ext in ['.so', '.dylib', '.dll']:
        candidate_path = PROJECT_ROOT / f"cvrp_nls/HGS-CVRP-main/build/libhgscvrp{ext}"
        if candidate_path.exists():
            hgs_found = True
            hgs_path = candidate_path
            break

    if hgs_found:
        print_status(f"✓ HGS-CVRP library found", "SUCCESS")
    else:
        print_status("⚠ HGS-CVRP library not found", "WARNING")
        print_status("  Run: ./scripts/setup_solvers.sh", "INFO")

    # Check Concorde
    concorde_path = PROJECT_ROOT / "tsp_nls/concorde/TSP/concorde"
    if concorde_path.exists():
        print_status("✓ Concorde TSP solver found", "SUCCESS")
        # Test concorde execution
        try:
            result = subprocess.run(
                [str(concorde_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode in [0, 1]:  # Some programs return 1 for --help
                print_status("✓ Concorde TSP solver executable", "SUCCESS")
            else:
                print_status("⚠ Concorde TSP solver execution failed", "WARNING")
                print_status("  Check solver installation or run: ./scripts/setup_solvers.sh", "INFO")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            print_status("⚠ Concorde TSP solver execution failed", "WARNING")
            print_status("  Check solver installation or run: ./scripts/setup_solvers.sh", "INFO")
    else:
        print_status("⚠ Concorde TSP solver not found", "WARNING")
        print_status("  Note: Concorde may not be available on Apple Silicon macOS", "INFO")
        print_status("  Run: ./scripts/setup_solvers.sh (may fail on Apple Silicon)", "INFO")

def check_module_imports():
    """Check that all GFACS modules can be imported."""
    print_status("Checking module imports...", "INFO")

    modules_to_test = [
        ("tsp_nls.aco", "TSP ACO module"),
        ("tsp_nls.net", "TSP Net module"),
        ("tsp_nls.utils", "TSP Utils module"),
        ("cvrp_nls.aco", "CVRP ACO module"),
        ("cvrp_nls.net", "CVRP Net module"),
        ("cvrp_nls.utils", "CVRP Utils module"),
        ("cvrptw_nls.aco", "CVRPTW ACO module"),
        ("cvrptw_nls.net", "CVRPTW Net module"),
        ("bpp.aco", "BPP ACO module"),
        ("bpp.net", "BPP Net module"),
        ("op.aco", "OP ACO module"),
        ("op.net", "OP Net module"),
        ("pctsp.aco", "PCTSP ACO module"),
        ("pctsp.net", "PCTSP Net module"),
        ("smtwtp.aco", "SMTWTP ACO module"),
        ("smtwtp.net", "SMTWTP Net module"),
        ("sop.aco", "SOP ACO module"),
        ("sop.net", "SOP Net module"),
    ]

    failed_imports = []

    for module, description in modules_to_test:
        try:
            importlib.import_module(module)
            print_status(f"✓ {description}", "SUCCESS")
        except (ImportError, FileNotFoundError, OSError) as e:
            # FileNotFoundError can occur when optional external libraries aren't built
            error_str = str(e).lower()
            if ("libhgscvrp" in error_str or "concorde" in error_str or
                "shared library file" in error_str):
                print_status(f"⚠ {description}: External solver not available (optional)", "WARNING")
            else:
                failed_imports.append((module, str(e)))
                print_status(f"✗ {description}: {e}", "ERROR")

    if failed_imports:
        error_msg = "Failed module imports:\n"
        for module, error in failed_imports:
            error_msg += f"  - {module}: {error}\n"
        error_msg += "\nThis may indicate:\n"
        error_msg += "  - Missing optional dependencies (check installation)\n"
        error_msg += "  - External solver libraries not built (run ./scripts/setup_solvers.sh)\n"
        error_msg += "  - Corrupted installation (try reinstalling)"
        raise VerificationError(error_msg)

def check_data_directories():
    """Check that data directories exist."""
    print_status("Checking data directories...", "INFO")

    data_dirs = [
        "data/tsp",
        "data/cvrp",
        "pretrained"
    ]

    for data_dir in data_dirs:
        full_path = PROJECT_ROOT / data_dir
        if full_path.exists():
            print_status(f"✓ {data_dir} exists", "SUCCESS")
        else:
            print_status(f"⚠ {data_dir} missing", "WARNING")

def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(
        description="GFACS Installation Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_install.py                    # Run all checks
  python verify_install.py --check-solvers   # Only check external solvers
  python verify_install.py --verbose         # Detailed output
  python verify_install.py --quiet           # Minimal output
        """
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output (only errors and final status)'
    )

    parser.add_argument(
        '--check-solvers',
        action='store_true',
        help='Only check external solvers'
    )

    parser.add_argument(
        '--check-problems',
        action='store_true',
        help='Only check problem modules'
    )

    args = parser.parse_args()

    # Handle quiet mode - create local print function
    local_print_status = print_status
    if args.quiet:
        def quiet_print(message, status="INFO"):
            if status in ["ERROR", "WARNING"] or "failed" in message.lower():
                print(f"[{status}] {message}")
        local_print_status = quiet_print

    # Show header unless quiet
    if not args.quiet:
        local_print_status("🚀 Starting GFACS installation verification", "INFO")
        print()

    try:
        # Determine which checks to run
        run_all = not (args.check_solvers or args.check_problems)

        if run_all or args.check_solvers:
            check_python_version()
            if not args.quiet:
                print()
            check_external_solvers()
            if not args.quiet:
                print()

        if run_all:
            check_core_dependencies()
            if not args.quiet:
                print()
            check_optional_dependencies()
            if not args.quiet:
                print()

        if run_all or args.check_problems:
            check_module_imports()
            if not args.quiet:
                print()
            check_data_directories()
            if not args.quiet:
                print()

        if not args.quiet:
            local_print_status("🎉 All checks passed! GFACS is ready to use.", "SUCCESS")
        return 0

    except VerificationError as e:
        if not args.quiet:
            print()
        local_print_status(f"❌ Verification failed: {e}", "ERROR")
        if not args.quiet:
            print()
            local_print_status("💡 For installation help, see: README.md or run ./scripts/setup_menu.sh", "INFO")
        return 1
    except Exception as e:
        if not args.quiet:
            print()
        local_print_status(f"❌ Unexpected error: {e}", "ERROR")
        if not args.quiet:
            print()
            local_print_status("💡 Please report this issue with full error details", "INFO")
        return 1

if __name__ == "__main__":
    sys.exit(main())
