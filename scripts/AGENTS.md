# Module: Scripts (`scripts/`)

## Overview

The Scripts module provides utility scripts for setting up, configuring, and managing GFACS installations. These scripts automate common tasks like environment setup, solver installation, and system verification.

## Scripts

### setup_env.sh
Environment setup script for configuring the GFACS development environment.

**Purpose:**
- Install system dependencies
- Configure Python environment
- Set up required packages
- Initialize development tools

**Usage:**
```bash
# Make executable and run
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

**Requirements:**
- Ubuntu/Debian-based Linux system
- sudo privileges for package installation
- Internet connection for downloads

**Actions Performed:**
- Updates package manager
- Installs system dependencies (build tools, libraries)
- Configures Python virtual environment
- Installs development dependencies
- Sets up environment variables

### setup_solvers.sh
External solver installation script for TSP and VRP solvers.

**Purpose:**
- Install Concorde TSP solver
- Install HGS-CVRP solver
- Configure solver paths
- Verify solver functionality

**Usage:**
```bash
# Make executable and run
chmod +x scripts/setup_solvers.sh
./scripts/setup_solvers.sh
```

**Requirements:**
- C/C++ compiler (gcc/g++)
- CMake build system
- Sufficient disk space (~500MB)
- Internet connection

**Solvers Installed:**
- **Concorde**: Exact TSP solver for verification
- **HGS-CVRP**: Hybrid Genetic Search for CVRP local search

### setup_menu.sh
Interactive setup and verification menu for comprehensive GFACS installation.

**Purpose:**
- Complete GFACS environment setup and verification
- Interactive menu for installation and configuration
- Environment status checking and dependency validation
- External solver installation
- Orchestrator execution with submenu

**Usage:**
```bash
# Make executable and run
chmod +x scripts/setup_menu.sh
./scripts/setup_menu.sh
```

**Features:**
- Comprehensive environment status checking
- Interactive setup menu system
- External solver building (HGS-CVRP, Concorde)
- Installation verification
- Orchestrator submenu for running experiments

### run.sh
Experiment execution script focused on running GFACS orchestrator experiments.

**Purpose:**
- Execute GFACS experiments with various configurations
- Support quick testing, full benchmarks, and custom problem sets
- Provide both command-line and interactive menu interfaces
- Generate structured output directories with comprehensive results
- Focus on experiment execution rather than environment setup

**Usage:**
```bash
# Quick test with TSP only
./run.sh quick

# Full experiment with all 8 problems
./run.sh full --experiment-name "production_run"

# Specific problems only
./run.sh problems --problems "tsp_nls cvrp_nls"

# Custom configuration
./run.sh custom --config config/benchmark.yaml

# Interactive menu mode (no arguments)
./run.sh
```

**Command Line Options:**
- `quick|full|problems|custom|help`: Command to execute
- `--problems "LIST"`: Space-separated list of problems (for problems command)
- `--config FILE`: Custom configuration file path (for custom command)
- `--experiment-name NAME`: Custom experiment name
- `--output-dir DIR`: Custom output directory
- `--verbose`: Enable verbose output
- `--dry-run`: Show what would be executed without running

### gfacs/verify_install.py
Installation verification script for GFACS components (located in gfacs/ directory).

**Purpose:**
- Test GFACS installation completeness
- Verify all dependencies are available
- Check solver functionality
- Validate problem modules

**Usage:**
```bash
# Run verification
python gfacs/verify_install.py

# Verbose output
python gfacs/verify_install.py --verbose

# Check specific components
python gfacs/verify_install.py --check-solvers --check-problems
```

**Verification Checks:**
- Python environment and packages
- External solvers (Concorde, HGS-CVRP)
- Problem module imports
- Neural network functionality
- Data loading capabilities
- GPU availability and CUDA setup

**Exit Codes:**
- `0`: All checks passed
- `1`: Some optional components missing
- `2`: Critical components failed
- `3`: Installation errors detected

## Dependencies

### System Requirements
- **OS**: Linux (Ubuntu 18.04+, CentOS 7+)
- **CPU**: x86_64 architecture
- **Memory**: 8GB+ RAM recommended
- **Storage**: 10GB+ free space
- **Network**: Internet connection required

### Software Dependencies
- **Python**: 3.8+ (3.11+ recommended)
- **Compiler**: GCC 7+ or Clang 6+
- **Build Tools**: CMake 3.10+, Make
- **Package Manager**: apt, yum, or equivalent

## Installation Process

### Automated Installation
```bash
# Run complete setup
./scripts/setup_menu.sh

# Or use individual scripts
./scripts/setup_env.sh
./scripts/setup_solvers.sh
./scripts/verify_install.py
```

### Manual Installation
For custom installations or troubleshooting:

1. **Environment Setup:**
   ```bash
   # Install system packages
   sudo apt update
   sudo apt install -y build-essential cmake python3-dev python3-pip
   ```

2. **Python Environment:**
   ```bash
   # Create virtual environment
   python3 -m venv gfacs_env
   source gfacs_env/bin/activate

   # Install Python packages
   pip install -r requirements.txt
   ```

3. **Solver Installation:**
   ```bash
   # Install Concorde
   cd tsp_nls
   chmod +x install_concorde.sh
   ./install_concorde.sh

   # Install HGS-CVRP
   cd ../cvrp_nls/HGS-CVRP-main
   mkdir build && cd build
   cmake ..
   make
   ```

### Verification
```bash
# Run verification script
python scripts/verify_install.py

# Check specific problems
python -c "from gfacs.tsp_nls.aco import ACO; print('TSP module OK')"
python -c "from gfacs.cvrp_nls.aco import ACO; print('CVRP module OK')"
```

## Troubleshooting

### Common Issues

**Setup Script Permissions:**
```bash
# Fix permissions
chmod +x scripts/*.sh
```

**Missing Dependencies:**
```bash
# Install missing packages
sudo apt install -y git wget unzip
```

**Python Virtual Environment:**
```bash
# Recreate environment
rm -rf gfacs_env
python3 -m venv gfacs_env
source gfacs_env/bin/activate
pip install -r requirements.txt
```

**Solver Compilation Errors:**
```bash
# Check compiler version
gcc --version

# Install development headers
sudo apt install -y libgmp-dev libmpfr-dev
```

**CUDA/GPU Issues:**
```bash
# Check CUDA installation
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Support Resources
- Check `verify_install.py` output for specific errors
- Review log files in `logs/` directory
- Consult problem-specific README files
- Check GitHub issues for known problems

## Development

### Adding New Scripts
- Follow naming convention: `snake_case.sh` or `snake_case.py`
- Include comprehensive error handling
- Add help text and usage examples
- Update this documentation

### Script Standards
- Use bash for shell scripts (`#!/bin/bash`)
- Include parameter validation
- Provide progress feedback
- Handle cleanup on failure
- Support `--help` and `--version` flags

## Integration

Scripts integrate with:
- **Main Repository**: Core GFACS functionality
- **Problem Modules**: Individual optimization problems
- **Configuration System**: Experiment setup
- **Orchestrator**: Experiment execution
- **Testing Framework**: Validation and verification