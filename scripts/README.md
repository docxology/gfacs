# Scripts Directory

This directory contains setup and utility scripts for GFACS.

## Setup Scripts

### setup_env.sh
Complete environment setup script that:
- Checks Python version and uv installation
- Creates/updates virtual environment
- Installs Python dependencies (PyTorch, PyTorch Geometric)
- Optionally installs external solvers (Concorde, HGS-CVRP)
- Runs installation verification

### setup_solvers.sh
Installs external optimization solvers:
- **HGS-CVRP**: Hybrid Genetic Search for CVRP
- **Concorde**: Exact TSP solver

### verify_install.py
Verifies GFACS installation and dependencies.

## Usage

```bash
# Complete setup (recommended)
./scripts/setup_env.sh

# Install solvers only
./scripts/setup_solvers.sh

# Verify installation
python scripts/verify_install.py
```

## Interactive Setup

Run `./run.sh` without arguments for an interactive menu with setup options.