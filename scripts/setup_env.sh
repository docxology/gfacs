#!/bin/bash
# GFACS Environment Setup Script
# This script sets up the development environment using uv

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up GFACS development environment${NC}"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ Error: pyproject.toml not found. Please run this script from the gfacs root directory.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11.0"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo -e "${RED}❌ Error: Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION${NC}"
    echo -e "${YELLOW}💡 Please install Python 3.11+ and try again.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python version check passed: $PYTHON_VERSION${NC}"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}📦 Installing uv package manager...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo -e "${GREEN}✅ uv is installed${NC}"

# Create virtual environment and install dependencies
echo -e "${YELLOW}🔧 Creating virtual environment and installing dependencies...${NC}"

# Sync with uv (creates .venv and installs all dependencies)
if ! uv sync --dev; then
    echo -e "${RED}❌ Failed to create virtual environment and install basic dependencies${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Basic environment setup complete!${NC}"

# Install PyTorch dependencies
echo -e "${YELLOW}🔥 Installing PyTorch dependencies...${NC}"

PYTORCH_INSTALLED=false

# Check for CUDA availability
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null 2>&1; then
    echo -e "${GREEN}🎮 CUDA detected, installing PyTorch with CUDA support...${NC}"
    # Try CUDA 12.1 first, fallback to other versions
    if uv add torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121; then
        echo -e "${GREEN}✅ PyTorch with CUDA 12.1 installed!${NC}"
        PYTORCH_INSTALLED=true
    elif uv add torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118; then
        echo -e "${GREEN}✅ PyTorch with CUDA 11.8 installed!${NC}"
        PYTORCH_INSTALLED=true
    else
        echo -e "${YELLOW}⚠️  CUDA PyTorch installation failed, falling back to CPU version...${NC}"
        if uv add torch torchvision torchaudio; then
            echo -e "${GREEN}✅ PyTorch CPU fallback installed!${NC}"
            PYTORCH_INSTALLED=true
        fi
    fi
else
    echo -e "${YELLOW}💻 No CUDA detected, installing CPU PyTorch...${NC}"
    # Try direct installation first (usually works better)
    if uv add torch torchvision torchaudio; then
        echo -e "${GREEN}✅ PyTorch CPU installed!${NC}"
        PYTORCH_INSTALLED=true
    else
        echo -e "${YELLOW}⚠️  Standard installation failed, trying PyTorch CPU index...${NC}"
        if uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cpu; then
            echo -e "${GREEN}✅ PyTorch CPU (via index) installed!${NC}"
            PYTORCH_INSTALLED=true
        fi
    fi
fi

if [ "$PYTORCH_INSTALLED" = false ]; then
    echo -e "${RED}❌ PyTorch installation failed${NC}"
    echo -e "${YELLOW}💡 You can try installing manually:${NC}"
    echo -e "${BLUE}   uv add torch torchvision torchaudio${NC}"
fi

# Install PyTorch Geometric
echo -e "${YELLOW}🔗 Installing PyTorch Geometric...${NC}"

# Determine PyTorch version for compatible PyG installation
PYTORCH_VERSION=$(uv run python -c "import torch; print(torch.__version__)" 2>/dev/null | cut -d'+' -f1)
if [ -z "$PYTORCH_VERSION" ]; then
    echo -e "${RED}❌ Failed to detect PyTorch version${NC}"
    PYTORCH_VERSION="2.9.1"  # Fallback version
    echo -e "${YELLOW}⚠️  Using fallback PyTorch version: $PYTORCH_VERSION${NC}"
fi
echo -e "${BLUE}📊 Detected PyTorch version: $PYTORCH_VERSION${NC}"

# Try to install PyTorch Geometric using uv pip (bypasses build isolation issues)
PYG_INSTALLED=false

# Install hatchling first to avoid build issues
echo -e "${YELLOW}📦 Installing build dependencies...${NC}"
if uv add hatchling; then
    echo -e "${GREEN}✅ Build dependencies installed!${NC}"

    # Now try installing PyTorch Geometric
    if uv pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv --no-build-isolation; then
        PYG_INSTALLED=true
        echo -e "${GREEN}✅ PyTorch Geometric installed successfully!${NC}"
    else
        echo -e "${YELLOW}⚠️  uv pip installation failed, trying with --no-deps...${NC}"
        if uv pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv --no-deps --no-build-isolation; then
            PYG_INSTALLED=true
            echo -e "${GREEN}✅ PyTorch Geometric installed (no-deps)!${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Build dependencies installation failed, trying direct pip...${NC}"
    if uv pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv --no-build-isolation; then
        PYG_INSTALLED=true
        echo -e "${GREEN}✅ PyTorch Geometric installed with pip!${NC}"
    fi
fi

if [ "$PYG_INSTALLED" = true ]; then
    echo -e "${GREEN}✅ PyTorch Geometric installed!${NC}"
else
    echo -e "${YELLOW}⚠️  PyTorch Geometric installation failed - some neural network features may not work${NC}"
    echo -e "${BLUE}💡 You can try installing manually later:${NC}"
    echo -e "${BLUE}   uv pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv --no-build-isolation${NC}"
    echo -e "${BLUE}   Basic GFACS functionality will still work${NC}"
fi

# Ask about installing external solvers
echo ""
echo -e "${YELLOW}🔧 External Solvers Setup${NC}"
echo -e "${BLUE}GFACS can use external solvers (Concorde TSP, HGS-CVRP) for improved performance.${NC}"
echo -e "${BLUE}This requires building C/C++ libraries and may take several minutes.${NC}"
echo ""
echo -n "Install external solvers? [y/N]: "
read -r install_solvers

if [[ "$install_solvers" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🏗️  Installing external solvers...${NC}"

    if [ -f "scripts/setup_solvers.sh" ]; then
        if bash scripts/setup_solvers.sh; then
            echo -e "${GREEN}✅ External solvers installed!${NC}"
        else
            echo -e "${YELLOW}⚠️  External solver installation failed${NC}"
            echo -e "${BLUE}💡 You can try installing manually later: bash scripts/setup_solvers.sh${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  setup_solvers.sh not found - skipping external solver installation${NC}"
    fi
else
    echo -e "${YELLOW}⏭️  Skipping external solver installation${NC}"
fi

# Run verification
echo ""
echo -e "${YELLOW}✅ Running installation verification...${NC}"

if [ -f "scripts/verify_install.py" ]; then
    if uv run python scripts/verify_install.py; then
        echo -e "${GREEN}✅ Verification complete!${NC}"
    else
        echo -e "${YELLOW}⚠️  Verification found some issues, but setup is complete${NC}"
        echo -e "${BLUE}💡 You can re-run verification later: uv run python scripts/verify_install.py${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  verify_install.py not found - skipping verification${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Complete GFACS setup finished!${NC}"
echo ""
echo -e "${GREEN}💡 To activate the environment:${NC}"
echo -e "${BLUE}   source .venv/bin/activate${NC}"
echo ""
echo -e "${GREEN}🚀 Ready to use GFACS! Try:${NC}"
echo -e "${BLUE}   ./run.sh${NC}                    # Interactive menu"
echo -e "${BLUE}   ./run.sh quick${NC}               # Quick test"
echo -e "${BLUE}   ./run.sh problems --problems tsp  # Run specific problem"
