#!/bin/bash
# GFACS External Solvers Setup Script
# This script builds HGS-CVRP and Concorde TSP solver

# Don't exit on any error - let caller handle failures gracefully

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Setting up external solvers for GFACS${NC}"

# Initialize success flags
HGS_SUCCESS=false
CONCORDE_SUCCESS=false

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ Error: pyproject.toml not found. Please run this script from the gfacs root directory.${NC}"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required build tools
echo -e "${YELLOW}🔍 Checking for required build tools...${NC}"

MISSING_TOOLS=()

if ! command_exists cmake; then
    MISSING_TOOLS+=("cmake")
fi

if ! command_exists make; then
    MISSING_TOOLS+=("make")
fi

if ! command_exists gcc || ! command_exists g++; then
    MISSING_TOOLS+=("gcc/g++")
fi

if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    echo -e "${RED}❌ Missing required build tools: ${MISSING_TOOLS[*]}${NC}"
    echo -e "${YELLOW}💡 Please install the missing tools and try again.${NC}"
    echo -e "${YELLOW}   On Ubuntu/Debian: sudo apt-get install build-essential cmake${NC}"
    echo -e "${YELLOW}   On macOS: xcode-select --install && brew install cmake${NC}"
    echo -e "${YELLOW}⚠️  External solver setup aborted due to missing tools${NC}"
    exit 0  # Don't fail the overall setup
fi

echo -e "${GREEN}✅ Build tools are available${NC}"

# Setup HGS-CVRP solver
echo -e "${YELLOW}🏗️  Building HGS-CVRP solver...${NC}"

cd cvrp_nls/HGS-CVRP-main

# Clean any previous build
if [ -d "build" ]; then
    rm -rf build
fi

mkdir -p build
cd build

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"
make lib

# Check for the built library (different extensions on different platforms)
LIB_FOUND=false
LIB_PATH=""

if [ -f "libhgscvrp.so" ]; then
    LIB_FOUND=true
    LIB_PATH="libhgscvrp.so"
elif [ -f "libhgscvrp.dylib" ]; then
    LIB_FOUND=true
    LIB_PATH="libhgscvrp.dylib"
elif [ -f "libhgscvrp.dll" ]; then
    LIB_FOUND=true
    LIB_PATH="libhgscvrp.dll"
fi

if [ "$LIB_FOUND" = false ]; then
    echo -e "${RED}❌ HGS-CVRP build failed - libhgscvrp library not found (.so, .dylib, or .dll)${NC}"
    echo -e "${YELLOW}💡 HGS-CVRP is optional, continuing setup...${NC}"
    HGS_SUCCESS=false
else
    HGS_SUCCESS=true
fi

if [ "$HGS_SUCCESS" = true ]; then
    echo -e "${GREEN}✅ HGS-CVRP solver built successfully ($LIB_PATH)${NC}"
fi

# Return to root directory
cd ../../../

# Setup Concorde TSP solver
echo -e "${YELLOW}🏗️  Building Concorde TSP solver...${NC}"

cd tsp_nls

# Run the existing installation script
chmod +x install_concorde.sh
if ./install_concorde.sh; then
    CONCORDE_SUCCESS=true
else
    # Check if it failed due to ARM64 macOS (expected failure)
    ARCH=$(uname -m)
    if [[ "$OSTYPE" == "darwin"* && "$ARCH" == "arm64" ]]; then
        echo -e "${YELLOW}⚠️  Concorde build failed (expected on Apple Silicon macOS)${NC}"
        echo -e "${BLUE}💡 Concorde is optional - TSP functionality works without it${NC}"
    else
        echo -e "${RED}❌ Concorde build failed${NC}"
        echo -e "${YELLOW}💡 Concorde is optional, continuing setup...${NC}"
    fi
    CONCORDE_SUCCESS=false
fi

if [ "$CONCORDE_SUCCESS" = true ]; then
    echo -e "${GREEN}✅ Concorde TSP solver built successfully${NC}"
fi

# Return to root directory
cd ..

echo ""
echo -e "${GREEN}🎉 External solvers setup complete!${NC}"
echo ""
echo -e "${GREEN}📋 Summary:${NC}"

if [ "$HGS_SUCCESS" = true ]; then
    echo -e "   ✅ HGS-CVRP: $(pwd)/cvrp_nls/HGS-CVRP-main/build/$LIB_PATH"
else
    echo -e "   ❌ HGS-CVRP: Build failed"
fi

if [ "$CONCORDE_SUCCESS" = true ]; then
    echo -e "   ✅ Concorde: $(pwd)/tsp_nls/concorde/TSP/concorde"
else
    echo -e "   ❌ Concorde: Build failed"
fi

echo ""
echo -e "${YELLOW}💡 Note: External solvers are optional but improve performance${NC}"
echo -e "${YELLOW}   You can now run: python scripts/verify_install.py${NC}"
