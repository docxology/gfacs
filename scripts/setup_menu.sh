#!/bin/bash
# ============================================================================
# GFACS Interactive Environment Setup Menu
# ============================================================================
#
# This script provides an interactive menu system for GFACS environment
# checking, setup, and orchestrator execution.
#
# Features:
#   - Comprehensive environment status checks
#   - Interactive menu-driven setup options
#   - Integration with existing setup scripts
#   - Orchestrator execution with submenu
#
# ============================================================================

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/config"
OUTPUTS_DIR="$REPO_ROOT/outputs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Global variables for status tracking
declare -A CHECK_RESULTS
declare -a FAILED_CHECKS
declare -a WARNING_CHECKS

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_header() {
    echo -e "${PURPLE}===============================================================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}===============================================================================${NC}"
}

# Check if we're in the right directory
check_repository() {
    if [ ! -f "$REPO_ROOT/pyproject.toml" ]; then
        log_error "pyproject.toml not found. Please run this script from the gfacs/scripts/ directory."
        exit 1
    fi
}

# Check Python version
check_python_version() {
    local check_name="python_version"
    local required_major=3
    local required_minor=11

    if ! command -v python3 &> /dev/null; then
        CHECK_RESULTS[$check_name]="missing"
        FAILED_CHECKS+=("$check_name")
        return
    fi

    local version=$(python3 --version 2>&1 | awk '{print $2}')
    local major=$(echo $version | cut -d. -f1)
    local minor=$(echo $version | cut -d. -f2)

    if [ "$major" -gt "$required_major" ] || ([ "$major" -eq "$required_major" ] && [ "$minor" -ge "$required_minor" ]); then
        CHECK_RESULTS[$check_name]="pass:$version"
    else
        CHECK_RESULTS[$check_name]="fail:$version"
        FAILED_CHECKS+=("$check_name")
    fi
}

# Check uv package manager
check_uv() {
    local check_name="uv"

    if command -v uv &> /dev/null; then
        local version=$(uv --version 2>&1 | awk '{print $2}')
        CHECK_RESULTS[$check_name]="pass:$version"
    else
        CHECK_RESULTS[$check_name]="missing"
        FAILED_CHECKS+=("$check_name")
    fi
}

# Check virtual environment
check_virtual_env() {
    local check_name="virtual_env"

    if [ -d "$REPO_ROOT/.venv" ] && [ -f "$REPO_ROOT/.venv/bin/python" ]; then
        # Check if virtual env is activated by comparing python paths
        local venv_python="$REPO_ROOT/.venv/bin/python"
        local current_python=$(which python3 2>/dev/null || echo "")

        if [ "$venv_python" = "$current_python" ]; then
            CHECK_RESULTS[$check_name]="active"
        else
            CHECK_RESULTS[$check_name]="exists"
        fi
    else
        CHECK_RESULTS[$check_name]="missing"
        FAILED_CHECKS+=("$check_name")
    fi
}

# Check PyTorch installation
check_pytorch() {
    local check_name="pytorch"

    if [ "${CHECK_RESULTS[virtual_env]}" = "missing" ]; then
        CHECK_RESULTS[$check_name]="venv_missing"
        WARNING_CHECKS+=("$check_name")
        return
    fi

    # Use uv run to check in virtual environment context
    if cd "$REPO_ROOT" && uv run python3 -c "import torch; print(torch.__version__)" &>/dev/null; then
        local version=$(cd "$REPO_ROOT" && uv run python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        CHECK_RESULTS[$check_name]="pass:$version"
    else
        CHECK_RESULTS[$check_name]="missing"
        WARNING_CHECKS+=("$check_name")
    fi
}

# Check PyTorch Geometric
check_pytorch_geometric() {
    local check_name="torch_geometric"

    if [ "${CHECK_RESULTS[virtual_env]}" = "missing" ]; then
        CHECK_RESULTS[$check_name]="venv_missing"
        WARNING_CHECKS+=("$check_name")
        return
    fi

    if cd "$REPO_ROOT" && uv run python3 -c "import torch_geometric; print(torch_geometric.__version__)" &>/dev/null; then
        local version=$(cd "$REPO_ROOT" && uv run python3 -c "import torch_geometric; print(torch_geometric.__version__)" 2>/dev/null)
        CHECK_RESULTS[$check_name]="pass:$version"
    else
        CHECK_RESULTS[$check_name]="missing"
        WARNING_CHECKS+=("$check_name")
    fi
}

# Check HGS-CVRP solver
check_hgs_solver() {
    local check_name="hgs_cvrp"
    local hgs_path="$REPO_ROOT/cvrp_nls/HGS-CVRP-main/build/libhgscvrp.so"

    if [ -f "$hgs_path" ]; then
        CHECK_RESULTS[$check_name]="pass:built"
    else
        CHECK_RESULTS[$check_name]="missing"
        WARNING_CHECKS+=("$check_name")
    fi
}

# Check Concorde solver
check_concorde_solver() {
    local check_name="concorde"
    local concorde_path="$REPO_ROOT/tsp_nls/concorde/TSP/concorde"

    if [ -f "$concorde_path" ]; then
        CHECK_RESULTS[$check_name]="pass:built"
    else
        CHECK_RESULTS[$check_name]="missing"
        WARNING_CHECKS+=("$check_name")
    fi
}

# Check data directories
check_data_directories() {
    local check_name="data_dirs"
    local missing_dirs=()

    local data_dirs=("data/tsp" "data/cvrp" "pretrained")

    for dir in "${data_dirs[@]}"; do
        if [ ! -d "$REPO_ROOT/$dir" ]; then
            missing_dirs+=("$dir")
        fi
    done

    if [ ${#missing_dirs[@]} -eq 0 ]; then
        CHECK_RESULTS[$check_name]="pass:all_present"
    else
        CHECK_RESULTS[$check_name]="missing:${missing_dirs[*]}"
        WARNING_CHECKS+=("$check_name")
    fi
}

# Check core Python dependencies
check_core_dependencies() {
    local check_name="core_deps"

    if [ "${CHECK_RESULTS[virtual_env]}" = "missing" ]; then
        CHECK_RESULTS[$check_name]="venv_missing"
        WARNING_CHECKS+=("$check_name")
        return
    fi

    local missing_deps=()
    local deps=("numpy" "numba" "pandas" "scipy" "tqdm" "wandb")

    for dep in "${deps[@]}"; do
        if ! cd "$REPO_ROOT" && uv run python3 -c "import $dep" &>/dev/null; then
            missing_deps+=("$dep")
        fi
    done

    if [ ${#missing_deps[@]} -eq 0 ]; then
        CHECK_RESULTS[$check_name]="pass:all_installed"
    else
        CHECK_RESULTS[$check_name]="missing:${missing_deps[*]}"
        WARNING_CHECKS+=("$check_name")
    fi
}

# Main environment checking function
check_environment() {
    log_info "Checking GFACS environment..."

    # Reset global variables
    unset CHECK_RESULTS
    declare -gA CHECK_RESULTS
    FAILED_CHECKS=()
    WARNING_CHECKS=()

    # Run all checks
    check_python_version
    check_uv
    check_virtual_env
    check_pytorch
    check_pytorch_geometric
    check_hgs_solver
    check_concorde_solver
    check_data_directories
    check_core_dependencies

    log_info "Environment check complete"
}

# Display status for a single check
display_check_status() {
    local check_name="$1"
    local display_name="$2"
    local status="${CHECK_RESULTS[$check_name]}"

    case $status in
        pass:*)
            local version=$(echo "$status" | cut -d: -f2-)
            echo -e "  ${GREEN}✅${NC} $display_name: $version"
            ;;
        active)
            echo -e "  ${GREEN}✅${NC} $display_name: Active"
            ;;
        exists)
            echo -e "  ${GREEN}✅${NC} $display_name: Exists (not activated)"
            ;;
        fail:*)
            local version=$(echo "$status" | cut -d: -f2-)
            echo -e "  ${RED}❌${NC} $display_name: $version (requires 3.11+)"
            ;;
        missing)
            echo -e "  ${RED}❌${NC} $display_name: Not found"
            ;;
        venv_missing)
            echo -e "  ${YELLOW}⚠️${NC}  $display_name: Virtual environment required"
            ;;
        *)
            if [[ $status == missing:* ]]; then
                local items=$(echo "$status" | cut -d: -f2-)
                echo -e "  ${YELLOW}⚠️${NC}  $display_name: Missing - $items"
            else
                echo -e "  ${YELLOW}⚠️${NC}  $display_name: $status"
            fi
            ;;
    esac
}

# Display environment status
display_status() {
    log_header "GFACS Environment Status"

    # Count totals
    local total_checks=9
    local passed_checks=0
    local failed_count=${#FAILED_CHECKS[@]}
    local warning_count=${#WARNING_CHECKS[@]}

    # Count passed checks
    for check in "${!CHECK_RESULTS[@]}"; do
        case ${CHECK_RESULTS[$check]} in
            pass:*|active|exists)
                ((passed_checks++))
                ;;
        esac
    done

    echo ""
    echo "System Requirements:"
    display_check_status "python_version" "Python Version"
    display_check_status "uv" "uv Package Manager"

    echo ""
    echo "Virtual Environment:"
    display_check_status "virtual_env" "Virtual Environment"

    echo ""
    echo "PyTorch Dependencies:"
    display_check_status "pytorch" "PyTorch"
    display_check_status "torch_geometric" "PyTorch Geometric"

    echo ""
    echo "External Solvers:"
    display_check_status "hgs_cvrp" "HGS-CVRP Solver"
    display_check_status "concorde" "Concorde TSP Solver"

    echo ""
    echo "Data & Dependencies:"
    display_check_status "data_dirs" "Data Directories"
    display_check_status "core_deps" "Core Dependencies"

    echo ""
    echo "Summary: $passed_checks/$total_checks checks passed"

    if [ $failed_count -gt 0 ]; then
        echo -e "${RED}Failed checks: $failed_count${NC}"
        echo "  Critical issues that prevent GFACS from running"
    fi

    if [ $warning_count -gt 0 ]; then
        echo -e "${YELLOW}Warnings: $warning_count${NC}"
        echo "  Optional components that can be installed later"
    fi

    if [ $passed_checks -eq $total_checks ]; then
        echo ""
        echo -e "${GREEN}🎉 Environment is fully configured and ready!${NC}"
    fi

    echo ""
}

# Show main menu
show_main_menu() {
    log_header "GFACS Setup Menu"

    echo "Choose an option:"
    echo ""
    echo "1. 🛠️  Setup Environment"
    echo "   Install Python dependencies and create virtual environment"
    echo ""
    echo "2. 🔧 Setup External Solvers"
    echo "   Build HGS-CVRP and Concorde TSP solvers"
    echo ""
    echo "3. ✅ Verify Installation"
    echo "   Run comprehensive installation verification"
    echo ""
    echo "4. 🚀 Run Orchestrator"
    echo "   Execute GFACS experiments with interactive submenu"
    echo ""
    echo "5. 📊 Show Environment Status"
    echo "   Display current environment check results"
    echo ""
    echo "0. Exit"
    echo ""
    echo -n "Enter your choice [0-5]: "
}

# Show orchestrator submenu
show_orchestrator_menu() {
    log_header "GFACS Orchestrator Menu"

    echo "Choose orchestrator mode:"
    echo ""
    echo "1. ⚡ Quick Test (TSP only)"
    echo "   Fast validation with Traveling Salesman Problem"
    echo ""
    echo "2. 🔬 Full Experiment"
    echo "   Complete run with all 8 problem types"
    echo ""
    echo "3. 🎯 Custom Problems"
    echo "   Select specific problems to run"
    echo ""
    echo "4. 📄 Custom Configuration"
    echo "   Use custom YAML configuration file"
    echo ""
    echo "5. 🔙 Back to Main Menu"
    echo ""
    echo -n "Enter your choice [1-5]: "
}

# Run setup environment
run_setup_environment() {
    log_header "Setting up GFACS Environment"

    local setup_script="$SCRIPT_DIR/setup_env.sh"

    if [ ! -f "$setup_script" ]; then
        log_error "Setup script not found: $setup_script"
        echo "Press Enter to continue..."
        read
        return 1
    fi

    echo "This will:"
    echo "  - Install uv package manager (if not present)"
    echo "  - Create virtual environment (.venv/)"
    echo "  - Install Python dependencies"
    echo ""
    echo -n "Continue? [y/N]: "

    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "Setup cancelled"
        return 0
    fi

    log_info "Running environment setup..."
    echo ""

    # Run setup script and capture output
    if cd "$REPO_ROOT" && bash "$setup_script"; then
        log_success "Environment setup completed successfully!"
        echo ""
        echo -n "Press Enter to re-check environment..."
        read
        return 0
    else
        log_error "Environment setup failed"
        echo ""
        echo -n "Press Enter to continue..."
        read
        return 1
    fi
}

# Run setup solvers
run_setup_solvers() {
    log_header "Setting up External Solvers"

    local setup_script="$SCRIPT_DIR/setup_solvers.sh"

    if [ ! -f "$setup_script" ]; then
        log_error "Setup script not found: $setup_script"
        echo "Press Enter to continue..."
        read
        return 1
    fi

    echo "This will:"
    echo "  - Check for required build tools (cmake, gcc, make)"
    echo "  - Build HGS-CVRP solver (C++ library)"
    echo "  - Build Concorde TSP solver (C executable)"
    echo ""
    echo "Note: This may take several minutes and requires build tools."
    echo ""
    echo -n "Continue? [y/N]: "

    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "Solver setup cancelled"
        return 0
    fi

    log_info "Running solver setup..."
    echo ""

    # Run setup script and capture output
    if cd "$REPO_ROOT" && bash "$setup_script"; then
        log_success "Solver setup completed successfully!"
        echo ""
        echo -n "Press Enter to re-check environment..."
        read
        return 0
    else
        log_error "Solver setup failed"
        echo ""
        echo -n "Press Enter to continue..."
        read
        return 1
    fi
}

# Run verify installation
run_verify_installation() {
    log_header "Verifying GFACS Installation"

    local verify_script="$SCRIPT_DIR/verify_install.py"

    if [ ! -f "$verify_script" ]; then
        log_error "Verification script not found: $verify_script"
        echo "Press Enter to continue..."
        read
        return 1
    fi

    echo "This will run comprehensive installation verification including:"
    echo "  - Python version and dependencies"
    echo "  - PyTorch and PyTorch Geometric"
    echo "  - External solvers (HGS-CVRP, Concorde)"
    echo "  - Module imports and data directories"
    echo ""
    echo -n "Run verification? [Y/n]: "

    read -r response
    if [[ "$response" =~ ^[Nn]$ ]]; then
        log_info "Verification cancelled"
        return 0
    fi

    log_info "Running installation verification..."
    echo ""

    # Run verification script
    if cd "$REPO_ROOT" && python3 "$verify_script"; then
        log_success "Verification completed successfully!"
    else
        log_error "Verification found issues"
    fi

    echo ""
    echo -n "Press Enter to continue..."
    read
}

# Run orchestrator
run_orchestrator() {
    while true; do
        clear
        show_orchestrator_menu
        read -r choice

        case $choice in
            1)
                # Quick test
                log_info "Running quick test (TSP only)..."
                echo ""
                if cd "$REPO_ROOT" && bash "$SCRIPT_DIR/run.sh" --from-menu quick; then
                    log_success "Quick test completed successfully!"
                else
                    log_error "Quick test failed"
                fi
                echo ""
                echo -n "Press Enter to continue..."
                read
                ;;
            2)
                # Full experiment
                log_info "Running full experiment..."
                echo ""
                echo -e "${YELLOW}Warning: This may take 20-60 minutes to complete${NC}"
                echo -n "Continue? [y/N]: "
                read -r confirm
                if [[ "$confirm" =~ ^[Yy]$ ]]; then
                    if cd "$REPO_ROOT" && bash "$SCRIPT_DIR/run.sh" --from-menu full; then
                        log_success "Full experiment completed successfully!"
                    else
                        log_error "Full experiment failed"
                    fi
                else
                    log_info "Full experiment cancelled"
                fi
                echo ""
                echo -n "Press Enter to continue..."
                read
                ;;
            3)
                # Custom problems
                echo ""
                echo "Available problems: tsp_nls, cvrp_nls, cvrptw_nls, bpp, op, pctsp, smtvtp, sop"
                echo -n "Enter problems (space-separated): "
                read -r problems
                if [ -n "$problems" ]; then
                    log_info "Running custom problems: $problems"
                    echo ""
                    if cd "$REPO_ROOT" && bash "$SCRIPT_DIR/run.sh" --from-menu problems --problems "$problems"; then
                        log_success "Custom problems run completed!"
                    else
                        log_error "Custom problems run failed"
                    fi
                else
                    log_info "No problems specified"
                fi
                echo ""
                echo -n "Press Enter to continue..."
                read
                ;;
            4)
                # Custom configuration
                echo ""
                echo -n "Enter configuration file path: "
                read -r config_file
                if [ -n "$config_file" ] && [ -f "$config_file" ]; then
                    log_info "Running with custom config: $config_file"
                    echo ""
                    if cd "$REPO_ROOT" && bash "$SCRIPT_DIR/run.sh" --from-menu custom --config "$config_file"; then
                        log_success "Custom configuration run completed!"
                    else
                        log_error "Custom configuration run failed"
                    fi
                elif [ -n "$config_file" ]; then
                    log_error "Configuration file not found: $config_file"
                else
                    log_info "No configuration file specified"
                fi
                echo ""
                echo -n "Press Enter to continue..."
                read
                ;;
            5)
                # Back to main menu
                return 0
                ;;
            *)
                log_error "Invalid choice. Please select 1-5."
                echo ""
                echo -n "Press Enter to continue..."
                read
                ;;
        esac
    done
}

# Main menu loop
main_menu_loop() {
    while true; do
        clear
        display_status
        show_main_menu
        read -r choice

        case $choice in
            1)
                # Setup environment
                if run_setup_environment; then
                    check_environment
                fi
                ;;
            2)
                # Setup external solvers
                if run_setup_solvers; then
                    check_environment
                fi
                ;;
            3)
                # Verify installation
                run_verify_installation
                ;;
            4)
                # Run orchestrator
                run_orchestrator
                ;;
            5)
                # Show environment status (already shown above)
                echo ""
                echo -n "Press Enter to continue..."
                read
                ;;
            0)
                # Exit
                log_info "Goodbye!"
                exit 0
                ;;
            *)
                log_error "Invalid choice. Please select 0-5."
                echo ""
                echo -n "Press Enter to continue..."
                read
                ;;
        esac
    done
}

# Main function
main() {
    # Check repository
    check_repository

    # Initial environment check
    check_environment

    # Start menu loop
    main_menu_loop
}

# Run main function with all arguments
main "$@"