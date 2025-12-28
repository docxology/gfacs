#!/bin/bash

# ============================================================================
# GFACS Orchestrator Runner
# ============================================================================
#
# This script provides easy access to run GFACS orchestrations with
# pre-configured options for different use cases.
#
# Usage:
#   ./run.sh                    # Interactive menu mode
#   ./run.sh [command] [options] # Command-line mode
#
# Commands:
#   quick      - Run quick test with TSP only (fast validation)
#   full       - Run full experiment with all 8 problems
#   problems   - Run specific problems (requires --problems flag)
#   custom     - Run with custom configuration file
#   help       - Show this help message
#
# Options:
#   --problems "prob1 prob2"  - Space-separated list of problems
#   --config /path/to/config  - Custom configuration file
#   --experiment-name name    - Custom experiment name
#   --output-dir /path/to/dir - Custom output directory
#   --verbose                 - Enable verbose output
#   --dry-run                 - Show what would be executed
#
# Examples:
#   ./run.sh quick
#   ./run.sh full --experiment-name "production_run_v1"
#   ./run.sh problems --problems "tsp_nls cvrp_nls"
#   ./run.sh custom --config config/benchmark.yaml
#
# ============================================================================

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
CONFIG_DIR="$REPO_ROOT/config"
OUTPUTS_DIR="$REPO_ROOT/outputs"

# Default values
COMMAND=""
EXPERIMENT_NAME=""
CONFIG_FILE=""
PROBLEMS=""
OUTPUT_DIR="$OUTPUTS_DIR"
VERBOSE=false
DRY_RUN=false
FROM_MENU=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

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

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1" >&2
    fi
}

log_header() {
    echo -e "${PURPLE}===============================================================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}===============================================================================${NC}"
}

# Show usage information
show_help() {
    cat << EOF
GFACS Orchestrator Runner

USAGE:
    $0                    Interactive menu for setup and execution
    $0 [COMMAND] [OPTIONS] Direct command execution

COMMANDS:
    quick                Run quick test with TSP only (fast validation)
    full                 Run full experiment with all 8 problems
    problems             Run specific problems (use --problems to specify)
    custom               Run with custom configuration file
    help                 Show this help message

OPTIONS:
    --problems "LIST"     Space-separated list of problems to run
    --config FILE         Custom configuration file path
    --experiment-name NAME Custom experiment name
    --output-dir DIR      Custom output directory
    --verbose             Enable verbose output
    --dry-run             Show what would be executed without running
    --from-menu           Suppress verbose output (used by setup menu)

EXAMPLES:
    $0 quick
    $0 full --experiment-name "production_run_v1"
    $0 problems --problems "tsp_nls cvrp_nls"
    $0 custom --config config/benchmark.yaml
    $0 help

ENVIRONMENT:
    This script requires uv package manager and Python dependencies.
    Run 'uv sync' first if dependencies are not installed.

OUTPUT:
    All results are saved to outputs/{experiment_name}_{timestamp}/
    including configs, logs, data, visualizations, and animations.
EOF
}

# Show main interactive menu
show_main_menu() {
    log_header "GFACS Interactive Menu"

    echo "Choose an option:"
    echo ""
    echo "1. 🛠️  Environment Setup"
    echo "   Complete GFACS setup with all dependencies and verification"
    echo ""
    echo "2. 🧪 Test Suite"
    echo "   Run comprehensive test suite with coverage reporting"
    echo ""
    echo "3. 🎯 Run One Example"
    echo "   Select and run a single problem example"
    echo ""
    echo "4. 🚀 Run All Examples"
    echo "   Execute full experiment with all 8 problem types"
    echo ""
    echo "0. Exit"
    echo ""
    echo -n "Enter your choice [0-4]: "
}

# Show problem selection submenu
show_problem_menu() {
    log_header "Select Problem Type"

    echo "Available problem types:"
    echo ""
    echo "1.  TSP (Traveling Salesman Problem)"
    echo "2.  CVRP (Capacitated Vehicle Routing Problem)"
    echo "3.  CVRPTW (CVRP with Time Windows)"
    echo "4.  BPP (Bin Packing Problem)"
    echo "5.  OP (Orieneering Problem)"
    echo "6.  PCTSP (Prize Collecting TSP)"
    echo "7.  SMTWTP (Single Machine Total Weighted Tardiness)"
    echo "8.  SOP (Sequential Ordering Problem)"
    echo ""
    echo "9.  🔙 Back to Main Menu"
    echo ""
    echo -n "Enter your choice [1-9]: "
}

# Check prerequisites
check_prerequisites() {
    log_verbose "Checking prerequisites..."

    # Check if uv is available
    if ! command -v uv &> /dev/null; then
        log_error "uv package manager not found. Please install uv first:"
        log_error "curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi

    # Check if we're in the right directory
    if [ ! -f "pyproject.toml" ]; then
        log_error "pyproject.toml not found. Are you in the GFACS repository root?"
        exit 1
    fi

    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        log_warning "Virtual environment not found. Installing dependencies..."
        if [ "$DRY_RUN" = false ]; then
            uv sync
        fi
    fi

    log_verbose "Prerequisites check passed"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            quick|full|problems|custom|help)
                if [ -n "$COMMAND" ]; then
                    log_error "Multiple commands specified. Use only one command."
                    exit 1
                fi
                COMMAND="$1"
                ;;
            --problems)
                shift
                PROBLEMS="$1"
                ;;
            --config)
                shift
                CONFIG_FILE="$1"
                ;;
            --experiment-name)
                shift
                EXPERIMENT_NAME="$1"
                ;;
            --output-dir)
                shift
                OUTPUT_DIR="$1"
                ;;
            --verbose)
                VERBOSE=true
                ;;
            --dry-run)
                DRY_RUN=true
                ;;
            --from-menu)
                FROM_MENU=true
                ;;
            *)
                log_error "Unknown option: $1"
                echo ""
                show_help
                exit 1
                ;;
        esac
        shift
    done
}

# Validate arguments
validate_args() {
    # Set default command if none specified
    if [ -z "$COMMAND" ]; then
        COMMAND="quick"
        log_info "No command specified, defaulting to 'quick'"
    fi

    # Validate command-specific requirements
    case $COMMAND in
        problems)
            if [ -z "$PROBLEMS" ]; then
                log_error "Command 'problems' requires --problems flag"
                exit 1
            fi
            ;;
        custom)
            if [ -z "$CONFIG_FILE" ]; then
                log_error "Command 'custom' requires --config flag"
                exit 1
            fi
            if [ ! -f "$CONFIG_FILE" ]; then
                log_error "Configuration file not found: $CONFIG_FILE"
                exit 1
            fi
            ;;
    esac

    # Set default experiment name if not provided
    if [ -z "$EXPERIMENT_NAME" ]; then
        case $COMMAND in
            quick)
                EXPERIMENT_NAME="quick_test"
                ;;
            full)
                EXPERIMENT_NAME="full_experiment"
                ;;
            problems)
                EXPERIMENT_NAME="custom_problems"
                ;;
            custom)
                EXPERIMENT_NAME="custom_config"
                ;;
        esac
    fi
}

# Build orchestrator command
build_command() {
    local cmd="uv run gfacs-orchestrator"

    # Add experiment name
    cmd="$cmd --experiment-name \"$EXPERIMENT_NAME\""

    # Add output directory
    cmd="$cmd --output-dir \"$OUTPUT_DIR\""

    # Add command-specific arguments
    case $COMMAND in
        quick)
            cmd="$cmd --quick"
            ;;
        full)
            # No additional args needed for full run
            ;;
        problems)
            cmd="$cmd --problems $PROBLEMS"
            ;;
        custom)
            cmd="$cmd --config \"$CONFIG_FILE\""
            ;;
    esac

    echo "$cmd"
}

# Execute orchestrator
run_orchestrator() {
    local cmd="$1"

    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN - Would execute:"
        echo "$cmd"
        return 0
    fi

    log_info "Starting GFACS orchestrator..."
    log_verbose "Command: $cmd"

    # Execute the command
    if eval "$cmd"; then
        if [ "$FROM_MENU" = false ]; then
            log_success "Orchestrator completed successfully!"
            log_info "Check the output directory for results:"
            log_info "  $OUTPUT_DIR/${EXPERIMENT_NAME}_*/"
        fi
    else
        local exit_code=$?
        if [ "$FROM_MENU" = false ]; then
            log_error "Orchestrator failed with exit code $exit_code"
            log_info "Check logs in the output directory for details"
        fi
        exit $exit_code
    fi
}

# Show summary
show_summary() {
    # Skip summary when called from menu to avoid redundant output
    if [ "$FROM_MENU" = true ]; then
        return
    fi

    log_info "Run Summary:"
    echo "  Command: $COMMAND"
    echo "  Experiment: $EXPERIMENT_NAME"
    echo "  Output Directory: $OUTPUT_DIR"

    case $COMMAND in
        quick)
            echo "  Description: Quick test with TSP only"
            ;;
        full)
            echo "  Description: Full experiment with all 8 problems"
            ;;
        problems)
            echo "  Description: Custom problems - $PROBLEMS"
            ;;
        custom)
            echo "  Description: Custom config - $CONFIG_FILE"
            ;;
    esac

    if [ "$DRY_RUN" = true ]; then
        echo "  Mode: Dry run (no execution)"
    else
        echo "  Mode: Live execution"
    fi
}

# Menu handler functions
run_environment_setup() {
    log_header "Environment Setup"

    local setup_script="$REPO_ROOT/scripts/setup_env.sh"

    if [ ! -f "$setup_script" ]; then
        log_error "Setup script not found: $setup_script"
        echo ""
        echo -n "Press Enter to continue..."
        read
        return 1
    fi

    echo "This will perform a complete GFACS setup:"
    echo "  - Check Python version and uv installation"
    echo "  - Create/update virtual environment (.venv/)"
    echo "  - Install Python dependencies (PyTorch, PyTorch Geometric)"
    echo "  - Optionally install external solvers (Concorde, HGS-CVRP)"
    echo "  - Run installation verification"
    echo ""
    echo -n "Continue? [y/N]: "

    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "Environment setup cancelled"
        return 0
    fi

    log_info "Running environment setup..."
    echo ""

    if cd "$REPO_ROOT" && bash "$setup_script"; then
        log_success "Environment setup completed successfully!"
        echo ""
        echo -n "Press Enter to continue..."
        read
    else
        log_error "Environment setup failed"
        echo ""
        echo -n "Press Enter to continue..."
        read
        return 1
    fi
}

run_test_suite() {
    log_header "Test Suite Execution"

    echo "Test suite options:"
    echo ""
    echo "1. Run all tests"
    echo "2. Run tests with coverage"
    echo "3. Run specific test module"
    echo ""
    echo -n "Enter your choice [1-3]: "

    read -r choice

    local pytest_cmd="uv run pytest"

    case $choice in
        1)
            pytest_cmd="$pytest_cmd -v"
            ;;
        2)
            pytest_cmd="$pytest_cmd --cov=gfacs --cov-report=html --cov-report=term"
            ;;
        3)
            echo ""
            echo -n "Enter test module (e.g., test_orchestrator, test_tsp_nls): "
            read -r module
            if [ -n "$module" ]; then
                pytest_cmd="$pytest_cmd tests/$module/ -v"
            else
                log_info "No module specified, running all tests"
                pytest_cmd="$pytest_cmd -v"
            fi
            ;;
        *)
            log_info "Invalid choice, running all tests"
            pytest_cmd="$pytest_cmd -v"
            ;;
    esac

    log_info "Running test suite..."
    echo ""

    if cd "$REPO_ROOT" && eval "$pytest_cmd"; then
        log_success "Test suite completed successfully!"
    else
        local exit_code=$?
        log_error "Test suite failed with exit code $exit_code"
    fi

    echo ""
    echo -n "Press Enter to continue..."
    read
}

select_problem() {
    while true; do
        # Only clear if we have a terminal
        if [ -t 1 ]; then
            clear
        fi
        show_problem_menu
        read -r choice

        case $choice in
            1) echo "tsp_nls"; return 0 ;;
            2) echo "cvrp_nls"; return 0 ;;
            3) echo "cvrptw_nls"; return 0 ;;
            4) echo "bpp"; return 0 ;;
            5) echo "op"; return 0 ;;
            6) echo "pctsp"; return 0 ;;
            7) echo "smtwtp"; return 0 ;;
            8) echo "sop"; return 0 ;;
            9) return 1 ;;  # Back to main menu
            *)
                log_error "Invalid choice. Please select 1-9."
                echo ""
                echo -n "Press Enter to continue..."
                read
                ;;
        esac
    done
}

run_all_examples() {
    log_header "Run All Examples"

    echo -e "${YELLOW}Warning: This will run experiments on all 8 problem types.${NC}"
    echo "This may take 20-60 minutes to complete depending on your system."
    echo ""
    echo -n "Continue? [y/N]: "

    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "Full experiment cancelled"
        return 0
    fi

    log_info "Starting full experiment with all problem types..."
    echo ""

    # Set up orchestrator command
    COMMAND="full"
    EXPERIMENT_NAME="full_experiment"
    OUTPUT_DIR="$OUTPUTS_DIR"

    validate_args
    local cmd
    cmd=$(build_command)

    if run_orchestrator "$cmd"; then
        log_success "All examples completed successfully!"
    else
        log_error "Some examples failed - check logs for details"
    fi

    echo ""
    echo -n "Press Enter to continue..."
    read
}

# Main menu loop
run_menu_loop() {
    while true; do
        # Only clear if we have a terminal
        if [ -t 1 ]; then
            clear
        fi
        show_main_menu
        read -r choice

        case $choice in
            1)
                # Environment setup
                run_environment_setup
                ;;
            2)
                # Test suite
                run_test_suite
                ;;
            3)
                # Run one example
                local selected_problem
                selected_problem=$(select_problem)

                if [ -n "$selected_problem" ]; then
                    log_header "Run Single Example: $selected_problem"

                    # Set up orchestrator command for single problem
                    COMMAND="problems"
                    PROBLEMS="$selected_problem"
                    EXPERIMENT_NAME="single_${selected_problem}"
                    OUTPUT_DIR="$OUTPUTS_DIR"

                    validate_args
                    local cmd
                    cmd=$(build_command)

                    if run_orchestrator "$cmd"; then
                        log_success "Example $selected_problem completed successfully!"
                    else
                        log_error "Example $selected_problem failed"
                    fi

                    echo ""
                    echo -n "Press Enter to continue..."
                    read
                fi
                ;;
            4)
                # Run all examples
                run_all_examples
                ;;
            0)
                # Exit
                log_info "Goodbye!"
                exit 0
                ;;
            *)
                log_error "Invalid choice. Please select 0-4."
                echo ""
                echo -n "Press Enter to continue..."
                read
                ;;
        esac
    done
}

# Main execution
main() {
    # Check if arguments are provided
    if [ $# -eq 0 ]; then
        # No arguments provided, show interactive menu
        run_menu_loop
    else
        # Arguments provided, use command-line interface
        # Parse command line arguments
        parse_args "$@"

        # Handle help command
        if [ "$COMMAND" = "help" ]; then
            show_help
            exit 0
        fi

        # Validate arguments
        validate_args

        # Check prerequisites
        check_prerequisites

        # Show summary
        show_summary

        # Build and execute command
        local cmd
        cmd=$(build_command)

        echo ""
        run_orchestrator "$cmd"
    fi
}

# Run main function with all arguments
main "$@"
