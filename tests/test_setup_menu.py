"""Integration tests for the GFACS setup menu system."""

import pytest
import subprocess
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestSetupMenuIntegration:
    """Integration tests for setup_menu.sh functionality."""

    def test_setup_menu_exists_and_executable(self):
        """Test that setup_menu.sh exists and is executable."""
        script_path = Path("scripts/setup_menu.sh")
        assert script_path.exists(), "setup_menu.sh should exist"
        assert os.access(script_path, os.X_OK), "setup_menu.sh should be executable"

    def test_setup_menu_help_display(self):
        """Test that setup menu displays help information."""
        script_path = Path("scripts/setup_menu.sh")

        # This would require interactive testing, so we'll just check the script structure
        with open(script_path, 'r') as f:
            content = f.read()

        # Check for key functions
        assert "check_environment()" in content, "Should contain check_environment function"
        assert "display_status()" in content, "Should contain display_status function"
        assert "show_main_menu()" in content, "Should contain show_main_menu function"
        assert "run_setup_environment()" in content, "Should contain run_setup_environment function"

    def test_environment_check_functions_exist(self):
        """Test that all environment check functions are defined."""
        script_path = Path("scripts/setup_menu.sh")

        with open(script_path, 'r') as f:
            content = f.read()

        # Check for individual check functions
        check_functions = [
            "check_python_version",
            "check_uv",
            "check_virtual_env",
            "check_pytorch",
            "check_pytorch_geometric",
            "check_hgs_solver",
            "check_concorde_solver",
            "check_data_directories",
            "check_core_dependencies"
        ]

        for func in check_functions:
            assert f"{func}()" in content, f"Should contain {func} function"

    def test_menu_option_parsing(self):
        """Test that menu options are properly defined."""
        script_path = Path("scripts/setup_menu.sh")

        with open(script_path, 'r') as f:
            content = f.read()

        # Check for menu options
        assert "1. 🛠️  Setup Environment" in content, "Should contain setup environment option"
        assert "2. 🔧 Setup External Solvers" in content, "Should contain setup solvers option"
        assert "3. ✅ Verify Installation" in content, "Should contain verify installation option"
        assert "4. 🚀 Run Orchestrator" in content, "Should contain run orchestrator option"
        assert "5. 📊 Show Environment Status" in content, "Should contain show status option"
        assert "0. Exit" in content, "Should contain exit option"

    def test_orchestrator_submenu_options(self):
        """Test that orchestrator submenu options are defined."""
        script_path = Path("scripts/setup_menu.sh")

        with open(script_path, 'r') as f:
            content = f.read()

        # Check for orchestrator submenu options
        assert "1. ⚡ Quick Test (TSP only)" in content, "Should contain quick test option"
        assert "2. 🔬 Full Experiment" in content, "Should contain full experiment option"
        assert "3. 🎯 Custom Problems" in content, "Should contain custom problems option"
        assert "4. 📄 Custom Configuration" in content, "Should contain custom config option"
        assert "5. 🔙 Back to Main Menu" in content, "Should contain back to main menu option"

    @patch('subprocess.run')
    @patch('builtins.input')
    def test_run_setup_environment_integration(self, mock_input, mock_subprocess):
        """Test integration of run_setup_environment function."""
        # Mock user input to proceed with setup
        mock_input.return_value = "y"

        # Mock subprocess to simulate successful setup
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        # Import and test the function (this would require more complex mocking)
        # For now, just verify the function exists and has proper structure
        script_path = Path("scripts/setup_menu.sh")

        with open(script_path, 'r') as f:
            content = f.read()

        # Check that the function calls the expected setup script
        assert "SCRIPT_DIR/setup_env.sh" in content, "Should call setup_env.sh script"
        assert "bash" in content, "Should execute bash scripts"

    def test_color_output_definitions(self):
        """Test that color codes are properly defined."""
        script_path = Path("scripts/setup_menu.sh")

        with open(script_path, 'r') as f:
            content = f.read()

        # Check for color definitions
        colors = ["RED", "GREEN", "YELLOW", "BLUE", "PURPLE", "CYAN", "NC"]
        for color in colors:
            assert f"{color}=" in content, f"Should define {color} color"

    def test_error_handling_patterns(self):
        """Test that error handling patterns are implemented."""
        script_path = Path("scripts/setup_menu.sh")

        with open(script_path, 'r') as f:
            content = f.read()

        # Check for error handling patterns
        assert "log_error" in content, "Should contain error logging"
        assert "exit 1" in content, "Should contain exit on error"
        assert "Press Enter to continue" in content, "Should prompt user to continue on error"

    def test_repository_validation(self):
        """Test that repository validation is implemented."""
        script_path = Path("scripts/setup_menu.sh")

        with open(script_path, 'r') as f:
            content = f.read()

        # Check for repository checks
        assert "check_repository" in content, "Should contain repository check function"
        assert "pyproject.toml" in content, "Should check for pyproject.toml"

    def test_from_menu_flag_integration(self):
        """Test that --from-menu flag is properly integrated."""
        script_path = Path("scripts/setup_menu.sh")

        with open(script_path, 'r') as f:
            content = f.read()

        # Check that run.sh is called with --from-menu flag
        assert "--from-menu" in content, "Should pass --from-menu flag to run.sh"

    def test_script_header_and_documentation(self):
        """Test that script has proper header and documentation."""
        script_path = Path("scripts/setup_menu.sh")

        with open(script_path, 'r') as f:
            lines = f.readlines()

        # Check header
        assert lines[0].startswith("#!/bin/bash"), "Should start with shebang"
        assert "GFACS Interactive Environment Setup Menu" in ''.join(lines[1:4]), "Should have descriptive header"

        # Check for usage documentation
        content = ''.join(lines)
        assert "This script provides an interactive menu system" in content, "Should have description"

    def test_main_function_structure(self):
        """Test that main function has proper structure."""
        script_path = Path("scripts/setup_menu.sh")

        with open(script_path, 'r') as f:
            content = f.read()

        # Check main function components
        assert "main()" in content, "Should contain main function"
        assert "check_repository" in content, "Should call repository check"
        assert "check_environment" in content, "Should check environment"
        assert "main_menu_loop" in content, "Should start menu loop"

    @pytest.mark.parametrize("menu_option", ["1", "2", "3", "4", "5", "0"])
    def test_menu_option_validation(self, menu_option):
        """Test that menu options are properly handled."""
        # This test would require more complex mocking of the interactive menu
        # For now, just verify the case statement exists
        script_path = Path("scripts/setup_menu.sh")

        with open(script_path, 'r') as f:
            content = f.read()

        # Check that case statement handles all options
        assert f'case $choice in' in content, "Should contain case statement for menu options"

        if menu_option in ["1", "2", "3", "4", "5"]:
            assert f'{menu_option})' in content, f"Should handle option {menu_option}"
        elif menu_option == "0":
            assert '0)' in content, "Should handle exit option"
        else:
            assert '*)' in content, "Should handle invalid options"

    def test_dependency_script_calls(self):
        """Test that the menu calls the correct dependency scripts."""
        script_path = Path("scripts/setup_menu.sh")

        with open(script_path, 'r') as f:
            content = f.read()

        # Check for script calls
        assert "SCRIPT_DIR/setup_env.sh" in content, "Should call setup_env.sh"
        assert "SCRIPT_DIR/setup_solvers.sh" in content, "Should call setup_solvers.sh"
        assert "python3" in content, "Should call python scripts"
        assert "bash \"$SCRIPT_DIR/run.sh\"" in content, "Should call run.sh"

    def test_status_display_formatting(self):
        """Test that status display functions are properly formatted."""
        script_path = Path("scripts/setup_menu.sh")

        with open(script_path, 'r') as f:
            content = f.read()

        # Check for status display functions
        assert "display_check_status()" in content, "Should contain display_check_status function"
        assert "✅" in content, "Should contain success emoji"
        assert "❌" in content, "Should contain error emoji"
        assert "⚠️" in content, "Should contain warning emoji"

    def test_script_modularity(self):
        """Test that the script is properly modularized."""
        script_path = Path("scripts/setup_menu.sh")

        with open(script_path, 'r') as f:
            content = f.read()

        # Count function definitions (rough check for modularity)
        function_count = content.count("() {")
        assert function_count > 10, f"Script should be well-modularized, found {function_count} functions"

        # Check for key architectural functions
        key_functions = [
            "check_environment",
            "display_status",
            "show_main_menu",
            "run_setup_environment",
            "run_setup_solvers",
            "run_verify_installation",
            "run_orchestrator"
        ]

        for func in key_functions:
            assert f"{func}()" in content, f"Should contain {func} function"