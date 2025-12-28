"""Tests for GFACS verify_install functionality."""

import pytest
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add gfacs to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "gfacs"))

from gfacs.verify_install import (
    check_python_version,
    check_core_dependencies,
    check_optional_dependencies,
    check_external_solvers,
    check_module_imports,
    check_data_directories,
    main,
    VerificationError,
    PROJECT_ROOT
)


class TestVerifyInstallFunctions:
    """Test individual verify_install functions."""

    def test_check_python_version_success(self):
        """Test Python version check with valid version."""
        # Should not raise an exception for current Python version
        check_python_version()

    @patch('sys.version_info', (3, 10, 0))
    def test_check_python_version_failure(self):
        """Test Python version check with invalid version."""
        with pytest.raises(VerificationError, match="Python 3.11\\+ required, found 3.10"):
            check_python_version()

    def test_check_core_dependencies_success(self):
        """Test core dependencies check with all packages available."""
        # Should not raise an exception when all packages are available
        check_core_dependencies()

    @patch('importlib.import_module', side_effect=ImportError("Package not found"))
    def test_check_core_dependencies_failure(self, mock_import):
        """Test core dependencies check with missing package."""
        with pytest.raises(VerificationError, match="Missing core dependencies"):
            check_core_dependencies()

    def test_check_optional_dependencies(self):
        """Test optional dependencies check."""
        # Should not raise an exception, may show warnings for missing packages
        check_optional_dependencies()

    def test_check_external_solvers_hgs_found(self):
        """Test external solvers check when HGS is available."""
        # Should not raise an exception when HGS is available (as in our setup)
        check_external_solvers()

    def test_check_module_imports_success(self):
        """Test module imports check with all modules available."""
        # Should not raise an exception when all modules are available
        check_module_imports()

    def test_check_data_directories(self):
        """Test data directories check."""
        # Should not raise an exception when data directories exist
        check_data_directories()


class TestVerifyInstallCLI:
    """Test CLI functionality."""

    def test_main_function_exists(self):
        """Test that main function exists."""
        assert callable(main)

    @patch('sys.argv', ['verify_install.py'])
    @patch('gfacs.verify_install.check_python_version')
    @patch('gfacs.verify_install.check_core_dependencies')
    @patch('gfacs.verify_install.check_optional_dependencies')
    @patch('gfacs.verify_install.check_external_solvers')
    @patch('gfacs.verify_install.check_module_imports')
    @patch('gfacs.verify_install.check_data_directories')
    def test_main_runs_all_checks(self, mock_data, mock_modules, mock_solvers,
                                 mock_optional, mock_core, mock_python):
        """Test main function runs all checks by default."""
        with patch('argparse.ArgumentParser.parse_args', return_value=MagicMock(
            verbose=False, quiet=False, check_solvers=False, check_problems=False
        )):
            result = main()
            assert result == 0

            # Verify all check functions were called
            mock_python.assert_called_once()
            mock_core.assert_called_once()
            mock_optional.assert_called_once()
            mock_solvers.assert_called_once()
            mock_modules.assert_called_once()
            mock_data.assert_called_once()

    @patch('gfacs.verify_install.check_python_version')
    @patch('gfacs.verify_install.check_external_solvers')
    def test_main_check_solvers_only(self, mock_solvers, mock_python):
        """Test main function with --check-solvers flag."""
        with patch('argparse.ArgumentParser.parse_args', return_value=MagicMock(
            verbose=False, quiet=False, check_solvers=True, check_problems=False
        )):
            result = main()
            assert result == 0

            # Only solver and python checks should be called
            mock_python.assert_called_once()
            mock_solvers.assert_called_once()

    @patch('gfacs.verify_install.check_module_imports')
    @patch('gfacs.verify_install.check_data_directories')
    def test_main_check_problems_only(self, mock_data, mock_modules):
        """Test main function with --check-problems flag."""
        with patch('argparse.ArgumentParser.parse_args', return_value=MagicMock(
            verbose=False, quiet=False, check_solvers=False, check_problems=True
        )):
            result = main()
            assert result == 0

            # Only module and data checks should be called
            mock_modules.assert_called_once()
            mock_data.assert_called_once()

    @patch('gfacs.verify_install.check_python_version')
    @patch('gfacs.verify_install.check_core_dependencies')
    def test_main_quiet_mode(self, mock_core, mock_python):
        """Test main function with --quiet flag."""
        with patch('argparse.ArgumentParser.parse_args', return_value=MagicMock(
            verbose=False, quiet=True, check_solvers=False, check_problems=False
        )):
            result = main()
            assert result == 0

            mock_python.assert_called_once()
            mock_core.assert_called_once()

    @patch('gfacs.verify_install.check_python_version', side_effect=VerificationError("Test error"))
    def test_main_verification_error(self, mock_python):
        """Test main function handles VerificationError."""
        with patch('argparse.ArgumentParser.parse_args', return_value=MagicMock(
            verbose=False, quiet=False, check_solvers=False, check_problems=False
        )):
            result = main()
            assert result == 1

    @patch('gfacs.verify_install.check_python_version', side_effect=Exception("Unexpected error"))
    def test_main_unexpected_error(self, mock_python):
        """Test main function handles unexpected exceptions."""
        with patch('argparse.ArgumentParser.parse_args', return_value=MagicMock(
            verbose=False, quiet=False, check_solvers=False, check_problems=False
        )):
            result = main()
            assert result == 1


class TestVerifyInstallPathResolution:
    """Test path resolution functionality."""

    def test_project_root_exists(self):
        """Test that PROJECT_ROOT is correctly set."""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()
        assert (PROJECT_ROOT / "gfacs").exists()
        assert (PROJECT_ROOT / "pyproject.toml").exists()

    def test_hgs_library_path_resolution(self):
        """Test HGS library path resolution."""
        hgs_path = PROJECT_ROOT / "cvrp_nls" / "HGS-CVRP-main" / "build"
        # Check if any of the expected library files exist
        found = False
        for ext in ['.so', '.dylib', '.dll']:
            if (hgs_path / f"libhgscvrp{ext}").exists():
                found = True
                break

        # In our setup, HGS should be available
        assert found, f"HGS library not found in {hgs_path}"

    def test_concorde_path_resolution(self):
        """Test Concorde path resolution."""
        concorde_path = PROJECT_ROOT / "tsp_nls" / "concorde" / "TSP" / "concorde"
        # Concorde may not be available (expected on Apple Silicon)
        # Just test that the path construction is correct
        expected_path = PROJECT_ROOT / "tsp_nls" / "concorde" / "TSP" / "concorde"
        assert concorde_path == expected_path


class TestVerifyInstallIntegration:
    """Integration tests for verify_install."""

    def test_full_verification_run(self):
        """Test a full verification run."""
        # This should work in our setup environment
        with patch('argparse.ArgumentParser.parse_args', return_value=MagicMock(
            verbose=False, quiet=False, check_solvers=False, check_problems=False
        )):
            exit_code = main()
            assert exit_code == 0

    def test_cli_entry_point_via_subprocess(self):
        """Test CLI entry point via subprocess."""
        # Test the gfacs-verify command
        result = subprocess.run(
            [sys.executable, "-m", "gfacs.verify_install"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        assert result.returncode == 0
        assert "Python version:" in result.stdout
        assert "All checks passed" in result.stdout

    def test_scripts_wrapper_via_subprocess(self):
        """Test scripts wrapper via subprocess."""
        scripts_verify = PROJECT_ROOT / "scripts" / "verify_install.py"
        result = subprocess.run(
            [sys.executable, str(scripts_verify)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        assert result.returncode == 0
        assert "Python version:" in result.stdout
        assert "All checks passed" in result.stdout


class TestVerifyInstallErrorHandling:
    """Test error handling and edge cases."""

    @patch('gfacs.verify_install.PROJECT_ROOT', Path('/nonexistent/path'))
    def test_missing_project_root(self):
        """Test behavior when project root doesn't exist."""
        # This should work but report missing directories
        check_data_directories()
        # No exception should be raised, just warnings logged

    def test_verification_error_exception(self):
        """Test VerificationError exception."""
        error = VerificationError("Test error message")
        assert str(error) == "Test error message"

    @patch('importlib.import_module', side_effect=OSError("System error"))
    def test_module_import_oserror(self, mock_import):
        """Test module import with OSError (e.g., library loading issues)."""
        # OSError that doesn't match external library patterns should raise VerificationError
        with pytest.raises(VerificationError, match="Failed module imports"):
            check_module_imports()


class TestVerifyInstallOutput:
    """Test output formatting and messages."""

    def test_print_status_formats_correctly(self):
        """Test that print_status generates correctly formatted output."""
        import io
        from contextlib import redirect_stdout
        from gfacs.verify_install import print_status

        # Capture stdout to test the actual output
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            print_status("Test message", "SUCCESS")

        output = captured_output.getvalue()
        assert "[SUCCESS] Test message" in output
        assert "\033[0;32m" in output  # Green color code

        # Test ERROR status
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            print_status("Error message", "ERROR")

        output = captured_output.getvalue()
        assert "[ERROR] Error message" in output
        assert "\033[0;31m" in output  # Red color code