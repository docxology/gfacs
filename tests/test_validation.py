"""Tests for GFACS validation utilities."""

import pytest
import numpy as np
from pathlib import Path

# Optional imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from gfacs.utils import (
        GFACSValidator, ValidationError,
        validate_positive_int, validate_non_negative_float,
        validate_distance_matrix, validate_coordinates,
        validate_file_exists, validate_directory_exists,
        validate_experiment_config
    )
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False
    GFACSValidator = None
    ValidationError = Exception


@pytest.mark.skipif(not HAS_VALIDATION, reason="Validation utilities not available")
class TestGFACSValidator:
    """Test GFACS validation utilities."""

    def test_validate_positive_number(self):
        """Test positive number validation."""
        # Valid cases
        assert GFACSValidator.validate_positive_number(5.0, "test") == 5.0
        assert GFACSValidator.validate_positive_number(1, "test") == 1.0
        assert GFACSValidator.validate_positive_number("3.14", "test") == 3.14

        # Invalid cases
        with pytest.raises(ValidationError):
            GFACSValidator.validate_positive_number(0, "test")

        with pytest.raises(ValidationError):
            GFACSValidator.validate_positive_number(-1, "test")

        with pytest.raises(ValidationError):
            GFACSValidator.validate_positive_number("invalid", "test")

    def test_validate_positive_number_allow_zero(self):
        """Test positive number validation with zero allowed."""
        assert GFACSValidator.validate_positive_number(0, "test", allow_zero=True) == 0.0
        assert GFACSValidator.validate_positive_number(5.0, "test", allow_zero=True) == 5.0

        with pytest.raises(ValidationError):
            GFACSValidator.validate_positive_number(-1, "test", allow_zero=True)

    def test_validate_integer(self):
        """Test integer validation with bounds."""
        # Valid cases
        assert GFACSValidator.validate_integer(5, "test") == 5
        assert GFACSValidator.validate_integer("10", "test") == 10
        assert GFACSValidator.validate_integer(5, "test", min_value=0, max_value=10) == 5

        # Invalid cases
        with pytest.raises(ValidationError):
            GFACSValidator.validate_integer(5, "test", min_value=10)

        with pytest.raises(ValidationError):
            GFACSValidator.validate_integer(15, "test", max_value=10)

        with pytest.raises(ValidationError):
            GFACSValidator.validate_integer("invalid", "test")

    def test_validate_probability(self):
        """Test probability validation."""
        assert GFACSValidator.validate_probability(0.5, "test") == 0.5
        assert GFACSValidator.validate_probability(0, "test") == 0.0
        assert GFACSValidator.validate_probability(1, "test") == 1.0

        with pytest.raises(ValidationError):
            GFACSValidator.validate_probability(-0.1, "test")

        with pytest.raises(ValidationError):
            GFACSValidator.validate_probability(1.1, "test")

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_validate_tensor(self):
        """Test tensor validation."""
        # Valid tensor
        tensor = torch.randn(5, 5)
        result = GFACSValidator.validate_tensor(tensor, "test")
        assert torch.equal(result, tensor)

        # Wrong shape
        with pytest.raises(ValidationError):
            GFACSValidator.validate_tensor(tensor, "test", expected_shape=(3, 3))

        # None when not allowed
        with pytest.raises(ValidationError):
            GFACSValidator.validate_tensor(None, "test")

        # None when allowed
        assert GFACSValidator.validate_tensor(None, "test", allow_none=True) is None

    def test_validate_array(self):
        """Test array validation."""
        # Valid array
        array = np.random.randn(5, 5)
        result = GFACSValidator.validate_array(array, "test")
        assert np.array_equal(result, array)

        # Wrong shape
        with pytest.raises(ValidationError):
            GFACSValidator.validate_array(array, "test", expected_shape=(3, 3))

        # None when not allowed
        with pytest.raises(ValidationError):
            GFACSValidator.validate_array(None, "test")

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_validate_distance_matrix(self):
        """Test distance matrix validation."""
        # Create valid distance matrix
        n_nodes = 5
        coordinates = torch.rand(n_nodes, 2)
        diff = coordinates.unsqueeze(0) - coordinates.unsqueeze(1)
        distances = torch.sqrt((diff ** 2).sum(dim=-1))
        distances.fill_diagonal_(1e9)

        result = validate_distance_matrix(distances)
        assert torch.equal(result, distances)

        # Invalid: negative values
        invalid_distances = distances.clone()
        invalid_distances[0, 1] = -1
        with pytest.raises(ValidationError):
            validate_distance_matrix(invalid_distances)

        # Invalid: not square
        invalid_distances = torch.rand(3, 4)
        with pytest.raises(ValidationError):
            validate_distance_matrix(invalid_distances)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_validate_coordinates(self):
        """Test coordinate validation."""
        # Valid coordinates
        coords = torch.rand(10, 2)
        result = validate_coordinates(coords)
        assert torch.equal(result, coords)

        # Wrong dimensions
        invalid_coords = torch.rand(10, 3)
        with pytest.raises(ValidationError):
            validate_coordinates(invalid_coords)

        invalid_coords = torch.rand(10,)
        with pytest.raises(ValidationError):
            validate_coordinates(invalid_coords)

    def test_validate_file_path(self, tmp_path):
        """Test file path validation."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        # Valid file
        result = GFACSValidator.validate_file_path(str(test_file))
        assert result == test_file

        # Non-existent file
        with pytest.raises(ValidationError):
            GFACSValidator.validate_file_path(str(tmp_path / "nonexistent.txt"))

    def test_validate_directory_path(self, tmp_path):
        """Test directory path validation."""
        # Valid directory
        result = GFACSValidator.validate_directory_path(str(tmp_path))
        assert result == tmp_path

        # Non-existent directory
        nonexistent = tmp_path / "nonexistent"
        with pytest.raises(ValidationError):
            GFACSValidator.validate_directory_path(str(nonexistent), must_exist=True)

        # Create if missing
        result = GFACSValidator.validate_directory_path(str(nonexistent), create_if_missing=True)
        assert result == nonexistent
        assert nonexistent.exists()

    def test_validate_choice(self):
        """Test choice validation."""
        choices = ["option1", "option2", "option3"]

        assert GFACSValidator.validate_choice("option1", choices, "test") == "option1"
        assert GFACSValidator.validate_choice("option2", choices, "test") == "option2"

        with pytest.raises(ValidationError):
            GFACSValidator.validate_choice("invalid", choices, "test")

    def test_validate_problem_config(self):
        """Test problem configuration validation."""
        # Valid config
        config = {
            "name": "tsp_nls",
            "size": 100,
            "enabled": True,
            "n_ants": 50,
            "n_iterations": 100
        }
        result = GFACSValidator.validate_problem_config(config)
        assert result["name"] == "tsp_nls"
        assert result["size"] == 100

        # Invalid problem name
        invalid_config = config.copy()
        invalid_config["name"] = "invalid_problem"
        with pytest.raises(ValidationError):
            GFACSValidator.validate_problem_config(invalid_config)

        # Invalid size
        invalid_config = config.copy()
        invalid_config["size"] = -1
        with pytest.raises(ValidationError):
            GFACSValidator.validate_problem_config(invalid_config)

    def test_validate_experiment_config(self):
        """Test experiment configuration validation."""
        config = {
            "experiment_name": "test_experiment",
            "problems": [
                {
                    "name": "tsp_nls",
                    "size": 50,
                    "enabled": True
                }
            ],
            "base_output_dir": "outputs",
            "enable_visualizations": True,
            "enable_animations": False,
            "log_level": "INFO",
            "max_parallel_problems": 1,
            "seed": 42
        }

        result = validate_experiment_config(config)
        assert result["experiment_name"] == "test_experiment"
        assert len(result["problems"]) == 1

        # Invalid log level
        invalid_config = config.copy()
        invalid_config["log_level"] = "INVALID"
        with pytest.raises(ValidationError):
            validate_experiment_config(invalid_config)

    def test_convenience_functions(self):
        """Test convenience validation functions."""
        assert validate_positive_int(5, "test") == 5
        assert validate_non_negative_float(0.0, "test") == 0.0

        with pytest.raises(ValidationError):
            validate_positive_int(0, "test")

        with pytest.raises(ValidationError):
            validate_non_negative_float(-1.0, "test")