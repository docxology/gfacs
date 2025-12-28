"""Validation utilities for GFACS.

This module provides comprehensive input validation, error handling,
and data integrity checks for GFACS components.
"""

import os
import numbers
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
import numpy as np

# Optional torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class GFACSValidator:
    """Comprehensive validator for GFACS inputs and configurations."""

    @staticmethod
    def validate_positive_number(value: Any, name: str, allow_zero: bool = False) -> float:
        """Validate that a value is a positive number.

        Args:
            value: Value to validate
            name: Name of the parameter for error messages
            allow_zero: Whether to allow zero values

        Returns:
            Validated float value

        Raises:
            ValidationError: If validation fails
        """
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{name} must be a number, got {type(value).__name__}: {value}")

        if allow_zero and num < 0:
            raise ValidationError(f"{name} must be non-negative, got {num}")
        elif not allow_zero and num <= 0:
            raise ValidationError(f"{name} must be positive, got {num}")

        return num

    @staticmethod
    def validate_integer(value: Any, name: str, min_value: Optional[int] = None,
                        max_value: Optional[int] = None) -> int:
        """Validate that a value is an integer within bounds.

        Args:
            value: Value to validate
            name: Name of the parameter for error messages
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)

        Returns:
            Validated integer value

        Raises:
            ValidationError: If validation fails
        """
        try:
            num = int(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{name} must be an integer, got {type(value).__name__}: {value}")

        if min_value is not None and num < min_value:
            raise ValidationError(f"{name} must be >= {min_value}, got {num}")

        if max_value is not None and num > max_value:
            raise ValidationError(f"{name} must be <= {max_value}, got {num}")

        return num

    @staticmethod
    def validate_probability(value: Any, name: str) -> float:
        """Validate that a value is a probability (0 <= value <= 1).

        Args:
            value: Value to validate
            name: Name of the parameter for error messages

        Returns:
            Validated probability value

        Raises:
            ValidationError: If validation fails
        """
        try:
            prob = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{name} must be a number, got {type(value).__name__}: {value}")

        if not (0 <= prob <= 1):
            raise ValidationError(f"{name} must be in range [0, 1], got {prob}")

        return prob

    @staticmethod
    def validate_tensor(tensor: Any, name: str, expected_shape: Optional[Tuple] = None,
                       expected_dtype: Optional[type] = None, allow_none: bool = False) -> Any:
        """Validate PyTorch tensor properties.

        Args:
            tensor: Tensor to validate
            name: Name of the tensor for error messages
            expected_shape: Expected tensor shape (None for any shape)
            expected_dtype: Expected tensor dtype (None for any dtype)
            allow_none: Whether None is acceptable

        Returns:
            Validated tensor

        Raises:
            ValidationError: If validation fails
        """
        if not HAS_TORCH:
            raise ValidationError("PyTorch not available for tensor validation")

        if tensor is None:
            if allow_none:
                return tensor
            raise ValidationError(f"{name} cannot be None")

        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(f"{name} must be a PyTorch tensor, got {type(tensor).__name__}")

        if expected_shape is not None:
            if len(tensor.shape) != len(expected_shape):
                raise ValidationError(f"{name} has wrong number of dimensions: "
                                    f"expected {len(expected_shape)}, got {len(tensor.shape)}")

            for i, (expected, actual) in enumerate(zip(expected_shape, tensor.shape)):
                if expected is not None and expected != actual:
                    raise ValidationError(f"{name} dimension {i}: expected {expected}, got {actual}")

        if expected_dtype is not None and tensor.dtype != expected_dtype:
            raise ValidationError(f"{name} has wrong dtype: expected {expected_dtype}, got {tensor.dtype}")

        return tensor

    @staticmethod
    def validate_array(array: Any, name: str, expected_shape: Optional[Tuple] = None,
                      expected_dtype: Optional[type] = None, allow_none: bool = False) -> Any:
        """Validate NumPy array properties.

        Args:
            array: Array to validate
            name: Name of the array for error messages
            expected_shape: Expected array shape (None for any shape)
            expected_dtype: Expected array dtype (None for any dtype)
            allow_none: Whether None is acceptable

        Returns:
            Validated array

        Raises:
            ValidationError: If validation fails
        """
        if array is None:
            if allow_none:
                return array
            raise ValidationError(f"{name} cannot be None")

        if not isinstance(array, np.ndarray):
            raise ValidationError(f"{name} must be a NumPy array, got {type(array).__name__}")

        if expected_shape is not None:
            if len(array.shape) != len(expected_shape):
                raise ValidationError(f"{name} has wrong number of dimensions: "
                                    f"expected {len(expected_shape)}, got {len(array.shape)}")

            for i, (expected, actual) in enumerate(zip(expected_shape, array.shape)):
                if expected is not None and expected != actual:
                    raise ValidationError(f"{name} dimension {i}: expected {expected}, got {actual}")

        if expected_dtype is not None and array.dtype != expected_dtype:
            raise ValidationError(f"{name} has wrong dtype: expected {expected_dtype}, got {array.dtype}")

        return array

    @staticmethod
    def validate_distance_matrix(matrix: Any, name: str = "distance_matrix") -> Any:
        """Validate distance matrix properties.

        Args:
            matrix: Distance matrix to validate
            name: Name for error messages

        Returns:
            Validated distance matrix

        Raises:
            ValidationError: If validation fails
        """
        # Validate as tensor or array
        if HAS_TORCH and isinstance(matrix, torch.Tensor):
            matrix = GFACSValidator.validate_tensor(matrix, name, expected_shape=(None, None))
            if matrix.shape[0] != matrix.shape[1]:
                raise ValidationError(f"{name} must be square, got shape {matrix.shape}")
        else:
            matrix = GFACSValidator.validate_array(matrix, name, expected_shape=(None, None))
            if matrix.shape[0] != matrix.shape[1]:
                raise ValidationError(f"{name} must be square, got shape {matrix.shape}")

        # Check for non-negative values (except possibly diagonal)
        if torch.is_tensor(matrix):
            if (matrix < 0).any():
                raise ValidationError(f"{name} contains negative values")
            # Check diagonal is high value (infinity representation)
            diagonal = matrix.diagonal()
            if not (diagonal > 1e6).all():
                raise ValidationError(f"{name} diagonal should contain high values (infinity), "
                                    f"found values: {diagonal}")
        else:
            if (matrix < 0).any():
                raise ValidationError(f"{name} contains negative values")
            diagonal = np.diag(matrix)
            if not (diagonal > 1e6).all():
                raise ValidationError(f"{name} diagonal should contain high values (infinity), "
                                    f"found values: {diagonal}")

        return matrix

    @staticmethod
    def validate_coordinates(coordinates: Any, name: str = "coordinates") -> Any:
        """Validate coordinate array/matrix.

        Args:
            coordinates: Coordinate array to validate
            name: Name for error messages

        Returns:
            Validated coordinates

        Raises:
            ValidationError: If validation fails
        """
        if HAS_TORCH and isinstance(coordinates, torch.Tensor):
            coordinates = GFACSValidator.validate_tensor(coordinates, name, expected_shape=(None, 2))
        else:
            coordinates = GFACSValidator.validate_array(coordinates, name, expected_shape=(None, 2))

        if coordinates.shape[1] != 2:
            raise ValidationError(f"{name} must have shape (n_points, 2), got {coordinates.shape}")

        return coordinates

    @staticmethod
    def validate_file_path(path: Union[str, Path], name: str = "file_path",
                          must_exist: bool = True, writable: bool = False) -> Path:
        """Validate file path.

        Args:
            path: File path to validate
            name: Name for error messages
            must_exist: Whether file must exist
            writable: Whether file must be writable

        Returns:
            Validated Path object

        Raises:
            ValidationError: If validation fails
        """
        try:
            path_obj = Path(path)
        except Exception as e:
            raise ValidationError(f"Invalid {name}: {e}")

        if must_exist and not path_obj.exists():
            raise ValidationError(f"{name} does not exist: {path_obj}")

        if writable:
            try:
                # Check if directory is writable
                parent = path_obj.parent
                if parent.exists():
                    if not os.access(parent, os.W_OK):
                        raise ValidationError(f"{name} parent directory is not writable: {parent}")
                else:
                    # Try to create parent directory
                    parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValidationError(f"Cannot write to {name}: {e}")

        return path_obj

    @staticmethod
    def validate_directory_path(path: Union[str, Path], name: str = "directory",
                               must_exist: bool = True, create_if_missing: bool = False) -> Path:
        """Validate directory path.

        Args:
            path: Directory path to validate
            name: Name for error messages
            must_exist: Whether directory must exist
            create_if_missing: Whether to create directory if missing

        Returns:
            Validated Path object

        Raises:
            ValidationError: If validation fails
        """
        try:
            path_obj = Path(path)
        except Exception as e:
            raise ValidationError(f"Invalid {name}: {e}")

        if create_if_missing and not path_obj.exists():
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValidationError(f"Cannot create {name}: {e}")
        elif must_exist and not path_obj.exists():
            raise ValidationError(f"{name} does not exist: {path_obj}")

        if path_obj.exists() and not path_obj.is_dir():
            raise ValidationError(f"{name} is not a directory: {path_obj}")

        return path_obj

    @staticmethod
    def validate_choice(value: Any, choices: List[Any], name: str) -> Any:
        """Validate that value is one of the allowed choices.

        Args:
            value: Value to validate
            choices: List of allowed values
            name: Name for error messages

        Returns:
            Validated value

        Raises:
            ValidationError: If validation fails
        """
        if value not in choices:
            raise ValidationError(f"{name} must be one of {choices}, got {value}")

        return value

    @staticmethod
    def validate_problem_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate problem configuration dictionary.

        Args:
            config: Problem configuration to validate

        Returns:
            Validated configuration

        Raises:
            ValidationError: If validation fails
        """
        required_keys = ['name', 'size']
        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Problem config missing required key: {key}")

        # Validate problem name
        valid_problems = ['tsp_nls', 'cvrp_nls', 'cvrptw_nls', 'bpp', 'op', 'pctsp', 'smtwtp', 'sop']
        config['name'] = GFACSValidator.validate_choice(config['name'], valid_problems, 'problem.name')

        # Validate problem size
        config['size'] = GFACSValidator.validate_integer(config['size'], 'problem.size', min_value=5, max_value=2000)

        # Validate optional parameters
        if 'enabled' in config:
            if not isinstance(config['enabled'], bool):
                raise ValidationError("problem.enabled must be boolean")

        if 'n_ants' in config:
            config['n_ants'] = GFACSValidator.validate_integer(config['n_ants'], 'problem.n_ants', min_value=1, max_value=1000)

        if 'n_iterations' in config:
            config['n_iterations'] = GFACSValidator.validate_integer(config['n_iterations'], 'problem.n_iterations', min_value=1, max_value=10000)

        if 'device' in config:
            valid_devices = ['cpu', 'cuda']
            if isinstance(config['device'], str):
                config['device'] = GFACSValidator.validate_choice(config['device'], valid_devices, 'problem.device')
            elif HAS_TORCH and isinstance(config['device'], torch.device):
                pass  # Accept torch.device objects
            else:
                raise ValidationError("problem.device must be string or torch.device")

        return config

    @staticmethod
    def validate_experiment_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experiment configuration dictionary.

        Args:
            config: Experiment configuration to validate

        Returns:
            Validated configuration

        Raises:
            ValidationError: If validation fails
        """
        # Validate experiment name
        if 'experiment_name' in config:
            if not isinstance(config['experiment_name'], str) or not config['experiment_name'].strip():
                raise ValidationError("experiment_name must be non-empty string")

        # Validate problems list
        if 'problems' in config:
            if not isinstance(config['problems'], list):
                raise ValidationError("problems must be a list")

            validated_problems = []
            for i, problem in enumerate(config['problems']):
                try:
                    validated_problems.append(GFACSValidator.validate_problem_config(problem))
                except ValidationError as e:
                    raise ValidationError(f"Problem {i}: {e}") from e

            config['problems'] = validated_problems

        # Validate output directory
        if 'base_output_dir' in config:
            config['base_output_dir'] = str(GFACSValidator.validate_directory_path(
                config['base_output_dir'], 'base_output_dir', create_if_missing=True))

        # Validate boolean flags
        bool_flags = ['enable_visualizations', 'enable_animations']
        for flag in bool_flags:
            if flag in config and not isinstance(config[flag], bool):
                raise ValidationError(f"{flag} must be boolean")

        # Validate log level
        if 'log_level' in config:
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            config['log_level'] = GFACSValidator.validate_choice(config['log_level'], valid_levels, 'log_level')

        # Validate parallel execution
        if 'max_parallel_problems' in config:
            config['max_parallel_problems'] = GFACSValidator.validate_integer(
                config['max_parallel_problems'], 'max_parallel_problems', min_value=1, max_value=16)

        # Validate seed
        if 'seed' in config:
            config['seed'] = GFACSValidator.validate_integer(config['seed'], 'seed', min_value=0)

        return config


# Convenience functions for common validations
def validate_positive_int(value: Any, name: str) -> int:
    """Validate positive integer."""
    return GFACSValidator.validate_integer(value, name, min_value=1)


def validate_non_negative_float(value: Any, name: str) -> float:
    """Validate non-negative float."""
    return GFACSValidator.validate_positive_number(value, name, allow_zero=True)


def validate_distance_matrix(matrix: Any) -> Any:
    """Validate distance matrix."""
    return GFACSValidator.validate_distance_matrix(matrix)


def validate_coordinates(coords: Any) -> Any:
    """Validate coordinates."""
    return GFACSValidator.validate_coordinates(coords)


def validate_file_exists(path: Union[str, Path]) -> Path:
    """Validate that file exists."""
    return GFACSValidator.validate_file_path(path, must_exist=True)


def validate_directory_exists(path: Union[str, Path]) -> Path:
    """Validate that directory exists."""
    return GFACSValidator.validate_directory_path(path, must_exist=True)


def validate_experiment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate complete experiment configuration."""
    return GFACSValidator.validate_experiment_config(config)