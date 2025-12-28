"""Utility modules for GFACS experiments.

This package contains utilities for:
- Experiment I/O management
- Visualization and animation generation
- Structured logging
- Configuration management
"""

from .logging import get_logger, setup_experiment_logging
from .io import get_io_manager, setup_experiment_io
from .profiling import (
    profile_function, profile_context, profile_gpu_memory,
    PerformanceProfiler, benchmark_function,
    profile_aco, profile_nn, profile_io, profile_slow
)
from .validation import (
    GFACSValidator, ValidationError,
    validate_positive_int, validate_non_negative_float,
    validate_distance_matrix, validate_coordinates,
    validate_file_exists, validate_directory_exists,
    validate_experiment_config
)

# Optional imports for visualization and animation modules
try:
    from .visualization import get_visualizer, save_experiment_visualizations
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

try:
    from .animations import create_animation, get_animator
    _HAS_ANIMATIONS = True
except ImportError:
    _HAS_ANIMATIONS = False

# Build __all__ based on available modules
__all__ = [
    "get_logger",
    "setup_experiment_logging",
    "get_io_manager",
    "setup_experiment_io",
    "profile_function",
    "profile_context",
    "profile_gpu_memory",
    "PerformanceProfiler",
    "benchmark_function",
    "profile_aco",
    "profile_nn",
    "profile_io",
    "profile_slow",
    "GFACSValidator",
    "ValidationError",
    "validate_positive_int",
    "validate_non_negative_float",
    "validate_distance_matrix",
    "validate_coordinates",
    "validate_file_exists",
    "validate_directory_exists",
    "validate_experiment_config",
]

if _HAS_VISUALIZATION:
    __all__.extend([
        "get_visualizer",
        "save_experiment_visualizations",
    ])

if _HAS_ANIMATIONS:
    __all__.extend(["create_animation", "get_animator"])
