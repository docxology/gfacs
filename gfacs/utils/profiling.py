"""Performance profiling utilities for GFACS.

This module provides profiling decorators and utilities for measuring
performance characteristics of algorithms and functions.
"""

import time
import functools
import os
from typing import Dict, Any, Optional, Callable, TypeVar
from contextlib import contextmanager

# Optional imports
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from .logging import get_logger

logger = get_logger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class PerformanceProfiler:
    """Performance profiling utility for tracking execution metrics."""

    def __init__(self, name: str = "profiler"):
        self.name = name
        self.reset()

    def reset(self):
        """Reset all profiling metrics."""
        self.metrics = {
            'calls': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_time': 0.0,
            'memory_peak': 0,
            'memory_avg': 0,
            'memory_samples': [],
        }

    def start_measurement(self) -> Dict[str, Any]:
        """Start a performance measurement."""
        start_time = time.perf_counter()

        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            memory_start = process.memory_info().rss / 1024 / 1024  # MB
        else:
            process = None
            memory_start = 0.0

        return {
            'start_time': start_time,
            'memory_start': memory_start,
            'process': process
        }

    def end_measurement(self, measurement: Dict[str, Any]) -> Dict[str, Any]:
        """End a performance measurement and calculate metrics."""
        end_time = time.perf_counter()
        elapsed_time = end_time - measurement['start_time']

        if HAS_PSUTIL and measurement['process'] is not None:
            process = measurement['process']
            memory_end = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_end - measurement['memory_start']
        else:
            memory_end = 0.0
            memory_used = 0.0

        # Update metrics
        self.metrics['calls'] += 1
        self.metrics['total_time'] += elapsed_time
        self.metrics['min_time'] = min(self.metrics['min_time'], elapsed_time)
        self.metrics['max_time'] = max(self.metrics['max_time'], elapsed_time)
        self.metrics['avg_time'] = self.metrics['total_time'] / self.metrics['calls']

        # Memory tracking (only if psutil available)
        if HAS_PSUTIL:
            self.metrics['memory_samples'].append(memory_used)
            self.metrics['memory_peak'] = max(self.metrics['memory_peak'], memory_end)
            self.metrics['memory_avg'] = sum(self.metrics['memory_samples']) / len(self.metrics['memory_samples'])

        return {
            'elapsed_time': elapsed_time,
            'memory_used': memory_used,
            'memory_end': memory_end
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'name': self.name,
            **self.metrics
        }

    def log_summary(self, level: str = "INFO"):
        """Log performance summary."""
        summary = self.get_summary()
        msg = (f"Performance [{summary['name']}]: "
               f"calls={summary['calls']}, "
               ".3f"
               ".3f"
               ".1f")
        getattr(logger, level.lower())(msg)


def profile_function(func: Optional[F] = None, *, name: Optional[str] = None, log_level: str = "DEBUG") -> F:
    """Decorator to profile function performance.

    Args:
        func: Function to profile
        name: Custom name for profiling (defaults to function name)
        log_level: Logging level for performance metrics

    Returns:
        Decorated function with profiling
    """
    def decorator(func: F) -> F:
        profiler_name = name or f"{func.__module__}.{func.__qualname__}"
        profiler = PerformanceProfiler(profiler_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            measurement = profiler.start_measurement()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                metrics = profiler.end_measurement(measurement)

                # Log individual call performance
                if log_level.lower() != "none":
                    msg = (".3f")
                    getattr(logger, log_level.lower())(msg)

        # Attach profiler to function for external access
        wrapper._profiler = profiler
        return wrapper

    if func is None:
        return decorator
    return decorator(func)


@contextmanager
def profile_context(name: str = "context", log_level: str = "DEBUG"):
    """Context manager for profiling code blocks.

    Args:
        name: Name for the profiling context
        log_level: Logging level for metrics
    """
    profiler = PerformanceProfiler(name)
    measurement = profiler.start_measurement()

    try:
        yield profiler
    finally:
        metrics = profiler.end_measurement(measurement)

        if log_level.lower() != "none":
            msg = (".3f")
            getattr(logger, log_level.lower())(msg)


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information if CUDA is available.

    Returns:
        Dictionary with GPU memory metrics
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return {}

    try:
        # Get current GPU memory usage
        allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        reserved = torch.cuda.memory_reserved() / 1024 / 1024    # MB
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        return {
            'gpu_memory_allocated': allocated,
            'gpu_memory_reserved': reserved,
            'gpu_memory_peak': max_allocated
        }
    except Exception:
        return {}


def profile_gpu_memory(func: Optional[F] = None, *, name: Optional[str] = None) -> F:
    """Decorator to profile GPU memory usage.

    Args:
        func: Function to profile
        name: Custom name for profiling

    Returns:
        Decorated function with GPU memory profiling
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            result = func(*args, **kwargs)

            if HAS_TORCH and torch.cuda.is_available():
                gpu_info = get_gpu_memory_info()
                if gpu_info:
                    profiler_name = name or f"{func.__module__}.{func.__qualname__}"
                    msg = (f"GPU Memory [{profiler_name}]: "
                           ".1f")
                    logger.debug(msg)

            return result
        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def benchmark_function(func: F, *args, iterations: int = 10, warmup: int = 2, **kwargs) -> Dict[str, Any]:
    """Benchmark a function over multiple iterations.

    Args:
        func: Function to benchmark
        *args: Positional arguments for the function
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations
        **kwargs: Keyword arguments for the function

    Returns:
        Dictionary with benchmark results
    """
    func_name = getattr(func, '__name__', str(type(func).__name__))
    profiler = PerformanceProfiler(f"benchmark_{func_name}")

    # Warmup iterations
    for _ in range(warmup):
        func(*args, **kwargs)

    # Benchmark iterations
    for _ in range(iterations):
        measurement = profiler.start_measurement()
        func(*args, **kwargs)
        profiler.end_measurement(measurement)

    return profiler.get_summary()


# Convenience decorators for common use cases
profile_aco = functools.partial(profile_function, log_level="INFO")  # ACO algorithms
profile_nn = functools.partial(profile_function, log_level="DEBUG")   # Neural network operations
profile_io = functools.partial(profile_function, log_level="DEBUG")   # I/O operations
profile_slow = functools.partial(profile_function, log_level="WARNING")  # Slow operations