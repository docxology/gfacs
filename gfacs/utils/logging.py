"""Logging utilities for GFACS experiments.

This module provides structured logging with file rotation,
console output, and performance metrics tracking for experiments.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
import datetime


class GFACSLogger:
    """Structured logger for GFACS experiments.

    Provides:
    - Console logging with colors
    - File logging with rotation
    - JSON structured logging for analysis
    - Performance metrics logging
    - Memory usage tracking
    """

    def __init__(
        self,
        name: str = "gfacs",
        log_dir: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        enable_file_logging: bool = True,
        enable_json_logging: bool = True,
        max_bytes: int = 10*1024*1024,  # 10MB
        backup_count: int = 5,
    ):
        """Initialize GFACS logger.

        Args:
            name: Logger name
            log_dir: Directory for log files (default: logs/)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_file_logging: Whether to log to files
            enable_json_logging: Whether to log in JSON format
            max_bytes: Maximum log file size before rotation
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.enable_file_logging = enable_file_logging
        self.enable_json_logging = enable_json_logging
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Add console handler
        self._add_console_handler()

        # Add file handlers if enabled
        if enable_file_logging:
            self._setup_file_logging()

        # Add JSON handler if enabled
        if enable_json_logging:
            self._add_json_handler()

    def _add_console_handler(self) -> None:
        """Add colored console handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)

        # Colored formatter
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[36m',    # Cyan
                'INFO': '\033[32m',     # Green
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',    # Red
                'CRITICAL': '\033[35m', # Magenta
                'RESET': '\033[0m'      # Reset
            }

            def format(self, record):
                color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
                reset = self.COLORS['RESET']
                record.levelname = f"{color}{record.levelname}{reset}"
                return super().format(record)

        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _setup_file_logging(self) -> None:
        """Setup file logging with rotation."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file with rotation
        main_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        main_handler.setLevel(self.log_level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        main_handler.setFormatter(formatter)
        self.logger.addHandler(main_handler)

        # Error log file (only errors and above)
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_error.log",
            maxBytes=self.max_bytes // 2,  # Smaller for errors
            backupCount=self.backup_count
        )
        error_handler.setLevel(logging.ERROR)

        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_handler)

    def _add_json_handler(self) -> None:
        """Add JSON structured logging handler."""
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}.jsonl",
            maxBytes=self.max_bytes * 2,  # Larger for JSON
            backupCount=self.backup_count
        )
        json_handler.setLevel(self.log_level)

        # JSON formatter
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                import json
                import datetime

                log_entry = {
                    'timestamp': datetime.datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno,
                }

                # Add any extra fields from the record
                if hasattr(record, 'extra_data'):
                    log_entry.update(record.extra_data)

                return json.dumps(log_entry)

        json_formatter = JSONFormatter()
        json_handler.setFormatter(json_formatter)
        self.logger.addHandler(json_handler)

    def log_performance_metrics(
        self,
        metrics: Dict[str, Any],
        prefix: str = ""
    ) -> None:
        """Log performance metrics.

        Args:
            metrics: Dictionary of metric names and values
            prefix: Optional prefix for metric names
        """
        formatted_metrics = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted_metrics.append(f"{prefix}{key}={value:.4f}")
            else:
                formatted_metrics.append(f"{prefix}{key}={value}")

        self.logger.info(f"Performance metrics: {' | '.join(formatted_metrics)}")

    def log_memory_usage(self, device: str = "cpu") -> None:
        """Log current memory usage.

        Args:
            device: Device to check memory for ('cpu' or 'cuda')
        """
        try:
            if device == "cuda" and hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available():
                memory_allocated = __import__('torch').cuda.memory_allocated() / 1024**2  # MB
                memory_reserved = __import__('torch').cuda.memory_reserved() / 1024**2    # MB
                self.logger.debug(f"GPU memory - allocated: {memory_allocated:.1f}MB, reserved: {memory_reserved:.1f}MB")
            else:
                # For CPU, we could add psutil-based memory monitoring if available
                self.logger.debug("CPU memory monitoring available")
        except ImportError:
            self.logger.debug("Memory monitoring not available")

    def log_experiment_start(
        self,
        experiment_name: str,
        config: Dict[str, Any]
    ) -> None:
        """Log experiment start with configuration.

        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary
        """
        self.logger.info(f"🚀 Starting experiment: {experiment_name}")
        self.logger.info(f"Configuration: {config}")

    def log_experiment_end(
        self,
        experiment_name: str,
        results: Dict[str, Any],
        duration: float
    ) -> None:
        """Log experiment completion.

        Args:
            experiment_name: Name of the experiment
            results: Results dictionary
            duration: Experiment duration in seconds
        """
        self.logger.info(f"✅ Completed experiment: {experiment_name} in {duration:.2f}s")
        self.logger.info(f"Final results: {results}")

    def log_problem_start(self, problem_name: str, config: Dict[str, Any]) -> None:
        """Log problem simulation start.

        Args:
            problem_name: Name of the problem
            config: Problem configuration
        """
        self.logger.info(f"🔄 Starting {problem_name} simulation")

    def log_problem_end(self, problem_name: str, results: Dict[str, Any], duration: float) -> None:
        """Log problem simulation completion.

        Args:
            problem_name: Name of the problem
            results: Problem results
            duration: Problem duration in seconds
        """
        self.logger.info(f"✅ Completed {problem_name} in {duration:.2f}s - Best cost: {results.get('best_cost', 'N/A')}")

    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self.logger


# Global logger instance
_default_logger: Optional[GFACSLogger] = None


def get_logger(
    name: str = "gfacs",
    log_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> logging.Logger:
    """Get or create a GFACS logger instance.

    Args:
        name: Logger name
        log_dir: Log directory
        **kwargs: Additional arguments for GFACSLogger

    Returns:
        Logger instance
    """
    global _default_logger

    if _default_logger is None or _default_logger.name != name:
        _default_logger = GFACSLogger(name=name, log_dir=log_dir, **kwargs)

    return _default_logger.get_logger()


def setup_experiment_logging(
    experiment_name: str,
    log_dir: Union[str, Path] = "logs",
    **logger_kwargs
) -> logging.Logger:
    """Setup logging for an experiment.

    Args:
        experiment_name: Name of the experiment
        log_dir: Base log directory
        **logger_kwargs: Additional logger arguments

    Returns:
        Configured logger instance
    """
    experiment_log_dir = Path(log_dir) / experiment_name
    logger = get_logger(
        name=experiment_name,
        log_dir=str(experiment_log_dir),
        **logger_kwargs
    )

    # Log experiment setup
    logger.info(f"📝 Experiment logging setup complete in {experiment_log_dir}")

    return logger
