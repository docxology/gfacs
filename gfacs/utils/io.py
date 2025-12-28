"""Input/Output persistence utilities for GFACS experiments.

This module provides comprehensive I/O functionality for saving and loading
experiment data, configurations, results, and visualizations.
"""

import json
import pickle
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import datetime

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class ExperimentIO:
    """Input/Output manager for GFACS experiments.

    Provides:
    - Experiment directory management
    - Input data persistence
    - Results saving
    - Checkpoint management
    - Metadata tracking
    """

    def __init__(self, base_dir: Union[str, Path] = "outputs"):
        """Initialize experiment I/O manager.

        Args:
            base_dir: Base directory for experiment outputs
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def create_experiment_dir(
        self,
        experiment_name: str,
        timestamp: bool = True
    ) -> Path:
        """Create experiment directory.

        Args:
            experiment_name: Name of the experiment
            timestamp: Whether to add timestamp to directory name

        Returns:
            Path to experiment directory
        """
        if timestamp:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = self.base_dir / f"{experiment_name}_{timestamp_str}"
        else:
            exp_dir = self.base_dir / experiment_name

        exp_dir.mkdir(exist_ok=True)
        return exp_dir

    def save_config(
        self,
        config: Any,
        experiment_dir: Path
    ) -> None:
        """Save experiment configuration.

        Args:
            config: Experiment configuration (dataclass or dict)
            experiment_dir: Experiment directory
        """
        config_dir = experiment_dir / "config"
        config_dir.mkdir(exist_ok=True)

        config_file = config_dir / "experiment.yaml"

        # Convert config to dict if it's a dataclass
        if hasattr(config, '__dict__'):
            config_dict = vars(config)
        elif hasattr(config, '__dataclass_fields__'):
            import dataclasses
            config_dict = dataclasses.asdict(config)
        else:
            config_dict = dict(config)

        # Ensure all nested dataclasses are also converted to dicts
        def convert_dataclasses_to_dict(obj):
            if isinstance(obj, dict):
                return {k: convert_dataclasses_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dataclasses_to_dict(item) for item in obj]
            elif hasattr(obj, '__dataclass_fields__'):
                import dataclasses
                return convert_dataclasses_to_dict(dataclasses.asdict(obj))
            elif callable(obj):
                raise TypeError(f"Cannot serialize callable object of type {type(obj)}")
            else:
                return obj

        config_dict = convert_dataclasses_to_dict(config_dict)

        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def save_input_data(
        self,
        data: Dict[str, Any],
        experiment_dir: Path,
        filename: str = "input_data.pkl"
    ) -> None:
        """Save input data (coordinates, demands, constraints, etc.).

        Args:
            data: Input data dictionary
            experiment_dir: Experiment directory
            filename: Output filename
        """
        input_dir = experiment_dir / "data" / "inputs"
        input_dir.mkdir(exist_ok=True, parents=True)

        filepath = input_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def save_problem_instance(
        self,
        problem_data: Dict[str, Any],
        experiment_dir: Path,
        problem_name: str,
        instance_name: str = "problem_instance"
    ) -> None:
        """Save problem instance data.

        Args:
            problem_data: Problem instance data
            experiment_dir: Experiment directory
            problem_name: Name of the problem (e.g., 'tsp_nls')
            instance_name: Name for the instance
        """
        problem_input_dir = experiment_dir / "data" / "inputs" / problem_name
        problem_input_dir.mkdir(exist_ok=True, parents=True)

        filepath = problem_input_dir / f"{instance_name}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(problem_data, f)

        # Also save as JSON if all data is serializable
        try:
            json_data = self._make_json_serializable(problem_data)
            json_file = problem_input_dir / f"{instance_name}.json"
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
        except (TypeError, ValueError):
            pass  # Skip JSON saving if not serializable

    def save_results(
        self,
        results: Dict[str, Any],
        experiment_dir: Path,
        filename: str = "results.json"
    ) -> None:
        """Save experiment results.

        Args:
            results: Results dictionary
            experiment_dir: Experiment directory
            filename: Output filename
        """
        results_dir = experiment_dir / "data" / "results"
        results_dir.mkdir(exist_ok=True, parents=True)

        filepath = results_dir / filename

        # Make results JSON serializable
        json_results = self._make_json_serializable(results)

        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)

        # Also save as pickle for full fidelity
        pickle_file = results_dir / filename.replace('.json', '.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(results, f)

    def save_problem_results(
        self,
        results: Dict[str, Any],
        experiment_dir: Path,
        problem_name: str,
        filename: str = "results.json"
    ) -> None:
        """Save problem-specific results.

        Args:
            results: Problem results dictionary
            experiment_dir: Experiment directory
            problem_name: Name of the problem
            filename: Output filename
        """
        problem_results_dir = experiment_dir / "data" / "results" / "per_problem" / problem_name
        problem_results_dir.mkdir(exist_ok=True, parents=True)

        filepath = problem_results_dir / filename

        # Make results JSON serializable
        json_results = self._make_json_serializable(results)

        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)

    def save_training_metrics(
        self,
        metrics: Dict[str, List[float]],
        experiment_dir: Path,
        problem_name: str,
        filename: str = "training_metrics.json"
    ) -> None:
        """Save training metrics over time.

        Args:
            metrics: Dictionary of metric lists
            experiment_dir: Experiment directory
            problem_name: Name of the problem
            filename: Output filename
        """
        metrics_dir = experiment_dir / "data" / "metrics" / problem_name
        metrics_dir.mkdir(exist_ok=True, parents=True)

        filepath = metrics_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]],
        epoch: int,
        metrics: Optional[Dict[str, float]],
        experiment_dir: Path,
        problem_name: str,
        filename: Optional[str] = None
    ) -> Path:
        """Save model checkpoint.

        Args:
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            epoch: Current epoch
            metrics: Current metrics
            experiment_dir: Experiment directory
            problem_name: Name of the problem
            filename: Checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        checkpoints_dir = experiment_dir / "data" / "checkpoints" / problem_name
        checkpoints_dir.mkdir(exist_ok=True, parents=True)

        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"

        filepath = checkpoints_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'metrics': metrics,
            'timestamp': datetime.datetime.now().isoformat(),
            'problem': problem_name,
        }

        import torch
        torch.save(checkpoint, filepath)
        return filepath

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        import torch
        return torch.load(checkpoint_path, map_location='cpu')

    def save_metadata(
        self,
        metadata: Dict[str, Any],
        experiment_dir: Path,
        filename: str = "metadata.json"
    ) -> None:
        """Save experiment metadata.

        Args:
            metadata: Metadata dictionary
            experiment_dir: Experiment directory
        """
        metadata_dir = experiment_dir / "data"
        metadata_dir.mkdir(exist_ok=True, parents=True)

        filepath = metadata_dir / filename

        json_metadata = self._make_json_serializable(metadata)

        with open(filepath, 'w') as f:
            json.dump(json_metadata, f, indent=2)

    def save_summary(
        self,
        config: Any,
        results: Dict[str, Any],
        experiment_dir: Path
    ) -> None:
        """Save experiment summary.

        Args:
            config: Experiment configuration
            results: Experiment results
            experiment_dir: Experiment directory
        """
        summary = {
            'experiment_name': getattr(config, 'experiment_name', 'unknown'),
            'description': getattr(config, 'description', ''),
            'start_time': results.get('start_time'),
            'end_time': results.get('end_time'),
            'duration': results.get('duration'),
            'problems_run': list(results.get('problem_results', {}).keys()),
            'total_problems': len(results.get('problem_results', {})),
            'overall_best_cost': results.get('overall_best_cost'),
            'config_summary': {
                'timestamp': datetime.datetime.now().isoformat(),
                'problems': getattr(config, 'problems', []),
                'base_output_dir': str(experiment_dir),
            }
        }

        # Add per-problem summaries
        if 'problem_results' in results:
            summary['problem_summaries'] = {}
            for problem_name, problem_results in results['problem_results'].items():
                summary['problem_summaries'][problem_name] = {
                    'best_cost': problem_results.get('best_cost'),
                    'duration': problem_results.get('duration'),
                    'iterations': problem_results.get('iterations', 0),
                }

        summary_file = experiment_dir / "EXPERIMENT_SUMMARY.md"
        with open(summary_file, 'w') as f:
            f.write("# GFACS Experiment Summary\n\n")
            f.write(f"**Experiment:** {summary['experiment_name']}\n")
            f.write(f"**Date:** {summary['config_summary']['timestamp']}\n")
            f.write(f"**Duration:** {summary.get('duration', 'N/A')} seconds\n\n")

            if summary['description']:
                f.write(f"**Description:** {summary['description']}\n\n")

            f.write("## Problems Executed\n\n")
            for problem in summary['problems_run']:
                f.write(f"- {problem}\n")
            f.write("\n")

            f.write("## Results Summary\n\n")
            if 'problem_summaries' in summary:
                f.write("| Problem | Best Cost | Duration | Iterations |\n")
                f.write("|---------|-----------|----------|------------|\n")
                for problem_name, prob_summary in summary['problem_summaries'].items():
                    f.write(f"| {problem_name} | {prob_summary.get('best_cost', 'N/A')} | {prob_summary.get('duration', 'N/A')} | {prob_summary.get('iterations', 'N/A')} |\n")
                f.write("\n")

            f.write("## Output Structure\n\n")
            f.write("```\n")
            f.write(f"{experiment_dir.name}/\n")
            f.write("├── config/                 # All configurations\n")
            f.write("├── logs/                   # All logs\n")
            f.write("├── data/                   # All input/output data\n")
            f.write("│   ├── inputs/            # Input instances\n")
            f.write("│   ├── outputs/           # Solution outputs\n")
            f.write("│   └── results/           # Aggregated results\n")
            f.write("├── visualizations/        # Static plots\n")
            f.write("└── animations/            # Animated visualizations\n")
            f.write("```\n\n")

            f.write("## File Descriptions\n\n")
            f.write("- `config/experiment.yaml` - Main experiment configuration\n")
            f.write("- `logs/orchestrator.log` - Main orchestrator execution log\n")
            f.write("- `data/results/summary.json` - Overall experiment results\n")
            f.write("- `visualizations/` - Static plots and charts\n")
            f.write("- `animations/` - Interactive animations and videos\n")

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON serializable version
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif HAS_PANDAS and isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif hasattr(obj, 'tolist'):  # NumPy arrays, PyTorch tensors
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, '__dict__'):
            # For custom objects, convert their __dict__
            return self._make_json_serializable(obj.__dict__)
        else:
            # Convert to string as fallback
            return str(obj)

    def list_experiments(self) -> List[Path]:
        """List all experiment directories.

        Returns:
            List of experiment directory paths
        """
        return [d for d in self.base_dir.iterdir() if d.is_dir()]

    def load_experiment_results(self, experiment_dir: Union[str, Path]) -> Dict[str, Any]:
        """Load experiment results.

        Args:
            experiment_dir: Experiment directory path

        Returns:
            Dictionary containing all experiment data
        """
        exp_dir = Path(experiment_dir)

        results = {}

        # Load config
        config_file = exp_dir / "config" / "experiment.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                results['config'] = yaml.safe_load(f)

        # Load summary
        summary_file = exp_dir / "data" / "results" / "summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                results['summary'] = json.load(f)

        # Load metadata
        metadata_file = exp_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                results['metadata'] = json.load(f)

        return results


# Global I/O manager instance
_io_manager: Optional[ExperimentIO] = None


def get_io_manager(base_dir: Union[str, Path] = "outputs") -> ExperimentIO:
    """Get global I/O manager instance."""
    global _io_manager
    if _io_manager is None:
        _io_manager = ExperimentIO(base_dir)
    return _io_manager


def setup_experiment_io(
    experiment_name: str,
    config: Any,
    base_dir: Union[str, Path] = "outputs"
) -> tuple[Path, ExperimentIO]:
    """Setup I/O for an experiment.

    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        base_dir: Base output directory

    Returns:
        Tuple of (experiment_dir, io_manager)
    """
    io_manager = get_io_manager(base_dir)
    experiment_dir = io_manager.create_experiment_dir(experiment_name)

    # Save configuration
    io_manager.save_config(config, experiment_dir)

    return experiment_dir, io_manager
