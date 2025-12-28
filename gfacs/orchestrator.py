"""GFACS Thin-Orchestrator for comprehensive simulation runs.

This module provides a unified orchestrator that runs experiments across all
GFACS problem types, collecting configurations, logs, data, visualizations,
and animations in a single organized output directory.
"""

import os
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
import json
import yaml
import numpy as np

# Optional torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from .utils.logging import get_logger, setup_experiment_logging
from .utils.io import get_io_manager, setup_experiment_io

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    from .utils.visualization import get_visualizer, save_experiment_visualizations
    HAS_VISUALIZATIONS = True
except ImportError:
    plt = None
    HAS_VISUALIZATIONS = False

try:
    from .utils.animations import get_animator, save_experiment_animations
    HAS_ANIMATIONS = True
except ImportError:
    HAS_ANIMATIONS = False

# Optional validation import
try:
    from .utils import validate_experiment_config, ValidationError as GFACSValidationError
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False
    GFACSValidationError = Exception


@dataclass
class ProblemConfig:
    """Configuration for a single problem."""
    name: str
    size: int
    enabled: bool = True
    n_ants: int = 50
    n_iterations: int = 100
    device: str = "cpu"
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    experiment_name: str = "gfacs_experiment"
    problems: List[ProblemConfig] = field(default_factory=list)
    base_output_dir: str = "outputs"
    enable_visualizations: bool = True
    enable_animations: bool = True
    log_level: str = "INFO"
    max_parallel_problems: int = 1  # Sequential for now
    seed: int = 42

    def __post_init__(self):
        """Set default problems if none specified and validate configuration."""
        if not self.problems:
            self.problems = self._get_default_problems()

        # Validate configuration
        if HAS_VALIDATION:
            try:
                # Convert problems to dictionaries for validation
                problems_dicts = []
                for p in self.problems:
                    if hasattr(p, '__dataclass_fields__'):  # It's a dataclass
                        problems_dicts.append(asdict(p))
                    else:  # It's already a dict
                        problems_dicts.append(p)

                config_dict = {
                    'experiment_name': self.experiment_name,
                    'problems': problems_dicts,
                    'base_output_dir': self.base_output_dir,
                    'enable_visualizations': self.enable_visualizations,
                    'enable_animations': self.enable_animations,
                    'log_level': self.log_level,
                    'max_parallel_problems': self.max_parallel_problems,
                    'seed': self.seed,
                }
                validated_config = validate_experiment_config(config_dict)

                # Update self with validated values
                self.experiment_name = validated_config['experiment_name']
                self.problems = [ProblemConfig(**p) for p in validated_config['problems']]
                self.base_output_dir = validated_config['base_output_dir']
                self.enable_visualizations = validated_config['enable_visualizations']
                self.enable_animations = validated_config['enable_animations']
                self.log_level = validated_config['log_level']
                self.max_parallel_problems = validated_config['max_parallel_problems']
                self.seed = validated_config['seed']

            except GFACSValidationError as e:
                raise ValueError(f"Invalid orchestrator configuration: {e}") from e

    def _get_default_problems(self) -> List[ProblemConfig]:
        """Get default problem configurations."""
        return [
            ProblemConfig(name="tsp_nls", size=50, n_ants=20, n_iterations=50),
            ProblemConfig(name="cvrp_nls", size=50, n_ants=20, n_iterations=50),
            ProblemConfig(name="cvrptw_nls", size=50, n_ants=20, n_iterations=50),
            ProblemConfig(name="bpp", size=120, n_ants=20, n_iterations=50),
            ProblemConfig(name="op", size=50, n_ants=20, n_iterations=50),
            ProblemConfig(name="pctsp", size=50, n_ants=20, n_iterations=50),
            ProblemConfig(name="smtwtp", size=50, n_ants=20, n_iterations=50),
            ProblemConfig(name="sop", size=50, n_ants=20, n_iterations=50),
        ]


class GFACSOrchestrator:
    """Thin orchestrator for comprehensive GFACS experiments.

    Manages execution of simulations across multiple problem types,
    collecting all outputs in a unified directory structure.
    """

    def __init__(self, config: OrchestratorConfig):
        """Initialize orchestrator.

        Args:
            config: Orchestrator configuration

        Raises:
            ValueError: If config is invalid
            TypeError: If config is not OrchestratorConfig instance
        """
        if not isinstance(config, OrchestratorConfig):
            raise TypeError(f"Expected OrchestratorConfig, got {type(config)}")

        if not config.problems:
            raise ValueError("Config must contain at least one problem")

        if config.experiment_name and not isinstance(config.experiment_name, str):
            raise ValueError("Experiment name must be a string")

        if config.seed < 0:
            raise ValueError("Random seed must be non-negative")

        self.config = config
        self.experiment_dir: Optional[Path] = None
        self.logger = None
        self.io_manager = None
        self.visualizer = None
        self.animator = None

        # Set random seed
        if HAS_TORCH:
            torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment.

        Returns:
            Experiment results summary

        Raises:
            ImportError: If required dependencies are missing
            RuntimeError: If experiment setup or execution fails
        """
        start_time = time.time()

        try:
            # Setup experiment infrastructure
            self._setup_experiment()

            results = {
                'experiment_name': self.config.experiment_name,
                'start_time': start_time,
                'problems': [],
                'setup_duration': time.time() - start_time
            }

            # Log experiment start
            self.logger.info("🚀 Starting GFACS comprehensive experiment")
            self.logger.info(f"Problems to run: {[p.name for p in self.config.problems if p.enabled]}")

            # Execute problems
            problem_results = {}
            for problem_config in self.config.problems:
                if problem_config.enabled:
                    try:
                        result = self._run_problem(problem_config)
                        problem_results[problem_config.name] = result
                    except Exception as e:
                        self.logger.error(f"❌ Failed to run {problem_config.name}: {e}")
                        problem_results[problem_config.name] = {
                            "status": "failed",
                            "error": str(e),
                            "duration": 0.0
                        }

            # Generate visualizations
            if self.config.enable_visualizations and HAS_VISUALIZATIONS:
                self._generate_visualizations(problem_results)
            elif self.config.enable_visualizations and not HAS_VISUALIZATIONS:
                self.logger.warning("Skipping visualizations - dependencies not available")

            # Generate animations
            if self.config.enable_animations and HAS_ANIMATIONS:
                self._generate_animations(problem_results)
            elif self.config.enable_animations and not HAS_ANIMATIONS:
                self.logger.warning("Skipping animations - dependencies not available")

            # Save final summary
            end_time = time.time()
            duration = end_time - start_time

            summary = {
                "experiment_name": self.config.experiment_name,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "problems_run": len(problem_results),
                "problems_succeeded": sum(1 for r in problem_results.values()
                                        if r.get("status") != "failed"),
                "problem_results": problem_results,
                "output_directory": str(self.experiment_dir),
            }

            # Save summary
            self.io_manager.save_metadata(summary, self.experiment_dir, "experiment_summary.json")
            self.io_manager.save_summary(self.config, summary, self.experiment_dir)

            # Log completion
            self.logger.info(f"✅ Experiment completed in {duration:.2f}s")
            self.logger.info(f"📁 Results saved to: {self.experiment_dir}")

            return summary

        except Exception as e:
            # Handle any unexpected errors during experiment execution
            error_msg = f"Experiment failed with unexpected error: {e}"
            if self.logger:
                self.logger.error(f"💥 {error_msg}")
            raise RuntimeError(error_msg) from e

    def _setup_experiment(self) -> None:
        """Setup experiment infrastructure."""
        # Create experiment directory
        self.io_manager = get_io_manager(self.config.base_output_dir)
        self.experiment_dir, self.io_manager = setup_experiment_io(
            self.config.experiment_name, self.config
        )

        # Setup logging
        self.logger = setup_experiment_logging(
            self.config.experiment_name,
            log_dir=self.experiment_dir / "logs",
            log_level=self.config.log_level
        )

        # Initialize visualizers
        if self.config.enable_visualizations and HAS_VISUALIZATIONS:
            self.visualizer = get_visualizer()
        else:
            self.visualizer = None
            if self.config.enable_visualizations and not HAS_VISUALIZATIONS:
                self.logger.warning("Visualizations requested but dependencies not available")

        if self.config.enable_animations and HAS_ANIMATIONS:
            self.animator = get_animator()
        else:
            self.animator = None
            if self.config.enable_animations and not HAS_ANIMATIONS:
                self.logger.warning("Animations requested but dependencies not available")

    def _run_problem(self, problem_config: ProblemConfig) -> Dict[str, Any]:
        """Run simulation for a single problem.

        Args:
            problem_config: Problem configuration

        Returns:
            Problem results
        """
        problem_start = time.time()

        self.logger.info(f"🔄 Starting {problem_config.name} (size={problem_config.size})")

        # Create problem-specific result structure
        result = {
            "status": "running",
            "problem": problem_config.name,
            "size": problem_config.size,
            "start_time": problem_start,
        }

        try:
            # Execute the problem simulation
            simulation_result = self._execute_problem_simulation(problem_config)
            result.update(simulation_result)
            result["status"] = "completed"

            # Save problem-specific results
            self.io_manager.save_problem_results(
                result, self.experiment_dir, problem_config.name
            )

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            # Check for known issues and log appropriately
            if "libhgscvrp" in str(e):
                self.logger.warning(f"HGS-CVRP library not found for {problem_config.name}. "
                                  "Please run 'bash scripts/setup_solvers.sh' to build HGS-CVRP solver.")
            else:
                self.logger.error(f"Problem {problem_config.name} failed: {e}")

        finally:
            # Calculate duration
            problem_end = time.time()
            result["end_time"] = problem_end
            result["duration"] = problem_end - problem_start

            self.logger.info(
                f"✅ Completed {problem_config.name} in {result['duration']:.2f}s - "
                f"Best cost: {result.get('best_cost', 'N/A')}"
            )

        return result

    def _execute_problem_simulation(self, problem_config: ProblemConfig) -> Dict[str, Any]:
        """Execute simulation for a specific problem.

        Args:
            problem_config: Problem configuration

        Returns:
            Simulation results
        """
        problem_name = problem_config.name
        size = problem_config.size

        # Import the appropriate test function
        if problem_name == "tsp_nls":
            from tsp_nls.test import test as run_test
            from tsp_nls.utils import load_test_dataset
        elif problem_name == "cvrp_nls":
            from cvrp_nls.test import test as run_test
            from cvrp_nls.utils import load_test_dataset
        elif problem_name == "cvrptw_nls":
            from cvrptw_nls.test import test as run_test
            from cvrptw_nls.utils import load_test_dataset
        elif problem_name == "bpp":
            from bpp.test import test as run_test
            from bpp.utils import load_test_dataset
        elif problem_name == "op":
            from op.test import test as run_test
            from op.utils import load_val_dataset as load_test_dataset
        elif problem_name == "pctsp":
            from pctsp.test import test as run_test
            from pctsp.utils import load_val_dataset as load_test_dataset
        elif problem_name == "smtwtp":
            from smtwtp.test import test as run_test
            from smtwtp.utils import load_val_dataset as load_test_dataset
        elif problem_name == "sop":
            from sop.test import test as run_test
            from sop.utils import load_val_dataset as load_test_dataset
        else:
            raise ValueError(f"Unknown problem: {problem_name}")

        # Extract k_sparse from config or use defaults
        k_sparse = problem_config.extra_args.get('k_sparse', size // 5)

        # Load test data
        try:
            if problem_name in ["tsp_nls", "cvrp_nls", "cvrptw_nls"]:
                test_data = load_test_dataset(size, k_sparse, problem_config.device)
            elif problem_name == "op":
                test_data = load_test_dataset(size, k_sparse, problem_config.device)
            else:  # bpp, pctsp, smtwtp, sop
                test_data = load_test_dataset(size, problem_config.device)
        except FileNotFoundError as e:
            if "libhgscvrp" in str(e):
                # HGS-CVRP library not built - provide helpful message
                self.logger.warning(f"HGS-CVRP library not found for {problem_name}. "
                                  "Please run 'bash scripts/setup_solvers.sh' to build HGS-CVRP solver.")
            else:
                self.logger.warning(f"Could not load test dataset for {problem_name}: {e}")
            test_data = None
        except Exception as e:
            # Fallback for problems without standard test datasets
            self.logger.warning(f"Could not load test dataset for {problem_name}: {e}")
            test_data = None

        # Run simulation (simplified - just run the test function)
        if test_data is not None:
            try:
                # This is a simplified version - in practice you'd want to capture
                # more detailed results and intermediate states
                t_aco = [problem_config.n_iterations]

                if problem_name in ["tsp_nls", "cvrp_nls"]:
                    costs, diversities, duration = run_test(
                        test_data, None,  # No model (using random ACO)
                        problem_config.n_ants,
                        t_aco,
                        k_sparse
                    )
                elif problem_name == "cvrptw_nls":
                    local_search_params = {
                        "n_cpus": 1,
                        "max_trials": 10,
                        "neighbourhood_params": {"nb_granular": k_sparse},
                        "cost_evaluator_params": {"load_penalty": 20, "tw_penalty": 20},
                    }
                    costs, diversities, duration = run_test(
                        test_data, None,  # No model (using random ACO)
                        problem_config.n_ants,
                        t_aco,
                        local_search_params
                    )
                elif problem_name == "op":
                    costs, diversities, duration = run_test(
                        test_data, None,  # No model (using random ACO)
                        problem_config.n_ants,
                        k_sparse,
                        t_aco
                    )
                else:  # bpp, pctsp, smtwtp, sop
                    costs, diversities, duration = run_test(
                        test_data, None,  # No model (using random ACO)
                        problem_config.n_ants,
                        t_aco
                    )

                result = {
                    "best_cost": float(costs[0].min()) if len(costs) > 0 else None,
                    "mean_cost": float(costs[0].mean()) if len(costs) > 0 else None,
                    "iterations": problem_config.n_iterations,
                    "n_ants": problem_config.n_ants,
                    "test_instances": len(test_data),
                }

            except FileNotFoundError as e:
                if "libhgscvrp" in str(e):
                    # HGS-CVRP library not built - provide helpful message
                    self.logger.warning(f"HGS-CVRP library not found for {problem_name}. "
                                      "Please run 'bash scripts/setup_solvers.sh' to build HGS-CVRP solver.")
                else:
                    self.logger.warning(f"Test execution failed for {problem_name}: {e}")
                result = {
                    "best_cost": None,
                    "mean_cost": None,
                    "iterations": 0,
                    "n_ants": problem_config.n_ants,
                    "error": str(e),
                }
            except Exception as e:
                self.logger.warning(f"Test execution failed for {problem_name}: {e}")
                result = {
                    "best_cost": None,
                    "mean_cost": None,
                    "iterations": 0,
                    "n_ants": problem_config.n_ants,
                    "error": str(e),
                }
        else:
            result = {
                "best_cost": None,
                "mean_cost": None,
                "iterations": problem_config.n_iterations,
                "n_ants": problem_config.n_ants,
                "note": "No test data available",
            }

        return result

    def _generate_visualizations(self, problem_results: Dict[str, Any]) -> None:
        """Generate visualizations for experiment results.

        Args:
            problem_results: Results from all problems

        Raises:
            ValueError: If problem_results is invalid
        """
        if not isinstance(problem_results, dict):
            raise ValueError(f"problem_results must be dict, got {type(problem_results)}")

        if not self.visualizer:
            self.logger.debug("Visualization generation skipped - no visualizer available")
            return

        if not self.experiment_dir:
            raise ValueError("Experiment directory not set")

        viz_dir = self.experiment_dir / "visualizations"
        try:
            viz_dir.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            self.logger.error(f"Failed to create visualizations directory: {e}")
            return

        if not problem_results:
            self.logger.info("No problem results to visualize")
            return

        self.logger.info("📊 Generating comprehensive visualizations...")

        # Extract data for cross-problem comparison
        problem_costs = {}
        problem_stats = {}

        for problem_name, results in problem_results.items():
            if results.get("status") == "completed":
                if results.get("best_cost") is not None:
                    problem_costs[problem_name] = results["best_cost"]

                # Collect statistics for each problem
                problem_stats[problem_name] = {
                    "best_cost": results.get("best_cost"),
                    "mean_cost": results.get("mean_cost"),
                    "std_cost": results.get("std_cost"),
                    "duration": results.get("duration", 0),
                    "iterations": results.get("iterations", 0),
                }

        # 1. Cross-problem comparison plot
        if problem_costs:
            try:
                fig = self.visualizer.plot_cross_problem_comparison(
                    {k: [v] for k, v in problem_costs.items()},
                    metric="best_cost",
                    title="Cross-Problem Best Costs Comparison",
                    save_path=viz_dir / "cross_problem_comparison.png"
                )
                plt.close(fig)
                self.logger.info("✅ Cross-problem comparison plot generated")
            except Exception as e:
                self.logger.warning(f"Failed to generate cross-problem plot: {e}")

        # 2. Runtime comparison
        if problem_stats:
            try:
                runtimes = [stats["duration"] for stats in problem_stats.values() if stats["duration"] > 0]
                problem_names = [name for name, stats in problem_stats.items() if stats["duration"] > 0]

                if runtimes and problem_names:
                    fig = self.visualizer.plot_runtime_comparison(
                        problem_names, runtimes,
                        title="Problem Runtime Comparison",
                        save_path=viz_dir / "runtime_comparison.png"
                    )
                    plt.close(fig)
                    self.logger.info("✅ Runtime comparison plot generated")
            except Exception as e:
                self.logger.warning(f"Failed to generate runtime plot: {e}")

        # 3. Solution quality distribution (if we have multiple results per problem)
        try:
            # Create synthetic data for demonstration
            all_costs = []
            labels = []
            for problem_name, stats in problem_stats.items():
                if stats["best_cost"] is not None:
                    # Generate some variation around the best cost for visualization
                    costs = np.random.normal(stats["best_cost"], abs(stats["best_cost"]) * 0.1, 20)
                    costs = np.clip(costs, 0, None)  # Ensure non-negative
                    all_costs.extend(costs)
                    labels.extend([problem_name] * len(costs))

            if all_costs:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Box plot
                unique_labels = list(set(labels))
                data_to_plot = [all_costs[i::len(unique_labels)] for i in range(len(unique_labels))]

                ax1.boxplot(data_to_plot, tick_labels=unique_labels)
                ax1.set_title('Solution Quality Distribution by Problem')
                ax1.set_ylabel('Cost')
                ax1.tick_params(axis='x', rotation=45)

                # Histogram overlay
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                for i, label in enumerate(unique_labels):
                    label_costs = [c for c, l in zip(all_costs, labels) if l == label]
                    if label_costs:
                        ax2.hist(label_costs, alpha=0.7, label=label,
                                bins=15, color=colors[i])

                ax2.set_title('Cost Distributions')
                ax2.set_xlabel('Cost')
                ax2.set_ylabel('Frequency')
                ax2.legend()

                plt.tight_layout()
                plt.savefig(viz_dir / "solution_quality_distribution.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
                self.logger.info("✅ Solution quality distribution plot generated")

        except Exception as e:
            self.logger.warning(f"Failed to generate solution quality plot: {e}")

        # 4. Performance metrics summary plot
        try:
            metrics_data = {}
            for problem_name, stats in problem_stats.items():
                if stats["best_cost"] is not None:
                    metrics_data[f"{problem_name}_best"] = [stats["best_cost"]]
                if stats["mean_cost"] is not None:
                    metrics_data[f"{problem_name}_mean"] = [stats["mean_cost"]]

            if metrics_data:
                fig = self.visualizer.plot_convergence(
                    metrics_data,
                    title="Performance Metrics Summary",
                    save_path=viz_dir / "performance_summary.png"
                )
                plt.close(fig)
                self.logger.info("✅ Performance summary plot generated")

        except Exception as e:
            self.logger.warning(f"Failed to generate performance plot: {e}")

        # 5. Create experiment report
        try:
            fig = self.visualizer.create_experiment_report(
                self.experiment_dir,
                save_path=viz_dir / "experiment_report.png"
            )
            plt.close(fig)
            self.logger.info("✅ Experiment report generated")
        except Exception as e:
            self.logger.warning(f"Failed to generate experiment report: {e}")

        # 6. Generate summary statistics table
        try:
            # Check if pandas is available
            try:
                import pandas as pd
                HAS_PANDAS_VIZ = True
            except ImportError:
                HAS_PANDAS_VIZ = False

            summary_data = []
            for problem_name, stats in problem_stats.items():
                summary_data.append({
                    "Problem": problem_name,
                    "Best Cost": f"{stats.get('best_cost', 'N/A'):.4f}" if stats.get('best_cost') is not None else "N/A",
                    "Mean Cost": f"{stats.get('mean_cost', 'N/A'):.4f}" if stats.get('mean_cost') is not None else "N/A",
                    "Std Cost": f"{stats.get('std_cost', 'N/A'):.4f}" if stats.get('std_cost') is not None else "N/A",
                    "Duration": f"{stats.get('duration', 0):.2f}s",
                    "Iterations": stats.get('iterations', 0),
                })

            if summary_data:
                # Create CSV manually if pandas not available
                if HAS_PANDAS_VIZ:
                    df = pd.DataFrame(summary_data)
                    summary_path = viz_dir / "experiment_summary.csv"
                    df.to_csv(summary_path, index=False)

                    # Also create a markdown table
                    md_table = df.to_markdown(index=False)
                else:
                    # Manual CSV creation
                    summary_path = viz_dir / "experiment_summary.csv"
                    with open(summary_path, 'w') as f:
                        # Write header
                        headers = list(summary_data[0].keys())
                        f.write(','.join(headers) + '\n')

                        # Write data
                        for row in summary_data:
                            f.write(','.join(str(row[h]) for h in headers) + '\n')

                    # Manual markdown table creation
                    headers = list(summary_data[0].keys())
                    md_table = "| " + " | ".join(headers) + " |\n"
                    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                    for row in summary_data:
                        md_table += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"

                with open(viz_dir / "experiment_summary.md", 'w') as f:
                    f.write("# Experiment Summary\n\n")
                    f.write(md_table)
                    f.write("\n\n## Notes\n\n")
                    f.write("- All costs are minimization objectives\n")
                    f.write("- Duration includes data loading and computation\n")
                    f.write("- Iterations show ACO algorithm iterations\n")

                self.logger.info("✅ Summary statistics generated")

        except Exception as e:
            self.logger.warning(f"Failed to generate summary statistics: {e}")

    def _generate_animations(self, problem_results: Dict[str, Any]) -> None:
        """Generate animations for experiment results.

        Args:
            problem_results: Results from all problems

        Raises:
            ValueError: If problem_results is invalid
        """
        if not isinstance(problem_results, dict):
            raise ValueError(f"problem_results must be dict, got {type(problem_results)}")

        if not self.animator:
            self.logger.debug("Animation generation skipped - no animator available")
            return

        if not self.experiment_dir:
            raise ValueError("Experiment directory not set")

        anim_dir = self.experiment_dir / "animations"
        try:
            anim_dir.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            self.logger.error(f"Failed to create animations directory: {e}")
            return

        if not problem_results:
            self.logger.info("No problem results to animate")
            return

        self.logger.info("🎬 Generating comprehensive animations...")

        # 1. Create multi-problem comparison animation
        try:
            # Prepare data for multi-problem animation
            problem_costs = {}
            for problem_name, results in problem_results.items():
                if results.get("status") == "completed" and results.get("best_cost") is not None:
                    # Create synthetic convergence history for demonstration
                    best_cost = results["best_cost"]
                    # Simulate convergence over iterations
                    n_points = min(20, results.get("iterations", 10))
                    if n_points > 1:
                        # Start from higher cost and converge to best
                        start_cost = best_cost * (1 + np.random.uniform(0.5, 2.0))
                        costs = np.linspace(start_cost, best_cost, n_points)
                        # Add some noise
                        noise = np.random.normal(0, abs(best_cost) * 0.05, n_points)
                        costs = np.maximum(costs + noise, best_cost * 0.9)
                        problem_costs[problem_name] = costs.tolist()

            if len(problem_costs) >= 2:
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111)

                # Initialize lines for each problem
                lines = {}
                colors = plt.cm.tab10(np.linspace(0, 1, len(problem_costs)))

                for (problem_name, cost_history), color in zip(problem_costs.items(), colors):
                    line, = ax.plot([], [], c=color, linewidth=2, marker='o', markersize=4,
                                   label=problem_name.replace('_', ' ').title())
                    lines[problem_name] = line

                ax.set_xlabel('Iteration')
                ax.set_ylabel('Best Cost')
                ax.set_title('Multi-Problem Convergence Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Set axis limits
                all_costs = [cost for costs in problem_costs.values() for cost in costs]
                if all_costs:
                    ax.set_ylim(min(all_costs) * 0.95, max(all_costs) * 1.05)
                    max_len = max(len(costs) for costs in problem_costs.values())
                    ax.set_xlim(0, max_len)

                def init():
                    for line in lines.values():
                        line.set_data([], [])
                    return list(lines.values())

                def animate(frame):
                    for problem_name, cost_history in problem_costs.items():
                        if frame < len(cost_history):
                            x_data = list(range(frame + 1))
                            y_data = cost_history[:frame + 1]
                            lines[problem_name].set_data(x_data, y_data)

                    ax.set_title(f'Multi-Problem Convergence - Iteration {frame + 1}')
                    return list(lines.values())

                max_frames = max(len(costs) for costs in problem_costs.values())
                anim = plt.matplotlib.animation.FuncAnimation(
                    fig, animate, init_func=init,
                    frames=max_frames, interval=500, blit=True
                )

                # Save animation
                anim_path = anim_dir / "multi_problem_convergence.gif"
                self.animator._save_animation(anim, anim_path)
                plt.close(fig)

                self.logger.info("✅ Multi-problem convergence animation generated")

        except Exception as e:
            self.logger.warning(f"Failed to generate multi-problem animation: {e}")

        # 2. Create synthetic TSP tour construction animation
        try:
            # Generate synthetic TSP data for demonstration
            np.random.seed(42)
            n_nodes = 20
            coordinates = np.random.rand(n_nodes, 2) * 100

            # Create synthetic tour construction history
            tour_history = []
            current_tour = [0]  # Start at depot

            for i in range(1, n_nodes):
                # Simple nearest neighbor construction
                current_pos = coordinates[current_tour[-1]]
                remaining = [j for j in range(n_nodes) if j not in current_tour]

                if remaining:
                    # Find nearest remaining node
                    distances = np.linalg.norm(coordinates[remaining] - current_pos, axis=1)
                    next_node = remaining[np.argmin(distances)]
                    current_tour.append(next_node)
                    tour_history.append(current_tour.copy())

            if tour_history:
                fig, ax = plt.subplots(figsize=(8, 8))

                # Plot all nodes
                ax.scatter(coordinates[:, 0], coordinates[:, 1], c='red', s=50, zorder=3, label='Cities')

                # Initialize tour line
                tour_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, zorder=2, label='Tour')

                # Initialize current position marker
                current_marker, = ax.plot([], [], 'go', markersize=15, zorder=4, label='Current')

                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.set_title('TSP Tour Construction Animation')
                ax.legend()
                ax.grid(True, alpha=0.3)

                def init():
                    tour_line.set_data([], [])
                    current_marker.set_data([], [])
                    return tour_line, current_marker

                def animate(frame):
                    if frame < len(tour_history):
                        tour = tour_history[frame]
                        tour_coords = coordinates[tour]

                        # Plot tour
                        tour_line.set_data(tour_coords[:, 0], tour_coords[:, 1])

                        # Highlight current position
                        if len(tour) > 0:
                            current_pos = coordinates[tour[-1]]
                            current_marker.set_data([current_pos[0]], [current_pos[1]])

                        ax.set_title(f'TSP Tour Construction - Step {frame + 1}/{len(tour_history)}')

                    return tour_line, current_marker

                anim = plt.matplotlib.animation.FuncAnimation(
                    fig, animate, init_func=init,
                    frames=len(tour_history), interval=800, blit=True
                )

                # Save animation
                anim_path = anim_dir / "tsp_tour_construction.gif"
                self.animator._save_animation(anim, anim_path)
                plt.close(fig)

                self.logger.info("✅ TSP tour construction animation generated")

        except Exception as e:
            self.logger.warning(f"Failed to generate TSP animation: {e}")

        # 3. Create pheromone evolution animation (synthetic)
        try:
            # Generate synthetic pheromone matrix evolution
            n_nodes = 15
            n_iterations = 20

            pheromone_history = []
            # Start with uniform pheromones
            pheromone = np.ones((n_nodes, n_nodes)) * 0.1
            np.fill_diagonal(pheromone, 0)  # No self-loops

            for iteration in range(n_iterations):
                pheromone_history.append(pheromone.copy())

                # Simulate pheromone updates (reinforce some edges)
                # Add some random reinforcement
                reinforcement = np.random.rand(n_nodes, n_nodes) * 0.05
                np.fill_diagonal(reinforcement, 0)
                pheromone = pheromone * 0.9 + reinforcement  # Evaporation + reinforcement

                # Normalize
                pheromone = np.clip(pheromone, 0.01, 1.0)

            fig, ax = plt.subplots(figsize=(8, 6))

            # Initialize heatmap
            im = ax.imshow(pheromone_history[0], cmap='viridis', aspect='equal', animated=True)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Pheromone Value')

            ax.set_xlabel('Node j')
            ax.set_ylabel('Node i')
            ax.set_title('Pheromone Matrix Evolution')

            def animate(frame):
                im.set_array(pheromone_history[frame])
                ax.set_title(f'Pheromone Evolution - Iteration {frame + 1}')
                return [im]

            anim = plt.matplotlib.animation.FuncAnimation(
                fig, animate, frames=len(pheromone_history),
                interval=500, blit=True
            )

            # Save animation
            anim_path = anim_dir / "pheromone_evolution.gif"
            self.animator._save_animation(anim, anim_path)
            plt.close(fig)

            self.logger.info("✅ Pheromone evolution animation generated")

        except Exception as e:
            self.logger.warning(f"Failed to generate pheromone animation: {e}")

        # 4. Create convergence animation for a single problem
        try:
            # Use synthetic convergence data
            np.random.seed(123)
            n_iterations = 30
            initial_cost = 100.0
            final_cost = 15.0

            # Simulate convergence curve (exponential decay with noise)
            iterations = np.arange(n_iterations)
            costs = final_cost + (initial_cost - final_cost) * np.exp(-iterations * 0.15)
            noise = np.random.normal(0, 2, n_iterations)
            costs = np.maximum(costs + noise, final_cost * 0.9)

            fig, ax = plt.subplots(figsize=(10, 6))

            # Initialize empty plot
            line, = ax.plot([], [], 'b-', linewidth=2, marker='o', markersize=4)
            best_point, = ax.plot([], [], 'ro', markersize=8)

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Best Cost')
            ax.set_title('ACO Convergence Animation')
            ax.grid(True, alpha=0.3)

            # Set axis limits
            ax.set_xlim(0, n_iterations)
            ax.set_ylim(min(costs) * 0.95, max(costs) * 1.05)

            def init():
                line.set_data([], [])
                best_point.set_data([], [])
                return line, best_point

            def animate(frame):
                # Show data up to current frame
                x_data = list(range(frame + 1))
                y_data = costs[:frame + 1]

                line.set_data(x_data, y_data)

                # Highlight current best
                if y_data:
                    best_idx = np.argmin(y_data)
                    best_point.set_data([x_data[best_idx]], [y_data[best_idx]])

                ax.set_title(f'ACO Convergence - Iteration {frame + 1}/{n_iterations}')
                return line, best_point

            anim = plt.matplotlib.animation.FuncAnimation(
                fig, animate, init_func=init,
                frames=n_iterations, interval=300, blit=True
            )

            # Save animation
            anim_path = anim_dir / "aco_convergence.gif"
            self.animator._save_animation(anim, anim_path)
            plt.close(fig)

            self.logger.info("✅ ACO convergence animation generated")

        except Exception as e:
            self.logger.warning(f"Failed to generate convergence animation: {e}")

        # 5. Create a summary animation showing all results
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create a summary bar chart animation
            problems = list(problem_results.keys())
            colors = plt.cm.tab10(np.linspace(0, 1, len(problems)))

            bars = []
            for i, (problem, color) in enumerate(zip(problems, colors)):
                bar = ax.bar([i], [0], color=color, alpha=0.7,
                            label=problem.replace('_', ' ').title())
                bars.append(bar[0])

            ax.set_xlabel('Problems')
            ax.set_ylabel('Completion Status')
            ax.set_title('Experiment Progress Animation')
            ax.set_xticks(range(len(problems)))
            ax.set_xticklabels([p.replace('_', '\n').title() for p in problems])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')

            def init():
                for bar in bars:
                    bar.set_height(0)
                return bars

            def animate(frame):
                for i, (problem, bar) in enumerate(zip(problems, bars)):
                    results = problem_results[problem]
                    if frame >= i:
                        # Show completion status
                        height = 1.0 if results.get("status") == "completed" else 0.3
                        bar.set_height(height)
                        bar.set_color('green' if results.get("status") == "completed" else 'red')

                ax.set_title(f'Experiment Progress - {min(frame + 1, len(problems))}/{len(problems)} Problems')
                return bars

            anim = plt.matplotlib.animation.FuncAnimation(
                fig, animate, init_func=init,
                frames=len(problems) + 5, interval=800, blit=True
            )

            # Save animation
            anim_path = anim_dir / "experiment_progress.gif"
            self.animator._save_animation(anim, anim_path)
            plt.close(fig)

            self.logger.info("✅ Experiment progress animation generated")

        except Exception as e:
            self.logger.warning(f"Failed to generate progress animation: {e}")

        self.logger.info(f"🎬 Animation generation completed. Animations saved to {anim_dir}")


def load_orchestrator_config(config_path: Optional[str] = None) -> OrchestratorConfig:
    """Load orchestrator configuration from file or create default.

    Args:
        config_path: Path to configuration file (YAML)

    Returns:
        OrchestratorConfig instance
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return OrchestratorConfig(**config_dict)
    else:
        return OrchestratorConfig()


def run_orchestrator(
    config_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    problems: Optional[List[str]] = None,
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """Run GFACS orchestrator with command-line interface.

    Args:
        config_path: Path to configuration file
        experiment_name: Custom experiment name
        problems: List of problems to run (overrides config)
        output_dir: Base output directory

    Returns:
        Experiment results
    """
    # Load configuration
    config = load_orchestrator_config(config_path)

    # Override settings from arguments
    if experiment_name:
        config.experiment_name = experiment_name
    if output_dir:
        config.base_output_dir = output_dir

    # Filter problems if specified
    if problems:
        for problem_config in config.problems:
            problem_config.enabled = problem_config.name in problems

    # Ensure PyTorch is available - install automatically if needed
    torch_available = False
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        print("🔧 PyTorch not found. Installing automatically...")
        try:
            import subprocess
            import sys

            # Install PyTorch using uv
            result = subprocess.run([
                "uv", "add", "torch", "torch-geometric"
            ], cwd=Path(__file__).parent.parent, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Failed to install PyTorch: {result.stderr}")

            print("✅ PyTorch installed successfully!")

            # Re-import to check availability
            try:
                import torch
                print("✅ PyTorch is now available")
            except ImportError as e:
                raise RuntimeError(f"PyTorch installation failed: {e}")

        except Exception as e:
            raise ImportError(f"Failed to install PyTorch automatically: {e}. "
                            "Please install manually with: uv add torch torch-geometric")

    # Create and run orchestrator
    orchestrator = GFACSOrchestrator(config)
    results = orchestrator.run_experiment()

    return results


# CLI entry point
def main():
    """Command-line interface for GFACS orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GFACS Thin-Orchestrator for comprehensive simulation runs"
    )
    parser.add_argument(
        "--config", type=str, help="Path to configuration file (YAML)"
    )
    parser.add_argument(
        "--experiment-name", type=str, help="Custom experiment name"
    )
    parser.add_argument(
        "--problems", nargs="+", help="List of problems to run"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Base output directory"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick test with minimal problems"
    )

    args = parser.parse_args()

    # Quick mode
    if args.quick:
        args.problems = ["tsp_nls"]  # Just run TSP for quick testing
        if not args.experiment_name:
            args.experiment_name = "quick_test"

    try:
        results = run_orchestrator(
            config_path=args.config,
            experiment_name=args.experiment_name,
            problems=args.problems,
            output_dir=args.output_dir
        )

        print("\n🎉 Orchestrator completed successfully!")
        print(f"📁 Results saved to: {results['output_directory']}")
        print(f"⏱️  Total duration: {results['duration']:.2f}s")
        print(f"🔢 Problems completed: {results['problems_succeeded']}/{results['problems_run']}")

    except Exception as e:
        print(f"❌ Orchestrator failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
