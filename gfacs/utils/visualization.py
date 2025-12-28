"""Visualization utilities for GFACS experiments.

This module provides plotting and visualization functions for analyzing
solutions, training progress, and performance metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import datetime

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Set matplotlib backend for headless operation
import matplotlib
matplotlib.use('Agg')

# Set seaborn style if available
if HAS_SEABORN:
    sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class GFACSVisualizer:
    """Visualization utilities for GFACS experiments."""

    def __init__(self, style: str = "whitegrid"):
        """Initialize visualizer.

        Args:
            style: Seaborn style to use

        Raises:
            ValueError: If style is invalid
        """
        if not isinstance(style, str):
            raise ValueError(f"style must be string, got {type(style)}")

        if HAS_SEABORN:
            try:
                sns.set_style(style)
                self.colors = sns.color_palette("husl", 10)
            except ValueError as e:
                raise ValueError(f"Invalid seaborn style '{style}': {e}")
        else:
            # Fallback color palette
            self.colors = plt.cm.tab10.colors

    def plot_tsp_tour(
        self,
        coordinates: Union[np.ndarray, torch.Tensor],
        tour: Optional[Union[np.ndarray, torch.Tensor]] = None,
        title: str = "TSP Tour",
        save_path: Optional[Union[str, Path]] = None,
        show_numbers: bool = False
    ) -> plt.Figure:
        """Plot TSP tour.

        Args:
            coordinates: Node coordinates [n_nodes, 2]
            tour: Tour indices (if None, plot points only)
            title: Plot title
            save_path: Path to save plot
            show_numbers: Whether to show node numbers

        Returns:
            Matplotlib figure

        Raises:
            ValueError: If inputs are invalid
        """
        # Input validation
        if coordinates is None:
            raise ValueError("coordinates cannot be None")

        coords_array = np.asarray(coordinates)
        if coords_array.ndim != 2 or coords_array.shape[1] != 2:
            raise ValueError(f"coordinates must be shape (n_nodes, 2), got {coords_array.shape}")

        if coords_array.shape[0] == 0:
            raise ValueError("coordinates cannot be empty")

        if tour is not None:
            tour_array = np.asarray(tour)
            if tour_array.ndim != 1:
                raise ValueError(f"tour must be 1D array, got shape {tour_array.shape}")

            # Check that tour indices are valid
            n_nodes = coords_array.shape[0]
            if len(tour_array) > 0 and (tour_array.min() < 0 or tour_array.max() >= n_nodes):
                raise ValueError(f"tour indices must be in range [0, {n_nodes-1}], got [{tour_array.min()}, {tour_array.max()}]")

        if save_path is not None:
            save_path = Path(save_path)
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.cpu().numpy()
        if isinstance(tour, torch.Tensor):
            tour = tour.cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot nodes
        ax.scatter(coordinates[:, 0], coordinates[:, 1],
                  c='red', s=50, zorder=3, label='Cities')

        # Show node numbers if requested
        if show_numbers:
            for i, (x, y) in enumerate(coordinates):
                ax.annotate(f'{i}', (x, y), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)

        # Plot tour if provided
        if tour is not None:
            tour_coords = coordinates[tour]
            # Close the tour
            tour_coords = np.vstack([tour_coords, tour_coords[0]])

            # Plot tour edges
            ax.plot(tour_coords[:, 0], tour_coords[:, 1],
                   'b-', linewidth=2, alpha=0.7, zorder=2, label='Tour')

            # Highlight start/end
            ax.scatter(tour_coords[0, 0], tour_coords[0, 1],
                      c='green', s=100, marker='*', zorder=4, label='Start')

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_cvrp_routes(
        self,
        coordinates: Union[np.ndarray, torch.Tensor],
        routes: List[Union[np.ndarray, torch.Tensor]],
        demands: Optional[Union[np.ndarray, torch.Tensor]] = None,
        capacity: float = 1.0,
        title: str = "CVRP Routes",
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Plot CVRP routes.

        Args:
            coordinates: Node coordinates [n_nodes, 2]
            routes: List of routes (each route is a sequence of node indices)
            demands: Node demands for sizing
            capacity: Vehicle capacity
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.cpu().numpy()
        if demands is not None and isinstance(demands, torch.Tensor):
            demands = demands.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot depot
        depot = coordinates[0]
        ax.scatter(depot[0], depot[1], c='red', s=200,
                  marker='s', zorder=4, label='Depot')

        # Plot customers
        customers = coordinates[1:]
        if demands is not None:
            customer_demands = demands[1:]
            sizes = 50 + (customer_demands / capacity) * 150
        else:
            sizes = 50

        ax.scatter(customers[:, 0], customers[:, 1],
                  c='blue', s=sizes, alpha=0.7, zorder=3, label='Customers')

        # Plot routes
        colors = self.colors[:len(routes)]
        for i, (route, color) in enumerate(zip(routes, colors)):
            if len(route) <= 2:  # Skip empty routes
                continue

            route_coords = coordinates[route]
            ax.plot(route_coords[:, 0], route_coords[:, 1],
                   c=color, linewidth=3, alpha=0.8,
                   label=f'Vehicle {i+1}')

            # Mark route start
            start_coord = route_coords[0]
            ax.scatter(start_coord[0], start_coord[1],
                      c=color, s=80, marker='o', edgecolors='black', zorder=4)

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_convergence(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Training Convergence",
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Plot training convergence curves.

        Args:
            metrics: Dictionary of metric lists
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))

        if len(metrics) == 1:
            axes = [axes]

        for ax, (metric_name, values) in zip(axes, metrics.items()):
            ax.plot(values, linewidth=2, marker='o', markersize=3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} vs Epoch')
            ax.grid(True, alpha=0.3)

        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_solution_quality_distribution(
        self,
        solutions: List[float],
        optimal: Optional[float] = None,
        title: str = "Solution Quality Distribution",
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Plot distribution of solution qualities.

        Args:
            solutions: List of solution costs/objectives
            optimal: Optimal solution value (if known)
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        ax1.hist(solutions, bins=30, alpha=0.7, edgecolor='black')
        if optimal is not None:
            ax1.axvline(optimal, color='red', linestyle='--',
                       linewidth=2, label=f'Optimal: {optimal:.4f}')
        ax1.set_xlabel('Solution Cost')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Solution Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(solutions, vert=False)
        if optimal is not None:
            ax2.axvline(optimal, color='red', linestyle='--',
                       linewidth=2, label=f'Optimal: {optimal:.4f}')
        ax2.set_xlabel('Solution Cost')
        ax2.set_title('Solution Box Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_pheromone_matrix(
        self,
        pheromone: Union[np.ndarray, torch.Tensor],
        title: str = "Pheromone Matrix",
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Plot pheromone matrix as heatmap.

        Args:
            pheromone: Pheromone matrix
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        if isinstance(pheromone, torch.Tensor):
            pheromone = pheromone.cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(pheromone, cmap='viridis', aspect='equal')

        ax.set_xlabel('Node j')
        ax.set_ylabel('Node i')
        ax.set_title(title)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Pheromone Value')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_parameter_sensitivity(
        self,
        parameter_values: List[float],
        performances: List[float],
        parameter_name: str = "Parameter",
        title: str = "Parameter Sensitivity Analysis",
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Plot parameter sensitivity analysis.

        Args:
            parameter_values: Parameter values tested
            performances: Performance metrics for each value
            parameter_name: Name of the parameter
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(parameter_values, performances,
               marker='o', linewidth=2, markersize=6)
        ax.set_xlabel(parameter_name)
        ax.set_ylabel('Performance Metric')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add value labels
        for x, y in zip(parameter_values, performances):
            ax.annotate(f'{y:.4f}', (x, y), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_runtime_comparison(
        self,
        methods: List[str],
        runtimes: List[float],
        title: str = "Runtime Comparison",
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Plot runtime comparison across methods.

        Args:
            methods: Method names
            runtimes: Runtime values in seconds
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(methods, runtimes, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Method')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, runtime in zip(bars, runtimes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(runtimes)*0.01,
                   '.2f', ha='center', va='bottom', fontsize=9)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_cross_problem_comparison(
        self,
        problem_results: Dict[str, Dict[str, float]],
        metric: str = "best_cost",
        title: str = "Cross-Problem Performance Comparison",
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Plot cross-problem performance comparison.

        Args:
            problem_results: Dictionary of problem results
            metric: Metric to compare
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        problems = list(problem_results.keys())
        values = [problem_results[p].get(metric, 0) for p in problems]

        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.bar(problems, values, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Problem')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   '.4f', ha='center', va='bottom', fontsize=9)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_experiment_report(
        self,
        experiment_dir: Union[str, Path],
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Create comprehensive experiment report visualization.

        Args:
            experiment_dir: Experiment directory
            save_path: Path to save report

        Returns:
            Matplotlib figure with multiple subplots
        """
        exp_dir = Path(experiment_dir)

        # Try to load metrics from various locations
        metrics_files = [
            exp_dir / "data" / "results" / "summary.json",
            exp_dir / "data" / "metrics" / "training_metrics.json"
        ]

        fig = plt.figure(figsize=(16, 12))

        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Try to load and plot data
        metrics_data = {}
        for metrics_file in metrics_files:
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        if metrics_file.suffix == '.json':
                            metrics_data.update(json.load(f))
                except:
                    pass

        # Plot metrics if available
        if metrics_data:
            # Training metrics
            ax1 = fig.add_subplot(gs[0, 0])
            for key, values in metrics_data.items():
                if isinstance(values, list) and len(values) > 1:
                    ax1.plot(values, label=key.replace('_', ' ').title())
            ax1.set_title('Training Metrics')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Performance summary
            ax2 = fig.add_subplot(gs[0, 1])
            # Create a simple bar chart of final metrics
            final_metrics = {}
            for key, values in metrics_data.items():
                if isinstance(values, list) and values:
                    final_metrics[key] = values[-1]

            if final_metrics:
                keys = list(final_metrics.keys())
                values = list(final_metrics.values())
                ax2.bar(keys, values, alpha=0.7)
                ax2.set_title('Final Metrics')
                ax2.set_ylabel('Value')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # Placeholder for additional plots
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.text(0.5, 0.5, 'Experiment\nVisualization\nPlaceholder',
                transform=ax3.transAxes, ha='center', va='center', fontsize=14)
        ax3.set_title('Additional Analysis')
        ax3.axis('off')

        # Experiment info
        ax4 = fig.add_subplot(gs[1, :])
        info_text = f"Experiment: {exp_dir.name}\n"
        info_text += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        if metrics_data:
            info_text += f"Metrics Available: {list(metrics_data.keys())}\n"
        else:
            info_text += "No metrics data found\n"

        ax4.text(0.1, 0.5, info_text,
                transform=ax4.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace')
        ax4.set_title('Experiment Information')
        ax4.axis('off')

        fig.suptitle(f'GFACS Experiment Report - {exp_dir.name}',
                    fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


# Global visualizer instance
_visualizer: Optional[GFACSVisualizer] = None


def get_visualizer() -> GFACSVisualizer:
    """Get global visualizer instance."""
    global _visualizer
    if _visualizer is None:
        _visualizer = GFACSVisualizer()
    return _visualizer


# Convenience functions for common visualizations
def plot_tsp_solution(
    coordinates: Union[np.ndarray, torch.Tensor],
    tour: Union[np.ndarray, torch.Tensor],
    cost: float,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """Plot TSP solution.

    Args:
        coordinates: Node coordinates [n_nodes, 2]
        tour: Tour indices
        cost: Tour cost
        save_path: Path to save plot
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if title is None:
        title = f"TSP Solution (Cost: {cost:.4f})"

    visualizer = get_visualizer()
    return visualizer.plot_tsp_tour(coordinates, tour, title, save_path)


def plot_training_progress(
    metrics: Dict[str, List[float]],
    experiment_dir: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Plot training progress.

    Args:
        metrics: Training metrics
        experiment_dir: Experiment directory for title
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    exp_dir = Path(experiment_dir)
    title = f"Training Progress - {exp_dir.name}"

    visualizer = get_visualizer()
    return visualizer.plot_convergence(metrics, title, save_path)


def save_experiment_visualizations(
    experiment_dir: Union[str, Path],
    coordinates: Optional[Union[np.ndarray, torch.Tensor]] = None,
    tour: Optional[Union[np.ndarray, torch.Tensor]] = None,
    cost: Optional[float] = None,
    metrics: Optional[Dict[str, List[float]]] = None,
    pheromone_matrix: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> None:
    """Save all relevant visualizations for an experiment.

    Args:
        experiment_dir: Experiment directory
        coordinates: Node coordinates (for TSP/CVRP)
        tour: Solution tour/route
        cost: Solution cost
        metrics: Training metrics
        pheromone_matrix: Pheromone matrix for visualization
    """
    exp_dir = Path(experiment_dir)
    viz_dir = exp_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True, parents=True)

    visualizer = get_visualizer()

    # Plot solution if coordinates and tour provided
    if coordinates is not None and tour is not None:
        if cost is not None:
            title = f"Solution (Cost: {cost:.4f})"
        else:
            title = "Solution"

        if coordinates.shape[0] <= 100:  # Only plot for small instances
            visualizer.plot_tsp_tour(
                coordinates, tour, title,
                save_path=viz_dir / "solution.png"
            )

    # Plot training progress if metrics provided
    if metrics is not None and len(metrics) > 0:
        visualizer.plot_convergence(
            metrics, "Training Progress",
            save_path=viz_dir / "training_progress.png"
        )

    # Plot pheromone matrix if provided
    if pheromone_matrix is not None:
        visualizer.plot_pheromone_matrix(
            pheromone_matrix, "Final Pheromone Matrix",
            save_path=viz_dir / "pheromone_matrix.png"
        )

    # Create comprehensive report
    visualizer.create_experiment_report(
        exp_dir, save_path=viz_dir / "experiment_report.png"
    )
