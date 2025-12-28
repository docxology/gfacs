"""Animation utilities for GFACS experiments.

This module provides animation generation for visualizing algorithm progress,
solution construction, and dynamic processes in combinatorial optimization.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import imageio
import datetime

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Set seaborn style
sns.set_style("whitegrid")


class GFACSAnimator:
    """Animation utilities for GFACS experiments."""

    def __init__(self, fps: int = 10, bitrate: int = 1800):
        """Initialize animator.

        Args:
            fps: Frames per second for animations
            bitrate: Video bitrate

        Raises:
            ValueError: If fps or bitrate are invalid
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if bitrate <= 0:
            raise ValueError(f"bitrate must be positive, got {bitrate}")

        self.fps = fps
        self.bitrate = bitrate
        self.colors = sns.color_palette("husl", 10)

    def create_tsp_construction_animation(
        self,
        coordinates: Union[np.ndarray, torch.Tensor],
        tour_history: List[Union[np.ndarray, torch.Tensor]],
        costs: Optional[List[float]] = None,
        title: str = "TSP Tour Construction",
        save_path: Optional[Union[str, Path]] = None
    ) -> animation.Animation:
        """Create animation of TSP tour construction.

        Args:
            coordinates: Node coordinates [n_nodes, 2]
            tour_history: List of partial tours at each step
            costs: Cost history for display
            title: Animation title
            save_path: Path to save animation

        Returns:
            Matplotlib animation object

        Raises:
            ValueError: If inputs are invalid
        """
        # Input validation
        if coordinates is None:
            raise ValueError("coordinates cannot be None")

        coords_array = np.asarray(coordinates)
        if coords_array.ndim != 2 or coords_array.shape[1] != 2:
            raise ValueError(f"coordinates must be shape (n_nodes, 2), got {coords_array.shape}")

        if not tour_history:
            raise ValueError("tour_history cannot be empty")

        if len(tour_history) == 0:
            raise ValueError("tour_history must contain at least one tour")

        if costs is not None and len(costs) != len(tour_history):
            raise ValueError(f"costs length {len(costs)} must match tour_history length {len(tour_history)}")

        if save_path is not None:
            save_path = Path(save_path)
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot all nodes
        nodes = ax.scatter(coordinates[:, 0], coordinates[:, 1],
                          c='red', s=50, zorder=3, label='Cities')

        # Initialize tour line
        tour_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, zorder=2, label='Tour')

        # Initialize current position marker
        current_marker, = ax.plot([], [], 'go', markersize=15, zorder=4, label='Current')

        # Setup plot
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        def init():
            tour_line.set_data([], [])
            current_marker.set_data([], [])
            return tour_line, current_marker

        def animate(frame):
            if frame < len(tour_history):
                tour = tour_history[frame]
                if isinstance(tour, torch.Tensor):
                    tour = tour.cpu().numpy()

                # Plot current tour
                if len(tour) > 1:
                    tour_coords = coordinates[tour]
                    tour_line.set_data(tour_coords[:, 0], tour_coords[:, 1])

                # Highlight current position
                if len(tour) > 0:
                    current_pos = coordinates[tour[-1]]
                    current_marker.set_data([current_pos[0]], [current_pos[1]])

                # Update title with cost if available
                if costs and frame < len(costs):
                    ax.set_title(f"{title} - Cost: {costs[frame]:.4f}")
                else:
                    ax.set_title(f"{title} - Step {frame}")

            return tour_line, current_marker

        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(tour_history), interval=500, blit=True
        )

        if save_path:
            self._save_animation(anim, save_path)

        return anim

    def create_pheromone_evolution_animation(
        self,
        pheromone_history: List[Union[np.ndarray, torch.Tensor]],
        title: str = "Pheromone Matrix Evolution",
        save_path: Optional[Union[str, Path]] = None
    ) -> animation.Animation:
        """Create animation of pheromone matrix evolution.

        Args:
            pheromone_history: List of pheromone matrices over time
            title: Animation title
            save_path: Path to save animation

        Returns:
            Matplotlib animation object
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Initialize heatmap
        first_matrix = pheromone_history[0]
        if isinstance(first_matrix, torch.Tensor):
            first_matrix = first_matrix.cpu().numpy()

        im = ax.imshow(first_matrix, cmap='viridis', aspect='equal', animated=True)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Pheromone Value')

        ax.set_xlabel('Node j')
        ax.set_ylabel('Node i')
        ax.set_title(title)

        def animate(frame):
            matrix = pheromone_history[frame]
            if isinstance(matrix, torch.Tensor):
                matrix = matrix.cpu().numpy()

            im.set_array(matrix)
            ax.set_title(f"{title} - Iteration {frame}")
            return [im]

        anim = animation.FuncAnimation(
            fig, animate, frames=len(pheromone_history),
            interval=500, blit=True
        )

        if save_path:
            self._save_animation(anim, save_path)

        return anim

    def create_convergence_animation(
        self,
        cost_history: List[float],
        title: str = "ACO Convergence",
        save_path: Optional[Union[str, Path]] = None
    ) -> animation.Animation:
        """Create animation of convergence over iterations.

        Args:
            cost_history: List of best costs over iterations
            title: Animation title
            save_path: Path to save animation

        Returns:
            Matplotlib animation object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Initialize empty plot
        line, = ax.plot([], [], 'b-', linewidth=2, marker='o', markersize=4)
        best_point, = ax.plot([], [], 'ro', markersize=8)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Cost')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Set axis limits
        max_cost = max(cost_history) if cost_history else 1.0
        min_cost = min(cost_history) if cost_history else 0.0
        ax.set_xlim(0, len(cost_history))
        ax.set_ylim(min_cost * 0.95, max_cost * 1.05)

        def init():
            line.set_data([], [])
            best_point.set_data([], [])
            return line, best_point

        def animate(frame):
            # Show data up to current frame
            x_data = list(range(frame + 1))
            y_data = cost_history[:frame + 1]

            line.set_data(x_data, y_data)

            # Highlight current best
            if y_data:
                best_idx = np.argmin(y_data)
                best_point.set_data([x_data[best_idx]], [y_data[best_idx]])

            ax.set_title(f"{title} - Iteration {frame + 1}")
            return line, best_point

        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(cost_history), interval=300, blit=True
        )

        if save_path:
            self._save_animation(anim, save_path)

        return anim

    def create_cvrp_route_animation(
        self,
        coordinates: Union[np.ndarray, torch.Tensor],
        route_history: List[List[Union[np.ndarray, torch.Tensor]]],
        title: str = "CVRP Route Construction",
        save_path: Optional[Union[str, Path]] = None
    ) -> animation.Animation:
        """Create animation of CVRP route construction.

        Args:
            coordinates: Node coordinates [n_nodes, 2]
            route_history: List of routes at each step (list of routes per vehicle)
            title: Animation title
            save_path: Path to save animation

        Returns:
            Matplotlib animation object
        """
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot depot and customers
        depot = coordinates[0]
        ax.scatter(depot[0], depot[1], c='red', s=200, marker='s', zorder=4, label='Depot')
        customers = coordinates[1:]
        ax.scatter(customers[:, 0], customers[:, 1], c='blue', s=50, alpha=0.7, zorder=3, label='Customers')

        # Initialize route lines for each vehicle
        route_lines = []
        colors = self.colors[:len(route_history[0]) if route_history else 3]  # Default 3 vehicles

        for i, color in enumerate(colors):
            line, = ax.plot([], [], c=color, linewidth=3, alpha=0.8,
                           label=f'Vehicle {i+1}', zorder=2)
            route_lines.append(line)

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        def init():
            for line in route_lines:
                line.set_data([], [])
            return route_lines

        def animate(frame):
            if frame < len(route_history):
                routes = route_history[frame]

                for i, (line, color) in enumerate(zip(route_lines, colors)):
                    if i < len(routes):
                        route = routes[i]
                        if isinstance(route, torch.Tensor):
                            route = route.cpu().numpy()

                        if len(route) > 1:
                            route_coords = coordinates[route]
                            line.set_data(route_coords[:, 0], route_coords[:, 1])
                        else:
                            line.set_data([], [])

                ax.set_title(f"{title} - Step {frame}")

            return route_lines

        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(route_history), interval=500, blit=True
        )

        if save_path:
            self._save_animation(anim, save_path)

        return anim

    def create_multi_problem_comparison_animation(
        self,
        problem_results: Dict[str, List[float]],
        title: str = "Multi-Problem Performance Comparison",
        save_path: Optional[Union[str, Path]] = None
    ) -> animation.Animation:
        """Create animation comparing multiple problems.

        Args:
            problem_results: Dict of problem_name -> cost_history
            title: Animation title
            save_path: Path to save animation

        Returns:
            Matplotlib animation object
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Initialize lines for each problem
        lines = {}
        colors = self.colors[:len(problem_results)]

        for (problem_name, cost_history), color in zip(problem_results.items(), colors):
            line, = ax.plot([], [], c=color, linewidth=2, marker='o', markersize=4,
                           label=problem_name.replace('_', ' ').title())
            lines[problem_name] = line

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Cost')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set axis limits based on all data
        all_costs = [cost for costs in problem_results.values() for cost in costs]
        if all_costs:
            ax.set_ylim(min(all_costs) * 0.95, max(all_costs) * 1.05)
            max_len = max(len(costs) for costs in problem_results.values())
            ax.set_xlim(0, max_len)

        def init():
            for line in lines.values():
                line.set_data([], [])
            return list(lines.values())

        def animate(frame):
            for problem_name, cost_history in problem_results.items():
                if frame < len(cost_history):
                    x_data = list(range(frame + 1))
                    y_data = cost_history[:frame + 1]
                    lines[problem_name].set_data(x_data, y_data)

            ax.set_title(f"{title} - Iteration {frame + 1}")
            return list(lines.values())

        max_frames = max(len(costs) for costs in problem_results.values())
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=max_frames, interval=300, blit=True
        )

        if save_path:
            self._save_animation(anim, save_path)

        return anim

    def _save_animation(
        self,
        anim: animation.Animation,
        save_path: Union[str, Path],
        writer: str = 'pillow'
    ) -> None:
        """Save animation to file.

        Args:
            anim: Matplotlib animation
            save_path: Path to save animation
            writer: Animation writer ('pillow' for GIF, 'ffmpeg' for MP4)
        """
        save_path = Path(save_path)

        if save_path.suffix.lower() == '.gif':
            writer = 'pillow'
        elif save_path.suffix.lower() == '.mp4':
            writer = 'ffmpeg'
        else:
            # Default to GIF
            save_path = save_path.with_suffix('.gif')
            writer = 'pillow'

        try:
            if writer == 'pillow':
                anim.save(save_path, writer='pillow', fps=self.fps)
            elif writer == 'ffmpeg':
                Writer = animation.writers['ffmpeg']
                writer_obj = Writer(fps=self.fps, bitrate=self.bitrate)
                anim.save(save_path, writer=writer_obj)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Failed to save animation: {e}")
            # Fallback: save as series of PNGs
            self._save_as_png_series(anim, save_path)

    def _save_as_png_series(
        self,
        anim: animation.Animation,
        base_path: Union[str, Path]
    ) -> None:
        """Save animation as series of PNG files.

        Args:
            anim: Matplotlib animation
            base_path: Base path for PNG files
        """
        base_path = Path(base_path)
        png_dir = base_path.parent / f"{base_path.stem}_frames"
        png_dir.mkdir(exist_ok=True)

        # Save each frame as PNG
        for i in range(anim._frames):
            anim._func(i)  # Update animation to frame i
            frame_path = png_dir / "04d"
            anim._fig.savefig(frame_path, dpi=100, bbox_inches='tight')

        print(f"Animation frames saved to {png_dir}")


# Global animator instance
_animator: Optional[GFACSAnimator] = None


def get_animator() -> GFACSAnimator:
    """Get global animator instance."""
    global _animator
    if _animator is None:
        _animator = GFACSAnimator()
    return _animator


# Convenience functions for common animations
def create_tsp_tour_animation(
    coordinates: Union[np.ndarray, torch.Tensor],
    tour_history: List[Union[np.ndarray, torch.Tensor]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "TSP Tour Construction"
) -> animation.Animation:
    """Create TSP tour construction animation.

    Args:
        coordinates: Node coordinates
        tour_history: History of tour construction
        save_path: Path to save animation
        title: Animation title

    Returns:
        Matplotlib animation
    """
    animator = get_animator()
    return animator.create_tsp_construction_animation(
        coordinates, tour_history, title=title, save_path=save_path
    )


def create_convergence_animation(
    cost_history: List[float],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "ACO Convergence"
) -> animation.Animation:
    """Create convergence animation.

    Args:
        cost_history: History of best costs
        save_path: Path to save animation
        title: Animation title

    Returns:
        Matplotlib animation
    """
    animator = get_animator()
    return animator.create_convergence_animation(
        cost_history, title=title, save_path=save_path
    )


def create_pheromone_animation(
    pheromone_history: List[Union[np.ndarray, torch.Tensor]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Pheromone Evolution"
) -> animation.Animation:
    """Create pheromone evolution animation.

    Args:
        pheromone_history: History of pheromone matrices
        save_path: Path to save animation
        title: Animation title

    Returns:
        Matplotlib animation
    """
    animator = get_animator()
    return animator.create_pheromone_evolution_animation(
        pheromone_history, title=title, save_path=save_path
    )


def create_animation(
    animation_type: str,
    **kwargs
) -> animation.Animation:
    """Create animation of specified type.

    Args:
        animation_type: Type of animation ('tsp', 'convergence', 'pheromone', 'cvrp', 'comparison')
        **kwargs: Animation-specific arguments

    Returns:
        Matplotlib animation
    """
    animator = get_animator()

    if animation_type == 'tsp':
        return animator.create_tsp_construction_animation(**kwargs)
    elif animation_type == 'convergence':
        return animator.create_convergence_animation(**kwargs)
    elif animation_type == 'pheromone':
        return animator.create_pheromone_evolution_animation(**kwargs)
    elif animation_type == 'cvrp':
        return animator.create_cvrp_route_animation(**kwargs)
    elif animation_type == 'comparison':
        return animator.create_multi_problem_comparison_animation(**kwargs)
    else:
        raise ValueError(f"Unknown animation type: {animation_type}")


def save_experiment_animations(
    experiment_dir: Union[str, Path],
    coordinates: Optional[Union[np.ndarray, torch.Tensor]] = None,
    tour_history: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
    cost_history: Optional[List[float]] = None,
    pheromone_history: Optional[List[Union[np.ndarray, torch.Tensor]]] = None
) -> None:
    """Save all relevant animations for an experiment.

    Args:
        experiment_dir: Experiment directory
        coordinates: Node coordinates for TSP
        tour_history: TSP tour construction history
        cost_history: Convergence cost history
        pheromone_history: Pheromone matrix evolution
    """
    exp_dir = Path(experiment_dir)
    anim_dir = exp_dir / "animations"
    anim_dir.mkdir(exist_ok=True, parents=True)

    animator = get_animator()

    # TSP tour construction animation
    if coordinates is not None and tour_history is not None:
        animator.create_tsp_construction_animation(
            coordinates, tour_history,
            title="TSP Tour Construction",
            save_path=anim_dir / "tsp_tour.gif"
        )

    # Convergence animation
    if cost_history is not None and len(cost_history) > 1:
        animator.create_convergence_animation(
            cost_history,
            title="ACO Convergence",
            save_path=anim_dir / "convergence.gif"
        )

    # Pheromone evolution animation
    if pheromone_history is not None and len(pheromone_history) > 1:
        animator.create_pheromone_evolution_animation(
            pheromone_history,
            title="Pheromone Evolution",
            save_path=anim_dir / "pheromone_evolution.gif"
        )


def get_animator() -> GFACSAnimator:
    """Get global animator instance.

    Returns:
        GFACSAnimator: Global animator instance
    """
    return GFACSAnimator()
