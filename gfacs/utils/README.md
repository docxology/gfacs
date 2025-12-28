# GFACS Utilities

## Overview

The GFACS utilities package provides essential infrastructure for experiment management, logging, I/O operations, visualization, and animation generation. These utilities enable reproducible research, comprehensive performance tracking, and rich data presentation across all GFACS problem types.

## Modules Overview

### Logging (`logging.py`)
Structured logging with file rotation, JSON output, and performance metrics tracking.

### I/O (`io.py`)
Comprehensive experiment data management including configurations, results, checkpoints, and metadata.

### Visualization (`visualization.py`)
Static plotting utilities for analysis, convergence curves, solution quality distributions, and performance comparisons.

### Animation (`animations.py`)
Dynamic visualization generation showing algorithm progress, pheromone evolution, and convergence over time.

## Quick Start

### Basic Experiment Setup

```python
from gfacs.utils import (
    setup_experiment_logging,
    setup_experiment_io,
    get_visualizer,
    get_animator
)

# Setup logging
logger = setup_experiment_logging("my_experiment")

# Setup I/O management
experiment_dir, io_manager = setup_experiment_io("my_experiment", config)

# Get visualization tools
visualizer = get_visualizer()
animator = get_animator()

logger.info("✅ Experiment infrastructure ready")
```

## Logging Usage

### Experiment Logging Setup

```python
from gfacs.utils import setup_experiment_logging

# Setup with default settings
logger = setup_experiment_logging("my_experiment")

# Setup with custom log directory
logger = setup_experiment_logging(
    "my_experiment",
    log_dir="custom_logs",
    log_level="DEBUG"
)
```

### Performance Metrics Logging

```python
# Log training metrics
metrics = {
    "loss": 0.123,
    "accuracy": 0.95,
    "best_cost": 42.5
}
logger.log_performance_metrics(metrics, prefix="epoch_10_")

# Log memory usage
logger.log_memory_usage(device="cuda")
```

### Structured Experiment Logging

```python
# Log experiment lifecycle
logger.log_experiment_start("tsp_experiment", {"size": 100, "ants": 50})

# Log problem execution
logger.log_problem_start("tsp_nls", {"iterations": 100})
logger.log_problem_end("tsp_nls", {"best_cost": 15.3}, 45.2)

# Log experiment completion
logger.log_experiment_end("tsp_experiment", {"total_cost": 15.3}, 120.5)
```

## I/O Operations

### Experiment Directory Management

```python
from gfacs.utils import setup_experiment_io

# Create experiment directory with timestamp
experiment_dir, io_manager = setup_experiment_io("benchmark_run", config)

print(f"Experiment directory: {experiment_dir}")
# outputs/benchmark_run_20241226_143022/
```

### Data Persistence

```python
# Save configuration
io_manager.save_config(config, experiment_dir)

# Save input data
input_data = {
    "coordinates": coordinates,
    "demands": demands,
    "distances": distance_matrix
}
io_manager.save_input_data(input_data, experiment_dir)

# Save problem-specific data
problem_data = {"coordinates": coords, "distances": dist}
io_manager.save_problem_instance(
    problem_data, experiment_dir, "tsp_nls", "instance_100"
)
```

### Results Management

```python
# Save experiment results
results = {
    "best_cost": 15.3,
    "mean_cost": 18.7,
    "std_cost": 2.1,
    "convergence": cost_history,
    "duration": 120.5
}
io_manager.save_results(results, experiment_dir)

# Save problem-specific results
problem_results = {
    "iterations": 100,
    "best_solution": best_tour,
    "cost_history": costs
}
io_manager.save_problem_results(
    problem_results, experiment_dir, "tsp_nls"
)
```

### Checkpoint Management

```python
# Save model checkpoint
checkpoint_path = io_manager.save_checkpoint(
    model_state_dict,
    optimizer_state_dict,
    epoch=50,
    metrics={"loss": 0.05, "cost": 15.3},
    experiment_dir,
    "tsp_nls"
)

# Load checkpoint
checkpoint = io_manager.load_checkpoint(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Visualization

### TSP Tour Visualization

```python
from gfacs.utils import get_visualizer

visualizer = get_visualizer()

# Plot TSP solution
fig = visualizer.plot_tsp_tour(
    coordinates=coordinates,
    tour=best_tour,
    title=f"TSP Solution (Cost: {best_cost:.2f})",
    save_path=experiment_dir / "visualizations" / "tsp_solution.png",
    show_numbers=True
)
```

### Convergence Curves

```python
# Plot training progress
metrics = {
    "loss": loss_history,
    "best_cost": cost_history
}

fig = visualizer.plot_convergence(
    metrics,
    title="Training Convergence",
    save_path=experiment_dir / "visualizations" / "convergence.png"
)
```

### Solution Quality Analysis

```python
# Plot solution distribution
fig = visualizer.plot_solution_quality_distribution(
    solutions=all_costs,
    optimal=known_optimal,
    title="Solution Quality Distribution",
    save_path=experiment_dir / "visualizations" / "solution_distribution.png"
)
```

### Cross-Problem Comparison

```python
# Compare performance across problems
problem_results = {
    "tsp_nls": {"best_cost": 15.3},
    "cvrp_nls": {"best_cost": 28.7},
    "op": {"best_cost": 45.2}
}

fig = visualizer.plot_cross_problem_comparison(
    problem_results,
    metric="best_cost",
    title="Problem Performance Comparison",
    save_path=experiment_dir / "visualizations" / "comparison.png"
)
```

### Pheromone Matrix Visualization

```python
# Visualize pheromone evolution
fig = visualizer.plot_pheromone_matrix(
    pheromone=pheromone_matrix,
    title="Final Pheromone Matrix",
    save_path=experiment_dir / "visualizations" / "pheromone.png"
)
```

## Animation

### Convergence Animation

```python
from gfacs.utils import get_animator

animator = get_animator()

# Create convergence animation
anim = animator.create_convergence_animation(
    cost_history=cost_history,
    title="ACO Convergence Over Time",
    save_path=experiment_dir / "animations" / "convergence.gif"
)
```

### TSP Tour Construction

```python
# Animate tour construction process
tour_history = [partial_tour_1, partial_tour_2, ..., final_tour]

anim = animator.create_tsp_construction_animation(
    coordinates=coordinates,
    tour_history=tour_history,
    costs=cost_history,
    title="TSP Tour Construction",
    save_path=experiment_dir / "animations" / "tour_construction.gif"
)
```

### Pheromone Evolution

```python
# Animate pheromone matrix changes
pheromone_history = [initial_matrix, iter1_matrix, iter2_matrix, ..., final_matrix]

anim = animator.create_pheromone_evolution_animation(
    pheromone_history=pheromone_history,
    title="Pheromone Matrix Evolution",
    save_path=experiment_dir / "animations" / "pheromone_evolution.gif"
)
```

### Multi-Problem Comparison

```python
# Animate convergence comparison across problems
problem_histories = {
    "tsp_nls": tsp_cost_history,
    "cvrp_nls": cvrp_cost_history,
    "op": op_cost_history
}

anim = animator.create_multi_problem_comparison_animation(
    problem_results=problem_histories,
    title="Multi-Problem Convergence Comparison",
    save_path=experiment_dir / "animations" / "multi_problem_comparison.gif"
)
```

## Automated Experiment Reporting

### Complete Experiment Visualization

```python
from gfacs.utils import save_experiment_visualizations, save_experiment_animations

# Generate all visualizations automatically
save_experiment_visualizations(
    experiment_dir=experiment_dir,
    coordinates=coordinates,
    tour=best_tour,
    cost=best_cost,
    metrics={"loss": loss_history, "cost": cost_history},
    pheromone_matrix=final_pheromone
)

# Generate all animations
save_experiment_animations(
    experiment_dir=experiment_dir,
    coordinates=coordinates,
    tour_history=tour_history,
    cost_history=cost_history,
    pheromone_history=pheromone_history
)
```

### Experiment Report Generation

```python
# Create comprehensive experiment report
fig = visualizer.create_experiment_report(
    experiment_dir=experiment_dir,
    save_path=experiment_dir / "visualizations" / "experiment_report.png"
)
```

## Configuration

### Custom Visualizer Settings

```python
# Custom styling
visualizer = get_visualizer()
visualizer.style = "darkgrid"  # seaborn style

# Custom color palette
visualizer.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
```

### Custom Animator Settings

```python
# High-quality animations
animator = get_animator()
animator.fps = 15  # Higher frame rate
animator.bitrate = 2400  # Higher quality

# Create MP4 instead of GIF
anim = animator.create_convergence_animation(
    cost_history,
    save_path=experiment_dir / "animations" / "convergence.mp4"
)
```

## Best Practices

### Experiment Organization

```python
# Consistent directory structure
experiment_name = f"{problem}_{size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Setup all utilities
logger = setup_experiment_logging(experiment_name)
experiment_dir, io_manager = setup_experiment_io(experiment_name, config)
visualizer = get_visualizer()
animator = get_animator()

# Log configuration
logger.info(f"Configuration: {config}")
io_manager.save_config(config, experiment_dir)
```

### Memory Management

```python
# Monitor memory usage during long experiments
logger.log_memory_usage(device="cuda")

# Clean up large objects
del large_intermediate_data
torch.cuda.empty_cache()
```

### Error Handling

```python
try:
    # Generate visualization
    fig = visualizer.plot_tsp_tour(coordinates, tour)
    visualizer._save_fig(fig, save_path)
except Exception as e:
    logger.warning(f"Visualization failed: {e}")
    # Continue with experiment
```

### Performance Optimization

```python
# Batch visualization operations
visualization_tasks = [
    ("tsp_solution", lambda: visualizer.plot_tsp_tour(coords, tour)),
    ("convergence", lambda: visualizer.plot_convergence(metrics)),
    ("pheromone", lambda: visualizer.plot_pheromone_matrix(pheromone))
]

for name, plot_func in visualization_tasks:
    try:
        fig = plot_func()
        fig.savefig(experiment_dir / "visualizations" / f"{name}.png")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to create {name}: {e}")
```

## Troubleshooting

### Common Issues

**1. Visualization Dependencies Missing**
```bash
pip install matplotlib seaborn pandas
# Or use conda: conda install matplotlib seaborn pandas
```

**2. Animation Generation Fails**
```bash
# For GIF support
pip install pillow

# For MP4 support
pip install ffmpeg-python
# Or install ffmpeg system package
```

**3. Memory Issues During Animation**
```python
# Reduce animation quality
animator = get_animator()
animator.fps = 5  # Lower frame rate
animator.bitrate = 1200  # Lower quality

# Or disable animations for large experiments
config.enable_animations = False
```

**4. File Permission Errors**
```python
# Ensure write permissions
experiment_dir.mkdir(exist_ok=True, parents=True)

# Or specify absolute paths
import os
os.makedirs(str(experiment_dir), exist_ok=True)
```

**5. Large Log Files**
```python
# Reduce logging level
logger = setup_experiment_logging(
    "experiment",
    log_level="INFO"  # Instead of DEBUG
)

# Or increase rotation size
logger = get_logger("experiment", max_bytes=50*1024*1024)  # 50MB
```

## Integration Examples

### With GFACS Orchestrator

```python
from gfacs import GFACSOrchestrator, OrchestratorConfig
from gfacs.utils import setup_experiment_logging

# Custom orchestrator with enhanced logging
config = OrchestratorConfig(
    experiment_name="enhanced_experiment",
    enable_visualizations=True,
    enable_animations=True
)

# Setup custom logging
logger = setup_experiment_logging(
    config.experiment_name,
    log_level="DEBUG"
)

orchestrator = GFACSOrchestrator(config)
results = orchestrator.run_experiment()

logger.info(f"Enhanced experiment completed: {results}")
```

### With Individual Problem Training

```python
from gfacs.tsp_nls.train import train_instance
from gfacs.utils import setup_experiment_logging, get_io_manager, get_visualizer

# Setup utilities
logger = setup_experiment_logging("tsp_training")
io_manager = get_io_manager()
visualizer = get_visualizer()

# Training loop with comprehensive logging
for epoch in range(100):
    loss = train_instance(model, optimizer, data, n_ants=50)
    logger.log_performance_metrics({"loss": loss}, prefix=f"epoch_{epoch}_")

    if epoch % 10 == 0:
        # Save checkpoint
        io_manager.save_checkpoint(
            model.state_dict(), optimizer.state_dict(),
            epoch, {"loss": loss}, experiment_dir, "tsp_nls"
        )

        # Generate progress visualization
        metrics = {"loss": loss_history[:epoch+1]}
        visualizer.plot_convergence(
            metrics,
            save_path=experiment_dir / f"progress_epoch_{epoch}.png"
        )
```

## Performance Characteristics

### Resource Usage

| Module | Memory | Disk I/O | CPU |
|--------|--------|----------|-----|
| Logging | Low | Medium | Low |
| I/O | Medium | High | Medium |
| Visualization | High | Medium | High |
| Animation | Very High | High | High |

### Scalability Guidelines

**Small Experiments (n ≤ 100):**
- All utilities enabled
- High-quality animations (15 FPS)
- Comprehensive visualizations

**Medium Experiments (100 < n ≤ 500):**
- Visualizations enabled, animations optional
- Reduce animation FPS to 5-10
- Use PNG series for large animations

**Large Experiments (n > 500):**
- Minimal visualizations
- Disable animations
- Focus on logging and data persistence

## Extension Points

### Custom Visualizations

```python
from gfacs.utils import get_visualizer

class ExtendedVisualizer(GFACSVisualizer):
    def plot_custom_analysis(self, data, **kwargs):
        fig, ax = plt.subplots()
        # Custom plotting logic
        return fig

# Use extended visualizer
visualizer = ExtendedVisualizer()
```

### Custom I/O Formats

```python
class ExtendedIOManager(ExperimentIO):
    def save_custom_data(self, data, path):
        # Custom serialization
        pass

# Use extended I/O manager
io_manager = ExtendedIOManager()
```

### Custom Logging

```python
from gfacs.utils import get_logger

class CustomLogger(GFACSLogger):
    def log_custom_metric(self, metric_name, value):
        self.logger.info(f"CUSTOM_METRIC {metric_name}: {value}")

# Use custom logger
logger = CustomLogger("experiment")
```

## License

Part of GFACS - MIT License