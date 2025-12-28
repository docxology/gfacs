# GFACS Thin-Orchestrator Documentation

## Overview

The GFACS Thin-Orchestrator is a comprehensive simulation runner that executes experiments across all 8 GFACS combinatorial optimization problems, collecting configurations, logs, full data, visualizations, and animations in a single organized output folder.

## Architecture

```
outputs/
└── {experiment_id}/                    # Single experiment folder
    ├── config/                         # All configurations
    │   ├── experiment.yaml             # Main experiment config
    │   └── orchestrator.yaml           # Orchestrator defaults
    ├── logs/                           # All logs
    │   ├── orchestrator.log           # Main orchestrator log
    │   ├── tsp_nls.log                 # Problem-specific logs
    │   └── ...
    ├── data/                           # All input/output data
    │   ├── inputs/                     # Input instances
    │   ├── outputs/                    # Solution outputs
    │   ├── results/                    # Aggregated results
    │   │   ├── summary.json
    │   │   ├── metrics.csv
    │   │   └── per_problem/
    │   └── metrics/                    # Training metrics
    ├── visualizations/                 # Static plots
    │   ├── convergence/                # Convergence curves
    │   ├── solutions/                  # Solution visualizations
    │   ├── comparisons/                # Cross-problem comparisons
    │   └── analysis/                   # Analysis plots
    ├── animations/                     # Animated visualizations
    │   ├── tsp_tours/                  # Animated TSP tours
    │   ├── cvrp_routes/                # Animated CVRP routes
    │   ├── convergence/                # Animated convergence
    │   └── pheromone/                  # Animated pheromone evolution
    └── README.md                        # Experiment documentation
```

## Quick Start

### 1. Basic Usage

```bash
# Run comprehensive experiment with all problems
gfacs-orchestrator

# Run specific problems only
gfacs-orchestrator --problems tsp_nls cvrp_nls

# Run with custom configuration
gfacs-orchestrator --config my_config.yaml --experiment-name my_experiment

# Quick test run (TSP only)
gfacs-orchestrator --quick
```

### 2. Output Structure

After running, all outputs are organized in `outputs/{experiment_name}_{timestamp}/`:

```
my_experiment_20241226_143052/
├── config/
│   ├── experiment.yaml
│   └── orchestrator.yaml
├── logs/
│   ├── orchestrator.log
│   └── tsp_nls.log
├── data/
│   ├── results/
│   │   ├── summary.json
│   │   └── per_problem/
│   └── metrics/
├── visualizations/
│   ├── cross_problem_comparison.png
│   ├── runtime_comparison.png
│   └── experiment_report.png
└── animations/
    ├── multi_problem_convergence.gif
    ├── tsp_tour_construction.gif
    └── pheromone_evolution.gif
```

## Configuration

### Default Configuration

The orchestrator uses sensible defaults for all problems:

```yaml
experiment_name: "gfacs_experiment"
problems:
  - name: "tsp_nls"
    size: 50
    enabled: true
    n_ants: 20
    n_iterations: 50
  - name: "cvrp_nls"
    size: 50
    enabled: true
    n_ants: 20
    n_iterations: 50
  # ... all 8 problems configured
enable_visualizations: true
enable_animations: true
log_level: "INFO"
```

### Custom Configuration

Create a custom `config.yaml`:

```yaml
experiment_name: "my_custom_experiment"
problems:
  - name: "tsp_nls"
    size: 100
    n_ants: 50
    n_iterations: 100
  - name: "cvrp_nls"
    size: 100
    n_ants: 50
    n_iterations: 100
enable_visualizations: true
enable_animations: true
log_level: "DEBUG"
```

Run with custom config:
```bash
gfacs-orchestrator --config config.yaml
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `experiment_name` | Name for experiment | "gfacs_experiment" |
| `problems` | List of problem configurations | All 8 problems |
| `enable_visualizations` | Generate static plots | true |
| `enable_animations` | Generate animated visualizations | true |
| `log_level` | Logging verbosity | "INFO" |
| `base_output_dir` | Base output directory | "outputs" |

#### Problem Configuration

Each problem can be configured individually:

```yaml
problems:
  - name: "tsp_nls"           # Problem identifier
    size: 100                 # Instance size
    enabled: true            # Whether to run this problem
    n_ants: 50               # Number of ants
    n_iterations: 100        # ACO iterations
    device: "cpu"            # Computation device
    extra_args:              # Problem-specific arguments
      k_sparse: 20
```

## Problems Supported

| Problem | Identifier | Description | Key Metrics |
|---------|------------|-------------|-------------|
| Traveling Salesman | `tsp_nls` | Tour optimization with neural heuristics | Tour length, computation time |
| Capacitated VRP | `cvrp_nls` | Vehicle routing with capacity constraints | Total distance, route count |
| VRP with Time Windows | `cvrptw_nls` | Routing with time window constraints | Distance, time feasibility |
| Bin Packing | `bpp` | Item assignment to capacity-constrained bins | Number of bins used |
| Orienteering | `op` | Prize maximization under distance constraints | Total prize collected |
| Prize Collecting TSP | `pctsp` | TSP with selective node visiting | Prize vs distance trade-off |
| Single Machine Scheduling | `smtwtp` | Job scheduling with weighted tardiness | Total weighted tardiness |
| Sequential Ordering | `sop` | Task ordering with precedence constraints | Processing cost |

## Output Details

### Logs

- **`orchestrator.log`**: Main orchestrator execution log
- **`{problem}.log`**: Problem-specific execution logs
- Structured logging with timestamps and log levels

### Data

#### Results Summary (`data/results/summary.json`)
```json
{
  "experiment_name": "gfacs_experiment",
  "start_time": 1703600000.0,
  "end_time": 1703600500.0,
  "duration": 500.0,
  "problems_run": 8,
  "problems_succeeded": 8,
  "problem_results": {
    "tsp_nls": {
      "status": "completed",
      "best_cost": 15.67,
      "mean_cost": 18.23,
      "std_cost": 2.45,
      "duration": 45.2,
      "iterations": 50
    }
  }
}
```

#### Per-Problem Results (`data/results/per_problem/{problem}/results.json`)
Detailed results for each problem including solution quality metrics.

#### Metrics (`data/metrics/{problem}/training_metrics.json`)
Training convergence data and performance metrics.

### Visualizations

#### Static Plots (`visualizations/`)
- **`cross_problem_comparison.png`**: Best costs across all problems
- **`runtime_comparison.png`**: Execution times comparison
- **`solution_quality_distribution.png`**: Cost distributions
- **`performance_summary.png`**: Performance metrics overview
- **`experiment_report.png`**: Comprehensive experiment summary

#### Data Tables
- **`experiment_summary.csv`**: Tabular results summary
- **`experiment_summary.md`**: Markdown-formatted summary

### Animations

#### Dynamic Visualizations (`animations/`)
- **`multi_problem_convergence.gif`**: Convergence comparison across problems
- **`tsp_tour_construction.gif`**: Step-by-step TSP tour building
- **`pheromone_evolution.gif`**: Pheromone matrix evolution over time
- **`aco_convergence.gif`**: Single problem convergence animation
- **`experiment_progress.gif`**: Overall experiment progress

## Usage Examples

### Basic Experiment

```bash
# Run all problems with defaults
gfacs-orchestrator --experiment-name "full_test"

# Output will be in: outputs/full_test_20241226_143052/
```

### Custom Problem Selection

```bash
# Run only TSP and CVRP
gfacs-orchestrator --problems tsp_nls cvrp_nls --experiment-name "routing_only"

# Run with larger instances
gfacs-orchestrator --config large_instances.yaml
```

### Configuration File Example

```yaml
# large_instances.yaml
experiment_name: "large_scale_test"
problems:
  - name: "tsp_nls"
    size: 200
    n_ants: 100
    n_iterations: 200
  - name: "cvrp_nls"
    size: 100
    n_ants: 50
    n_iterations: 100
enable_visualizations: true
enable_animations: false  # Skip animations for speed
log_level: "INFO"
```

### Quick Testing

```bash
# Quick test with just TSP
gfacs-orchestrator --quick --experiment-name "quick_test"

# Minimal output for development
gfacs-orchestrator --problems tsp_nls --enable-animations false
```

## API Usage

### Programmatic Usage

```python
from gfacs.orchestrator import GFACSOrchestrator, OrchestratorConfig, ProblemConfig

# Create custom configuration
config = OrchestratorConfig(
    experiment_name="api_test",
    problems=[
        ProblemConfig(name="tsp_nls", size=50, n_ants=20),
        ProblemConfig(name="cvrp_nls", size=50, n_ants=20)
    ],
    enable_animations=False  # Skip animations for speed
)

# Run experiment
orchestrator = GFACSOrchestrator(config)
results = orchestrator.run_experiment()

print(f"Experiment completed in {results['duration']:.2f}s")
print(f"Problems succeeded: {results['problems_succeeded']}/{results['problems_run']}")
```

### Loading Configurations

```python
from gfacs.orchestrator import load_orchestrator_config

# Load from file
config = load_orchestrator_config("my_config.yaml")

# Load defaults
config = load_orchestrator_config()
```

## Performance Considerations

### Resource Requirements

| Problem Size | Memory | Time | GPU Recommended |
|-------------|--------|------|-----------------|
| Small (≤50) | 2-4GB | 1-5min | No |
| Medium (50-200) | 4-8GB | 5-30min | Recommended |
| Large (200+) | 8-16GB+ | 30min+ | Yes |

### Optimization Tips

1. **Disable animations** for large experiments:
   ```bash
   gfacs-orchestrator --enable-animations false
   ```

2. **Run subset of problems** for testing:
   ```bash
   gfacs-orchestrator --problems tsp_nls cvrp_nls
   ```

3. **Use appropriate instance sizes** based on available resources

4. **Monitor logs** for progress and potential issues

## Troubleshooting

### Common Issues

#### Out of Memory
```
Solution: Reduce problem sizes or disable visualizations
gfacs-orchestrator --problems tsp_nls --enable-visualizations false
```

#### Missing Dependencies
```
Error: torch_geometric not found
Solution: Install dependencies
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
```

#### Long Runtime
```
Solution: Use smaller instances or fewer problems
gfacs-orchestrator --quick
```

### Log Analysis

Check logs for detailed error information:
```bash
# View main log
cat outputs/experiment_*/logs/orchestrator.log

# View problem-specific logs
cat outputs/experiment_*/logs/tsp_nls.log
```

### Debugging

Enable debug logging:
```bash
gfacs-orchestrator --log-level DEBUG
```

## Extending the Orchestrator

### Adding New Problems

1. Create problem module in `gfacs/problems/`
2. Implement standard interface (`train`, `test` functions)
3. Add problem configuration to orchestrator
4. Update visualization/animation generators

### Custom Visualizations

```python
from gfacs.orchestrator import GFACSOrchestrator

class CustomOrchestrator(GFACSOrchestrator):
    def _generate_visualizations(self, problem_results):
        # Custom visualization logic
        super()._generate_visualizations(problem_results)
        # Add custom plots...
```

## File Organization

```
/gfacs/
├── orchestrator.py           # Main orchestrator class
├── utils/
│   ├── io.py                # I/O management
│   ├── logging.py           # Logging utilities
│   ├── visualization.py     # Static plots
│   └── animations.py        # Animated visualizations
├── problems/                # Problem implementations
│   ├── tsp_nls/
│   ├── cvrp_nls/
│   └── ...
└── config/
    └── orchestrator.yaml     # Default configuration
```

## Contributing

1. Follow the established configuration format
2. Add comprehensive logging
3. Include visualization support
4. Update documentation
5. Add tests for new functionality

## License

This orchestrator is part of the GFACS project and follows the same MIT license.
