# GFACS Thin-Orchestrator

## Overview

The GFACS Thin-Orchestrator executes simulations across all 8 GFACS combinatorial optimization problems, collecting and organizing all outputs (configurations, logs, data, visualizations, and animations) in a single structured output folder.

## Key Features

✅ **Single Output Folder**: All experiment artifacts in one organized directory
✅ **Comprehensive Logging**: Structured logs for monitoring and debugging
✅ **Rich Visualizations**: Static plots and animated demonstrations
✅ **Automated Organization**: Consistent file structure across experiments
✅ **Configurable Execution**: YAML-based configuration for flexible experiments
✅ **Problem Integration**: Unified interface to all 8 GFACS problems
✅ **Performance Monitoring**: Execution times, resource usage, and metrics
✅ **Reproducibility**: Complete configuration and data preservation

## Quick Start

### Basic Usage

```bash
# Run comprehensive experiment (all 8 problems)
gfacs-orchestrator

# Run specific problems only
gfacs-orchestrator --problems tsp_nls cvrp_nls

# Quick test (TSP only)
gfacs-orchestrator --quick
```

### Output Structure

After execution, all results are organized in:
```
outputs/{experiment_name}_{timestamp}/
├── config/                    # All configurations
├── logs/                      # Execution logs
├── data/                      # Input/output data
├── visualizations/           # Static plots
└── animations/               # Animated visualizations
```

## Execution Flow

```mermaid
graph TB
    subgraph "Initialization"
        CLI[CLI Command] --> LoadConfig[Load Configuration<br/>YAML or defaults]
        LoadConfig --> Validate[Validate Parameters]
        Validate --> CreateOrch[Create Orchestrator]
    end

    subgraph "Setup Phase"
        CreateOrch --> SetupExp[Setup Experiment<br/>Directories & Infrastructure]
        SetupExp --> SetupLog[Initialize Logging<br/>Files & handlers]
        SetupExp --> SetupIO[Initialize I/O Manager<br/>Paths & formats]
        SetupExp --> InitViz[Initialize Visualizers<br/>Matplotlib, animations]
    end

    subgraph "Execution Phase"
        InitViz --> ProblemLoop[For Each Problem]
        ProblemLoop --> LoadProb[Load Problem Module<br/>Dynamic import]
        LoadProb --> RunSim[Run Simulation<br/>test.py execution]
        RunSim --> CollectRes[Collect Results<br/>Costs, metrics, data]
        CollectRes --> SaveProb[Save Problem Results<br/>JSON, data files]
        SaveProb --> NextProb{More Problems?}
        NextProb -->|Yes| ProblemLoop
    end

    subgraph "Analysis Phase"
        NextProb -->|No| GenViz[Generate Visualizations<br/>Cross-problem plots]
        GenViz --> GenAnim[Generate Animations<br/>Dynamic demonstrations]
        GenAnim --> AggRes[Aggregate Results<br/>Summary statistics]
        AggRes --> SaveSum[Save Experiment Summary<br/>Final reports]
    end

    subgraph "Cleanup"
        SaveSum --> Cleanup[Cleanup Resources<br/>Close files, connections]
        Cleanup --> Report[Generate Report<br/>Execution summary]
        Report --> Complete[Experiment Complete]
    end
```

## Installation

The orchestrator is included with GFACS. Ensure you have all dependencies:

```bash
# Install GFACS with dependencies
pip install -e .

# Or using uv
uv sync
```

## Configuration

### Default Configuration

Runs all 8 problems with sensible defaults:
- TSP, CVRP, CVRPTW, BPP, OP, PCTSP, SMTWTP, SOP
- Small instances (size 50) for quick execution
- All visualizations and animations enabled
- Comprehensive logging

### Custom Configuration

Create `config/orchestrator.yaml`:

```yaml
experiment_name: "my_experiment"
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
log_level: "INFO"
```

Run with custom config:
```bash
gfacs-orchestrator --config config/orchestrator.yaml
```

## Generated Outputs

### 📊 Visualizations
- **Cross-problem comparisons**: Best costs across all problems
- **Runtime analysis**: Execution time comparisons
- **Solution distributions**: Quality histograms and box plots
- **Performance summaries**: Comprehensive metrics overview

### 🎬 Animations
- **Multi-problem convergence**: Simultaneous convergence tracking
- **TSP tour construction**: Step-by-step tour building animation
- **Pheromone evolution**: Dynamic pheromone matrix visualization
- **ACO convergence**: Single-problem optimization progress
- **Experiment progress**: Overall execution status

### 📈 Data & Metrics
- **Complete results**: JSON summaries with statistics
- **Performance metrics**: CSV exports for analysis
- **Training data**: Convergence curves and intermediate results
- **Configuration preservation**: Exact settings used

### 📝 Documentation
- **Experiment README**: Auto-generated summary with key findings
- **Structured logs**: Timestamped execution records
- **Metadata**: Complete experiment context and parameters

## Usage Examples

### Development Testing

```bash
# Quick TSP-only test
gfacs-orchestrator --quick --experiment-name "dev_test"

# Debug mode with detailed logging
gfacs-orchestrator --problems tsp_nls --log-level DEBUG
```

### Production Experiments

```bash
# Full experiment with custom settings
gfacs-orchestrator --config production.yaml --experiment-name "prod_run_v1"

# Large-scale evaluation
gfacs-orchestrator --config large_scale.yaml
```

### Research Benchmarking

```bash
# Standardized benchmark across all problems
gfacs-orchestrator --config benchmark.yaml --experiment-name "benchmark_$(date +%Y%m%d)"

# Compare specific algorithms
gfacs-orchestrator --problems tsp_nls cvrp_nls --experiment-name "routing_comparison"
```

## Performance Characteristics

### Execution Times (Approximate)

| Configuration | Problems | Time | Memory |
|---------------|----------|------|--------|
| Quick (TSP only) | 1 | 2-5 min | 2GB |
| Default (all) | 8 | 20-60 min | 8GB |
| Large instances | 8 | 2-6 hours | 16GB+ |

### Resource Requirements

- **CPU**: Multi-core recommended for parallel execution
- **Memory**: 8GB+ for full experiments
- **Storage**: 1GB+ per experiment for all outputs
- **GPU**: Optional, improves performance for neural components

## Advanced Usage

### Programmatic API

```python
from gfacs.orchestrator import GFACSOrchestrator, OrchestratorConfig

# Custom experiment
config = OrchestratorConfig(
    experiment_name="custom_run",
    problems=[
        ProblemConfig(name="tsp_nls", size=100, n_ants=50),
    ],
    enable_animations=False  # Skip for speed
)

orchestrator = GFACSOrchestrator(config)
results = orchestrator.run_experiment()
```

### Configuration Templates

#### Quick Testing
```yaml
experiment_name: "quick_test"
problems:
  - name: "tsp_nls"
    size: 20
    n_ants: 10
    n_iterations: 10
enable_visualizations: true
enable_animations: false
```

#### Full Benchmark
```yaml
experiment_name: "full_benchmark"
problems:
  - name: "tsp_nls"
    size: 100
    n_ants: 50
    n_iterations: 100
  - name: "cvrp_nls"
    size: 100
    n_ants: 50
    n_iterations: 100
  # ... all problems
enable_visualizations: true
enable_animations: true
log_level: "INFO"
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce problem sizes or disable visualizations
2. **Long Runtimes**: Use `--quick` or reduce iterations
3. **Missing Dependencies**: Check PyTorch Geometric installation
4. **Permission Errors**: Ensure write access to output directory

### Debugging

```bash
# Enable detailed logging
gfacs-orchestrator --log-level DEBUG --problems tsp_nls

# Check logs after failure
cat outputs/*/logs/orchestrator.log
```

## Integration with GFACS

The orchestrator seamlessly integrates with all GFACS components:

- **Problems**: All 8 combinatorial optimization problems
- **Algorithms**: ACO variants, neural heuristics, local search
- **Data**: Built-in datasets and custom instance support
- **Training**: Neural network training with experiment tracking
- **Testing**: Standardized evaluation and benchmarking

## Architecture

### Core Components

1. **Experiment Management**: Configuration loading, directory setup
2. **Problem Execution**: Standardized interface to all GFACS problems
3. **Data Collection**: Automatic gathering of results and metrics
4. **Visualization Generation**: Static plots and animated demonstrations
5. **Output Organization**: Structured file system with metadata

### Extension Points

- **Custom Problems**: Add new optimization problems
- **Visualization Plugins**: Extend plotting capabilities
- **Animation Generators**: Add new animation types
- **Metrics Collectors**: Custom performance monitoring

## Contributing

1. Maintain backward compatibility
2. Add comprehensive logging
3. Include visualization support
4. Update configuration schemas
5. Add tests for new features

## License

Part of GFACS - MIT License
