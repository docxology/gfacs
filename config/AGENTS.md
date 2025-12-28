# Module: Configuration (`config/`)

## Overview

The Configuration module provides centralized configuration management for GFACS experiments. It includes YAML-based configuration files and Python utilities for loading, validating, and managing experiment settings across all problem types.

## Files

### orchestrator.yaml
Main configuration file for GFACS orchestrator experiments.

**Structure:**
```yaml
experiment_name: "gfacs_experiment"
description: "Full GFACS experiment running all 8 combinatorial optimization problems"

# Problem configurations
problems:
  - name: "tsp_nls"
    size: 50
    enabled: true
    n_ants: 20
    n_iterations: 50
    device: "cpu"
    extra_args:
      k_sparse: 20

# Output configuration
base_output_dir: "outputs"
enable_visualizations: true
enable_animations: true

# Logging configuration
log_level: "INFO"

# Resource configuration
max_parallel_problems: 1  # Sequential execution for stability

# Reproducibility
seed: 42
```

**Configuration Sections:**

### Experiment Configuration
- `experiment_name`: Unique identifier for the experiment
- `description`: Human-readable experiment description
- `seed`: Random seed for reproducibility

### Problem Configuration
Each problem entry supports:
- `name`: Problem identifier ('tsp_nls', 'cvrp_nls', 'cvrptw_nls', 'bpp', 'op', 'pctsp', 'smtwtp', 'sop')
- `size`: Problem instance size (number of nodes/items)
- `enabled`: Whether to run this problem
- `n_ants`: Number of ants per iteration
- `n_iterations`: Number of ACO iterations
- `device`: Computation device ('cpu', 'cuda')
- `extra_args`: Problem-specific additional arguments

### Output Configuration
- `base_output_dir`: Base directory for all outputs
- `enable_visualizations`: Generate static plots and charts
- `enable_animations`: Generate animated visualizations
- `log_level`: Logging verbosity ('DEBUG', 'INFO', 'WARNING', 'ERROR')

### Resource Management
- `max_parallel_problems`: Maximum concurrent problem execution

## Usage

### Loading Configuration

```python
from gfacs.orchestrator import load_orchestrator_config

# Load from YAML file
config = load_orchestrator_config("config/orchestrator.yaml")

# Use default configuration
config = load_orchestrator_config()
```

### Configuration Validation

The orchestrator automatically validates configuration parameters:
- Problem names must be valid GFACS problems
- Device must be 'cpu' or 'cuda'
- Problem sizes must be positive integers
- Log levels must be standard Python logging levels

## Configuration Examples

### Quick Test Configuration
```yaml
experiment_name: "quick_test"
problems:
  - name: "tsp_nls"
    size: 50
    n_ants: 10
    n_iterations: 20
enable_visualizations: false
enable_animations: false
```

### Full Benchmark Configuration
```yaml
experiment_name: "comprehensive_benchmark"
problems:
  - name: "tsp_nls"
    size: 200
    n_ants: 100
    n_iterations: 200
  - name: "cvrp_nls"
    size: 100
    n_ants: 50
  - name: "cvrptw_nls"
    size: 50
    n_ants: 30
enable_visualizations: true
enable_animations: true
log_level: "INFO"
seed: 42
```

### Development Configuration
```yaml
experiment_name: "development_run"
problems:
  - name: "bpp"
    size: 120
    n_ants: 20
    n_iterations: 10
    device: "cpu"
enable_visualizations: true
enable_animations: false
log_level: "DEBUG"
```

## Configuration Schema

### ProblemConfig Dataclass
```python
@dataclass
class ProblemConfig:
    name: str
    size: int
    enabled: bool = True
    n_ants: int = 50
    n_iterations: int = 100
    device: str = "cpu"
    extra_args: Dict[str, Any] = field(default_factory=dict)
```

### OrchestratorConfig Dataclass
```python
@dataclass
class OrchestratorConfig:
    experiment_name: str = "gfacs_experiment"
    problems: List[ProblemConfig] = field(default_factory=list)
    base_output_dir: str = "outputs"
    enable_visualizations: bool = True
    enable_animations: bool = True
    log_level: str = "INFO"
    max_parallel_problems: int = 1
    seed: int = 42
```

## Validation Rules

### Problem Validation
- Names must be one of: 'tsp_nls', 'cvrp_nls', 'cvrptw_nls', 'bpp', 'op', 'pctsp', 'smtwtp', 'sop'
- Sizes must be positive integers
- Devices must be 'cpu' or 'cuda'

### Global Validation
- At least one problem must be enabled
- Base output directory must be writable
- Log level must be valid Python logging level
- Seed must be non-negative integer

## Integration

The configuration system integrates with:
- **Orchestrator**: Main experiment execution engine
- **Problem Modules**: Individual problem implementations
- **Output Management**: Result collection and organization
- **Logging System**: Structured logging configuration

## Best Practices

### Configuration Design
- Use descriptive experiment names
- Start with small problem sizes for testing
- Enable visualizations only when needed
- Use appropriate log levels for debugging

### File Organization
- Keep configuration files version controlled
- Use consistent naming conventions
- Document custom configurations
- Separate development and production configs

### Performance Tuning
- Adjust `n_ants` and `n_iterations` based on problem size
- Use GPU acceleration when available
- Configure parallel execution carefully
- Monitor resource usage with appropriate logging