# Module: Single Machine Total Weighted Tardiness Problem (`smtwtp/`)

## Overview

The Single Machine Total Weighted Tardiness Problem module implements ACO with GFlowNet sampling for scheduling jobs on a single machine. The goal is to minimize the total weighted tardiness when jobs have due times, processing times, and importance weights.

**Key Features:**
- Job scheduling optimization
- Weighted tardiness minimization
- Single machine constraint
- Time-dependent scheduling decisions
- Neural heuristic learning for job ordering

## Core Classes

### ACO Class - Single Machine Scheduling ACO

```python
class ACO:
    def __init__(
        self,
        due_time,        # [n,] - job due times
        weights,         # [n,] - job importance weights
        processing_time, # [n,] - job processing times
        n_ants=20,
        decay=0.9,
        alpha=1.0,
        beta=1.0,
        elitist=False,
        min_max=False,
        pheromone=None,
        heuristic=None,
        min=None,       # MIN value for MAX-MIN system
        device='cpu'
    )
```

**Parameters:**
- `due_time` (torch.Tensor): Job due times [n_jobs]
- `weights` (torch.Tensor): Job importance weights [n_jobs]
- `processing_time` (torch.Tensor): Job processing times [n_jobs]
- `n_ants` (int): Number of ants per iteration
- `decay` (float): Pheromone evaporation rate (0.9)
- `alpha` (float): Pheromone influence (1.0)
- `beta` (float): Heuristic influence (1.0)
- `elitist` (bool): Enable elitist ant system
- `min_max` (bool): Enable MAX-MIN ant system
- `min` (float, optional): Minimum pheromone value
- `device` (str): Computation device ('cpu', 'cuda')

#### Key Methods

**sample(invtemp=1.0, inference=False)**
Generate job schedules using ACO.

**Parameters:**
- `invtemp` (float): Inverse temperature for sampling
- `inference` (bool): Use greedy sampling

**Returns:**
- `objs` (torch.Tensor): Total weighted tardiness values [n_ants]
- `log_probs` (torch.Tensor): Action log probabilities
- `schedules` (torch.Tensor): Job schedules/permutations

**run(n_iterations)**
Execute full ACO algorithm for scheduling.

**Parameters:**
- `n_iterations` (int): Number of ACO iterations

**Returns:**
- Best total weighted tardiness found

**update_pheromone(paths, costs)**
Update pheromone trails based on solution quality.

**Parameters:**
- `paths` (torch.Tensor): Solution paths
- `costs` (torch.Tensor): Corresponding costs

**gen_path_costs(schedules)**
Compute total weighted tardiness for schedules.

**Parameters:**
- `schedules` (torch.Tensor): Job permutations [batch_size, n_jobs]

**Returns:**
- `tardiness` (torch.Tensor): Total weighted tardiness [batch_size]

## Problem Formulation

### Weighted Tardiness
For each job i in schedule:
```
completion_time_i = Σ_{j=1 to i} processing_time_{schedule[j]}
tardiness_i = max(0, completion_time_i - due_time_{schedule[i]})
weighted_tardiness_i = weights_{schedule[i]} × tardiness_i
```

**Total Objective**: Σᵢ weighted_tardiness_i (minimize)

### Key Characteristics
- **Single Machine**: Jobs processed sequentially
- **No Preemptions**: Once started, jobs run to completion
- **Due Times**: Desired completion times for each job
- **Weights**: Importance multipliers for tardiness penalties

## Utility Functions

### Data Processing

**gen_pyg_data(due_time, weights, processing_time, k_sparse)**
Create PyTorch Geometric data for scheduling.

```python
def gen_pyg_data(
    due_time: torch.Tensor,        # [n_jobs]
    weights: torch.Tensor,         # [n_jobs]
    processing_time: torch.Tensor, # [n_jobs]
    k_sparse: int
) -> tuple:
```

**Returns:**
- `pyg_data`: Graph with node features [due_time, weight, processing_time]
- `due_time`: Due time vector
- `weights`: Weight vector
- `processing_time`: Processing time vector

### Dataset Management

**load_val_dataset(n_jobs, device)**
Load or generate validation dataset.

**load_test_dataset(n_jobs, device)**
Load test dataset.

## Neural Network Architecture

### Net Class
Scheduling-specific neural network.

```python
class Net(nn.Module):
    def __init__(self, gfn=False, Z_out_dim=1):
        # Node features: [due_time, weight, processing_time]
        self.emb_net = EmbNet(feats=3)
```

**Key Features:**
- Temporal features in node embeddings
- Scheduling constraint awareness
- Priority learning for job ordering

## Training and Testing

### Training Function

```python
def train_instance(
    model,
    optimizer,
    data,  # List of (pyg_data, due_time, weights, processing_time)
    n_ants,
    cost_w=1.0,
    invtemp=1.0,
    guided_exploration=False,
    beta=100.0
)
```

### Testing Function

```python
def test(dataset, model, n_ants, t_aco, k_sparse)
```

## Usage Examples

### Basic Job Scheduling

```python
import torch
from gfacs.smtwtp.aco import ACO
from gfacs.smtwtp.net import Net
from gfacs.smtwtp.utils import gen_pyg_data

# Generate scheduling instance
n_jobs = 20
due_time = torch.rand(n_jobs) * 100 + 50      # Due times 50-150
weights = torch.rand(n_jobs) * 5 + 1          # Weights 1-6
processing_time = torch.rand(n_jobs) * 10 + 5  # Processing times 5-15

pyg_data, due_time, weights, processing_time = gen_pyg_data(
    due_time, weights, processing_time, k_sparse=10
)

# Load model and solve
model = Net(gfn=True)
model.load_state_dict(torch.load('pretrained/smtwtp/20/model.pt'))

heu_vec = model(pyg_data)
heu_mat = model.reshape(pyg_data, heu_vec)

aco = ACO(
    due_time=due_time,
    weights=weights,
    processing_time=processing_time,
    heuristic=heu_mat
)

tardiness, log_probs, schedules = aco.sample()
```

### Schedule Analysis

```python
# Analyze best schedule
best_idx = tardiness.argmin()
best_schedule = schedules[best_idx]
best_tardiness = tardiness[best_idx]

# Compute completion times
completion_times = torch.cumsum(processing_time[best_schedule], dim=0)

# Compute individual tardiness
individual_tardiness = torch.clamp(
    completion_times - due_time[best_schedule], min=0
)
weighted_tardiness = weights[best_schedule] * individual_tardiness

print(f"Total weighted tardiness: {best_tardiness}")
print(f"Average job tardiness: {individual_tardiness.mean()}")
```

### Command Line Usage

```bash
# Train SMTWTP model
python train.py 100 --epochs 50 --batch-size 20

# Test model
python test.py 100 -p pretrained/smtwtp/100/model.pt
```

## Problem Characteristics

### Scheduling Constraints
- **Single Machine**: Only one job processed at a time
- **No Preemptions**: Jobs run to completion once started
- **Sequential Processing**: Jobs processed in scheduled order
- **Start Time**: All jobs start at time 0

### Objective Components
- **Tardiness**: max(0, completion_time - due_time)
- **Weights**: Importance multipliers for each job
- **Total Cost**: Σ (weight_i × tardiness_i)
- **Early Jobs**: No penalty for completing before due time

### Decision Variables
- **Job Order**: Permutation of jobs to process
- **Completion Times**: Determined by processing order
- **Tardiness Calculation**: Based on completion vs. due times

## Performance Characteristics

### Time Complexity
- **ACO Sampling**: O(n_ants × n_jobs²) - permutation construction
- **Tardiness Calculation**: O(n_jobs) per schedule
- **Neural Forward Pass**: O(n_jobs × k_sparse × depth)

### Space Complexity
- **Job Features**: O(n_jobs × 3) - due_time, weight, processing_time
- **Pheromone Matrix**: O(n_jobs²)
- **Schedule Storage**: O(n_ants × n_jobs)

### Instance Scaling
- **Small instances** (n ≤ 50): Exact solutions possible
- **Medium instances** (50 < n ≤ 200): Good approximation quality
- **Large instances** (n > 200): Computationally intensive

## Implementation Notes

### Permutation Construction
- **Dummy Node**: Uses n+1 nodes with dummy start node
- **Path Building**: Constructs permutation through graph traversal
- **Feasibility**: All schedules are automatically feasible
- **Completeness**: All jobs scheduled exactly once

### Tardiness Computation
- **Cumulative Times**: Completion time is sum of previous processing times
- **Vectorized**: Efficient batch computation using torch.cumsum
- **Weighted**: Different importance weights for different jobs
- **Non-negative**: Only positive tardiness contributes to cost

### Neural Learning
- **Temporal Features**: Due times and processing times in embeddings
- **Priority Learning**: Learn job importance and urgency
- **Sequence Prediction**: Predict good processing orders

## Testing and Validation

### Unit Tests
- Tardiness calculation accuracy
- Schedule feasibility validation
- Completion time computation
- Weighted objective functions

### Integration Tests
- End-to-end scheduling optimization
- Permutation construction validity
- Objective function correctness

### Validation Metrics
- Average weighted tardiness
- Schedule makespan (total completion time)
- Job tardiness distributions
- Comparison with dispatching rules (EDD, WSPT, etc.)
