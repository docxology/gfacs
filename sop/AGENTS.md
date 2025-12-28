# Module: Sequential Ordering Problem (`sop/`)

## Overview

The Sequential Ordering Problem module implements ACO with GFlowNet sampling for SOP. The goal is to find a minimum-cost ordering of tasks subject to precedence constraints, where each task has specific processing costs and precedence relationships.

**Key Features:**
- Precedence constraint satisfaction
- Sequential ordering optimization
- Task processing costs
- Constraint-aware path construction
- Neural heuristic learning for ordering decisions

## Core Classes

### ACO Class - Sequential Ordering ACO

```python
class ACO:
    def __init__(
        self,
        distances,     # (n, n) - processing cost matrix
        prec_cons,     # (n, n) - precedence constraints
        n_ants=20,
        decay=0.9,
        alpha=1.0,
        beta=1.0,
        elitist=False,
        min_max=False,
        pheromone=None,
        heuristic=None,
        min=None,      # MIN value for MAX-MIN system
        device='cpu'
    )
```

**Parameters:**
- `distances` (torch.Tensor): Processing cost matrix [n_tasks, n_tasks]
- `prec_cons` (torch.Tensor): Precedence constraints [n_tasks, n_tasks]
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
Generate precedence-feasible task orderings.

**Parameters:**
- `invtemp` (float): Inverse temperature for sampling
- `inference` (bool): Use greedy sampling

**Returns:**
- `objs` (torch.Tensor): Total processing costs [n_ants]
- `log_probs` (torch.Tensor): Action log probabilities
- `sequences` (torch.Tensor): Task sequences

**run(n_iterations)**
Execute full ACO algorithm for sequential ordering.

**Parameters:**
- `n_iterations` (int): Number of ACO iterations

**Returns:**
- Best total cost found

**update_pheromone(paths, costs)**
Update pheromone trails based on solution quality.

**Parameters:**
- `paths` (torch.Tensor): Solution paths
- `costs` (torch.Tensor): Corresponding costs

**gen_path_costs(sequences)**
Compute total processing costs for sequences.

**Parameters:**
- `sequences` (torch.Tensor): Task sequences [batch_size, n_tasks]

**Returns:**
- `costs` (torch.Tensor): Total processing costs [batch_size]

**check_precedence(sequence)**
Validate precedence constraint satisfaction.

**Parameters:**
- `sequence` (torch.Tensor): Task sequence to validate

**Returns:**
- `is_feasible` (bool): Precedence feasibility

## Problem Formulation

### Sequential Ordering Problem
The SOP involves ordering tasks with:
- **Processing Costs**: Cost of processing task i after task j
- **Precedence Constraints**: Task i must precede task j (i → j)
- **Objective**: Minimize total processing cost of valid sequence

### Precedence Constraints
- **Matrix Format**: prec_cons[i,j] = 1 if i must precede j
- **Transitive**: If i → j and j → k, then i → k
- **Acyclic**: No cycles in precedence graph
- **Feasibility**: Sequence must respect all constraints

## Utility Functions

### Data Processing

**gen_distance_matrix(coordinates, cost_matrix=None)**
Create distance/cost matrix from coordinates or provided matrix.

**gen_pyg_data(coordinates, prec_cons, k_sparse)**
Create PyTorch Geometric data for SOP.

```python
def gen_pyg_data(
    coordinates: torch.Tensor,  # [n_tasks, 2]
    prec_cons: torch.Tensor,    # [n_tasks, n_tasks]
    k_sparse: int
) -> tuple:
```

**Returns:**
- `pyg_data`: Graph with node features [x, y]
- `distances`: Cost matrix
- `prec_cons`: Precedence constraint matrix

### Dataset Management

**load_val_dataset(n_tasks, k_sparse, device)**
Load or generate validation dataset.

**load_test_dataset(n_tasks, k_sparse, device)**
Load test dataset.

## Neural Network Architecture

### Net Class
SOP-specific neural network.

```python
class Net(nn.Module):
    def __init__(self, gfn=False, Z_out_dim=1):
        # Node features: [x, y] (coordinates)
        self.emb_net = EmbNet(feats=2)
```

**Key Features:**
- Spatial features for task positioning
- Constraint-aware decision making
- Ordering preference learning

## Training and Testing

### Training Function

```python
def train_instance(
    model,
    optimizer,
    data,  # List of (pyg_data, distances, prec_cons)
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

### Basic Sequential Ordering

```python
import torch
from gfacs.sop.aco import ACO
from gfacs.sop.net import Net
from gfacs.sop.utils import gen_pyg_data

# Generate SOP instance
n_tasks = 20
coordinates = torch.rand(n_tasks, 2)

# Create precedence constraints (random DAG)
prec_cons = torch.zeros(n_tasks, n_tasks)
for i in range(n_tasks):
    for j in range(i+1, min(i+4, n_tasks)):
        if torch.rand(1) > 0.5:
            prec_cons[i,j] = 1

pyg_data, distances, prec_cons = gen_pyg_data(
    coordinates, prec_cons, k_sparse=10
)

# Load model and solve
model = Net(gfn=True)
model.load_state_dict(torch.load('pretrained/sop/20/model.pt'))

heu_vec = model(pyg_data)
heu_mat = model.reshape(pyg_data, heu_vec)

aco = ACO(
    distances=distances,
    prec_cons=prec_cons,
    heuristic=heu_mat
)

costs, log_probs, sequences = aco.sample()
```

### Precedence Validation

```python
# Check constraint satisfaction
best_idx = costs.argmin()
best_sequence = sequences[best_idx]
is_feasible = aco.check_precedence(best_sequence)

if is_feasible:
    print(f"Best feasible sequence cost: {costs[best_idx]}")
else:
    print("Best sequence violates precedence constraints")
```

### Command Line Usage

```bash
# Train SOP model
python train.py 100 --epochs 50 --batch-size 20

# Test model
python test.py 100 -p pretrained/sop/100/model.pt
```

## Problem Characteristics

### Constraints
- **Precedence Relations**: Partial order on tasks
- **Transitivity**: Constraint closure under transitivity
- **Acyclicity**: No circular dependencies
- **Feasibility**: Topological ordering exists

### Objective Function
- **Processing Costs**: Task-dependent transition costs
- **Sequence Dependent**: Cost of i→j may differ from j→i
- **Total Cost**: Sum of transition costs in sequence
- **Minimization**: Find minimum cost valid sequence

### Decision Variables
- **Task Order**: Permutation respecting precedence
- **Feasibility**: Must satisfy all precedence constraints
- **Completeness**: All tasks must be included
- **Optimality**: Minimum total processing cost

## Performance Characteristics

### Time Complexity
- **ACO Sampling**: O(n_ants × n_tasks²) - constraint-aware construction
- **Precedence Checking**: O(n_tasks²) per sequence
- **Cost Calculation**: O(n_tasks) per sequence
- **Neural Forward Pass**: O(n_tasks × k_sparse × depth)

### Space Complexity
- **Cost Matrix**: O(n_tasks²)
- **Precedence Matrix**: O(n_tasks²)
- **Sequence Storage**: O(n_ants × n_tasks)

### Instance Scaling
- **Small instances** (n ≤ 50): Exact solutions possible
- **Medium instances** (50 < n ≤ 100): Good approximation quality
- **Constraint Density**: Performance depends on precedence graph structure

## Implementation Notes

### Precedence-Aware Construction
- **Feasible Moves**: Only select tasks that don't violate constraints
- **Partial Order**: Respect precedence relations during construction
- **Backtracking**: Avoid dead-ends in constraint graph
- **Completeness**: Ensure all tasks are eventually scheduled

### Constraint Representation
- **Adjacency Matrix**: Binary matrix for precedence relations
- **Transitive Closure**: Implicit constraint propagation
- **Feasibility Checking**: Validate sequence against constraints
- **Efficient Lookups**: Fast constraint violation detection

### Neural Learning
- **Spatial Embeddings**: Learn from task coordinates
- **Constraint Awareness**: Neural network respects precedence
- **Ordering Preferences**: Learn effective sequencing strategies
- **GFlowNet**: Learn distribution over valid orderings

## Testing and Validation

### Unit Tests
- Precedence constraint validation
- Sequence feasibility checking
- Cost calculation accuracy
- Neural network predictions

### Integration Tests
- End-to-end sequential ordering
- Constraint satisfaction rates
- Objective function optimization

### Validation Metrics
- Precedence feasibility rate
- Average sequence cost
- Constraint violation analysis
- Comparison with heuristic ordering methods
