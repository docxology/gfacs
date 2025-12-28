# Module: Bin Packing Problem (`bpp/`)

## Overview

The Bin Packing Problem module implements ACO with GFlowNet sampling for the one-dimensional bin packing problem. This module focuses on assigning items to bins of fixed capacity to minimize the number of bins used.

**Key Features:**
- Capacity-constrained item assignment
- Fitness-based optimization (bin usage efficiency)
- Neural heuristic learning for packing decisions
- Bin packing specific ACO adaptations
- No local search (NP-hard bin packing)

## Core Classes

### ACO Class - Bin Packing ACO

```python
class ACO:
    def __init__(
        self,  # 0: depot
        demand,   # (n, ) - item sizes
        capacity=1.0,    # Bin capacity (normalized)
        n_ants=20,
        decay=0.9,
        alpha=1,
        beta=1,
        elitist=False,
        pheromone=None,
        heuristic=None,
        device='cpu'
    )
```

**Parameters:**
- `demand` (torch.Tensor): Item sizes [n_items] (normalized to [0, 1])
- `capacity` (float): Bin capacity (normalized to 1.0, default: 1.0)
- `n_ants` (int): Number of ants per iteration (default: 20)
- `decay` (float): Pheromone evaporation rate (default: 0.9)
- `alpha` (int): Pheromone influence (default: 1)
- `beta` (int): Heuristic influence (default: 1)
- `elitist` (bool): Enable elitist ant system (default: False)
- `pheromone` (torch.Tensor, optional): Initial pheromone matrix
- `heuristic` (torch.Tensor, optional): Heuristic matrix
- `device` (str): Computation device ('cpu', 'cuda', default: 'cpu')

#### Key Methods

**sample(invtemp=1.0, return_sol=False, K=1)**
Generate bin packing solutions using ACO.

**Parameters:**
- `invtemp` (float): Inverse temperature for sampling
- `return_sol` (bool): Return solutions along with objectives
- `K` (int): Number of solution attempts per ant

**Returns:**
- `objs` (torch.Tensor): Fitness values (lower is better)
- `log_probs` (torch.Tensor): Action log probabilities
- `sols` (torch.Tensor, optional): Packing solutions if return_sol=True

**run(n_iterations)**
Execute full ACO algorithm for bin packing.

**Parameters:**
- `n_iterations` (int): Number of ACO iterations

**Returns:**
- Best fitness value found

**update_pheromone(sols, fits)**
Update pheromone trails based on solution quality.

**Parameters:**
- `sols` (torch.Tensor): Packing solutions
- `fits` (torch.Tensor): Corresponding fitness values

**gen_sol(invtemp=1.0, require_prob=False)**
Generate single bin packing solution.

**Parameters:**
- `invtemp` (float): Inverse temperature
- `require_prob` (bool): Return log probabilities

**Returns:**
- `sol` (torch.Tensor): Packing solution [n_bins, n_items] (binary)
- `log_probs` (torch.Tensor, optional): Log probabilities if required

**gen_sol_objs(sols)**
Compute fitness values for packing solutions.

**Parameters:**
- `sols` (torch.Tensor): Packing solutions [batch_size, n_bins, n_items]

**Returns:**
- `objs` (torch.Tensor): Fitness values (sum of squared bin usages)

## Utility Functions

### Fitness Calculation

**cal_fitness(s, demand, n_bins)**
Compute fitness for bin packing solutions (numba-compiled).

```python
@numba.njit()
def cal_fitness(s: np.ndarray, demand: np.ndarray, n_bins: np.ndarray) -> np.ndarray:
    """
    Calculate fitness as sum of squared bin utilization.

    Args:
        s: Solution matrix [batch_size, max_bins, n_items]
        demand: Item demands [n_items]
        n_bins: Number of bins used per solution [batch_size]

    Returns:
        Fitness values [batch_size]
    """
```

### Data Processing

**gen_pyg_data(demand, k_sparse)**
Create PyTorch Geometric data for bin packing.

```python
def gen_pyg_data(demand: torch.Tensor, k_sparse: int) -> torch_geometric.data.Data:
    """
    Convert demand vector to graph format.

    Args:
        demand: Item demands [n_items]
        k_sparse: Sparsity parameter

    Returns:
        PyTorch Geometric Data instance
    """
```

### Dataset Management

**load_val_dataset(n_items, device)**
Load or generate validation dataset.

**load_test_dataset(n_items, device)**
Load test dataset.

## Neural Network Architecture

### Net Class
Bin packing specific neural network.

```python
class Net(nn.Module):
    def __init__(self, gfn=False, Z_out_dim=1):
        # Node features: [demand] (single feature per item)
        self.emb_net = EmbNet(feats=1)
```

**Key Characteristics:**
- Single feature per item (demand/size)
- Edge features represent item-bin compatibility
- Heuristic predicts packing preferences

## Training and Testing

### Training Function

```python
def train_instance(
    model,
    optimizer,
    data,  # List of pyg_data instances
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

### Basic Bin Packing

```python
import torch
from gfacs.bpp.aco import ACO
from gfacs.bpp.net import Net
from gfacs.bpp.utils import gen_pyg_data

# Generate bin packing instance
n_items = 100
demand = torch.rand(n_items) * 0.8 + 0.1  # Items sized 0.1-0.9
capacity = 1.0

pyg_data = gen_pyg_data(demand, k_sparse=20)

# Load model and solve
model = Net(gfn=True)
model.load_state_dict(torch.load('pretrained/bpp/100/model.pt'))

heu_vec = model(pyg_data)
heu_mat = model.reshape(pyg_data, heu_vec)

aco = ACO(demand=demand, capacity=capacity, heuristic=heu_mat)

# Sample solutions
objs, log_probs, sols = aco.sample(return_sol=True)
best_fitness = objs.min()
best_solution = sols[objs.argmin()]
```

### Fitness Evaluation

```python
# Evaluate solution quality
n_bins_used = (sols.sum(dim=-1) > 0).sum(dim=-1)  # Bins with items
fitness = cal_fitness(sols.numpy(), demand.numpy(), n_bins_used.numpy())

print(f"Best fitness: {fitness.min()}")
print(f"Bins used: {n_bins_used[fitness.argmin()]}")
```

### Command Line Usage

```bash
# Train BPP model
python train.py 120 --epochs 50 --batch-size 20

# Test model
python test.py 120 -p pretrained/bpp/120/model.pt
```

## Problem Representation

### Solution Format
- **Matrix**: [n_bins, n_items] binary matrix
- **Row i**: Items assigned to bin i
- **Column j**: Bin assignment of item j
- **Constraints**: Sum of demands per bin ≤ capacity

### Fitness Function
- **Formula**: Σᵢ (usageᵢ)² where usageᵢ = demand_sumᵢ / capacity
- **Optimization**: Minimize fitness (encourages balanced bin usage)
- **Perfect Packing**: Fitness approaches 0 for well-balanced solutions

### Heuristic Information
- **Item-Bin Compatibility**: Based on remaining capacity
- **Neural Learning**: GFlowNet learns effective packing strategies
- **Pheromone**: Tracks successful item-bin assignments

## Performance Characteristics

### Time Complexity
- **ACO Sampling**: O(n_ants × n_items × n_bins)
- **Fitness Calculation**: O(batch_size × n_bins × n_items)
- **Neural Forward Pass**: O(n_items × k_sparse × depth)

### Space Complexity
- **Solution Matrix**: O(n_bins × n_items)
- **Pheromone Matrix**: O(n_items × n_items)
- **Neural Network**: O(n_items × k_sparse × units)

### Instance Scaling
- **Small instances** (n ≤ 100): Fast, exact solutions possible
- **Large instances** (n > 500): Approximation algorithms needed
- **Capacity ratios**: Performance depends on item size distribution

## Implementation Notes

### Bin Packing Specifics
- **No Local Search**: Bin packing has no known efficient local search
- **Multiple Solutions**: Each ant generates multiple packing attempts
- **Fitness Landscape**: Deceptive landscape requires good exploration
- **Capacity Constraints**: Hard constraints checked during construction

### Neural Architecture
- **Single Feature**: Only item demand as node feature
- **Edge Features**: Compatibility based on size differences
- **GFlowNet Mode**: Learns distribution over packing strategies
- **Parameter Learning**: Learns when to start new bins

### Validation
- **Feasibility**: All solutions satisfy capacity constraints
- **Completeness**: All items assigned to exactly one bin
- **Optimality**: Minimize number of bins (equivalent to minimize fitness)

## Testing and Validation

### Unit Tests
- Solution feasibility checking
- Fitness calculation accuracy
- Capacity constraint validation
- Neural network forward passes

### Integration Tests
- End-to-end bin packing solving
- Model training convergence
- Multiple solution quality comparison

### Validation Metrics
- Solution feasibility rate (should be 100%)
- Average bin utilization
- Fitness value distributions
- Comparison with baseline heuristics
