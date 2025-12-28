# Module: Orienteering Problem (`op/`)

## Overview

The Orienteering Problem module implements ACO with GFlowNet sampling for the orienteering problem. The goal is to find a path that maximizes collected prizes while respecting a maximum tour length constraint.

**Key Features:**
- Prize maximization under distance constraints
- Selective node visiting (not all nodes need to be visited)
- Path length constraints
- Neural heuristic learning for prize collection decisions
- MAX-MIN pheromone bounds for stability

## Core Classes

### ACO Class - Orienteering Problem ACO

```python
class ACO:
    def __init__(
        self,
        distances,     # (n, n) - distance matrix
        prizes,        # (n, ) - node prizes (depot = 0)
        max_len,       # Maximum tour length
        n_ants=20,
        decay=0.9,
        alpha=1.0,
        beta=1.0,
        elitist=False,
        min_max=False,
        pheromone=None,
        heuristic=None,
        min=None,      # MIN value for MAX-MIN system
        device='cpu',
        k_sparse=None
    )
```

**Parameters:**
- `distances` (torch.Tensor): Distance matrix [n_nodes, n_nodes]
- `prizes` (torch.Tensor): Node prizes [n_nodes] (depot prize = 0)
- `max_len` (float): Maximum allowed tour length
- `n_ants` (int): Number of ants per iteration
- `decay` (float): Pheromone evaporation rate (0.9)
- `alpha` (float): Pheromone influence (1.0)
- `beta` (float): Heuristic influence (1.0)
- `elitist` (bool): Enable elitist ant system
- `min_max` (bool): Enable MAX-MIN ant system
- `min` (float, optional): Minimum pheromone value for MAX-MIN
- `device` (str): Computation device ('cpu', 'cuda')

#### Key Methods

**sample(invtemp=1.0, inference=False, start_node=0)**
Generate prize-collecting paths under length constraints.

**Parameters:**
- `invtemp` (float): Inverse temperature for sampling
- `inference` (bool): Use greedy sampling for inference
- `start_node` (int): Starting node index (0 for depot)

**Returns:**
- `objs` (torch.Tensor): Prize values (higher is better) [n_ants]
- `log_probs` (torch.Tensor): Action log probabilities
- `paths` (torch.Tensor): Generated paths

**run(n_iterations, start_node=0)**
Execute full ACO algorithm for orienteering.

**Parameters:**
- `n_iterations` (int): Number of ACO iterations
- `start_node` (int): Starting node

**Returns:**
- Best prize value found

**update_pheromone(paths, costs)**
Update pheromone trails based on solution quality.

**Parameters:**
- `paths` (torch.Tensor): Solution paths
- `costs` (torch.Tensor): Corresponding costs

**gen_path_costs(paths)**
Compute prizes and check length constraints.

**Parameters:**
- `paths` (torch.Tensor): Paths to evaluate [batch_size, max_length]

**Returns:**
- `prizes` (torch.Tensor): Collected prizes [batch_size]
- `lengths` (torch.Tensor): Tour lengths [batch_size]
- `is_feasible` (torch.Tensor): Feasibility flags [batch_size]

## Utility Functions

### Data Processing

**gen_distance_matrix(coordinates)**
Compute Euclidean distance matrix from coordinates.

**gen_pyg_data(coordinates, prizes, max_len, k_sparse)**
Create PyTorch Geometric data for orienteering.

```python
def gen_pyg_data(
    coordinates: torch.Tensor,  # [n_nodes, 2]
    prizes: torch.Tensor,       # [n_nodes]
    max_len: float,
    k_sparse: int
) -> tuple:
```

**Returns:**
- `pyg_data`: Graph with node features [x, y, prize]
- `distances`: Distance matrix
- `prizes`: Prize vector

### Dataset Management

**load_val_dataset(n_nodes, k_sparse, device)**
Load or generate validation dataset.

**load_test_dataset(n_nodes, k_sparse, device)**
Load test dataset.

## Neural Network Architecture

### Net Class
Orienteering-specific neural network.

```python
class Net(nn.Module):
    def __init__(self, gfn=False, Z_out_dim=1):
        # Node features: [x, y, prize]
        self.emb_net = EmbNet(feats=3)
```

**Key Features:**
- Prize information in node embeddings
- Length constraint awareness
- Selective visiting decisions

## Training and Testing

### Training Function

```python
def train_instance(
    model,
    optimizer,
    data,  # List of (pyg_data, distances, prizes)
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

### Basic Orienteering

```python
import torch
from gfacs.op.aco import ACO
from gfacs.op.net import Net
from gfacs.op.utils import gen_pyg_data

# Generate orienteering instance
n_nodes = 50
coordinates = torch.rand(n_nodes, 2)
prizes = torch.rand(n_nodes) * 10  # Prizes 0-10
prizes[0] = 0  # Depot has no prize
max_len = 5.0  # Maximum tour length

pyg_data, distances, prizes = gen_pyg_data(
    coordinates, prizes, max_len, k_sparse=20
)

# Load model and solve
model = Net(gfn=True)
model.load_state_dict(torch.load('pretrained/op/50/model.pt'))

heu_vec = model(pyg_data)
heu_mat = model.reshape(pyg_data, heu_vec)

aco = ACO(
    distances=distances,
    prizes=prizes,
    max_len=max_len,
    heuristic=heu_mat
)

prizes_collected, log_probs, paths = aco.sample()
lengths = aco.gen_path_costs(paths)[1]  # Get lengths
```

### Feasibility Checking

```python
# Check constraint satisfaction
prizes, lengths, feasible = aco.gen_path_costs(paths)

valid_solutions = paths[feasible]
valid_prizes = prizes[feasible]
valid_lengths = lengths[feasible]

print(f"Feasible solutions: {feasible.sum()}/{len(feasible)}")
print(f"Best prize: {valid_prizes.max()}")
```

### Command Line Usage

```bash
# Train OP model
python train.py 100 --epochs 50 --batch-size 20

# Test model
python test.py 100 -p pretrained/op/100/model.pt
```

## Problem Characteristics

### Objective Function
- **Prize Maximization**: Σᵢ prizeᵢ for visited nodes i
- **Length Constraint**: Tour length ≤ max_len
- **Selective Visiting**: Not all nodes need to be visited
- **Start/End**: Always start and end at depot (node 0)

### Constraints
- **Tour Length**: Distance traveled ≤ max_len
- **Connectivity**: Path must be connected
- **Start Point**: Always begin at depot
- **End Point**: Must return to depot

### Solution Representation
- **Path**: Sequence of nodes starting and ending at depot
- **Visited Set**: Subset of nodes included in path
- **Feasibility**: Length constraint satisfaction

## Performance Characteristics

### Time Complexity
- **ACO Sampling**: O(n_ants × n_nodes × path_length)
- **Feasibility Check**: O(path_length) per path
- **Neural Forward Pass**: O(n_nodes × k_sparse × depth)

### Space Complexity
- **Distance Matrix**: O(n_nodes²)
- **Prize Vector**: O(n_nodes)
- **Path Storage**: O(n_ants × max_path_length)

### Instance Scaling
- **Small instances** (n ≤ 50): Exact solutions possible
- **Medium instances** (50 < n ≤ 200): Good approximation quality
- **Constraint Tightness**: Performance depends on max_len vs. instance diameter

## Implementation Notes

### MAX-MIN Pheromone System
- **Bounds**: Pheromone values constrained to [min, max]
- **Stability**: Prevents pheromone explosion
- **Adaptation**: Bounds adjust during optimization
- **Convergence**: Faster convergence than standard ACO

### Selective Node Visiting
- **Decision Making**: Neural network decides which nodes to visit
- **Path Construction**: Partial paths until length limit reached
- **Prize Trade-off**: Balance between prizes collected and travel cost
- **Early Termination**: Stop when length constraint would be violated

### Feasibility Handling
- **Constraint Checking**: Length validation during and after construction
- **Penalty Methods**: Infeasible solutions receive zero fitness
- **Repair Strategies**: No repair (infeasible solutions discarded)
- **Population Management**: Only feasible solutions contribute to pheromone update

## Testing and Validation

### Unit Tests
- Path feasibility validation
- Prize calculation accuracy
- Length constraint checking
- Neural network predictions

### Integration Tests
- End-to-end orienteering solving
- Constraint satisfaction rates
- Prize collection optimization

### Validation Metrics
- Feasibility rate (percentage of valid solutions)
- Average prize collected
- Solution length distributions
- Comparison with baseline heuristics (nearest neighbor, etc.)
