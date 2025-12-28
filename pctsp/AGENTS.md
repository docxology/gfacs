# Module: Prize Collecting TSP (`pctsp/`)

## Overview

The Prize Collecting Traveling Salesman Problem module implements ACO with GFlowNet sampling for PCTSP. The goal is to find a tour that visits all nodes while maximizing collected prizes and minimizing penalties for non-visited nodes.

**Key Features:**
- Prize collection incentives
- Penalty costs for unvisited nodes
- Complete tour requirement (TSP constraint)
- Trade-off between tour length and prize/penalty balance
- Neural heuristic learning for visiting order optimization

## Core Classes

### ACO Class - Prize Collecting TSP ACO

```python
class ACO:
    def __init__(
        self,
        distances,     # (n, n) - distance matrix
        prizes,        # (n, ) - node prizes
        penalties,     # (n, ) - penalties for not visiting
        n_ants=20,
        heuristic=None,
        pheromone=None,
        decay=0.9,
        alpha=1.0,
        beta=1.0,
        elitist=False,
        maxmin=False,
        rank_based=False,
        n_elites=None,
        shift_cost=True,
        use_local_search=False,
        device='cpu'
    )
```

**Parameters:**
- `distances` (torch.Tensor): Distance matrix [n_nodes, n_nodes]
- `prizes` (torch.Tensor): Node prizes [n_nodes] (depot prize = 0)
- `penalties` (torch.Tensor): Penalties for not visiting [n_nodes]
- `n_ants` (int): Number of ants per iteration
- `decay` (float): Pheromone evaporation rate (0.9)
- `alpha` (float): Pheromone influence (1.0)
- `beta` (float): Heuristic influence (1.0)
- `elitist` (bool): Enable elitist ant system
- `maxmin` (bool): Enable MAX-MIN ant system
- `rank_based` (bool): Enable rank-based ant system
- `device` (str): Computation device ('cpu', 'cuda')

#### Key Methods

**sample(invtemp=1.0, inference=False, start_node=0)**
Generate complete tours with prize collection.

**Parameters:**
- `invtemp` (float): Inverse temperature for sampling
- `inference` (bool): Use greedy sampling
- `start_node` (int): Starting node index

**Returns:**
- `objs` (torch.Tensor): Objective values (prize - penalty - distance)
- `log_probs` (torch.Tensor): Action log probabilities
- `paths` (torch.Tensor): Generated tours

**run(n_iterations, start_node=0)**
Execute full ACO algorithm for PCTSP.

**Parameters:**
- `n_iterations` (int): Number of ACO iterations
- `start_node` (int): Starting node

**Returns:**
- Best objective value found

**update_pheromone(paths, costs)**
Update pheromone trails based on solution quality.

**Parameters:**
- `paths` (torch.Tensor): Solution paths
- `costs` (torch.Tensor): Corresponding costs

**gen_path_costs(paths)**
Compute objective function (prizes - penalties - distance).

**Parameters:**
- `paths` (torch.Tensor): Tours to evaluate [batch_size, n_nodes]

**Returns:**
- `objectives` (torch.Tensor): Objective values [batch_size]

## Objective Function

### PCTSP Formulation
```
Objective = Σᵢ prizeᵢ - Σⱼ penaltyⱼ - distance_penalty × tour_length
```

Where:
- **prizeᵢ**: Prize collected for visiting node i
- **penaltyⱼ**: Penalty for not visiting node j
- **tour_length**: Total distance traveled
- **distance_penalty**: Weight for tour length cost

### Key Characteristics
- **Complete Coverage**: All nodes must be visited (TSP constraint)
- **Prize Incentives**: Rewards for visiting high-prize nodes
- **Penalty Costs**: Costs for leaving nodes unvisited
- **Trade-off**: Balance between prizes collected, penalties avoided, and travel cost

## Utility Functions

### Data Processing

**gen_distance_matrix(coordinates)**
Compute Euclidean distance matrix.

**gen_pyg_data(coordinates, prizes, penalties, k_sparse)**
Create PyTorch Geometric data for PCTSP.

```python
def gen_pyg_data(
    coordinates: torch.Tensor,  # [n_nodes, 2]
    prizes: torch.Tensor,       # [n_nodes]
    penalties: torch.Tensor,    # [n_nodes]
    k_sparse: int
) -> tuple:
```

**Returns:**
- `pyg_data`: Graph with node features [x, y, prize, penalty]
- `distances`: Distance matrix
- `prizes`: Prize vector
- `penalties`: Penalty vector

### Dataset Management

**load_val_dataset(n_nodes, k_sparse, device)**
Load or generate validation dataset.

**load_test_dataset(n_nodes, k_sparse, device)**
Load test dataset.

## Neural Network Architecture

### Net Class
PCTSP-specific neural network.

```python
class Net(nn.Module):
    def __init__(self, gfn=False, Z_out_dim=1):
        # Node features: [x, y, prize, penalty]
        self.emb_net = EmbNet(feats=4)
```

**Key Features:**
- Prize and penalty information in embeddings
- Decision making considers both incentives and costs
- TSP completeness constraint awareness

## Training and Testing

### Training Function

```python
def train_instance(
    model,
    optimizer,
    data,  # List of (pyg_data, distances, prizes, penalties)
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

### Basic PCTSP Solving

```python
import torch
from gfacs.pctsp.aco import ACO
from gfacs.pctsp.net import Net
from gfacs.pctsp.utils import gen_pyg_data

# Generate PCTSP instance
n_nodes = 50
coordinates = torch.rand(n_nodes, 2)
prizes = torch.rand(n_nodes) * 10     # Prizes 0-10
penalties = torch.rand(n_nodes) * 5   # Penalties 0-5
prizes[0] = penalties[0] = 0          # Depot has no prize/penalty

pyg_data, distances, prizes, penalties = gen_pyg_data(
    coordinates, prizes, penalties, k_sparse=20
)

# Load model and solve
model = Net(gfn=True)
model.load_state_dict(torch.load('pretrained/pctsp/50/model.pt'))

heu_vec = model(pyg_data)
heu_mat = model.reshape(pyg_data, heu_vec)

aco = ACO(
    distances=distances,
    prizes=prizes,
    penalties=penalties,
    heuristic=heu_mat
)

objectives, log_probs, tours = aco.sample()
tour_lengths = aco.gen_path_costs(tours)  # Additional length info
```

### Objective Analysis

```python
# Analyze solution components
objectives, log_probs, tours = aco.sample()

# Best solution
best_idx = objectives.argmax()
best_tour = tours[best_idx]
best_objective = objectives[best_idx]

# Component breakdown
tour_length = aco.gen_path_costs(best_tour.unsqueeze(0))
total_prizes = prizes.sum()
total_penalties = penalties.sum()  # All penalties since complete tour

print(f"Best objective: {best_objective}")
print(f"Tour length: {tour_length}")
print(f"Total prizes: {total_prizes}")
print(f"Penalties avoided: {total_penalties}")
```

### Command Line Usage

```bash
# Train PCTSP model
python train.py 100 --epochs 50 --batch-size 20

# Test model
python test.py 100 -p pretrained/pctsp/100/model.pt
```

## Problem Characteristics

### Constraints
- **Complete Tour**: All nodes must be visited exactly once
- **Connectivity**: Tour must be a valid Hamiltonian cycle
- **Start/End**: Begin and end at depot (node 0)
- **No Subtours**: Single connected tour

### Decision Variables
- **Tour Order**: Sequence in which nodes are visited
- **Prize Collection**: Automatic for all visited nodes
- **Penalty Avoidance**: All penalties avoided (complete tour)
- **Length Cost**: Distance traveled through tour

### Objective Components
- **Positive**: Prizes collected (Σ visited node prizes)
- **Negative**: Tour length cost
- **Penalty-Free**: No penalties since all nodes visited

## Performance Characteristics

### Time Complexity
- **ACO Sampling**: O(n_ants × n_nodes²) - complete tour construction
- **Objective Calculation**: O(n_nodes) per tour
- **Neural Forward Pass**: O(n_nodes × k_sparse × depth)

### Space Complexity
- **Distance Matrix**: O(n_nodes²)
- **Prize/Penalty Vectors**: O(n_nodes)
- **Tour Storage**: O(n_ants × n_nodes)

### Instance Scaling
- **Small instances** (n ≤ 50): Exact solutions possible
- **Medium instances** (50 < n ≤ 100): Good approximation quality
- **Large instances** (n > 100): Computationally intensive

## Implementation Notes

### TSP vs. Selective Visiting
- **PCTSP**: Complete coverage required, all penalties avoided
- **OP**: Selective visiting, length-constrained
- **Trade-off**: PCTSP focuses on order optimization under completeness

### Objective Function Design
- **Prize Maximization**: Incentivize visiting high-value nodes
- **Length Minimization**: Standard TSP distance cost
- **Penalty Handling**: No penalties in PCTSP (complete tours)

### Neural Learning
- **Node Embeddings**: Include both prize and penalty information
- **Decision Making**: Learn effective visiting order
- **GFlowNet**: Learn distribution over optimal tour orders

## Testing and Validation

### Unit Tests
- Tour completeness validation
- Objective calculation accuracy
- Prize and penalty accounting
- Neural network predictions

### Integration Tests
- End-to-end PCTSP solving
- Tour validity checking
- Objective function optimization

### Validation Metrics
- Tour completeness rate (should be 100%)
- Average objective value
- Prize collection efficiency
- Tour length vs. prize trade-off analysis
