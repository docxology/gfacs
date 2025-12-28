# Module: CVRPTW with PyVRP Local Search (`cvrptw_nls/`)

## Overview

The Capacitated Vehicle Routing Problem with Time Windows module implements ACO with GFlowNet sampling for CVRPTW. This module integrates with PyVRP library for time window constraint handling and local search optimization.

**Key Features:**
- Capacity and time window constraints
- PyVRP integration for constraint-aware local search
- Temporal feasibility validation
- Multi-dimensional constraint optimization
- Neural heuristic learning for time-aware routing

## Core Classes

### ACO Class - Time-Window Constrained Vehicle Routing

```python
class ACO:
    def __init__(
        self,
        distances,      # (n, n) - distance matrix
        demands,        # (n, ) - customer demands
        windows,        # (n, 2) - time windows [early, late]
        service_time: float = 0.0,
        n_ants: int = 20,
        decay: float = 0.9,
        alpha: float = 1.0,
        beta: float = 1.0,
        elitist: bool = False,
        maxmin: bool = False,
        rank_based: bool = False,
        n_elites: int | None = None,
        pheromone: torch.Tensor | None = None,
        heuristic: torch.Tensor | None = None,
        k_sparse: int | None = None,
        local_search_type: str | None = 'pyvrp',
        device: str = 'cpu'
    )
```

**Parameters:**
- `distances` (torch.Tensor): Distance matrix [n_nodes, n_nodes]
- `demands` (torch.Tensor): Customer demands [n_nodes] (depot = 0)
- `windows` (torch.Tensor): Time windows [n_nodes, 2] as [early, late]
- `service_time` (float): Service time at each customer (0.0)
- `n_ants` (int): Number of ants per iteration (20)
- `decay` (float): Pheromone evaporation rate (0.9)
- `alpha` (float): Pheromone influence (1.0)
- `beta` (float): Heuristic influence (1.0)
- `local_search_type` (str, optional): Local search type ('pyvrp', None)
- `device` (str): Computation device ('cpu', 'cuda')

#### Key Methods

**sample(invtemp=1.0, inference=False)**
Generate time-feasible vehicle routes.

**Parameters:**
- `invtemp` (float): Inverse temperature for sampling
- `inference` (bool): Use greedy sampling

**Returns:**
- `costs` (torch.Tensor): Total route costs [n_ants]
- `log_probs` (torch.Tensor): Action log probabilities
- `paths` (torch.Tensor): Routes with depot separators

**local_search(paths, inference=False)**
Apply PyVRP local search with time window constraints.

**Parameters:**
- `paths` (torch.Tensor): Input routes
- `inference` (bool): Inference mode flag

**Returns:**
- `improved_paths` (torch.Tensor): Time-feasible optimized routes

**gen_path_costs(paths)**
Compute route costs considering time window penalties.

**Parameters:**
- `paths` (torch.Tensor): Routes to evaluate

**Returns:**
- `costs` (torch.Tensor): Penalized route costs

**gen_numpy_path_costs(paths)**
Compute route costs for NumPy paths with time window penalties.

**Parameters:**
- `paths` (numpy.ndarray): Routes to evaluate

**Returns:**
- `costs` (numpy.ndarray): Penalized route costs

**validate_time_windows(paths)**
Check time window feasibility of routes.

**Parameters:**
- `paths` (torch.Tensor): Routes to validate

**Returns:**
- `is_feasible` (bool): Time feasibility
- `violations` (dict): Time window violations

**run(n_iterations)**
Execute full ACO algorithm for CVRPTW.

**Parameters:**
- `n_iterations` (int): Number of ACO iterations

**Returns:**
- `lowest_cost` (float): Best solution cost found
- `diversity` (float): Solution diversity measure
- `duration` (float): Execution time in seconds

**update_pheromone(paths, costs)**
Update pheromone trails based on solution quality.

**Parameters:**
- `paths` (torch.Tensor): Solution routes
- `costs` (torch.Tensor): Corresponding route costs

**neural_local_search(paths, inference=False, T_nls=1)**
Apply neural-guided local search with time window constraints.

**Parameters:**
- `paths` (torch.Tensor): Input routes to optimize
- `inference` (bool): Use inference mode
- `T_nls` (int): Local search iterations

**Returns:**
- `optimized_paths` (torch.Tensor): Time-feasible optimized routes

## PyVRP Integration

### Local Search Function
Interface to PyVRP's constraint-aware local search.

```python
def pyvrp_batched_local_search(
    positions: np.ndarray,    # [n_nodes, 2]
    demands: np.ndarray,      # [n_nodes]
    windows: np.ndarray,      # [n_nodes, 2]
    routes: list,             # Initial routes
    max_iterations: int = 1000
) -> list:
```

**Parameters:**
- `positions`: Node coordinates
- `demands`: Customer demands
- `windows`: Time windows [early, late]
- `routes`: Initial solution routes
- `max_iterations`: Local search iterations

**Returns:**
- `optimized_routes`: Time-feasible improved routes

### Data Conversion Functions

**make_data(positions, demands, windows, distances)**
Create PyVRP ProblemData instance.

```python
def make_data(
    positions: np.ndarray,
    demands: np.ndarray,
    windows: np.ndarray,
    distances: np.ndarray
) -> pyvrp.ProblemData:
```

**make_solution(data, path)**
Convert route tensor to PyVRP Solution.

```python
def make_solution(
    data: pyvrp.ProblemData,
    path: np.ndarray
) -> pyvrp.Solution:
```

## Utility Functions

### Route Processing
Same as CVRP module: `get_subroutes()`, `merge_subroutes()`

### Data Processing

**gen_distance_matrix(coordinates)**
Compute Euclidean distance matrix.

**gen_pyg_data(coordinates, demands, windows, k_sparse)**
Create PyTorch Geometric data with time windows.

```python
def gen_pyg_data(
    coordinates: torch.Tensor,  # [n_nodes, 2]
    demands: torch.Tensor,      # [n_nodes]
    windows: torch.Tensor,      # [n_nodes, 2]
    k_sparse: int
) -> tuple:
```

**Returns:**
- `pyg_data`: Graph with node features [x, y, demand, early_time, late_time]
- `distances`: Distance matrix
- `demands`: Demand vector
- `windows`: Time window matrix

### Dataset Management

**load_val_dataset(n_nodes, k_sparse, device)**
Generate validation dataset with time windows.

**load_test_dataset(n_nodes, k_sparse, device)**
Load test dataset with time constraints.

**gen_instance(n_nodes)**
Generate random CVRPTW instance.

```python
def gen_instance(n_nodes: int) -> tuple:
    """
    Generate random CVRPTW instance.

    Returns:
        coordinates, demands, windows, distances
    """
```

## Neural Network Architecture

### Net Class
Extended for time window features.

```python
class Net(nn.Module):
    def __init__(self, gfn: bool = False, Z_out_dim: int = 1):
        # Node features: [x, y, demand, early_time, late_time]
        self.emb_net = EmbNet(feats=5)  # Extended feature dimension
```

**Key Extensions:**
- Time window features in node embeddings
- Temporal feasibility awareness
- Constraint-aware routing decisions

## Training and Testing

### Training Function

```python
def train_instance(
    model,
    optimizer,
    data,  # (pyg_data, distances, demands, windows)
    n_ants,
    cost_w=1.0,
    invtemp=1.0,
    guided_exploration=False,
    shared_energy_norm=False,
    beta=100.0,
    it=0,
    local_search_params=None
)
```

**Additional Parameters:**
- `local_search_params`: PyVRP local search configuration

### Testing Function

```python
def test(dataset, model, n_ants, t_aco, k_sparse)
```

Includes time window validation and feasibility checking.

## Usage Examples

### Basic CVRPTW Solving

```python
import torch
from gfacs.cvrptw_nls.aco import ACO
from gfacs.cvrptw_nls.net import Net
from gfacs.cvrptw_nls.utils import gen_pyg_data

# Generate CVRPTW instance
n_nodes = 50
coordinates = torch.rand(n_nodes, 2)
demands = torch.rand(n_nodes) * 0.5
demands[0] = 0  # Depot
windows = torch.zeros(n_nodes, 2)
windows[:, 1] = 1.0  # Time horizon [0, 1]
# Add random time windows
for i in range(1, n_nodes):
    early = torch.rand(1) * 0.8
    late = early + 0.2 + torch.rand(1) * 0.3
    windows[i] = torch.cat([early, late])

pyg_data, distances, demands, windows = gen_pyg_data(
    coordinates, demands, windows, k_sparse=20
)

# Load model and solve
model = Net(gfn=True)
model.load_state_dict(torch.load('pretrained/cvrptw_nls/50/model.pt'))

heu_vec = model(pyg_data)
heu_mat = model.reshape(pyg_data, heu_vec)

aco = ACO(
    distances=distances,
    demands=demands,
    windows=windows,
    heuristic=heu_mat,
    local_search_type='pyvrp'
)

costs, log_probs, routes = aco.sample()
improved_routes = aco.local_search(routes)
```

### Time Window Validation

```python
# Check time feasibility
is_feasible, violations = aco.validate_time_windows(routes)

if not is_feasible:
    print(f"Time window violations: {violations}")
```

### Command Line Usage

```bash
# Train CVRPTW model
python train.py 100 --epochs 50

# Test model
python test.py 100 -p pretrained/cvrptw_nls/100/model.pt
```

## External Dependencies

### PyVRP Library
- **Installation**: `pip install pyvrp`
- **Purpose**: Time window constraint handling and local search
- **Integration**: Native Python API with efficient C++ backend

### Key PyVRP Components Used
- **ProblemData**: Instance representation with time windows
- **Solution**: Route solution with feasibility checking
- **LocalSearch**: Constraint-aware improvement operators
- **CostEvaluator**: Penalized cost computation

## Performance Characteristics

### Time Complexity
- **ACO Sampling**: O(n_ants × n_nodes × n_routes)
- **PyVRP Local Search**: O(n_nodes × iterations) - efficient implementation
- **Time Validation**: O(route_length) per route

### Space Complexity
- **Extended Features**: O(n_nodes × 5) for node features
- **Time Windows**: O(n_nodes × 2) storage
- **PyVRP Data**: O(n_nodes) problem representation

### Constraint Handling
- **Capacity**: Same as CVRP (normalized to 1.0)
- **Time Windows**: [early, late] pairs per customer
- **Service Time**: Fixed service duration
- **Penalties**: Large penalties for infeasible solutions

## Implementation Notes

### Time Window Representation
- **Normalization**: Time horizon normalized to [0, 1]
- **Depot Window**: Usually [0, 1] (full horizon)
- **Customer Windows**: Random subsets of time horizon
- **Validation**: Arrival time ∈ [early, late] for each customer

### PyVRP Integration
- **Data Scaling**: Coordinates scaled to integers (×10^4)
- **Time Scaling**: Time windows scaled to integers
- **Capacity Scaling**: Demands scaled to integers
- **Solution Conversion**: PyVRP Solution ↔ tensor routes

### Feasibility Checking
- **Hard Constraints**: Capacity and time windows
- **Soft Constraints**: Service times, waiting times
- **Penalization**: Large cost penalties for violations
- **Repair**: Local search attempts feasibility repair

## Testing and Validation

### Unit Tests
- Time window validation
- Route feasibility checking
- PyVRP data conversion
- Cost computation with penalties

### Integration Tests
- End-to-end CVRPTW solving
- Time window constraint satisfaction
- PyVRP local search effectiveness

### Validation Metrics
- Time window feasibility rate
- Capacity constraint satisfaction
- Solution cost vs. time penalties
- Local search improvement quality
