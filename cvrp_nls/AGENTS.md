# Module: CVRP with Neural Local Search (`cvrp_nls/`)

## Overview

The Capacitated Vehicle Routing Problem with Neural Local Search module implements ACO with GFlowNet sampling for the CVRP. The module integrates with the HGS-CVRP C++ solver for high-quality local search and handles capacity constraints for vehicle routing.

**Key Features:**
- Capacity-constrained vehicle routing
- HGS-CVRP integration for local search
- Multi-route solution representation
- Demand and capacity validation
- Neural heuristic learning for routing decisions

## Core Classes

### ACO Class - Capacitated Vehicle Routing ACO

```python
class ACO:
    def __init__(
        self,
        distances: torch.Tensor,  # (n, n)
        demand: torch.Tensor,     # (n, )
        capacity: float = 1.0,
        positions: torch.Tensor | None = None,
        n_ants: int = 20,
        heuristic: torch.Tensor | None = None,
        k_sparse: int | None = None,
        pheromone: torch.Tensor | None = None,
        decay: float = 0.9,
        alpha: float = 1.0,
        beta: float = 1.0,
        elitist: bool = False,
        maxmin: bool = False,
        rank_based: bool = False,
        n_elites: int | None = None,
        smoothing: bool = False,
        smoothing_thres: int = 5,
        smoothing_delta: float = 0.5,
        shift_cost: bool = True,
        local_search_type: str | None = 'nls',
        device: str = 'cpu'
    )
```

**Parameters:**
- `distances` (torch.Tensor): Distance matrix [n_nodes, n_nodes]
- `demand` (torch.Tensor): Customer demands [n_nodes] (depot demand = 0)
- `capacity` (float): Vehicle capacity (normalized to 1.0)
- `positions` (torch.Tensor, optional): Node coordinates [n_nodes, 2]
- `n_ants` (int): Number of ants per iteration
- `heuristic` (torch.Tensor, optional): Heuristic matrix [n_nodes, n_nodes]
- `k_sparse` (int, optional): Sparsity parameter for heuristic generation
- `pheromone` (torch.Tensor, optional): Initial pheromone matrix
- `decay` (float): Pheromone evaporation rate (0.9)
- `alpha` (float): Pheromone influence (1.0)
- `beta` (float): Heuristic influence (1.0)
- `local_search_type` (str, optional): Local search type ('nls', None)
- `device` (str): Computation device ('cpu', 'cuda')

#### Key Methods

**sample(invtemp=1.0, inference=False, start_node=0)**
Generate vehicle routes using ACO with capacity constraints.

**Parameters:**
- `invtemp` (float): Inverse temperature for sampling
- `inference` (bool): Use greedy sampling for inference
- `start_node` (int): Depot node index (default=0)

**Returns:**
- `costs` (torch.Tensor): Total route costs [n_ants]
- `log_probs` (torch.Tensor): Action log probabilities
- `paths` (torch.Tensor): Routes with depot separators [n_ants, max_length]

**local_search(paths, inference=False)**
Apply HGS-CVRP local search to improve routes.

**Parameters:**
- `paths` (torch.Tensor): Input routes [batch_size, max_length]
- `inference` (bool): Run in inference mode

**Returns:**
- `improved_paths` (torch.Tensor): Locally optimized routes

**gen_path_costs(paths)**
Compute total distance costs for routes.

**Parameters:**
- `paths` (torch.Tensor): Routes with depot separators

**Returns:**
- `costs` (torch.Tensor): Total route costs

**gen_path(require_prob=False, invtemp=1.0, paths=None)**
Generate CVRP solution paths using ACO construction.

**Parameters:**
- `require_prob` (bool): Whether to return log probabilities
- `invtemp` (float): Inverse temperature for sampling
- `paths` (torch.Tensor, optional): Pre-existing partial paths

**Returns:**
- `paths` (torch.Tensor): Complete solution paths
- `log_probs` (torch.Tensor, optional): Log probabilities if require_prob=True

**validate_routes(paths)**
Validate route feasibility (capacity and connectivity).

**Parameters:**
- `paths` (torch.Tensor): Routes to validate

**Returns:**
- `is_valid` (bool): Route feasibility
- `violations` (dict): Capacity and connectivity violations

**multiple_swap_star(paths, indexes=None, inference=False)**
Apply parallel local search to multiple solution paths.

**Parameters:**
- `paths` (torch.Tensor): Solution paths to optimize
- `indexes` (list, optional): Specific path indices to optimize
- `inference` (bool): Use inference mode (more iterations)

**Returns:**
- `optimized_paths` (torch.Tensor): Locally optimized paths

**distances_cpu**
Distance matrix as NumPy array (CPU).

**Returns:**
- `distances` (numpy.ndarray): Distance matrix for local search

**demand_cpu**
Customer demands as NumPy array (CPU).

**Returns:**
- `demand` (numpy.ndarray): Demand vector for local search

**positions_cpu**
Node positions as NumPy array (CPU).

**Returns:**
- `positions` (numpy.ndarray, optional): Position coordinates for local search

**run(n_iterations)**
Execute full ACO algorithm for CVRP.

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

## Utility Functions

### Route Processing

**get_subroutes(route, end_with_zero=True)**
Split route into individual vehicle routes.

```python
def get_subroutes(route: torch.Tensor, end_with_zero: bool = True) -> list:
    """
    Split route at depot nodes (0) to get individual vehicle routes.

    Args:
        route: Route with depot separators
        end_with_zero: Include depot at end of each subroute

    Returns:
        List of subroutes for each vehicle
    """
```

**merge_subroutes(subroutes, length, device)**
Merge individual routes back into single route representation.

```python
def merge_subroutes(
    subroutes: list,
    length: int,
    device: str
) -> torch.Tensor:
```

### Data Processing

**gen_distance_matrix(coordinates)**
Compute Euclidean distance matrix from coordinates.

**gen_pyg_data(coordinates, demand, k_sparse)**
Create PyTorch Geometric data for CVRP instances.

```python
def gen_pyg_data(
    coordinates: torch.Tensor,  # [n_nodes, 2]
    demand: torch.Tensor,       # [n_nodes]
    k_sparse: int
) -> tuple[torch_geometric.data.Data, torch.Tensor, torch.Tensor]:
```

**Returns:**
- `pyg_data`: Graph data with node features [coordinates, demand]
- `distances`: Distance matrix
- `demand`: Demand vector

### Dataset Management

**load_val_dataset(n_nodes, k_sparse, device)**
Load or generate validation dataset.

**load_test_dataset(n_nodes, k_sparse, device, filename=None)**
Load test dataset with optional filename.

## HGS-CVRP Integration

### Solver Class
Interface to HGS-CVRP C++ local search solver.

```python
class Solver:
    def __init__(self, parameters: AlgorithmParameters, verbose: bool = False)

    def local_search(
        self,
        data: dict,
        routes: list,
        count: int = 1,
        rounding: bool = True
    ) -> list:
```

**Parameters:**
- `data`: Problem data dictionary with coordinates, demands, capacity
- `routes`: Initial routes as list of lists
- `count`: Number of local search iterations
- `rounding`: Round solution to integers

### AlgorithmParameters
Configuration for HGS-CVRP solver.

**Key Parameters:**
- `nbGranular`: Granular search neighborhood size
- `mu`: Granular search parameter
- `lambda`: Penalty parameter
- `nbClose`: Number of closest customers considered

## Neural Network Architecture

### Net Class
Same architecture as TSP module but adapted for CVRP features.

```python
class Net(nn.Module):
    def __init__(self, gfn: bool = False, Z_out_dim: int = 1):
        # Node features: [x, y, demand] instead of [x, y]
        self.emb_net = EmbNet(feats=3)  # coordinates + demand
```

**Key Difference from TSP:**
- Node features include demand: [x, y, demand]
- Routing decisions consider capacity constraints
- Multiple routes per solution

### Neural Swap Star Function

**neural_swapstar(demand, distances, heu_dist, positions, p, disturb=5, limit=10000)**
Neural-guided local search combining distance and heuristic information.

**Parameters:**
- `demand` (numpy.ndarray): Customer demands
- `distances` (numpy.ndarray): Distance matrix
- `heu_dist` (numpy.ndarray): Heuristic distance matrix
- `positions` (numpy.ndarray): Node coordinates
- `p` (list): Initial route permutation
- `disturb` (int): Disturbance iterations with heuristic (default: 5)
- `limit` (int): Full local search iterations (default: 10000)

**Returns:**
- `optimized_route` (list): Locally optimized route

## Training and Testing

### Training Function
Similar to TSP but with capacity constraint handling.

```python
def train_instance(
    model,
    optimizer,
    data,  # List of (pyg_data, distances, demand)
    n_ants,
    cost_w=1.0,
    invtemp=1.0,
    guided_exploration=False,
    beta=100.0
)
```

### Testing Function
Includes route validation and capacity checks.

```python
def test(
    dataset,  # List of (pyg_data, distances, demand)
    model,
    n_ants,
    t_aco,
    k_sparse
)
```

## Usage Examples

### Basic CVRP Solving

```python
import torch
from gfacs.cvrp_nls.aco import ACO
from gfacs.cvrp_nls.net import Net
from gfacs.cvrp_nls.utils import gen_pyg_data

# Generate CVRP instance
n_nodes = 50
coordinates = torch.rand(n_nodes, 2)
demand = torch.rand(n_nodes) * 0.5  # Normalized demands
demand[0] = 0  # Depot has no demand

pyg_data, distances, demand = gen_pyg_data(coordinates, demand, k_sparse=20)

# Load model and generate heuristic
model = Net(gfn=True)
model.load_state_dict(torch.load('pretrained/cvrp_nls/50/model.pt'))
heu_vec = model(pyg_data)
heu_mat = model.reshape(pyg_data, heu_vec)

# Solve with ACO + HGS local search
aco = ACO(
    distances=distances,
    demand=demand,
    capacity=1.0,
    heuristic=heu_mat,
    local_search_type='nls'
)

costs, log_probs, routes = aco.sample()
improved_routes = aco.local_search(routes)
improved_costs = aco.gen_path_costs(improved_routes)
```

### Route Validation

```python
# Check route feasibility
is_valid, violations = aco.validate_routes(routes)

if not is_valid:
    print(f"Capacity violations: {violations['capacity']}")
    print(f"Connectivity issues: {violations['connectivity']}")
```

### Command Line Usage

```bash
# Train CVRP model
python train.py 100 --epochs 50 --batch-size 20

# Test with TAM datasets
python train.py 100 --tam

# Test model
python test.py 100 -p pretrained/cvrp_nls/100/model.pt
```

## External Dependencies

### HGS-CVRP C++ Solver
- **Location**: `HGS-CVRP-main/` directory
- **Build Requirements**: CMake, C++ compiler
- **Installation**:
  ```bash
  cd HGS-CVRP-main
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make lib
  ```
- **Output**: `libhgscvrp.so` shared library

### Integration Details
- **Interface**: ctypes binding to C API
- **Data Format**: Dictionary with coordinates, demands, capacity
- **Route Format**: List of lists (one per vehicle)
- **Performance**: Fast local search for CVRP instances

## Performance Characteristics

### Time Complexity
- **ACO Sampling**: O(n_ants × n_nodes × n_routes)
- **HGS Local Search**: O(n_nodes × iterations) - efficient C++ implementation
- **Capacity Checking**: O(route_length) per route

### Space Complexity
- **Distance Matrix**: O(n_nodes²)
- **Route Storage**: O(n_ants × max_route_length)
- **HGS Data**: O(n_nodes) for problem data

### Scalability
- **Small instances** (n ≤ 100): Fast convergence, high quality
- **Large instances** (n > 100): Memory intensive, HGS provides quality boost

## Implementation Notes

### Route Representation
- **Format**: Single tensor with depot (0) separators
- **Example**: [0, 1, 3, 5, 0, 2, 4, 0] = routes [[1,3,5], [2,4]]
- **Validation**: Capacity and connectivity checks

### Capacity Constraints
- **Normalization**: All demands normalized to [0, 1]
- **Capacity**: Fixed at 1.0 (normalized)
- **Validation**: Cumulative demand ≤ capacity per route

### HGS Integration
- **Data Conversion**: PyTorch tensors → C-compatible arrays
- **Memory Management**: Proper cleanup of C structures
- **Error Handling**: Exception handling for solver failures

## Testing and Validation

### Unit Tests
- Route splitting and merging
- Capacity constraint validation
- Cost calculation accuracy
- HGS integration testing

### Integration Tests
- End-to-end CVRP solving
- Model training pipeline
- Benchmark dataset evaluation

### Validation Metrics
- Route feasibility (capacity, connectivity)
- Solution cost vs. known optima
- HGS improvement quality
