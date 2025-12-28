# Module: TSP with Neural Local Search (`tsp_nls/`)

## Overview

The TSP with Neural Local Search module implements Ant Colony Optimization combined with GFlowNet sampling and neural-guided local search for the Traveling Salesman Problem. The module integrates custom 2-opt local search with optional Concorde TSP solver verification.

**Key Features:**
- GFlowNet-based heuristic learning
- Guided exploration using post-local-search costs
- Parallel 2-opt local search implementation
- TSPLIB benchmark support
- Multiple ACO variants (AS, Elitist, MAX-MIN, Rank-based)

## Core Classes

### ACO Class - Ant Colony Optimization Implementation

```python
class ACO:
    def __init__(
        self,
        distances: torch.Tensor,
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
- `distances` (torch.Tensor): Distance matrix of shape [n_nodes, n_nodes]
- `n_ants` (int): Number of ants per iteration
- `heuristic` (torch.Tensor, optional): Heuristic matrix of shape [n_nodes, n_nodes]
- `k_sparse` (int, optional): Sparsity parameter for heuristic generation
- `pheromone` (torch.Tensor, optional): Initial pheromone matrix
- `decay` (float): Pheromone evaporation rate (0.9)
- `alpha` (float): Pheromone influence parameter (1.0)
- `beta` (float): Heuristic influence parameter (1.0)
- `elitist` (bool): Enable elitist ant system
- `maxmin` (bool): Enable MAX-MIN ant system
- `rank_based` (bool): Enable rank-based ant system
- `n_elites` (int, optional): Number of elite ants for rank-based system
- `smoothing` (bool): Enable pheromone smoothing
- `smoothing_thres` (int): Smoothing threshold iterations
- `smoothing_delta` (float): Smoothing delta parameter
- `shift_cost` (bool): Shift costs to positive values
- `local_search_type` (str, optional): Local search type ('nls', '2opt', None)
- `device` (str): Device for tensor operations ('cpu', 'cuda')

#### Key Methods

**sample(invtemp=1.0, inference=False, start_node=None)**
Generate paths using current pheromone and heuristic information.

**Parameters:**
- `invtemp` (float): Inverse temperature for sampling (1.0)
- `inference` (bool): Use greedy sampling for inference
- `start_node` (int, optional): Fixed starting node index

**Returns:**
- `costs` (torch.Tensor): Path costs of shape [n_ants]
- `log_probs` (torch.Tensor): Log probabilities of shape [n_ants, n_nodes-1] or None if inference
- `paths` (torch.Tensor): Generated paths of shape [n_nodes, n_ants]

**local_search(paths, inference=False)**
Apply local search improvements to generated paths.

**Parameters:**
- `paths` (torch.Tensor): Input paths of shape [batch_size, n_nodes]
- `inference` (bool): Run in inference mode (no gradients)

**Returns:**
- `improved_paths` (torch.Tensor): Locally optimized paths

**run(n_iterations, start_node=None)**
Execute full ACO algorithm with pheromone updates.

**Parameters:**
- `n_iterations` (int): Number of ACO iterations
- `start_node` (int, optional): Fixed starting node

**Returns:**
- `best_cost` (float): Best solution cost found
- `best_path` (torch.Tensor): Best solution path

**update_pheromone(paths, costs)**
Update pheromone trails based on solution quality.

**Parameters:**
- `paths` (torch.Tensor): Solution paths
- `costs` (torch.Tensor): Corresponding costs

**gen_path_costs(paths)**
Compute costs for given paths.

**Parameters:**
- `paths` (torch.Tensor): Paths of shape [n_nodes, n_ants]

**Returns:**
- `costs` (torch.Tensor): Path costs of shape [n_ants]

**gen_numpy_path_costs(paths)**
Compute costs for given NumPy paths.

**Parameters:**
- `paths` (numpy.ndarray): Paths of shape [n_ants, n_nodes]

**Returns:**
- `costs` (numpy.ndarray): Path costs of shape [n_ants]

### ACO_NP Class - NumPy-based ACO Implementation

NumPy implementation of ACO for environments without PyTorch or for specific use cases.

```python
class ACO_NP:
    def __init__(
        self,
        distances: np.ndarray,
        n_ants: int = 20,
        heuristic: np.ndarray | None = None,
        k_sparse: int | None = None,
        pheromone: np.ndarray | None = None,
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
        local_search_type: str | None = 'nls'
    )
```

Similar interface to torch-based ACO but operates on NumPy arrays.

## Neural Network Classes

### EmbNet - Edge Embedding Network

```python
class EmbNet(nn.Module):
    def __init__(
        self,
        depth: int = 12,
        feats: int = 2,
        units: int = 32,
        act_fn: str = 'silu',
        agg_fn: str = 'mean'
    )
```

Graph neural network for learning edge embeddings from node coordinates.

**Parameters:**
- `depth` (int): Number of message passing layers (12)
- `feats` (int): Input node feature dimension (2 for coordinates)
- `units` (int): Hidden dimension size (32)
- `act_fn` (str): Activation function name ('silu')
- `agg_fn` (str): Global aggregation function ('mean')

**forward(x, edge_index, edge_attr)**
Process graph data through embedding network.

**Parameters:**
- `x` (torch.Tensor): Node features of shape [n_nodes, feats]
- `edge_index` (torch.Tensor): Edge indices of shape [2, n_edges]
- `edge_attr` (torch.Tensor): Edge attributes of shape [n_edges, 1]

**Returns:**
- `edge_embeddings` (torch.Tensor): Edge embeddings of shape [n_edges, units]

### ParNet - Parameter Prediction Network

```python
class ParNet(MLP):
    def __init__(
        self,
        depth: int = 3,
        units: int = 32,
        preds: int = 1,
        act_fn: str = 'silu'
    )
```

MLP for predicting heuristic parameters from edge embeddings.

### Net - Main Neural Network

```python
class Net(nn.Module):
    def __init__(
        self,
        gfn: bool = False,
        Z_out_dim: int = 1,
        start_node: int | None = None
    )
```

Complete neural network combining embedding and parameter networks.

**Parameters:**
- `gfn` (bool): Enable GFlowNet mode with log-Z estimation
- `Z_out_dim` (int): Output dimension for log-Z network
- `start_node` (int, optional): Fixed start node (changes input features)

**forward(pyg, return_logZ=False)**
Process PyTorch Geometric data.

**Parameters:**
- `pyg` (torch_geometric.data.Data): Graph data instance
- `return_logZ` (bool): Return log-Z estimates for GFlowNet training

**Returns:**
- `heu_vec` (torch.Tensor): Heuristic values
- `logZs` (tuple, optional): Log-Z estimates if return_logZ=True

**reshape(pyg, heu_vec)**
Reshape heuristic vector into matrix form.

**Parameters:**
- `pyg` (torch_geometric.data.Data): Graph data
- `heu_vec` (torch.Tensor): Flat heuristic vector

**Returns:**
- `heu_mat` (torch.Tensor): Heuristic matrix of shape [n_nodes, n_nodes]

## Utility Functions

### Data Processing

**gen_distance_matrix(tsp_coordinates)**
Compute Euclidean distance matrix from node coordinates.

```python
def gen_distance_matrix(tsp_coordinates: torch.Tensor) -> torch.Tensor:
    """
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        distance_matrix: torch tensor [n_nodes, n_nodes] for EUC distances
    """
```

**gen_pyg_data(tsp_coordinates, k_sparse, start_node=None)**
Convert TSP coordinates to PyTorch Geometric format with sparse edges.

```python
def gen_pyg_data(
    tsp_coordinates: torch.Tensor,
    k_sparse: int,
    start_node: int | None = None
) -> tuple[torch_geometric.data.Data, torch.Tensor]:
```

**Parameters:**
- `tsp_coordinates` (torch.Tensor): Node coordinates [n_nodes, 2]
- `k_sparse` (int): Number of edges to keep per node
- `start_node` (int, optional): Fixed starting node

**Returns:**
- `pyg_data` (torch_geometric.data.Data): Graph data
- `distances` (torch.Tensor): Distance matrix

**gen_pyg_data_tsplib(tsp_coordinates, k_sparse, start_node=None)**
TSPLIB-specific version of data generation.

### Dataset Management

**load_val_dataset(n_nodes, k_sparse, device, start_node=None)**
Load or generate validation dataset.

**load_test_dataset(n_nodes, k_sparse, device, start_node=None, filename=None)**
Load test dataset with optional custom filename.

## Training Functions

### train_instance(model, optimizer, data, n_ants, ...)**
Train model on a single problem instance.

**Parameters:**
- `model` (Net): Neural network model
- `optimizer` (torch.optim.Optimizer): Optimizer instance
- `data` (list): List of (pyg_data, distances) tuples
- `n_ants` (int): Number of ants for sampling
- Additional parameters for guided exploration, temperature, etc.

**Returns:**
- Training loss and logging metrics

### Main training function in train.py
Entry point for training with argument parsing and W&B integration.

## Testing Functions

### infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse)**
Run inference on a single instance.

### test(dataset, model, n_ants, t_aco, k_sparse)**
Test model on dataset with multiple ACO iterations.

### Main testing function in test.py
Command-line interface for testing with various ACO algorithms.

## Local Search Implementation

### two_opt_once(distmat, tour, fixed_i=0)
Apply single 2-opt improvement step (numba-compiled).

### _two_opt_python(distmat, tour, max_iterations=1000)
Complete 2-opt optimization for single tour.

### batched_two_opt_python(dist, tours, max_iterations=1000)
Parallel 2-opt optimization for multiple tours.

**Parameters:**
- `dist` (np.ndarray): Distance matrix
- `tours` (np.ndarray): Tours to optimize [batch_size, n_nodes]
- `max_iterations` (int): Maximum improvement iterations

**Returns:**
- `optimized_tours` (np.ndarray): Locally optimized tours

## External Dependencies

### Concorde TSP Solver
- **Purpose**: Exact TSP solving for verification and benchmarking
- **Integration**: Command-line execution via subprocess
- **Location**: `concorde/` directory after installation

### Installation
```bash
# Install Concorde (from tsp_nls directory)
chmod +x install_concorde.sh
./install_concorde.sh
```

## Usage Examples

### Basic TSP Solving

```python
import torch
from gfacs.tsp_nls.aco import ACO
from gfacs.tsp_nls.net import Net
from gfacs.tsp_nls.utils import gen_pyg_data

# Generate problem instance
n_nodes = 50
coordinates = torch.rand(n_nodes, 2)
pyg_data, distances = gen_pyg_data(coordinates, k_sparse=20)

# Load trained model
model = Net(gfn=True)
model.load_state_dict(torch.load('pretrained/tsp_nls/50/model.pt'))

# Generate heuristic
heu_vec = model(pyg_data)
heu_mat = model.reshape(pyg_data, heu_vec)

# Solve with ACO + local search
aco = ACO(distances, n_ants=20, heuristic=heu_mat, local_search_type='nls')
costs, log_probs, paths = aco.sample()
improved_paths = aco.local_search(paths)
improved_costs = aco.gen_path_costs(improved_paths)
```

### Training Loop

```python
from gfacs.tsp_nls.train import train_instance

# Training parameters
model = Net(gfn=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
data = [(pyg_data, distances)]  # List of instances

# Train for one epoch
loss = train_instance(
    model=model,
    optimizer=optimizer,
    data=data,
    n_ants=20,
    guided_exploration=True,
    beta=100.0
)
```

### Command Line Usage

```bash
# Train model
python train.py 200 --epochs 50 --batch-size 20

# Test model
python test.py 200 -p pretrained/tsp_nls/200/model.pt --n-ants 50

# Test on TSPLIB
python test_tsplib.py 200 -p pretrained/tsp_nls/200/model.pt
```

## Performance Characteristics

### Time Complexity
- **ACO Sampling**: O(n_ants × n_nodes × n_edges)
- **Neural Forward Pass**: O(n_nodes × k_sparse × depth)
- **2-opt Local Search**: O(n_nodes²) per tour
- **Batch Processing**: Parallel across n_ants

### Space Complexity
- **Distance Matrix**: O(n_nodes²)
- **Neural Network**: O(n_nodes × k_sparse × units)
- **Pheromone Matrix**: O(n_nodes²)

### Memory Requirements
- **Small instances** (n ≤ 100): ~100MB
- **Medium instances** (100 < n ≤ 500): ~1GB
- **Large instances** (n > 500): 4GB+

## Implementation Notes

### Numba Optimizations
- Path generation uses numba-compiled functions for performance
- Distance calculations are vectorized
- Local search operates on NumPy arrays for efficiency

### Device Support
- Automatic device placement for tensors
- CPU fallback for systems without CUDA
- Mixed precision support for training

### Numerical Stability
- Cost shifting to avoid negative values
- Epsilon addition to prevent division by zero
- Log probability clipping for GFlowNet training

## Testing and Validation

### Unit Tests
- ACO path generation and cost calculation
- Neural network forward passes
- Local search improvements
- Data loading and preprocessing

### Integration Tests
- End-to-end training pipeline
- Model checkpoint loading/saving
- TSPLIB benchmark evaluation

### Performance Benchmarks
- Convergence speed vs. baseline ACO
- Solution quality on standard instances
- Training time per epoch
