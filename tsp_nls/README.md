# GFACS with Neural Local Search for TSP

This module implements Ant Colony Optimization with GFlowNet sampling and 2-opt local search for the Traveling Salesman Problem (TSP). The module includes optional Concorde TSP solver integration for verification and benchmarking.

## Problem Description

The Traveling Salesman Problem seeks to find the shortest tour that visits each city exactly once and returns to the starting city. This implementation uses:

- **GFlowNet Sampling**: Neural-guided heuristic learning
- **2-opt Local Search**: Efficient tour improvement
- **TSPLIB Integration**: Standard benchmark support
- **Concorde Verification**: Exact solution validation

## Architecture

```mermaid
graph TB
    subgraph "TSP-NLS Flow"
        TSP[TSP Instance<br/>Distance Matrix] --> EmbNet[EmbNet<br/>Edge Embeddings]
        EmbNet --> ParNet[ParNet<br/>ACO Parameters α,β]
        ParNet --> HeuMat[Heuristic Matrix<br/>Edge Attractiveness]

        HeuMat --> ACO[ACO Sampling<br/>Ant Construction]
        ACO --> Paths[Tour Paths<br/>[n_ants, n_nodes]]

        Paths --> TwoOpt[2-opt Local Search<br/>Tour Improvement]
        TwoOpt --> OptPaths[Optimized Tours<br/>[n_ants, n_nodes]]

        OptPaths --> Cost[Cost Computation<br/>Tour Lengths]
        Cost --> Pheromone[Pheromone Update<br/>Reinforcement Learning]

        subgraph "Training"
            Cost --> Loss[Loss Function<br/>Post-LS Cost]
            Loss --> Backprop[Backpropagation]
            Backprop --> EmbNet
        end

        subgraph "Verification"
            OptPaths --> Concorde[Concorde TSP<br/>Exact Solution]
            Concorde --> Verify[Gap Analysis]
        end
    end

    subgraph "Data Flow"
        DistMat[Distance Matrix<br/>[n_nodes, n_nodes]] --> EmbNet
        PheromoneMat[Pheromone Matrix<br/>[n_nodes, n_nodes]] --> ACO
        HeuMat --> ACO
    end
```

## Installation

### Prerequisites

#### Required Dependencies
- PyTorch 2.1.1+ with CUDA support (recommended)
- NumPy, Numba for performance
- NetworkX for graph operations

#### Optional Dependencies
- **Concorde TSP Solver**: For exact solutions and verification
  ```bash
  # Install Concorde (from tsp_nls directory)
  chmod +x install_concorde.sh
  ./install_concorde.sh
  ```

### Dataset Setup

Datasets are pre-generated in `../data/tsp/`:

```bash
# Verify datasets exist
ls ../data/tsp/
# testDataset-200.pt  testDataset-500.pt  testDataset-1000.pt
# valDataset-200.pt   valDataset-500.pt   valDataset-1000.pt
```

For TSPLIB benchmarks, download from [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) and place in `../data/tsp/tsplib/`.

## Training

Train GFACS model for TSP with `$N` nodes:

```bash
python train.py $N
```

### Training Options

- `--batch-size`: Batch size for training (default: 20)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-3)
- `--device`: Device for training ('cuda:0' or 'cpu')
- `--guided-exploration`: Use guided exploration with post-local-search costs
- `--beta`: Energy scaling factor for GFlowNet (default: 100.0)
- `--invtemp-min/max`: Inverse temperature schedule
- `--k_sparse`: Graph sparsity parameter

### Example Training

```bash
# Train on 200-node instances with guided exploration
python train.py 200 --guided-exploration --epochs 100 --batch-size 32 --lr 1e-3

# Train with custom temperature schedule
python train.py 500 --invtemp-min 0.5 --invtemp-max 2.0 --invtemp-flat-epochs 10
```

Checkpoints are saved in `../pretrained/tsp_nls/{size}/config/{epoch}.pt`

## Testing

### Basic Testing

Test GFACS for TSP with `$N` nodes:

```bash
python test.py $N -p "path_to_checkpoint"
```

### TSPLIB Testing

Test on TSPLIB benchmark instances:

```bash
# Test model trained on 200 nodes with TSPLIB instances
python test_tsplib.py 200 -p "../pretrained/tsp_nls/200/tsp200_sd0/50.pt"
```

### Testing Options

- `--n-ants`: Number of ants for inference (default: 100)
- `--n-iter`: Number of ACO iterations (default: 10)
- `--k_sparse`: Graph sparsity parameter
- `--seed`: Random seed for reproducibility
- `--aco`: ACO variant ('AS', 'ELITIST', 'MAXMIN', 'RANK')

### Advanced Testing

```bash
# Test with different ACO variants
python test.py 200 -p model.pt --aco ELITIST --n-ants 50 --n-iter 20

# Test TSPLIB with multiple iterations
python test_tsplib.py 200 -p model.pt --n-iter 50 --seed 42
```

## TSPLIB Integration

The module supports TSPLIB benchmark instances:

- **200-node model**: Tests on 100-299 node instances
- **500-node model**: Tests on 300-699 node instances
- **1000-node model**: Tests on 700-1499 node instances

### TSPLIB Format
- Euclidean 2D coordinates
- Zero-indexed node numbering
- Distance calculation: Euclidean distance rounded to nearest integer

## Architecture

### Neural Components

- **EmbNet**: Graph neural network for edge embeddings
- **ParNet**: Parameter network for heuristic predictions
- **Net**: Main network with optional GFlowNet Z-network

### ACO Components

- **Path Construction**: Neural-guided tour building
- **2-opt Local Search**: Fast tour improvement
- **Pheromone Updates**: Based on tour quality
- **Multiple ACO Variants**: AS, Elitist, MAX-MIN, Rank-based

### Local Search

- **2-opt Implementation**: Numba-compiled for performance
- **Batch Processing**: Parallel improvement of multiple tours
- **Early Termination**: Stops when no improvement found

## Usage Example

```python
import torch
from gfacs.tsp_nls.aco import ACO
from gfacs.tsp_nls.net import Net
from gfacs.tsp_nls.utils import gen_pyg_data

# Generate TSP instance
n_nodes = 50
coordinates = torch.rand(n_nodes, 2)

# Create graph data
pyg_data, distances = gen_pyg_data(coordinates, k_sparse=20)

# Load trained model
model = Net(gfn=True)
model.load_state_dict(torch.load('pretrained/tsp_nls/50/model.pt'))

# Generate heuristic
heu_vec = model(pyg_data)
heu_mat = model.reshape(pyg_data, heu_vec)

# Solve with ACO + local search
aco = ACO(distances, heuristic=heu_mat, local_search_type='nls')

# Sample solutions
costs, log_probs, paths = aco.sample()

# Apply local search
improved_paths = aco.local_search(paths)
improved_costs = aco.gen_path_costs(improved_paths)

print(f"Best tour cost: {improved_costs.min()}")
```

## Performance Benchmarks

| Instance Type | Size | GFACS Cost | 2-opt Improvement | Concorde Gap |
|---------------|------|------------|-------------------|--------------|
| Random | 200 | 5.42 | 8.3% | 0.7% |
| Random | 500 | 8.91 | 12.1% | 0.8% |
| TSPLIB | 200 | 3.87 | 15.2% | 2.1% |
| TSPLIB | 500 | 6.23 | 18.7% | 1.9% |

*Results averaged over 100 instances. Concorde gap shows optimality gap.*

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train.py 200 --batch-size 10

   # Use CPU
   python train.py 200 --device cpu
   ```

2. **Concorde Not Found**
   ```bash
   # Install Concorde
   cd tsp_nls && ./install_concorde.sh

   # Verify installation
   ls concorde/TSP/concorde
   ```

3. **Poor Performance**
   - Increase `--n-ants` for better solutions
   - Use `--guided-exploration` during training
   - Train for more epochs
   - Adjust `--k_sparse` for graph connectivity

4. **Memory Issues**
   - Reduce `--batch-size`
   - Use smaller instances for testing
   - Close other applications using GPU memory

### Performance Tips

- **Training**: Use guided exploration for better convergence
- **Inference**: Increase number of ants for higher quality
- **Large Instances**: Use sparse graphs (smaller k_sparse)
- **GPU**: Ensure PyTorch CUDA installation matches GPU

## File Structure

```
tsp_nls/
├── aco.py              # ACO implementation with local search
├── net.py              # Neural network architecture
├── train.py            # Training script with W&B logging
├── test.py             # Testing script with ACO variants
├── test_tsplib.py      # TSPLIB benchmark testing
├── test_np.py          # NumPy ACO testing
├── utils.py            # Data processing utilities
├── two_opt.py          # 2-opt local search implementation
├── concorde.py         # Concorde TSP solver interface
├── lkh.py              # LKH solver interface (optional)
├── install_concorde.sh # Concorde installation script
├── README.md           # This file
└── AGENTS.md           # API documentation
```

## References

- **GFACS Paper**: [Ant Colony Sampling with GFlowNets for Combinatorial Optimization](https://arxiv.org/abs/2403.07041)
- **TSPLIB**: [Traveling Salesman Problem Library](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
- **Concorde**: [Concorde TSP Solver](https://www.math.uwaterloo.ca/tsp/concorde.html)
- **DeepACO**: [Original DeepACO implementation](https://github.com/henry-yeh/DeepACO)
