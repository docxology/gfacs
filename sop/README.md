# GFACS for Sequential Ordering Problem

This module implements ACO with GFlowNet sampling for the Sequential Ordering Problem (SOP). The goal is to find minimum-cost sequences that respect precedence constraints between tasks.

## Problem Description

SOP involves ordering tasks with:
- **Precedence Constraints**: Task i must precede task j
- **Processing Costs**: Cost of transitioning between tasks
- **Constraint Satisfaction**: Must respect all precedence relations
- **Optimal Sequencing**: Minimum total transition cost

## Installation

### Prerequisites
- PyTorch 2.1.1+ (CUDA recommended)
- NumPy, Numba

## Dataset Generation

```bash
python utils.py
```

## Training

```bash
python train.py $N
```

## Testing

```bash
python test.py $N -p "checkpoint"
```

## Key Features

- **Precedence Constraints**: Partial order on tasks
- **Feasible Sequences**: Topological ordering requirement
- **Transition Costs**: Task-dependent processing costs
- **Constraint-Aware**: Neural network respects precedence

## Usage Example

```python
from gfacs.sop.aco import ACO

aco = ACO(distances=costs, prec_cons=constraints, heuristic=heu_mat)
costs, log_probs, sequences = aco.sample()
```

## File Structure

```
sop/
├── aco.py
├── net.py
├── train.py
├── test.py
├── utils.py
├── README.md
└── AGENTS.md
```
