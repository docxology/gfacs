# GFACS for Single Machine Total Weighted Tardiness Problem

This module implements ACO with GFlowNet sampling for the Single Machine Total Weighted Tardiness Problem (SMTWTP). The goal is to schedule jobs to minimize total weighted tardiness on a single machine.

## Problem Description

SMTWTP involves scheduling jobs with:
- **Due Times**: Desired completion times for each job
- **Processing Times**: Time required to process each job
- **Weights**: Importance multipliers for tardiness
- **Single Machine**: Jobs processed sequentially
- **Tardiness**: max(0, completion_time - due_time)

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

- **Job Scheduling**: Permutation-based job ordering
- **Weighted Tardiness**: Importance-weighted penalties
- **Completion Times**: Sequential processing calculation
- **Neural Learning**: Learns scheduling priorities

## Usage Example

```python
from gfacs.smtwtp.aco import ACO

aco = ACO(due_time=due_times, weights=weights, processing_time=processing_times, heuristic=heu_mat)
tardiness, log_probs, schedules = aco.sample()
```

## File Structure

```
smtwtp/
├── aco.py
├── net.py
├── train.py
├── test.py
├── utils.py
├── README.md
└── AGENTS.md
```
