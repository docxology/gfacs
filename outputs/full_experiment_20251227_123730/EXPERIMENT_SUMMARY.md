# GFACS Experiment Summary

**Experiment:** full_experiment
**Date:** 2025-12-27T12:37:33.834890
**Duration:** 3.0198540687561035 seconds

## Problems Executed

- tsp_nls
- cvrp_nls
- cvrptw_nls
- bpp
- op
- pctsp
- smtwtp
- sop

## Results Summary

| Problem | Best Cost | Duration | Iterations |
|---------|-----------|----------|------------|
| tsp_nls | None | 2.426510810852051 | 0 |
| cvrp_nls | None | 0.0019230842590332031 | 0 |
| cvrptw_nls | None | 0.021129131317138672 | 0 |
| bpp | None | 0.0021462440490722656 | 0 |
| op | None | 0.0019791126251220703 | 0 |
| pctsp | None | 0.0018761157989501953 | 0 |
| smtwtp | None | 0.001514434814453125 | 0 |
| sop | None | 0.0012869834899902344 | 0 |

## Output Structure

```
full_experiment_20251227_123730/
├── config/                 # All configurations
├── logs/                   # All logs
├── data/                   # All input/output data
│   ├── inputs/            # Input instances
│   ├── outputs/           # Solution outputs
│   └── results/           # Aggregated results
├── visualizations/        # Static plots
└── animations/            # Animated visualizations
```

## File Descriptions

- `config/experiment.yaml` - Main experiment configuration
- `logs/orchestrator.log` - Main orchestrator execution log
- `data/results/summary.json` - Overall experiment results
- `visualizations/` - Static plots and charts
- `animations/` - Interactive animations and videos
