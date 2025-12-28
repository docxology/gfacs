# GFACS Experiment Summary

**Experiment:** full_experiment
**Date:** 2025-12-27T16:04:28.536573
**Duration:** 3.997418165206909 seconds

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
| tsp_nls | None | 3.393218994140625 | 50 |
| cvrp_nls | None | 0.004044055938720703 | 50 |
| cvrptw_nls | None | 0.027904033660888672 | 50 |
| bpp | None | 0.004137992858886719 | 50 |
| op | None | 0.002674102783203125 | 50 |
| pctsp | None | 0.0028302669525146484 | 50 |
| smtwtp | None | 0.0031518936157226562 | 50 |
| sop | None | 0.0022809505462646484 | 50 |

## Output Structure

```
full_experiment_20251227_160424/
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
