# GFACS Experiment Summary

**Experiment:** test_full_fixes
**Date:** 2025-12-27T12:40:27.368815
**Duration:** 2.0546340942382812 seconds

## Problems Executed

- tsp_nls
- cvrp_nls
- bpp

## Results Summary

| Problem | Best Cost | Duration | Iterations |
|---------|-----------|----------|------------|
| tsp_nls | None | 1.537438154220581 | 0 |
| cvrp_nls | None | 0.0005021095275878906 | 0 |
| bpp | None | 0.0029518604278564453 | 0 |

## Output Structure

```
test_full_fixes_20251227_124025/
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
