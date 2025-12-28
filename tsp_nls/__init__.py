"""TSP with Neural Local Search module.

This module implements Ant Colony Optimization with GFlowNet sampling
and 2-opt local search for the Traveling Salesman Problem.
"""

from .aco import ACO, ACO_NP
from . import utils

# Optional imports for neural network components
try:
    from .net import Net, EmbNet, ParNet
    _HAS_NEURAL_NETWORKS = True
except ImportError:
    Net = EmbNet = ParNet = None
    _HAS_NEURAL_NETWORKS = False

# Build __all__ based on available components
__all__ = [
    "ACO",
    "ACO_NP",
    "utils",
]

if _HAS_NEURAL_NETWORKS:
    __all__.extend([
        "Net",
        "EmbNet",
        "ParNet",
    ])
