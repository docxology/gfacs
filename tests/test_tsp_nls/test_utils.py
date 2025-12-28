"""Tests for TSP utilities."""

import pytest

# Optional torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

if HAS_TORCH:
    from tsp_nls.utils import gen_distance_matrix
    if HAS_TORCH_GEOMETRIC:
        from tsp_nls.utils import gen_pyg_data
    else:
        gen_pyg_data = None
else:
    gen_distance_matrix = gen_pyg_data = None


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCH_GEOMETRIC, reason="PyTorch or PyTorch Geometric not available")
class TestTSPUtils:
    """Test suite for TSP utility functions."""

    def test_gen_distance_matrix(self, small_tsp_instance):
        """Test distance matrix generation."""
        coordinates = small_tsp_instance['coordinates']
        n_nodes = small_tsp_instance['n_nodes']

        distances = gen_distance_matrix(coordinates)

        # Check shape
        assert distances.shape == (n_nodes, n_nodes)

        # Check diagonal is large value (1e9)
        assert torch.all(torch.diag(distances) == 1e9)

        # Check symmetry
        assert torch.allclose(distances, distances.t())

        # Check non-negative off-diagonal
        off_diag = distances[~torch.eye(n_nodes, dtype=bool)]
        assert torch.all(off_diag >= 0)

    def test_gen_pyg_data(self, small_tsp_instance):
        """Test PyTorch Geometric data generation."""
        coordinates = small_tsp_instance['coordinates']
        n_nodes = small_tsp_instance['n_nodes']
        k_sparse = 5

        pyg_data, distances = gen_pyg_data(coordinates, k_sparse)

        # Check data attributes
        assert hasattr(pyg_data, 'x')
        assert hasattr(pyg_data, 'edge_index')
        assert hasattr(pyg_data, 'edge_attr')

        # Check node features (coordinates)
        assert pyg_data.x.shape == (n_nodes, 2)
        assert torch.allclose(pyg_data.x, coordinates)

        # Check edge attributes shape
        n_edges = pyg_data.edge_index.shape[1]
        assert pyg_data.edge_attr.shape == (n_edges, 1)

        # Check distance matrix
        assert distances.shape == (n_nodes, n_nodes)

        # Verify edge distances match matrix
        for i, (u, v) in enumerate(pyg_data.edge_index.t()):
            expected_dist = distances[u, v]
            actual_dist = pyg_data.edge_attr[i, 0]
            assert torch.allclose(actual_dist, expected_dist)

    def test_gen_pyg_data_k_sparse(self, small_tsp_instance):
        """Test k-sparse graph generation."""
        coordinates = small_tsp_instance['coordinates']
        n_nodes = small_tsp_instance['n_nodes']
        k_sparse = 3

        pyg_data, _ = gen_pyg_data(coordinates, k_sparse)

        # Count edges per node
        edge_index = pyg_data.edge_index
        node_degrees = torch.bincount(edge_index[0], minlength=n_nodes)

        # Each node should have at most k_sparse outgoing edges
        assert torch.all(node_degrees <= k_sparse)

        # Check total edges
        expected_max_edges = n_nodes * k_sparse
        assert edge_index.shape[1] <= expected_max_edges

    def test_gen_pyg_data_fixed_start(self, small_tsp_instance):
        """Test data generation with fixed start node."""
        coordinates = small_tsp_instance['coordinates']
        n_nodes = small_tsp_instance['n_nodes']
        k_sparse = 5
        start_node = 2

        pyg_data, _ = gen_pyg_data(coordinates, k_sparse, start_node=start_node)

        # Node features should include start node indicator
        assert pyg_data.x.shape == (n_nodes, 1)
        assert pyg_data.x[start_node, 0] == 1.0
        assert torch.all(pyg_data.x[torch.arange(n_nodes) != start_node, 0] == 0.0)
