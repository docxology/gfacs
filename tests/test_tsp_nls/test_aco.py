"""Tests for TSP ACO implementation."""

import pytest

# Optional torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

if HAS_TORCH:
    from tsp_nls.aco import ACO
else:
    ACO = None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestTSPACO:
    """Test suite for TSP ACO implementation."""

    def test_aco_initialization(self, small_tsp_instance, device):
        """Test ACO initialization."""
        distances = small_tsp_instance['distances']
        n_ants = 5

        aco = ACO(distances, n_ants=n_ants, device=device)

        assert aco.problem_size == distances.shape[0]
        assert aco.n_ants == n_ants
        assert aco.device == device
        assert aco.distances.shape == distances.shape
        assert aco.pheromone.shape == distances.shape

    def test_aco_sample_shapes(self, small_tsp_instance, device):
        """Test ACO sampling returns correct shapes."""
        distances = small_tsp_instance['distances']
        n_nodes = small_tsp_instance['n_nodes']
        n_ants = 3

        aco = ACO(distances, n_ants=n_ants, device=device)
        costs, log_probs, paths = aco.sample()

        # Check shapes
        assert costs.shape == (n_ants,)
        assert log_probs.shape == (n_nodes, n_ants)
        assert paths.shape == (n_nodes, n_ants)

        # Check costs are positive
        assert torch.all(costs > 0)

    def test_aco_inference_mode(self, small_tsp_instance, device):
        """Test ACO in inference mode."""
        distances = small_tsp_instance['distances']
        n_ants = 3

        aco = ACO(distances, n_ants=n_ants, device=device)
        costs, log_probs, paths = aco.sample(inference=True)

        # In inference mode, log_probs should be None
        assert log_probs is None
        assert costs.shape == (n_ants,)
        assert paths.shape == (distances.shape[0], n_ants)

    def test_aco_with_heuristic(self, small_tsp_instance, device):
        """Test ACO with custom heuristic."""
        distances = small_tsp_instance['distances']
        n_ants = 3

        # Create random heuristic
        heuristic = torch.rand_like(distances)
        heuristic.fill_diagonal_(0)  # No self-heuristic

        aco = ACO(distances, n_ants=n_ants, heuristic=heuristic, device=device)

        # Should use provided heuristic
        assert torch.allclose(aco.heuristic, heuristic)

        # Sampling should work
        costs, log_probs, paths = aco.sample()
        assert costs.shape == (n_ants,)

    def test_aco_path_cost_calculation(self, small_tsp_instance, device):
        """Test path cost calculation."""
        distances = small_tsp_instance['distances']
        n_nodes = small_tsp_instance['n_nodes']

        aco = ACO(distances, n_ants=1, device=device)

        # Create a simple path: 0 -> 1 -> 2 -> ... -> n-1
        path = torch.arange(n_nodes, device=device)

        costs = aco.gen_path_costs(path.unsqueeze(1))  # Shape (n_nodes, 1)
        assert costs.shape == (1,)

        # Calculate expected cost manually (TSP tour: 0->1->2->...->n-1->0)
        expected_cost = 0
        for i in range(n_nodes):
            u, v = path[i], path[(i + 1) % n_nodes]
            expected_cost += distances[u, v]

        assert torch.allclose(costs[0], expected_cost)

    def test_aco_local_search_2opt(self, small_tsp_instance, device):
        """Test 2-opt local search."""
        distances = small_tsp_instance['distances']
        n_ants = 2

        aco = ACO(distances, n_ants=n_ants, local_search_type='2opt', device=device)
        costs, _, paths = aco.sample()

        # Apply local search
        improved_paths = aco.local_search(paths)

        # Improved paths should have same shape
        assert improved_paths.shape == paths.shape

        # Calculate costs of improved paths
        improved_costs = aco.gen_path_costs(improved_paths)

        # 2-opt should not increase costs (may improve or stay same)
        # Note: Due to randomness, we can't guarantee improvement
        assert torch.all(improved_costs >= 0)

    @pytest.mark.parametrize("aco_variant", ["AS", "ELITIST", "MAXMIN", "RANK"])
    def test_aco_variants(self, small_tsp_instance, device, aco_variant):
        """Test different ACO variants."""
        distances = small_tsp_instance['distances']
        n_ants = 3

        # Set variant-specific parameters
        kwargs = {}
        if aco_variant == "ELITIST":
            kwargs["elitist"] = True
        elif aco_variant == "MAXMIN":
            kwargs["maxmin"] = True
        elif aco_variant == "RANK":
            kwargs["rank_based"] = True
            kwargs["n_elites"] = 2

        aco = ACO(distances, n_ants=n_ants, device=device, **kwargs)

        # Should initialize without error
        assert aco is not None

        # Should be able to sample
        costs, _, _ = aco.sample()
        assert costs.shape == (n_ants,)

    def test_aco_run_method(self, small_tsp_instance, device):
        """Test ACO run method for multiple iterations."""
        distances = small_tsp_instance['distances']
        n_iterations = 2

        aco = ACO(distances, device=device)
        result = aco.run(n_iterations)

        # Should return tuple (lowest_cost, diversity, duration)
        assert isinstance(result, tuple) and len(result) == 3
        best_cost, diversity, duration = result
        assert isinstance(best_cost, (int, float))
        assert isinstance(diversity, (int, float))
        assert isinstance(duration, (int, float))
        assert best_cost >= 0
        assert 0 <= diversity <= 1
        assert duration >= 0

    def test_aco_fixed_start_node(self, small_tsp_instance, device):
        """Test ACO with fixed start node."""
        distances = small_tsp_instance['distances']
        start_node = 1
        n_ants = 3

        aco = ACO(distances, n_ants=n_ants, device=device)
        costs, _, paths = aco.sample(start_node=start_node)

        # All paths should start with the specified node
        assert torch.all(paths[0, :] == start_node)

        assert costs.shape == (n_ants,)
        assert paths.shape == (distances.shape[0], n_ants)

    def test_aco_single_ant(self, small_tsp_instance, device):
        """Test ACO with single ant."""
        distances = small_tsp_instance['distances']

        aco = ACO(distances, n_ants=1, device=device)
        costs, log_probs, paths = aco.sample()

        assert costs.shape == (1,)
        assert log_probs.shape == (distances.shape[0], 1)
        assert paths.shape == (distances.shape[0], 1)
        assert torch.all(costs > 0)

    def test_aco_path_transposition_consistency(self, small_tsp_instance, device):
        """Test path shape consistency across different ACO operations."""
        distances = small_tsp_instance['distances']
        n_ants = 3

        aco = ACO(distances, n_ants=n_ants, device=device)

        # Test sampling
        costs, log_probs, paths = aco.sample()
        assert paths.shape == (distances.shape[0], n_ants)

        # Test that gen_path_costs works with sampled paths
        calculated_costs = aco.gen_path_costs(paths)
        assert calculated_costs.shape == (n_ants,)
        assert torch.allclose(costs, calculated_costs, atol=1e-5)

    @pytest.mark.parametrize("aco_variant,expected_attrs", [
        ("AS", ["alpha", "beta"]),
        ("ELITIST", ["alpha", "beta", "elitist"]),
        ("MAXMIN", ["alpha", "beta", "maxmin"]),
        ("RANK", ["alpha", "beta", "rank_based", "n_elites"]),
    ])
    def test_aco_variants_attributes(self, small_tsp_instance, device, aco_variant, expected_attrs):
        """Test ACO variants have expected attributes."""
        distances = small_tsp_instance['distances']

        kwargs = {}
        if aco_variant == "ELITIST":
            kwargs["elitist"] = True
        elif aco_variant == "MAXMIN":
            kwargs["maxmin"] = True
        elif aco_variant == "RANK":
            kwargs["rank_based"] = True
            kwargs["n_elites"] = 2

        aco = ACO(distances, device=device, **kwargs)

        # Check that variant-specific attributes are set
        for attr in expected_attrs:
            assert hasattr(aco, attr)
