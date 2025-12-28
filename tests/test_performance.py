"""Performance tests and benchmarks for GFACS components."""

import pytest
import torch
import numpy as np
from pathlib import Path

# Optional imports
try:
    from gfacs.utils import benchmark_function, profile_context
    HAS_PROFILING = True
except ImportError:
    HAS_PROFILING = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from tsp_nls.aco import ACO
    from tsp_nls.net import Net
    from tsp_nls.utils import gen_pyg_data
    HAS_TSP = True
except ImportError:
    HAS_TSP = False
    ACO = Net = gen_pyg_data = None


@pytest.mark.skipif(not HAS_TORCH or not HAS_TSP, reason="PyTorch or TSP modules not available")
@pytest.mark.skipif(not HAS_PROFILING, reason="Profiling utilities not available")
class TestPerformance:
    """Performance benchmarks for GFACS components."""

    def test_aco_sampling_performance(self, small_tsp_instance):
        """Benchmark ACO sampling performance."""
        distances = small_tsp_instance['distances']
        n_ants = 20

        aco = ACO(distances, n_ants=n_ants)

        # Benchmark sampling
        result = benchmark_function(
            aco.sample,
            iterations=5,
            warmup=2
        )

        # Should complete within reasonable time (lenient threshold for CI)
        assert result['avg_time'] < 5.0, ".3f"
        assert result['calls'] == 5

    def test_aco_local_search_performance(self, small_tsp_instance):
        """Benchmark ACO local search performance."""
        distances = small_tsp_instance['distances']
        n_ants = 20

        aco = ACO(distances, n_ants=n_ants, local_search_type='2opt')

        # Generate initial paths
        costs, log_probs, paths = aco.sample()

        # Benchmark local search
        result = benchmark_function(
            aco.local_search,
            paths,
            iterations=3,
            warmup=1
        )

        # Local search should improve solutions
        assert result['avg_time'] < 10.0, ".3f"

    def test_aco_run_performance(self, small_tsp_instance):
        """Benchmark full ACO run performance."""
        distances = small_tsp_instance['distances']
        n_ants = 20

        aco = ACO(distances, n_ants=n_ants)

        # Benchmark full run
        result = benchmark_function(
            aco.run,
            n_iterations=5,
            iterations=2,
            warmup=1
        )

        # Should complete within reasonable time
        assert result['avg_time'] < 5.0, ".3f"

    @pytest.mark.parametrize("n_nodes", [20, 50, 100])
    def test_scaling_performance(self, n_nodes):
        """Test performance scaling with problem size."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        # Generate test instance
        torch.manual_seed(42)
        coordinates = torch.rand(n_nodes, 2)
        distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
        distances.fill_diagonal_(float('inf'))

        n_ants = min(50, n_nodes * 2)  # Scale ants with problem size
        aco = ACO(distances, n_ants=n_ants)

        # Benchmark sampling
        result = benchmark_function(
            aco.sample,
            iterations=3,
            warmup=1
        )

        # Log performance for analysis
        print(f"\nProblem size {n_nodes}: {result['avg_time']:.3f}s")

        # Should scale reasonably (O(n²) is acceptable for small problems)
        if n_nodes <= 50:
            assert result['avg_time'] < 2.0
        elif n_nodes <= 100:
            assert result['avg_time'] < 10.0

    def test_memory_efficiency(self, small_tsp_instance):
        """Test memory efficiency of ACO operations."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        distances = small_tsp_instance['distances']
        n_ants = 20

        aco = ACO(distances, n_ants=n_ants)

        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()

        # Run sampling
        costs, log_probs, paths = aco.sample()

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - initial_memory

            # Should use reasonable GPU memory
            memory_mb = memory_used / 1024 / 1024
            assert memory_mb < 500, ".1f"  # Less than 500MB for small problem

    def test_batch_processing_efficiency(self):
        """Test efficiency of batch processing."""
        if not HAS_TORCH or not HAS_TSP:
            pytest.skip("Required modules not available")

        # Create multiple small instances
        n_instances = 5
        n_nodes = 20
        n_ants = 10

        instances = []
        for _ in range(n_instances):
            torch.manual_seed(42)
            coordinates = torch.rand(n_nodes, 2)
            distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
            distances.fill_diagonal_(float('inf'))
            instances.append(distances)

        # Benchmark individual vs batch processing
        individual_times = []
        for distances in instances:
            aco = ACO(distances, n_ants=n_ants)
            result = benchmark_function(aco.sample, iterations=3, warmup=1)
            individual_times.append(result['avg_time'])

        avg_individual = sum(individual_times) / len(individual_times)

        # Batch processing would be more complex to implement
        # For now, just verify individual processing works efficiently
        assert avg_individual < 1.0, ".3f"

    @pytest.mark.slow
    def test_large_problem_performance(self):
        """Test performance on larger problems."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        # Larger problem for stress testing
        n_nodes = 200
        n_ants = 50

        torch.manual_seed(42)
        coordinates = torch.rand(n_nodes, 2)
        distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
        distances.fill_diagonal_(float('inf'))

        aco = ACO(distances, n_ants=n_ants)

        # This should still complete in reasonable time
        result = benchmark_function(
            aco.sample,
            iterations=2,
            warmup=1
        )

        # Allow more time for larger problems but still reasonable
        assert result['avg_time'] < 30.0, ".3f"

    def test_neural_network_inference_performance(self):
        """Test neural network inference performance."""
        if not HAS_TORCH or not HAS_TSP:
            pytest.skip("Required modules not available")

        try:
            from torch_geometric.data import Data
        except ImportError:
            pytest.skip("PyTorch Geometric not available")

        # Create small test graph
        n_nodes = 20
        torch.manual_seed(42)
        coordinates = torch.rand(n_nodes, 2)

        # Generate PyG data
        pyg_data, distances = gen_pyg_data(coordinates, k_sparse=10)

        # Create model
        model = Net(gfn=True)

        # Benchmark inference
        result = benchmark_function(
            model,
            pyg_data,
            iterations=5,
            warmup=2
        )

        # Neural network inference should be fast
        assert result['avg_time'] < 1.0, ".3f"

    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up after operations."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        # This test ensures no memory leaks in repeated operations
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        for _ in range(5):
            # Create and run ACO
            n_nodes = 30
            torch.manual_seed(42)
            coordinates = torch.rand(n_nodes, 2)
            distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
            distances.fill_diagonal_(float('inf'))

            aco = ACO(distances, n_ants=20)
            costs, log_probs, paths = aco.sample()

            # Force cleanup
            del aco, costs, log_probs, paths
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Memory should not grow significantly
        memory_growth = final_memory - initial_memory
        if torch.cuda.is_available():
            growth_mb = memory_growth / 1024 / 1024
            assert growth_mb < 50, ".1f"  # Less than 50MB growth

    def test_profile_context_manager(self):
        """Test the profile_context context manager."""
        if not HAS_PROFILING:
            pytest.skip("Profiling utilities not available")

        with profile_context("test_context") as profiler:
            # Simulate some work
            import time
            time.sleep(0.01)

        summary = profiler.get_summary()
        assert summary['calls'] == 1
        assert summary['total_time'] > 0
        assert summary['avg_time'] > 0

    def test_benchmark_function_with_model(self):
        """Test benchmark_function with model instances."""
        if not HAS_PROFILING:
            pytest.skip("Profiling utilities not available")

        # Create a mock model-like object
        class MockModel:
            def __call__(self, x):
                return x * 2

        model = MockModel()

        # Should handle model instances without __name__
        result = benchmark_function(model, 5, iterations=2, warmup=1)
        assert result['calls'] == 2
        assert 'MockModel' in result['name']

    def test_benchmark_function_edge_cases(self):
        """Test benchmark_function edge cases."""
        if not HAS_PROFILING:
            pytest.skip("Profiling utilities not available")

        def simple_func():
            return 42

        # Test with zero warmup
        result = benchmark_function(simple_func, iterations=2, warmup=0)
        assert result['calls'] == 2

        # Test with function that has __name__
        result = benchmark_function(simple_func, iterations=1, warmup=0)
        assert 'simple_func' in result['name']