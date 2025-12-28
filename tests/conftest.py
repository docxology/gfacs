"""Shared fixtures and configuration for GFACS tests."""

import pytest
import numpy as np
from pathlib import Path

# Optional imports with fallbacks
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import torch_geometric
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    torch_geometric = None
    HAS_TORCH_GEOMETRIC = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    imageio = None
    HAS_IMAGEIO = False

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    matplotlib = None
    HAS_MATPLOTLIB = False

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)


@pytest.fixture
def device():
    """Provide appropriate device for testing."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def small_tsp_instance():
    """Create a small TSP instance for testing (10 nodes)."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    n_nodes = 10
    torch.manual_seed(42)
    coordinates = torch.rand(n_nodes, 2)

    # Calculate distance matrix
    distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
    distances.fill_diagonal_(float('inf'))  # Prevent self-loops

    return {
        'coordinates': coordinates,
        'distances': distances,
        'n_nodes': n_nodes
    }


@pytest.fixture
def small_cvrp_instance():
    """Create a small CVRP instance for testing (10 nodes)."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    n_nodes = 10
    torch.manual_seed(42)
    coordinates = torch.rand(n_nodes, 2)
    demands = torch.rand(n_nodes) * 0.5
    demands[0] = 0  # Depot has no demand

    # Calculate distance matrix
    distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
    distances.fill_diagonal_(float('inf'))

    return {
        'coordinates': coordinates,
        'distances': distances,
        'demands': demands,
        'capacity': 1.0,
        'n_nodes': n_nodes
    }


@pytest.fixture
def small_cvrptw_instance():
    """Create a small CVRPTW instance for testing (10 nodes)."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    n_nodes = 10
    torch.manual_seed(42)
    coordinates = torch.rand(n_nodes, 2)
    demands = torch.rand(n_nodes) * 0.5
    demands[0] = 0

    # Time windows: [early, late] normalized to [0, 1]
    windows = torch.zeros(n_nodes, 2)
    windows[:, 1] = 1.0  # Full time horizon
    for i in range(1, n_nodes):
        early = torch.rand(1) * 0.6
        late = early + 0.3 + torch.rand(1) * 0.2
        windows[i] = torch.cat([early, late])

    # Calculate distance matrix
    distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
    distances.fill_diagonal_(float('inf'))

    return {
        'coordinates': coordinates,
        'distances': distances,
        'demands': demands,
        'windows': windows,
        'capacity': 1.0,
        'n_nodes': n_nodes
    }


@pytest.fixture
def small_bpp_instance():
    """Create a small BPP instance for testing (20 items)."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    n_items = 20
    torch.manual_seed(42)
    item_sizes = torch.rand(n_items) * 0.8 + 0.1  # Sizes 0.1-0.9
    capacity = 1.0

    return {
        'item_sizes': item_sizes,
        'capacity': capacity,
        'n_items': n_items
    }


@pytest.fixture
def small_op_instance():
    """Create a small OP instance for testing (15 nodes)."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    n_nodes = 15
    torch.manual_seed(42)
    coordinates = torch.rand(n_nodes, 2)
    prizes = torch.rand(n_nodes) * 10
    prizes[0] = 0  # Depot has no prize
    max_len = 4.0

    return {
        'coordinates': coordinates,
        'prizes': prizes,
        'max_len': max_len,
        'n_nodes': n_nodes
    }


@pytest.fixture
def small_pctsp_instance():
    """Create a small PCTSP instance for testing (15 nodes)."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    n_nodes = 15
    torch.manual_seed(42)
    coordinates = torch.rand(n_nodes, 2)
    prizes = torch.rand(n_nodes) * 10
    penalties = torch.rand(n_nodes) * 5
    prizes[0] = penalties[0] = 0

    return {
        'coordinates': coordinates,
        'prizes': prizes,
        'penalties': penalties,
        'n_nodes': n_nodes
    }


@pytest.fixture
def small_smtwtp_instance():
    """Create a small SMTWTP instance for testing (10 jobs)."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    n_jobs = 10
    torch.manual_seed(42)
    due_times = torch.rand(n_jobs) * 100 + 50
    weights = torch.rand(n_jobs) * 5 + 1
    processing_times = torch.rand(n_jobs) * 10 + 5

    return {
        'due_times': due_times,
        'weights': weights,
        'processing_times': processing_times,
        'n_jobs': n_jobs
    }


@pytest.fixture
def small_sop_instance():
    """Create a small SOP instance for testing (12 tasks)."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    n_tasks = 12
    torch.manual_seed(42)
    coordinates = torch.rand(n_tasks, 2)

    # Create precedence constraints (random DAG)
    prec_cons = torch.zeros(n_tasks, n_tasks)
    for i in range(n_tasks):
        for j in range(i+1, min(i+3, n_tasks)):
            if torch.rand(1) > 0.5:
                prec_cons[i,j] = 1

    # Processing costs
    processing_costs = torch.rand(n_tasks, n_tasks) * 10

    return {
        'coordinates': coordinates,
        'prec_cons': prec_cons,
        'processing_costs': processing_costs,
        'n_tasks': n_tasks
    }


@pytest.fixture
def mock_model():
    """Create a mock neural network model for testing."""
    from unittest.mock import MagicMock

    model = MagicMock()
    model.eval.return_value = None
    model.train.return_value = None

    return model


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def requires_torch():
    """Skip test if PyTorch is not available."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")


@pytest.fixture
def requires_torch_geometric():
    """Skip test if PyTorch Geometric is not available."""
    if not HAS_TORCH_GEOMETRIC:
        pytest.skip("PyTorch Geometric not available")


@pytest.fixture
def requires_visualization():
    """Skip test if visualization libraries are not available."""
    missing = []
    if not HAS_MATPLOTLIB:
        missing.append("matplotlib")
    if not HAS_IMAGEIO:
        missing.append("imageio")

    if missing:
        pytest.skip(f"Visualization libraries not available: {', '.join(missing)}")


@pytest.fixture
def torch_device():
    """Provide PyTorch device, skip if not available."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def logger(tmp_path):
    """Create a real GFACS logger for testing."""
    from gfacs.utils.logging import setup_experiment_logging
    experiment_dir = tmp_path / "test_experiment"
    experiment_dir.mkdir(exist_ok=True)
    logger = setup_experiment_logging("test_experiment", experiment_dir, log_level="DEBUG")
    return logger


@pytest.fixture
def io_manager(tmp_path):
    """Create a real ExperimentIO manager for testing."""
    from gfacs.utils.io import ExperimentIO
    base_dir = tmp_path / "outputs"
    base_dir.mkdir(exist_ok=True)
    io_manager = ExperimentIO(base_dir)
    return io_manager


@pytest.fixture
def real_logger():
    """Create a real GFACS logger for testing."""
    from gfacs.utils.logging import GFACSLogger
    return GFACSLogger("test")


@pytest.fixture
def real_io_manager(tmp_path):
    """Create a real experiment I/O manager for testing."""
    from gfacs.utils.io import ExperimentIO
    return ExperimentIO(tmp_path)


@pytest.fixture
def real_visualizer():
    """Create a real GFACS visualizer for testing."""
    if not HAS_MATPLOTLIB:
        pytest.skip("Matplotlib not available for visualization tests")
    from gfacs.utils.visualization import get_visualizer
    return get_visualizer()


@pytest.fixture
def real_animator():
    """Create a real GFACS animator for testing."""
    if not HAS_ANIMATIONS:
        pytest.skip("Animation dependencies not available")
    from gfacs.utils.animations import get_animator
    return get_animator()


@pytest.fixture
def minimal_problem_config():
    """Create a minimal problem configuration for testing."""
    from gfacs.orchestrator import ProblemConfig
    return ProblemConfig(
        name="tsp_nls",
        size=10,  # Very small for fast tests
        n_ants=5,
        n_iterations=5
    )


@pytest.fixture
def minimal_orchestrator_config(minimal_problem_config):
    """Create a minimal orchestrator configuration for testing."""
    from gfacs.orchestrator import OrchestratorConfig
    return OrchestratorConfig(
        experiment_name="test_experiment",
        problems=[minimal_problem_config],
        enable_visualizations=False,  # Disable for faster tests
        enable_animations=False,  # Disable for faster tests
        log_level="INFO"
    )


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    if HAS_TORCH:
        torch.manual_seed(42)
    np.random.seed(42)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "torch: marks tests that require PyTorch"
    )
    config.addinivalue_line(
        "markers", "torch_geometric: marks tests that require PyTorch Geometric"
    )
    config.addinivalue_line(
        "markers", "visualization: marks tests that require visualization libraries"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location and skip if dependencies missing."""
    for item in items:
        # Mark GPU tests
        if "gpu" in str(item.fspath) or "cuda" in str(item.fspath):
            item.add_marker(pytest.mark.gpu)

        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if "slow" in item.keywords or "performance" in str(item.fspath):
            item.add_marker(pytest.mark.slow)

        # Automatically skip tests requiring unavailable dependencies
        test_file = str(item.fspath)

        # PyTorch required tests
        if any(pattern in test_file for pattern in ["aco.py", "net.py", "train.py"]) or "torch" in item.name:
            item.add_marker(pytest.mark.torch)
            if not HAS_TORCH:
                item.add_marker(pytest.mark.skip(reason="PyTorch not available"))

        # PyTorch Geometric required tests
        if "torch_geometric" in item.name or any(pattern in test_file for pattern in ["geometric", "pyg"]):
            item.add_marker(pytest.mark.torch_geometric)
            if not HAS_TORCH_GEOMETRIC:
                item.add_marker(pytest.mark.skip(reason="PyTorch Geometric not available"))

        # Visualization required tests
        if any(pattern in test_file for pattern in ["visualization", "animation", "plot"]):
            item.add_marker(pytest.mark.visualization)
            if not (HAS_MATPLOTLIB and HAS_IMAGEIO):
                missing = []
                if not HAS_MATPLOTLIB:
                    missing.append("matplotlib")
                if not HAS_IMAGEIO:
                    missing.append("imageio")
                item.add_marker(pytest.mark.skip(reason=f"Visualization libraries not available: {', '.join(missing)}"))
