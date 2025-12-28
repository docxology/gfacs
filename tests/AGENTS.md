# Module: Tests (`tests/`)

## Overview

The Tests module provides comprehensive testing infrastructure for GFACS. It includes unit tests, integration tests, performance benchmarks, and validation suites to ensure code quality, correctness, and performance across all GFACS components.

## Test Structure

```
tests/
├── __init__.py                     # Test package initialization
├── conftest.py                     # pytest configuration and fixtures
├── test_*.py                       # Main test files
├── test_bpp/                       # Bin Packing Problem tests
├── test_cvrp_nls/                  # CVRP with Neural Local Search tests
├── test_cvrptw_nls/                # CVRPTW with PyVRP tests
├── test_op/                        # Orienteering Problem tests
├── test_pctsp/                     # Prize Collecting TSP tests
├── test_smtwtp/                    # SMTWTP tests
├── test_sop/                       # Sequential Ordering Problem tests
├── test_tsp_nls/                   # TSP with Neural Local Search tests
│   ├── __pycache__/               # Python bytecode
│   ├── test_aco.py                 # ACO algorithm tests
│   ├── test_net.py                 # Neural network tests
│   └── test_utils.py               # Utility function tests
```

## Test Categories

### Unit Tests
**Purpose:** Test individual functions and classes in isolation
**Scope:** Single functions, methods, and small components
**Location:** `test_*.py` files and module-specific directories

**Coverage Areas:**
- Algorithm correctness (ACO, neural networks)
- Data processing utilities
- Configuration validation
- Error handling

### Integration Tests
**Purpose:** Test component interactions and end-to-end workflows
**Scope:** Multi-component interactions, data pipelines
**Location:** Module-specific test directories

**Coverage Areas:**
- Complete problem solving pipelines
- Model training and inference
- Data loading and preprocessing
- External solver integration

### Performance Tests
**Purpose:** Validate performance characteristics and scalability
**Scope:** Execution time, memory usage, convergence
**Location:** `test_performance.py`

**Coverage Areas:**
- Algorithm convergence speed
- Memory consumption patterns
- Scalability with problem size
- GPU utilization efficiency

### Validation Tests
**Purpose:** Ensure solution quality and correctness
**Scope:** Solution feasibility, optimality bounds
**Location:** `test_validation.py` and problem-specific tests

**Coverage Areas:**
- Solution constraint satisfaction
- Objective function accuracy
- Benchmark performance validation
- Statistical significance testing

## Test Fixtures

### conftest.py
Global pytest configuration and shared fixtures.

**Key Fixtures:**
```python
@pytest.fixture
def sample_tsp_instance():
    """Generate sample TSP instance for testing."""
    coordinates = torch.rand(10, 2)
    return coordinates

@pytest.fixture
def sample_cvrp_instance():
    """Generate sample CVRP instance for testing."""
    coordinates = torch.rand(10, 2)
    demands = torch.rand(10) * 0.5 + 0.1
    capacity = 1.0
    return coordinates, demands, capacity

@pytest.fixture
def mock_model():
    """Create mock neural network for testing."""
    model = Net(gfn=False)
    return model
```

### Problem-Specific Fixtures
Each problem module provides specialized test fixtures:
- **TSP**: Coordinate generation, distance matrices
- **CVRP**: Demand vectors, capacity constraints
- **BPP**: Item sizes, bin configurations
- **OP**: Prize values, length constraints

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_orchestrator.py

# Run tests for specific module
pytest tests/test_tsp_nls/

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=gfacs --cov-report=html
```

### Test Selection
```bash
# Run tests by pattern
pytest -k "test_aco"

# Run tests by module
pytest tests/test_bpp/

# Run slow tests only
pytest -m slow

# Skip integration tests
pytest -m "not integration"
```

### Test Configuration
```bash
# Parallel execution
pytest -n auto

# Stop on first failure
pytest -x

# Detailed failure information
pytest --tb=long

# Generate JUnit XML report
pytest --junitxml=reports/test_results.xml
```

## Test Organization

### Test Naming Convention
- **Unit Tests**: `test_function_name()` or `test_method_name()`
- **Integration Tests**: `test_integration_feature()`
- **Performance Tests**: `test_performance_scenario()`
- **Validation Tests**: `test_validation_criteria()`

### Test Structure Pattern
```python
class TestComponentName:
    """Test suite for ComponentName."""

    def test_initialization(self):
        """Test component initialization."""
        # Arrange
        # Act
        # Assert

    def test_core_functionality(self):
        """Test core component functionality."""
        # Test implementation

    @pytest.mark.parametrize("param", [value1, value2])
    def test_parameterized_behavior(self, param):
        """Test behavior with different parameters."""
        # Parameterized test
```

## Test Coverage

### Coverage Goals
- **Unit Tests**: >90% coverage for core algorithms
- **Integration Tests**: Complete pipeline coverage
- **Performance Tests**: Key performance scenarios
- **Validation Tests**: All constraint types and objectives

### Coverage Measurement
```bash
# Generate coverage report
pytest --cov=gfacs --cov-report=html

# Check coverage by module
pytest --cov=gfacs --cov-report=term-missing

# Coverage thresholds
pytest --cov=gfacs --cov-fail-under=85
```

## Continuous Integration

### CI Pipeline
- **Automated Testing**: Run on every push and PR
- **Coverage Reporting**: Upload coverage reports
- **Quality Gates**: Block merges below coverage thresholds
- **Performance Regression**: Detect performance degradation

### CI Configuration
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov=gfacs --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Test Data Management

### Test Datasets
- **Synthetic Data**: Generated for unit tests
- **Benchmark Data**: Subset of full benchmarks
- **Edge Cases**: Special instances for validation
- **Regression Data**: Known failure cases

### Data Generation
```python
def generate_test_tsp_instance(n_nodes=10):
    """Generate small TSP instance for testing."""
    coordinates = torch.rand(n_nodes, 2)
    return coordinates

def generate_edge_case_instance():
    """Generate instance that tests edge cases."""
    # Specific configuration for edge case testing
    pass
```

## Mocking and Stubs

### External Dependencies
- **Solvers**: Mock Concorde, HGS-CVRP for fast testing
- **File I/O**: Use temporary files and in-memory data
- **Network Calls**: Mock external API dependencies
- **GPU Operations**: CPU fallbacks for CI environments

### Mock Implementation
```python
@pytest.fixture
def mock_concorde_solver():
    """Mock Concorde TSP solver for testing."""
    with patch('gfacs.tsp_nls.concorde.solve_tsp') as mock:
        mock.return_value = optimal_tour
        yield mock
```

## Performance Testing

### Benchmark Tests
```python
def test_tsp_convergence_speed(benchmark):
    """Benchmark TSP solving performance."""
    instance = generate_test_tsp_instance(50)

    def solve_instance():
        # Solving logic
        pass

    benchmark(solve_instance)
```

### Memory Profiling
```python
def test_memory_usage():
    """Test memory consumption patterns."""
    from memory_profiler import memory_usage

    def solve_large_instance():
        # Large instance solving
        pass

    mem_usage = memory_usage(solve_large_instance)
    assert max(mem_usage) < 1000  # MB
```

## Test Maintenance

### Test Updates
- **Code Changes**: Update tests when implementation changes
- **New Features**: Add tests for new functionality
- **Bug Fixes**: Add regression tests for fixed bugs
- **Refactoring**: Update test structure as needed

### Test Quality
- **Flaky Tests**: Identify and fix non-deterministic tests
- **Slow Tests**: Optimize or mark slow tests appropriately
- **Redundant Tests**: Remove duplicate test coverage
- **Documentation**: Keep test documentation current

## Debugging Tests

### Common Issues
- **Import Errors**: Check Python path and dependencies
- **Fixture Problems**: Verify fixture dependencies
- **GPU Tests**: Handle CUDA availability gracefully
- **Randomness**: Use seeded random number generators

### Debugging Tools
```bash
# Debug specific test
pytest tests/test_specific.py::TestClass::test_method -s

# PDB debugging
pytest --pdb

# Print debugging
pytest -s --capture=no
```

## Contributing Tests

### Adding New Tests
1. **Identify Coverage Gap**: Determine what needs testing
2. **Create Test File**: Follow naming conventions
3. **Implement Tests**: Use appropriate fixtures and patterns
4. **Run Tests**: Verify tests pass and provide coverage
5. **Document Tests**: Add docstrings and comments

### Test Guidelines
- **Isolation**: Tests should not depend on external state
- **Repeatability**: Tests should produce consistent results
- **Speed**: Keep unit tests fast, mark slow tests
- **Clarity**: Use descriptive names and documentation

## Integration with Development

### TDD Workflow
- **Write Tests First**: Define expected behavior
- **Implement Code**: Make tests pass
- **Refactor**: Improve code while maintaining tests
- **Continuous Testing**: Run tests frequently during development

### Pre-commit Hooks
```bash
# Run tests before commit
pre-commit install
# Configure to run pytest in pre-commit-config.yaml
```

### IDE Integration
- **Test Runners**: Configure pytest in IDE
- **Coverage Display**: Show coverage in editor
- **Debugging**: Set breakpoints in tests
- **Auto-run**: Run tests on file save