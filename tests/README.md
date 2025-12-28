# Tests Directory

This directory contains the comprehensive test suite for GFACS.

## Test Structure

- **Unit tests**: Individual component testing
- **Integration tests**: End-to-end pipeline testing
- **Performance tests**: Benchmarking and profiling
- **Validation tests**: Solution quality verification

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gfacs --cov-report=html

# Run specific module tests
pytest tests/test_tsp_nls/

# Run performance tests
pytest tests/test_performance.py
```

## Test Coverage

The test suite aims for >80% code coverage across all modules.

## Continuous Integration

Tests run automatically on every push and pull request.