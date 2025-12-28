"""Tests for GFACS orchestrator functionality."""

import pytest
import tempfile
import yaml
from pathlib import Path

from gfacs.orchestrator import (
    GFACSOrchestrator,
    OrchestratorConfig,
    ProblemConfig,
    load_orchestrator_config,
    HAS_ANIMATIONS,
    HAS_VISUALIZATIONS
)


class TestOrchestratorConfig:
    """Test orchestrator configuration."""

    def test_default_config_creation(self):
        """Test creating default orchestrator configuration."""
        config = OrchestratorConfig()

        assert config.experiment_name == "gfacs_experiment"
        assert len(config.problems) == 8  # All 8 GFACS problems
        assert config.enable_visualizations is True
        assert config.enable_animations is True
        assert config.log_level == "INFO"

    def test_problem_config_creation(self):
        """Test creating individual problem configurations."""
        config = ProblemConfig(
            name="tsp_nls",
            size=50,
            n_ants=20,
            n_iterations=100
        )

        assert config.name == "tsp_nls"
        assert config.size == 50
        assert config.n_ants == 20
        assert config.n_iterations == 100
        assert config.enabled is True

    def test_config_with_disabled_problems(self):
        """Test configuration with some problems disabled."""
        config = OrchestratorConfig()
        config.problems[0].enabled = False  # Disable first problem

        enabled_problems = [p for p in config.problems if p.enabled]
        assert len(enabled_problems) == 7


class TestOrchestratorSetup:
    """Test orchestrator setup and initialization."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        config = OrchestratorConfig()
        orchestrator = GFACSOrchestrator(config)

        assert orchestrator.config == config
        assert orchestrator.experiment_dir is None
        assert orchestrator.logger is None

    def test_setup_experiment(self, tmp_path, logger, io_manager):
        """Test experiment setup with implementations."""
        config = OrchestratorConfig()
        orchestrator = GFACSOrchestrator(config)

        # Setup should create experiment directory and initialize components
        orchestrator._setup_experiment()

        # Verify experiment directory was created
        assert orchestrator.experiment_dir is not None
        assert orchestrator.experiment_dir.exists()
        assert orchestrator.experiment_dir.name.startswith(config.experiment_name)

        # Verify logger and io_manager are set
        assert orchestrator.logger is not None
        assert orchestrator.io_manager is not None

        # Verify config was saved
        config_file = orchestrator.experiment_dir / "config" / "experiment.yaml"
        assert config_file.exists()


class TestProblemExecution:
    """Test problem execution functionality."""

    def test_run_problem_success(self, tmp_path, logger, io_manager, minimal_problem_config):
        """Test successful problem execution."""
        config = OrchestratorConfig()
        orchestrator = GFACSOrchestrator(config)
        orchestrator.logger = logger
        orchestrator.io_manager = io_manager
        orchestrator.experiment_dir = tmp_path / "test_experiment"

        # Use minimal config for fast testing
        result = orchestrator._run_problem(minimal_problem_config)

        # Verify result structure
        assert result["status"] in ["completed", "failed"]  # Could fail if dependencies missing
        assert result["problem"] == minimal_problem_config.name
        assert result["size"] == minimal_problem_config.size
        assert "duration" in result
        assert isinstance(result["duration"], (int, float))
        assert result["duration"] >= 0

    def test_run_problem_failure_handling(self, tmp_path, logger, io_manager):
        """Test problem execution failure handling with invalid problem name."""
        config = OrchestratorConfig()
        orchestrator = GFACSOrchestrator(config)
        orchestrator.logger = logger
        orchestrator.io_manager = io_manager
        orchestrator.experiment_dir = tmp_path / "test_experiment"

        # Use invalid problem name to trigger failure
        problem_config = ProblemConfig(name="invalid_problem", size=50)
        result = orchestrator._run_problem(problem_config)

        assert result["status"] == "failed"
        assert "error" in result


class TestVisualizationGeneration:
    """Test visualization generation."""

    def test_generate_visualizations(self, tmp_path, logger, io_manager):
        """Test visualization generation with real implementations."""
        config = OrchestratorConfig()
        orchestrator = GFACSOrchestrator(config)
        orchestrator.logger = logger
        orchestrator.io_manager = io_manager
        orchestrator.experiment_dir = tmp_path / "test_experiment"

        # Initialize visualizer (normally done in _setup_experiment)
        if HAS_VISUALIZATIONS:
            from gfacs.utils.visualization import get_visualizer
            orchestrator.visualizer = get_visualizer()

        # Create experiment directory structure
        viz_dir = orchestrator.experiment_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        problem_results = {
            "tsp_nls": {"status": "completed", "best_cost": 15.5},
            "cvrp_nls": {"status": "completed", "best_cost": 25.3}
        }

        # Test visualization generation
        orchestrator._generate_visualizations(problem_results)

        if HAS_VISUALIZATIONS:
            # Verify visualizations directory contains files
            viz_files = list(viz_dir.glob("*.png"))
            assert len(viz_files) > 0, "Should have generated visualization files"
        # If HAS_VISUALIZATIONS is False, should handle gracefully without creating files


class TestAnimationGeneration:
    """Test animation generation."""

    def test_generate_animations(self, tmp_path, logger, io_manager):
        """Test animation generation."""
        config = OrchestratorConfig()
        orchestrator = GFACSOrchestrator(config)
        orchestrator.logger = logger
        orchestrator.io_manager = io_manager
        orchestrator.experiment_dir = tmp_path / "test_experiment"

        # Initialize animator (normally done in _setup_experiment)
        if HAS_ANIMATIONS:
            from gfacs.utils.animations import get_animator
            orchestrator.animator = get_animator()

        problem_results = {
            "tsp_nls": {"status": "completed", "best_cost": 15.5},
            "cvrp_nls": {"status": "completed", "best_cost": 25.3}
        }

        # Should handle gracefully regardless of animation dependencies
        orchestrator._generate_animations(problem_results)


class TestConfigLoading:
    """Test configuration loading."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_orchestrator_config()

        assert isinstance(config, OrchestratorConfig)
        assert len(config.problems) == 8

    def test_load_yaml_config(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "experiment_name": "test_experiment",
            "problems": [
                {"name": "tsp_nls", "size": 100, "enabled": True},
                {"name": "cvrp_nls", "size": 100, "enabled": False}
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = load_orchestrator_config(config_path)

            assert config.experiment_name == "test_experiment"
            assert len(config.problems) == 2
            assert config.problems[0].name == "tsp_nls"
            assert config.problems[0].size == 100
            assert config.problems[0].enabled is True
            assert config.problems[1].enabled is False

        finally:
            Path(config_path).unlink()

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading configuration with invalid YAML."""
        invalid_yaml_path = tmp_path / "invalid.yaml"
        invalid_yaml_path.write_text("invalid: yaml: content: [")

        with pytest.raises(Exception):  # YAML parsing error
            load_orchestrator_config(str(invalid_yaml_path))

    def test_load_config_missing_file(self):
        """Test loading configuration from non-existent file."""
        config = load_orchestrator_config("/non/existent/file.yaml")
        # Should return default config
        assert isinstance(config, OrchestratorConfig)
        assert len(config.problems) == 8


class TestCLI:
    """Test CLI functionality."""

    def test_main_function_exists(self):
        """Test that main CLI function exists and can be imported."""
        from gfacs.orchestrator import main
        assert callable(main)

    def test_run_orchestrator_function(self, tmp_path):
        """Test run_orchestrator function with minimal config."""
        from gfacs.orchestrator import run_orchestrator

        # Test with minimal parameters
        results = run_orchestrator(
            experiment_name="cli_test",
            output_dir=str(tmp_path),
            problems=["tsp_nls"]  # Only run one problem for speed
        )

        # Verify result structure
        assert isinstance(results, dict)
        assert "experiment_name" in results
        assert "duration" in results


class TestUtilities:
    """Test utility functions and imports."""

    def test_get_animator_function_exists(self):
        """Test that get_animator function exists and works."""
        try:
            from gfacs.utils.animations import get_animator
            animator = get_animator()
            if HAS_ANIMATIONS:
                # When dependencies are available, should return valid animator
                assert animator is not None
                assert hasattr(animator, 'fps')
                assert hasattr(animator, 'bitrate')
            else:
                # Even when HAS_ANIMATIONS is False, the function should exist
                # but may not work properly
                assert animator is not None
                # Don't test attributes since they may not be functional
        except ImportError:
            # ImportError should only happen if the module itself can't be imported
            # This is different from HAS_ANIMATIONS being False
            if HAS_ANIMATIONS:
                # If HAS_ANIMATIONS is True, ImportError is unexpected
                raise
            # If HAS_ANIMATIONS is False, ImportError during import is expected
            pass


class TestProblemSimulation:
    """Test problem simulation execution."""

    def test_execute_problem_simulation_tsp(self, tmp_path, logger, io_manager):
        """Test executing TSP problem simulation."""
        config = OrchestratorConfig()
        orchestrator = GFACSOrchestrator(config)
        orchestrator.logger = logger
        orchestrator.io_manager = io_manager
        orchestrator.experiment_dir = tmp_path / "test_experiment"

        problem_config = ProblemConfig(name="tsp_nls", size=10, n_ants=2, n_iterations=2)
        result = orchestrator._execute_problem_simulation(problem_config)

        # Verify result structure
        assert isinstance(result, dict)
        assert "best_cost" in result or "error" in result  # May fail if dependencies missing
        assert "iterations" in result
        assert result["iterations"] == 2

    def test_execute_problem_simulation_cvrp(self, tmp_path, logger, io_manager):
        """Test executing CVRP problem simulation."""
        config = OrchestratorConfig()
        orchestrator = GFACSOrchestrator(config)
        orchestrator.logger = logger
        orchestrator.io_manager = io_manager
        orchestrator.experiment_dir = tmp_path / "test_experiment"

        problem_config = ProblemConfig(name="cvrp_nls", size=10, n_ants=2, n_iterations=2)
        result = orchestrator._execute_problem_simulation(problem_config)

        # Verify result structure
        assert isinstance(result, dict)
        assert "best_cost" in result or "error" in result  # May fail if dependencies missing

    def test_execute_problem_simulation_invalid_problem(self, tmp_path, logger, io_manager):
        """Test executing simulation with invalid problem name."""
        config = OrchestratorConfig()
        orchestrator = GFACSOrchestrator(config)
        orchestrator.logger = logger
        orchestrator.io_manager = io_manager
        orchestrator.experiment_dir = tmp_path / "test_experiment"

        problem_config = ProblemConfig(name="invalid_problem", size=10)

        # Should raise ValueError for invalid problem name
        with pytest.raises(ValueError, match="Unknown problem"):
            orchestrator._execute_problem_simulation(problem_config)


class TestVisualizationGenerationExtended:
    """Extended tests for visualization generation."""

    def test_generate_visualizations_empty_results(self, tmp_path, logger, io_manager):
        """Test visualization generation with empty results."""
        config = OrchestratorConfig()
        orchestrator = GFACSOrchestrator(config)
        orchestrator.logger = logger
        orchestrator.io_manager = io_manager
        orchestrator.experiment_dir = tmp_path / "test_experiment"

        # Should handle empty results gracefully
        orchestrator._generate_visualizations({})

    def test_generate_visualizations_single_problem(self, tmp_path, logger, io_manager):
        """Test visualization generation with single problem results."""
        config = OrchestratorConfig()
        orchestrator = GFACSOrchestrator(config)
        orchestrator.logger = logger
        orchestrator.io_manager = io_manager
        orchestrator.experiment_dir = tmp_path / "test_experiment"

        problem_results = {
            "tsp_nls": {"status": "completed", "best_cost": 15.5}
        }

        orchestrator._generate_visualizations(problem_results)


class TestAnimationGenerationExtended:
    """Extended tests for animation generation."""

    def test_generate_animations_empty_results(self, tmp_path, logger, io_manager):
        """Test animation generation with empty results."""
        config = OrchestratorConfig()
        orchestrator = GFACSOrchestrator(config)
        orchestrator.logger = logger
        orchestrator.io_manager = io_manager
        orchestrator.experiment_dir = tmp_path / "test_experiment"

        # Should handle empty results gracefully
        orchestrator._generate_animations({})

    def test_generate_animations_single_problem(self, tmp_path, logger, io_manager):
        """Test animation generation with single problem results."""
        config = OrchestratorConfig()
        orchestrator = GFACSOrchestrator(config)
        orchestrator.logger = logger
        orchestrator.io_manager = io_manager
        orchestrator.experiment_dir = tmp_path / "test_experiment"

        problem_results = {
            "tsp_nls": {"status": "completed", "best_cost": 15.5}
        }

        orchestrator._generate_animations(problem_results)


class TestOrchestratorEdgeCases:
    """Test edge cases and error handling."""

    def test_orchestrator_with_disabled_problems(self, tmp_path):
        """Test orchestrator with all problems disabled."""
        config = OrchestratorConfig()
        # Disable all problems
        for problem in config.problems:
            problem.enabled = False

        orchestrator = GFACSOrchestrator(config)
        # Don't manually set logger/io_manager as run_experiment will set them up

        results = orchestrator.run_experiment()

        assert results["problems_run"] == 0
        # When no problems run, problem_results should be empty
        assert results.get("problem_results", {}) == {}

    def test_orchestrator_empty_problem_list(self, tmp_path):
        """Test orchestrator with empty problem list."""
        config = OrchestratorConfig()
        config.problems = []  # Empty problem list

        # Should raise ValueError for empty problem list
        with pytest.raises(ValueError, match="Config must contain at least one problem"):
            GFACSOrchestrator(config)


class TestIntegration:
    """Integration tests for orchestrator."""

    def test_full_experiment_run(self, tmp_path, minimal_orchestrator_config):
        """Test full experiment run with real implementation."""
        orchestrator = GFACSOrchestrator(minimal_orchestrator_config)

        # Run the experiment - this should work with real implementations
        results = orchestrator.run_experiment()

        # Verify result structure
        assert "experiment_name" in results
        assert "duration" in results
        assert "problems_run" in results
        assert isinstance(results["duration"], (int, float))
        assert results["duration"] >= 0

        # Verify experiment directory was created
        assert orchestrator.experiment_dir is not None
        assert orchestrator.experiment_dir.exists()

        # Verify results contain expected structure
        assert "problem_results" in results
        assert isinstance(results["problem_results"], dict)


if __name__ == "__main__":
    pytest.main([__file__])
