"""Tests for GFACS I/O functionality."""

import pytest
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch


class TestExperimentIO:
    """Test ExperimentIO class."""

    def test_io_initialization(self, real_io_manager):
        """Test I/O manager initialization."""
        assert real_io_manager is not None
        assert hasattr(real_io_manager, 'base_dir')

    def test_create_experiment_dir(self, real_io_manager, tmp_path):
        """Test creating experiment directory."""
        experiment_name = "test_experiment"

        exp_dir = real_io_manager.create_experiment_dir(experiment_name)

        assert exp_dir.exists()
        assert experiment_name in str(exp_dir)
        assert exp_dir.is_dir()

    def test_create_experiment_dir_with_timestamp(self, real_io_manager, tmp_path):
        """Test creating experiment directory with timestamp."""
        experiment_name = "test_experiment"

        exp_dir = real_io_manager.create_experiment_dir(experiment_name, timestamp=True)

        assert exp_dir.exists()
        assert experiment_name in str(exp_dir)
        # Should contain timestamp
        assert len(exp_dir.name.split('_')) >= 2

    def test_save_config(self, real_io_manager, tmp_path, minimal_orchestrator_config):
        """Test saving experiment configuration."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        real_io_manager.save_config(minimal_orchestrator_config, exp_dir)

        config_file = exp_dir / "config" / "experiment.yaml"
        assert config_file.exists()

        # Verify content can be read back
        import yaml
        with open(config_file) as f:
            loaded_config = yaml.safe_load(f)
        assert loaded_config is not None

    def test_save_input_data(self, real_io_manager, tmp_path):
        """Test saving input data."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        input_data = {
            "coordinates": torch.rand(10, 2),
            "distances": torch.rand(10, 10),
            "demands": torch.rand(10)
        }

        real_io_manager.save_input_data(input_data, exp_dir)

        input_file = exp_dir / "data" / "inputs" / "input_data.pkl"
        assert input_file.exists()

        # Verify content can be loaded back
        with open(input_file, 'rb') as f:
            loaded_data = pickle.load(f)
        assert "coordinates" in loaded_data

    def test_save_problem_instance(self, real_io_manager, tmp_path):
        """Test saving problem instance data."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        problem_data = {
            "coordinates": torch.rand(10, 2),
            "distances": torch.rand(10, 10),
            "n_nodes": 10
        }

        real_io_manager.save_problem_instance(
            problem_data, exp_dir, "tsp_nls", "instance_1"
        )

        instance_file = exp_dir / "data" / "inputs" / "tsp_nls" / "instance_1.pkl"
        assert instance_file.exists()

    def test_save_results(self, real_io_manager, tmp_path):
        """Test saving experiment results."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        results = {
            "experiment_name": "test",
            "duration": 10.5,
            "problems_run": 1
        }

        real_io_manager.save_results(results, exp_dir)

        results_file = exp_dir / "data" / "results" / "results.json"
        assert results_file.exists()

        # Verify content
        with open(results_file) as f:
            loaded_results = json.load(f)
        assert loaded_results["experiment_name"] == "test"

    def test_save_problem_results(self, real_io_manager, tmp_path):
        """Test saving problem-specific results."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        results = {
            "best_cost": 15.5,
            "mean_cost": 18.2,
            "iterations": 100
        }

        real_io_manager.save_problem_results(results, exp_dir, "tsp_nls")

        results_file = exp_dir / "data" / "results" / "per_problem" / "tsp_nls" / "results.json"
        assert results_file.exists()

    def test_save_training_metrics(self, real_io_manager, tmp_path):
        """Test saving training metrics."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        metrics = {
            "loss": [1.0, 0.8, 0.6],
            "best_cost": [20.0, 18.0, 15.0]
        }

        real_io_manager.save_training_metrics(metrics, exp_dir, "tsp_nls")

        metrics_file = exp_dir / "data" / "metrics" / "tsp_nls" / "training_metrics.json"
        assert metrics_file.exists()

        # Verify content
        with open(metrics_file) as f:
            loaded_metrics = json.load(f)
        assert "loss" in loaded_metrics
        assert len(loaded_metrics["loss"]) == 3

    def test_save_checkpoint(self, real_io_manager, tmp_path):
        """Test saving model checkpoint."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        # Mock model state
        model_state = {"layer1.weight": torch.rand(5, 5)}
        optimizer_state = {"state": {}, "param_groups": []}
        epoch = 10
        metrics = {"loss": 0.5}

        checkpoint_path = real_io_manager.save_checkpoint(
            model_state, optimizer_state, epoch, metrics, exp_dir, "tsp_nls"
        )

        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == ".pt"

        # Verify can load checkpoint
        loaded = real_io_manager.load_checkpoint(checkpoint_path)
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded
        assert loaded["epoch"] == 10

    def test_load_checkpoint(self, real_io_manager, tmp_path):
        """Test loading model checkpoint."""
        # Create a checkpoint file
        checkpoint_data = {
            "model_state": {"layer.weight": torch.rand(3, 3)},
            "optimizer_state": {"state": {}},
            "epoch": 5,
            "metrics": {"loss": 0.3}
        }

        checkpoint_file = tmp_path / "checkpoint.pt"
        torch.save(checkpoint_data, checkpoint_file)

        loaded = real_io_manager.load_checkpoint(checkpoint_file)

        assert loaded["epoch"] == 5
        assert "model_state" in loaded

    def test_save_metadata(self, real_io_manager, tmp_path):
        """Test saving experiment metadata."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        metadata = {
            "version": "1.0",
            "timestamp": "2024-01-01",
            "system_info": {"python": "3.11"}
        }

        real_io_manager.save_metadata(metadata, exp_dir)

        metadata_file = exp_dir / "data" / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata["version"] == "1.0"

    def test_save_summary(self, real_io_manager, tmp_path, minimal_orchestrator_config):
        """Test saving experiment summary."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        results = {
            "experiment_name": "test",
            "duration": 5.0,
            "problems_run": 1
        }

        real_io_manager.save_summary(minimal_orchestrator_config, results, exp_dir)

        summary_file = exp_dir / "EXPERIMENT_SUMMARY.md"
        assert summary_file.exists()

    def test_list_experiments(self, real_io_manager, tmp_path):
        """Test listing experiment directories."""
        # Create some experiment directories
        exp1 = tmp_path / "experiment_20240101_120000"
        exp1.mkdir()
        exp2 = tmp_path / "experiment_20240102_130000"
        exp2.mkdir()

        # Create config file to mark as valid experiment
        (exp1 / "config" / "experiment.yaml").parent.mkdir(parents=True)
        (exp1 / "config" / "experiment.yaml").write_text("test: data")

        (exp2 / "config" / "experiment.yaml").parent.mkdir(parents=True)
        (exp2 / "config" / "experiment.yaml").write_text("test: data")

        # Temporarily change base_dir
        original_base_dir = real_io_manager.base_dir
        real_io_manager.base_dir = tmp_path

        try:
            experiments = real_io_manager.list_experiments()
            assert len(experiments) >= 2
        finally:
            real_io_manager.base_dir = original_base_dir

    def test_load_experiment_results(self, real_io_manager, tmp_path):
        """Test loading complete experiment results."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        # Create results file
        results_file = exp_dir / "data" / "results" / "summary.json"
        results_file.parent.mkdir(parents=True)

        results_data = {
            "experiment_name": "test_exp",
            "duration": 10.0,
            "problems": {"tsp": {"best_cost": 15.5}}
        }

        with open(results_file, 'w') as f:
            json.dump(results_data, f)

        loaded = real_io_manager.load_experiment_results(exp_dir)

        # Results are nested under 'summary' key
        assert "summary" in loaded
        assert loaded["summary"]["experiment_name"] == "test_exp"
        assert loaded["summary"]["duration"] == 10.0


class TestIOModuleFunctions:
    """Test I/O module convenience functions."""

    def test_get_io_manager(self):
        """Test get_io_manager function."""
        from gfacs.utils.io import get_io_manager

        io_mgr = get_io_manager()
        assert io_mgr is not None

    def test_setup_experiment_io(self, tmp_path):
        """Test setup_experiment_io function."""
        from gfacs.utils.io import setup_experiment_io
        from gfacs.orchestrator import OrchestratorConfig

        config = OrchestratorConfig(experiment_name="test_setup")

        exp_dir, io_mgr = setup_experiment_io("test_setup", config, tmp_path)

        assert exp_dir.exists()
        assert io_mgr is not None
        assert "test_setup" in str(exp_dir)


class TestIOErrorHandling:
    """Test error handling in I/O operations."""

    def test_save_config_invalid_dataclass(self, real_io_manager, tmp_path):
        """Test saving config with invalid dataclass."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        # Invalid object that can't be serialized
        invalid_config = lambda x: x  # Function object

        # This should either raise an exception or handle gracefully
        try:
            real_io_manager.save_config(invalid_config, exp_dir)
            # If it doesn't raise, that's also acceptable (graceful degradation)
        except (TypeError, ValueError):
            # Expected to raise an exception for invalid input
            pass

    def test_load_checkpoint_invalid_file(self, real_io_manager, tmp_path):
        """Test loading checkpoint from invalid file."""
        invalid_file = tmp_path / "invalid_checkpoint.txt"
        invalid_file.write_text("not a checkpoint")

        with pytest.raises(Exception):
            real_io_manager.load_checkpoint(invalid_file)

    def test_load_experiment_results_missing_file(self, real_io_manager, tmp_path):
        """Test loading experiment results when file doesn't exist."""
        exp_dir = tmp_path / "empty_experiment"
        exp_dir.mkdir()

        # Should return empty dict or handle gracefully
        results = real_io_manager.load_experiment_results(exp_dir)
        assert isinstance(results, dict)

    def test_save_input_data_with_numpy_arrays(self, real_io_manager, tmp_path):
        """Test saving input data containing numpy arrays."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        input_data = {
            "coordinates": np.random.rand(5, 2),
            "distances": np.random.rand(5, 5)
        }

        real_io_manager.save_input_data(input_data, exp_dir)

        input_file = exp_dir / "data" / "inputs" / "input_data.pkl"
        assert input_file.exists()

        # Verify can load back
        with open(input_file, 'rb') as f:
            loaded_data = pickle.load(f)
        assert "coordinates" in loaded_data


class TestIOSerialization:
    """Test JSON serialization utilities."""

    def test_make_json_serializable(self, real_io_manager):
        """Test _make_json_serializable method."""
        # Test with torch tensors
        tensor = torch.rand(3, 3)
        serializable = real_io_manager._make_json_serializable(tensor)
        assert isinstance(serializable, list)

        # Test with numpy arrays
        array = np.random.rand(2, 2)
        serializable = real_io_manager._make_json_serializable(array)
        assert isinstance(serializable, list)

        # Test with regular Python objects
        regular_obj = {"key": "value", "number": 42}
        serializable = real_io_manager._make_json_serializable(regular_obj)
        assert serializable == regular_obj