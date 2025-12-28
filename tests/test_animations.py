"""Tests for GFACS animations functionality."""

import pytest
import numpy as np
import torch
from pathlib import Path


class TestGFACSAnimator:
    """Test GFACSAnimator class."""

    def test_animator_initialization(self, real_animator):
        """Test animator initialization with default parameters."""
        assert real_animator.fps == 10
        assert real_animator.bitrate == 1800
        assert hasattr(real_animator, 'colors')
        assert len(real_animator.colors) == 10

    def test_animator_custom_parameters(self):
        """Test animator initialization with custom parameters."""
        from gfacs.utils.animations import GFACSAnimator
        animator = GFACSAnimator(fps=30, bitrate=2400)

        assert animator.fps == 30
        assert animator.bitrate == 2400

    def test_create_tsp_construction_animation(self, tmp_path, real_animator):
        """Test creating TSP tour construction animation."""
        # Create simple test data
        coordinates = torch.rand(5, 2)
        tour_history = [torch.randperm(5) for _ in range(3)]

        save_path = tmp_path / "tsp_animation.gif"
        animation = real_animator.create_tsp_construction_animation(
            coordinates, tour_history, save_path=save_path
        )

        # Verify animation object is created
        assert animation is not None

    def test_create_convergence_animation(self, tmp_path, real_animator):
        """Test creating convergence animation."""
        cost_history = [10.0, 8.5, 7.2, 6.1, 5.8]

        save_path = tmp_path / "convergence_animation.gif"
        animation = real_animator.create_convergence_animation(
            cost_history, save_path=save_path
        )

        assert animation is not None

    def test_create_pheromone_animation(self, tmp_path, real_animator):
        """Test creating pheromone evolution animation."""
        pheromone_history = [torch.rand(5, 5) for _ in range(3)]

        save_path = tmp_path / "pheromone_animation.gif"
        animation = real_animator.create_pheromone_evolution_animation(
            pheromone_history, save_path=save_path
        )

        assert animation is not None


class TestAnimationConvenienceFunctions:
    """Test convenience animation functions."""

    def test_create_tsp_tour_animation(self, tmp_path):
        """Test create_tsp_tour_animation convenience function."""
        from gfacs.utils.animations import create_tsp_tour_animation

        coordinates = torch.rand(5, 2)
        tour_history = [torch.randperm(5) for _ in range(2)]

        save_path = tmp_path / "tsp_tour.gif"
        animation = create_tsp_tour_animation(
            coordinates, tour_history, save_path=save_path
        )

        assert animation is not None

    def test_create_pheromone_animation_function(self, tmp_path):
        """Test create_pheromone_animation convenience function."""
        from gfacs.utils.animations import create_pheromone_animation

        pheromone_history = [torch.rand(5, 5) for _ in range(2)]

        save_path = tmp_path / "pheromone.gif"
        animation = create_pheromone_animation(
            pheromone_history, save_path=save_path
        )

        assert animation is not None

    def test_create_animation_factory(self, tmp_path):
        """Test create_animation factory function."""
        from gfacs.utils.animations import create_animation

        # Test invalid animation type
        with pytest.raises(ValueError, match="Unknown animation type"):
            create_animation("invalid_type")


class TestGetAnimator:
    """Test get_animator function."""

    def test_get_animator_singleton(self):
        """Test that get_animator returns singleton instance."""
        from gfacs.utils.animations import get_animator

        animator1 = get_animator()
        animator2 = get_animator()

        assert animator1 is animator2

    def test_get_animator_attributes(self):
        """Test get_animator returns properly configured instance."""
        from gfacs.utils.animations import get_animator

        animator = get_animator()

        assert hasattr(animator, 'fps')
        assert hasattr(animator, 'bitrate')
        assert hasattr(animator, 'colors')


class TestSaveExperimentAnimations:
    """Test save_experiment_animations function."""

    def test_save_experiment_animations_empty(self, tmp_path):
        """Test save_experiment_animations with no data."""
        from gfacs.utils.animations import save_experiment_animations

        experiment_dir = tmp_path / "experiment"
        save_experiment_animations(experiment_dir)

        # Should create animations directory even with no data
        assert (experiment_dir / "animations").exists()

    def test_save_experiment_animations_with_data(self, tmp_path):
        """Test save_experiment_animations with actual data."""
        from gfacs.utils.animations import save_experiment_animations

        experiment_dir = tmp_path / "experiment"
        coordinates = torch.rand(5, 2)
        tour_history = [torch.randperm(5) for _ in range(2)]
        cost_history = [10.0, 8.0]

        save_experiment_animations(
            experiment_dir,
            coordinates=coordinates,
            tour_history=tour_history,
            cost_history=cost_history
        )

        assert (experiment_dir / "animations").exists()


class TestAnimationErrorHandling:
    """Test error handling in animation functions."""

    def test_invalid_coordinates(self, tmp_path, real_animator):
        """Test animation creation with invalid coordinates."""
        # Empty coordinates
        coordinates = torch.empty(0, 2)
        tour_history = []

        animation = real_animator.create_tsp_construction_animation(
            coordinates, tour_history
        )
        # Should handle gracefully or raise appropriate error
        assert animation is not None

    def test_invalid_cost_history(self, tmp_path, real_animator):
        """Test convergence animation with invalid cost history."""
        # Empty cost history
        cost_history = []

        animation = real_animator.create_convergence_animation(cost_history)
        # Should handle gracefully
        assert animation is not None

    def test_invalid_pheromone_history(self, tmp_path, real_animator):
        """Test pheromone animation with invalid history."""
        # Empty pheromone history
        pheromone_history = []

        animation = real_animator.create_pheromone_evolution_animation(pheromone_history)
        # Should handle gracefully
        assert animation is not None


class TestAnimationWithNumpyArrays:
    """Test animations work with numpy arrays instead of torch tensors."""

    def test_numpy_coordinates(self, tmp_path, real_animator):
        """Test TSP animation with numpy coordinates."""
        coordinates = np.random.rand(5, 2)
        tour_history = [np.random.permutation(5) for _ in range(2)]

        animation = real_animator.create_tsp_construction_animation(
            coordinates, tour_history
        )

        assert animation is not None

    def test_numpy_pheromone(self, tmp_path, real_animator):
        """Test pheromone animation with numpy arrays."""
        pheromone_history = [np.random.rand(5, 5) for _ in range(2)]

        animation = real_animator.create_pheromone_evolution_animation(
            pheromone_history
        )

        assert animation is not None