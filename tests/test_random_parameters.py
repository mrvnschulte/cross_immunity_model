"""
Tests for random_parameters.py
"""

import numpy as np
import numpy.testing as npt
import pytest

from cross_immunity_model.random_parameters import random_simplex_points


# ──────────────────────────────────────────
# random_simplex_points
# ──────────────────────────────────────────


class TestRandomSimplexPoints:

    def test_output_shape(self):
        result = random_simplex_points(10, 5)
        assert result.shape == (10, 5)

    def test_returns_ndarray(self):
        assert isinstance(random_simplex_points(5, 3), np.ndarray)

    def test_rows_sum_to_one(self):
        result = random_simplex_points(50, 4, rng=np.random.default_rng(0))
        npt.assert_array_almost_equal(result.sum(axis=1), np.ones(50))

    def test_all_entries_non_negative(self):
        result = random_simplex_points(50, 4, rng=np.random.default_rng(0))
        assert np.all(result >= 0.0)

    def test_all_entries_at_most_one(self):
        result = random_simplex_points(50, 4, rng=np.random.default_rng(0))
        assert np.all(result <= 1.0)

    def test_single_point(self):
        result = random_simplex_points(1, 3, rng=np.random.default_rng(0))
        assert result.shape == (1, 3)
        npt.assert_almost_equal(result.sum(), 1.0)

    def test_dim_one_gives_all_ones(self):
        """Simplex of dimension 1 has only one vertex: x=[1]."""
        result = random_simplex_points(10, 1, rng=np.random.default_rng(0))
        npt.assert_array_almost_equal(result, np.ones((10, 1)))

    def test_rng_none_produces_valid_output(self):
        """When rng=None, function must initialise its own generator."""
        result = random_simplex_points(5, 3, rng=None)
        assert result.shape == (5, 3)
        npt.assert_array_almost_equal(result.sum(axis=1), np.ones(5))

    def test_reproducible_with_same_rng(self):
        r1 = random_simplex_points(10, 4, rng=np.random.default_rng(42))
        r2 = random_simplex_points(10, 4, rng=np.random.default_rng(42))
        npt.assert_array_equal(r1, r2)

    def test_different_rng_seed_gives_different_output(self):
        r1 = random_simplex_points(10, 4, rng=np.random.default_rng(0))
        r2 = random_simplex_points(10, 4, rng=np.random.default_rng(1))
        assert not np.allclose(r1, r2)
