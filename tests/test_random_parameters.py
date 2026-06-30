"""
Tests for random_parameters.py
"""

import numpy as np
import numpy.testing as npt
import pytest

from cross_immunity_model.random_parameters import (
    random_simplex_points,
    sample_random_parameters,
    sample_symmetric_beta_matrix,
)


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


# ──────────────────────────────────────────
# sample_symmetric_beta_matrix
# ──────────────────────────────────────────


class TestSampleSymmetricBetaMatrix:

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        A = sample_symmetric_beta_matrix(4, 2.0, 1.0, rng)
        assert A.shape == (4, 4)

    def test_returns_ndarray(self):
        rng = np.random.default_rng(0)
        assert isinstance(
            sample_symmetric_beta_matrix(3, 1.0, 1.0, rng), np.ndarray
        )

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(0)
        A = sample_symmetric_beta_matrix(5, 2.0, 1.0, rng)
        npt.assert_array_equal(np.diag(A), np.ones(5))

    def test_is_symmetric(self):
        rng = np.random.default_rng(0)
        A = sample_symmetric_beta_matrix(5, 2.0, 1.0, rng)
        npt.assert_array_equal(A, A.T)

    def test_off_diagonal_in_unit_interval(self):
        """Off-diagonal entries are sampled from Beta ∈ (0, 1)."""
        rng = np.random.default_rng(0)
        A = sample_symmetric_beta_matrix(10, 2.0, 1.0, rng)
        mask = ~np.eye(10, dtype=bool)
        assert np.all(A[mask] >= 0.0)
        assert np.all(A[mask] <= 1.0)

    def test_n_equals_one(self):
        """1×1 matrix must be [[1.0]]."""
        rng = np.random.default_rng(0)
        A = sample_symmetric_beta_matrix(1, 2.0, 1.0, rng)
        npt.assert_array_equal(A, np.array([[1.0]]))

    def test_n_equals_two(self):
        """2×2 matrix: diagonal 1, off-diagonal entries equal."""
        rng = np.random.default_rng(0)
        A = sample_symmetric_beta_matrix(2, 2.0, 1.0, rng)
        assert A[0, 0] == 1.0
        assert A[1, 1] == 1.0
        assert A[0, 1] == A[1, 0]

    def test_reproducible_with_same_rng(self):
        A1 = sample_symmetric_beta_matrix(
            4, 2.0, 1.0, np.random.default_rng(7)
        )
        A2 = sample_symmetric_beta_matrix(
            4, 2.0, 1.0, np.random.default_rng(7)
        )
        npt.assert_array_equal(A1, A2)

    def test_different_seed_gives_different_output(self):
        A1 = sample_symmetric_beta_matrix(
            4, 2.0, 1.0, np.random.default_rng(0)
        )
        A2 = sample_symmetric_beta_matrix(
            4, 2.0, 1.0, np.random.default_rng(1)
        )
        assert not np.allclose(A1, A2)

    def test_high_alpha_shifts_mean_toward_one(self):
        """Beta(alpha, beta) with large alpha/beta → mean near 1."""
        rng = np.random.default_rng(0)
        A = sample_symmetric_beta_matrix(20, 100.0, 1.0, rng)
        mask = ~np.eye(20, dtype=bool)
        assert A[mask].mean() > 0.9

    def test_low_alpha_shifts_mean_toward_zero(self):
        """Beta(alpha, beta) with small alpha/beta → mean near 0."""
        rng = np.random.default_rng(0)
        A = sample_symmetric_beta_matrix(20, 1.0, 100.0, rng)
        mask = ~np.eye(20, dtype=bool)
        assert A[mask].mean() < 0.1

    def test_number_of_unique_off_diagonal_entries(self):
        """Upper and lower triangle share values → n*(n-1)/2 unique pairs."""
        n = 5
        rng = np.random.default_rng(0)
        A = sample_symmetric_beta_matrix(n, 2.0, 1.0, rng)
        i_upper, j_upper = np.triu_indices(n, k=1)
        i_lower, j_lower = np.tril_indices(n, k=-1)
        npt.assert_array_equal(A[i_upper, j_upper], A[j_upper, i_upper])
        # Symmetry: lower triangle equals transposed upper triangle
        npt.assert_array_equal(A[i_lower, j_lower], A[j_lower, i_lower])


# ──────────────────────────────────────────
# sample_random_parameters
# ──────────────────────────────────────────


class TestSampleRandomParameters:

    # ── return types and shapes ───────────

    def test_returns_three_arrays(self):
        result = sample_random_parameters(3, seed=0)
        assert len(result) == 3

    def test_beta_shape(self):
        beta, _, _ = sample_random_parameters(4, seed=0)
        assert beta.shape == (4,)

    def test_gamma_shape(self):
        _, gamma, _ = sample_random_parameters(4, seed=0)
        assert gamma.shape == (4,)

    def test_sigma_shape(self):
        _, _, sigma = sample_random_parameters(4, seed=0)
        assert sigma.shape == (4, 4)

    def test_all_return_ndarray(self):
        for arr in sample_random_parameters(3, seed=0):
            assert isinstance(arr, np.ndarray)

    def test_n_variants_one(self):
        beta, gamma, sigma = sample_random_parameters(1, seed=0)
        assert beta.shape == (1,)
        assert gamma.shape == (1,)
        assert sigma.shape == (1, 1)

    # ── sigma structure ───────────────────

    def test_sigma_diagonal_is_one(self):
        _, _, sigma = sample_random_parameters(4, seed=0)
        npt.assert_array_equal(np.diag(sigma), np.ones(4))

    def test_sigma_is_symmetric(self):
        _, _, sigma = sample_random_parameters(4, seed=0)
        npt.assert_array_equal(sigma, sigma.T)

    def test_sigma_off_diagonal_in_unit_interval(self):
        _, _, sigma = sample_random_parameters(5, seed=0)
        mask = ~np.eye(5, dtype=bool)
        assert np.all(sigma[mask] >= 0.0)
        assert np.all(sigma[mask] <= 1.0)

    # ── reproducibility ───────────────────

    def test_reproducible_with_same_seed(self):
        b1, g1, s1 = sample_random_parameters(3, seed=42)
        b2, g2, s2 = sample_random_parameters(3, seed=42)
        npt.assert_array_equal(b1, b2)
        npt.assert_array_equal(g1, g2)
        npt.assert_array_equal(s1, s2)

    def test_different_seeds_give_different_output(self):
        b1, g1, s1 = sample_random_parameters(3, seed=0)
        b2, g2, s2 = sample_random_parameters(3, seed=1)
        assert not np.allclose(b1, b2)

    def test_rng_takes_precedence_over_seed(self):
        """Passing an rng object overrides the seed argument."""
        rng = np.random.default_rng(99)
        b1, g1, s1 = sample_random_parameters(3, rng=rng, seed=0)
        rng = np.random.default_rng(99)
        b2, g2, s2 = sample_random_parameters(3, rng=rng, seed=999)
        npt.assert_array_equal(b1, b2)

    def test_seed_none_produces_valid_output(self):
        """seed=None (default) must still return arrays of correct shape."""
        beta, gamma, sigma = sample_random_parameters(3)
        assert beta.shape == (3,)
        assert gamma.shape == (3,)
        assert sigma.shape == (3, 3)

    # ── distribution parameters ───────────

    def test_beta_mean_parameter(self):
        """With zero std, all beta values equal beta_mean."""
        beta, _, _ = sample_random_parameters(
            100, beta_mean=0.5, beta_std=0.0, seed=0
        )
        npt.assert_array_almost_equal(beta, 0.5 * np.ones(100))

    def test_gamma_mean_parameter(self):
        """With zero std, all gamma values equal gamma_mean."""
        _, gamma, _ = sample_random_parameters(
            100, gamma_mean=0.2, gamma_std=0.0, seed=0
        )
        npt.assert_array_almost_equal(gamma, 0.2 * np.ones(100))

    def test_sigma_alpha_beta_affect_off_diagonal(self):
        """High alpha/beta ratio pushes off-diagonal entries toward 1."""
        _, _, sigma_high = sample_random_parameters(
            10, sigma_alpha=100.0, sigma_beta=1.0, seed=0
        )
        _, _, sigma_low = sample_random_parameters(
            10, sigma_alpha=1.0, sigma_beta=100.0, seed=0
        )
        mask = ~np.eye(10, dtype=bool)
        assert sigma_high[mask].mean() > sigma_low[mask].mean()

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_shapes_for_various_n_variants(self, n):
        beta, gamma, sigma = sample_random_parameters(n, seed=0)
        assert beta.shape == (n,)
        assert gamma.shape == (n,)
        assert sigma.shape == (n, n)
