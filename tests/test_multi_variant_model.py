"""
Tests for multi_variant_model.py

Note: The two formulations (matrix and for-loop) are only equivalent when
      sigma[k, k] = 1 for all k (prior infection with variant k confers full
      protection against itself). All fixtures enforce this convention.
"""

import numpy as np
import numpy.testing as npt
import pytest

from cross_immunity_model.multi_variant_model import (
    multi_variant_model,
    multi_variant_model_for_loops,
    multi_variant_model_with_matrices,
    multi_variant_model_sparse,
)
from cross_immunity_model.matrices import (
    construct_matrices,
    construct_sparse_operators,
)


# ──────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────


@pytest.fixture
def params_n1():
    """n=1 parameters. sigma[0,0]=1 is required for both formulations to agree."""
    return (
        np.array([0.3]),  # beta
        np.array([0.1]),  # gamma
        np.array([[1.0]]),  # sigma (diagonal = 1)
        0.0,  # b
    )


@pytest.fixture
def params_n2():
    """n=2 parameters with sigma diagonal = 1."""
    return (
        np.array([0.3, 0.4]),
        np.array([0.1, 0.12]),
        np.array([[1.0, 0.5], [0.5, 1.0]]),
        0.0,
    )


@pytest.fixture
def state_n1_mixed():
    """
    n=1, sums to 1:
      x[0] = I[0,0] = 0.7  (susceptible, unprotected)
      x[1] = I[0,1] = 0.2  (protected against v1)
      x[2] = I[1,1] = 0.1  (infectious v1)
    """
    return np.array([0.7, 0.2, 0.1])


@pytest.fixture
def state_n1_disease_free():
    """n=1: entire population susceptible."""
    return np.array([1.0, 0.0, 0.0])


@pytest.fixture
def state_n2_mixed():
    """
    n=2, sums to 1:
      x[0] = I[0,0] = 0.6   (susceptible, unprotected)
      x[1] = I[0,1] = 0.05  (protected v1)
      x[2] = I[1,1] = 0.05  (infectious v1)
      x[3] = I[0,2] = 0.05  (protected v2)
      x[4] = I[2,2] = 0.05  (infectious v2)
      x[5] = I[0,3] = 0.05  (protected both)
      x[6] = I[1,3] = 0.05  (infectious v1, protected both)
      x[7] = I[2,3] = 0.05  (infectious v2, protected both)
      x[8] = I[3,3] = 0.05  (infectious both, protected both)
    """
    x = np.zeros(9)
    x[0] = 0.6
    x[1:] = 0.05
    return x


@pytest.fixture
def state_n2_disease_free():
    """n=2: entire population susceptible."""
    x = np.zeros(9)
    x[0] = 1.0
    return x


@pytest.fixture
def matrices_n1(params_n1):
    _, gamma, sigma, _ = params_n1
    return construct_matrices(gamma, sigma)


@pytest.fixture
def matrices_n2(params_n2):
    _, gamma, sigma, _ = params_n2
    return construct_matrices(gamma, sigma)


@pytest.fixture
def sparse_operators_n1(params_n1):
    _, gamma, sigma, _ = params_n1
    return construct_sparse_operators(gamma, sigma)


@pytest.fixture
def sparse_operators_n2(params_n2):
    _, gamma, sigma, _ = params_n2
    return construct_sparse_operators(gamma, sigma)


# ──────────────────────────────────────────
# multi_variant_model
# ──────────────────────────────────────────


class TestMultiVariantModel:

    @pytest.mark.parametrize("n, x_len", [(1, 3), (2, 9)])
    def test_output_shape(self, n, x_len, params_n1, params_n2):
        beta, gamma, sigma, b = params_n1 if n == 1 else params_n2
        x = np.zeros(x_len)
        x[0] = 1.0
        assert multi_variant_model(0.0, x, beta, gamma, sigma, b).shape == (
            x_len,
        )

    def test_returns_ndarray(self, state_n1_mixed, params_n1):
        beta, gamma, sigma, b = params_n1
        assert isinstance(
            multi_variant_model(0.0, state_n1_mixed, beta, gamma, sigma, b),
            np.ndarray,
        )

    def test_t_is_unused(self, state_n1_mixed, params_n1):
        beta, gamma, sigma, b = params_n1
        dx0 = multi_variant_model(0.0, state_n1_mixed, beta, gamma, sigma, b)
        dx99 = multi_variant_model(99.0, state_n1_mixed, beta, gamma, sigma, b)
        npt.assert_array_equal(dx0, dx99)

    def test_known_values_n1(self, state_n1_mixed, params_n1):
        """
        For beta=[0.3], gamma=[0.1], sigma=[[1]], b=0, x=[0.7, 0.2, 0.1]:
          dI[0,0] = -beta*S*I_v                      = -0.021
          dI[0,1] = +gamma*I[1,1]                    = +0.01
          dI[1,1] = -gamma*I[1,1] + beta*I[0,0]*I_v  = -0.01 + 0.021 = +0.011
        """
        beta, gamma, sigma, b = params_n1
        dx = multi_variant_model(0.0, state_n1_mixed, beta, gamma, sigma, b)
        npt.assert_array_almost_equal(dx, [-0.021, 0.01, 0.011])

    @pytest.mark.parametrize("b", [0.0, 0.01])
    def test_mass_conservation_n1(self, state_n1_mixed, params_n1, b):
        beta, gamma, sigma, _ = params_n1
        dx = multi_variant_model(0.0, state_n1_mixed, beta, gamma, sigma, b)
        npt.assert_almost_equal(dx.sum(), 0.0, decimal=12)

    @pytest.mark.parametrize("b", [0.0, 0.01])
    def test_mass_conservation_n2(self, state_n2_mixed, params_n2, b):
        beta, gamma, sigma, _ = params_n2
        dx = multi_variant_model(0.0, state_n2_mixed, beta, gamma, sigma, b)
        npt.assert_almost_equal(dx.sum(), 0.0, decimal=12)

    def test_disease_free_equilibrium_n1(
        self, state_n1_disease_free, params_n1
    ):
        beta, gamma, sigma, b = params_n1
        dx = multi_variant_model(
            0.0, state_n1_disease_free, beta, gamma, sigma, b
        )
        npt.assert_array_almost_equal(dx, np.zeros(3))

    def test_disease_free_equilibrium_n2(
        self, state_n2_disease_free, params_n2
    ):
        beta, gamma, sigma, b = params_n2
        dx = multi_variant_model(
            0.0, state_n2_disease_free, beta, gamma, sigma, b
        )
        npt.assert_array_almost_equal(dx, np.zeros(9))

    def test_disease_free_equilibrium_with_birth_n1(
        self, state_n1_disease_free, params_n1
    ):
        """At DFE with sum(x)=1, dx=0 even for b>0 (births and deaths cancel)."""
        beta, gamma, sigma, _ = params_n1
        dx = multi_variant_model(
            0.0, state_n1_disease_free, beta, gamma, sigma, b=0.05
        )
        npt.assert_array_almost_equal(dx, np.zeros(3))

    def test_zero_beta_no_new_infections_n1(self, state_n1_mixed, params_n1):
        """With beta=0, only recovery: S is constant, I decreases, R increases."""
        _, gamma, sigma, b = params_n1
        dx = multi_variant_model(
            0.0, state_n1_mixed, np.array([0.0]), gamma, sigma, b
        )
        npt.assert_almost_equal(dx[0], 0.0)
        assert dx[1] > 0
        assert dx[2] < 0

    def test_higher_beta_increases_infection_rate_n1(
        self, state_n1_mixed, params_n1
    ):
        """Higher transmission rate must increase dI[1,1]."""
        _, gamma, sigma, b = params_n1
        dx_low = multi_variant_model(
            0.0, state_n1_mixed, np.array([0.1]), gamma, sigma, b
        )
        dx_high = multi_variant_model(
            0.0, state_n1_mixed, np.array([0.5]), gamma, sigma, b
        )
        assert dx_high[2] > dx_low[2]


# ──────────────────────────────────────────
# multi_variant_model_for_loops
# ──────────────────────────────────────────


class TestMultiVariantModelForLoops:

    @pytest.mark.parametrize("n, x_len", [(1, 3), (2, 9)])
    def test_output_shape(self, n, x_len, params_n1, params_n2):
        beta, gamma, sigma, b = params_n1 if n == 1 else params_n2
        x = np.zeros(x_len)
        x[0] = 1.0
        assert multi_variant_model_for_loops(
            0.0, x, beta, gamma, sigma, b
        ).shape == (x_len,)

    def test_returns_ndarray(self, state_n1_mixed, params_n1):
        beta, gamma, sigma, b = params_n1
        assert isinstance(
            multi_variant_model_for_loops(
                0.0, state_n1_mixed, beta, gamma, sigma, b
            ),
            np.ndarray,
        )

    def test_t_is_unused(self, state_n1_mixed, params_n1):
        beta, gamma, sigma, b = params_n1
        dx0 = multi_variant_model_for_loops(
            0.0, state_n1_mixed, beta, gamma, sigma, b
        )
        dx99 = multi_variant_model_for_loops(
            99.0, state_n1_mixed, beta, gamma, sigma, b
        )
        npt.assert_array_equal(dx0, dx99)

    def test_known_values_n1(self, state_n1_mixed, params_n1):
        beta, gamma, sigma, b = params_n1
        dx = multi_variant_model_for_loops(
            0.0, state_n1_mixed, beta, gamma, sigma, b
        )
        npt.assert_array_almost_equal(dx, [-0.021, 0.01, 0.011])

    @pytest.mark.parametrize("b", [0.0, 0.01])
    def test_mass_conservation_n1(self, state_n1_mixed, params_n1, b):
        beta, gamma, sigma, _ = params_n1
        dx = multi_variant_model_for_loops(
            0.0, state_n1_mixed, beta, gamma, sigma, b
        )
        npt.assert_almost_equal(dx.sum(), 0.0, decimal=12)

    @pytest.mark.parametrize("b", [0.0, 0.01])
    def test_mass_conservation_n2(self, state_n2_mixed, params_n2, b):
        beta, gamma, sigma, _ = params_n2
        dx = multi_variant_model_for_loops(
            0.0, state_n2_mixed, beta, gamma, sigma, b
        )
        npt.assert_almost_equal(dx.sum(), 0.0, decimal=12)

    def test_disease_free_equilibrium_n1(
        self, state_n1_disease_free, params_n1
    ):
        beta, gamma, sigma, b = params_n1
        dx = multi_variant_model_for_loops(
            0.0, state_n1_disease_free, beta, gamma, sigma, b
        )
        npt.assert_array_almost_equal(dx, np.zeros(3))

    def test_disease_free_equilibrium_n2(
        self, state_n2_disease_free, params_n2
    ):
        beta, gamma, sigma, b = params_n2
        dx = multi_variant_model_for_loops(
            0.0, state_n2_disease_free, beta, gamma, sigma, b
        )
        npt.assert_array_almost_equal(dx, np.zeros(9))


# ──────────────────────────────────────────
# Consistency: multi_variant_model vs multi_variant_model_for_loops
# ──────────────────────────────────────────


class TestConsistencyModelVsForLoops:

    @pytest.mark.parametrize("b", [0.0, 0.01])
    def test_n1_mixed_state(self, state_n1_mixed, params_n1, b):
        beta, gamma, sigma, _ = params_n1
        dx_mat = multi_variant_model(
            0.0, state_n1_mixed, beta, gamma, sigma, b
        )
        dx_loop = multi_variant_model_for_loops(
            0.0, state_n1_mixed, beta, gamma, sigma, b
        )
        npt.assert_array_almost_equal(dx_mat, dx_loop)

    @pytest.mark.parametrize("b", [0.0, 0.01])
    def test_n2_mixed_state(self, state_n2_mixed, params_n2, b):
        beta, gamma, sigma, _ = params_n2
        dx_mat = multi_variant_model(
            0.0, state_n2_mixed, beta, gamma, sigma, b
        )
        dx_loop = multi_variant_model_for_loops(
            0.0, state_n2_mixed, beta, gamma, sigma, b
        )
        npt.assert_array_almost_equal(dx_mat, dx_loop)

    def test_n1_disease_free(self, state_n1_disease_free, params_n1):
        beta, gamma, sigma, b = params_n1
        dx_mat = multi_variant_model(
            0.0, state_n1_disease_free, beta, gamma, sigma, b
        )
        dx_loop = multi_variant_model_for_loops(
            0.0, state_n1_disease_free, beta, gamma, sigma, b
        )
        npt.assert_array_almost_equal(dx_mat, dx_loop)

    def test_n2_disease_free(self, state_n2_disease_free, params_n2):
        beta, gamma, sigma, b = params_n2
        dx_mat = multi_variant_model(
            0.0, state_n2_disease_free, beta, gamma, sigma, b
        )
        dx_loop = multi_variant_model_for_loops(
            0.0, state_n2_disease_free, beta, gamma, sigma, b
        )
        npt.assert_array_almost_equal(dx_mat, dx_loop)

    def test_n2_random_normalized_state(self, params_n2):
        """Agreement holds for any normalized random state."""
        rng = np.random.default_rng(42)
        x = rng.random(9)
        x /= x.sum()
        beta, gamma, sigma, b = params_n2
        dx_mat = multi_variant_model(0.0, x, beta, gamma, sigma, b)
        dx_loop = multi_variant_model_for_loops(0.0, x, beta, gamma, sigma, b)
        npt.assert_array_almost_equal(dx_mat, dx_loop, decimal=10)


# ──────────────────────────────────────────
# multi_variant_model_with_matrices
# ──────────────────────────────────────────


class TestMultiVariantModelWithMatrices:

    def test_output_shape_n1(self, state_n1_mixed, params_n1, matrices_n1):
        beta, gamma, sigma, b = params_n1
        V, G, T, S = matrices_n1
        dx = multi_variant_model_with_matrices(
            0.0, state_n1_mixed, beta, gamma, sigma, V, G, T, S, b
        )
        assert dx.shape == (3,)

    def test_output_shape_n2(self, state_n2_mixed, params_n2, matrices_n2):
        beta, gamma, sigma, b = params_n2
        V, G, T, S = matrices_n2
        dx = multi_variant_model_with_matrices(
            0.0, state_n2_mixed, beta, gamma, sigma, V, G, T, S, b
        )
        assert dx.shape == (9,)

    def test_returns_ndarray(self, state_n1_mixed, params_n1, matrices_n1):
        beta, gamma, sigma, b = params_n1
        V, G, T, S = matrices_n1
        assert isinstance(
            multi_variant_model_with_matrices(
                0.0, state_n1_mixed, beta, gamma, sigma, V, G, T, S, b
            ),
            np.ndarray,
        )

    def test_t_is_unused(self, state_n1_mixed, params_n1, matrices_n1):
        beta, gamma, sigma, b = params_n1
        V, G, T, S = matrices_n1
        dx0 = multi_variant_model_with_matrices(
            0.0, state_n1_mixed, beta, gamma, sigma, V, G, T, S, b
        )
        dx99 = multi_variant_model_with_matrices(
            99.0, state_n1_mixed, beta, gamma, sigma, V, G, T, S, b
        )
        npt.assert_array_equal(dx0, dx99)

    @pytest.mark.parametrize("b", [0.0, 0.01])
    def test_mass_conservation_n1(
        self, state_n1_mixed, params_n1, matrices_n1, b
    ):
        beta, gamma, sigma, _ = params_n1
        V, G, T, S = matrices_n1
        dx = multi_variant_model_with_matrices(
            0.0, state_n1_mixed, beta, gamma, sigma, V, G, T, S, b
        )
        npt.assert_almost_equal(dx.sum(), 0.0, decimal=12)

    @pytest.mark.parametrize("b", [0.0, 0.01])
    def test_mass_conservation_n2(
        self, state_n2_mixed, params_n2, matrices_n2, b
    ):
        beta, gamma, sigma, _ = params_n2
        V, G, T, S = matrices_n2
        dx = multi_variant_model_with_matrices(
            0.0, state_n2_mixed, beta, gamma, sigma, V, G, T, S, b
        )
        npt.assert_almost_equal(dx.sum(), 0.0, decimal=12)

    def test_disease_free_equilibrium_n1(
        self, state_n1_disease_free, params_n1, matrices_n1
    ):
        beta, gamma, sigma, b = params_n1
        V, G, T, S = matrices_n1
        dx = multi_variant_model_with_matrices(
            0.0, state_n1_disease_free, beta, gamma, sigma, V, G, T, S, b
        )
        npt.assert_array_almost_equal(dx, np.zeros(3))

    def test_consistency_with_multi_variant_model_n1(
        self, state_n1_mixed, params_n1, matrices_n1
    ):
        beta, gamma, sigma, b = params_n1
        V, G, T, S = matrices_n1
        dx_base = multi_variant_model(
            0.0, state_n1_mixed, beta, gamma, sigma, b
        )
        dx_mat = multi_variant_model_with_matrices(
            0.0, state_n1_mixed, beta, gamma, sigma, V, G, T, S, b
        )
        npt.assert_array_almost_equal(dx_base, dx_mat)

    def test_consistency_with_multi_variant_model_n2(
        self, state_n2_mixed, params_n2, matrices_n2
    ):
        beta, gamma, sigma, b = params_n2
        V, G, T, S = matrices_n2
        dx_base = multi_variant_model(
            0.0, state_n2_mixed, beta, gamma, sigma, b
        )
        dx_mat = multi_variant_model_with_matrices(
            0.0, state_n2_mixed, beta, gamma, sigma, V, G, T, S, b
        )
        npt.assert_array_almost_equal(dx_base, dx_mat)

    # ── compute_dn_variant ──────────────────────────────────────────────────

    def test_compute_dn_variant_output_shape(
        self, state_n2_mixed, params_n2, matrices_n2
    ):
        beta, gamma, sigma, b = params_n2
        V, G, T, S = matrices_n2
        n = len(beta)
        x_ext = np.concatenate([state_n2_mixed, np.zeros(n)])
        dx = multi_variant_model_with_matrices(
            0.0,
            x_ext,
            beta,
            gamma,
            sigma,
            V,
            G,
            T,
            S,
            b,
            compute_dn_variant=True,
        )
        assert dx.shape == (9 + n,)

    def test_compute_dn_variant_non_negative(
        self, state_n2_mixed, params_n2, matrices_n2
    ):
        """New infection rates per variant must be ≥ 0."""
        beta, gamma, sigma, b = params_n2
        V, G, T, S = matrices_n2
        n = len(beta)
        x_ext = np.concatenate([state_n2_mixed, np.zeros(n)])
        dx = multi_variant_model_with_matrices(
            0.0,
            x_ext,
            beta,
            gamma,
            sigma,
            V,
            G,
            T,
            S,
            b,
            compute_dn_variant=True,
        )
        assert np.all(dx[-n:] >= -1e-12)

    def test_compute_dn_variant_zero_at_disease_free(
        self, state_n2_disease_free, params_n2, matrices_n2
    ):
        """At disease-free state there are no new infections."""
        beta, gamma, sigma, b = params_n2
        V, G, T, S = matrices_n2
        n = len(beta)
        x_ext = np.concatenate([state_n2_disease_free, np.zeros(n)])
        dx = multi_variant_model_with_matrices(
            0.0,
            x_ext,
            beta,
            gamma,
            sigma,
            V,
            G,
            T,
            S,
            b,
            compute_dn_variant=True,
        )
        npt.assert_array_almost_equal(dx[-n:], np.zeros(n))

    # ── compute_dn ─────────────────────────────────────────────────────────

    def test_compute_dn_output_shape(
        self, state_n2_mixed, params_n2, matrices_n2
    ):
        beta, gamma, sigma, b = params_n2
        V, G, T, S = matrices_n2
        x_ext = np.concatenate([state_n2_mixed, np.zeros(1)])
        dx = multi_variant_model_with_matrices(
            0.0, x_ext, beta, gamma, sigma, V, G, T, S, b, compute_dn=True
        )
        assert dx.shape == (9 + 1,)

    def test_compute_dn_non_negative(
        self, state_n2_mixed, params_n2, matrices_n2
    ):
        """Total new infection rate must be ≥ 0."""
        beta, gamma, sigma, b = params_n2
        V, G, T, S = matrices_n2
        x_ext = np.concatenate([state_n2_mixed, np.zeros(1)])
        dx = multi_variant_model_with_matrices(
            0.0, x_ext, beta, gamma, sigma, V, G, T, S, b, compute_dn=True
        )
        assert dx[-1] >= -1e-12

    def test_compute_dn_equals_sum_of_dn_variant(
        self, state_n2_mixed, params_n2, matrices_n2
    ):
        """Total dn must equal the sum of per-variant dn values."""
        beta, gamma, sigma, b = params_n2
        V, G, T, S = matrices_n2
        n = len(beta)
        x_variant = np.concatenate([state_n2_mixed, np.zeros(n)])
        x_total = np.concatenate([state_n2_mixed, np.zeros(1)])
        dx_variant = multi_variant_model_with_matrices(
            0.0,
            x_variant,
            beta,
            gamma,
            sigma,
            V,
            G,
            T,
            S,
            b,
            compute_dn_variant=True,
        )
        dx_total = multi_variant_model_with_matrices(
            0.0, x_total, beta, gamma, sigma, V, G, T, S, b, compute_dn=True
        )
        npt.assert_almost_equal(dx_variant[-n:].sum(), dx_total[-1])

    def test_base_states_unaffected_by_compute_dn_variant(
        self, state_n2_mixed, params_n2, matrices_n2
    ):
        """The first 9 elements of dx must not change when compute_dn_variant=True."""
        beta, gamma, sigma, b = params_n2
        V, G, T, S = matrices_n2
        n = len(beta)
        dx_base = multi_variant_model_with_matrices(
            0.0, state_n2_mixed, beta, gamma, sigma, V, G, T, S, b
        )
        x_ext = np.concatenate([state_n2_mixed, np.zeros(n)])
        dx_ext = multi_variant_model_with_matrices(
            0.0,
            x_ext,
            beta,
            gamma,
            sigma,
            V,
            G,
            T,
            S,
            b,
            compute_dn_variant=True,
        )
        npt.assert_array_almost_equal(dx_base, dx_ext[:9])

    def test_base_states_unaffected_by_compute_dn(
        self, state_n2_mixed, params_n2, matrices_n2
    ):
        """The first 9 elements of dx must not change when compute_dn_variant=True."""
        beta, gamma, sigma, b = params_n2
        V, G, T, S = matrices_n2
        n = len(beta)
        dx_base = multi_variant_model_with_matrices(
            0.0, state_n2_mixed, beta, gamma, sigma, V, G, T, S, b
        )
        x_ext = np.concatenate([state_n2_mixed, np.zeros(n)])
        dx_ext = multi_variant_model_with_matrices(
            0.0,
            x_ext,
            beta,
            gamma,
            sigma,
            V,
            G,
            T,
            S,
            b,
            compute_dn=True,
        )
        npt.assert_array_almost_equal(dx_base, dx_ext[:9])


# ──────────────────────────────────────────
# multi_variant_model_sparse
# ──────────────────────────────────────────


class TestMultiVariantModelSparse:

    def test_output_shape_n1(
        self,
        state_n1_mixed,
        params_n1,
        sparse_operators_n1,
    ):
        beta, _, _, b = params_n1

        dx = multi_variant_model_sparse(
            0.0,
            state_n1_mixed,
            beta,
            sparse_operators_n1,
            b,
        )

        assert dx.shape == (3,)

    def test_output_shape_n2(
        self,
        state_n2_mixed,
        params_n2,
        sparse_operators_n2,
    ):
        beta, _, _, b = params_n2

        dx = multi_variant_model_sparse(
            0.0,
            state_n2_mixed,
            beta,
            sparse_operators_n2,
            b,
        )

        assert dx.shape == (9,)

    def test_returns_ndarray(
        self,
        state_n1_mixed,
        params_n1,
        sparse_operators_n1,
    ):
        beta, _, _, b = params_n1

        result = multi_variant_model_sparse(
            0.0,
            state_n1_mixed,
            beta,
            sparse_operators_n1,
            b,
        )

        assert isinstance(result, np.ndarray)

    def test_t_is_unused(
        self,
        state_n1_mixed,
        params_n1,
        sparse_operators_n1,
    ):
        beta, _, _, b = params_n1

        dx0 = multi_variant_model_sparse(
            0.0,
            state_n1_mixed,
            beta,
            sparse_operators_n1,
            b,
        )
        dx99 = multi_variant_model_sparse(
            99.0,
            state_n1_mixed,
            beta,
            sparse_operators_n1,
            b,
        )

        npt.assert_array_equal(dx0, dx99)

    def test_known_values_n1(
        self,
        state_n1_mixed,
        params_n1,
        sparse_operators_n1,
    ):
        beta, _, _, b = params_n1

        dx = multi_variant_model_sparse(
            0.0,
            state_n1_mixed,
            beta,
            sparse_operators_n1,
            b,
        )

        npt.assert_array_almost_equal(dx, [-0.021, 0.01, 0.011])

    @pytest.mark.parametrize("b", [0.0, 0.01])
    def test_mass_conservation_n1(
        self,
        state_n1_mixed,
        params_n1,
        sparse_operators_n1,
        b,
    ):
        beta, _, _, _ = params_n1

        dx = multi_variant_model_sparse(
            0.0,
            state_n1_mixed,
            beta,
            sparse_operators_n1,
            b,
        )

        npt.assert_almost_equal(dx.sum(), 0.0, decimal=12)

    @pytest.mark.parametrize("b", [0.0, 0.01])
    def test_mass_conservation_n2(
        self,
        state_n2_mixed,
        params_n2,
        sparse_operators_n2,
        b,
    ):
        beta, _, _, _ = params_n2

        dx = multi_variant_model_sparse(
            0.0,
            state_n2_mixed,
            beta,
            sparse_operators_n2,
            b,
        )

        npt.assert_almost_equal(dx.sum(), 0.0, decimal=12)

    def test_disease_free_equilibrium_n1(
        self,
        state_n1_disease_free,
        params_n1,
        sparse_operators_n1,
    ):
        beta, _, _, _ = params_n1

        dx = multi_variant_model_sparse(
            0.0,
            state_n1_disease_free,
            beta,
            sparse_operators_n1,
            b=0.05,
        )

        npt.assert_array_almost_equal(dx, np.zeros(3))

    def test_disease_free_equilibrium_n2(
        self,
        state_n2_disease_free,
        params_n2,
        sparse_operators_n2,
    ):
        beta, _, _, _ = params_n2

        dx = multi_variant_model_sparse(
            0.0,
            state_n2_disease_free,
            beta,
            sparse_operators_n2,
            b=0.05,
        )

        npt.assert_array_almost_equal(dx, np.zeros(9))

    @pytest.mark.parametrize("b", [0.0, 0.01])
    def test_matches_dense_implementation_n1(
        self,
        state_n1_mixed,
        params_n1,
        matrices_n1,
        sparse_operators_n1,
        b,
    ):
        beta, gamma, sigma, _ = params_n1
        V, G, T, S = matrices_n1

        dx_dense = multi_variant_model_with_matrices(
            0.0,
            state_n1_mixed,
            beta,
            gamma,
            sigma,
            V,
            G,
            T,
            S,
            b,
        )
        dx_sparse = multi_variant_model_sparse(
            0.0,
            state_n1_mixed,
            beta,
            sparse_operators_n1,
            b,
        )

        npt.assert_allclose(
            dx_sparse,
            dx_dense,
            rtol=1e-12,
            atol=1e-14,
        )

    @pytest.mark.parametrize("b", [0.0, 0.01])
    def test_matches_dense_implementation_n2(
        self,
        state_n2_mixed,
        params_n2,
        matrices_n2,
        sparse_operators_n2,
        b,
    ):
        beta, gamma, sigma, _ = params_n2
        V, G, T, S = matrices_n2

        dx_dense = multi_variant_model_with_matrices(
            0.0,
            state_n2_mixed,
            beta,
            gamma,
            sigma,
            V,
            G,
            T,
            S,
            b,
        )
        dx_sparse = multi_variant_model_sparse(
            0.0,
            state_n2_mixed,
            beta,
            sparse_operators_n2,
            b,
        )

        npt.assert_allclose(
            dx_sparse,
            dx_dense,
            rtol=1e-12,
            atol=1e-14,
        )

    def test_matches_dense_implementation_n3_with_asymmetric_sigma(self):
        rng = np.random.default_rng(42)
        n = 3

        beta = np.array([0.21, 0.26, 0.18])
        gamma = np.array([0.10, 0.12, 0.08])
        sigma = np.array(
            [
                [1.0, 0.25, 0.70],
                [0.40, 1.0, 0.10],
                [0.65, 0.85, 1.0],
            ]
        )
        b = 0.001
        x = rng.dirichlet(np.ones(3**n))

        V, G, T, S = construct_matrices(gamma, sigma)
        sparse_operators = construct_sparse_operators(gamma, sigma)

        dx_dense = multi_variant_model_with_matrices(
            0.0,
            x,
            beta,
            gamma,
            sigma,
            V,
            G,
            T,
            S,
            b,
        )
        dx_sparse = multi_variant_model_sparse(
            0.0,
            x,
            beta,
            sparse_operators,
            b,
        )

        npt.assert_allclose(
            dx_sparse,
            dx_dense,
            rtol=1e-12,
            atol=1e-14,
        )
