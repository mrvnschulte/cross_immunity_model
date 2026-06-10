"""
Tests for multi_variant_equilibrium_equations.py

Note: eigenvalues_linearized_system calls build_jacobian_function which
constructs the symbolic Jacobian via sympy on the first call for a given n.
Subsequent calls use lru_cache. First-run tests may take several seconds.

Key analytical facts used in this file (n=1, b > 0):
  R0    = beta / (gamma + b)
  I*    = b * (R0 - 1) / beta          [endemic infectious fraction]
  S*    = (gamma + b) / beta = 1 / R0  [endemic susceptible fraction]
  R*    = gamma * I* / b               [endemic recovered fraction]

Jacobian at DFE for n=1 is upper triangular with diagonal (-b, -b, beta-gamma-b).
  -> eigenvalues: -b (×2) and beta-gamma-b
  -> max_real_part > 0 iff R0 > 1   (unstable DFE)
  -> max_real_part < 0 iff R0 < 1   (stable DFE, dominant eig = -b)
"""

import numpy as np
import numpy.testing as npt
import pytest

from cross_immunity_model.multi_variant_equilibrium_equations import (
    check_stability_of_variants,
    eigenvalues_linearized_system,
    find_multi_variant_equilibrium,
    find_stable_multi_variant_equilibrium,
    multi_variant_equilibrium_compartments,
    multi_variant_equilibrium_equation,
    solve_multi_variant_equilibrium_equation,
)
from cross_immunity_model.matrices import construct_matrices
from cross_immunity_model.model_transformations import (
    infectious_per_variant,
    map_I_to_x,
)
from cross_immunity_model.invasion_reproduction_number import (
    invasion_reproduction_number_multiple_variants,
)


### ──────────────────────────────────────────

### Module-level helpers

### ──────────────────────────────────────────


def _r0_n1(p):
    return p["beta"][0] / (p["gamma"][0] + p["b"])


def _endemic_I_variant_n1(p):
    """Analytical endemic I_variant for n=1: b*(R0-1)/beta."""
    r0 = _r0_n1(p)
    return np.array([p["b"] * (r0 - 1) / p["beta"][0]])


def _endemic_S_n1(p):
    """Analytical endemic susceptible fraction for n=1: 1/R0."""
    return 1.0 / _r0_n1(p)


def _compartments(I_v, p, mats):
    V, G, T, S = mats
    return multi_variant_equilibrium_compartments(
        I_v, p["beta"], p["gamma"], p["sigma"], V, G, T, S, p["b"]
    )


def _equation(I_v, p, mats, **kwargs):
    V, G, T, S = mats
    return multi_variant_equilibrium_equation(
        I_v, p["beta"], p["gamma"], p["sigma"], V, G, T, S, p["b"], **kwargs
    )


def _solve(p, mats, I_guess, **kwargs):
    V, G, T, S = mats
    return solve_multi_variant_equilibrium_equation(
        p["beta"],
        p["gamma"],
        p["sigma"],
        V,
        G,
        T,
        S,
        p["b"],
        I_guess,
        **kwargs
    )


def _find(p, mats, I_guess, **kwargs):
    V, G, T, S = mats
    return find_multi_variant_equilibrium(
        p["beta"],
        p["gamma"],
        p["sigma"],
        V,
        G,
        T,
        S,
        p["b"],
        I_guess,
        **kwargs
    )


def _check_stability(coexisting, candidates, p, mats=None):
    V, G, T, S = (None, None, None, None) if mats is None else mats
    return check_stability_of_variants(
        coexisting,
        candidates,
        p["beta"],
        p["gamma"],
        p["sigma"],
        p["b"],
        V=V,
        G=G,
        T=T,
        S=S,
    )


def _find_stable(p, **kwargs):
    return find_stable_multi_variant_equilibrium(
        p["beta"], p["gamma"], p["sigma"], p["b"], **kwargs
    )


def _eigvals(x_eq, p):
    return eigenvalues_linearized_system(
        x_eq, p["beta"], p["gamma"], p["sigma"], p["b"]
    )


### ──────────────────────────────────────────

### Fixtures

### ──────────────────────────────────────────


@pytest.fixture
def p_n1():
    """n=1, R0 = 0.3/0.15 = 2.0 > 1: endemic equilibrium exists."""
    return dict(
        beta=np.array([0.3]),
        gamma=np.array([0.1]),
        sigma=np.array([[1.0]]),
        b=0.05,
    )


@pytest.fixture
def p_n1_stable():
    """n=1, R0 = 0.05/0.15 = 1/3 < 1: DFE is the only stable equilibrium."""
    return dict(
        beta=np.array([0.05]),
        gamma=np.array([0.1]),
        sigma=np.array([[1.0]]),
        b=0.05,
    )


@pytest.fixture
def p_n2():
    """
    n=2 with coexistence equilibrium.
    R0_1 = 0.4/0.10 = 4.0, R0_2 = 0.35/0.09 ≈ 3.89, sigma_12 = 0.3.
    """
    return dict(
        beta=np.array([0.4, 0.35]),
        gamma=np.array([0.08, 0.07]),
        sigma=np.array([[1.0, 0.3], [0.3, 1.0]]),
        b=0.02,
    )


@pytest.fixture
def p_n2_stable():
    """n=2, both R0 < 1: DFE is stable."""
    return dict(
        beta=np.array([0.05, 0.04]),
        gamma=np.array([0.1, 0.1]),
        sigma=np.array([[1.0, 0.5], [0.5, 1.0]]),
        b=0.05,
    )


@pytest.fixture
def mats_n1(p_n1):
    return construct_matrices(p_n1["gamma"], p_n1["sigma"])


@pytest.fixture
def mats_n1_stable(p_n1_stable):
    return construct_matrices(p_n1_stable["gamma"], p_n1_stable["sigma"])


@pytest.fixture
def mats_n2(p_n2):
    return construct_matrices(p_n2["gamma"], p_n2["sigma"])


@pytest.fixture
def mats_n2_stable(p_n2_stable):
    return construct_matrices(p_n2_stable["gamma"], p_n2_stable["sigma"])


@pytest.fixture
def I_v_endemic_n1(p_n1):
    """Analytical endemic infectious fraction for n=1."""
    return _endemic_I_variant_n1(p_n1)


@pytest.fixture
def I_v_zero_n1():
    return np.zeros(1)


@pytest.fixture
def I_v_zero_n2():
    return np.zeros(2)


### ──────────────────────────────────────────

### multi_variant_equilibrium_compartments

### ──────────────────────────────────────────


class TestMultiVariantEquilibriumCompartments:

    # ── Return type and shape ─────────────

    @pytest.mark.parametrize("n", [1, 2])
    def test_output_shape(self, n, p_n1, p_n2, mats_n1, mats_n2):
        p, mats = (p_n1, mats_n1) if n == 1 else (p_n2, mats_n2)
        assert _compartments(np.zeros(n), p, mats).shape == (2**n, 2**n)

    def test_returns_ndarray(self, p_n1, mats_n1, I_v_zero_n1):
        assert isinstance(
            _compartments(I_v_zero_n1, p_n1, mats_n1), np.ndarray
        )

    # ── DFE (I_v = 0) ────────────────────

    def test_dfe_I00_equals_one_n1(self, p_n1, mats_n1, I_v_zero_n1):
        npt.assert_almost_equal(
            _compartments(I_v_zero_n1, p_n1, mats_n1)[0, 0], 1.0
        )

    def test_dfe_I00_equals_one_n2(self, p_n2, mats_n2, I_v_zero_n2):
        npt.assert_almost_equal(
            _compartments(I_v_zero_n2, p_n2, mats_n2)[0, 0], 1.0
        )

    def test_dfe_all_others_zero_n1(self, p_n1, mats_n1, I_v_zero_n1):
        result = _compartments(I_v_zero_n1, p_n1, mats_n1)
        off = result.copy()
        off[0, 0] = 0.0
        npt.assert_array_almost_equal(off, np.zeros((2, 2)))

    def test_dfe_mass_conservation_n1(self, p_n1, mats_n1, I_v_zero_n1):
        npt.assert_almost_equal(
            _compartments(I_v_zero_n1, p_n1, mats_n1).sum(), 1.0
        )

    def test_dfe_mass_conservation_n2(self, p_n2, mats_n2, I_v_zero_n2):
        npt.assert_almost_equal(
            _compartments(I_v_zero_n2, p_n2, mats_n2).sum(), 1.0
        )

    # ── Endemic equilibrium ───────────────

    def test_endemic_mass_conservation_n1(self, p_n1, mats_n1, I_v_endemic_n1):
        npt.assert_almost_equal(
            _compartments(I_v_endemic_n1, p_n1, mats_n1).sum(), 1.0
        )

    def test_endemic_non_negative_n1(self, p_n1, mats_n1, I_v_endemic_n1):
        assert np.all(_compartments(I_v_endemic_n1, p_n1, mats_n1) >= -1e-12)

    def test_endemic_S_equals_one_over_R0(self, p_n1, mats_n1, I_v_endemic_n1):
        """At endemic equilibrium, I[0,0] = S* = 1/R0 = (gamma+b)/beta."""
        result = _compartments(I_v_endemic_n1, p_n1, mats_n1)
        npt.assert_almost_equal(result[0, 0], _endemic_S_n1(p_n1))

    def test_endemic_I11_equals_I_variant(self, p_n1, mats_n1, I_v_endemic_n1):
        """I[1,1] = I_variant[0] at endemic equilibrium (self-consistency)."""
        result = _compartments(I_v_endemic_n1, p_n1, mats_n1)
        npt.assert_almost_equal(result[1, 1], I_v_endemic_n1[0])

    def test_endemic_infectious_consistency(
        self, p_n1, mats_n1, I_v_endemic_n1
    ):
        """infectious_per_variant(I_star) == I_variant at endemic equilibrium."""
        V, G, T, S = mats_n1
        I_star = _compartments(I_v_endemic_n1, p_n1, mats_n1)
        I_v_computed = infectious_per_variant(I_star, V, matrix_form=True)
        npt.assert_array_almost_equal(I_v_computed, I_v_endemic_n1)


### ──────────────────────────────────────────

### multi_variant_equilibrium_equation

### ──────────────────────────────────────────


class TestMultiVariantEquilibriumEquation:

    def test_returns_ndarray(self, p_n1, mats_n1, I_v_zero_n1):
        result = _equation(
            I_v_zero_n1, p_n1, mats_n1, use_inv_formulation=False
        )
        assert isinstance(result, np.ndarray)

    # ── Disease-free equilibrium ──────────

    def test_dfe_is_fixed_point_no_inv(self, p_n1, mats_n1, I_v_zero_n1):
        """DFE (I_v=0) is trivially a fixed point of the non-inv formulation."""
        diff = _equation(I_v_zero_n1, p_n1, mats_n1, use_inv_formulation=False)
        npt.assert_array_almost_equal(diff, np.zeros(1))

    def test_dfe_inv_empty_coexisting_returns_empty_array(
        self, p_n1, mats_n1, I_v_zero_n1
    ):
        """With coexisting=[], inv formulation returns an empty residual."""
        diff = _equation(
            I_v_zero_n1,
            p_n1,
            mats_n1,
            coexisting_variants=[],
            use_inv_formulation=True,
        )
        assert len(diff) == 0

    def test_dfe_inv_residual_equals_r0_minus_one(
        self, p_n1, mats_n1, I_v_zero_n1
    ):
        """At DFE with coexisting=[0]: inv residual = inv_r0[0] - 1 = R0 - 1."""
        diff = _equation(
            I_v_zero_n1,
            p_n1,
            mats_n1,
            coexisting_variants=[0],
            use_inv_formulation=True,
        )
        npt.assert_almost_equal(diff[0], _r0_n1(p_n1) - 1.0)

    # ── Endemic equilibrium ───────────────

    def test_endemic_is_fixed_point_no_inv(
        self, p_n1, mats_n1, I_v_endemic_n1
    ):
        """At endemic equilibrium: non-inv residual ≈ 0."""
        diff = _equation(
            I_v_endemic_n1, p_n1, mats_n1, use_inv_formulation=False
        )
        npt.assert_array_almost_equal(diff, np.zeros(1), decimal=6)

    def test_endemic_is_fixed_point_inv(self, p_n1, mats_n1, I_v_endemic_n1):
        """At endemic equilibrium: inv_r0[0] = 1, inv residual ≈ 0."""
        diff = _equation(
            I_v_endemic_n1,
            p_n1,
            mats_n1,
            coexisting_variants=[0],
            use_inv_formulation=True,
        )
        npt.assert_array_almost_equal(diff, np.zeros(1), decimal=6)

    # ── Output length ─────────────────────

    def test_output_length_no_inv_n2(self, p_n2, mats_n2, I_v_zero_n2):
        """non-inv: output length equals number of variants."""
        diff = _equation(I_v_zero_n2, p_n2, mats_n2, use_inv_formulation=False)
        assert len(diff) == 2

    @pytest.mark.parametrize("n_cov", [0, 1, 2])
    def test_output_length_inv_n2(self, p_n2, mats_n2, I_v_zero_n2, n_cov):
        """inv: output length equals len(coexisting_variants)."""
        diff = _equation(
            I_v_zero_n2,
            p_n2,
            mats_n2,
            coexisting_variants=list(range(n_cov)),
            use_inv_formulation=True,
        )
        assert len(diff) == n_cov

    # ── Non-fixed point ────────────────────

    def test_arbitrary_I_not_fixed_point_no_inv(self, p_n1, mats_n1):
        """An arbitrary I_v ≠ endemic I_v is not a fixed point."""
        I_arb = np.array([0.4])  # far from endemic (≈ 0.167)
        diff = _equation(I_arb, p_n1, mats_n1, use_inv_formulation=False)
        assert not np.allclose(diff, 0.0, atol=1e-3)


### ──────────────────────────────────────────

### solve_multi_variant_equilibrium_equation

### ──────────────────────────────────────────


class TestSolveMultiVariantEquilibriumEquation:

    def test_returns_ndarray(self, p_n1, mats_n1):
        assert isinstance(_solve(p_n1, mats_n1, np.array([0.1])), np.ndarray)

    def test_output_length_n1(self, p_n1, mats_n1):
        assert len(_solve(p_n1, mats_n1, np.array([0.1]))) == 1

    def test_output_length_n2(self, p_n2, mats_n2):
        assert len(_solve(p_n2, mats_n2, np.array([0.05, 0.05]))) == 2

    def test_result_within_unit_bounds(self, p_n1, mats_n1):
        result = _solve(p_n1, mats_n1, np.array([0.1]))
        assert np.all(result >= 0) and np.all(result <= 1)

    # ── Solution satisfies equilibrium equation ──

    def test_satisfies_equation_no_inv_n1(self, p_n1, mats_n1):
        """Returned I_variant satisfies non-inv equilibrium equation."""
        I_var = _solve(
            p_n1,
            mats_n1,
            np.array([0.1]),
            free_indices=[0],
            use_inv_formulation=False,
        )
        residual = _equation(I_var, p_n1, mats_n1, use_inv_formulation=False)
        npt.assert_array_almost_equal(residual, np.zeros(1), decimal=5)

    def test_satisfies_equation_inv_n1(self, p_n1, mats_n1):
        """Returned I_variant satisfies invasion reproduction number equation."""
        I_var = _solve(
            p_n1,
            mats_n1,
            np.array([0.1]),
            free_indices=[0],
            use_inv_formulation=True,
        )
        residual = _equation(
            I_var,
            p_n1,
            mats_n1,
            coexisting_variants=[0],
            use_inv_formulation=True,
        )
        npt.assert_array_almost_equal(residual, np.zeros(1), decimal=5)

    # ── Analytical consistency ─────────────

    def test_matches_analytical_endemic_n1(
        self, p_n1, mats_n1, I_v_endemic_n1
    ):
        """Numerical solution matches analytical endemic I_variant for n=1."""
        I_var = _solve(
            p_n1,
            mats_n1,
            np.array([0.1]),
            free_indices=[0],
            use_inv_formulation=True,
        )
        npt.assert_almost_equal(I_var[0], I_v_endemic_n1[0], decimal=4)

    # ── free_indices ──────────────────────

    def test_free_indices_fixes_other_variants_at_zero(self, p_n2, mats_n2):
        """With free_indices=[0], variant 1 stays at 0."""
        I_var = _solve(
            p_n2,
            mats_n2,
            np.array([0.05, 0.0]),
            free_indices=[0],
            use_inv_formulation=True,
        )
        npt.assert_almost_equal(I_var[1], 0.0)
        assert I_var[0] > 0

    # ── Custom bounds ─────────────────────

    def test_custom_bounds_respected(self, p_n1, mats_n1):
        """Result must stay within custom [0.05, 0.3] bounds."""
        I_var = _solve(
            p_n1,
            mats_n1,
            np.array([0.1]),
            free_indices=[0],
            bounds=(0.05, 0.3),
        )
        assert 0.05 <= I_var[0] <= 0.3 + 1e-8


### ──────────────────────────────────────────

### find_multi_variant_equilibrium

### ──────────────────────────────────────────


class TestFindMultiVariantEquilibrium:

    def test_returns_tuple_of_three(self, p_n1, mats_n1):
        assert len(_find(p_n1, mats_n1, np.array([0.1]))) == 3

    def test_types(self, p_n1, mats_n1):
        I_var, I_eq, x_eq = _find(p_n1, mats_n1, np.array([0.1]))
        assert isinstance(I_var, np.ndarray)
        assert isinstance(I_eq, np.ndarray)
        assert isinstance(x_eq, np.ndarray)

    def test_I_eq_shape_n1(self, p_n1, mats_n1):
        _, I_eq, _ = _find(p_n1, mats_n1, np.array([0.1]))
        assert I_eq.shape == (2, 2)

    def test_x_eq_length_n1(self, p_n1, mats_n1):
        _, _, x_eq = _find(p_n1, mats_n1, np.array([0.1]))
        assert len(x_eq) == 3

    def test_x_eq_equals_map_I_to_x(self, p_n1, mats_n1):
        """x_eq must be consistent with I_eq via map_I_to_x."""
        I_var, I_eq, x_eq = _find(p_n1, mats_n1, np.array([0.1]))
        npt.assert_array_almost_equal(x_eq, map_I_to_x(I_eq))

    def test_x_eq_sums_to_one(self, p_n1, mats_n1):
        _, _, x_eq = _find(p_n1, mats_n1, np.array([0.1]))
        npt.assert_almost_equal(x_eq.sum(), 1.0)

    # ── DFE via empty free_indices ────────

    def test_empty_free_indices_gives_dfe(self, p_n1, mats_n1):
        """free_indices=[] bypasses solver and returns the DFE."""
        I_var, I_eq, x_eq = _find(p_n1, mats_n1, np.zeros(1), free_indices=[])
        npt.assert_array_almost_equal(I_var, np.zeros(1))
        npt.assert_almost_equal(x_eq[0], 1.0)  # all mass at susceptible

    # ── Endemic equilibrium ───────────────

    def test_endemic_I_variant_positive(self, p_n1, mats_n1):
        I_var, _, _ = _find(p_n1, mats_n1, np.array([0.1]), free_indices=[0])
        assert I_var[0] > 0

    def test_endemic_matches_analytical_n1(
        self, p_n1, mats_n1, I_v_endemic_n1
    ):
        I_var, _, _ = _find(p_n1, mats_n1, np.array([0.1]), free_indices=[0])
        npt.assert_almost_equal(I_var[0], I_v_endemic_n1[0], decimal=4)

    def test_endemic_x_eq_S_equals_one_over_R0(self, p_n1, mats_n1):
        """At endemic equilibrium, susceptible fraction = 1/R0."""
        _, I_eq, _ = _find(p_n1, mats_n1, np.array([0.1]), free_indices=[0])
        npt.assert_almost_equal(I_eq[0, 0], _endemic_S_n1(p_n1), decimal=4)

    # ── n=2 case ──────────────────────────

    def test_n2_x_eq_sums_to_one(self, p_n2, mats_n2):
        _, _, x_eq = _find(p_n2, mats_n2, np.array([0.05, 0.05]))
        npt.assert_almost_equal(x_eq.sum(), 1.0)

    def test_n2_I_variant_non_negative(self, p_n2, mats_n2):
        I_var, _, _ = _find(p_n2, mats_n2, np.array([0.05, 0.05]))
        assert np.all(I_var >= 0)


### ──────────────────────────────────────────

### check_stability_of_variants

### ──────────────────────────────────────────


class TestCheckStabilityOfVariants:

    def test_returns_tuple_of_four(self, p_n1, mats_n1):
        assert len(_check_stability([], [0], p_n1, mats_n1)) == 4

    def test_types(self, p_n1, mats_n1):
        coexisting, I_var, x_eq, conv = _check_stability(
            [], [0], p_n1, mats_n1
        )
        assert isinstance(coexisting, list)
        assert isinstance(I_var, np.ndarray)
        assert isinstance(x_eq, np.ndarray)
        assert isinstance(conv, int)

    # ── DFE stable (R0 < 1) ──────────────

    def test_dfe_stable_when_r0_lt_1(self, p_n1_stable, mats_n1_stable):
        """R0<1: DFE is stable, coexisting=[], convergence=1."""
        coexisting, I_var, x_eq, conv = _check_stability(
            [], [0], p_n1_stable, mats_n1_stable
        )
        assert coexisting == []
        assert conv == 1

    def test_dfe_I_variant_zero_when_r0_lt_1(
        self, p_n1_stable, mats_n1_stable
    ):
        _, I_var, _, _ = _check_stability([], [0], p_n1_stable, mats_n1_stable)
        npt.assert_array_almost_equal(I_var, np.zeros(1))

    def test_dfe_x_eq_sums_to_one(self, p_n1_stable, mats_n1_stable):
        _, _, x_eq, _ = _check_stability([], [0], p_n1_stable, mats_n1_stable)
        npt.assert_almost_equal(x_eq.sum(), 1.0)

    # ── Endemic found (R0 > 1) ────────────

    def test_endemic_found_when_r0_gt_1(self, p_n1, mats_n1):
        """R0>1: variant 0 is added to equilibrium, convergence=1."""
        coexisting, I_var, x_eq, conv = _check_stability(
            [], [0], p_n1, mats_n1
        )
        assert 0 in coexisting
        assert conv == 1

    def test_endemic_I_variant_positive(self, p_n1, mats_n1):
        """All coexisting variants have I_variant > 0 at equilibrium."""
        coexisting, I_var, _, _ = _check_stability([], [0], p_n1, mats_n1)
        for k in coexisting:
            assert I_var[k] > 0

    def test_endemic_x_eq_sums_to_one(self, p_n1, mats_n1):
        _, _, x_eq, _ = _check_stability([], [0], p_n1, mats_n1)
        npt.assert_almost_equal(x_eq.sum(), 1.0)

    def test_endemic_I_variant_matches_analytical_n1(
        self, p_n1, mats_n1, I_v_endemic_n1
    ):
        _, I_var, _, _ = _check_stability([], [0], p_n1, mats_n1)
        npt.assert_almost_equal(I_var[0], I_v_endemic_n1[0], decimal=4)

    # ── Matrices computed internally ──────

    def test_matrices_computed_if_none(self, p_n1):
        """If V=G=T=S=None, matrices are computed internally → same result."""
        coex_expl, I_expl, _, _ = _check_stability([], [0], p_n1, mats=None)
        assert 0 in coex_expl
        assert I_expl[0] > 0

    def test_explicit_and_implicit_matrices_agree(self, p_n1, mats_n1):
        coex_e, I_e, _, _ = _check_stability([], [0], p_n1, mats_n1)
        coex_i, I_i, _, _ = _check_stability([], [0], p_n1, mats=None)
        assert coex_e == coex_i
        npt.assert_array_almost_equal(I_e, I_i, decimal=4)

    # ── n=2 case ──────────────────────────

    def test_n2_convergence_is_1(self, p_n2, mats_n2):
        _, _, _, conv = _check_stability([], [0, 1], p_n2, mats_n2)
        assert conv == 1

    def test_n2_x_eq_sums_to_one(self, p_n2, mats_n2):
        _, _, x_eq, _ = _check_stability([], [0, 1], p_n2, mats_n2)
        npt.assert_almost_equal(x_eq.sum(), 1.0)

    def test_n2_dfe_stable(self, p_n2_stable, mats_n2_stable):
        coexisting, _, _, conv = _check_stability(
            [], [0, 1], p_n2_stable, mats_n2_stable
        )
        assert coexisting == []
        assert conv == 1


### ──────────────────────────────────────────

### find_stable_multi_variant_equilibrium

### ──────────────────────────────────────────


class TestFindStableMultiVariantEquilibrium:

    def test_returns_tuple_of_four(self, p_n1):
        assert len(_find_stable(p_n1)) == 4

    def test_types(self, p_n1):
        coexisting, I_var, x_eq, conv = _find_stable(p_n1)
        assert isinstance(coexisting, list)
        assert isinstance(I_var, np.ndarray)
        assert isinstance(x_eq, np.ndarray)
        assert isinstance(conv, int)

    def test_x_eq_sums_to_one(self, p_n1):
        _, _, x_eq, _ = _find_stable(p_n1)
        npt.assert_almost_equal(x_eq.sum(), 1.0)

    # ── ordered_check=True ───────────────

    def test_dfe_r0_lt_1_ordered(self, p_n1_stable):
        """R0<1, ordered: DFE stable (coexisting=[], conv=1)."""
        coexisting, I_var, _, conv = _find_stable(
            p_n1_stable, ordered_check=True
        )
        assert coexisting == []
        assert conv == 1
        npt.assert_array_almost_equal(I_var, np.zeros(1))

    def test_endemic_r0_gt_1_ordered(self, p_n1):
        """R0>1, ordered: endemic equilibrium found (conv=1)."""
        coexisting, I_var, _, conv = _find_stable(p_n1, ordered_check=True)
        assert 0 in coexisting
        assert conv == 1
        assert I_var[0] > 0

    # ── ordered_check=False ──────────────

    def test_dfe_r0_lt_1_unordered(self, p_n1_stable):
        """R0<1, unordered code path: coexisting=[], I_variant=0."""
        coexisting, I_var, _, conv = _find_stable(
            p_n1_stable, ordered_check=False
        )
        assert coexisting == []
        assert conv == 1
        npt.assert_array_almost_equal(I_var, np.zeros(1))

    def test_endemic_r0_gt_1_unordered(self, p_n1):
        """R0>1, unordered: endemic equilibrium found (conv=1)."""
        coexisting, I_var, _, conv = _find_stable(p_n1, ordered_check=False)
        assert 0 in coexisting
        assert conv == 1
        assert I_var[0] > 0

    # ── Consistency ordered vs unordered ──

    def test_both_formulations_agree_n1(self, p_n1):
        """Ordered and unordered give the same I_variant for n=1."""
        _, I_ord, _, _ = _find_stable(p_n1, ordered_check=True)
        _, I_unord, _, _ = _find_stable(p_n1, ordered_check=False)
        npt.assert_array_almost_equal(I_ord, I_unord, decimal=4)

    # ── Explicit matrices ─────────────────

    def test_explicit_matrices_same_result(self, p_n1, mats_n1):
        """Explicit V,G,T,S gives same result as computing them internally."""
        V, G, T, S = mats_n1
        _, I_expl, _, _ = find_stable_multi_variant_equilibrium(
            p_n1["beta"],
            p_n1["gamma"],
            p_n1["sigma"],
            p_n1["b"],
            V=V,
            G=G,
            T=T,
            S=S,
        )
        _, I_impl, _, _ = _find_stable(p_n1)
        npt.assert_array_almost_equal(I_expl, I_impl, decimal=4)

    # ── n=2 case ──────────────────────────

    def test_n2_convergence_1_ordered(self, p_n2):
        _, _, _, conv = _find_stable(p_n2, ordered_check=True)
        assert conv == 1

    def test_n2_convergence_1_unordered(self, p_n2):
        _, _, _, conv = _find_stable(p_n2, ordered_check=False)
        assert conv == 1

    def test_n2_x_eq_sums_to_one(self, p_n2):
        _, _, x_eq, _ = _find_stable(p_n2)
        npt.assert_almost_equal(x_eq.sum(), 1.0)

    def test_n2_dfe_r0_lt_1_ordered(self, p_n2_stable):
        """n=2, all R0<1, ordered: coexisting=[], conv=1."""
        coexisting, I_var, _, conv = _find_stable(
            p_n2_stable, ordered_check=True
        )
        assert coexisting == []
        assert conv == 1
        npt.assert_array_almost_equal(I_var, np.zeros(2))

    def test_n2_dfe_r0_lt_1_unordered(self, p_n2_stable):
        """n=2, all R0<1, ordered: coexisting=[], conv=1."""
        coexisting, I_var, _, conv = _find_stable(
            p_n2_stable, ordered_check=False
        )
        assert coexisting == []
        assert conv == 1
        npt.assert_array_almost_equal(I_var, np.zeros(2))


### ──────────────────────────────────────────

### eigenvalues_linearized_system

### ──────────────────────────────────────────


class TestEigenvaluesLinearizedSystem:

    # ── Return types and shapes ──────────

    def test_returns_tuple_of_three(self, p_n1):
        result = _eigvals(np.array([1.0, 0.0, 0.0]), p_n1)
        assert len(result) == 3

    def test_eigvals_is_ndarray(self, p_n1):
        eigvals, _, _ = _eigvals(np.array([1.0, 0.0, 0.0]), p_n1)
        assert isinstance(eigvals, np.ndarray)

    def test_eigvals_length_n1(self, p_n1):
        eigvals, _, _ = _eigvals(np.array([1.0, 0.0, 0.0]), p_n1)
        assert len(eigvals) == 3

    def test_eigvects_shape_n1(self, p_n1):
        _, _, eigvects = _eigvals(np.array([1.0, 0.0, 0.0]), p_n1)
        assert eigvects.shape == (3, 3)

    def test_max_real_part_is_float(self, p_n1):
        _, max_real_part, _ = _eigvals(np.array([1.0, 0.0, 0.0]), p_n1)
        assert isinstance(max_real_part, float)

    # ── DFE stability ─────────────────────

    def test_dfe_unstable_when_r0_gt_1(self, p_n1):
        """At DFE with R0>1: dominant eigenvalue > 0 (unstable DFE)."""
        _, max_real_part, _ = _eigvals(np.array([1.0, 0.0, 0.0]), p_n1)
        assert max_real_part > 0

    def test_dfe_stable_when_r0_lt_1(self, p_n1_stable):
        """At DFE with R0<1 and b>0: all eigenvalues negative (stable DFE)."""
        _, max_real_part, _ = _eigvals(np.array([1.0, 0.0, 0.0]), p_n1_stable)
        assert max_real_part < 0

    def test_dominant_eigval_at_dfe_equals_beta_minus_gamma_minus_b_r0_gt_1(
        self, p_n1
    ):
        """At DFE with R0>1: max_real_part = beta - gamma - b."""
        _, max_real_part, _ = _eigvals(np.array([1.0, 0.0, 0.0]), p_n1)
        expected = p_n1["beta"][0] - p_n1["gamma"][0] - p_n1["b"]
        npt.assert_almost_equal(max_real_part, expected, decimal=8)

    def test_dominant_eigval_at_dfe_equals_minus_b_r0_lt_1(self, p_n1_stable):
        """
        At DFE with R0<1 (beta < gamma): max eigenvalue = -b.
        The other non-trivial eigenvalue beta-gamma-b < -b, so -b dominates.
        """
        _, max_real_part, _ = _eigvals(np.array([1.0, 0.0, 0.0]), p_n1_stable)
        npt.assert_almost_equal(max_real_part, -p_n1_stable["b"], decimal=8)

    def test_dfe_eigvals_contain_beta_minus_gamma_minus_b(self, p_n1):
        """beta-gamma-b must appear in the eigenvalues at DFE."""
        eigvals, _, _ = _eigvals(np.array([1.0, 0.0, 0.0]), p_n1)
        target = p_n1["beta"][0] - p_n1["gamma"][0] - p_n1["b"]
        assert any(np.abs(eigvals.real - target) < 1e-6)

    def test_dfe_eigvals_contain_minus_b(self, p_n1):
        """At DFE for n=1 with b>0: -b appears in the eigenvalues."""
        eigvals, _, _ = _eigvals(np.array([1.0, 0.0, 0.0]), p_n1)
        assert any(np.abs(eigvals.real - (-p_n1["b"])) < 1e-6)

    def test_all_dfe_eigvals_negative_when_r0_lt_1(self, p_n1_stable):
        """At DFE with R0<1 and b>0: all eigenvalues have Re < 0."""
        eigvals, _, _ = _eigvals(np.array([1.0, 0.0, 0.0]), p_n1_stable)
        assert np.all(eigvals.real < 0)

    # ── Endemic equilibrium stability ─────

    def test_endemic_stable_when_r0_gt_1(self, p_n1, mats_n1, I_v_endemic_n1):
        """At endemic equilibrium with R0>1: max real part of eigenvalues < 0."""
        I_star = _compartments(I_v_endemic_n1, p_n1, mats_n1)
        x_eq = map_I_to_x(I_star)
        _, max_real_part, _ = _eigvals(x_eq, p_n1)
        assert max_real_part < 0

    def test_endemic_all_eigvals_negative_n1(
        self, p_n1, mats_n1, I_v_endemic_n1
    ):
        """At endemic equilibrium all eigenvalues have Re < 0."""
        I_star = _compartments(I_v_endemic_n1, p_n1, mats_n1)
        x_eq = map_I_to_x(I_star)
        eigvals, _, _ = _eigvals(x_eq, p_n1)
        assert np.all(eigvals.real < 0)

    # ── max_real_part consistency ─────────

    def test_max_real_part_equals_max_of_eigvals(self, p_n1):
        """max_real_part is exactly np.max(eigvals.real)."""
        eigvals, max_real_part, _ = _eigvals(np.array([1.0, 0.0, 0.0]), p_n1)
        npt.assert_almost_equal(max_real_part, np.max(eigvals.real))

    def test_max_real_part_consistency_at_endemic(
        self, p_n1, mats_n1, I_v_endemic_n1
    ):
        """max_real_part equals max(eigvals.real) at endemic equilibrium."""
        I_star = _compartments(I_v_endemic_n1, p_n1, mats_n1)
        x_eq = map_I_to_x(I_star)
        eigvals, max_real_part, _ = _eigvals(x_eq, p_n1)
        npt.assert_almost_equal(max_real_part, np.max(eigvals.real))

    # ── Eigenvector consistency ───────────

    def test_eigvects_satisfy_Av_equals_lambda_v(self, p_n1):
        """J @ v ≈ lambda * v for each eigenpair at DFE."""
        from cross_immunity_model.symbolic_system import (
            build_jacobian_function,
        )
        from cross_immunity_model.utils import number_of_variants

        x_eq = np.array([1.0, 0.0, 0.0])
        eigvals, _, eigvects = _eigvals(x_eq, p_n1)

        n_variants = number_of_variants(x_eq)
        J_func = build_jacobian_function(n_variants)
        sigma_flat = np.array(p_n1["sigma"]).flatten()
        args = (
            list(x_eq)
            + list(p_n1["beta"])
            + list(p_n1["gamma"])
            + list(sigma_flat)
            + [p_n1["b"]]
        )
        J_num = np.array(J_func(*args), dtype=float)

        for i, lam in enumerate(eigvals):
            v = eigvects[:, i]
            npt.assert_array_almost_equal(J_num @ v, lam * v, decimal=8)

    # ── n=2 case ──────────────────────────

    def test_eigvals_length_n2(self, p_n2):
        """n=2: state has 9 components → 9 eigenvalues."""
        x_eq = np.zeros(9)
        x_eq[0] = 1.0
        eigvals, _, _ = _eigvals(x_eq, p_n2)
        assert len(eigvals) == 9

    def test_dfe_unstable_n2(self, p_n2):
        """At DFE with R0>1 for n=2: max real part > 0."""
        x_eq = np.zeros(9)
        x_eq[0] = 1.0
        _, max_real_part, _ = _eigvals(x_eq, p_n2)
        assert max_real_part > 0

    def test_endemic_stable_n2(self, p_n2):
        """At a stable endemic equilibrium for n=2: max real part < 0."""
        coexisting, I_var, x_eq, conv = find_stable_multi_variant_equilibrium(
            p_n2["beta"], p_n2["gamma"], p_n2["sigma"], p_n2["b"]
        )
        if conv == 1:
            _, max_real_part, _ = _eigvals(x_eq, p_n2)
            assert max_real_part < 0
