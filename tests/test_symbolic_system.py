"""
Tests for symbolic_system.py

Note: build_jacobian_function uses lru_cache(maxsize=8). The first call
for a given n computes the symbolic Jacobian and may take several seconds.
Subsequent calls with the same n return the cached result immediately.
"""

import numpy as np
import numpy.testing as npt
import pytest
from sympy import Symbol, lambdify

from cross_immunity_model.symbolic_system import (
    build_symbolic_system,
    build_jacobian_function,
)
from cross_immunity_model.multi_variant_model import multi_variant_model


# ──────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────


def _lambdify_dx(dx_symb, x_symb, beta_symb, gamma_symb, sigma_symb, b_symb):
    """Return a callable f(*args) from symbolic dx."""
    all_symbols = (
        list(x_symb)
        + list(beta_symb)
        + list(gamma_symb)
        + list(sigma_symb.flatten())
        + [b_symb]
    )
    return lambdify(all_symbols, list(dx_symb), modules="numpy")


def _args(p: dict) -> list:
    """Flatten a params dict into a positional argument list for lambdified
    functions. Order matches the symbol ordering in build_symbolic_system."""
    return (
        list(p["x"])
        + list(p["beta"])
        + list(p["gamma"])
        + list(p["sigma"].flatten())
        + [p["b"]]
    )


def _jac(n: int, p: dict) -> np.ndarray:
    """Evaluate the lambdified Jacobian for n variants at params p."""
    return np.array(build_jacobian_function(n)(*_args(p)), dtype=float)


def _finite_difference_jacobian(
    f, x: np.ndarray, eps: float = 1e-7
) -> np.ndarray:
    """Central-difference Jacobian of f at x."""
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        J[:, i] = (f(xp) - f(xm)) / (2 * eps)
    return J


# ──────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────


@pytest.fixture
def symb_n1():
    return build_symbolic_system(1)


@pytest.fixture
def symb_n2():
    return build_symbolic_system(2)


@pytest.fixture
def p_n1():
    return dict(
        x=np.array([0.7, 0.2, 0.1]),
        beta=np.array([0.3]),
        gamma=np.array([0.1]),
        sigma=np.array([[1.0]]),
        b=0.0,
    )


@pytest.fixture
def p_n2():
    x = np.zeros(9)
    x[0] = 0.6
    x[1:] = 0.05
    return dict(
        x=x,
        beta=np.array([0.3, 0.4]),
        gamma=np.array([0.1, 0.12]),
        sigma=np.array([[1.0, 0.5], [0.5, 1.0]]),
        b=0.0,
    )


@pytest.fixture
def dfe_n1():
    """Disease-free equilibrium for n=1: entire population susceptible."""
    return dict(
        x=np.array([1.0, 0.0, 0.0]),
        beta=np.array([0.3]),
        gamma=np.array([0.1]),
        sigma=np.array([[1.0]]),
        b=0.0,
    )


@pytest.fixture
def dfe_n1_stable():
    """DFE with R0 = 0.5 < 1: stable equilibrium."""
    return dict(
        x=np.array([1.0, 0.0, 0.0]),
        beta=np.array([0.05]),
        gamma=np.array([0.1]),
        sigma=np.array([[1.0]]),
        b=0.0,
    )


# ──────────────────────────────────────────
# build_symbolic_system
# ──────────────────────────────────────────


class TestBuildSymbolicSystem:

    # ── Output structure ──────────────────

    @pytest.mark.parametrize("n", [1, 2])
    def test_returns_six_elements(self, n):
        assert len(build_symbolic_system(n)) == 6

    @pytest.mark.parametrize("n", [1, 2])
    def test_dx_length(self, n):
        dx, *_ = build_symbolic_system(n)
        assert len(dx) == 3**n

    @pytest.mark.parametrize("n", [1, 2])
    def test_x_length(self, n):
        _, x, *_ = build_symbolic_system(n)
        assert len(x) == 3**n

    @pytest.mark.parametrize("n", [1, 2])
    def test_beta_length(self, n):
        _, _, beta, *_ = build_symbolic_system(n)
        assert len(beta) == n

    @pytest.mark.parametrize("n", [1, 2])
    def test_gamma_length(self, n):
        _, _, _, gamma, *_ = build_symbolic_system(n)
        assert len(gamma) == n

    @pytest.mark.parametrize("n", [1, 2])
    def test_sigma_shape(self, n):
        _, _, _, _, sigma, _ = build_symbolic_system(n)
        assert sigma.shape == (n, n)

    # ── Return types ──────────────────────

    def test_dx_is_ndarray(self, symb_n1):
        dx, *_ = symb_n1
        assert isinstance(dx, np.ndarray)

    def test_x_is_list(self, symb_n1):
        _, x, *_ = symb_n1
        assert isinstance(x, list)

    def test_beta_is_ndarray(self, symb_n1):
        _, _, beta, *_ = symb_n1
        assert isinstance(beta, np.ndarray)

    def test_gamma_is_ndarray(self, symb_n1):
        _, _, _, gamma, *_ = symb_n1
        assert isinstance(gamma, np.ndarray)

    def test_sigma_is_ndarray(self, symb_n1):
        _, _, _, _, sigma, _ = symb_n1
        assert isinstance(sigma, np.ndarray)

    def test_b_is_symbol(self, symb_n1):
        *_, b = symb_n1
        assert isinstance(b, Symbol)

    def test_x_elements_are_symbols(self, symb_n1):
        _, x, *_ = symb_n1
        assert all(isinstance(xi, Symbol) for xi in x)

    def test_beta_elements_are_symbols(self, symb_n1):
        _, _, beta, *_ = symb_n1
        assert all(isinstance(s, Symbol) for s in beta)

    def test_gamma_elements_are_symbols(self, symb_n1):
        _, _, _, gamma, *_ = symb_n1
        assert all(isinstance(s, Symbol) for s in gamma)

    def test_sigma_elements_are_symbols(self, symb_n1):
        _, _, _, _, sigma, _ = symb_n1
        assert all(isinstance(s, Symbol) for s in sigma.flatten())

    # ── Symbol names (n=1, ordering unambiguous) ──

    def test_x_symbol_names_n1(self, symb_n1):
        _, x, *_ = symb_n1
        assert [str(xi) for xi in x] == ["x_0", "x_1", "x_2"]

    def test_beta_symbol_name_n1(self, symb_n1):
        _, _, beta, *_ = symb_n1
        assert str(beta[0]) == "beta_1"

    def test_gamma_symbol_name_n1(self, symb_n1):
        _, _, _, gamma, *_ = symb_n1
        assert str(gamma[0]) == "gamma_1"

    def test_sigma_symbol_name_n1(self, symb_n1):
        _, _, _, _, sigma, _ = symb_n1
        assert str(sigma[0, 0]) == "sigma_11"

    def test_b_symbol_name(self, symb_n1):
        *_, b = symb_n1
        assert str(b) == "b"

    # ── Symbol name structure (n=2) ──

    def test_x_symbol_names_unique_n2(self, symb_n2):
        _, x, *_ = symb_n2
        names = [str(xi) for xi in x]
        assert all(name.startswith("x_") for name in names)
        assert len(set(names)) == 9

    def test_beta_prefix_n2(self, symb_n2):
        _, _, beta, *_ = symb_n2
        assert all(str(s).startswith("beta_") for s in beta)

    def test_gamma_prefix_n2(self, symb_n2):
        _, _, _, gamma, *_ = symb_n2
        assert all(str(s).startswith("gamma_") for s in gamma)

    def test_sigma_prefix_and_unique_n2(self, symb_n2):
        _, _, _, _, sigma, _ = symb_n2
        names = [str(s) for s in sigma.flatten()]
        assert all(name.startswith("sigma_") for name in names)
        assert len(set(names)) == 4  # sigma_11, sigma_12, sigma_21, sigma_22

    # ── Numerical consistency ──

    def test_consistent_with_numerical_model_n1(self, symb_n1, p_n1):
        dx_s, x_s, beta_s, gamma_s, sigma_s, b_s = symb_n1
        f = _lambdify_dx(dx_s, x_s, beta_s, gamma_s, sigma_s, b_s)
        result_symb = np.array(f(*_args(p_n1)), dtype=float)
        result_num = multi_variant_model(
            0.0,
            p_n1["x"],
            p_n1["beta"],
            p_n1["gamma"],
            p_n1["sigma"],
            p_n1["b"],
        )
        npt.assert_array_almost_equal(result_symb, result_num)

    def test_consistent_with_numerical_model_n2(self, symb_n2, p_n2):
        dx_s, x_s, beta_s, gamma_s, sigma_s, b_s = symb_n2
        f = _lambdify_dx(dx_s, x_s, beta_s, gamma_s, sigma_s, b_s)
        result_symb = np.array(f(*_args(p_n2)), dtype=float)
        result_num = multi_variant_model(
            0.0,
            p_n2["x"],
            p_n2["beta"],
            p_n2["gamma"],
            p_n2["sigma"],
            p_n2["b"],
        )
        npt.assert_array_almost_equal(result_symb, result_num)

    # ── Mass conservation ──

    def test_mass_conservation_n1(self, symb_n1, p_n1):
        dx_s, x_s, beta_s, gamma_s, sigma_s, b_s = symb_n1
        f = _lambdify_dx(dx_s, x_s, beta_s, gamma_s, sigma_s, b_s)
        result = np.array(f(*_args(p_n1)), dtype=float)
        npt.assert_almost_equal(result.sum(), 0.0)

    def test_mass_conservation_n2(self, symb_n2, p_n2):
        dx_s, x_s, beta_s, gamma_s, sigma_s, b_s = symb_n2
        f = _lambdify_dx(dx_s, x_s, beta_s, gamma_s, sigma_s, b_s)
        result = np.array(f(*_args(p_n2)), dtype=float)
        npt.assert_almost_equal(result.sum(), 0.0)

    # ── Disease-free equilibrium ──

    def test_disease_free_equilibrium_n1(self, symb_n1, dfe_n1):
        dx_s, x_s, beta_s, gamma_s, sigma_s, b_s = symb_n1
        f = _lambdify_dx(dx_s, x_s, beta_s, gamma_s, sigma_s, b_s)
        result = np.array(f(*_args(dfe_n1)), dtype=float)
        npt.assert_array_almost_equal(result, np.zeros(3))


# ──────────────────────────────────────────
# build_jacobian_function
# ──────────────────────────────────────────


class TestBuildJacobianFunction:

    # ── Basic properties ──

    def test_returns_callable_n1(self):
        assert callable(build_jacobian_function(1))

    def test_returns_callable_n2(self):
        assert callable(build_jacobian_function(2))

    def test_is_cached(self):
        """lru_cache: same object returned on repeated calls with same n."""
        assert build_jacobian_function(1) is build_jacobian_function(1)

    def test_different_n_gives_different_functions(self):
        assert build_jacobian_function(1) is not build_jacobian_function(2)

    def test_output_shape_n1(self, p_n1):
        assert _jac(1, p_n1).shape == (3, 3)

    def test_output_shape_n2(self, p_n2):
        assert _jac(2, p_n2).shape == (9, 9)

    # ── Mass conservation: column sums = 0 ──

    def test_column_sums_zero_n1(self, p_n1):
        """d/dt(sum x) = 0 for all x implies each column of J sums to 0."""
        npt.assert_array_almost_equal(_jac(1, p_n1).sum(axis=0), np.zeros(3))

    def test_column_sums_zero_n2(self, p_n2):
        npt.assert_array_almost_equal(_jac(2, p_n2).sum(axis=0), np.zeros(9))

    # ── Consistency with finite differences ──

    def test_consistent_with_finite_differences_n1(self, p_n1):
        J_symb = _jac(1, p_n1)

        def f(x):
            return multi_variant_model(
                0.0, x, p_n1["beta"], p_n1["gamma"], p_n1["sigma"], p_n1["b"]
            )

        npt.assert_array_almost_equal(
            J_symb, _finite_difference_jacobian(f, p_n1["x"]), decimal=5
        )

    def test_consistent_with_finite_differences_n2(self, p_n2):
        J_symb = _jac(2, p_n2)

        def f(x):
            return multi_variant_model(
                0.0, x, p_n2["beta"], p_n2["gamma"], p_n2["sigma"], p_n2["b"]
            )

        npt.assert_array_almost_equal(
            J_symb, _finite_difference_jacobian(f, p_n2["x"]), decimal=5
        )

    # ── DFE Jacobian structure (n=1) ──────
    #
    # At DFE x=[1,0,0] (all susceptible, I_variant=0):
    #   dx[0] = dI[0,0]/dt = -beta * x[0] * x[2]
    #   dx[1] = dI[0,1]/dt = +gamma * x[2]
    #   dx[2] = dI[1,1]/dt = +beta * x[0] * x[2] - gamma * x[2]
    #
    # Jacobian:
    #   J[:,0] = J[:,1] = 0   (no active transmission at DFE)
    #   J[0,2] = -beta,  J[1,2] = +gamma,  J[2,2] = beta - gamma

    def test_dfe_j02_equals_minus_beta(self, dfe_n1):
        """J[0,2] = d(dS_unprotected)/d(I) = -beta*S = -beta at DFE."""
        J = _jac(1, dfe_n1)
        npt.assert_almost_equal(J[0, 2], -float(dfe_n1["beta"][0]))

    def test_dfe_j12_equals_gamma(self, dfe_n1):
        """J[1,2] = d(dS_protected)/d(I) = gamma at DFE (recovery inflow)."""
        J = _jac(1, dfe_n1)
        npt.assert_almost_equal(J[1, 2], float(dfe_n1["gamma"][0]))

    def test_dfe_j22_equals_beta_minus_gamma(self, dfe_n1):
        """
        J[2,2] = beta*S - gamma = beta - gamma at DFE.
        Sign determines stability: positive iff R0 > 1.
        """
        J = _jac(1, dfe_n1)
        beta, gamma = dfe_n1["beta"][0], dfe_n1["gamma"][0]
        npt.assert_almost_equal(J[2, 2], float(beta - gamma))

    def test_dfe_inactive_columns_are_zero_n1(self, dfe_n1):
        """
        At DFE, I_variant=0: perturbations of S compartments don't
        affect any rates, so J[:,0] = J[:,1] = 0.
        """
        J = _jac(1, dfe_n1)
        npt.assert_array_almost_equal(J[:, 0], np.zeros(3))
        npt.assert_array_almost_equal(J[:, 1], np.zeros(3))

    # ── Stability via dominant eigenvalue ──

    def test_dfe_dominant_eigenvalue_positive_when_r0_gt_1(self, dfe_n1):
        """
        beta=0.3, gamma=0.1 → R0=3 > 1.
        Dominant eigenvalue of J at DFE must be positive (unstable DFE).
        """
        eigenvalues = np.linalg.eigvals(_jac(1, dfe_n1))
        assert np.max(eigenvalues.real) > 0

    def test_dfe_dominant_eigenvalue_equals_beta_minus_gamma(self, dfe_n1):
        """
        At DFE the only non-trivial eigenvalue is beta-gamma (from J[2,2]).
        The other two eigenvalues are 0 (J[:,0]=J[:,1]=0).
        """
        J = _jac(1, dfe_n1)
        eigenvalues = sorted(np.linalg.eigvals(J).real)
        beta, gamma = dfe_n1["beta"][0], dfe_n1["gamma"][0]
        npt.assert_almost_equal(max(eigenvalues), float(beta - gamma))
