"""
Tests for two_variant_model.py

Compartment mapping (0-based Python index):
  x[0]=x1  infectious v1, unprotected (susceptible to v2)
  x[1]=x2  infectious v1, immune to v2
  x[2]=x3  co-infectious (v1 + v2)
  x[3]=x4  infectious v2, unprotected (susceptible to v1)
  x[4]=x5  infectious v2, immune to v1
  x[5]=x6  fully susceptible
  x[6]=x7  recovered v1, susceptible to v2
  x[7]=x8  recovered v2, susceptible to v1
  x[8]=x9  recovered both, fully protected

Mass conservation: sum(dx/dt) = b*(1 - sum(x))  [always holds]
"""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.optimize import fsolve

from cross_immunity_model.two_variant_model import (
    two_variant_model,
    susceptibles,
    protected_1,
    protected_2,
    protected_12,
)
from cross_immunity_model.two_variant_equilibrium_equations import (
    endemic_equilibrium_equation,
    x_bar as compute_xbar,
)


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────


def _call(x, p):
    return two_variant_model(
        0.0,
        x,
        p["beta1"],
        p["beta2"],
        p["gamma1"],
        p["gamma2"],
        p["sigma"],
        p["b"],
    )


def _find_endemic_equilibrium(p, i0=(0.1, 0.1)):
    """Find endemic equilibrium via fsolve on the characteristic equation."""
    return fsolve(
        lambda i: endemic_equilibrium_equation(
            i,
            p["beta1"],
            p["beta2"],
            p["gamma1"],
            p["gamma2"],
            p["sigma"],
            p["b"],
        ),
        list(i0),
    )


def _xbar(i, p):
    return compute_xbar(
        i,
        p["beta1"],
        p["beta2"],
        p["gamma1"],
        p["gamma2"],
        p["sigma"],
        p["b"],
    )


# ──────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────


@pytest.fixture
def p():
    """Default asymmetric parameter set with b>0."""
    return dict(
        beta1=0.3, beta2=0.4, gamma1=0.1, gamma2=0.12, sigma=0.5, b=0.1
    )


@pytest.fixture
def p_no_b():
    """Parameter set without demographics (b=0)."""
    return dict(
        beta1=0.3, beta2=0.4, gamma1=0.1, gamma2=0.12, sigma=0.5, b=0.0
    )


@pytest.fixture
def p_sym():
    """Symmetric parameter set (beta1=beta2, gamma1=gamma2)."""
    return dict(
        beta1=0.5, beta2=0.5, gamma1=0.1, gamma2=0.1, sigma=0.3, b=0.05
    )


@pytest.fixture
def p_coexistence():
    """
    Parameters producing a two-strain coexistence equilibrium.
    R0_1 = 0.4/0.10 = 4.0 > 1, R0_2 = 0.35/0.09 ≈ 3.89 > 1.
    Both invasion reproduction numbers > 1 at sigma=0.3.
    """
    return dict(
        beta1=0.4, beta2=0.35, gamma1=0.08, gamma2=0.07, sigma=0.3, b=0.02
    )


@pytest.fixture
def dfe():
    """Disease-free equilibrium: entire population in x6."""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def state_mixed():
    """
    Mixed state with sum=1 for known-value tests (b=0):
      x1=x4=0.1 (infectious), x6=0.8 (susceptible), all others 0.
    """
    return [0.1, 0.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.0]


@pytest.fixture
def state_valid():
    """A non-degenerate unit state for structural tests."""
    x = [0.1, 0.05, 0.02, 0.1, 0.05, 0.4, 0.05, 0.05, 0.18]
    assert abs(sum(x) - 1.0) < 1e-12
    return x


# ──────────────────────────────────────────
# two_variant_model
# ──────────────────────────────────────────


class TestTwoVariantModel:

    # ── Output structure ─────────────────

    def test_returns_list(self, dfe, p):
        assert isinstance(_call(dfe, p), list)

    def test_output_length_9(self, dfe, p):
        assert len(_call(dfe, p)) == 9

    def test_t_is_unused(self, state_mixed, p):
        f0 = two_variant_model(
            0.0,
            state_mixed,
            p["beta1"],
            p["beta2"],
            p["gamma1"],
            p["gamma2"],
            p["sigma"],
            p["b"],
        )
        f99 = two_variant_model(
            99.9,
            state_mixed,
            p["beta1"],
            p["beta2"],
            p["gamma1"],
            p["gamma2"],
            p["sigma"],
            p["b"],
        )
        npt.assert_array_equal(f0, f99)

    # ── Mass conservation ─────────────────
    # sum(dx/dt) = b*(1 - sum(x))

    @pytest.mark.parametrize("b", [0.0, 0.05, 0.1])
    def test_mass_conservation_unit_state(self, state_mixed, p, b):
        """sum(f) = 0 when sum(x)=1, regardless of b."""
        npt.assert_almost_equal(
            sum(_call(state_mixed, p | {"b": b})),
            0.0,
            decimal=12,  # ← geändert
        )

    @pytest.mark.parametrize("b", [0.0, 0.05, 0.1])
    def test_mass_conservation_dfe(self, dfe, p, b):
        npt.assert_almost_equal(
            sum(_call(dfe, p | {"b": b})), 0.0, decimal=12  # ← geändert
        )

    def test_mass_flow_non_unit_sum(self, state_valid, p):
        """sum(dx/dt) = b*(1-sum(x)) for any state (b>0)."""
        x_sub = [xi * 0.8 for xi in state_valid]  # sum(x) = 0.8
        f = _call(x_sub, p)
        npt.assert_almost_equal(sum(f), p["b"] * (1 - sum(x_sub)), decimal=12)

    # ── Disease-free equilibrium ──────────

    def test_dfe_is_fixed_point_with_b(self, dfe, p):
        npt.assert_array_almost_equal(_call(dfe, p), [0.0] * 9)

    def test_dfe_is_fixed_point_no_b(self, dfe, p_no_b):
        npt.assert_array_almost_equal(_call(dfe, p_no_b), [0.0] * 9)

    # ── Known values (b=0, state_mixed) ──
    # x1=x4=0.1, x6=0.8, all others 0; i_1=i_2=0.1
    # beta1=0.3, beta2=0.4, gamma1=0.1, gamma2=0.12, sigma=0.5, b=0

    def test_known_f1(self, state_mixed, p_no_b):
        # -0.1*0.1 - 0.4*0.1*0.1 + 0.5*0.3*0.8*0.1 = -0.002
        npt.assert_almost_equal(_call(state_mixed, p_no_b)[0], -0.002)

    def test_known_f2(self, state_mixed, p_no_b):
        # sigma*beta1*x6*i1 = 0.5*0.3*0.8*0.1 = 0.012
        npt.assert_almost_equal(_call(state_mixed, p_no_b)[1], 0.012)

    def test_known_f3(self, state_mixed, p_no_b):
        # beta1*x4*i1 + beta2*x1*i2 = 0.003 + 0.004 = 0.007
        npt.assert_almost_equal(_call(state_mixed, p_no_b)[2], 0.007)

    def test_known_f4(self, state_mixed, p_no_b):
        # -0.12*0.1 - 0.3*0.1*0.1 + 0.5*0.4*0.8*0.1 = 0.001
        npt.assert_almost_equal(_call(state_mixed, p_no_b)[3], 0.001)

    def test_known_f5(self, state_mixed, p_no_b):
        # sigma*beta2*x6*i2 = 0.5*0.4*0.8*0.1 = 0.016
        npt.assert_almost_equal(_call(state_mixed, p_no_b)[4], 0.016)

    def test_known_f6(self, state_mixed, p_no_b):
        # -beta1*x6*i1 - beta2*x6*i2 = -0.024 - 0.032 = -0.056
        npt.assert_almost_equal(_call(state_mixed, p_no_b)[5], -0.056)

    def test_known_f7(self, state_mixed, p_no_b):
        # gamma1*x1 = 0.1*0.1 = 0.01
        npt.assert_almost_equal(_call(state_mixed, p_no_b)[6], 0.01)

    def test_known_f8(self, state_mixed, p_no_b):
        # gamma2*x4 = 0.12*0.1 = 0.012
        npt.assert_almost_equal(_call(state_mixed, p_no_b)[7], 0.012)

    def test_known_f9(self, state_mixed, p_no_b):
        # gamma1*x2 + gamma2*x5 = 0 (x2=x5=0)
        npt.assert_almost_equal(_call(state_mixed, p_no_b)[8], 0.0)

    # ── Zero transmission ─────────────────

    def test_zero_beta_x1_decreases(self, p):
        """beta=0, b=0: dx1/dt = -gamma1*x1 < 0 when x1>0."""
        p2 = p | {"beta1": 0.0, "beta2": 0.0, "b": 0.0}  # ← geändert
        x = [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]
        assert _call(x, p2)[0] < 0.0

    def test_zero_beta_x4_decreases(self, p):
        """beta=0, b=0: dx4/dt = -gamma2*x4 < 0 when x4>0."""
        p2 = p | {"beta1": 0.0, "beta2": 0.0, "b": 0.0}  # ← geändert
        x = [0.0, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0, 0.0, 0.0]
        assert _call(x, p2)[3] < 0.0

    def test_zero_i1_no_v1_transmission(self, p):
        """i_1=0: no inflow into x1 or x2 via v1-infection."""
        x = [0.0, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0, 0.0, 0.0]
        f = _call(x, p | {"b": 0.0})  # ← geändert
        npt.assert_almost_equal(f[0], 0.0)
        npt.assert_almost_equal(f[1], 0.0)

    def test_zero_i2_no_v2_transmission(self, p):
        """i_2=0: no inflow into x4 or x5 via v2-infection."""
        x = [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]
        f = _call(x, p | {"b": 0.0})  # ← geändert
        npt.assert_almost_equal(f[3], 0.0)
        npt.assert_almost_equal(f[4], 0.0)

    # ── Cross-immunity edge cases ─────────

    def test_sigma_zero_f2_only_from_x8(self, p):
        """
        sigma=0: f2 = beta1*x8*i1 + gamma2*x3 - b*x2.
        The sigma*beta1*x6*i1 term vanishes entirely.
        """
        p2 = p | {"sigma": 0.0, "b": 0.0}  # ← geändert
        x = [0.1, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.2, 0.1]
        i1 = x[0] + x[1] + x[2]
        f = _call(x, p2)
        expected_f2 = p2["beta1"] * x[7] * i1
        npt.assert_almost_equal(f[1], expected_f2)

    def test_sigma_zero_f5_only_from_x7(self, p):
        """
        sigma=0: f5 = beta2*x7*i2 + gamma1*x3 - b*x5.
        The sigma*beta2*x6*i2 term vanishes.
        """
        p2 = p | {"sigma": 0.0, "b": 0.0}  # ← geändert
        x = [0.0, 0.0, 0.0, 0.1, 0.0, 0.6, 0.2, 0.0, 0.1]
        i2 = x[2] + x[3] + x[4]
        f = _call(x, p2)
        expected_f5 = p2["beta2"] * x[6] * i2
        npt.assert_almost_equal(f[4], expected_f5)

    def test_sigma_one_no_x6_to_x1_inflow(self, p):
        """
        sigma=1: (1-sigma)*beta1*x6*i1 = 0.
        f1 depends only on outflow terms.
        """
        p2 = p | {"sigma": 1.0, "b": 0.0}  # ← geändert
        x = [0.1, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1]
        i2 = x[2] + x[3] + x[4]
        f = _call(x, p2)
        expected_f1 = -(p2["gamma1"] * x[0] + p2["beta2"] * x[0] * i2)
        npt.assert_almost_equal(f[0], expected_f1)

    def test_sigma_one_no_x6_to_x4_inflow(self, p):
        """sigma=1: (1-sigma)*beta2*x6*i2 = 0."""
        p2 = p | {"sigma": 1.0, "b": 0.0}  # ← geändert
        x = [0.0, 0.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.1]
        i1 = x[0] + x[1] + x[2]
        f = _call(x, p2)
        expected_f4 = -(p2["gamma2"] * x[3] + p2["beta1"] * x[3] * i1)
        npt.assert_almost_equal(f[3], expected_f4)

    # ── Symmetry ──────────────────────────

    def test_symmetric_f1_equals_f4(self, p_sym):
        """Equal params, x1=x4, i_1=i_2 → f1=f4."""
        x = [0.1, 0.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.0]
        f = _call(x, p_sym)
        npt.assert_almost_equal(f[0], f[3])

    def test_symmetric_f2_equals_f5(self, p_sym):
        """Equal params, symmetric state → f2=f5."""
        x = [0.1, 0.05, 0.0, 0.1, 0.05, 0.7, 0.0, 0.0, 0.0]
        f = _call(x, p_sym)
        npt.assert_almost_equal(f[1], f[4])

    def test_symmetric_f7_equals_f8(self, p_sym):
        """Equal params, x1=x4, x7=x8, i_1=i_2 → f7=f8."""
        x = [0.1, 0.0, 0.0, 0.1, 0.0, 0.6, 0.05, 0.05, 0.1]
        f = _call(x, p_sym)
        npt.assert_almost_equal(f[6], f[7])

    def test_symmetric_f3_consistent(self, p_sym):
        """f3 = beta1*x4*i1 + beta2*x1*i2; symmetric state → both terms equal."""
        x = [0.1, 0.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.0]
        i1 = x[0] + x[1] + x[2]
        i2 = x[2] + x[3] + x[4]
        expected = p_sym["beta1"] * x[3] * i1 + p_sym["beta2"] * x[0] * i2
        npt.assert_almost_equal(_call(x, p_sym)[2], expected)

    # ── Integration: consistency with equilibrium equations ──

    def test_endemic_equilibrium_is_fixed_point(self, p_coexistence):
        """
        The endemic equilibrium from two_variant_equilibrium_equations must
        satisfy two_variant_model(t, x_eq, ...) ≈ 0.
        """
        i_star = _find_endemic_equilibrium(p_coexistence)
        x_eq = _xbar(i_star, p_coexistence)
        f = _call(x_eq, p_coexistence)
        npt.assert_array_almost_equal(f, [0.0] * 9, decimal=6)

    def test_endemic_equilibrium_sum_is_one(self, p_coexistence):
        """x_bar at endemic equilibrium must sum to 1."""
        i_star = _find_endemic_equilibrium(p_coexistence)
        x_eq = _xbar(i_star, p_coexistence)
        npt.assert_almost_equal(sum(x_eq), 1.0)

    def test_higher_beta_increases_susceptible_outflow(self, p_no_b):
        """Higher beta1 leads to faster depletion of x6."""
        x = [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]
        f_low = _call(x, p_no_b | {"beta1": 0.2})  # ← geändert
        f_high = _call(x, p_no_b | {"beta1": 0.6})  # ← geändert
        assert f_high[5] < f_low[5]


# ──────────────────────────────────────────
# susceptibles
# ──────────────────────────────────────────


class TestSusceptibles:

    def test_returns_x6_by_index(self):
        x = list(range(1, 10))  # [1, 2, ..., 9]
        assert susceptibles(x) == 6  # x[5] = 6

    def test_at_dfe_returns_one(self, dfe):
        npt.assert_almost_equal(susceptibles(dfe), 1.0)

    def test_zero_when_all_infected(self):
        x = [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        npt.assert_almost_equal(susceptibles(x), 0.0)

    def test_within_unit_interval_for_valid_state(self, state_valid):
        assert 0.0 <= susceptibles(state_valid) <= 1.0


# ──────────────────────────────────────────
# protected_1
# ──────────────────────────────────────────


class TestProtected1:

    def test_returns_x1_plus_x7(self):
        x = list(range(1, 10))
        assert protected_1(x) == 1 + 7  # x[0] + x[6]

    def test_at_dfe_returns_zero(self, dfe):
        npt.assert_almost_equal(protected_1(dfe), 0.0)

    def test_known_value(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        npt.assert_almost_equal(protected_1(x), 0.1 + 0.7)  # = 0.8

    def test_within_unit_interval_for_valid_state(self, state_valid):
        assert 0.0 <= protected_1(state_valid) <= 1.0


# ──────────────────────────────────────────
# protected_2
# ──────────────────────────────────────────


class TestProtected2:

    def test_returns_x4_plus_x8(self):
        x = list(range(1, 10))
        assert protected_2(x) == 4 + 8  # x[3] + x[7]

    def test_at_dfe_returns_zero(self, dfe):
        npt.assert_almost_equal(protected_2(dfe), 0.0)

    def test_known_value(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        npt.assert_almost_equal(protected_2(x), 0.4 + 0.8)  # = 1.2

    def test_within_unit_interval_for_valid_state(self, state_valid):
        assert 0.0 <= protected_2(state_valid) <= 1.0

    def test_symmetric_state_equals_protected_1(self):
        """x[0]=x[3] and x[6]=x[7] → protected_1 = protected_2."""
        x = [0.1, 0.0, 0.0, 0.1, 0.0, 0.6, 0.1, 0.1, 0.0]
        npt.assert_almost_equal(protected_1(x), protected_2(x))


# ──────────────────────────────────────────
# protected_12
# ──────────────────────────────────────────


class TestProtected12:

    def test_returns_x2_plus_x3_plus_x5_plus_x9(self):
        x = list(range(1, 10))
        # x[1]+x[2]+x[4]+x[8] = 2+3+5+9 = 19
        assert protected_12(x) == 19

    def test_at_dfe_returns_zero(self, dfe):
        npt.assert_almost_equal(protected_12(dfe), 0.0)

    def test_known_value(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # x[1]+x[2]+x[4]+x[8] = 0.2+0.3+0.5+0.9 = 1.9
        npt.assert_almost_equal(protected_12(x), 1.9)

    def test_within_unit_interval_for_valid_state(self, state_valid):
        assert 0.0 <= protected_12(state_valid) <= 1.0


# ──────────────────────────────────────────
# Partition property: all helpers sum to sum(x)
# ──────────────────────────────────────────


class TestHelperFunctionsPartition:

    def _total(self, x):
        return (
            susceptibles(x) + protected_1(x) + protected_2(x) + protected_12(x)
        )

    def test_sum_equals_sum_x(self, state_valid):
        """The four helpers exactly partition all nine compartments."""
        npt.assert_almost_equal(self._total(state_valid), sum(state_valid))

    def test_sum_equals_one_for_unit_state(self, state_valid):
        npt.assert_almost_equal(self._total(state_valid), 1.0)

    def test_covers_all_nine_indices(self):
        """
        Using x=[0,1,...,8] ensures every index is counted exactly once:
        sum of helper functions must equal sum(range(9)) = 36.
        """
        x = list(range(9))
        assert self._total(x) == sum(range(9))

    def test_dfe_partition(self, dfe):
        """At DFE: susceptibles=1, all others=0."""
        npt.assert_almost_equal(susceptibles(dfe), 1.0)
        npt.assert_almost_equal(protected_1(dfe), 0.0)
        npt.assert_almost_equal(protected_2(dfe), 0.0)
        npt.assert_almost_equal(protected_12(dfe), 0.0)

    def test_endemic_equilibrium_partition_is_one(self, p_coexistence):
        """At the endemic equilibrium: all helpers sum to 1 (sum(x_eq)=1)."""
        i_star = _find_endemic_equilibrium(p_coexistence)
        x_eq = _xbar(i_star, p_coexistence)
        npt.assert_almost_equal(self._total(x_eq), 1.0)
