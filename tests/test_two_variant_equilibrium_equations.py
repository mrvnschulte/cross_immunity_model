"""
Tests for two_variant_equilibrium_equations.py

Notation used in comments:
  alpha = beta_1 * i_1,  beta_t = beta_2 * i_2  (transmission pressures)
  Compartment order in x_bar: [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9]
  i_1 = x_1 + x_2 + x_3,  i_2 = x_3 + x_4 + x_5  (consistency conditions)
"""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.optimize import fsolve

from cross_immunity_model.two_variant_equilibrium_equations import (
    x_1,
    x_2,
    x_3,
    x_4,
    x_5,
    x_6,
    x_7,
    x_8,
    x_9,
    endemic_equilibrium_equation,
    x_bar,
)

# ──────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────

ALL_FUNS = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9]


def _args(p, i_1, i_2):
    """Positional argument tuple for all compartment functions."""
    return (
        i_1,
        i_2,
        p["beta_1"],
        p["beta_2"],
        p["gamma_1"],
        p["gamma_2"],
        p["sigma"],
        p["b"],
    )


def _xbar(i, p):
    return x_bar(
        i,
        p["beta_1"],
        p["beta_2"],
        p["gamma_1"],
        p["gamma_2"],
        p["sigma"],
        p["b"],
    )


def _eq(i, p):
    return endemic_equilibrium_equation(
        i,
        p["beta_1"],
        p["beta_2"],
        p["gamma_1"],
        p["gamma_2"],
        p["sigma"],
        p["b"],
    )


def _find_equilibrium(p, i0=(0.05, 0.05)):
    """Find endemic equilibrium numerically via fsolve."""
    return fsolve(lambda i: _eq(i, p), list(i0))


# ──────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────


@pytest.fixture
def p():
    """Asymmetric parameter set for analytical known-value tests."""
    return dict(
        beta_1=0.3, beta_2=0.4, gamma_1=0.1, gamma_2=0.12, sigma=0.5, b=0.1
    )


@pytest.fixture
def p_sym():
    """Symmetric parameter set (beta_1=beta_2, gamma_1=gamma_2)."""
    return dict(
        beta_1=0.5, beta_2=0.5, gamma_1=0.1, gamma_2=0.1, sigma=0.3, b=0.05
    )


@pytest.fixture
def p_coexistence():
    """
    Parameters ensuring a two-strain coexistence equilibrium exists.
    R0_1 = 0.4 / (0.08 + 0.02) = 4.0
    R0_2 = 0.35 / (0.07 + 0.02) ≈ 3.89
    """
    return dict(
        beta_1=0.4, beta_2=0.35, gamma_1=0.08, gamma_2=0.07, sigma=0.3, b=0.02
    )


# ──────────────────────────────────────────
# x_6  (not infectious, susceptible)
# ──────────────────────────────────────────


class TestX6:

    def test_known_value(self, p):
        # b=0.1, beta_1*i_1=0.06, beta_2*i_2=0.04 → 0.1/0.2 = 0.5
        npt.assert_almost_equal(x_6(*_args(p, 0.2, 0.1)), 0.5)

    def test_disease_free_gives_one(self, p):
        # i_1=i_2=0 → b/b = 1
        npt.assert_almost_equal(x_6(*_args(p, 0.0, 0.0)), 1.0)

    def test_returns_float(self, p):
        assert isinstance(x_6(*_args(p, 0.1, 0.1)), float)

    def test_between_zero_and_one(self, p):
        v = x_6(*_args(p, 0.1, 0.1))
        assert 0.0 <= v <= 1.0

    def test_decreases_with_higher_i1(self, p):
        assert x_6(*_args(p, 0.3, 0.1)) < x_6(*_args(p, 0.1, 0.1))

    def test_decreases_with_higher_i2(self, p):
        assert x_6(*_args(p, 0.1, 0.3)) < x_6(*_args(p, 0.1, 0.1))

    def test_independent_of_gamma_and_sigma(self, p):
        """x_6 formula: b/(beta_1*i_1+beta_2*i_2+b) — no gamma or sigma."""
        p_alt = p.copy()
        p_alt["gamma_1"] = 99.0
        p_alt["gamma_2"] = 99.0
        p_alt["sigma"] = 0.99
        npt.assert_almost_equal(
            x_6(*_args(p, 0.1, 0.1)),
            x_6(*_args(p_alt, 0.1, 0.1)),
        )


# ──────────────────────────────────────────
# x_1  (infectious with variant 1, protected against variant 1)
# ──────────────────────────────────────────


class TestX1:

    def test_known_value(self, p):
        # x_6(0.2, 0.1) = 0.5
        # x_1 = 0.5*0.3*0.5*0.2 / (0.1+0.04+0.1) = 0.015/0.24 = 0.0625
        npt.assert_almost_equal(x_1(*_args(p, 0.2, 0.1)), 0.0625)

    def test_zero_when_i1_is_zero(self, p):
        npt.assert_almost_equal(x_1(*_args(p, 0.0, 0.1)), 0.0)

    def test_zero_when_sigma_is_one(self, p):
        """Full cross-immunity: (1-sigma)=0 → x_1=0."""
        npt.assert_almost_equal(
            x_1(*_args(p | {"sigma": 1.0}, 0.1, 0.1)), 0.0  # ← geändert
        )

    def test_non_negative(self, p):
        assert x_1(*_args(p, 0.1, 0.1)) >= 0.0

    def test_linear_in_one_minus_sigma(self, p):
        """x_1 ∝ (1-sigma); x_6 and denominator do not depend on sigma."""
        r_a = x_1(*_args(p | {"sigma": 0.2}, 0.1, 0.1))  # ← geändert
        r_b = x_1(*_args(p | {"sigma": 0.6}, 0.1, 0.1))  # ← geändert
        npt.assert_almost_equal(r_a / (1 - 0.2), r_b / (1 - 0.6))

    def test_symmetric_equals_x4(self, p_sym):
        """Equal params and i_1=i_2 → x_1 = x_4 by symmetry."""
        npt.assert_almost_equal(
            x_1(*_args(p_sym, 0.15, 0.15)),
            x_4(*_args(p_sym, 0.15, 0.15)),
        )


# ──────────────────────────────────────────
# x_4  (infectious with variant 2, protected against variant 2)
# ──────────────────────────────────────────


class TestX4:

    def test_zero_when_i2_is_zero(self, p):
        npt.assert_almost_equal(x_4(*_args(p, 0.1, 0.0)), 0.0)

    def test_zero_when_sigma_is_one(self, p):
        npt.assert_almost_equal(
            x_4(*_args(p | {"sigma": 1.0}, 0.1, 0.1)), 0.0  # ← geändert
        )

    def test_non_negative(self, p):
        assert x_4(*_args(p, 0.1, 0.1)) >= 0.0

    def test_linear_in_one_minus_sigma(self, p):
        r_a = x_4(*_args(p | {"sigma": 0.2}, 0.1, 0.1))  # ← geändert
        r_b = x_4(*_args(p | {"sigma": 0.6}, 0.1, 0.1))  # ← geändert
        npt.assert_almost_equal(r_a / (1 - 0.2), r_b / (1 - 0.6))


# ──────────────────────────────────────────
# x_7  (not infectious, protected against variant 1)
# ──────────────────────────────────────────


class TestX7:

    def test_zero_when_i1_is_zero(self, p):
        """x_1=0 when i_1=0 → x_7 = gamma_1*x_1/denom = 0."""
        npt.assert_almost_equal(x_7(*_args(p, 0.0, 0.1)), 0.0)

    def test_non_negative(self, p):
        assert x_7(*_args(p, 0.1, 0.1)) >= 0.0

    def test_formula_matches_components(self, p):
        """x_7 = gamma_1 * x_1 / (beta_2*i_2 + b)."""
        i_1, i_2 = 0.1, 0.1
        a = _args(p, i_1, i_2)
        expected = p["gamma_1"] * x_1(*a) / (p["beta_2"] * i_2 + p["b"])
        npt.assert_almost_equal(x_7(*a), expected)

    def test_symmetric_equals_x8(self, p_sym):
        """Equal params and i_1=i_2 → x_7 = x_8 by symmetry."""
        npt.assert_almost_equal(
            x_7(*_args(p_sym, 0.15, 0.15)),
            x_8(*_args(p_sym, 0.15, 0.15)),
        )


# ──────────────────────────────────────────
# x_8  (not infectious, protected against variant 2)
# ──────────────────────────────────────────


class TestX8:

    def test_zero_when_i2_is_zero(self, p):
        """x_4=0 when i_2=0 → x_8 = gamma_2*x_4/denom = 0."""
        npt.assert_almost_equal(x_8(*_args(p, 0.1, 0.0)), 0.0)

    def test_non_negative(self, p):
        assert x_8(*_args(p, 0.1, 0.1)) >= 0.0

    def test_formula_matches_components(self, p):
        """x_8 = gamma_2 * x_4 / (beta_1*i_1 + b)."""
        i_1, i_2 = 0.1, 0.1
        a = _args(p, i_1, i_2)
        expected = p["gamma_2"] * x_4(*a) / (p["beta_1"] * i_1 + p["b"])
        npt.assert_almost_equal(x_8(*a), expected)


# ──────────────────────────────────────────
# x_3  (infectious with both variants, protected against both variants)
# ──────────────────────────────────────────


class TestX3:

    def test_zero_when_i1_is_zero(self, p):
        """x_1=0 and beta_1*i_1=0 when i_1=0 → both numerator terms = 0."""
        npt.assert_almost_equal(x_3(*_args(p, 0.0, 0.1)), 0.0)

    def test_zero_when_i2_is_zero(self, p):
        """x_4=0 and beta_2*i_2=0 when i_2=0 → both numerator terms = 0."""
        npt.assert_almost_equal(x_3(*_args(p, 0.1, 0.0)), 0.0)

    def test_non_negative(self, p):
        assert x_3(*_args(p, 0.1, 0.1)) >= 0.0

    def test_formula_matches_components(self, p):
        """x_3 = (beta_1*x_4*i_1 + beta_2*x_1*i_2) / (gamma_1+gamma_2+b)."""
        i_1, i_2 = 0.1, 0.1
        a = _args(p, i_1, i_2)
        expected = (
            p["beta_1"] * x_4(*a) * i_1 + p["beta_2"] * x_1(*a) * i_2
        ) / (p["gamma_1"] + p["gamma_2"] + p["b"])
        npt.assert_almost_equal(x_3(*a), expected)


# ──────────────────────────────────────────
# x_2  (infectious with variant 1, protected against both variants)
# ──────────────────────────────────────────


class TestX2:

    def test_zero_when_i1_is_zero(self, p):
        """All terms involve beta_1*i_1 or gamma_2*x_3(i_1=0)=0."""
        npt.assert_almost_equal(x_2(*_args(p, 0.0, 0.1)), 0.0)

    def test_non_negative(self, p):
        assert x_2(*_args(p, 0.1, 0.1)) >= 0.0

    def test_formula_matches_components(self, p):
        """x_2 = (sigma*beta_1*x_6*i_1 + beta_1*x_8*i_1 + gamma_2*x_3)/(gamma_1+b)."""
        i_1, i_2 = 0.1, 0.1
        a = _args(p, i_1, i_2)
        expected = (
            p["sigma"] * p["beta_1"] * x_6(*a) * i_1
            + p["beta_1"] * x_8(*a) * i_1
            + p["gamma_2"] * x_3(*a)
        ) / (p["gamma_1"] + p["b"])
        npt.assert_almost_equal(x_2(*a), expected)

    def test_symmetric_equals_x5(self, p_sym):
        """Equal params and i_1=i_2 → x_2 = x_5 by symmetry."""
        npt.assert_almost_equal(
            x_2(*_args(p_sym, 0.15, 0.15)),
            x_5(*_args(p_sym, 0.15, 0.15)),
        )


# ──────────────────────────────────────────
# x_5  (infectious with variant 2, protected against both variants)
# ──────────────────────────────────────────


class TestX5:

    def test_zero_when_i2_is_zero(self, p):
        npt.assert_almost_equal(x_5(*_args(p, 0.1, 0.0)), 0.0)

    def test_non_negative(self, p):
        assert x_5(*_args(p, 0.1, 0.1)) >= 0.0

    def test_formula_matches_components(self, p):
        """x_5 = (sigma*beta_2*x_6*i_2 + beta_2*x_7*i_2 + gamma_1*x_3)/(gamma_2+b)."""
        i_1, i_2 = 0.1, 0.1
        a = _args(p, i_1, i_2)
        expected = (
            p["sigma"] * p["beta_2"] * x_6(*a) * i_2
            + p["beta_2"] * x_7(*a) * i_2
            + p["gamma_1"] * x_3(*a)
        ) / (p["gamma_2"] + p["b"])
        npt.assert_almost_equal(x_5(*a), expected)


# ──────────────────────────────────────────
# x_9  (not infectious, protected against both variants)
# ──────────────────────────────────────────


class TestX9:

    def test_zero_at_disease_free(self, p):
        """x_2=x_5=0 at DFE → x_9 = 0."""
        npt.assert_almost_equal(x_9(*_args(p, 0.0, 0.0)), 0.0)

    def test_non_negative(self, p):
        assert x_9(*_args(p, 0.1, 0.1)) >= 0.0

    def test_formula_matches_components(self, p):
        """x_9 = (gamma_1*x_2 + gamma_2*x_5) / b."""
        i_1, i_2 = 0.1, 0.1
        a = _args(p, i_1, i_2)
        expected = (p["gamma_1"] * x_2(*a) + p["gamma_2"] * x_5(*a)) / p["b"]
        npt.assert_almost_equal(x_9(*a), expected)


# ──────────────────────────────────────────
# All compartments: structural properties
# ──────────────────────────────────────────


class TestAllCompartments:

    def test_disease_free_state(self, p):
        """At DFE (i_1=i_2=0): x_6=1, all others=0."""
        a = _args(p, 0.0, 0.0)
        expected = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        for fn, exp in zip(ALL_FUNS, expected):
            npt.assert_almost_equal(fn(*a), exp, err_msg=f"{fn.__name__}")

    def test_all_non_negative(self, p):
        a = _args(p, 0.1, 0.1)
        for fn in ALL_FUNS:
            assert fn(*a) >= 0.0, f"{fn.__name__} returned a negative value"

    def test_all_return_float(self, p):
        a = _args(p, 0.1, 0.1)
        for fn in ALL_FUNS:
            assert isinstance(
                fn(*a), float
            ), f"{fn.__name__} did not return float"

    @pytest.mark.parametrize("i_1, i_2", [(0.0, 0.0), (0.1, 0.05), (0.3, 0.2)])
    def test_sum_equals_one(self, p, i_1, i_2):
        """
        Sum of all 9 compartments = 1 for any i_1, i_2 (b > 0).
        """
        a = _args(p, i_1, i_2)
        npt.assert_almost_equal(sum(fn(*a) for fn in ALL_FUNS), 1.0)


# ──────────────────────────────────────────
# endemic_equilibrium_equation
# ──────────────────────────────────────────


class TestEndemicEquilibriumEquation:

    def test_returns_list_of_length_2(self, p):
        result = _eq([0.1, 0.1], p)
        assert isinstance(result, list) and len(result) == 2

    def test_disease_free_is_fixed_point(self, p):
        """DFE (i_1=i_2=0) always satisfies the equations (trivially)."""
        npt.assert_array_almost_equal(_eq([0.0, 0.0], p), [0.0, 0.0])

    def test_arbitrary_i_not_fixed_point(self, p):
        """An arbitrary (0.5, 0.5) is not a fixed point for these params."""
        assert not np.allclose(_eq([0.5, 0.5], p), [0.0, 0.0])

    def test_endemic_equilibrium_is_fixed_point(self, p_coexistence):
        """Numerically found equilibrium must satisfy the equations to high precision."""
        i_star = _find_equilibrium(p_coexistence, i0=(0.1, 0.1))
        npt.assert_array_almost_equal(
            _eq(i_star, p_coexistence), [0.0, 0.0], decimal=8
        )

    def test_endemic_equilibrium_i_positive(self, p_coexistence):
        i_star = _find_equilibrium(p_coexistence, i0=(0.1, 0.1))
        assert i_star[0] > 0 and i_star[1] > 0

    def test_endemic_equilibrium_i_less_than_one(self, p_coexistence):
        i_star = _find_equilibrium(p_coexistence, i0=(0.1, 0.1))
        assert i_star[0] < 1.0 and i_star[1] < 1.0

    def test_symmetric_equilibrium_has_equal_i(self, p_sym):
        """For symmetric parameters, coexistence equilibrium has i_1 = i_2."""
        i_star = _find_equilibrium(p_sym, i0=(0.1, 0.1))
        npt.assert_almost_equal(i_star[0], i_star[1], decimal=8)

    def test_equation_0_is_x1_plus_x2_plus_x3_minus_i1(self, p):
        """eq[0] = x_1 + x_2 + x_3 - i_1 (infectious consistency for v1)."""
        i_1, i_2 = 0.1, 0.1
        a = _args(p, i_1, i_2)
        expected = x_1(*a) + x_2(*a) + x_3(*a) - i_1
        npt.assert_almost_equal(_eq([i_1, i_2], p)[0], expected)

    def test_equation_1_is_x3_plus_x4_plus_x5_minus_i2(self, p):
        """eq[1] = x_3 + x_4 + x_5 - i_2 (infectious consistency for v2)."""
        i_1, i_2 = 0.1, 0.1
        a = _args(p, i_1, i_2)
        expected = x_3(*a) + x_4(*a) + x_5(*a) - i_2
        npt.assert_almost_equal(_eq([i_1, i_2], p)[1], expected)


# ──────────────────────────────────────────
# x_bar
# x_bar index map: [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9]
#                    0    1    2    3    4    5    6    7    8
# ──────────────────────────────────────────


class TestXBar:

    def test_returns_list_of_length_9(self, p):
        result = _xbar([0.1, 0.1], p)
        assert isinstance(result, list) and len(result) == 9

    def test_disease_free_state(self, p):
        """At DFE: x_bar = [0,0,0,0,0,1,0,0,0]."""
        npt.assert_array_almost_equal(
            _xbar([0.0, 0.0], p),
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        )

    def test_all_non_negative(self, p):
        assert all(v >= 0.0 for v in _xbar([0.1, 0.1], p))

    def test_sum_equals_one(self, p):
        npt.assert_almost_equal(sum(_xbar([0.1, 0.1], p)), 1.0)

    def test_consistent_with_individual_functions(self, p):
        """x_bar must match calling x_1,...,x_9 individually."""
        i_1, i_2 = 0.12, 0.08
        a = _args(p, i_1, i_2)
        expected = [fn(*a) for fn in ALL_FUNS]
        npt.assert_array_almost_equal(_xbar([i_1, i_2], p), expected)

    def test_i1_consistency_at_equilibrium(self, p_coexistence):
        """At endemic equilibrium: i_1 = x_bar[0]+x_bar[1]+x_bar[2]."""
        i_star = _find_equilibrium(p_coexistence, i0=(0.1, 0.1))
        result = _xbar(i_star, p_coexistence)
        npt.assert_almost_equal(result[0] + result[1] + result[2], i_star[0])

    def test_i2_consistency_at_equilibrium(self, p_coexistence):
        """At endemic equilibrium: i_2 = x_bar[2]+x_bar[3]+x_bar[4]."""
        i_star = _find_equilibrium(p_coexistence, i0=(0.1, 0.1))
        result = _xbar(i_star, p_coexistence)
        npt.assert_almost_equal(result[2] + result[3] + result[4], i_star[1])

    def test_symmetric_x1_equals_x4_at_equilibrium(self, p_sym):
        """For symmetric params at coexistence equilibrium: x_bar[0]=x_bar[3]."""
        i_star = _find_equilibrium(p_sym, i0=(0.1, 0.1))
        result = _xbar(i_star, p_sym)
        npt.assert_almost_equal(result[0], result[3])

    def test_symmetric_x7_equals_x8_at_equilibrium(self, p_sym):
        """For symmetric params at coexistence equilibrium: x_bar[6]=x_bar[7]."""
        i_star = _find_equilibrium(p_sym, i0=(0.1, 0.1))
        result = _xbar(i_star, p_sym)
        npt.assert_almost_equal(result[6], result[7])

    def test_symmetric_x2_equals_x5_at_equilibrium(self, p_sym):
        """For symmetric params at coexistence equilibrium: x_bar[1]=x_bar[4]."""
        i_star = _find_equilibrium(p_sym, i0=(0.1, 0.1))
        result = _xbar(i_star, p_sym)
        npt.assert_almost_equal(result[1], result[4])
