"""
Tests for invasion_reproduction_number.py
"""

import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import patch

from cross_immunity_model.invasion_reproduction_number import (
    invasion_reproduction_number,
    invasion_reproduction_number_multiple_variants,
    critical_sigma,
    critical_r01,
    critical_r02,
)

MODULE = "cross_immunity_model.invasion_reproduction_number"


# ──────────────────────────────────────────
# invasion_reproduction_number
# ──────────────────────────────────────────


class TestInvasionReproductionNumber:

    def test_known_value(self):
        # r01=2, r02=3, sigma=0.5: 2 - 0.5*(2/3)*(3-1) = 2 - 2/3 = 4/3
        result = invasion_reproduction_number(2.0, 3.0, 0.5)
        npt.assert_almost_equal(result, 4 / 3)

    def test_zero_cross_immunity_returns_r01(self):
        # sigma=0 → invasion_r0 = r01
        result = invasion_reproduction_number(3.0, 2.0, sigma=0.0)
        npt.assert_almost_equal(result, 3.0)

    def test_full_cross_immunity_equal_r0_returns_one(self):
        # sigma=1, r01=r02 → r01 - (r01/r01)*(r01-1) = 1
        result = invasion_reproduction_number(2.0, 2.0, sigma=1.0)
        npt.assert_almost_equal(result, 1.0)

    def test_returns_float(self):
        result = invasion_reproduction_number(2.0, 3.0, 0.5)
        assert isinstance(result, float)

    def test_consistency_with_critical_sigma(self):
        # invasion_r0 must equal 1 at the critical sigma (by definition)
        r01, r02 = 2.0, 4.0
        sigma_c = critical_sigma(r01, r02)
        result = invasion_reproduction_number(r01, r02, sigma_c)
        npt.assert_almost_equal(result, 1.0)

    def test_consistency_with_critical_r01(self):
        # invasion_r0 must equal 1 at the critical r01 (by definition)
        r02, sigma = 3.0, 0.5
        r01_c = critical_r01(r02, sigma)
        result = invasion_reproduction_number(r01_c, r02, sigma)
        npt.assert_almost_equal(result, 1.0)

    def test_consistency_with_critical_r02(self):
        # invasion_r0 must equal 1 at the critical r02 (by definition)
        r01, sigma = 1.5, 0.5
        r02_c = critical_r02(r01, sigma)
        result = invasion_reproduction_number(r01, r02_c, sigma)
        npt.assert_almost_equal(result, 1.0)


# ──────────────────────────────────────────
# invasion_reproduction_number_multiple_variants
# ──────────────────────────────────────────


class TestInvasionReproductionNumberMultipleVariants:

    def test_output_is_elementwise_product_of_s_and_r0(self):
        mock_S = np.array([0.5, 0.4])
        mock_r0 = np.array([2.0, 3.0])
        with patch(f"{MODULE}.susceptibles_per_variant", return_value=mock_S):
            with patch(
                f"{MODULE}.basic_reproduction_number", return_value=mock_r0
            ):
                result = invasion_reproduction_number_multiple_variants(
                    x_eq=np.ones(4),
                    beta=[1.0, 1.5],
                    gamma=[0.5, 0.5],
                    b=0.01,
                    V=None,
                )
        npt.assert_array_almost_equal(result, mock_S * mock_r0)

    def test_returns_ndarray(self):
        mock_S = np.array([0.5, 0.4])
        mock_r0 = np.array([2.0, 3.0])
        with patch(f"{MODULE}.susceptibles_per_variant", return_value=mock_S):
            with patch(
                f"{MODULE}.basic_reproduction_number", return_value=mock_r0
            ):
                result = invasion_reproduction_number_multiple_variants(
                    x_eq=np.ones(4),
                    beta=[1.0, 1.5],
                    gamma=[0.5, 0.5],
                    b=0.01,
                    V=None,
                )
        assert isinstance(result, np.ndarray)

    def test_calls_susceptibles_per_variant_with_correct_args(self):
        x_eq = np.ones(4)
        V = np.eye(2)
        with patch(
            f"{MODULE}.susceptibles_per_variant",
            return_value=np.array([0.5, 0.5]),
        ) as mock_susc:
            with patch(
                f"{MODULE}.basic_reproduction_number",
                return_value=np.array([2.0, 2.0]),
            ):
                invasion_reproduction_number_multiple_variants(
                    x_eq=x_eq, beta=[1.0, 1.5], gamma=[0.5, 0.5], b=0.01, V=V
                )
        mock_susc.assert_called_once_with(x_eq, V)

    def test_calls_basic_reproduction_number_with_correct_args(self):
        beta = [1.0, 1.5]
        gamma = [0.5, 0.5]
        b = 0.01
        with patch(
            f"{MODULE}.susceptibles_per_variant",
            return_value=np.array([0.5, 0.5]),
        ):
            with patch(
                f"{MODULE}.basic_reproduction_number",
                return_value=np.array([2.0, 3.0]),
            ) as mock_r0_fn:
                invasion_reproduction_number_multiple_variants(
                    x_eq=np.ones(4), beta=beta, gamma=gamma, b=b, V=None
                )
        mock_r0_fn.assert_called_once_with(beta, gamma, b)


# ──────────────────────────────────────────
# critical_sigma
# ──────────────────────────────────────────


class TestCriticalSigma:

    def test_r02_less_than_r01_returns_inf(self):
        # Variant 1 can always invade if it has higher R0
        result = critical_sigma(r01=3.0, r02=2.0)
        assert np.isinf(result)

    def test_r02_equal_r01_returns_one(self):
        # sigma_c = r02*(r01-1) / (r01*(r02-1)) = 1 when r01=r02
        result = critical_sigma(r01=2.0, r02=2.0)
        npt.assert_almost_equal(result, 1.0)

    def test_known_value(self):
        # r01=2, r02=4: sigma_c = 4*(2-1) / (2*(4-1)) = 4/6 = 2/3
        result = critical_sigma(r01=2.0, r02=4.0)
        npt.assert_almost_equal(result, 2 / 3)

    def test_scalar_input_returns_float(self):
        result = critical_sigma(2.0, 3.0)
        assert isinstance(result, float)

    def test_array_input_returns_ndarray(self):
        result = critical_sigma(2.0, np.array([2.0, 3.0, 4.0]))
        assert isinstance(result, np.ndarray)

    def test_array_input_correct_shape(self):
        r02 = np.array([2.0, 3.0, 4.0])
        result = critical_sigma(r01=2.0, r02=r02)
        assert result.shape == r02.shape

    def test_array_mixed_inf_and_values(self):
        # r01=2.0, r02=[1.5, 2.0, 4.0] → [inf, 1.0, 2/3]
        result = critical_sigma(r01=2.0, r02=np.array([1.5, 2.0, 4.0]))
        assert np.isinf(result[0])
        npt.assert_almost_equal(result[1], 1.0)
        npt.assert_almost_equal(result[2], 2 / 3)

    def test_result_is_non_negative(self):
        r02 = np.array([2.0, 3.0, 4.0])
        result = critical_sigma(r01=2.0, r02=r02)
        assert np.all(result >= 0)

    def test_consistency_invasion_r0_equals_one_at_critical_sigma(self):
        r01, r02 = 2.0, 4.0
        sigma_c = critical_sigma(r01, r02)
        result = invasion_reproduction_number(r01, r02, sigma_c)
        npt.assert_almost_equal(result, 1.0)


# ──────────────────────────────────────────
# critical_r01
# ──────────────────────────────────────────


class TestCriticalR01:

    def test_r02_less_than_one_returns_one(self):
        # No endemic state for variant 2 → threshold is r01=1
        result = critical_r01(r02=0.5, sigma=0.5)
        npt.assert_almost_equal(result, 1.0)

    def test_r02_equal_one_returns_one(self):
        result = critical_r01(r02=1.0, sigma=0.5)
        npt.assert_almost_equal(result, 1.0)

    def test_known_value(self):
        # r02=2, sigma=0.5: r01_c = 1/(1 - 0.5/2*(2-1)) = 1/(0.75) = 4/3
        result = critical_r01(r02=2.0, sigma=0.5)
        npt.assert_almost_equal(result, 4 / 3)

    def test_zero_cross_immunity_returns_one(self):
        # sigma=0 → r01_c = 1/(1 - 0) = 1
        result = critical_r01(r02=3.0, sigma=0.0)
        npt.assert_almost_equal(result, 1.0)

    def test_scalar_input_returns_float(self):
        result = critical_r01(r02=2.0, sigma=0.5)
        assert isinstance(result, float)

    def test_array_input_returns_ndarray(self):
        result = critical_r01(r02=np.array([0.5, 2.0, 3.0]), sigma=0.5)
        assert isinstance(result, np.ndarray)

    def test_array_input_correct_shape(self):
        r02 = np.array([0.5, 2.0, 3.0])
        result = critical_r01(r02, sigma=0.5)
        assert result.shape == r02.shape

    def test_array_mixed_values(self):
        # r02=[0.5, 1.0, 2.0], sigma=0.5 → [1.0, 1.0, 4/3]
        r02 = np.array([0.5, 1.0, 2.0])
        result = critical_r01(r02, sigma=0.5)
        npt.assert_almost_equal(result[0], 1.0)
        npt.assert_almost_equal(result[1], 1.0)
        npt.assert_almost_equal(result[2], 4 / 3)

    def test_result_greater_or_equal_one(self):
        r02 = np.array([0.5, 2.0, 3.0, 5.0])
        result = critical_r01(r02, sigma=0.5)
        assert np.all(result >= 1.0)

    def test_consistency_invasion_r0_equals_one_at_critical_r01(self):
        r02, sigma = 3.0, 0.5
        r01_c = critical_r01(r02, sigma)
        result = invasion_reproduction_number(r01_c, r02, sigma)
        npt.assert_almost_equal(result, 1.0)


# ──────────────────────────────────────────
# critical_r02
# ──────────────────────────────────────────


class TestCriticalR02:

    def test_r01_less_than_one_returns_one(self):
        # No endemic state for variant 1 → threshold is r02=1
        result = critical_r02(r01=0.5, sigma=0.5)
        npt.assert_almost_equal(result, 1.0)

    def test_known_value(self):
        # r01=1.5, sigma=0.5: r02_star = 0.5/(0.5 + 1/1.5 - 1) = 0.5/(1/6) = 3.0
        result = critical_r02(r01=1.5, sigma=0.5)
        npt.assert_almost_equal(result, 3.0)

    def test_sigma_zero_returns_inf(self):
        # sigma=0, r01>1: r02_star = 0 → inf
        result = critical_r02(r01=2.0, sigma=0.0)
        assert np.isinf(result)

    def test_scalar_input_returns_float(self):
        result = critical_r02(r01=1.5, sigma=0.5)
        assert isinstance(result, float)

    def test_array_all_above_one_returns_ndarray(self):
        result = critical_r02(r01=np.array([1.5, 2.0]), sigma=0.8)
        assert isinstance(result, np.ndarray)

    def test_array_all_above_one_correct_shape(self):
        r01 = np.array([1.5, 2.0])
        result = critical_r02(r01, sigma=0.8)
        assert result.shape == r01.shape

    def test_array_all_below_one_returns_ones(self):
        r01 = np.array([0.3, 0.5, 0.9])
        result = critical_r02(r01, sigma=0.5)
        npt.assert_array_almost_equal(result, np.ones(3))

    def test_result_positive_or_inf(self):
        r01 = np.array([1.5, 2.0, 3.0])
        result = critical_r02(r01, sigma=0.8)
        assert np.all((result > 0) | np.isinf(result))

    def test_consistency_invasion_r0_equals_one_at_critical_r02(self):
        r01, sigma = 1.5, 0.5
        r02_c = critical_r02(r01, sigma)
        result = invasion_reproduction_number(r01, r02_c, sigma)
        npt.assert_almost_equal(result, 1.0)
