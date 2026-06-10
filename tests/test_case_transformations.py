import numpy as np
import pytest
from cross_immunity_model.case_transformations import (
    cumulative_to_daily_cases,
    relative_to_absolute_cases,
    moving_average,
    cases_to_incidence,
    cumulative_cases_to_incidence,
)


# ──────────────────────────────────────────
# cumulative_to_daily_cases
# ──────────────────────────────────────────


class TestCumulativeToDailyCases:

    def test_basic_1d(self):
        result = cumulative_to_daily_cases([0, 1, 3, 6])
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_constant_input_gives_zeros(self):
        result = cumulative_to_daily_cases([5, 5, 5])
        np.testing.assert_array_equal(result, [0, 0])

    def test_2d_input(self):
        cumulative = np.array([[0, 0], [1, 2], [3, 5]])
        result = cumulative_to_daily_cases(cumulative)
        expected = np.array([[1, 2], [2, 3]])
        np.testing.assert_array_equal(result, expected)

    def test_returns_ndarray(self):
        result = cumulative_to_daily_cases([0, 1, 2])
        assert isinstance(result, np.ndarray)


# ──────────────────────────────────────────
# relative_to_absolute_cases
# ──────────────────────────────────────────


class TestRelativeToAbsoluteCases:

    def test_basic(self):
        result = relative_to_absolute_cases([0.1, 0.2], population=1000)
        np.testing.assert_array_almost_equal(result, [100.0, 200.0])

    def test_zero_cases(self):
        result = relative_to_absolute_cases([0.0, 0.0], population=5000)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_returns_ndarray(self):
        result = relative_to_absolute_cases([0.5], population=100)
        assert isinstance(result, np.ndarray)


# ──────────────────────────────────────────
# moving_average
# ──────────────────────────────────────────


class TestMovingAverage:

    def test_basic_1d(self):
        result = moving_average([1, 2, 3, 4, 5, 6, 7], window=7)
        np.testing.assert_array_almost_equal(result, [4.0])

    def test_output_length_1d(self):
        cases = np.ones(10)
        result = moving_average(cases, window=7)
        assert len(result) == 10 - 7 + 1

    def test_uniform_input_unchanged(self):
        cases = np.full(10, 3.0)
        result = moving_average(cases, window=5)
        np.testing.assert_array_almost_equal(result, np.full(6, 3.0))

    def test_2d_input(self):
        cases = np.ones((10, 2))
        result = moving_average(cases, window=7)
        assert result.shape == (4, 2)


# ──────────────────────────────────────────
# cases_to_incidence
# ──────────────────────────────────────────


class TestCasesToIncidence:

    def test_basic(self):
        result = cases_to_incidence(cases=[100], population=100_000)
        np.testing.assert_almost_equal(result[0], 100.0)

    def test_zero_cases(self):
        result = cases_to_incidence(cases=[0], population=50_000)
        np.testing.assert_almost_equal(result[0], 0.0)

    def test_scaling(self):
        # 1 case in 1 million inhabitants → 0.1 per 100k
        result = cases_to_incidence([1], population=1_000_000)
        np.testing.assert_almost_equal(result[0], 0.1)


# ──────────────────────────────────────────
# cumulative_cases_to_incidence (Integration test)
# ──────────────────────────────────────────


class TestCumulativeCasesToIncidence:

    def test_output_shape(self):
        cumulative = np.arange(20)  # 20 days, growing linearly
        result = cumulative_cases_to_incidence(
            cumulative, population=100_000, window=7
        )
        # diff → 19 values, moving_average with window=7 → 13 values
        assert result.shape == (13,)

    def test_constant_daily_cases(self):
        # 7 new cases/day → incidence = 7 * 1e5 / 100_000 = 7.0
        cumulative = np.arange(0, 8 * 7, 7) / 100_000  # 0, 7, 14, ...
        result = cumulative_cases_to_incidence(
            cumulative, population=100_000, window=7
        )
        np.testing.assert_array_almost_equal(result, np.full(len(result), 7.0))
