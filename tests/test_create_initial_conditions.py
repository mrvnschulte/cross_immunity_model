"""
Tests for create_initial_conditions.py
"""

import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import patch

from cross_immunity_model.create_initial_conditions import (
    create_protection_state_from_seroprevalence,
    distribute_infectious_fraction,
    create_initial_conditions_from_seroprevalence,
)

MODULE = "cross_immunity_model.create_initial_conditions"


# ──────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────


@pytest.fixture
def simple_I_matrix():
    """
    2x2 protection state matrix for n=1.
    Equivalent to create_protection_state_from_seroprevalence(0.5, [0.8]).
    """
    return np.array([[0.6, 0.4], [0.0, 0.0]], dtype=float)


# ──────────────────────────────────────────
# create_protection_state_from_seroprevalence
# ──────────────────────────────────────────


class TestCreateProtectionStateFromSeroprevalence:

    @pytest.mark.parametrize(
        "n_variants, expected_dim",
        [
            (1, 2),
            (2, 4),
            (3, 8),
        ],
    )
    def test_output_shape(self, n_variants, expected_dim):
        result = create_protection_state_from_seroprevalence(
            0.5, [0.8] * n_variants
        )
        assert result.shape == (expected_dim, expected_dim)

    def test_returns_ndarray(self):
        result = create_protection_state_from_seroprevalence(0.5, [0.8])
        assert isinstance(result, np.ndarray)

    def test_sum_equals_one(self):
        result = create_protection_state_from_seroprevalence(0.5, [0.8, 0.6])
        npt.assert_almost_equal(result.sum(), 1.0)

    def test_non_negative(self):
        result = create_protection_state_from_seroprevalence(0.5, [0.8, 0.6])
        assert np.all(result >= 0)

    def test_only_first_row_populated(self):
        # No infectious individuals before distribute_infectious_fraction
        result = create_protection_state_from_seroprevalence(0.5, [0.8])
        npt.assert_array_almost_equal(result[1:, :], np.zeros((1, 2)))

    def test_zero_seroprevalence_all_mass_at_state_zero(self):
        result = create_protection_state_from_seroprevalence(0.0, [0.8])
        npt.assert_almost_equal(result[0, 0], 1.0)
        off_origin = result.copy()
        off_origin[0, 0] = 0.0
        npt.assert_array_almost_equal(off_origin, np.zeros_like(result))

    def test_full_seroprevalence_and_full_protection_one_variant(self):
        # fraction=1, p=1 → all mass at I[0, 1] (protected, not infectious)
        result = create_protection_state_from_seroprevalence(1.0, [1.0])
        npt.assert_almost_equal(result[0, 1], 1.0)
        npt.assert_almost_equal(result[0, 0], 0.0)

    def test_full_seroprevalence_and_no_protection_one_variant(self):
        # fraction=1, p=0 → all mass stays at I[0, 0] (unprotected)
        result = create_protection_state_from_seroprevalence(1.0, [0.0])
        npt.assert_almost_equal(result[0, 0], 1.0)

    def test_known_values_one_variant(self):
        # fraction=0.5, p=0.8:
        # I[0, 0] = (1 - 0.5) + 0.5 * (1 - 0.8) = 0.5 + 0.1 = 0.6
        # I[0, 1] = 0.5 * 0.8 = 0.4
        result = create_protection_state_from_seroprevalence(0.5, [0.8])
        npt.assert_almost_equal(result[0, 0], 0.6)
        npt.assert_almost_equal(result[0, 1], 0.4)

    def test_matrix_form_false_calls_map_I_to_x(self):
        mock_x = np.array([0.6, 0.4, 0.0, 0.0])
        with patch(f"{MODULE}.map_I_to_x", return_value=mock_x) as mock:
            result = create_protection_state_from_seroprevalence(
                0.5, [0.8], matrix_form=False
            )
        mock.assert_called_once()
        npt.assert_array_equal(result, mock_x)

    def test_matrix_form_true_does_not_call_map_I_to_x(self):
        with patch(f"{MODULE}.map_I_to_x") as mock:
            create_protection_state_from_seroprevalence(
                0.5, [0.8], matrix_form=True
            )
        mock.assert_not_called()


# ──────────────────────────────────────────
# distribute_infectious_fraction
# ──────────────────────────────────────────


class TestDistributeInfectiousFraction:

    def test_returns_ndarray(self, simple_I_matrix):
        with patch(
            f"{MODULE}.susceptibles_per_variant", return_value=np.array([0.6])
        ):
            result = distribute_infectious_fraction(simple_I_matrix, [0.1])
        assert isinstance(result, np.ndarray)

    def test_output_shape_unchanged(self, simple_I_matrix):
        with patch(
            f"{MODULE}.susceptibles_per_variant", return_value=np.array([0.6])
        ):
            result = distribute_infectious_fraction(
                simple_I_matrix.copy(), [0.1]
            )
        assert result.shape == (2, 2)

    def test_sum_preserved(self, simple_I_matrix):
        with patch(
            f"{MODULE}.susceptibles_per_variant", return_value=np.array([0.6])
        ):
            result = distribute_infectious_fraction(
                simple_I_matrix.copy(), [0.1]
            )
        npt.assert_almost_equal(result.sum(), 1.0)

    def test_non_negative(self, simple_I_matrix):
        with patch(
            f"{MODULE}.susceptibles_per_variant", return_value=np.array([0.6])
        ):
            result = distribute_infectious_fraction(
                simple_I_matrix.copy(), [0.1]
            )
        assert np.all(result >= 0)

    def test_zero_infectious_fraction_preserves_matrix(self, simple_I_matrix):
        I_original = simple_I_matrix.copy()
        with patch(
            f"{MODULE}.susceptibles_per_variant", return_value=np.array([0.6])
        ):
            result = distribute_infectious_fraction(
                simple_I_matrix.copy(), [0.0]
            )
        npt.assert_array_almost_equal(result, I_original)

    def test_modifies_input_in_place(self, simple_I_matrix):
        with patch(
            f"{MODULE}.susceptibles_per_variant", return_value=np.array([0.6])
        ):
            result = distribute_infectious_fraction(simple_I_matrix, [0.1])
        assert result is simple_I_matrix

    def test_infectious_mass_equals_input_fraction(self, simple_I_matrix):
        # The total infectious mass (row 1+) should equal infectious_with_variant[0]
        with patch(
            f"{MODULE}.susceptibles_per_variant", return_value=np.array([0.6])
        ):
            result = distribute_infectious_fraction(
                simple_I_matrix.copy(), [0.1]
            )
        npt.assert_almost_equal(result[1:, :].sum(), 0.1)


# ──────────────────────────────────────────
# create_initial_conditions_from_seroprevalence
# ──────────────────────────────────────────


class TestCreateInitialConditionsFromSeroprevalence:

    def test_raises_value_error_on_length_mismatch(self):
        with pytest.raises(ValueError, match="must be of same length"):
            create_initial_conditions_from_seroprevalence(
                fraction_with_antibodies=0.5,
                protection_against_variant=[0.8, 0.6],
                infectious_with_variant=[0.01],
            )

    def test_returns_ndarray(self):
        with patch(f"{MODULE}.map_I_to_x", return_value=np.ones(4)):
            with patch(
                f"{MODULE}.susceptibles_per_variant",
                return_value=np.array([0.6]),
            ):
                result = create_initial_conditions_from_seroprevalence(
                    0.5, [0.8], [0.0]
                )
        assert isinstance(result, np.ndarray)

    def test_calls_map_I_to_x(self):
        with patch(f"{MODULE}.map_I_to_x", return_value=np.ones(4)) as mock:
            with patch(
                f"{MODULE}.susceptibles_per_variant",
                return_value=np.array([0.6]),
            ):
                create_initial_conditions_from_seroprevalence(
                    0.5, [0.8], [0.0]
                )
        mock.assert_called_once()

    def test_returns_output_of_map_I_to_x(self):
        mock_x = np.array([0.6, 0.3, 0.0, 0.1])
        with patch(f"{MODULE}.map_I_to_x", return_value=mock_x):
            with patch(
                f"{MODULE}.susceptibles_per_variant",
                return_value=np.array([0.6]),
            ):
                result = create_initial_conditions_from_seroprevalence(
                    0.5, [0.8], [0.1]
                )
        npt.assert_array_equal(result, mock_x)

    def test_sum_equals_one_with_zero_infectious(self):
        # Integration test: map_I_to_x replaced by flatten to check conservation
        with patch(
            f"{MODULE}.susceptibles_per_variant", return_value=np.array([0.6])
        ):
            with patch(
                f"{MODULE}.map_I_to_x", side_effect=lambda x: x.flatten()
            ):
                result = create_initial_conditions_from_seroprevalence(
                    0.5, [0.8], [0.0]
                )
        npt.assert_almost_equal(result.sum(), 1.0)
