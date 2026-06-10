"""
Tests for model_transformations.py
"""

import numpy as np
import numpy.testing as npt
import pytest

from cross_immunity_model.model_transformations import (
    map_x_to_I,
    map_I_to_x,
    susceptibles_per_variant,
    infectious_per_variant,
    project_n_variants,
    reduce_variants,
)
from cross_immunity_model.matrices import variant_in_set_matrix


# ──────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────


@pytest.fixture
def state_n1():
    """
    n=1: x=[0.7, 0.2, 0.1]
    I[0,0]=0.7 (susceptible, not infectious)
    I[0,1]=0.2 (protected, not infectious)
    I[1,1]=0.1 (infectious)
    """
    x = np.array([0.7, 0.2, 0.1])
    I = np.array([[0.7, 0.2], [0.0, 0.1]])
    return x, I


@pytest.fixture
def state_n2_all_susceptible():
    """n=2: all mass at I[0,0]=1 (all susceptible, not infectious)"""
    x = np.zeros(9)
    x[0] = 1.0
    I = np.zeros((4, 4))
    I[0, 0] = 1.0
    return x, I


@pytest.fixture
def state_n2_all_protected():
    """n=2: all mass at I[0,3]=1 (fully protected against both variants, not infectious)"""
    # j=3 (binary: 11) = protected against both; k=0 (not infectious)
    # In x: j=3 → x[5]=I[0,3], x[6]=I[1,3], x[7]=I[2,3], x[8]=I[3,3]
    x = np.zeros(9)
    x[5] = 1.0  # I[0,3]
    I = np.zeros((4, 4))
    I[0, 3] = 1.0
    return x, I


# ──────────────────────────────────────────
# map_x_to_I
# ──────────────────────────────────────────


class TestMapXToI:

    @pytest.mark.parametrize("n, x_len", [(1, 3), (2, 9), (3, 27)])
    def test_output_shape(self, n, x_len):
        I = map_x_to_I(np.zeros(x_len))
        assert I.shape == (2**n, 2**n)

    def test_returns_ndarray(self):
        assert isinstance(map_x_to_I(np.zeros(3)), np.ndarray)

    def test_known_values_n1(self, state_n1):
        x, I_expected = state_n1
        npt.assert_array_almost_equal(map_x_to_I(x), I_expected)

    def test_zero_vector_gives_zero_matrix(self):
        npt.assert_array_equal(map_x_to_I(np.zeros(9)), np.zeros((4, 4)))

    def test_sum_preserved(self, state_n1):
        x, _ = state_n1
        npt.assert_almost_equal(map_x_to_I(x).sum(), x.sum())

    @pytest.mark.parametrize("n, x_len", [(1, 3), (2, 9)])
    def test_only_valid_subset_entries_nonzero(self, n, x_len):
        """I[k,j] == 0 whenever k is not a subset of j (bitwise)."""
        I = map_x_to_I(np.ones(x_len))
        for k in range(2**n):
            for j in range(2**n):
                if (k & j) != k:
                    npt.assert_almost_equal(
                        I[k, j],
                        0.0,
                        err_msg=f"Expected I[{k},{j}]=0 (k not subset of j)",
                    )

    def test_dtype_respected(self):
        I = map_x_to_I(np.zeros(3), dtype=np.float32)
        assert I.dtype == np.float32


# ──────────────────────────────────────────
# map_I_to_x
# ──────────────────────────────────────────


class TestMapIToX:

    @pytest.mark.parametrize("n, x_len", [(1, 3), (2, 9), (3, 27)])
    def test_output_length(self, n, x_len):
        x = map_I_to_x(np.zeros((2**n, 2**n)))
        assert len(x) == x_len

    def test_returns_ndarray(self):
        assert isinstance(map_I_to_x(np.zeros((2, 2))), np.ndarray)

    def test_known_values_n1(self, state_n1):
        x_expected, I = state_n1
        npt.assert_array_almost_equal(map_I_to_x(I), x_expected)

    def test_zero_matrix_gives_zero_vector(self):
        npt.assert_array_equal(map_I_to_x(np.zeros((4, 4))), np.zeros(9))

    def test_sum_preserved(self, state_n1):
        _, I = state_n1
        npt.assert_almost_equal(map_I_to_x(I).sum(), I.sum())

    def test_dtype_respected(self):
        x = map_I_to_x(np.zeros((2, 2)), dtype=np.float32)
        assert x.dtype == np.float32


# ──────────────────────────────────────────
# Round-trip: map_x_to_I <-> map_I_to_x
# ──────────────────────────────────────────


class TestRoundTrip:

    @pytest.mark.parametrize("n, x_len", [(1, 3), (2, 9), (3, 27)])
    def test_x_roundtrip(self, n, x_len):
        rng = np.random.default_rng(42)
        x = rng.random(x_len)
        npt.assert_array_almost_equal(map_I_to_x(map_x_to_I(x)), x)

    @pytest.mark.parametrize("n", [1, 2])
    def test_I_roundtrip(self, n):
        """Only entries where k⊆j are set; others remain 0 through round-trip."""
        size = 2**n
        I = np.zeros((size, size))
        for k in range(size):
            for j in range(size):
                if (k & j) == k:
                    I[k, j] = float(k + j + 1)
        npt.assert_array_almost_equal(map_x_to_I(map_I_to_x(I)), I)

    def test_x_roundtrip_sum_preserved(self):
        rng = np.random.default_rng(0)
        x = rng.random(9)
        npt.assert_almost_equal(map_I_to_x(map_x_to_I(x)).sum(), x.sum())


# ──────────────────────────────────────────
# susceptibles_per_variant
# ──────────────────────────────────────────


class TestSusceptiblesPerVariant:

    @pytest.mark.parametrize("n, x_len", [(1, 3), (2, 9)])
    def test_output_shape(self, n, x_len):
        S = susceptibles_per_variant(np.zeros(x_len))
        assert S.shape == (n,)

    def test_returns_ndarray(self, state_n1):
        x, _ = state_n1
        assert isinstance(susceptibles_per_variant(x), np.ndarray)

    def test_known_value_n1_vector_form(self, state_n1):
        # S[0] = I[:,0].sum() = I[0,0] = 0.7
        x, _ = state_n1
        S = susceptibles_per_variant(x, matrix_form=False)
        npt.assert_almost_equal(S[0], 0.7)

    def test_known_value_n1_matrix_form(self, state_n1):
        _, I = state_n1
        S = susceptibles_per_variant(I, matrix_form=True)
        npt.assert_almost_equal(S[0], 0.7)

    def test_matrix_form_consistent_with_vector_form(self, state_n1):
        x, I = state_n1
        npt.assert_array_almost_equal(
            susceptibles_per_variant(x, matrix_form=False),
            susceptibles_per_variant(I, matrix_form=True),
        )

    def test_all_susceptible_gives_ones(self, state_n2_all_susceptible):
        x, _ = state_n2_all_susceptible
        S = susceptibles_per_variant(x)
        npt.assert_array_almost_equal(S, np.ones(2))

    def test_fully_protected_gives_zero(self, state_n2_all_protected):
        # Everyone is protected against both variants → S = [0, 0]
        x, _ = state_n2_all_protected
        S = susceptibles_per_variant(x)
        npt.assert_array_almost_equal(S, np.zeros(2))

    def test_values_between_zero_and_one(self, state_n1):
        x, _ = state_n1
        S = susceptibles_per_variant(x)
        assert np.all(S >= 0) and np.all(S <= 1 + 1e-10)

    def test_provided_V_matches_computed_V(self, state_n1):
        x, _ = state_n1
        V = variant_in_set_matrix(1)
        npt.assert_array_almost_equal(
            susceptibles_per_variant(x, V=V),
            susceptibles_per_variant(x, V=None),
        )


# ──────────────────────────────────────────
# infectious_per_variant
# ──────────────────────────────────────────


class TestInfectiousPerVariant:

    @pytest.mark.parametrize("n, x_len", [(1, 3), (2, 9)])
    def test_output_shape(self, n, x_len):
        I_var = infectious_per_variant(np.zeros(x_len))
        assert I_var.shape == (n,)

    def test_returns_ndarray(self, state_n1):
        x, _ = state_n1
        assert isinstance(infectious_per_variant(x), np.ndarray)

    def test_known_value_n1_vector_form(self, state_n1):
        # I_variant[0] = I[1,1] = 0.1
        x, _ = state_n1
        I_var = infectious_per_variant(x, matrix_form=False)
        npt.assert_almost_equal(I_var[0], 0.1)

    def test_known_value_n1_matrix_form(self, state_n1):
        _, I = state_n1
        I_var = infectious_per_variant(I, matrix_form=True)
        npt.assert_almost_equal(I_var[0], 0.1)

    def test_matrix_form_consistent_with_vector_form(self, state_n1):
        x, I = state_n1
        npt.assert_array_almost_equal(
            infectious_per_variant(x, matrix_form=False),
            infectious_per_variant(I, matrix_form=True),
        )

    def test_no_infectious_gives_zero(self, state_n2_all_susceptible):
        x, _ = state_n2_all_susceptible
        npt.assert_array_almost_equal(infectious_per_variant(x), np.zeros(2))

    def test_values_between_zero_and_one(self, state_n1):
        x, _ = state_n1
        I_var = infectious_per_variant(x)
        assert np.all(I_var >= 0) and np.all(I_var <= 1 + 1e-10)

    def test_susceptibles_plus_infectious_leq_one(self, state_n1):
        x, _ = state_n1
        S = susceptibles_per_variant(x)
        I_var = infectious_per_variant(x)
        assert np.all(S + I_var <= 1 + 1e-10)

    def test_provided_V_matches_computed_V(self, state_n1):
        x, _ = state_n1
        V = variant_in_set_matrix(1)
        npt.assert_array_almost_equal(
            infectious_per_variant(x, V=V),
            infectious_per_variant(x, V=None),
        )


# ──────────────────────────────────────────
# project_n_variants
# ──────────────────────────────────────────


class TestProjectNVariants:

    def test_output_shape(self):
        result = project_n_variants([0.1, 0.2], n=4, indices=[0, 2])
        assert result.shape == (4,)

    def test_returns_ndarray(self):
        assert isinstance(
            project_n_variants([0.1], n=3, indices=[1]), np.ndarray
        )

    def test_custom_indices_correct_positions(self):
        result = project_n_variants([0.3, 0.5], n=4, indices=[1, 3])
        npt.assert_almost_equal(result[1], 0.3)
        npt.assert_almost_equal(result[3], 0.5)

    def test_zeros_at_unused_positions(self):
        result = project_n_variants([0.3, 0.5], n=4, indices=[1, 3])
        npt.assert_almost_equal(result[0], 0.0)
        npt.assert_almost_equal(result[2], 0.0)

    def test_all_positions_filled(self):
        vals = [0.1, 0.2, 0.3]
        result = project_n_variants(vals, n=3, indices=[0, 1, 2])
        npt.assert_array_almost_equal(result, vals)

    def test_indices_none_equivalent_to_arange(self):
        # When indices=None, first indices are used
        vals = [0.1, 0.2, 0.3]
        result_none = project_n_variants(vals, n=4, indices=None)
        result_explicit = project_n_variants(vals, n=4, indices=[0, 1, 2])
        npt.assert_array_almost_equal(result_none, result_explicit)

    def test_single_value_at_last_position(self):
        result = project_n_variants([0.9], n=4, indices=[3])
        npt.assert_almost_equal(result[3], 0.9)
        npt.assert_array_almost_equal(result[:3], np.zeros(3))


# ──────────────────────────────────────────
# reduce_variants
# ──────────────────────────────────────────


class TestReduceVariants:

    def test_returns_ndarray(self):
        assert isinstance(reduce_variants([0.1, 0.2, 0.3]), np.ndarray)

    def test_indices_none_returns_full_array(self):
        full = [0.1, 0.2, 0.3]
        npt.assert_array_almost_equal(
            reduce_variants(full, indices=None), full
        )

    def test_single_index(self):
        npt.assert_array_almost_equal(
            reduce_variants([0.1, 0.2, 0.3], indices=[1]), [0.2]
        )

    def test_multiple_indices(self):
        npt.assert_array_almost_equal(
            reduce_variants([0.1, 0.2, 0.3, 0.4], indices=[0, 2]), [0.1, 0.3]
        )

    def test_output_length_matches_indices(self):
        result = reduce_variants([0.1, 0.2, 0.3, 0.4], indices=[1, 3])
        assert len(result) == 2

    def test_all_indices_returns_full(self):
        full = [0.1, 0.2, 0.3]
        npt.assert_array_almost_equal(
            reduce_variants(full, indices=[0, 1, 2]), full
        )


# ──────────────────────────────────────────
# project / reduce round-trip
# ──────────────────────────────────────────


class TestProjectReduceRoundTrip:

    def test_project_then_reduce_is_identity(self):
        reduced = np.array([0.2, 0.5])
        indices = [1, 3]
        projected = project_n_variants(reduced, n=4, indices=indices)
        recovered = reduce_variants(projected, indices=indices)
        npt.assert_array_almost_equal(recovered, reduced)

    def test_reduce_then_project_recovers_at_indices(self):
        full = np.array([0.1, 0.0, 0.3, 0.0])
        indices = [0, 2]
        reduced = reduce_variants(full, indices=indices)
        projected = project_n_variants(reduced, n=4, indices=indices)
        npt.assert_array_almost_equal(projected[indices], full[indices])
