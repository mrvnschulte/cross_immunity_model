"""
Tests for matrices.py
"""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.sparse import csr_matrix

from cross_immunity_model.matrices import (
    variant_in_set_matrix,
    compute_gamma_matrix,
    construct_transition_matrix,
    shift_matrix,
    construct_matrices,
    SparseModelOperators,
    construct_sparse_transition_matrices,
    construct_sparse_operators,
)

# ──────────────────────────────────────────
# Helpers (inline, no dependency on utils)
# ──────────────────────────────────────────


def _variant_in_set(k: int, j: int) -> bool:
    """Bit k-1 in j is 1?"""
    return bool(j & (1 << (k - 1)))


def _is_subset(i: int, j: int) -> bool:
    """i ⊆ j (bitwise)?"""
    return (i & j) == i


# ──────────────────────────────────────────
# variant_in_set_matrix
# ──────────────────────────────────────────


class TestVariantInSetMatrix:

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_output_shape(self, n):
        V = variant_in_set_matrix(n)
        assert V.shape == (n, 2**n)

    def test_dtype_is_int(self):
        V = variant_in_set_matrix(2)
        assert np.issubdtype(V.dtype, np.integer)

    def test_known_values_n1(self):
        # Bit 0: j=0 → 0, j=1 → 1
        V = variant_in_set_matrix(1)
        npt.assert_array_equal(V, [[0, 1]])

    def test_known_values_n2(self):
        # V[0] = [0,1,0,1], V[1] = [0,0,1,1]
        V = variant_in_set_matrix(2)
        npt.assert_array_equal(V[0], [0, 1, 0, 1])
        npt.assert_array_equal(V[1], [0, 0, 1, 1])

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_entries_are_binary(self, n):
        V = variant_in_set_matrix(n)
        assert np.all((V == 0) | (V == 1))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_each_variant_in_exactly_half_sets(self, n):
        # Each variant is in exactly 2^(n-1) of the 2^n sets
        V = variant_in_set_matrix(n)
        npt.assert_array_equal(V.sum(axis=1), np.full(n, 2 ** (n - 1)))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_column_sum_equals_popcount(self, n):
        # Column sum j = Number of bits in j
        V = variant_in_set_matrix(n)
        for j in range(2**n):
            npt.assert_equal(V[:, j].sum(), bin(j).count("1"))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_encoding_matches_bitwise_check(self, n):
        V = variant_in_set_matrix(n)
        for i in range(n):
            for j in range(2**n):
                expected = int((j & (1 << i)) != 0)
                assert V[i, j] == expected


# ──────────────────────────────────────────
# compute_gamma_matrix
# ──────────────────────────────────────────


class TestComputeGammaMatrix:

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_output_shape(self, n):
        G = compute_gamma_matrix([1.0] * n)
        assert G.shape == (2**n, 2**n)

    def test_known_values_n1(self):
        G = compute_gamma_matrix([2.0])
        expected = np.array([[0.0, 2.0], [0.0, 0.0]])
        npt.assert_array_almost_equal(G, expected)

    def test_known_values_n2(self):
        # G_prev = [[0,2],[0,0]], G = [[0,2,3,0],[0,0,0,3],[0,0,0,2],[0,0,0,0]]
        G = compute_gamma_matrix([2.0, 3.0])
        expected = np.array(
            [[0, 2, 3, 0], [0, 0, 0, 3], [0, 0, 0, 2], [0, 0, 0, 0]],
            dtype=float,
        )
        npt.assert_array_almost_equal(G, expected)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_is_upper_triangular(self, n):
        G = compute_gamma_matrix([1.0] * n)
        npt.assert_array_equal(G, np.triu(G))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_diagonal_is_zero(self, n):
        G = compute_gamma_matrix([1.0] * n)
        npt.assert_array_equal(np.diag(G), np.zeros(2**n))

    def test_non_negative(self):
        G = compute_gamma_matrix([1.0, 2.0, 3.0])
        assert np.all(G >= 0)

    def test_uniform_gamma_scales_linearly(self):
        G1 = compute_gamma_matrix([1.0, 1.0])
        G2 = compute_gamma_matrix([2.0, 2.0])
        npt.assert_array_almost_equal(G2, 2 * G1)

    def test_dtype_is_respected(self):
        G = compute_gamma_matrix([1.0, 2.0], dtype=np.float32)
        assert G.dtype == np.float32

    def test_returns_ndarray(self):
        G = compute_gamma_matrix([1.0])
        assert isinstance(G, np.ndarray)


# ──────────────────────────────────────────
# construct_transition_matrix
# ──────────────────────────────────────────


class TestConstructTransitionMatrix:

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_output_shape(self, n):
        sigma = np.full((n, n), 0.5)
        T = construct_transition_matrix(sigma)
        assert T.shape == (n, 2**n, 2**n)

    def test_non_negative(self):
        sigma = np.full((2, 2), 0.5)
        T = construct_transition_matrix(sigma)
        assert np.all(T >= 0)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_row_sum_is_zero_or_one(self, n):
        # T[k, i, :].sum() = 1 if variant k+1 not in set i, else 0
        sigma = np.full((n, n), 0.5)
        T = construct_transition_matrix(sigma)
        for k in range(n):
            for i in range(2**n):
                expected = 0.0 if _variant_in_set(k + 1, i) else 1.0
                npt.assert_almost_equal(T[k, i, :].sum(), expected)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_zero_where_variant_already_in_set(self, n):
        sigma = np.full((n, n), 0.5)
        T = construct_transition_matrix(sigma)
        for k in range(n):
            for i in range(2**n):
                if _variant_in_set(k + 1, i):
                    npt.assert_array_equal(T[k, i, :], np.zeros(2**n))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_zero_where_not_superset(self, n):
        # T[k, i, j] = 0 if i is not a subset of j
        sigma = np.full((n, n), 0.5)
        T = construct_transition_matrix(sigma)
        for k in range(n):
            for i in range(2**n):
                for j in range(2**n):
                    if not _is_subset(i, j):
                        npt.assert_almost_equal(T[k, i, j], 0.0)

    def test_known_values_n1(self):
        # T[0] = [[(1-s), s], [0, 0]]
        s = 0.6
        T = construct_transition_matrix([[s]])
        expected = np.array([[[1 - s, s], [0.0, 0.0]]])
        npt.assert_array_almost_equal(T, expected)

    def test_dtype_is_respected(self):
        sigma = np.full((2, 2), 0.5)
        T = construct_transition_matrix(sigma, dtype=np.float32)
        assert T.dtype == np.float32

    def test_returns_ndarray(self):
        T = construct_transition_matrix([[0.5]])
        assert isinstance(T, np.ndarray)


# ──────────────────────────────────────────
# shift_matrix
# ──────────────────────────────────────────


class TestShiftMatrix:

    @pytest.mark.parametrize("size, shift", [(4, 1), (8, 2), (4, 0)])
    def test_output_shape(self, size, shift):
        S = shift_matrix(size, shift)
        assert S.shape == (size, size)

    @pytest.mark.parametrize("size, shift", [(4, 1), (4, 2), (8, 3)])
    def test_equals_numpy_eye_with_negative_k(self, size, shift):
        npt.assert_array_equal(
            shift_matrix(size, shift), np.eye(size, k=-shift)
        )

    def test_shift_zero_is_identity(self):
        npt.assert_array_equal(shift_matrix(4, 0), np.eye(4))

    def test_shift_action_on_vector(self):
        # S @ [1,2,3,4] with shift=1 → [0,1,2,3]
        S = shift_matrix(4, 1)
        v = np.array([1.0, 2.0, 3.0, 4.0])
        npt.assert_array_equal(S @ v, [0.0, 1.0, 2.0, 3.0])

    def test_entries_are_zero_or_one(self):
        S = shift_matrix(6, 2)
        assert np.all((S == 0) | (S == 1))

    def test_row_sum_at_most_one(self):
        S = shift_matrix(5, 2)
        assert np.all(S.sum(axis=1) <= 1)

    def test_returns_ndarray(self):
        assert isinstance(shift_matrix(4, 1), np.ndarray)


# ──────────────────────────────────────────
# construct_matrices (Integration test)
# ──────────────────────────────────────────


class TestConstructMatrices:

    def test_returns_tuple_of_four(self):
        result = construct_matrices([1.0], [[0.5]])
        assert len(result) == 4

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_output_shapes(self, n):
        gamma = [1.0] * n
        sigma = np.full((n, n), 0.5)
        V, G, T, S = construct_matrices(gamma, sigma)
        size = 2**n
        assert V.shape == (n, size)
        assert G.shape == (size, size)
        assert T.shape == (n, size, size)
        assert S.shape == (n, size, size)

    def test_V_matches_variant_in_set_matrix(self):
        gamma = [1.0, 2.0]
        sigma = np.full((2, 2), 0.5)
        V, *_ = construct_matrices(gamma, sigma)
        npt.assert_array_equal(V, variant_in_set_matrix(2))

    def test_G_matches_compute_gamma_matrix(self):
        gamma = [1.0, 2.0]
        sigma = np.full((2, 2), 0.5)
        _, G, *_ = construct_matrices(gamma, sigma)
        npt.assert_array_equal(G, compute_gamma_matrix(gamma))

    def test_T_matches_construct_transition_matrix(self):
        gamma = [1.0, 2.0]
        sigma = np.full((2, 2), 0.5)
        _, _, T, _ = construct_matrices(gamma, sigma)
        npt.assert_array_equal(T, construct_transition_matrix(sigma))

    def test_S_matches_shift_matrices(self):
        n = 2
        gamma = [1.0, 2.0]
        sigma = np.full((n, n), 0.5)
        _, _, _, S = construct_matrices(gamma, sigma)
        size = 2**n
        for m in range(n):
            npt.assert_array_equal(S[m], shift_matrix(size, 2**m))

    def test_dtype_propagates_to_G_and_T(self):
        gamma = [1.0, 2.0]
        sigma = np.full((2, 2), 0.5)
        _, G, T, _ = construct_matrices(gamma, sigma, dtype=np.float32)
        assert G.dtype == np.float32
        assert T.dtype == np.float32


# ──────────────────────────────────────────
# construct_sparse_transition_matrices
# ──────────────────────────────────────────


class TestConstructSparseTransitionMatrices:

    @pytest.mark.parametrize(
        "sigma",
        [
            np.array([[0.6]]),
            np.array(
                [
                    [1.0, 0.25],
                    [0.75, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0, 0.0, 0.6],
                    [0.2, 1.0, 1.0],
                    [0.8, 0.4, 1.0],
                ]
            ),
        ],
        ids=["n1", "n2", "n3"],
    )
    def test_matches_dense_transition_matrices(self, sigma):
        n = sigma.shape[0]
        size = 2**n

        dense_transitions = construct_transition_matrix(sigma)
        sparse_transitions = construct_sparse_transition_matrices(sigma)

        assert isinstance(sparse_transitions, tuple)
        assert len(sparse_transitions) == n

        for k, transition in enumerate(sparse_transitions):
            assert isinstance(transition, csr_matrix)
            assert transition.shape == (size, size)
            assert np.all(transition.data != 0.0)

            npt.assert_allclose(
                transition.toarray(),
                dense_transitions[k],
                rtol=1e-12,
                atol=1e-14,
            )

    def test_rows_with_variant_already_present_are_zero(self):
        sigma = np.array(
            [
                [1.0, 0.3, 0.6],
                [0.4, 1.0, 0.2],
                [0.7, 0.5, 1.0],
            ]
        )
        transitions = construct_sparse_transition_matrices(sigma)
        states = np.arange(2 ** len(sigma))

        for k, transition in enumerate(transitions):
            bit = 1 << k
            rows_with_variant = states[(states & bit) != 0]

            assert transition[rows_with_variant, :].nnz == 0

    @pytest.mark.parametrize(
        "sigma",
        [
            np.array([0.2, 0.3]),
            np.ones((2, 3)),
        ],
    )
    def test_raises_value_error_for_non_square_sigma(self, sigma):
        with pytest.raises(ValueError, match="square matrix"):
            construct_sparse_transition_matrices(sigma)


# ──────────────────────────────────────────
# construct_sparse_operators
# ──────────────────────────────────────────


class TestConstructSparseOperators:

    def test_returns_expected_operator_data(self):
        gamma = np.array([0.1, 0.12, 0.08])
        sigma = np.array(
            [
                [1.0, 0.3, 0.6],
                [0.4, 1.0, 0.2],
                [0.7, 0.5, 1.0],
            ]
        )

        operators = construct_sparse_operators(gamma, sigma)
        n = len(gamma)
        size = 2**n
        expected_V = variant_in_set_matrix(n).astype(bool)

        assert isinstance(operators, SparseModelOperators)
        npt.assert_array_equal(operators.V, expected_V)
        npt.assert_array_equal(operators.gamma, gamma)
        npt.assert_allclose(
            operators.gamma_sum,
            expected_V.T @ gamma,
        )

        assert len(operators.source_rows) == n
        assert len(operators.target_rows) == n
        assert len(operators.transition_transposes) == n

        states = np.arange(size)

        for k in range(n):
            bit = 1 << k
            expected_source_rows = states[(states & bit) == 0]

            npt.assert_array_equal(
                operators.source_rows[k],
                expected_source_rows,
            )
            npt.assert_array_equal(
                operators.target_rows[k],
                expected_source_rows | bit,
            )

            assert np.all((operators.source_rows[k] & bit) == 0)
            assert np.all((operators.target_rows[k] & bit) != 0)
            assert isinstance(
                operators.transition_transposes[k],
                csr_matrix,
            )

    def test_transition_transposes_match_dense_transition_matrices(self):
        gamma = np.array([0.1, 0.12, 0.08])
        sigma = np.array(
            [
                [1.0, 0.3, 0.6],
                [0.4, 1.0, 0.2],
                [0.7, 0.5, 1.0],
            ]
        )

        dense_transitions = construct_transition_matrix(sigma)
        operators = construct_sparse_operators(gamma, sigma)

        for k, transition_transpose in enumerate(
            operators.transition_transposes
        ):
            npt.assert_allclose(
                transition_transpose.T.toarray(),
                dense_transitions[k],
                rtol=1e-12,
                atol=1e-14,
            )

    def test_raises_value_error_for_empty_gamma(self):
        with pytest.raises(ValueError, match="At least one variant"):
            construct_sparse_operators(
                np.array([]),
                np.empty((0, 0)),
            )

    def test_raises_value_error_for_incompatible_sigma_shape(self):
        with pytest.raises(ValueError, match="sigma must have shape"):
            construct_sparse_operators(
                np.array([0.1, 0.2]),
                np.ones((3, 3)),
            )
