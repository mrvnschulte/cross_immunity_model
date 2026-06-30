"""
Tests for utils.py
"""

import numpy as np
import numpy.testing as npt
import pytest

from cross_immunity_model.utils import (
    basic_reproduction_number,
    check_validity,
    hamming_weight_iterator,
    is_subset,
    number_of_variants,
    subsets_of,
    variant_in_set,
)


# ──────────────────────────────────────────
# basic_reproduction_number
# ──────────────────────────────────────────


class TestBasicReproductionNumber:

    def test_known_value_scalar(self):
        # beta=0.3, gamma=0.1, b=0 → R0 = 3.0
        npt.assert_almost_equal(basic_reproduction_number([0.3], [0.1]), [3.0])

    def test_known_value_with_b(self):
        # beta=0.3, gamma=0.1, b=0.1 → R0 = 0.3/0.2 = 1.5
        npt.assert_almost_equal(
            basic_reproduction_number([0.3], [0.1], b=0.1), [1.5]
        )

    def test_multiple_variants(self):
        npt.assert_array_almost_equal(
            basic_reproduction_number([0.3, 0.4], [0.1, 0.2]),
            [3.0, 2.0],
        )

    def test_returns_ndarray(self):
        assert isinstance(basic_reproduction_number([0.3], [0.1]), np.ndarray)

    def test_b_zero_default(self):
        """b=0 default matches explicit b=0."""
        npt.assert_array_almost_equal(
            basic_reproduction_number([0.3], [0.1]),
            basic_reproduction_number([0.3], [0.1], b=0.0),
        )

    def test_output_length_matches_input(self):
        r0 = basic_reproduction_number([0.1, 0.2, 0.3], [0.05, 0.1, 0.15])
        assert len(r0) == 3

    def test_r0_greater_one_when_beta_gt_gamma(self):
        r0 = basic_reproduction_number([0.5], [0.1])
        assert r0[0] > 1.0

    def test_r0_less_one_when_beta_lt_gamma(self):
        r0 = basic_reproduction_number([0.05], [0.1])
        assert r0[0] < 1.0

    def test_r0_equal_one_when_beta_equals_gamma(self):
        npt.assert_almost_equal(
            basic_reproduction_number([0.1], [0.1])[0], 1.0
        )

    def test_higher_b_reduces_r0(self):
        r0_low = basic_reproduction_number([0.3], [0.1], b=0.01)
        r0_high = basic_reproduction_number([0.3], [0.1], b=0.1)
        assert r0_high[0] < r0_low[0]

    def test_elementwise_formula(self):
        """R0 = beta / (gamma + b) elementwise."""
        beta = np.array([0.2, 0.5, 0.8])
        gamma = np.array([0.1, 0.2, 0.4])
        b = 0.05
        npt.assert_array_almost_equal(
            basic_reproduction_number(beta, gamma, b),
            beta / (gamma + b),
        )


# ──────────────────────────────────────────
# check_validity
# ──────────────────────────────────────────


class TestCheckValidity:

    def test_valid_state_returns_true(self):
        assert check_validity([0.0, 0.5, 1.0]) is True

    def test_all_zeros_is_valid(self):
        assert check_validity([0.0, 0.0, 0.0]) is True

    def test_all_ones_is_valid(self):
        assert check_validity([1.0, 1.0, 1.0]) is True

    def test_unit_state_is_valid(self):
        x = [0.1, 0.2, 0.3, 0.4]
        assert check_validity(x) is True

    def test_negative_value_is_invalid(self):
        assert check_validity([-0.01, 0.5, 0.5]) is False

    def test_value_exceeds_one_is_invalid(self):
        assert check_validity([0.5, 1.01, 0.0]) is False

    def test_all_negative_is_invalid(self):
        assert check_validity([-1.0, -0.5]) is False

    def test_returns_bool(self):
        assert isinstance(check_validity([0.5]), bool)

    def test_ndarray_input(self):
        assert check_validity(np.array([0.2, 0.3, 0.5])) is True

    @pytest.mark.parametrize("v", [0.0, 0.5, 1.0])
    def test_boundary_values_are_valid(self, v):
        assert check_validity([v]) is True

    @pytest.mark.parametrize("v", [-1e-10, 1.0 + 1e-10])
    def test_outside_boundary_is_invalid(self, v):
        assert check_validity([v]) is False


# ──────────────────────────────────────────
# hamming_weight_iterator
# ──────────────────────────────────────────


class TestHammingWeightIterator:

    def test_docstring_example(self):
        assert list(hamming_weight_iterator(8)) == [0, 1, 2, 4, 3, 5, 6, 7]

    def test_size_one(self):
        assert list(hamming_weight_iterator(1)) == [0]

    def test_size_two(self):
        # weight 0: 0; weight 1: 1
        assert list(hamming_weight_iterator(2)) == [0, 1]

    def test_size_four(self):
        # weight 0: [0]; weight 1: [1,2]; weight 2: [3]
        assert list(hamming_weight_iterator(4)) == [0, 1, 2, 3]

    def test_output_length(self):
        assert len(list(hamming_weight_iterator(8))) == 8

    @pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
    def test_is_permutation_of_range(self, size):
        assert sorted(hamming_weight_iterator(size)) == list(range(size))

    def test_non_decreasing_hamming_weight(self):
        """Hamming weight must be non-decreasing along the iteration."""
        prev_weight = 0
        for j in hamming_weight_iterator(16):
            w = bin(j).count("1")
            assert w >= prev_weight
            prev_weight = w

    def test_within_same_weight_ascending_order(self):
        """Within each Hamming-weight group, values must be ascending."""
        from itertools import groupby

        groups = groupby(
            hamming_weight_iterator(16), key=lambda j: bin(j).count("1")
        )
        for _, group in groups:
            vals = list(group)
            assert vals == sorted(vals)

    def test_returns_tuple(self):
        assert isinstance(hamming_weight_iterator(4), tuple)

    def test_zero_is_first(self):
        assert (hamming_weight_iterator(8))[0] == 0


# ──────────────────────────────────────────
# is_subset
# ──────────────────────────────────────────


class TestIsSubset:

    def test_empty_set_is_subset_of_everything(self):
        """0 (empty set) is a subset of every set."""
        for j in range(8):
            assert is_subset(0, j) is True

    def test_set_is_subset_of_itself(self):
        for i in range(8):
            assert is_subset(i, i) is True

    def test_known_true_cases(self):
        assert is_subset(1, 3) is True  # 001 ⊆ 011
        assert is_subset(2, 3) is True  # 010 ⊆ 011
        assert is_subset(3, 7) is True  # 011 ⊆ 111
        assert is_subset(5, 7) is True  # 101 ⊆ 111

    def test_known_false_cases(self):
        assert is_subset(3, 1) is False  # 011 ⊄ 001
        assert is_subset(3, 2) is False  # 011 ⊄ 010
        assert is_subset(7, 3) is False  # 111 ⊄ 011
        assert is_subset(6, 5) is False  # 110 ⊄ 101

    def test_returns_bool(self):
        assert isinstance(is_subset(0, 1), bool)

    def test_antisymmetry(self):
        """i ⊆ j and j ⊆ i iff i == j."""
        for i in range(8):
            for j in range(8):
                if is_subset(i, j) and is_subset(j, i):
                    assert i == j

    def test_transitivity(self):
        """i ⊆ j and j ⊆ k implies i ⊆ k."""
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    if is_subset(i, j) and is_subset(j, k):
                        assert is_subset(i, k)

    def test_superset_not_subset(self):
        """If j strictly contains i, j is not a subset of i."""
        assert is_subset(3, 1) is False
        assert is_subset(7, 1) is False

    def test_disjoint_sets_not_subsets(self):
        """Disjoint non-empty sets are not subsets of each other."""
        assert is_subset(1, 2) is False  # 001 and 010 disjoint
        assert is_subset(2, 1) is False


# ──────────────────────────────────────────
# number_of_variants
# ──────────────────────────────────────────


class TestNumberOfVariants:

    # ── vector form (base 3) ──────────────

    @pytest.mark.parametrize("n, x_len", [(1, 3), (2, 9), (3, 27)])
    def test_vector_form(self, n, x_len):
        assert number_of_variants(np.zeros(x_len), matrix_form=False) == n

    def test_vector_form_default(self):
        """matrix_form defaults to False."""
        assert number_of_variants(np.zeros(9)) == 2

    # ── matrix form (base 2) ─────────────

    @pytest.mark.parametrize("n, x_len", [(1, 2), (2, 4), (3, 8)])
    def test_matrix_form(self, n, x_len):
        assert (
            number_of_variants(np.zeros((x_len, x_len)), matrix_form=True) == n
        )

    # ── return type ──────────────────────

    def test_returns_int(self):
        assert isinstance(number_of_variants(np.zeros(9)), int)

    # ── list input ───────────────────────

    def test_list_input(self):
        assert number_of_variants([0, 0, 0]) == 1

    # ── consistency between forms ────────

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_vector_and_matrix_form_consistent(self, n):
        assert number_of_variants(np.zeros(3**n), matrix_form=False) == n
        assert (
            number_of_variants(np.zeros((2**n, 2**n)), matrix_form=True) == n
        )


# ──────────────────────────────────────────
# subsets_of
# ──────────────────────────────────────────


class TestSubsetsOf:

    def test_single_element(self):
        """subsets_of([0]) = [[0]] (only the singleton; empty excluded)."""
        assert subsets_of([0]) == [[0]]

    def test_two_elements(self):
        result = subsets_of([0, 1])
        assert result == [[0], [1], [0, 1]]

    def test_three_elements(self):
        result = subsets_of([0, 1, 2])
        expected = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        assert result == expected

    def test_count_for_n_elements(self):
        """subsets_of(n elements) → 2^n - 1 subsets (no empty set)."""
        for n in [1, 2, 3, 4]:
            assert len(subsets_of(list(range(n)))) == 2**n - 1

    def test_returns_list(self):
        assert isinstance(subsets_of([0, 1]), list)

    def test_each_subset_is_list(self):
        for subset in subsets_of([0, 1, 2]):
            assert isinstance(subset, list)

    def test_each_element_is_int(self):
        for subset in subsets_of([0, 1, 2]):
            assert all(isinstance(x, int) for x in subset)

    def test_no_empty_subset(self):
        for subset in subsets_of([0, 1, 2]):
            assert len(subset) > 0

    def test_full_set_is_last(self):
        result = subsets_of([0, 1, 2])
        assert result[-1] == [0, 1, 2]

    def test_singletons_are_first_n(self):
        result = subsets_of([0, 1, 2])
        assert result[:3] == [[0], [1], [2]]

    def test_empty_input_returns_empty_list(self):
        assert subsets_of([]) == []

    def test_preserves_input_values(self):
        """Non-zero-based indices must be preserved."""
        result = subsets_of([3, 5])
        assert [3] in result
        assert [5] in result
        assert [3, 5] in result

    def test_ordering_by_size(self):
        """All size-k subsets appear before size-(k+1) subsets."""
        result = subsets_of([0, 1, 2])
        sizes = [len(s) for s in result]
        assert sizes == sorted(sizes)


# ──────────────────────────────────────────
# variant_in_set
# ──────────────────────────────────────────


class TestVariantInSet:

    def test_variant_1_in_set_1(self):
        # i=1 (0b001): variant 1 (bit 0) is set
        assert variant_in_set(1, 1) is True

    def test_variant_1_not_in_set_2(self):
        # i=2 (0b010): bit 0 not set
        assert variant_in_set(1, 2) is False

    def test_variant_2_in_set_2(self):
        # i=2 (0b010): variant 2 (bit 1) is set
        assert variant_in_set(2, 2) is True

    def test_variant_2_not_in_set_1(self):
        assert variant_in_set(2, 1) is False

    def test_variant_1_in_set_3(self):
        # i=3 (0b011): both bits set
        assert variant_in_set(1, 3) is True

    def test_variant_2_in_set_3(self):
        assert variant_in_set(2, 3) is True

    def test_variant_3_in_set_4(self):
        # i=4 (0b100): bit 2 set → variant 3
        assert variant_in_set(3, 4) is True

    def test_variant_3_not_in_set_3(self):
        # i=3 (0b011): bits 0,1 set; bit 2 not set → variant 3 absent
        assert variant_in_set(3, 3) is False

    def test_variant_not_in_empty_set(self):
        """No variant is in the empty set (i=0)."""
        for k in range(1, 5):
            assert variant_in_set(k, 0) is False

    def test_all_variants_in_full_set(self):
        """For n=3, full set = 7 (0b111): all variants 1,2,3 present."""
        for k in range(1, 4):
            assert variant_in_set(k, 7) is True

    def test_returns_bool(self):
        assert isinstance(variant_in_set(1, 1), bool)

    def test_consistency_with_is_subset(self):
        """variant_in_set(k, i) ↔ is_subset(2^(k-1), i)."""
        for k in range(1, 5):
            for i in range(16):
                assert variant_in_set(k, i) == is_subset(2 ** (k - 1), i)

    def test_consistency_with_variant_in_set_matrix(self):
        """Must agree with entry V[k-1, i] from variant_in_set_matrix."""
        from cross_immunity_model.matrices import variant_in_set_matrix

        n = 3
        V = variant_in_set_matrix(n)
        for k in range(1, n + 1):
            for i in range(2**n):
                assert variant_in_set(k, i) == bool(V[k - 1, i])
