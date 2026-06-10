"""
Defines utility functions for the analysis and computation of the multi-variant
model, including:
    - basic_reproduction_number
    - check_validity
    - hamming_weight_iterator
    - is_subset
    - number_of_variants
    - random_simplex_points
    - subsets_of
    - variant_in_set
"""

from collections.abc import Iterator
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike


def basic_reproduction_number(
    beta: ArrayLike, gamma: ArrayLike, b: float = 0.0
) -> np.ndarray:
    """
    Returns the basic reproduction numbers for all variants.

    Parameters
    ----------
    beta : array-like
        Vector of transmission rates of the variants.
    gamma : array-like
        Vector of recovery rates of the variants.
    b : float, optional
        Natural death and birth rate. The default is 0.0.

    Returns
    -------
    r0 : np.ndarray
        Vector of the basic reproduction numbers of the variants.
    """
    beta = np.array(beta)
    gamma = np.array(gamma)
    r0 = beta / (gamma + b)
    return r0


def check_validity(x: ArrayLike) -> bool:
    """
    Checks biological feasibility of a state x.

    Parameters
    ----------
    x : array-like
        State of the system.

    Returns
    -------
    bool
        True, if x is biologically feasible. False, otherwise.
    """
    x = np.array(x)
    return bool(np.all((0 <= x) & (x <= 1)))


def hamming_weight_iterator(size: int) -> Iterator[int]:
    """
    Iterate over all integers in [0, size) ordered by ascending Hamming weight.

    Within the same Hamming weight, numbers are yielded in ascending order.

    Parameters
    ----------
    size : int
        Upper bound (exclusive) of the integer range to iterate over.

    Yields
    ------
    int
        Next integer in [0, size), ordered by number of set bits (1s in
        binary representation).

    Examples
    --------
    >>> list(hamming_weight_iterator(8))
    [0, 1, 2, 4, 3, 5, 6, 7]

    The order corresponds to grouping by Hamming weight:

    - weight 0: 0 (0b000)
    - weight 1: 1 (0b001), 2 (0b010), 4 (0b100)
    - weight 2: 3 (0b011), 5 (0b101), 6 (0b110)
    - weight 3: 7 (0b111)
    """
    for weight in range(size.bit_length()):
        for j in range(size):
            if bin(j).count("1") == weight:
                yield j


def is_subset(i: int, j: int) -> bool:
    """
    Checks whether i is a subset of j using binary representations.

    Parameters
    ----------
    i : int
        Binary (integer) representation of set i.
    j : int
        Binary (integer) representation of set j.

    Returns
    -------
    bool
        True, if i is a subset of j.
    """
    return i & j == i


def number_of_variants(x: ArrayLike, matrix_form: bool = False) -> int:
    """
    Returns the number of variants for a given state.

    Parameters
    ----------
    x : ArrayLike
        State of the system.
    matrix_form : bool, optional
        If True, the state is given in matrix form, else in vector form. The
        default is False.

    Returns
    -------
    n : int
        Number of variants of the system.
    """
    if matrix_form:
        base = 2
    else:
        base = 3
    n = int(round(np.log(len(x)) / np.log(base)))
    return n


def random_simplex_points(
    n: int, dim: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Generate random points on the standard simplex according to Dirichlet
    distribution.

    Parameters
    ----------
    n : int
        Number of points to generate.
    dim : int
        Dimension of the state.
    rng : np.random.Generator, optional
        Random number generator. If None is given, it is initialized

    Returns
    -------
    mats : np.ndarray
        Array of n random points on the simplex.
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha = np.ones(dim)
    mats = rng.dirichlet(alpha, size=n)
    return mats


def subsets_of(indices: ArrayLike) -> list:
    """
    Returns the list of subsets of indices, excluding the empty and the full
    set.

    Parameters
    ----------
    indices : array-like
        Iterable of indices to find subsets of.

    Returns
    -------
    subsets : list
        List of subsets of indices, excluding the empty and the full set.
    """
    subsets = [
        list(map(int, comb))
        for r in range(1, len(indices) + 1)
        for comb in combinations(indices, r)
    ]
    return subsets


def variant_in_set(k: int, i: int) -> bool:
    """
    Checks whether variant k is in set i.

    Parameters
    ----------
    k : int
        Index of variant k.
    i : int
        Binary (integer) representation of set i.

    Returns
    -------
    bool
        True, if variant k is in set i.
    """
    return (i & (1 << (k - 1))) != 0
