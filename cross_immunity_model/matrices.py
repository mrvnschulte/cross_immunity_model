"""
Defines functions to construct the matrices corresponding to the matrix-vector
formulation of the model.
"""

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .utils import is_subset, variant_in_set


def variant_in_set_matrix(n: int) -> np.ndarray:
    """
    Returns the variant in set matrix of n variants.

    Parameters
    ----------
    n : int
        Number of variants.

    Returns
    -------
    V : np.ndarray
        Variant in set matrix.
    """
    V = np.zeros((n, 2**n), dtype=int)
    indices = np.arange(2**n)
    for i in range(n):
        V[i] = (indices & (1 << i)) != 0
    return V


def compute_gamma_matrix(
    gamma: ArrayLike, dtype: DTypeLike | None = None
) -> np.ndarray:
    """
    Computes the gamma matrix based on the gamma vector.

    Parameters
    ----------
    gamma : array-like
        Vector of recovery rates.
    dtype : data-type or None, optional
        The desired data-type of the array. The default is None.

    Returns
    -------
    G : np.ndarray
        Gamma matrix.
    """
    n = len(gamma)
    if n == 1:
        G = np.array([[0, gamma[0]], [0, 0]], dtype=dtype)
    else:
        G_prev = compute_gamma_matrix(gamma[:-1], dtype=dtype)
        G = np.block(
            [
                [G_prev, gamma[-1] * np.eye(G_prev.shape[0], dtype=dtype)],
                [np.zeros_like(G_prev), G_prev],
            ]
        )

    return G


def construct_transition_matrix(
    sigma: ArrayLike, dtype: DTypeLike | None = None
) -> np.ndarray:
    """
    Construct the transition matrix based on infections.

    Parameters
    ----------
    sigma : array-like
        Matrix of cross-immunities.
    dtype : data-type or None, optional
        The desired data-type for the array. The default is None.

    Returns
    -------
    T : np.ndarray
        Transition matrix.
    """
    n = len(sigma)
    size = 2**n
    T = np.zeros((n, size, size), dtype=dtype)
    sigma = np.array(sigma)
    for k in range(n):
        for i in range(size):
            if not variant_in_set(k + 1, i):
                for j in range(size):
                    if is_subset(i, j):
                        T[k, i, j] = 1
                        for m in range(n):
                            if variant_in_set(m + 1, j - i):
                                T[k, i, j] *= sigma[k, m]
                            if not variant_in_set(m + 1, j):
                                T[k, i, j] *= 1 - sigma[k, m]
    return T


def shift_matrix(size: int, shift: int) -> np.ndarray:
    """
    Returns the shift matrix of given size and shift.

    Parameters
    ----------
    size : int
        Size of the matrix.
    shift : int
        Shift of the matrix.

    Returns
    -------
    S : np.ndarray
        Shift matrix of given size and shift.
    """
    S = np.eye(size, k=-shift)
    return S


def construct_matrices(
    gamma: ArrayLike, sigma: ArrayLike, dtype: DTypeLike | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs all matrices that are used in the matrix-formulation of the
    multi-variant model.

    Parameters
    ----------
    gamma : array-like
        Vector of recovery rates of the variants.
    sigma : array-like
        Matrix of cross-immunity factors.
    dtype : data-type | None, optional
        The desired data-type for the array.

    Returns
    -------
    V : np.ndarray
        Variant in set matrix.
    G : np.ndarray
        Gamma matrix.
    T : np.ndarray
        Transition matrix.
    S : np.ndarray
        Combined shift matrices.
    """
    n = len(gamma)
    size = 2**n
    V = variant_in_set_matrix(n)
    G = compute_gamma_matrix(gamma, dtype)
    T = construct_transition_matrix(sigma, dtype)
    S = np.array([shift_matrix(size, 2**m) for m in range(n)])

    return V, G, T, S
