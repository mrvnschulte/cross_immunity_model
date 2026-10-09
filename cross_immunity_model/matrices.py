"""
Defines functions to construct the matrices corresponding to the matrix-vector
formulation of the model.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from scipy.sparse import csr_matrix

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


@dataclass(frozen=True)
class SparseModelOperators:
    """
    Stores the matrices and indices used in the sparse formulation of the
    multi-variant model.

    Attributes
    ----------
    V : np.ndarray
        Variant-in-set matrix.
    gamma : np.ndarray
        Vector of recovery rates of the variants.
    gamma_sum : np.ndarray
        Sum of recovery rates for each infection set.
    source_rows : tuple[np.ndarray, ...]
        Indices of infection sets not containing the corresponding variant.
    target_rows : tuple[np.ndarray, ...]
        Indices of infection sets obtained by adding the corresponding variant.
    transition_transposes : tuple[csr_matrix, ...]
        Transposed sparse transition matrices.
    """

    V: np.ndarray
    gamma: np.ndarray
    gamma_sum: np.ndarray
    source_rows: tuple[np.ndarray, ...]
    target_rows: tuple[np.ndarray, ...]
    transition_transposes: tuple[csr_matrix, ...]


def construct_sparse_transition_matrices(
    sigma: ArrayLike,
) -> tuple[csr_matrix, ...]:
    """
    Constructs sparse transition matrices based on infections.

    Parameters
    ----------
    sigma : array-like
        Matrix of cross-immunity factors.

    Returns
    -------
    T : tuple[csr_matrix, ...]
        Sparse transition matrices.
    """
    sigma = np.asarray(sigma, dtype=float)

    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be a square matrix.")

    n = sigma.shape[0]
    size = 2**n
    states = np.arange(size, dtype=np.intp)
    transitions = []

    for k in range(n):
        bit_k = 1 << k
        rows = []
        cols = []
        data = []

        # T[k, i, j] can only be non-zero if k is not in i
        for protection_source in states[(states & bit_k) == 0]:
            # All protection_target states j for which i is a subset of j
            protection_target = states[
                (states & protection_source) == protection_source
            ]

            # Since protection_source is a subset of protection_target:
            # XOR equals the set difference used in the dense implementation
            newly_added = protection_target ^ protection_source

            values = np.ones(protection_target.size, dtype=float)

            for m in range(n):
                bit_m = 1 << m

                added_m = (newly_added & bit_m) != 0
                absent_m = (protection_target & bit_m) == 0

                values[added_m] *= sigma[k, m]
                values[absent_m] *= 1.0 - sigma[k, m]

            keep = values != 0.0

            if np.any(keep):
                rows.append(
                    np.full(
                        np.count_nonzero(keep),
                        protection_source,
                        dtype=np.intp,
                    )
                )
                cols.append(protection_target[keep])
                data.append(values[keep])

        if rows:
            transition = csr_matrix(
                (
                    np.concatenate(data),
                    (
                        np.concatenate(rows),
                        np.concatenate(cols),
                    ),
                ),
                shape=(size, size),
            )
        else:
            transition = csr_matrix((size, size), dtype=float)

        transitions.append(transition)

    return tuple(transitions)


def construct_sparse_operators(
    gamma: ArrayLike,
    sigma: ArrayLike,
) -> SparseModelOperators:
    """
    Constructs all operators that are used in the sparse formulation of the
    multi-variant model.

    Parameters
    ----------
    gamma : array-like
        Vector of recovery rates of the variants.
    sigma : array-like
        Matrix of cross-immunity factors.

    Returns
    -------
    operators : SparseModelOperators
        Operators used in the sparse formulation of the multi-variant model.
    """
    gamma = np.asarray(gamma, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    n = len(gamma)

    if n < 1:
        raise ValueError("At least one variant is required.")

    if sigma.shape != (n, n):
        raise ValueError("sigma must have shape (len(gamma), len(gamma)).")

    size = 2**n
    states = np.arange(size, dtype=np.intp)

    V = variant_in_set_matrix(n).astype(bool)

    # Rows without variant k and the corresponding rows after infection with k
    source_rows = tuple(states[(states & (1 << k)) == 0] for k in range(n))
    target_rows = tuple(
        source | (1 << k) for k, source in enumerate(source_rows)
    )

    transitions = construct_sparse_transition_matrices(sigma)

    # The RHS uses T[k].T @ I[source].T, so transpose once here
    transition_transposes = tuple(
        transition.T.tocsr() for transition in transitions
    )

    return SparseModelOperators(
        V=V,
        gamma=gamma,
        gamma_sum=V.T @ gamma,
        source_rows=source_rows,
        target_rows=target_rows,
        transition_transposes=transition_transposes,
    )
