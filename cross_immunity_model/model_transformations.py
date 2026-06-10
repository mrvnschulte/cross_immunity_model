"""
Defines functions to transform between model formulations, including
transformations between the vector and matrix formulation of the state, the
projection to susceptibles and infectious per variant and the projection onto
model states with a different number of variants.
"""

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .matrices import variant_in_set_matrix
from .utils import is_subset, number_of_variants


def map_x_to_I(x: ArrayLike, dtype: DTypeLike | None = None) -> np.ndarray:
    """
    Maps vector representation x to matrix representation I

    Parameters
    ----------
    x : array-like
        Vector representation of state.
    dtype : data-type, optional
        The desired data-type for the array.

    Returns
    -------
    I : np.ndarray
        Matrix representation of state.
    """
    n = number_of_variants(x, matrix_form=False)
    I = np.zeros((2**n, 2**n), dtype=dtype)
    counter = 0
    for j in range(2**n):
        for k in range(j + 1):
            if is_subset(k, j):
                I[k, j] = x[counter]
                counter += 1
    return I


def map_I_to_x(I: ArrayLike, dtype: DTypeLike | None = None) -> np.ndarray:
    """
    Maps matrix representation I back to vector representation x

    Parameters
    ----------
    I : array-like
        Matrix representation of state.
    dtype : data-type, optional
        The desired data-type for the array.

    Returns
    -------
    x : np.ndarray
        Vector representation of state.
    """
    n = number_of_variants(I, matrix_form=True)
    size = 2**n
    length_x = 3**n
    x = np.zeros(length_x, dtype=dtype)
    counter = 0
    for j in range(size):
        for k in range(j + 1):
            if is_subset(k, j):
                x[counter] = I[k, j]
                counter += 1
    return x


def susceptibles_per_variant(
    x: ArrayLike, V: np.ndarray | None = None, matrix_form: bool = False
) -> np.ndarray:
    """
    Returns the susceptibles per variant.

    Parameters
    ----------
    x : array-like
        State of the systen.
    V : np.ndarray, optional
        Variant-in-set matrix. If not given, it is computed. The default is
        None.
    matrix_form : bool, optional
        If True, the state is given in matrix form, else in vector form. The
        default is False.

    Returns
    -------
    S_variant : np.ndarray
        Susceptibles per variant.
    """
    n = number_of_variants(x, matrix_form=matrix_form)
    size = 2**n
    if V is None:
        V = variant_in_set_matrix(n)
    if matrix_form:
        I = x
    else:
        I = map_x_to_I(x)
    S_variant = (np.ones((size, 1)).T @ I @ (np.ones_like(V) - V).T).flatten()

    return S_variant


def infectious_per_variant(
    x: ArrayLike, V: np.ndarray | None = None, matrix_form: bool = False
) -> np.ndarray:
    """
    Returns the infectious fraction per variant.

    Parameters
    ----------
    x : ArrayLike
        State of the system.
    V : np.ndarray, optional
        Variant-in-set matrix. If not given, it is computed. The default is
        None.
    matrix_form : bool, optional
        If True, the state is given in matrix form, else in vector form. The
        default is False.

    Returns
    -------
    I_variant : np.ndarray
        Infectious fraction per variant.
    """
    n = number_of_variants(x, matrix_form=matrix_form)
    size = 2**n
    if V is None:
        V = variant_in_set_matrix(n)
    if matrix_form:
        I = x
    else:
        I = map_x_to_I(x)
    I_variant = (V @ I @ np.ones((size, 1))).flatten()

    return I_variant


def project_n_variants(
    I_variant_reduced: ArrayLike, n: int, indices: ArrayLike | None = None
) -> np.ndarray:
    """
    Project reduced I_variant onto full I_variant with n variants given
    indicated positions.

    Parameters
    ----------
    I_variant_reduced : array-like
        Reduced I_variant.
    n : int
        Number of variants in full system.
    indices : array-like or None, optional
        Positions of the reduced system in the full system. If None is given,
        the first positions are used. The default is None.

    Returns
    -------
    I_variant_full : np.ndarray
        Full I_variant.
    """
    if indices is None:
        indices = np.arange(len(I_variant_reduced))
    indices = np.array(indices)
    I_variant_full = np.zeros(n)
    I_variant_full[indices] = I_variant_reduced
    return I_variant_full


def reduce_variants(
    I_variant_full: ArrayLike, indices: ArrayLike | None = None
) -> np.ndarray:
    """
    Reduce full I_variant by using only indicated positions.

    Parameters
    ----------
    I_variant_full : array-like
        Full I_variant.
    indices : array-like or None, optional
        Positions of the variants to be used. If None is given, all are used.
        The default is None.

    Returns
    -------
    I_variant_reduced : np.ndarray
        Reduced I_variant.
    """
    I_variant_full = np.array(I_variant_full)
    if indices is not None:
        indices = np.array(indices)
        I_variant_reduced = I_variant_full[indices]
    else:
        I_variant_reduced = I_variant_full
    return I_variant_reduced
