"""
Defines the multi-variant model using naive and matrix-vector formulations.
"""

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .matrices import (
    compute_gamma_matrix,
    construct_transition_matrix,
    shift_matrix,
    SparseModelOperators,
    variant_in_set_matrix,
)
from .model_transformations import (
    infectious_per_variant,
    map_I_to_x,
    map_x_to_I,
    susceptibles_per_variant,
)
from .utils import (
    is_subset,
    number_of_variants,
    variant_in_set,
)


def multi_variant_model(
    t: float,
    x: ArrayLike,
    beta: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
    b: float = 0.0,
    dtype: DTypeLike | None = None,
) -> np.ndarray:
    """
    Computes the derivatives of a multi-variant model with cross-immunity.

    This function computes the derivatives of a multi-variant SIR-like model
    with cross-immunity.

    Parameters
    ----------
    t : float
        The current time (not used).
    x : array-like
        The current state of the system.
    beta : array-like
        Vector of transmission rates of the variants.
    gamma : array-like
        Vector of recovery rates of the variants.
    sigma : list of lists, or np.ndarray
        Matrix of cross-immunity factors.
    b : float, optional
        Natural death and birth rate. The default is 0.0.

    Returns
    -------
    dx : np.ndarray
        The derivatives of the state.
    """
    # Compute number of variants and number of protection/infection groups
    n = number_of_variants(x)
    size = 2**n

    # Transform state vector into corresponding matrix
    I = map_x_to_I(x, dtype=dtype)

    # Compute the infectious with each variant
    V = variant_in_set_matrix(n)
    I_variant = infectious_per_variant(I, V, matrix_form=True)
    # Compute the recoveries for each infection set
    gamma_sum = (V.T @ gamma).flatten()
    # Compute the transmission factor for each variant
    transmission_vector = beta * I_variant
    # Compute sum of transmission rates of all susceptible variables
    beta_sum = ((np.ones_like(V) - V).T @ (transmission_vector)).flatten()
    # Compute the entry matrix due to recovery
    gamma_matrix = compute_gamma_matrix(gamma)
    # Compute transitions due to infections
    T = construct_transition_matrix(sigma, dtype=dtype)
    S = np.array([shift_matrix(size, 2**m) for m in range(n)])
    W = transmission_vector[:, np.newaxis, np.newaxis] * T

    # Sum up all transitions and retransform to state
    dI = (
        -(gamma_sum * I.T).T
        - beta_sum * I
        + gamma_matrix @ I
        + np.sum(S @ I @ W, axis=0)
        - b * I
    )
    dI[0, 0] += b
    dx = map_I_to_x(dI, dtype=dtype)

    return dx


def multi_variant_model_for_loops(
    t: float,
    x: ArrayLike,
    beta: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
    b: float = 0.0,
) -> np.ndarray:
    """
    Computes the derivatives of a multi-variant model with cross-immunity.

    This function computes the derivatives of a multi-variant SIR-like model
    with cross-immunity.

    Parameters
    ----------
    t : float
        The current time (not used).
    x : array-like
        The current state of the system.
    beta : array-like
        Vector of transmission rates of the variants.
    gamma : array-like
        Vector of recovery rates of the variants.
    sigma : list of lists, or np.ndarray
        Matrix of cross-immunity factors.
    b : float, optional
        Natural death and birth rate. The default is 0.0.

    Returns
    -------
    dx : np.ndarray
        The derivatives of the state.
    """
    # Compute number of variants and number of protection/infection groups
    n = number_of_variants(x)
    size = 2**n

    # Transform state vector into corresponding matrix
    I = map_x_to_I(x)
    dI = np.zeros_like(I)

    # Compute infectious per variant
    I_variant = np.zeros(n)
    for k in range(n):
        for i in range(size):
            if variant_in_set(k + 1, i):
                for j in range(size):
                    if is_subset(i, j):
                        I_variant[k] += I[i, j]

    # Compute transition matrix of infections
    sigma_trans = np.ones((n, size, size))
    for k in range(n):
        for i in range(size):
            if not variant_in_set(k + 1, i):
                for j in range(size):
                    if variant_in_set(k + 1, j) and is_subset(i, j):
                        for m in range(n):
                            if m != k:
                                if variant_in_set(
                                    m + 1, j
                                ) and not variant_in_set(m + 1, i):
                                    sigma_trans[k, i, j] *= sigma[k, m]
                                if not variant_in_set(m + 1, j):
                                    sigma_trans[k, i, j] *= 1 - sigma[k, m]

    # Compute sum of all transitions
    for i in range(size):
        for j in range(size):
            if is_subset(i, j):
                for k in range(n):
                    if variant_in_set(k + 1, i):
                        dI[i, j] -= gamma[k] * I[i, j]
                        i_minus_k = i ^ (2**k)
                        for l in range(size):
                            if (
                                is_subset(i_minus_k, l)
                                and is_subset(l, j)
                                and not variant_in_set(k + 1, l)
                            ):
                                dI[i, j] += (
                                    sigma_trans[k, l, j]
                                    * beta[k]
                                    * I[i_minus_k, l]
                                    * I_variant[k]
                                )
                    if not variant_in_set(k + 1, j):
                        dI[i, j] -= beta[k] * I[i, j] * I_variant[k]
                    if not variant_in_set(k + 1, i) and variant_in_set(
                        k + 1, j
                    ):
                        i_plus_k = i ^ (2**k)
                        dI[i, j] += gamma[k] * I[i_plus_k, j]
                dI[i, j] -= b * I[i, j]

    dI[0, 0] += b
    dx = map_I_to_x(dI)

    return dx


def multi_variant_model_with_matrices(
    t: float,
    x: ArrayLike,
    beta: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
    V: np.ndarray,
    G: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    b: float = 0.0,
    compute_dn_variant: bool = False,
    compute_dn: bool = False,
) -> np.ndarray:
    """
    Computes the derivatives of a multi-variant model with cross-immunity.

    This function computes the derivatives of a multi-variant SIR-like model
    with cross-immunity.

    Parameters
    ----------
    t : float
        The current time (not used).
    x : array-like
        The current state of the system.
    beta : array-like
        Vector of transmission rates of the variants.
    gamma : array-like
        Vector of recovery rates of the variants.
    sigma : list of lists, or np.ndarray
        Matrix of cross-immunity factors.
    V : np.ndarray
        Variant-in-set matrix.
    G : np.ndarray
        Gamma matrix.
    T : np.ndarray
        Transmission matrix.
    S : np.ndarray
        Shift matrix.
    b : float, optional
        Natural death and birth rate. The default is 0.0.
    compute_dn_variant : bool, optional
        If True, the total number of infections per variant is computed. The
        default is False.
    compute_dn : bool, optional
        If True, the total number of infections is computed. The default is
        False.

    Returns
    -------
    dx : np.ndarray
        The derivatives of the state.
    """
    if compute_dn:
        I = map_x_to_I(x[:-1])
    elif compute_dn_variant:
        I = map_x_to_I(x[: -len(beta)])
    else:
        I = map_x_to_I(x)

    # Compute the infectious with each variant
    I_variant = infectious_per_variant(I, V, matrix_form=True)
    # Compute the recoveries for each infection set
    gamma_sum = (V.T @ gamma).flatten()
    # Compute the transmission factor for each variant
    transmission_vector = beta * I_variant
    # Compute sum of transmission rates of all susceptible variables
    beta_sum = ((np.ones_like(V) - V).T @ (transmission_vector)).flatten()
    # Compute transitions due to infections
    W = transmission_vector[:, np.newaxis, np.newaxis] * T

    # Sum up all transitions and retransform to state
    dI = (
        -(gamma_sum * I.T).T
        - beta_sum * I
        + G @ I
        + np.sum(S @ I @ W, axis=0)
        - b * I
    )
    dI[0, 0] += b
    dx = map_I_to_x(dI)

    # Compute derivative of total number of infections
    if compute_dn_variant:
        S_variant = susceptibles_per_variant(I, V, matrix_form=True)
        dn = beta * S_variant * I_variant
        dx = np.concatenate([dx, dn])

    if compute_dn:
        S_variant = susceptibles_per_variant(I, V, matrix_form=True)
        dn = np.sum(beta * S_variant * I_variant)
        dx = np.append(dx, dn)

    return dx


def multi_variant_model_sparse(
    t: float,
    x: ArrayLike,
    beta: ArrayLike,
    operators: SparseModelOperators,
    b: float = 0.0,
) -> np.ndarray:
    """
    Numerical multi-variant model using sparse transition matrices.

    gamma and sigma are contained in operators and must remain unchanged
    during integration.
    """
    beta = np.asarray(beta, dtype=float)

    I = map_x_to_I(x)

    # Equivalent to V @ I @ 1, but avoids an unnecessary matrix product
    I_variant = operators.V @ I.sum(axis=1)

    transmission_vector = beta * I_variant
    beta_sum = (~operators.V).T @ transmission_vector

    # Losses due to recovery, infection, and demographics
    dI = -I * (
        operators.gamma_sum[:, np.newaxis] + beta_sum[np.newaxis, :] + b
    )

    for k in range(len(beta)):
        source_rows = operators.source_rows[k]
        target_rows = operators.target_rows[k]

        # Equivalent to the dense term G @ I:
        # recovery removes variant k from the infectious-set row
        dI[source_rows, :] += operators.gamma[k] * I[target_rows, :]

        # Equivalent to:
        # S[k] @ I @ (transmission_vector[k] * T[k])
        #
        # (T[k].T @ I[source].T).T equals I[source] @ T[k]
        incoming = np.asarray(
            operators.transition_transposes[k] @ I[source_rows, :].T
        ).T

        dI[target_rows, :] += transmission_vector[k] * incoming

    dI[0, 0] += b

    return map_I_to_x(dI)
