"""
Defines functions to compute the state at given equilibrium as well as
functions to find the equilibrium and one computing the eigenvalues of the
linearized system at equilibrium to check for stability.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from .matrices import construct_matrices
from .model_transformations import (
    infectious_per_variant,
    map_I_to_x,
    project_n_variants,
    reduce_variants,
)
from .invasion_reproduction_number import (
    invasion_reproduction_number_multiple_variants,
)
from .symbolic_system import build_jacobian_function
from .utils import (
    basic_reproduction_number,
    hamming_weight_iterator,
    is_subset,
    number_of_variants,
    subsets_of,
)


def multi_variant_equilibrium_compartments(
    I_variant: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    V: np.ndarray,
    G: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    b: float,
) -> np.ndarray:
    """
    Computes the compartment sizes for multi-variant equilibria given the
    infectious fractions per variant.

    Parameters
    ----------
    I_variant : np.ndarray
        Infectious fraction per variant.
    beta : np.ndarray
        Array of transmission rates.
    gamma : np.ndarray
        Array of recovery rates.
    sigma : np.ndarray
        Array of cross-immunity factors.
    V : np.ndarray
        Variant-in-set matrix.
    G : np.ndarray
        Gamma matrix.
    T : np.ndarray
        Transmission matrix.
    S : np.ndarray
        Shift matrix.
    b : float
        Natural death and birth rate.

    Returns
    -------
    I_star : np.ndarray
        Compartment sizes.
    """
    n_variants = len(I_variant)
    size = 2**n_variants

    I_star = np.zeros((size, size))

    transmission_vector = beta * I_variant
    beta_sum = ((np.ones_like(V) - V).T @ (transmission_vector)).flatten()
    W = transmission_vector[:, np.newaxis, np.newaxis] * T
    gamma_sum = (V.T @ gamma).flatten()

    I_star[0, 0] = b / (gamma_sum[0] + beta_sum[0] + b)

    iteration_list = list(hamming_weight_iterator(size))

    for j in iteration_list:
        if j == 0:
            continue
        SIW = np.sum(S @ I_star @ W, axis=0)
        for i in reversed(iteration_list):
            if is_subset(i, j):
                GI_ij = G[i, :] @ I_star[:, j]
                I_star[i, j] = (SIW[i, j] + GI_ij) / (
                    gamma_sum[i] + beta_sum[j] + b
                )

    return I_star


def multi_variant_equilibrium_equation(
    I_variant: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    V: np.ndarray,
    G: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    b: float,
    coexisting_variants: list[int] | None = None,
    use_inv_formulation: bool = True,
) -> np.ndarray:
    """
    Computes the difference between the given I_variant and the computed one
    according to the compartment equilibrium equations for the given I_variant.

    The function can be used to solve for the equilibrium I_variant.

    Parameters
    ----------
    I_variant : np.ndarray
        Infectious fraction per variant.
    beta : np.ndarray
        Array of transmission rates.
    gamma : np.ndarray
        Array of recovery rates.
    sigma : np.ndarray
        Array of cross-immunity factors.
    V : np.ndarray
        Variant-in-set matrix.
    G : np.ndarray
        Gamma matrix.
    T : np.ndarray
        Transmission matrix.
    S : np.ndarray
        Shift matrix.
    b : float
        Natural death and birth rate.
    coexisting_variants : list of int, optional
        List of variants that should be present in equilibrium. Only used, if
        use_inv_formulation is True. If none is given, all variants are used.
    use_inv_formulation : bool, optional
        If True, the equilibrium is computed by setting the invasion
        reproduction number of coexisting variants to 1. The default is True.

    Returns
    -------
    diff : np.ndarray
        Difference between given and computed I_variant or invasion
        reproduction number of coexisting variants and 1, depending on
        use_inv_formulation.
    """
    I_star = multi_variant_equilibrium_compartments(
        I_variant, beta, gamma, sigma, V, G, T, S, b
    )
    if not use_inv_formulation:
        I_star_variant = infectious_per_variant(I_star, V, matrix_form=True)
        diff = I_star_variant - I_variant
    else:
        if coexisting_variants is None:
            coexisting_variants = list(range(len(beta)))
        x_eq = map_I_to_x(I_star)
        inv_r0 = invasion_reproduction_number_multiple_variants(
            x_eq, beta, gamma, b, V
        )
        diff = inv_r0[coexisting_variants] - 1
    return diff


def solve_multi_variant_equilibrium_equation(
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    V: np.ndarray,
    G: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    b: float,
    I_variant_guess: ArrayLike,
    free_indices: ArrayLike | None = None,
    bounds: tuple[ArrayLike, ArrayLike] = (0, 1),
    atol: float = 1e-4,
    use_inv_formulation: bool = True,
) -> np.ndarray:
    """
    Solves the multi-variant equilibrium equation with given initial guess and
    optionally also the set of free variants and bounds.

    Parameters
    ----------
    beta : np.ndarray
        Array of transmission rates.
    gamma : np.ndarray
        Array of recovery rates.
    sigma : np.ndarray
        Array of cross-immunity factors.
    V : np.ndarray
        Variant-in-set matrix.
    G : np.ndarray
        Gamma matrix.
    T : np.ndarray
        Transmission matrix.
    S : np.ndarray
        Shift matrix.
    b : float
        Natural death and birth rate.
    I_variant_guess : array-like
        Initial guess of equilibrium I_variant.
    free_indices : array-like or None, optional
        Indices of the variables which could exist. If None is given, all are
        used. The default is None.
    bounds : tuple[ArrayLike, ArrayLike], optional
        Bounds on the equilibrium I_variant. The default is (0, 1).
    atol : float, optional
        Tolerance for least square solver. The default is 1e-4.
    use_inv_formulation : bool, optional
        If True, the equilibrium is computed by setting the invasion
        reproduction number of coexisting variants to 1. The default is True.

    Returns
    -------
    I_variant : np.ndarray
        Equilibrium I_variant.
    """
    n = len(I_variant_guess)

    lb, ub = bounds
    if not np.isscalar(lb):
        lb = reduce_variants(lb, free_indices)
    if not np.isscalar(ub):
        ub = reduce_variants(ub, free_indices)
    bounds = (lb, ub)

    def multi_variant_equilibrium_equation_reduced(I_variant_free):
        I_variant_full = project_n_variants(I_variant_free, n, free_indices)

        if use_inv_formulation:
            return multi_variant_equilibrium_equation(
                I_variant_full, beta, gamma, sigma, V, G, T, S, b, free_indices
            )
        else:
            result = multi_variant_equilibrium_equation(
                I_variant_full,
                beta,
                gamma,
                sigma,
                V,
                G,
                T,
                S,
                b,
                use_inv_formulation=use_inv_formulation,
            )
            return reduce_variants(result, free_indices)

    I_variant_guess = reduce_variants(I_variant_guess, free_indices)

    res = least_squares(
        multi_variant_equilibrium_equation_reduced,
        I_variant_guess,
        bounds=bounds,
        ftol=atol,
        # xtol=atol,
        # gtol=atol,
    )
    I_variant = project_n_variants(res.x, n, free_indices)

    return I_variant


def find_multi_variant_equilibrium(
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    V: np.ndarray,
    G: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    b: float,
    I_variant_guess: ArrayLike,
    free_indices: ArrayLike | None = None,
    bounds: tuple[ArrayLike, ArrayLike] = (0, 1),
    atol: float = 1e-4,
    use_inv_formulation: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves the multi-variant equilibrium equation with given initial guess and
    optionally also the set of free variants and bounds and computes the state
    at equilibrium.

    Parameters
    ----------
    beta : np.ndarray
        Array of transmission rates.
    gamma : np.ndarray
        Array of recovery rates.
    sigma : np.ndarray
        Array of cross-immunity factors.
    V : np.ndarray
        Variant-in-set matrix.
    G : np.ndarray
        Gamma matrix.
    T : np.ndarray
        Transmission matrix.
    S : np.ndarray
        Shift matrix.
    b : float
        Natural death and birth rate.
    I_variant_guess : array-like
        Initial guess of equilibrium I_variant.
    free_indices : array-like or None, optional
        Indices of the variables which could exist. If None is given, all are
        used. The default is None.
    bounds : tuple[ArrayLike, ArrayLike], optional
        Bounds on the equilibrium I_variant. The default is (0, 1).
    atol : float, optional
        Tolerance for least square solver. The default is 1e-4.
    use_inv_formulation : bool, optional
        If True, the equilibrium is computed by setting the invasion
        reproduction number of coexisting variants to 1. The default is True.


    Returns
    -------
    I_variant : np.ndarray
        Equilibrium I_variant.
    I_eq : np.ndarray
        Equilibrium state in matrix form.
    x_eq : np.ndarray
        Equilibrium state in vector form.
    """
    if free_indices is None or len(free_indices) == 0:
        I_variant = np.zeros(len(beta))
    else:
        I_variant = solve_multi_variant_equilibrium_equation(
            beta,
            gamma,
            sigma,
            V,
            G,
            T,
            S,
            b,
            I_variant_guess,
            free_indices=free_indices,
            bounds=bounds,
            atol=atol,
            use_inv_formulation=use_inv_formulation,
        )
    I_eq = multi_variant_equilibrium_compartments(
        I_variant, beta, gamma, sigma, V, G, T, S, b
    )
    x_eq = map_I_to_x(I_eq)

    return I_variant, I_eq, x_eq


def check_stability_of_variants(
    coexisting_variants: ArrayLike,
    candidate_variants: ArrayLike,
    beta: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
    b: float,
    V: np.ndarray | None = None,
    G: np.ndarray | None = None,
    T: np.ndarray | None = None,
    S: np.ndarray | None = None,
    atol: float = 1e-4,
    verbosity_level: int = 0,
    use_inv_formulation: bool = True,
    _I_variant_prev: np.ndarray | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray, int]:
    """
    Checks the stability of the equilibrium with coexisting variants, while
    possibly adding candidate variants.

    Parameters
    ----------
    coexisting_variants : array-like
        List of variants (by index) that are coexisting in equilibrium.
    candidate_variants : array-like
        List of variants (by index) that can potentially be added to
        equilibrium.
    beta : np.ndarray
        Array of transmission rates.
    gamma : np.ndarray
        Array of recovery rates.
    sigma : np.ndarray
        Array of cross-immunity factors.
    b : float
        Natural birth and death rate.
    V : np.ndarray | None, optional
        Variant-in-set matrix. If a matrix is not given, they are computed. The
        default is None.
    G : np.ndarray | None, optional
        Gamma matrix. If a matrix is not given, they are computed. The default
        is None.
    T : np.ndarray | None, optional
        Transmission matrix. If a matrix is not given, they are computed. The
        default is None.
    S : np.ndarray | None, optional
        Shift matrix. If a matrix is not given, they are computed. The default
        is None.
    atol : float, optional
        Tolerance for invasion reproduction number of coexisting variants not
        being exactly 1. The default is 1e-4.
    verbosity_level : int, optional
        Verbosity level of the function. For
            - 0: No output
            - 1: Warnings are displayed if equilibrium of coexisting variants
                 is not found.
            - 2: Additionally, convergence message is printed.
            - 3: Additionally, the set of coexisting variants and the invasion
                 reproduction numbers at their equilibrium are printed.
        The default is 0.
    use_inv_formulation : bool, optional
        If True, the equilibrium is computed by setting the invasion
        reproduction number of coexisting variants to 1. The default is True.
    _I_variant_prev : np.ndarray | None, optional
        Warm-start vector from the parent recursive call. The known entries
        (coexisting variants from the previous level) are reused as the
        initial guess, and the new candidate entry is initialised to a small
        positive value. If None, the standard unit guess is used. Internal
        parameter – not intended to be set by the caller.

    Returns
    -------
    stable_coexisting_variants: list[int]
        List of coexisting variants in stable equilibrium.
    I_variant_eq : np.ndarray
        Equilibrium I_variant.
    x_eq : np.ndarray
        Equilibrium state in vector form.
    convergence : int
        Convergence signal. For
            - -1: Equilibrium of coexisting variants has not been found
            - 0: There are no more candidate variants (with invasion
                 reproduction greater than 1), but equilibrium is not stable
            - 1: Equilibrium is stable
    """
    if any(m is None for m in (V, G, T, S)):
        V, G, T, S = construct_matrices(gamma, sigma)

    # Find corresponding equilibrium and compute invasion reproduction number
    n_variants = len(beta)
    if _I_variant_prev is not None:
        I_variant_guess = _I_variant_prev.copy()
    else:
        I_variant_guess = np.zeros(n_variants)
        I_variant_guess[coexisting_variants] = 0.05

    I_variant_eq, I_eq, x_eq = find_multi_variant_equilibrium(
        beta,
        gamma,
        sigma,
        V,
        G,
        T,
        S,
        b,
        I_variant_guess,
        free_indices=coexisting_variants,
        atol=atol,
        use_inv_formulation=use_inv_formulation,
    )
    inv_r0 = invasion_reproduction_number_multiple_variants(
        x_eq, beta, gamma, b, V
    )

    if verbosity_level >= 3:
        print(
            f"Checking stability of equilibrium with variants: {coexisting_variants}"
        )
        print(f"Invasion reproduction number: {inv_r0}")

    # Check if correct equilibrium has been found
    if not np.allclose(inv_r0[coexisting_variants], 1, atol=atol):
        convergence = -1
        if verbosity_level >= 1:
            print(
                f"Warning: Equilibrium not found for coexisting variants {coexisting_variants}"
            )
        return coexisting_variants, I_variant_eq, x_eq, convergence

    # Check if equilibrium is stable
    other_variants = list(
        map(
            int,
            np.setdiff1d(
                np.arange(n_variants), coexisting_variants, assume_unique=True
            ),
        )
    )
    if np.all(inv_r0[other_variants] < 1):
        convergence = 1
        if verbosity_level >= 2:
            print(
                f"Stable equilibrium found with variants: {coexisting_variants}"
            )
        return coexisting_variants, I_variant_eq, x_eq, convergence

    # If it is not stable, sort variants by invasion reproduction number and
    # add them to equilibrium in this order, checking each for stability
    candidate_variants = sorted(
        candidate_variants,
        key=lambda i: inv_r0[i],
        reverse=True,
    )
    if len(candidate_variants) == 0 or inv_r0[candidate_variants[0]] < 1:
        convergence = 0
        return coexisting_variants, I_variant_eq, x_eq, convergence
    for i, candidate in enumerate(candidate_variants):
        if inv_r0[candidate] < 1:
            break
        new_coexisting_variants = coexisting_variants + [candidate]
        new_coexisting_variants.sort()
        new_candidate_variants = candidate_variants[i + 1 :]

        warm_start = I_variant_eq.copy()
        warm_start[candidate] = 0.05

        stable_coexisting_variants, I_variant_eq, x_eq, convergence = (
            check_stability_of_variants(
                new_coexisting_variants,
                new_candidate_variants,
                beta,
                gamma,
                sigma,
                b,
                V,
                G,
                T,
                S,
                atol=atol,
                verbosity_level=verbosity_level,
                use_inv_formulation=use_inv_formulation,
                _I_variant_prev=warm_start,
            )
        )
        if convergence == 1:
            break

    return stable_coexisting_variants, I_variant_eq, x_eq, convergence


def find_stable_multi_variant_equilibrium(
    beta: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
    b: float,
    V: np.ndarray | None = None,
    G: np.ndarray | None = None,
    T: np.ndarray | None = None,
    S: np.ndarray | None = None,
    atol: float = 1e-4,
    verbosity_level: int = 0,
    use_inv_formulation: bool = True,
    ordered_check: bool = True,
) -> tuple[list[int], np.ndarray, np.ndarray, int]:
    """
    Finds the set of coexisting variants in the stable equilibrium.

    Parameters
    ----------
    beta : ArrayLike
        Vector of transmission rates.
    gamma : ArrayLike
        Vector of recovery rates.
    sigma : ArrayLike
        Matrix of cross-immunity factors.
    b : float
        Natural birth and death rate.
    V : np.ndarray | None, optional
        Variant-in-set matrix. If a matrix is not given, they are computed. The
        default is None.
    G : np.ndarray | None, optional
        Gamma matrix. If a matrix is not given, they are computed. The default
        is None.
    T : np.ndarray | None, optional
        Transmission matrix. If a matrix is not given, they are computed. The
        default is None.
    S : np.ndarray | None, optional
        Shift matrix. If a matrix is not given, they are computed. The default
        is None.
    atol : float, optional
        Tolerance for invasion reproduction number of equilibrium variants not
        being exactly zero. The default is 1e-2.
    verbosity_level : int, optional
        Verbosity level of the function. For
            - 0: No output
            - 1: Warnings are displayed if equilibrium of coexisting variants
                 is not found.
            - 2: Additionally, convergence message is printed.
            - 3: Additionally, the set of coexisting variants and the invasion
                 reproduction numbers at their equilibrium are printed.
        The default is 0.
    use_inv_formulation : bool, optional
        If True, the equilibrium is computed by setting the invasion
        reproduction number of coexisting variants to 1. The default is True.
    ordered_check : bool, optional
        If True, subsets are checked by depth first search in order of
        ascending (invasion) reproduction number. If False, subsets are checked
        in order of ascending number of variants. The default is True.

    Returns
    -------
    coexisting_variants : list[int]
        List of coexisting variants in stable equilibrium.
    I_variant_eq : np.ndarray
        Infectious fraction per variant at equilibrium.
    x_eq : np.ndarray
        State at equilibrium.
    convergence : int
        Convergence signal. For
            - -1: Equilibrium of coexisting variants has not been found
            - 0: There are no more candidate variants (with invasion
                 reproduction greater than 1), but equilibrium is not stable
            - 1: Equilibrium is stable

    """
    if any(m is None for m in (V, G, T, S)):
        V, G, T, S = construct_matrices(gamma, sigma)

    if ordered_check:
        coexisting_variants, I_variant_eq, x_eq, convergence = (
            check_stability_of_variants(
                [],
                list(map(int, np.arange(len(beta)))),
                beta,
                gamma,
                sigma,
                b,
                V,
                G,
                T,
                S,
                atol=atol,
                verbosity_level=verbosity_level,
                use_inv_formulation=use_inv_formulation,
            )
        )

        return coexisting_variants, I_variant_eq, x_eq, convergence

    else:
        r0 = basic_reproduction_number(beta, gamma, b)
        n_variants = len(beta)
        if np.all(r0 < 1):
            coexisting_variants = []
            I_variant_eq = np.zeros(n_variants)
            x_eq = multi_variant_equilibrium_compartments(
                I_variant_eq, beta, gamma, sigma, V, G, T, S, b
            )
            convergence = 1
            return coexisting_variants, I_variant_eq, x_eq, convergence
        else:
            I_variant_guess = 0.1 * np.ones(n_variants)
            for subset in subsets_of(np.arange(n_variants)):
                I_variant_eq, I_eq, x_eq = find_multi_variant_equilibrium(
                    beta,
                    gamma,
                    sigma,
                    V,
                    G,
                    T,
                    S,
                    b,
                    I_variant_guess,
                    free_indices=subset,
                    atol=atol,
                    use_inv_formulation=use_inv_formulation,
                )
                inv_r0 = invasion_reproduction_number_multiple_variants(
                    x_eq, beta, gamma, b, V
                )

                if not np.allclose(inv_r0[subset], 1, atol=atol):
                    continue

                other_variants = list(
                    map(
                        int,
                        np.setdiff1d(
                            np.arange(n_variants),
                            subset,
                            assume_unique=True,
                        ),
                    )
                )
                if np.all(inv_r0[other_variants] < 1):
                    convergence = 1
                    return subset, I_variant_eq, x_eq, convergence
            I_variant_eq, I_eq, x_eq = find_multi_variant_equilibrium(
                beta,
                gamma,
                sigma,
                V,
                G,
                T,
                S,
                b,
                I_variant_guess,
                free_indices=list(np.arange(n_variants)),
                atol=atol,
                use_inv_formulation=use_inv_formulation,
            )
            convergence = 1
            return list(np.arange(n_variants)), I_variant_eq, x_eq, convergence


def eigenvalues_linearized_system(
    x_eq: ArrayLike,
    beta: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
    b: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Returns the eigenvalues, its maximal real part and the eigenvectors of the
    linearized system at the equilibrium x_eq.

    Parameters
    ----------
    x_eq : array-like
        State at equilibrium.
    beta : array-like
        Vector of transmission rates of the variants.
    gamma : array-like
        Vector of recovery rates of the variants.
    sigma : list of lists, or np.ndarray
        Matrix of cross-immunity factors.
    b : float
        Natural death and birth rate.

    Returns
    -------
    eigvals : np.ndarray
        Eigenvalues of the linearized system at the equilibrium.
    max_real_part : float
        Maximal real part of the eigenvalues of the linearized system.
    eigvects : np.ndarray
        Eigenvectors of the linearized system at the equilibrium.
    """
    n_variants = number_of_variants(x_eq)
    sigma = np.array(sigma)

    J_func = build_jacobian_function(n_variants)
    args = list(x_eq) + list(beta) + list(gamma) + list(sigma.flatten()) + [b]
    J_num = np.array(J_func(*args), dtype=float)
    eigvals, eigvects = np.linalg.eig(J_num)
    max_real_part = np.max(eigvals.real)

    return eigvals, max_real_part, eigvects
