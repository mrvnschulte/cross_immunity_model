"""
Creates random parameters and initial conditions for the cross-immunity model.
"""

import numpy as np


def sample_symmetric_beta_matrix(
    n: int, alpha: float, beta: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Samples a symmetic matrix of size n times n, where the main diagonal
    entries are 1 and the off-diagonal entries are sampled from a beta
    distribution with parameters alpha and beta.

    Parameters
    ----------
    n : int
        Size of the matrix.
    alpha : float
        Shape parameter alpha of beta distribution.
    beta : float
        Shape parameter beta of beta distribution.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    A : np.ndarray
        Matrix of size n times n with mentioned properties.
    """
    upper = np.zeros((n, n))
    idx = np.triu_indices(n, k=1)
    upper[idx] = rng.beta(a=alpha, b=beta, size=len(idx[0]))
    A = upper + upper.T
    np.fill_diagonal(A, 1.0)

    return A


def sample_random_parameters(
    n_variants: int,
    beta_mean: float = 0.2,
    beta_std: float = 0.01,
    gamma_mean: float = 0.1,
    gamma_std: float = 0.01,
    sigma_alpha: float = 1.0,
    sigma_beta: float = 1.0,
    rng: np.random.Generator = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Samples random parameters for cross immunity model with n variants.

    Beta and gamma parameters are sampled according to a normal distribution
    and sigma parameters are sampled according to a beta distribution.

    Parameters
    ----------
    n_variants : int
        Number of variants.
    beta_mean : float, optional
        Mean of normal distribution for beta. The default is 0.2.
    beta_std : float, optional
        Standard deviation of normal distribution for beta. The default is
        0.01.
    gamma_mean : float, optional
        Mean of normal distribution for gamma. The default is 0.1.
    gamma_std : float, optional
        Standard deviation of normal distribution for gamma. The default is
        0.01.
    sigma_alpha : float, optional
        Shape parameter alpha of beta distribution for sigma. The default is
        1.0.
    sigma_beta : float, optional
        Shape parameter beta of beta distribution for sigma. The default is
        1.0.
    rng : np.random.Generator, optional
        Random number generator. If none is given, it is generated. The default
        is None.
    seed : int | None, optional
        Seed of random number generator. Is only used, when no random number
        generator is passed. The default is None.

    Returns
    -------
    beta : np.ndarray
        Vector of transmission rates.
    gamma : np.ndarrray
        Vector of recovery rates.
    sigma : np.ndarray
        Matrix of cross-immunity factors.
    """
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    beta = rng.normal(loc=beta_mean, scale=beta_std, size=n_variants)
    gamma = rng.normal(loc=gamma_mean, scale=gamma_std, size=n_variants)
    sigma = sample_symmetric_beta_matrix(
        n_variants, sigma_alpha, sigma_beta, rng
    )

    return beta, gamma, sigma


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
