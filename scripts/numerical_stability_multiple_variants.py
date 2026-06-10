import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from cross_immunity_model import (
    construct_matrices,
    eigenvalues_linearized_system,
    find_multi_variant_equilibrium,
    find_stable_multi_variant_equilibrium,
    infectious_per_variant,
    invasion_reproduction_number_multiple_variants,
    multi_variant_model_with_matrices,
)


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


def check_stable_variants_algorithm(
    N_parameters,
    n_variants,
    beta_mean,
    beta_std,
    gamma_mean,
    gamma_std,
    sigma_alpha,
    sigma_beta,
    b,
    filename,
    dpi,
):
    max_real_parts = []
    colors = []
    N_wrong = []
    correct_real_part = []
    # V = variant_in_set_matrix(n_variants)
    for i in range(N_parameters):
        print(f"Compute set {i}")
        beta, gamma, sigma = sample_random_parameters(
            n_variants,
            beta_mean,
            beta_std,
            gamma_mean,
            gamma_std,
            sigma_alpha,
            sigma_beta,
            seed=i,
        )
        coexisting_variants, I_variant_eq, x_eq, convergence = (
            find_stable_multi_variant_equilibrium(
                beta, gamma, sigma, b, verbosity_level=1, atol=1e-4
            )
        )
        # if not convergence == 1:
        #     max_real_parts.append(0)
        #     colors.append("tab:green")
        #     continue
        # inv_r0 = invasion_reproduction_number_multiple_variants(
        #     x_eq, beta, gamma, b, V
        # )
        eigvals, max_real_part, eigvects = eigenvalues_linearized_system(
            x_eq, beta, gamma, sigma, b
        )
        max_real_parts.append(max_real_part)
        # colors.append(
        #     "tab:orange"
        #     if np.min(I_variant_eq[coexisting_variants]) < 1e-4
        #     else "tab:blue"
        # )  # neu
        if max_real_part > 0:
            # colors.append("tab:orange")
            # if np.min(I_variant_eq[coexisting_variants]) < 1e-4:
            x0 = 0.01 * np.ones(3**n_variants)
            x0[0] = 1 - sum(x0[1:])
            V, G, T, S = construct_matrices(gamma, sigma)
            sol = solve_ivp(
                multi_variant_model_with_matrices,
                (0, 10000),
                x0,
                args=(beta, gamma, sigma, V, G, T, S, b),
            )
            I_variant = infectious_per_variant(sol.y.T[-1], V)
            I_variant = np.maximum(I_variant, 0)
            inv_r0 = invasion_reproduction_number_multiple_variants(
                x_eq, beta, gamma, b, V
            )
            print(f"Coexisting variants: {coexisting_variants}")
            print(f"I_variant computed: {I_variant_eq}")
            print(f"Invasion reproduction number: {inv_r0}")
            print(f"I_variant solution: {I_variant}")
            print(f"beta: {beta}")
            print(f"sigma: {sigma}")
            coexisting_variants = list(np.where(I_variant > 1e-5)[0])
            I_variant_new, I_eq, x_eq = find_multi_variant_equilibrium(
                beta,
                gamma,
                sigma,
                V,
                G,
                T,
                S,
                b,
                I_variant,
                coexisting_variants,
            )
            eigvals, max_real_part, eigvects = eigenvalues_linearized_system(
                x_eq, beta, gamma, sigma, b
            )
            inv_r0 = invasion_reproduction_number_multiple_variants(
                x_eq, beta, gamma, b, V
            )
            print(f"New I_variant: {I_variant_new}")
            print(f"Maximal real part of solution: {max_real_part}")
            print(f"Invasion reproduction number: {inv_r0}")
            N_wrong.append(i)
            correct_real_part.append(max_real_part)
            if list(np.where(I_variant_new > 1e-4)[0]) == coexisting_variants:
                colors.append("tab:orange")
            else:
                colors.append("tab:purple")
            # print(beta)
            # print(sigma)
            print("-" * 60)
        else:
            colors.append("tab:blue")
        #     print(f"Coexisting variants: {coexisting_variants}")
        #     print(f"I_variant: {I_variant_eq}")
    plt.scatter(range(N_parameters), max_real_parts, c=colors)
    plt.scatter(N_wrong, correct_real_part, c="tab:red")
    plt.axhline(color="grey", linestyle="dotted")
    plt.xlabel("Index of parameter set")
    plt.ylabel("Maximal real part of eigenvalues")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=dpi)
    plt.show()


beta_mean = 0.3
beta_std = 0.025
gamma_mean = 0.1
gamma_std = 0
sigma_alpha = 9
sigma_beta = 1
b = 0.01
N_parameters = 1000
check_variants = [3, 4, 5, 6]  # , 7, 8, 9, 10]
dpi = 500
verbose = True

for n_variants in check_variants:
    filename = f"../plots/Numerical_stability_{n_variants}_variants.jpg"
    check_stable_variants_algorithm(
        N_parameters,
        n_variants,
        beta_mean,
        beta_std,
        gamma_mean,
        gamma_std,
        sigma_alpha,
        sigma_beta,
        b,
        filename,
        dpi,
    )
    if verbose:
        print(f"Finished computation for {n_variants} variants.")
