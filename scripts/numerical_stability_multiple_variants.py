"""
This script checks whether the algorithm to compute the coexisting variants
really finds a stable equilibrium.
"""

import matplotlib.pyplot as plt

from cross_immunity_model import (
    eigenvalues_linearized_system,
    find_stable_multi_variant_equilibrium,
    sample_random_parameters,
)


def check_stable_variants_algorithm(
    N_parameters: int,
    n_variants: int,
    beta_mean: float,
    beta_std: float,
    gamma_mean: float,
    gamma_std: float,
    sigma_alpha: float,
    sigma_beta: float,
    b: float,
    filename: str | None = None,
    dpi: int = 500,
    verbose_step: int | None = None,
):
    """
    Check whether the stable-equilibrium algorithm converges correctly.

    Generates ``N_parameters`` independent parameter sets for a model with
    ``n_variants`` variants and, for each set:

    1. Samples ``(beta, gamma, sigma)`` via ``sample_random_parameters``.
    2. Locates the stable equilibrium with
       ``find_stable_multi_variant_equilibrium``.
    3. Computes the eigenvalues of the Jacobian at that equilibrium via
       ``eigenvalues_linearized_system``.
    4. Records the maximal real part of those eigenvalues.

    A scatter plot of the maximal real parts against the parameter-set index
    is produced. All maximal real parts should be strictly negative if the
    algorithm converges to a stable equilibrium in every case.

    Parameters
    ----------
    N_parameters : int
        Number of parameter sets to be generated.
    n_variants : int
        Number of variants in the model.
    beta_mean : float
        Mean of the transmission rate.
    beta_std : float
        Standard deviation of the transmission rate.
    gamma_mean : float
        Mean of the recovery rate.
    gamma_std : float
        Standard deviation of the recovery rate.
    sigma_alpha : float
        Alpha parameter of the Beta distribution for cross-immunity factors.
    sigma_beta : float
        Beta parameter of the Beta distribution for cross-immunity factors.
    b : float
        Natural birth and death rate.
    filename : str or None, optional
        Path at which to save the figure. If None the figure is not saved.
        The default is None.
    dpi : int, optional
        Resolution in dots per inch used when saving the figure.
        The default is 500.
    verbose_step : int or None, optional
        If not None, a progress message is printed every ``verbose_step``
        iterations. The default is None.

    Returns
    -------
    None

    Side Effects
    ------------
    Displays a scatter plot via ``plt.show()``.
    Writes a JPEG file to ``filename`` when ``filename`` is not None.
    Prints a warning to stdout for each parameter set where
    ``find_stable_multi_variant_equilibrium`` does not converge
    (``convergence != 1``).

    Notes
    -----
    A maximal real part strictly below zero confirms asymptotic stability of
    the found equilibrium. Values at or above zero indicate either that the
    algorithm converged to an unstable equilibrium or that numerical errors
    are present.

    The ``i``-th parameter set is seeded with ``seed=i``, so results are
    fully reproducible.
    """
    max_real_parts = []
    for i in range(N_parameters):
        if verbose_step is not None:
            if (i + 1) % verbose_step == 0:
                print(f"Compute set {i+1}/{N_parameters}")
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
                beta, gamma, sigma, b, atol=1e-4
            )
        )
        if convergence != 1:
            print(
                f"Note: For seed {i} a stable equilibrium has not been found, probably due to not finding the correct equilibrium"
            )
        eigvals, max_real_part, eigvects = eigenvalues_linearized_system(
            x_eq, beta, gamma, sigma, b
        )
        max_real_parts.append(max_real_part)
    plt.scatter(range(N_parameters), max_real_parts)
    plt.axhline(color="grey", linestyle="dotted")
    plt.xlabel("Index of parameter set")
    plt.ylabel("Maximal real part of eigenvalues")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=dpi)
    plt.show()


# Parameter values
beta_mean = 0.3
beta_std = 0.025
gamma_mean = 0.1
gamma_std = 0
sigma_alpha = 9
sigma_beta = 1
b = 0.01
N_parameters = 1000
check_variants = [3, 4, 5, 6]
dpi = 500
savefig = False
verbose = True
verbose_step = 10

# Main part
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
        filename=filename,
        dpi=dpi,
        verbose_step=verbose_step,
    )
    if verbose:
        print(f"Finished computation for {n_variants} variants.")
