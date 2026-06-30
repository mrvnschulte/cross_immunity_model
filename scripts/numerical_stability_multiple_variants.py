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
    sample_random_parameters,
)


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
    verbose,
    verbose_step,
):
    max_real_parts = []
    colors = []
    N_wrong = []
    correct_real_part = []
    for i in range(N_parameters):
        if verbose:
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
savefig = False
verbose = True
verbose_step = 10

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
        verbose=verbose,
        verbose_step=verbose_step,
    )
    if verbose:
        print(f"Finished computation for {n_variants} variants.")
