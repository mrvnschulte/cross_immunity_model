"""
The possible equilibria of a three-variant model are computed and it is checked
whether other variants can invade the equilibrium and whether this is stable.
"""

import numpy as np

from cross_immunity_model import (
    basic_reproduction_number,
    construct_matrices,
    eigenvalues_linearized_system,
    find_multi_variant_equilibrium,
    invasion_reproduction_number_multiple_variants,
)


beta = [0.35, 0.3, 0.15]
gamma = [0.1, 0.1, 0.1]
sigma_12 = 0.5
sigma_23 = 0.41
b = 0.01
I_variant_guess = [0.4, 0.2, 0.1]
bounds = ([0, 0, 0], [1, 1, 1])
atol = 1e-4

sigma = np.array(
    [
        [1, sigma_12, sigma_12 * sigma_23],
        [sigma_12, 1, sigma_23],
        [sigma_12 * sigma_23, sigma_23, 1],
    ]
)
V, G, T, S = construct_matrices(gamma, sigma)
n_variants = len(beta)

variant_combinations = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]

for equilibrium_variants in variant_combinations:
    equilibrium_variants = np.array(equilibrium_variants)
    I_variant, I_eq, x_eq = find_multi_variant_equilibrium(
        beta,
        gamma,
        sigma,
        V,
        G,
        T,
        S,
        b,
        I_variant_guess,
        free_indices=equilibrium_variants,
        bounds=bounds,
    )
    print("-" * 60 + "\n")
    if np.any(np.isclose(I_variant[equilibrium_variants], 0, atol=atol)):
        print(f"There is no {equilibrium_variants+1}-equilibrium\n\n")
        continue

    r0 = basic_reproduction_number(beta, gamma, b)
    inv_r0 = invasion_reproduction_number_multiple_variants(
        x_eq, beta, gamma, b, V
    )
    for i in range(n_variants):
        if not i in equilibrium_variants:
            inv_r0_i = inv_r0[i]
            print(f"R0_{i+1}to{equilibrium_variants+1}={inv_r0_i}")
            if np.isclose(inv_r0_i, 1, atol=atol):
                print(
                    f"    -> {i+1} might invade the {equilibrium_variants+1}-equilibrium (unclear due to numerical errors)\n"
                )

            elif inv_r0_i > 1 + atol:
                print(
                    f"    -> {i+1} can invade the {equilibrium_variants+1}-equilibrium\n"
                )
            else:
                print(
                    f"    -> {i+1} cannot invade the {equilibrium_variants+1}-equilibrium\n"
                )

    eigvals, max_real_part, _ = eigenvalues_linearized_system(
        x_eq, beta, gamma, sigma, b
    )
    # print(f"Eigenvalues: {eigvals}")
    print(f"Maximal real part of eigenvalues: {max_real_part}")
    if max_real_part > 0:
        print(
            f"    -> The {equilibrium_variants+1}-equilibrium is unstable.\n\n"
        )
    else:
        print(
            f"    -> The {equilibrium_variants+1}-equilibrium is stable.\n\n"
        )
