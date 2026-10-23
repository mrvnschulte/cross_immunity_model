"""
Demonstrates a stable endemic equilibrium in a three-variant model that does
not contain the most transmissible variant.

The script:

1. Simulates a three-variant cross-immunity model.
2. Computes the equilibrium containing variants 1 and 2 only.
3. Verifies that variant 3, despite having the largest transmission rate,
   cannot invade this equilibrium.
4. Checks local stability using the eigenvalues of the linearized system.
5. Plots all compartments and the infectious fractions per variant.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from cross_immunity_model import (
    construct_matrices,
    eigenvalues_linearized_system,
    find_multi_variant_equilibrium,
    infectious_per_variant,
    invasion_reproduction_number_multiple_variants,
    multi_variant_model_with_matrices,
)


def plot_compartments(
    solution,
    filename: str | None = None,
) -> None:
    """
    Plot the trajectories of all model compartments.

    Parameters
    ----------
    solution : scipy.integrate.OdeResult
        Numerical solution returned by ``solve_ivp``.
    filename : str or None, optional
        Path at which to save the figure. If None, the figure is not saved.
        The default is None.
    """
    plt.figure()

    plt.plot(solution.t, solution.y.T)

    plt.xlabel(r"Time $t$", fontsize=fontsize_text)
    plt.ylabel("Fraction of the population", fontsize=fontsize_text)
    plt.xlim(t0, tf)
    plt.ylim(bottom=0)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid(alpha=alpha_grid)

    if filename is not None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_infectious_fractions(
    solution,
    infectious_fractions: np.ndarray,
    filename: str | None = None,
) -> None:
    """
    Plot infectious fractions for all variants.

    Parameters
    ----------
    solution : scipy.integrate.OdeResult
        Numerical solution returned by ``solve_ivp``.
    infectious_fractions : np.ndarray
        Array whose columns contain infectious fractions for each variant.
    filename : str or None, optional
        Path at which to save the figure. If None, the figure is not saved.
        The default is None.
    """
    plt.figure()

    plt.plot(
        solution.t,
        infectious_fractions,
        label=[r"Variant 1", r"Variant 2", r"Variant 3"],
    )

    plt.xlabel(r"Time $t$", fontsize=fontsize_text)
    plt.ylabel("Infectious fraction", fontsize=fontsize_text)
    plt.xlim(t0, tf)
    plt.ylim(bottom=0)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid(alpha=alpha_grid)
    plt.legend(fontsize=fontsize_ticks)

    if filename is not None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close()


# Plot settings
dpi = 1200
fontsize_text = 20
fontsize_ticks = 16
alpha_grid = 0.7

plot_folder = "../plots"
fileformat = "pdf"

# Model parameters
beta = np.array([0.30329949, 0.30299893, 0.30850361])
gamma = np.array([0.1, 0.1, 0.1])
b = 0.01
sigma = np.array(
    [
        [1.0, 0.79653764, 0.95123338],
        [0.79653764, 1.0, 0.98846650],
        [0.95123338, 0.98846650, 1.0],
    ]
)

n_variants = len(beta)
t0 = 0.0
tf = 1000.0

# The equilibrium is contains variants 1 and 2 only
equilibrium_variants = np.array([0, 1])
excluded_variant = 2

# Numerical tolerances
atol = 1e-4
stability_tol = 1e-8

# Construct matrices and initial state
V, G, T, S = construct_matrices(gamma, sigma)
x0 = 0.001 * np.ones(3**n_variants)
x0[0] = 1.0 - np.sum(x0[1:])

# Simulate the full three-variant model
solution = solve_ivp(
    multi_variant_model_with_matrices,
    (t0, tf),
    x0,
    t_eval=np.linspace(t0, tf, 1001),
    args=(beta, gamma, sigma, V, G, T, S, b),
)

if not solution.success:
    raise RuntimeError(f"Simulation failed: {solution.message}")

infectious_fractions = np.array(
    [infectious_per_variant(state, V) for state in solution.y.T]
)

plot_compartments(
    solution,
    filename=(
        f"{plot_folder}/stable_equilibrium_without_most_transmissible_"
        f"variant_compartments.{fileformat}"
    ),
)

plot_infectious_fractions(
    solution,
    infectious_fractions,
    filename=(
        f"{plot_folder}/stable_equilibrium_without_most_transmissible_"
        f"variant_infectious.{fileformat}"
    ),
)


# Compute the equilibrium with variants 1 and 2
infectious_guess = np.array([1.0, 1.0, 0.0])

infectious_equilibrium, _, x_equilibrium = find_multi_variant_equilibrium(
    beta,
    gamma,
    sigma,
    V,
    G,
    T,
    S,
    b,
    infectious_guess,
    free_indices=equilibrium_variants,
)

if np.any(
    np.isclose(
        infectious_equilibrium[equilibrium_variants],
        0.0,
        atol=atol,
    )
):
    raise RuntimeError(
        "No non-trivial equilibrium containing variants 1 and 2 was found."
    )


# Check invasion ability and local stability
invasion_r0 = invasion_reproduction_number_multiple_variants(
    x_equilibrium,
    beta,
    gamma,
    b,
    V,
)

_, max_real_part, _ = eigenvalues_linearized_system(
    x_equilibrium,
    beta,
    gamma,
    sigma,
    b,
)

variant_number = excluded_variant + 1
excluded_variant_invasion_r0 = invasion_r0[excluded_variant]

print("=" * 80)
print("Equilibrium without the most transmissible variant")
print("=" * 80)

print(f"Transmission rates beta: {beta}")
print(f"Most transmissible variant: {np.argmax(beta) + 1}")
print(f"Equilibrium variants: {equilibrium_variants + 1}")
print()

print("Infectious fractions at the computed equilibrium:")
for index, infectious_fraction in enumerate(infectious_equilibrium):
    print(f"  Variant {index + 1}: {infectious_fraction:.8f}")

print()
print(
    f"Invasion reproduction number of variant {variant_number}: "
    f"{excluded_variant_invasion_r0:.8f}"
)

if excluded_variant_invasion_r0 > 1.0 + atol:
    print(
        f"Variant {variant_number} can invade the equilibrium of variants "
        f"{equilibrium_variants + 1}."
    )
elif np.isclose(excluded_variant_invasion_r0, 1.0, atol=atol):
    print(
        f"It is numerically unclear whether variant {variant_number} can "
        "invade the equilibrium."
    )
else:
    print(
        f"Variant {variant_number} cannot invade the equilibrium of variants "
        f"{equilibrium_variants + 1}."
    )

print()
print(f"Maximal real part of Jacobian eigenvalues: {max_real_part:.8e}")

if max_real_part < -stability_tol:
    print("The equilibrium is locally asymptotically stable.")
elif max_real_part > stability_tol:
    print("The equilibrium is unstable.")
else:
    print("The stability classification is numerically inconclusive.")
