"""
Run the multi-variant model using different formulations and compare their
runtime for multiple numbers of variants.
"""

import timeit

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from cross_immunity_model import (
    construct_matrices,
    multi_variant_model,
    multi_variant_model_for_loops,
    multi_variant_model_with_matrices,
)


# Run the model with matrices computed internally (i.e. in each step)
def run_model_matrices_step():
    solve_ivp(
        multi_variant_model,
        (t0, tf),
        x0,
        t_eval=np.arange(tf),
        args=(beta, gamma, sigma),
    )


# Run the model with naive implementation using for loops and subset checks
def run_model_for_loops():
    solve_ivp(
        multi_variant_model_for_loops,
        (t0, tf),
        x0,
        t_eval=np.arange(tf),
        args=(beta, gamma, sigma),
    )


# Run the model with matrices as arguments (i.e. computed beforehand)
def run_model_matrices_once():
    V, G, T, S = construct_matrices(gamma, sigma)
    solve_ivp(
        multi_variant_model_with_matrices,
        (t0, tf),
        x0,
        t_eval=np.arange(tf),
        args=(beta, gamma, sigma, V, G, T, S),
    )


def build_base_params(
    n_variants: int, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int]:
    """
    Build parameters for given number of variants.

    Parameters
    ----------
    n_variants : int
        Number of variants.
    rng : np.random.Generator, optional
        Random number generator. The default is None.

    Returns
    -------
    x0 : np.ndarray
        Initial value.
    beta : np.ndarray
        Transmission rates.
    gamma : np.ndarray
        Recovery rates.
    sigma : np.ndarray
        Cross-immunity matrix.
    """
    if rng is None:
        rng = np.random.default_rng()
    x0 = 0.001 * np.ones(3**n_variants)
    x0[0] = 1 - np.sum(x0[1:])
    beta = np.random.uniform(0.15, 0.3, n_variants)
    gamma = 0.1 * np.ones(n_variants)
    sigma = 0.8 * np.ones((n_variants, n_variants))
    for i in range(n_variants):
        sigma[i, i] = 1
    return x0, beta, gamma, sigma


t0 = 0
tf = 500
n = 20
n_variants_max = 7
seed = 42
dpi = 500
verbose = True

rng = np.random.default_rng(seed=seed)
means_model_a = []
means_model_b = []
means_model_c = []
variants = list(range(1, n_variants_max + 1))

for n_variants in variants:
    x0, beta, gamma, sigma = build_base_params(n_variants)

    time_a = timeit.repeat(
        "run_model_for_loops()", globals=globals(), repeat=n, number=1
    )
    time_b = timeit.repeat(
        "run_model_matrices_step()", globals=globals(), repeat=n, number=1
    )
    time_c = timeit.repeat(
        "run_model_matrices_once()", globals=globals(), repeat=n, number=1
    )
    means_model_a.append(np.min(time_a))
    means_model_b.append(np.min(time_b))
    means_model_c.append(np.min(time_c))
    if verbose:
        print(f"Finished computation for {n_variants} variants")

plt.figure(figsize=(8, 6))
plt.plot(variants, means_model_a, label="Loops", marker="o")
plt.plot(variants, means_model_b, label="Matrices per step", marker="s")
plt.plot(variants, means_model_c, label="Matrices once", marker="^")
plt.xlabel("Number of variants")
plt.ylabel("Minimal runtime [s]")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../plots/Runtime_comparison.jpg", dpi=dpi)
plt.show()
