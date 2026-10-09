"""
Run the multi-variant model using different formulations and compare their
runtime for multiple numbers of variants.
"""

import csv
from pathlib import Path
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
        args=(beta, gamma, sigma, b),
    )


# Run the model with naive implementation using for loops and subset checks
def run_model_for_loops():
    solve_ivp(
        multi_variant_model_for_loops,
        (t0, tf),
        x0,
        t_eval=np.arange(tf),
        args=(beta, gamma, sigma, b),
    )


# Run the model with matrices as arguments (i.e. computed beforehand)
def run_model_matrices_once():
    V, G, T, S = construct_matrices(gamma, sigma)
    solve_ivp(
        multi_variant_model_with_matrices,
        (t0, tf),
        x0,
        t_eval=np.arange(tf),
        args=(beta, gamma, sigma, V, G, T, S, b),
    )


def build_base_params(
    n_variants: int,
    rng: np.random.Generator,
    initial_nonzero_mass: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build parameters for given number of variants.

    Parameters
    ----------
    n_variants : int
        Number of variants.
    rng : np.random.Generator
        Random number generator.
    initial_nonzero_mass : float
        Initial fraction of the population that is not fully susceptible.

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
    b : float
        Birth and death rate.
    """
    if n_variants < 1:
        raise ValueError("n_variants must be at least 1.")
    if not 0.0 <= initial_nonzero_mass < 1.0:
        raise ValueError("initial_nonzero_mass must be in [0, 1).")

    n_states = 3**n_variants
    x0 = np.empty(n_states, dtype=float)
    x0[0] = 1.0 - initial_nonzero_mass
    x0[1:] = initial_nonzero_mass / (n_states - 1)

    beta = rng.uniform(0.15, 0.3, size=n_variants)
    gamma = np.full(n_variants, 0.1)
    sigma = np.full((n_variants, n_variants), 0.8)
    np.fill_diagonal(sigma, 1.0)
    b = 0.001

    return x0, beta, gamma, sigma, b


dpi = 1200
fontsize_text = 20
fontsize_ticks = 16
alpha_grid = 0.7
folder = "../plots"
fileformat = "pdf"
data_folder = "../data"

t0 = 0
tf = 500
n_repeats = 1
n_variants_max = 7
seed = 42
verbose = True

rng = np.random.default_rng(seed=seed)
mins_model_a = []
mins_model_b = []
mins_model_c = []
means_model_a = []
means_model_b = []
means_model_c = []
variants = list(range(1, n_variants_max + 1))

for n_variants in variants:
    x0, beta, gamma, sigma, b = build_base_params(n_variants, rng)

    time_a = timeit.repeat(
        "run_model_for_loops()", globals=globals(), repeat=n_repeats, number=1
    )
    time_b = timeit.repeat(
        "run_model_matrices_step()",
        globals=globals(),
        repeat=n_repeats,
        number=1,
    )
    time_c = timeit.repeat(
        "run_model_matrices_once()",
        globals=globals(),
        repeat=n_repeats,
        number=1,
    )
    mins_model_a.append(np.min(time_a))
    mins_model_b.append(np.min(time_b))
    mins_model_c.append(np.min(time_c))
    means_model_a.append(np.mean(time_a))
    means_model_b.append(np.mean(time_b))
    means_model_c.append(np.mean(time_c))
    if verbose:
        print(f"Finished computation for {n_variants} variants")

csv_path = Path(data_folder) / "runtime_comparison.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)

with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "n_variants",
            "min_loops_s",
            "min_matrices_per_step_s",
            "min_matrices_once_s",
            "mean_loops_s",
            "mean_matrices_per_step_s",
            "mean_matrices_once_s",
            "min_speedup_matrices_per_step",
            "min_speedup_matrices_once",
            "mean_speedup_matrices_per_step",
            "mean_speedup_matrices_once",
        ]
    )

    for i, n_variants in enumerate(variants):
        min_speedup_step = mins_model_a[i] / mins_model_b[i]
        min_speedup_once = mins_model_a[i] / mins_model_c[i]
        mean_speedup_step = means_model_a[i] / means_model_b[i]
        mean_speedup_once = means_model_a[i] / means_model_c[i]

        writer.writerow(
            [
                n_variants,
                mins_model_a[i],
                mins_model_b[i],
                mins_model_c[i],
                means_model_a[i],
                means_model_b[i],
                means_model_c[i],
                min_speedup_step,
                min_speedup_once,
                mean_speedup_step,
                mean_speedup_once,
            ]
        )

if verbose:
    print(f"CSV file saved: {csv_path}")


plt.figure()
plt.plot(variants, mins_model_a, label="Loops", marker="o")
plt.plot(variants, mins_model_b, label="Matrices per step", marker="s")
plt.plot(variants, mins_model_c, label="Matrices once", marker="^")
plt.xlabel("Number of strains", fontsize=fontsize_text)
plt.ylabel("Minimal runtime [s]", fontsize=fontsize_text)
plt.xticks(variants, fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.yscale("log")
plt.legend(fontsize=fontsize_ticks)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/runtime_comparison_min.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()

plt.figure()
plt.plot(variants, means_model_a, label="Loops", marker="o")
plt.plot(variants, means_model_b, label="Matrices per step", marker="s")
plt.plot(variants, means_model_c, label="Matrices once", marker="^")
plt.xlabel("Number of strains", fontsize=fontsize_text)
plt.ylabel("Mean runtime [s]", fontsize=fontsize_text)
plt.xticks(variants, fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.yscale("log")
plt.legend(fontsize=fontsize_ticks)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/runtime_comparison_mean.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
