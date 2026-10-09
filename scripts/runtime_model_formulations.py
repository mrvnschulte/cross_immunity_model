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
    construct_sparse_operators,
    multi_variant_model,
    multi_variant_model_for_loops,
    multi_variant_model_sparse,
    multi_variant_model_with_matrices,
)


# Run the model with naive implementation using for loops and subset checks
def run_model_for_loops(
    x0: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    b: float,
) -> None:
    solve_ivp(
        multi_variant_model_for_loops,
        (t0, tf),
        x0,
        t_eval=np.arange(tf),
        args=(beta, gamma, sigma, b),
    )


# Run the model with matrices computed internally in each RHS evaluation
def run_model_matrices_step(
    x0: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    b: float,
) -> None:
    solve_ivp(
        multi_variant_model,
        (t0, tf),
        x0,
        t_eval=np.arange(tf),
        args=(beta, gamma, sigma, b),
    )


# Run the model with dense matrices constructed once per solve_ivp call
def run_model_matrices_once(
    x0: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    b: float,
) -> None:
    V, G, T, S = construct_matrices(gamma, sigma)

    solve_ivp(
        multi_variant_model_with_matrices,
        (t0, tf),
        x0,
        t_eval=np.arange(tf),
        args=(beta, gamma, sigma, V, G, T, S, b),
    )


# Run the sparse model with operators constructed once per solve_ivp call
def run_model_sparse(
    x0: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    b: float,
) -> None:
    operators = construct_sparse_operators(gamma, sigma)

    solve_ivp(
        multi_variant_model_sparse,
        (t0, tf),
        x0,
        t_eval=np.arange(tf),
        args=(beta, operators, b),
    )


def measure_once(function) -> float:
    """
    Measures one invocation of a zero-argument function.

    Parameters
    ----------
    function : callable
        Function to be measured.

    Returns
    -------
    runtime : float
        Runtime in seconds.
    """
    return timeit.timeit(function, number=1)


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
n_repeats = 10
n_variants_max = 7
# This benchmark only evaluates one RHS call and can therefore use more variants
n_variants_max_function_call = 10

seed = 42
verbose = True

Path(folder).mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(seed=seed)

### STEP 1: Compare complete solve_ivp integrations.

mins_model_a = []
mins_model_b = []
mins_model_c = []
mins_model_d = []

means_model_a = []
means_model_b = []
means_model_c = []
means_model_d = []

variants = list(range(1, n_variants_max + 1))

for n_variants in variants:
    time_a = []
    time_b = []
    time_c = []
    time_d = []

    for _ in range(n_repeats):
        # A new parameter set is created for every repeat. The same parameter
        # set is then used by all four model formulations.
        x0, beta, gamma, sigma, b = build_base_params(n_variants, rng)

        time_a.append(
            measure_once(
                lambda: run_model_for_loops(
                    x0,
                    beta,
                    gamma,
                    sigma,
                    b,
                )
            )
        )

        time_b.append(
            measure_once(
                lambda: run_model_matrices_step(
                    x0,
                    beta,
                    gamma,
                    sigma,
                    b,
                )
            )
        )

        time_c.append(
            measure_once(
                lambda: run_model_matrices_once(
                    x0,
                    beta,
                    gamma,
                    sigma,
                    b,
                )
            )
        )

        time_d.append(
            measure_once(
                lambda: run_model_sparse(
                    x0,
                    beta,
                    gamma,
                    sigma,
                    b,
                )
            )
        )

    mins_model_a.append(np.min(time_a))
    mins_model_b.append(np.min(time_b))
    mins_model_c.append(np.min(time_c))
    mins_model_d.append(np.min(time_d))

    means_model_a.append(np.mean(time_a))
    means_model_b.append(np.mean(time_b))
    means_model_c.append(np.mean(time_c))
    means_model_d.append(np.mean(time_d))

    if verbose:
        print(f"Finished integration benchmark for {n_variants} variants")


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
            "min_sparse_s",
            "mean_loops_s",
            "mean_matrices_per_step_s",
            "mean_matrices_once_s",
            "mean_sparse_s",
            "min_speedup_matrices_per_step",
            "min_speedup_matrices_once",
            "min_speedup_sparse",
            "mean_speedup_matrices_per_step",
            "mean_speedup_matrices_once",
            "mean_speedup_sparse",
        ]
    )

    for i, n_variants in enumerate(variants):
        min_speedup_step = mins_model_a[i] / mins_model_b[i]
        min_speedup_once = mins_model_a[i] / mins_model_c[i]
        min_speedup_sparse = mins_model_a[i] / mins_model_d[i]

        mean_speedup_step = means_model_a[i] / means_model_b[i]
        mean_speedup_once = means_model_a[i] / means_model_c[i]
        mean_speedup_sparse = means_model_a[i] / means_model_d[i]

        writer.writerow(
            [
                n_variants,
                mins_model_a[i],
                mins_model_b[i],
                mins_model_c[i],
                mins_model_d[i],
                means_model_a[i],
                means_model_b[i],
                means_model_c[i],
                means_model_d[i],
                min_speedup_step,
                min_speedup_once,
                min_speedup_sparse,
                mean_speedup_step,
                mean_speedup_once,
                mean_speedup_sparse,
            ]
        )

if verbose:
    print(f"Integration CSV file saved: {csv_path}")


plt.figure()
plt.plot(variants, mins_model_a, label="Loops", marker="o")
plt.plot(variants, mins_model_b, label="Matrices per step", marker="s")
plt.plot(variants, mins_model_c, label="Matrices precomputed", marker="^")
plt.plot(variants, mins_model_d, label="Sparse matrices", marker="D")
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
plt.plot(variants, means_model_c, label="Matrices precomputed", marker="^")
plt.plot(variants, means_model_d, label="Sparse matrices", marker="D")
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


### STEP 2: Compare one direct RHS function call without solve_ivp integration

mins_function_a = []
mins_function_b = []
mins_function_c = []
mins_function_d = []

means_function_a = []
means_function_b = []
means_function_c = []
means_function_d = []

function_call_variants = list(range(1, n_variants_max_function_call + 1))

for n_variants in function_call_variants:
    time_a = []
    time_b = []
    time_c = []
    time_d = []

    for _ in range(n_repeats):
        # The same parameter set is used for all formulations in this repeat
        x0, beta, gamma, sigma, b = build_base_params(n_variants, rng)

        # These constructions are excluded from the measurements below
        V, G, T, S = construct_matrices(gamma, sigma)
        sparse_operators = construct_sparse_operators(gamma, sigma)

        time_a.append(
            measure_once(
                lambda: multi_variant_model_for_loops(
                    0.0,
                    x0,
                    beta,
                    gamma,
                    sigma,
                    b,
                )
            )
        )

        # This formulation intentionally constructs its matrices internally
        time_b.append(
            measure_once(
                lambda: multi_variant_model(
                    0.0,
                    x0,
                    beta,
                    gamma,
                    sigma,
                    b,
                )
            )
        )

        time_c.append(
            measure_once(
                lambda: multi_variant_model_with_matrices(
                    0.0,
                    x0,
                    beta,
                    gamma,
                    sigma,
                    V,
                    G,
                    T,
                    S,
                    b,
                )
            )
        )

        time_d.append(
            measure_once(
                lambda: multi_variant_model_sparse(
                    0.0,
                    x0,
                    beta,
                    sparse_operators,
                    b,
                )
            )
        )

    mins_function_a.append(np.min(time_a))
    mins_function_b.append(np.min(time_b))
    mins_function_c.append(np.min(time_c))
    mins_function_d.append(np.min(time_d))

    means_function_a.append(np.mean(time_a))
    means_function_b.append(np.mean(time_b))
    means_function_c.append(np.mean(time_c))
    means_function_d.append(np.mean(time_d))

    if verbose:
        print("Finished function-call benchmark for " f"{n_variants} variants")


function_call_csv_path = (
    Path(data_folder) / "function_call_runtime_comparison.csv"
)

with function_call_csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow(
        [
            "n_variants",
            "min_loops_s",
            "min_matrices_per_step_s",
            "min_matrices_once_s",
            "min_sparse_s",
            "mean_loops_s",
            "mean_matrices_per_step_s",
            "mean_matrices_once_s",
            "mean_sparse_s",
            "min_speedup_matrices_per_step",
            "min_speedup_matrices_once",
            "min_speedup_sparse",
            "mean_speedup_matrices_per_step",
            "mean_speedup_matrices_once",
            "mean_speedup_sparse",
        ]
    )

    for i, n_variants in enumerate(function_call_variants):
        min_speedup_step = mins_function_a[i] / mins_function_b[i]
        min_speedup_once = mins_function_a[i] / mins_function_c[i]
        min_speedup_sparse = mins_function_a[i] / mins_function_d[i]

        mean_speedup_step = means_function_a[i] / means_function_b[i]
        mean_speedup_once = means_function_a[i] / means_function_c[i]
        mean_speedup_sparse = means_function_a[i] / means_function_d[i]

        writer.writerow(
            [
                n_variants,
                mins_function_a[i],
                mins_function_b[i],
                mins_function_c[i],
                mins_function_d[i],
                means_function_a[i],
                means_function_b[i],
                means_function_c[i],
                means_function_d[i],
                min_speedup_step,
                min_speedup_once,
                min_speedup_sparse,
                mean_speedup_step,
                mean_speedup_once,
                mean_speedup_sparse,
            ]
        )

if verbose:
    print(f"Function-call CSV file saved: {function_call_csv_path}")


plt.figure()
plt.plot(function_call_variants, mins_function_a, label="Loops", marker="o")
plt.plot(
    function_call_variants,
    mins_function_b,
    label="Matrices per step",
    marker="s",
)
plt.plot(
    function_call_variants,
    mins_function_c,
    label="Matrices precomputed",
    marker="^",
)
plt.plot(
    function_call_variants,
    mins_function_d,
    label="Sparse matrices",
    marker="D",
)
plt.xlabel("Number of strains", fontsize=fontsize_text)
plt.ylabel("Minimal runtime [s]", fontsize=fontsize_text)
plt.xticks(function_call_variants, fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.yscale("log")
plt.legend(fontsize=fontsize_ticks)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/function_call_runtime_comparison_min.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()

plt.figure()
plt.plot(function_call_variants, means_function_a, label="Loops", marker="o")
plt.plot(
    function_call_variants,
    means_function_b,
    label="Matrices per step",
    marker="s",
)
plt.plot(
    function_call_variants,
    means_function_c,
    label="Matrices precomputed",
    marker="^",
)
plt.plot(
    function_call_variants,
    means_function_d,
    label="Sparse matrices",
    marker="D",
)
plt.xlabel("Number of strains", fontsize=fontsize_text)
plt.ylabel("Mean runtime [s]", fontsize=fontsize_text)
plt.xticks(function_call_variants, fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.yscale("log")
plt.legend(fontsize=fontsize_ticks)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/function_call_runtime_comparison_mean.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
