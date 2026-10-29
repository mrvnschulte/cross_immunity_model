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
    sample_random_parameters,
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
    beta_mean: float,
    beta_std: float,
    gamma_mean: float,
    gamma_std: float,
    sigma_alpha: float,
    sigma_beta: float,
    rng: np.random.Generator,
    initial_nonzero_mass: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build parameters for given number of variants.

    Parameters
    ----------
    n_variants : int
        Number of variants.
    beta_mean : float
        Mean of normal distribution for beta.
    beta_std : float
        Standard deviation of normal distribution for beta.
    gamma_mean : float
        Mean of normal distribution for gamma.
    gamma_std : float
        Standard deviation of normal distribution for gamma.
    sigma_alpha : float
        Shape parameter alpha of beta distribution for sigma.
    sigma_beta : float
        Shape parameter beta of beta distribution for sigma.
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

    beta, gamma, sigma = sample_random_parameters(
        n_variants,
        beta_mean,
        beta_std,
        gamma_mean,
        gamma_std,
        sigma_alpha,
        sigma_beta,
    )

    return x0, beta, gamma, sigma


def save_runtime_data(
    csv_path: Path,
    variants: list[int],
    runtime_data: dict[str, list[float]],
) -> None:
    """
    Saves runtime data and speedups to a CSV file.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path of the CSV file.
    variants : list[int]
        Numbers of variants.
    runtime_data : dict[str, list[float]]
        Runtime data for all model formulations.
    """
    missing_columns = [
        column
        for column in RUNTIME_VALUE_COLUMNS
        if column not in runtime_data
    ]

    if missing_columns:
        raise ValueError(
            "Missing runtime columns: " f"{', '.join(missing_columns)}."
        )

    invalid_lengths = [
        column
        for column in RUNTIME_VALUE_COLUMNS
        if len(runtime_data[column]) != len(variants)
    ]

    if invalid_lengths:
        raise ValueError(
            "Runtime data has an invalid length for: "
            f"{', '.join(invalid_lengths)}."
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_COLUMNS,
        )
        writer.writeheader()

        for i, n_variants in enumerate(variants):
            row = {"n_variants": n_variants}

            for column in RUNTIME_VALUE_COLUMNS:
                row[column] = runtime_data[column][i]

            row["min_speedup_matrices_per_step"] = (
                row["min_loops_s"] / row["min_matrices_per_step_s"]
            )
            row["min_speedup_matrices_once"] = (
                row["min_loops_s"] / row["min_matrices_once_s"]
            )
            row["min_speedup_sparse"] = (
                row["min_loops_s"] / row["min_sparse_s"]
            )
            row["mean_speedup_matrices_per_step"] = (
                row["mean_loops_s"] / row["mean_matrices_per_step_s"]
            )
            row["mean_speedup_matrices_once"] = (
                row["mean_loops_s"] / row["mean_matrices_once_s"]
            )
            row["mean_speedup_sparse"] = (
                row["mean_loops_s"] / row["mean_sparse_s"]
            )

            writer.writerow(row)


def load_runtime_data(
    csv_path: Path,
) -> tuple[list[int], dict[str, list[float]]]:
    """
    Loads runtime data from a CSV file.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path of the CSV file.

    Returns
    -------
    variants : list[int]
        Numbers of variants.
    runtime_data : dict[str, list[float]]
        Runtime data for all model formulations.
    """
    try:
        csv_file = csv_path.open(
            "r",
            newline="",
            encoding="utf-8",
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. "
            "Set recompute = True to create it."
        ) from error

    with csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty: {csv_path}")

        required_columns = (
            "n_variants",
            *RUNTIME_VALUE_COLUMNS,
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in reader.fieldnames
        ]

        if missing_columns:
            raise ValueError(
                f"CSV file {csv_path} is missing columns: "
                f"{', '.join(missing_columns)}."
            )

        variants = []
        runtime_data = {column: [] for column in RUNTIME_VALUE_COLUMNS}

        for row_number, row in enumerate(reader, start=2):
            try:
                n_variants = int(row["n_variants"])
                values = {
                    column: float(row[column])
                    for column in RUNTIME_VALUE_COLUMNS
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid value in {csv_path} at line " f"{row_number}."
                ) from error

            variants.append(n_variants)

            for column, value in values.items():
                runtime_data[column].append(value)

    if not variants:
        raise ValueError(f"CSV file contains no runtime data: {csv_path}")

    return variants, runtime_data


dpi = 1200
fontsize_text = 20
fontsize_ticks = 16
alpha_grid = 0.7
folder = "../plots_test"
fileformat = "pdf"
data_folder = "../data_test"

t0 = 0
tf = 500
n_repeats = 10
n_variants_max = 4
# This benchmark only evaluates one RHS call and can therefore use more variants
n_variants_max_function_call = 4

beta_mean = 0.3
beta_std = 0.025
gamma_mean = 0.1
gamma_std = 0
sigma_alpha = 9
sigma_beta = 1
b = 0.01

seed = 42
verbose = True

recompute = True

RUNTIME_VALUE_COLUMNS = (
    "min_loops_s",
    "min_matrices_per_step_s",
    "min_matrices_once_s",
    "min_sparse_s",
    "mean_loops_s",
    "mean_matrices_per_step_s",
    "mean_matrices_once_s",
    "mean_sparse_s",
)

CSV_COLUMNS = (
    "n_variants",
    *RUNTIME_VALUE_COLUMNS,
    "min_speedup_matrices_per_step",
    "min_speedup_matrices_once",
    "min_speedup_sparse",
    "mean_speedup_matrices_per_step",
    "mean_speedup_matrices_once",
    "mean_speedup_sparse",
)


Path(folder).mkdir(parents=True, exist_ok=True)

csv_path = Path(data_folder) / "runtime_comparison.csv"
function_call_csv_path = (
    Path(data_folder) / "function_call_runtime_comparison.csv"
)

rng = np.random.default_rng(seed=seed)

### STEP 1: Compare complete solve_ivp integrations

if recompute:
    integration_variants = list(range(1, n_variants_max + 1))

    mins_model_a = []
    mins_model_b = []
    mins_model_c = []
    mins_model_d = []

    means_model_a = []
    means_model_b = []
    means_model_c = []
    means_model_d = []

    for n_variants in integration_variants:
        time_a = []
        time_b = []
        time_c = []
        time_d = []

        for _ in range(n_repeats):
            # A new parameter set is created for every repeat. The same
            # parameter set is used by all four model formulations.
            x0, beta, gamma, sigma = build_base_params(
                n_variants,
                beta_mean,
                beta_std,
                gamma_mean,
                gamma_std,
                sigma_alpha,
                sigma_beta,
                rng,
            )

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
            print(
                "Finished integration benchmark for " f"{n_variants} variants"
            )

    integration_runtime_data = {
        "min_loops_s": mins_model_a,
        "min_matrices_per_step_s": mins_model_b,
        "min_matrices_once_s": mins_model_c,
        "min_sparse_s": mins_model_d,
        "mean_loops_s": means_model_a,
        "mean_matrices_per_step_s": means_model_b,
        "mean_matrices_once_s": means_model_c,
        "mean_sparse_s": means_model_d,
    }

    save_runtime_data(
        csv_path,
        integration_variants,
        integration_runtime_data,
    )

    if verbose:
        print(f"Integration CSV file saved: {csv_path}")

else:
    integration_variants, integration_runtime_data = load_runtime_data(
        csv_path
    )

    if verbose:
        print(f"Integration CSV file loaded: {csv_path}")

mins_model_a = integration_runtime_data["min_loops_s"]
mins_model_b = integration_runtime_data["min_matrices_per_step_s"]
mins_model_c = integration_runtime_data["min_matrices_once_s"]
mins_model_d = integration_runtime_data["min_sparse_s"]

means_model_a = integration_runtime_data["mean_loops_s"]
means_model_b = integration_runtime_data["mean_matrices_per_step_s"]
means_model_c = integration_runtime_data["mean_matrices_once_s"]
means_model_d = integration_runtime_data["mean_sparse_s"]


fig = plt.figure()
plt.plot(integration_variants, mins_model_a, label="Loops", marker="o")
plt.plot(
    integration_variants,
    mins_model_b,
    label="Matrices per step",
    marker="s",
)
plt.plot(
    integration_variants,
    mins_model_c,
    label="Matrices precomputed",
    marker="^",
)
plt.plot(
    integration_variants,
    mins_model_d,
    label="Sparse matrices",
    marker="D",
)
plt.xlabel("Number of strains", fontsize=fontsize_text)
plt.ylabel("Minimal runtime [s]", fontsize=fontsize_text)
plt.xticks(integration_variants, fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.yscale("log")
fig.legend(
    fontsize=fontsize_ticks,
    loc="upper left",
    bbox_to_anchor=(-0.075, -0.05),
    ncol=2,
)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/runtime_comparison_min.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()

fig = plt.figure()
plt.plot(integration_variants, means_model_a, label="Loops", marker="o")
plt.plot(
    integration_variants,
    means_model_b,
    label="Matrices per step",
    marker="s",
)
plt.plot(
    integration_variants,
    means_model_c,
    label="Matrices precomputed",
    marker="^",
)
plt.plot(
    integration_variants,
    means_model_d,
    label="Sparse matrices",
    marker="D",
)
plt.xlabel("Number of strains", fontsize=fontsize_text)
plt.ylabel("Mean runtime [s]", fontsize=fontsize_text)
plt.xticks(integration_variants, fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.yscale("log")
fig.legend(
    fontsize=fontsize_ticks,
    loc="upper left",
    bbox_to_anchor=(-0.075, -0.05),
    ncol=2,
)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/runtime_comparison_mean.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()


### STEP 2: Compare one direct RHS function call without solve_ivp integration

if recompute:
    function_call_variants = list(range(1, n_variants_max_function_call + 1))

    mins_function_a = []
    mins_function_b = []
    mins_function_c = []
    mins_function_d = []

    means_function_a = []
    means_function_b = []
    means_function_c = []
    means_function_d = []

    for n_variants in function_call_variants:
        time_a = []
        time_b = []
        time_c = []
        time_d = []

        for _ in range(n_repeats):
            # The same parameter set is used for all formulations in this
            # repeat.
            x0, beta, gamma, sigma = build_base_params(
                n_variants,
                beta_mean,
                beta_std,
                gamma_mean,
                gamma_std,
                sigma_alpha,
                sigma_beta,
                rng,
            )

            # These constructions are excluded from the measurements below.
            V, G, T, S = construct_matrices(gamma, sigma)
            sparse_operators = construct_sparse_operators(
                gamma,
                sigma,
            )

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

            # This formulation intentionally constructs its matrices
            # internally.
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
            print(
                "Finished function-call benchmark for "
                f"{n_variants} variants"
            )

    function_call_runtime_data = {
        "min_loops_s": mins_function_a,
        "min_matrices_per_step_s": mins_function_b,
        "min_matrices_once_s": mins_function_c,
        "min_sparse_s": mins_function_d,
        "mean_loops_s": means_function_a,
        "mean_matrices_per_step_s": means_function_b,
        "mean_matrices_once_s": means_function_c,
        "mean_sparse_s": means_function_d,
    }

    save_runtime_data(
        function_call_csv_path,
        function_call_variants,
        function_call_runtime_data,
    )

    if verbose:
        print("Function-call CSV file saved: " f"{function_call_csv_path}")

else:
    function_call_variants, function_call_runtime_data = load_runtime_data(
        function_call_csv_path
    )

    if verbose:
        print("Function-call CSV file loaded: " f"{function_call_csv_path}")

mins_function_a = function_call_runtime_data["min_loops_s"]
mins_function_b = function_call_runtime_data["min_matrices_per_step_s"]
mins_function_c = function_call_runtime_data["min_matrices_once_s"]
mins_function_d = function_call_runtime_data["min_sparse_s"]

means_function_a = function_call_runtime_data["mean_loops_s"]
means_function_b = function_call_runtime_data["mean_matrices_per_step_s"]
means_function_c = function_call_runtime_data["mean_matrices_once_s"]
means_function_d = function_call_runtime_data["mean_sparse_s"]


fig = plt.figure()
plt.plot(
    function_call_variants,
    mins_function_a,
    label="Loops",
    marker="o",
)
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
fig.legend(
    fontsize=fontsize_ticks,
    loc="upper left",
    bbox_to_anchor=(-0.075, -0.05),
    ncol=2,
)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/function_call_runtime_comparison_min.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()

fig = plt.figure()
plt.plot(
    function_call_variants,
    means_function_a,
    label="Loops",
    marker="o",
)
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
fig.legend(
    fontsize=fontsize_ticks,
    loc="upper left",
    bbox_to_anchor=(-0.075, -0.05),
    ncol=2,
)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/function_call_runtime_comparison_mean.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
