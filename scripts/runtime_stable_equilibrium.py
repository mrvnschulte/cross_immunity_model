"""
Compare runtimes of different methods to find the stable multi-variant
equilibrium for varying numbers of variants:
  1. find_stable_multi_variant_equilibrium with ordered_check=True
  2. find_stable_multi_variant_equilibrium with ordered_check=False
  3. Long simulation of multi_variant_model_with_matrices until all
     derivatives fall below a threshold (deriv_tol)

For each variant count, N_parameters independent parameter sets are sampled
following the same procedure as in numerical_stability_multiple_variants.py.
Each method is timed once per parameter set; mean and standard deviation or
minima and maxima across parameter sets are shown in the final plot.
"""

import csv
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from cross_immunity_model import (
    construct_matrices,
    find_stable_multi_variant_equilibrium,
    infectious_per_variant,
    invasion_reproduction_number_multiple_variants,
    multi_variant_model_with_matrices,
    sample_random_parameters,
    solve_multi_variant_equilibrium_equation,
)


def find_equilibrium_by_simulation(
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    V: np.ndarray,
    G: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    b: float,
    x0: np.ndarray,
    deriv_tol: float = 1e-4,
    t_max: float = 1e6,
) -> tuple[np.ndarray, float]:
    """
    Integrates the multi-variant model until max(|dx/dt|) < deriv_tol or
    t_max is reached, and returns the final state as the approximate
    equilibrium together with the simulation time at which it stopped.

    Parameters
    ----------
    beta, gamma, sigma, V, G, T, S, b
        Model parameters and pre-computed matrices.
    x0 : np.ndarray
        Initial condition.
    deriv_tol : float
        Derivative magnitude threshold below which the state is considered
        stationary. The default is 1e-5.
    t_max : float
        Maximum integration time if the threshold is never reached.

    Returns
    -------
    x_eq : np.ndarray
        Approximate equilibrium state.
    t_stop : float
        Simulation time at which the integration stopped. Equals t_max if
        the derivative threshold was never reached.
    """

    def _equilibrium_reached(t, x, beta, gamma, sigma, V, G, T, S, b):
        dx = multi_variant_model_with_matrices(
            t, x, beta, gamma, sigma, V, G, T, S, b
        )
        return np.linalg.norm(dx) - deriv_tol

    # Terminate only when the derivative norm is decreasing through the
    # threshold (direction=-1) to avoid spurious triggers at t=0.
    _equilibrium_reached.terminal = True
    _equilibrium_reached.direction = -1

    sol = solve_ivp(
        multi_variant_model_with_matrices,
        (0.0, t_max),
        x0,
        args=(beta, gamma, sigma, V, G, T, S, b),
        events=_equilibrium_reached,
        method="RK45",
    )

    # sol.t_events[0] contains the time(s) the terminal event fired;
    # if it is empty the integrator reached t_max without converging.
    if len(sol.t_events[0]) > 0:
        t_stop = float(sol.t_events[0][0])
    else:
        t_stop = float(sol.t[-1])

    return sol.y[:, -1], t_stop


def runtime_csv_columns(deriv_tols: list[float]) -> list[str]:
    """
    Return CSV column names for the runtime benchmark data.

    Parameters
    ----------
    deriv_tols : list of float
        Derivative tolerances used by the simulation method.

    Returns
    -------
    list of str
        Ordered CSV column names.
    """
    columns = [
        "n_variants",
        "parameter_set",
        "seed",
        "runtime_ordered_s",
        "runtime_unordered_s",
        "error_ordered",
        "error_unordered",
        "inv_r0_distance_ordered",
        "inv_r0_distance_unordered",
    ]

    for tol in deriv_tols:
        tol_name = f"{tol:.0e}"

        columns.extend(
            [
                f"runtime_simulation_tol_{tol_name}_s",
                f"simulation_stop_time_tol_{tol_name}",
                f"error_simulation_tol_{tol_name}",
                f"inv_r0_distance_simulation_tol_{tol_name}",
            ]
        )

    return columns


def save_runtime_data(
    csv_filename: str | Path,
    variants: list[int],
    times_ordered: list[np.ndarray],
    times_unordered: list[np.ndarray],
    times_simulation: dict[float, list[np.ndarray]],
    t_stops: dict[float, list[np.ndarray]],
    errors_ordered: list[np.ndarray],
    errors_unordered: list[np.ndarray],
    errors_simulation: dict[float, list[np.ndarray]],
    inv_r0_dists_ordered: list[np.ndarray],
    inv_r0_dists_unordered: list[np.ndarray],
    inv_r0_dists_simulation: dict[float, list[np.ndarray]],
    deriv_tols: list[float],
) -> None:
    """
    Save raw runtime benchmark results to a CSV file.

    One row is written for each combination of number of variants and sampled
    parameter set. The CSV contains all values needed to recreate the runtime,
    stopping-time, error, and invasion-reproduction-number plots.

    Parameters
    ----------
    csv_filename : str or pathlib.Path
        Path of the output CSV file.
    variants : list of int
        Numbers of variants included in the benchmark.
    times_ordered, times_unordered : list of numpy.ndarray
        Runtime measurements for the ordered and unordered equilibrium search.
    times_simulation : dict of float to list of numpy.ndarray
        Runtime measurements for each simulation tolerance.
    t_stops : dict of float to list of numpy.ndarray
        Simulation stopping times for each simulation tolerance.
    errors_ordered, errors_unordered : list of numpy.ndarray
        Errors of equilibrium-search results relative to the reference solution.
    errors_simulation : dict of float to list of numpy.ndarray
        Simulation errors relative to the reference solution.
    inv_r0_dists_ordered, inv_r0_dists_unordered : list of numpy.ndarray
        Maximum distances of invasion reproduction numbers from one.
    inv_r0_dists_simulation : dict of float to list of numpy.ndarray
        Maximum distances for simulations at each tolerance.
    deriv_tols : list of float
        Derivative tolerances used by the simulation method.
    """
    csv_path = Path(csv_filename)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = runtime_csv_columns(deriv_tols)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for variant_index, n_variants in enumerate(variants):
            n_parameter_sets = len(times_ordered[variant_index])

            for parameter_set in range(n_parameter_sets):
                row = {
                    "n_variants": n_variants,
                    "parameter_set": parameter_set,
                    "seed": parameter_set,
                    "runtime_ordered_s": times_ordered[variant_index][
                        parameter_set
                    ],
                    "runtime_unordered_s": times_unordered[variant_index][
                        parameter_set
                    ],
                    "error_ordered": errors_ordered[variant_index][
                        parameter_set
                    ],
                    "error_unordered": errors_unordered[variant_index][
                        parameter_set
                    ],
                    "inv_r0_distance_ordered": inv_r0_dists_ordered[
                        variant_index
                    ][parameter_set],
                    "inv_r0_distance_unordered": inv_r0_dists_unordered[
                        variant_index
                    ][parameter_set],
                }

                for tol in deriv_tols:
                    tol_name = f"{tol:.0e}"

                    row[f"runtime_simulation_tol_{tol_name}_s"] = (
                        times_simulation[tol][variant_index][parameter_set]
                    )
                    row[f"simulation_stop_time_tol_{tol_name}"] = t_stops[tol][
                        variant_index
                    ][parameter_set]
                    row[f"error_simulation_tol_{tol_name}"] = (
                        errors_simulation[tol][variant_index][parameter_set]
                    )
                    row[f"inv_r0_distance_simulation_tol_{tol_name}"] = (
                        inv_r0_dists_simulation[tol][variant_index][
                            parameter_set
                        ]
                    )

                writer.writerow(row)

    print(f"Runtime CSV file saved: {csv_path}")


def load_runtime_data(
    csv_filename: str | Path,
    variants: list[int],
    deriv_tols: list[float],
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    dict[float, list[np.ndarray]],
    dict[float, list[np.ndarray]],
    list[np.ndarray],
    list[np.ndarray],
    dict[float, list[np.ndarray]],
    list[np.ndarray],
    list[np.ndarray],
    dict[float, list[np.ndarray]],
]:
    """
    Load raw runtime benchmark results from a CSV file.

    Parameters
    ----------
    csv_filename : str or pathlib.Path
        Path of the CSV file created by ``save_runtime_data``.
    variants : list of int
        Expected numbers of variants.
    deriv_tols : list of float
        Expected simulation derivative tolerances.

    Returns
    -------
    tuple
        Benchmark results in the same structure used by the plotting code.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing, values are invalid, or the CSV does
        not contain data for every requested number of variants.
    """
    csv_path = Path(csv_filename)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Runtime CSV file not found: {csv_path}. "
            "Set recompute = True to create it."
        )

    required_columns = set(runtime_csv_columns(deriv_tols))

    data_by_variant: dict[int, list[dict[str, str]]] = {
        n_variants: [] for n_variants in variants
    }

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError(f"Runtime CSV file is empty: {csv_path}")

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"Runtime CSV file {csv_path} is missing columns: "
                f"{', '.join(sorted(missing_columns))}."
            )

        for row in reader:
            n_variants = int(row["n_variants"])

            if n_variants in data_by_variant:
                data_by_variant[n_variants].append(row)

    missing_variants = [
        n_variants for n_variants, rows in data_by_variant.items() if not rows
    ]

    if missing_variants:
        raise ValueError(
            "Runtime CSV file contains no data for: " f"{missing_variants}."
        )

    times_ordered = []
    times_unordered = []
    times_simulation = {tol: [] for tol in deriv_tols}
    t_stops = {tol: [] for tol in deriv_tols}

    errors_ordered = []
    errors_unordered = []
    errors_simulation = {tol: [] for tol in deriv_tols}

    inv_r0_dists_ordered = []
    inv_r0_dists_unordered = []
    inv_r0_dists_simulation = {tol: [] for tol in deriv_tols}

    for n_variants in variants:
        rows = sorted(
            data_by_variant[n_variants],
            key=lambda row: int(row["parameter_set"]),
        )

        times_ordered.append(
            np.array(
                [float(row["runtime_ordered_s"]) for row in rows],
                dtype=float,
            )
        )
        times_unordered.append(
            np.array(
                [float(row["runtime_unordered_s"]) for row in rows],
                dtype=float,
            )
        )
        errors_ordered.append(
            np.array(
                [float(row["error_ordered"]) for row in rows],
                dtype=float,
            )
        )
        errors_unordered.append(
            np.array(
                [float(row["error_unordered"]) for row in rows],
                dtype=float,
            )
        )
        inv_r0_dists_ordered.append(
            np.array(
                [float(row["inv_r0_distance_ordered"]) for row in rows],
                dtype=float,
            )
        )
        inv_r0_dists_unordered.append(
            np.array(
                [float(row["inv_r0_distance_unordered"]) for row in rows],
                dtype=float,
            )
        )

        for tol in deriv_tols:
            tol_name = f"{tol:.0e}"

            times_simulation[tol].append(
                np.array(
                    [
                        float(row[f"runtime_simulation_tol_{tol_name}_s"])
                        for row in rows
                    ],
                    dtype=float,
                )
            )
            t_stops[tol].append(
                np.array(
                    [
                        float(row[f"simulation_stop_time_tol_{tol_name}"])
                        for row in rows
                    ],
                    dtype=float,
                )
            )
            errors_simulation[tol].append(
                np.array(
                    [
                        float(row[f"error_simulation_tol_{tol_name}"])
                        for row in rows
                    ],
                    dtype=float,
                )
            )
            inv_r0_dists_simulation[tol].append(
                np.array(
                    [
                        float(
                            row[f"inv_r0_distance_simulation_tol_{tol_name}"]
                        )
                        for row in rows
                    ],
                    dtype=float,
                )
            )

    return (
        times_ordered,
        times_unordered,
        times_simulation,
        t_stops,
        errors_ordered,
        errors_unordered,
        errors_simulation,
        inv_r0_dists_ordered,
        inv_r0_dists_unordered,
        inv_r0_dists_simulation,
    )


# Plot settings
dpi = 1200
savefig = True
verbose = True
verbose_step = 10
plot_folder = "../plots"
fileformat = "pdf"
fontsize_text = 20
fontsize_ticks = 16
alpha_grid = 0.7
data_folder = "../data"
recompute = False
runtime_csv_file = f"{data_folder}/runtime_stable_equilibrium_raw_results.csv"


# Parameter settings
beta_mean = 0.3
beta_std = 0.025
gamma_mean = 0.1
gamma_std = 0.0
sigma_alpha = 9
sigma_beta = 1
b = 0.01
atol = 1e-4
deriv_tols = [1e-4, 1e-5]
atol_ref = 1e-8
N_parameters = 1000
n_variants_max = 6

variants = list(range(1, n_variants_max + 1))

if recompute:
    times_ordered = []
    times_unordered = []
    times_simulation = {tol: [] for tol in deriv_tols}
    t_stops = {tol: [] for tol in deriv_tols}

    errors_ordered = []
    errors_unordered = []
    errors_simulation = {tol: [] for tol in deriv_tols}

    inv_r0_dists_ordered = []
    inv_r0_dists_unordered = []
    inv_r0_dists_simulation = {tol: [] for tol in deriv_tols}

    # Benchmark loop
    for n_variants in variants:
        t_ord = np.empty(N_parameters)
        t_unord = np.empty(N_parameters)
        t_sim = {tol: np.empty(N_parameters) for tol in deriv_tols}
        t_stop_arr = {tol: np.empty(N_parameters) for tol in deriv_tols}

        err_ord = np.empty(N_parameters)
        err_unord = np.empty(N_parameters)
        err_sim = {tol: np.empty(N_parameters) for tol in deriv_tols}

        dist_ord = np.empty(N_parameters)
        dist_unord = np.empty(N_parameters)
        dist_sim = {tol: np.empty(N_parameters) for tol in deriv_tols}

        for i in range(N_parameters):
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

            V, G, T, S = construct_matrices(gamma, sigma)

            x0 = 0.01 * np.ones(3**n_variants)
            x0[0] = 1.0 - np.sum(x0[1:])

            t_start = time.perf_counter()
            coexisting_variants, I_var_ord, x_eq_ord, _ = (
                find_stable_multi_variant_equilibrium(
                    beta,
                    gamma,
                    sigma,
                    b,
                    V=V,
                    G=G,
                    T=T,
                    S=S,
                    ordered_check=True,
                    atol=atol,
                )
            )
            t_ord[i] = time.perf_counter() - t_start

            t_start = time.perf_counter()
            coexisting_unord, I_var_unord, x_eq_unord, _ = (
                find_stable_multi_variant_equilibrium(
                    beta,
                    gamma,
                    sigma,
                    b,
                    V=V,
                    G=G,
                    T=T,
                    S=S,
                    ordered_check=False,
                    atol=atol,
                )
            )
            t_unord[i] = time.perf_counter() - t_start

            for tol in deriv_tols:
                t_start = time.perf_counter()

                x_eq_sim, t_stop_arr[tol][i] = find_equilibrium_by_simulation(
                    beta,
                    gamma,
                    sigma,
                    V,
                    G,
                    T,
                    S,
                    b,
                    x0,
                    deriv_tol=tol,
                )

                t_sim[tol][i] = time.perf_counter() - t_start

                I_var_sim = infectious_per_variant(
                    x_eq_sim,
                    V,
                    matrix_form=False,
                )

                if tol == deriv_tols[0]:
                    I_var_ref = solve_multi_variant_equilibrium_equation(
                        beta,
                        gamma,
                        sigma,
                        V,
                        G,
                        T,
                        S,
                        b,
                        I_variant_guess=I_var_sim,
                        atol=atol_ref,
                        free_indices=coexisting_variants,
                    )

                    err_ord[i] = np.linalg.norm(I_var_ord - I_var_ref)
                    err_unord[i] = np.linalg.norm(I_var_unord - I_var_ref)

                err_sim[tol][i] = np.linalg.norm(I_var_sim - I_var_ref)

                inv_r0_sim = invasion_reproduction_number_multiple_variants(
                    x_eq_sim,
                    beta,
                    gamma,
                    b,
                    V,
                )

                if len(coexisting_variants) > 0:
                    dist_sim[tol][i] = np.max(
                        np.abs(inv_r0_sim[coexisting_variants] - 1)
                    )
                else:
                    dist_sim[tol][i] = 0.0

            inv_r0_ord = invasion_reproduction_number_multiple_variants(
                x_eq_ord,
                beta,
                gamma,
                b,
                V,
            )
            inv_r0_unord = invasion_reproduction_number_multiple_variants(
                x_eq_unord,
                beta,
                gamma,
                b,
                V,
            )

            if len(coexisting_variants) > 0:
                dist_ord[i] = np.max(
                    np.abs(inv_r0_ord[coexisting_variants] - 1)
                )
                dist_unord[i] = np.max(
                    np.abs(inv_r0_unord[coexisting_unord] - 1)
                )
            else:
                dist_ord[i] = 0.0
                dist_unord[i] = 0.0

            if verbose and (i + 1) % verbose_step == 0:
                print(f"  {i + 1}/{N_parameters} parameter sets done")

        times_ordered.append(t_ord)
        times_unordered.append(t_unord)
        errors_ordered.append(err_ord)
        errors_unordered.append(err_unord)
        inv_r0_dists_ordered.append(dist_ord)
        inv_r0_dists_unordered.append(dist_unord)

        for tol in deriv_tols:
            times_simulation[tol].append(t_sim[tol])
            t_stops[tol].append(t_stop_arr[tol])
            errors_simulation[tol].append(err_sim[tol])
            inv_r0_dists_simulation[tol].append(dist_sim[tol])

        if verbose:
            print(f"Finished benchmarks for {n_variants} strain(s).")

    save_runtime_data(
        runtime_csv_file,
        variants,
        times_ordered,
        times_unordered,
        times_simulation,
        t_stops,
        errors_ordered,
        errors_unordered,
        errors_simulation,
        inv_r0_dists_ordered,
        inv_r0_dists_unordered,
        inv_r0_dists_simulation,
        deriv_tols,
    )

else:
    (
        times_ordered,
        times_unordered,
        times_simulation,
        t_stops,
        errors_ordered,
        errors_unordered,
        errors_simulation,
        inv_r0_dists_ordered,
        inv_r0_dists_unordered,
        inv_r0_dists_simulation,
    ) = load_runtime_data(
        runtime_csv_file,
        variants,
        deriv_tols,
    )

    print(f"Runtime CSV file loaded: {runtime_csv_file}")

# Aggregate statistics
means_ordered = [t.mean() for t in times_ordered]
means_unordered = [t.mean() for t in times_unordered]
stds_ordered = [t.std() for t in times_ordered]
stds_unordered = [t.std() for t in times_unordered]

means_err_ord = [e.mean() for e in errors_ordered]
means_err_unord = [e.mean() for e in errors_unordered]
mins_err_ord = [e.min() for e in errors_ordered]
mins_err_unord = [e.min() for e in errors_unordered]
maxs_err_ord = [e.max() for e in errors_ordered]
maxs_err_unord = [e.max() for e in errors_unordered]

means_dist_ord = [d.mean() for d in inv_r0_dists_ordered]
means_dist_unord = [d.mean() for d in inv_r0_dists_unordered]
mins_dist_ord = [d.min() for d in inv_r0_dists_ordered]
mins_dist_unord = [d.min() for d in inv_r0_dists_unordered]
maxs_dist_ord = [d.max() for d in inv_r0_dists_ordered]
maxs_dist_unord = [d.max() for d in inv_r0_dists_unordered]

# Per-tol aggregates
means_sim = {
    tol: [t.mean() for t in times_simulation[tol]] for tol in deriv_tols
}
stds_sim = {
    tol: [t.std() for t in times_simulation[tol]] for tol in deriv_tols
}
means_tstop = {tol: [t.mean() for t in t_stops[tol]] for tol in deriv_tols}
stds_tstop = {tol: [t.std() for t in t_stops[tol]] for tol in deriv_tols}
means_esim = {
    tol: [e.mean() for e in errors_simulation[tol]] for tol in deriv_tols
}
mins_esim = {
    tol: [e.min() for e in errors_simulation[tol]] for tol in deriv_tols
}
maxs_esim = {
    tol: [e.max() for e in errors_simulation[tol]] for tol in deriv_tols
}
means_dsim = {
    tol: [d.mean() for d in inv_r0_dists_simulation[tol]] for tol in deriv_tols
}
mins_dsim = {
    tol: [d.min() for d in inv_r0_dists_simulation[tol]] for tol in deriv_tols
}
maxs_dsim = {
    tol: [d.max() for d in inv_r0_dists_simulation[tol]] for tol in deriv_tols
}


# Plot 1: Runtimes
plt.figure()
sim_colors = {}

for means, stds, label, marker in (
    (means_ordered, stds_ordered, "Ordered search", "o"),
    (means_unordered, stds_unordered, "Naive search", "s"),
    *[
        (
            means_sim[tol],
            stds_sim[tol],
            f"Simulation (tol=$10^{{{int(np.log10(tol))}}}$)",
            m,
        )
        for tol, m in zip(deriv_tols, ["^", "v"])
    ],
):
    (line,) = plt.plot(variants, means, label=label, marker=marker)
    lower = np.maximum(np.array(means) - np.array(stds), 0)
    upper = np.array(means) + np.array(stds)
    plt.fill_between(variants, lower, upper, alpha=0.2, color=line.get_color())

    for tol in deriv_tols:
        if label == f"Simulation (tol=$10^{{{int(np.log10(tol))}}}$)":
            sim_colors[tol] = line.get_color()

plt.xlabel("Number of strains", fontsize=fontsize_text)
plt.ylabel("Mean runtime [s]", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.yscale("log")
plt.legend(
    fontsize=fontsize_ticks,
    loc="upper center",
    bbox_to_anchor=(0.4, -0.2),
    ncol=2,
)
plt.grid(True, alpha=alpha_grid)
if savefig:
    plt.savefig(
        f"{plot_folder}/runtime_stable_equilibrium.{fileformat}",
        dpi=dpi,
        bbox_inches="tight",
    )
plt.show()
plt.close()

# Plot 2: simulation stopping time t_stop
plt.figure()

for tol, marker in zip(deriv_tols, ["^", "v"]):
    means = means_tstop[tol]
    stds = stds_tstop[tol]
    (line,) = plt.plot(
        variants,
        means,
        marker=marker,
        label=f"Simulation (tol=$10^{{{int(np.log10(tol))}}}$)",
        color=sim_colors[tol],
    )
    lower = np.maximum(np.array(means) - np.array(stds), 0)
    upper = np.array(means) + np.array(stds)
    plt.fill_between(variants, lower, upper, alpha=0.2, color=line.get_color())

plt.xlabel("Number of strains", fontsize=fontsize_text)
plt.ylabel("Mean simulation\nstop time $t$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.legend(
    fontsize=fontsize_ticks,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.2),
    ncol=1,
)
plt.grid(True, alpha=alpha_grid)
if savefig:
    plt.savefig(
        f"{plot_folder}/runtime_simulation_tstop.{fileformat}",
        dpi=dpi,
        bbox_inches="tight",
    )
plt.show()
plt.close()

# Plot 3: accuracy vs reference solution

# plt.figure()

# for means, mins, maxs, label, marker in (
#     (
#         means_err_ord,
#         mins_err_ord,
#         maxs_err_ord,
#         "find_stable (ordered=True)",
#         "o",
#     ),
#     (
#         means_err_unord,
#         mins_err_unord,
#         maxs_err_unord,
#         "find_stable (ordered=False)",
#         "s",
#     ),
#     *[
#         (
#             means_esim[tol],
#             mins_esim[tol],
#             maxs_esim[tol],
#             f"Simulation (tol={tol:.0e})",
#             m,
#         )
#         for tol, m in zip(deriv_tols, ["^", "v"])
#     ],
# ):
#     (line,) = plt.plot(variants, means, label=label, marker=marker)
#     plt.fill_between(variants, mins, maxs, alpha=0.2, color=line.get_color())

# plt.xlabel("Number of variants")
# plt.ylabel(
#     r"Mean maximal error in $\mathrm{I}_\mathrm{variant}$"
#     f" (N={N_parameters})"
# )
# plt.yscale("log")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# if savefig:
#     plt.savefig("../plots/runtime_accuracy.jpg", dpi=dpi)
# plt.show()

# Plot 4: max |R_inv - 1| of coexisting variants
plt.figure()

for means, mins, maxs, label, marker, color in (
    (
        means_dist_ord,
        mins_dist_ord,
        maxs_dist_ord,
        "Equation based",
        "o",
        "tab:blue",
    ),
    *[
        (
            means_dsim[tol],
            mins_dsim[tol],
            maxs_dsim[tol],
            f"Simulation (tol=$10^{{{int(np.log10(tol))}}}$)",
            m,
            sim_colors[tol],
        )
        for tol, m in zip(deriv_tols, ["^", "v"])
    ],
):
    plt.plot(variants, means, label=label, marker=marker, color=color)
    plt.fill_between(variants, mins, maxs, alpha=0.2, color=color)

plt.axhline(
    atol,
    color="gray",
    linestyle="--",
    label=rf"$\varepsilon = 10^{{{int(np.log10(atol))}}}$",
)
plt.xlabel("Number of strains", fontsize=fontsize_text)
plt.ylabel(
    "Mean maximal error in\n" r"$R_{0,k\to M}=1$ for $k\in M$",
    fontsize=fontsize_text,
)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.yscale("log")
plt.legend(
    fontsize=fontsize_ticks,
    loc="upper center",
    bbox_to_anchor=(0.4, -0.2),
    ncol=2,
)
plt.grid(True, alpha=alpha_grid)
if savefig:
    plt.savefig(
        f"{plot_folder}/runtime_inv_r0_distance.{fileformat}",
        dpi=dpi,
        bbox_inches="tight",
    )
plt.show()
plt.close()
