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

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from cross_immunity_model import (
    construct_matrices,
    find_stable_multi_variant_equilibrium,
    multi_variant_model_with_matrices,
    sample_random_parameters,
    solve_multi_variant_equilibrium_equation,
    infectious_per_variant,
    invasion_reproduction_number_multiple_variants,
)


# --------------------------------------------------------------------------- #
# Simulation-based equilibrium finder
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Benchmark settings
# --------------------------------------------------------------------------- #

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
dpi = 500
savefig = True
verbose = True
verbose_step = 10

variants = list(range(1, n_variants_max + 1))

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


# --------------------------------------------------------------------------- #
# Benchmark loop
# --------------------------------------------------------------------------- #

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

        # --- Method 1: ordered_check=True ---
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

        # --- Method 2: ordered_check=False ---
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

        # --- Method 3: Simulation (all deriv_tol values) ---
        for tol in deriv_tols:
            t_start = time.perf_counter()
            x_eq_sim, t_stop_arr[tol][i] = find_equilibrium_by_simulation(
                beta, gamma, sigma, V, G, T, S, b, x0, deriv_tol=tol
            )
            t_sim[tol][i] = time.perf_counter() - t_start

            I_var_sim = infectious_per_variant(x_eq_sim, V, matrix_form=False)

            # Reference solution (computed only once)
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
                    I_variant_guess=I_var_ord,
                    atol=atol_ref,
                    free_indices=coexisting_variants,
                )
                err_ord[i] = np.linalg.norm(I_var_ord - I_var_ref)
                err_unord[i] = np.linalg.norm(I_var_unord - I_var_ref)

            err_sim[tol][i] = np.linalg.norm(I_var_sim - I_var_ref)

            inv_r0_sim = invasion_reproduction_number_multiple_variants(
                x_eq_sim, beta, gamma, b, V
            )

            if len(coexisting_variants) > 0:
                dist_sim[tol][i] = np.max(
                    np.abs(inv_r0_sim[coexisting_variants] - 1)
                )
            else:
                dist_sim[tol][i] = 0.0

        inv_r0_ord = invasion_reproduction_number_multiple_variants(
            x_eq_ord, beta, gamma, b, V
        )
        inv_r0_unord = invasion_reproduction_number_multiple_variants(
            x_eq_unord, beta, gamma, b, V
        )

        if len(coexisting_variants) > 0:
            dist_ord[i] = np.max(np.abs(inv_r0_ord[coexisting_variants] - 1))
            dist_unord[i] = np.max(np.abs(inv_r0_unord[coexisting_unord] - 1))
        else:
            dist_ord[i] = 0.0
            dist_unord[i] = 0.0

        if verbose and (i + 1) % verbose_step == 0:
            print(f"  {i+1}/{N_parameters} parameter sets done")

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
        print(f"Finished benchmarks for {n_variants} variant(s).")


# --------------------------------------------------------------------------- #
# Aggregate statistics
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Plot 1: wall-clock runtimes
# --------------------------------------------------------------------------- #

plt.figure()
sim_colors = {}

for means, stds, label, marker in (
    (means_ordered, stds_ordered, "find_stable (ordered=True)", "o"),
    (means_unordered, stds_unordered, "find_stable (ordered=False)", "s"),
    *[
        (means_sim[tol], stds_sim[tol], f"Simulation (tol={tol:.0e})", m)
        for tol, m in zip(deriv_tols, ["^", "v"])
    ],
):
    (line,) = plt.plot(variants, means, label=label, marker=marker)
    lower = np.maximum(np.array(means) - np.array(stds), 0)
    upper = np.array(means) + np.array(stds)
    plt.fill_between(variants, lower, upper, alpha=0.2, color=line.get_color())

    for tol in deriv_tols:
        if label == f"Simulation (tol={tol:.0e})":
            sim_colors[tol] = line.get_color()  # <--

plt.xlabel("Number of variants")
plt.ylabel(f"Mean runtime [s] (N={N_parameters})")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
if savefig:
    plt.savefig("../plots/runtime_wallclock.jpg", dpi=dpi)
plt.show()

# --------------------------------------------------------------------------- #
# Plot 2: simulation stopping time t_stop
# --------------------------------------------------------------------------- #

plt.figure()

for tol, marker in zip(deriv_tols, ["^", "v"]):
    means = means_tstop[tol]
    stds = stds_tstop[tol]
    (line,) = plt.plot(
        variants,
        means,
        marker=marker,
        label=f"Simulation (tol={tol:.0e})",
        color=sim_colors[tol],
    )
    lower = np.maximum(np.array(means) - np.array(stds), 0)
    upper = np.array(means) + np.array(stds)
    plt.fill_between(variants, lower, upper, alpha=0.2, color=line.get_color())

plt.xlabel("Number of variants")
plt.ylabel(f"Mean simulation stop time $t$ (N={N_parameters})")
plt.legend()
plt.grid(True)
plt.tight_layout()
if savefig:
    plt.savefig("../plots/runtime_simulation_tstop.jpg", dpi=dpi)
plt.show()

# --------------------------------------------------------------------------- #
# Plot 3: accuracy vs reference solution
# --------------------------------------------------------------------------- #

plt.figure()

for means, mins, maxs, label, marker in (
    (
        means_err_ord,
        mins_err_ord,
        maxs_err_ord,
        "find_stable (ordered=True)",
        "o",
    ),
    (
        means_err_unord,
        mins_err_unord,
        maxs_err_unord,
        "find_stable (ordered=False)",
        "s",
    ),
    *[
        (
            means_esim[tol],
            mins_esim[tol],
            maxs_esim[tol],
            f"Simulation (tol={tol:.0e})",
            m,
        )
        for tol, m in zip(deriv_tols, ["^", "v"])
    ],
):
    (line,) = plt.plot(variants, means, label=label, marker=marker)
    plt.fill_between(variants, mins, maxs, alpha=0.2, color=line.get_color())

plt.xlabel("Number of variants")
plt.ylabel(
    r"Mean maximal error in $\mathrm{I}_\mathrm{variant}$"
    f" (N={N_parameters})"
)
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
if savefig:
    plt.savefig("../plots/runtime_accuracy.jpg", dpi=dpi)
plt.show()

# --------------------------------------------------------------------------- #
# Plot 4: max |R_inv - 1| of coexisting variants
# --------------------------------------------------------------------------- #

plt.figure()

for means, mins, maxs, label, marker in (
    (
        means_dist_ord,
        mins_dist_ord,
        maxs_dist_ord,
        "find_stable (ordered=True)",
        "o",
    ),
    (
        means_dist_unord,
        mins_dist_unord,
        maxs_dist_unord,
        "find_stable (ordered=False)",
        "s",
    ),
    *[
        (
            means_dsim[tol],
            mins_dsim[tol],
            maxs_dsim[tol],
            f"Simulation (tol={tol:.0e})",
            m,
        )
        for tol, m in zip(deriv_tols, ["^", "v"])
    ],
):
    (line,) = plt.plot(variants, means, label=label, marker=marker)
    plt.fill_between(variants, mins, maxs, alpha=0.2, color=line.get_color())

plt.axhline(
    atol, color="gray", linestyle="--", linewidth=1, label=f"atol = {atol}"
)
plt.xlabel("Number of variants")
plt.ylabel(
    r"Mean maximal error in $R_{0,\mathrm{inv}}$"
    "\nof coexisting variants"
    f" (N={N_parameters})"
)
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
if savefig:
    plt.savefig("../plots/runtime_inv_r0_distance.jpg", dpi=dpi)
plt.show()
