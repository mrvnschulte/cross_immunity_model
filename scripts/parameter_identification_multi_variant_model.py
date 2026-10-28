"""
Fit a multi-variant cross-immunity model to variant-specific incidence data.

The fitting procedure is sequential:

1. Fit Delta parameters.
2. Re-fit Delta and fit Omicron BA.1.
3. Re-fit all variants including Omicron BA.2.

The fitted parameters are:
- beta for each active variant
- gamma for each active variant
- infectious_at_start for each active variant
- one shared detection rate for all variants

The cross-immunity matrix sigma and birth/death rate b are fixed.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from cross_immunity_model import (
    cases_to_incidence,
    construct_matrices,
    create_initial_conditions_from_seroprevalence,
    cumulative_cases_to_incidence,
    multi_variant_model_with_matrices,
)


def read_data(filename: str | Path) -> pd.DataFrame:
    """
    Read incidence data from a CSV file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the CSV file. The file must contain a column named "Date".

    Returns
    -------
    pandas.DataFrame
        Data frame indexed by date.
    """
    return pd.read_csv(
        filename,
        parse_dates=["Date"],
        index_col="Date",
    )


def params_to_sigma(
    sigma_params: np.ndarray | list[float],
    n_variants: int,
) -> np.ndarray:
    """
    Construct a symmetric cross-immunity matrix from upper-triangular values.

    For three variants, the parameter order is:

        [sigma_12, sigma_13, sigma_23]

    Parameters
    ----------
    sigma_params : array-like
        Upper-triangular off-diagonal entries.
    n_variants : int
        Number of variants.

    Returns
    -------
    np.ndarray
        Symmetric n_variants x n_variants cross-immunity matrix.
    """
    sigma_params = np.asarray(sigma_params, dtype=float)

    n_sigma_params = n_variants * (n_variants - 1) // 2

    if len(sigma_params) != n_sigma_params:
        raise ValueError(
            f"Expected {n_sigma_params} sigma parameters, "
            f"but received {len(sigma_params)}."
        )

    sigma = np.eye(n_variants)

    parameter_index = 0

    for i in range(n_variants):
        for j in range(i + 1, n_variants):
            sigma[i, j] = sigma_params[parameter_index]
            sigma[j, i] = sigma_params[parameter_index]
            parameter_index += 1

    return sigma


def run_model(
    beta: np.ndarray | list[float],
    gamma: np.ndarray | list[float],
    sigma: np.ndarray,
    infectious_at_start: np.ndarray | list[float],
    fraction_antibodies: float,
    antibody_effectiveness: np.ndarray | list[float],
    t0: float,
    tf: float,
    population: int,
    b: float,
) -> np.ndarray:
    """
    Run the multi-variant model and return modelled incidence.

    Returns
    -------
    np.ndarray
        Modelled 7-day average incidence per 100,000 inhabitants.
    """
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    infectious_at_start = np.asarray(infectious_at_start, dtype=float)

    n_variants = len(beta)
    t_eval = np.arange(t0, tf)

    x0 = create_initial_conditions_from_seroprevalence(
        fraction_antibodies,
        antibody_effectiveness,
        infectious_at_start,
    )

    cumulative_cases_start = np.zeros(n_variants)
    y0 = np.concatenate([x0, cumulative_cases_start])

    V, G, T, S = construct_matrices(gamma, sigma)

    solution = solve_ivp(
        multi_variant_model_with_matrices,
        (t0, tf),
        y0,
        t_eval=t_eval,
        args=(beta, gamma, sigma, V, G, T, S, b, True),
        rtol=1e-6,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")

    cumulative_cases = solution.y.T[:, -n_variants:]

    return cumulative_cases_to_incidence(
        cumulative_cases,
        population,
    )


def objective_active_variants(
    params: np.ndarray,
    active_variant_indices: list[int],
    beta_initial: np.ndarray,
    gamma_initial: np.ndarray,
    infectious_initial: np.ndarray,
    sigma_fixed: np.ndarray,
    fraction_antibodies: float,
    antibody_effectiveness: np.ndarray,
    t0: float,
    tf: float,
    population: int,
    b: float,
    reported_incidence: np.ndarray,
) -> float:
    """
    Compute the sum of squared errors for one sequential fitting stage.

    Parameter order
    ---------------
    beta for active variants,
    gamma for active variants,
    log10(infectious_at_start) for active variants,
    shared detection rate.

    Parameters of inactive variants remain fixed at their initial values.
    The cross-immunity matrix sigma remains fixed.
    """
    n_active = len(active_variant_indices)

    beta_active = params[:n_active]
    gamma_active = params[n_active : 2 * n_active]
    log_infectious_active = params[2 * n_active : 3 * n_active]
    detection_rate = params[-1]

    beta_model = beta_initial.copy()
    gamma_model = gamma_initial.copy()
    infectious_model = infectious_initial.copy()

    for local_index, variant_index in enumerate(active_variant_indices):
        beta_model[variant_index] = beta_active[local_index]
        gamma_model[variant_index] = gamma_active[local_index]
        infectious_model[variant_index] = (
            10.0 ** log_infectious_active[local_index]
        )

    try:
        incidence = run_model(
            beta=beta_model,
            gamma=gamma_model,
            sigma=sigma_fixed,
            infectious_at_start=infectious_model,
            fraction_antibodies=fraction_antibodies,
            antibody_effectiveness=antibody_effectiveness,
            t0=t0,
            tf=tf,
            population=population,
            b=b,
        )
    except (FloatingPointError, RuntimeError, ValueError):
        return 1e12

    if not np.all(np.isfinite(incidence)):
        return 1e12

    detected_incidence = detection_rate * incidence

    n_points = min(
        len(detected_incidence),
        len(reported_incidence),
    )

    residuals = (
        detected_incidence[:n_points, active_variant_indices]
        - reported_incidence[:n_points, active_variant_indices]
    )

    return float(np.nansum(residuals**2))


def fit_stage(
    active_variant_indices: list[int],
    beta_start: np.ndarray,
    gamma_start: np.ndarray,
    infectious_start: np.ndarray,
    detection_start: float,
    beta_initial: np.ndarray,
    gamma_initial: np.ndarray,
    infectious_initial: np.ndarray,
    sigma_fixed: np.ndarray,
    fraction_antibodies: float,
    antibody_effectiveness: np.ndarray,
    t0: float,
    tf: float,
    population: int,
    b: float,
    reported_incidence: np.ndarray,
    variants: list[str],
    stage_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fit one sequential stage.

    Returns
    -------
    beta_opt : np.ndarray
        Fitted beta values for active variants.
    gamma_opt : np.ndarray
        Fitted gamma values for active variants.
    infectious_opt : np.ndarray
        Fitted initial infectious fractions for active variants.
    detection_rate_opt : float
        Shared fitted detection rate.
    """
    n_active = len(active_variant_indices)

    p0 = np.concatenate(
        [
            np.asarray(beta_start, dtype=float),
            np.asarray(gamma_start, dtype=float),
            np.log10(np.asarray(infectious_start, dtype=float)),
            [detection_start],
        ]
    )

    bounds = (
        [(0.1, 5.0)] * n_active
        + [(0.05, 0.5)] * n_active
        + [(-20.0, -1.0)] * n_active
        + [(0.01, 1.0)]
    )

    print()
    print("=" * 80)
    print(f"Starting {stage_name}")
    print("=" * 80)

    result = minimize(
        objective_active_variants,
        p0,
        args=(
            active_variant_indices,
            beta_initial,
            gamma_initial,
            infectious_initial,
            sigma_fixed,
            fraction_antibodies,
            antibody_effectiveness,
            t0,
            tf,
            population,
            b,
            reported_incidence,
        ),
        method="Nelder-Mead",
        bounds=bounds,
        options={
            "maxiter": 30000,
            "disp": True,
        },
    )

    beta_opt = result.x[:n_active]
    gamma_opt = result.x[n_active : 2 * n_active]

    log_infectious_opt = result.x[2 * n_active : 3 * n_active]
    infectious_opt = 10.0**log_infectious_opt

    detection_rate_opt = float(result.x[-1])

    print()
    print(f"{stage_name} results:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Final SSE: {result.fun:.6f}")
    print(f"  Shared detection rate: {detection_rate_opt:.6f}")

    for local_index, variant_index in enumerate(active_variant_indices):
        print(f"  {variants[variant_index]}:")
        print(f"    beta = {beta_opt[local_index]:.6f}")
        print(f"    gamma = {gamma_opt[local_index]:.6f}")
        print(
            "    infectious_at_start = " f"{infectious_opt[local_index]:.6e}"
        )

    return (
        beta_opt,
        gamma_opt,
        infectious_opt,
        detection_rate_opt,
    )


def plot_initial_model(
    dates_reported: pd.DatetimeIndex,
    reported_incidence: np.ndarray,
    dates_model: pd.DatetimeIndex,
    incidence_model: np.ndarray,
    variants: list[str],
    colors: list[str],
    fontsize_text: int,
    fontsize_ticks: int,
    alpha_grid: float,
    dpi: int,
    filename: str | Path | None = None,
    fileformat: str = "pdf",
) -> None:
    """Plot reported incidence and incidence from initial parameter values."""
    fig = plt.figure()

    for i, variant in enumerate(variants):
        plt.plot(
            dates_reported,
            reported_incidence[:, i],
            label=f"{variant} (Data)",
            color=colors[i],
            marker="o",
            markersize=2,
            linestyle="none",
        )

        plt.plot(
            dates_model,
            incidence_model[:, i],
            label=f"{variant} (Initial model)",
            color=colors[i],
            linestyle="--",
            linewidth=2,
        )

    plt.xlabel("Date", fontsize=fontsize_text)
    plt.ylabel(
        "Cases per 100,000\n(7-day moving average)",
        fontsize=fontsize_text,
    )

    plt.xlim(dates_model[0], dates_model[-1])
    plt.ylim(bottom=0)
    plt.xticks(fontsize=fontsize_ticks, rotation=90)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid(alpha=alpha_grid)

    fig.legend(
        fontsize=fontsize_ticks,
        ncol=2,
        bbox_to_anchor=(0, 0.05),
        loc="upper left",
    )

    plt.tight_layout()

    if filename is not None:
        filename = Path(filename).with_suffix(f".{fileformat}")
        filename.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            filename,
            dpi=dpi,
            bbox_inches="tight",
        )

    plt.show()
    plt.close()


def plot_final_model(
    dates_reported: pd.DatetimeIndex,
    reported_incidence: np.ndarray,
    dates_model: pd.DatetimeIndex,
    detected_incidence: np.ndarray,
    variants: list[str],
    colors: list[str],
    fontsize_text: int,
    fontsize_ticks: int,
    alpha_grid: float,
    dpi: int,
    filename: str | Path,
    fileformat: str,
) -> None:
    """Plot reported incidence and the final fitted model."""
    fig = plt.figure()

    for i, variant in enumerate(variants):
        plt.plot(
            dates_reported,
            reported_incidence[:, i],
            label=f"{variant} (Data)",
            color=colors[i],
            marker="o",
            markersize=2,
            linestyle="none",
        )

        plt.plot(
            dates_model,
            detected_incidence[:, i],
            label=f"{variant} (Model)",
            color=colors[i],
            linestyle="-",
            linewidth=2,
        )

    plt.xlabel("Date", fontsize=fontsize_text)
    plt.ylabel(
        "Cases per 100,000\n(7-day moving average)",
        fontsize=fontsize_text,
    )

    plt.xlim(dates_model[0], dates_model[-1])
    plt.ylim(bottom=0)
    plt.xticks(fontsize=fontsize_ticks, rotation=90)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid(alpha=alpha_grid)

    fig.legend(
        fontsize=fontsize_ticks,
        ncol=2,
        bbox_to_anchor=(0, 0.05),
        loc="upper left",
    )

    plt.tight_layout()

    filename = Path(filename).with_suffix(f".{fileformat}")
    filename.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        filename,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()


def save_fitted_parameters(
    filename: str | Path,
    variants: list[str],
    beta: np.ndarray,
    gamma: np.ndarray,
    infectious_at_start: np.ndarray,
    sigma: np.ndarray,
    detection_rate: float,
) -> None:
    """Save final fitted and fixed model parameters to a CSV file."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    n_variants = len(variants)

    fieldnames = [
        "variant",
        "beta",
        "gamma",
        "infectious_at_start",
        "detection_rate",
    ]

    for i in range(n_variants):
        for j in range(i + 1, n_variants):
            fieldnames.append(f"sigma_{i + 1}{j + 1}")

    with filename.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for i, variant in enumerate(variants):
            row = {
                "variant": variant,
                "beta": beta[i],
                "gamma": gamma[i],
                "infectious_at_start": infectious_at_start[i],
                "detection_rate": detection_rate,
            }

            for first_index in range(n_variants):
                for second_index in range(first_index + 1, n_variants):
                    row[f"sigma_{first_index + 1}{second_index + 1}"] = sigma[
                        first_index,
                        second_index,
                    ]

            writer.writerow(row)


# =============================================================================
# Plot settings
# =============================================================================

plot_folder = Path("../plots")
data_folder = Path("../data")

fileformat = "pdf"
dpi = 1200

fontsize_text = 16
fontsize_ticks = 14
alpha_grid = 0.7

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
]

initial_plot_filename = (
    plot_folder / "multi_variant_parameter_identification_initial"
)

final_plot_filename = (
    plot_folder / "multi_variant_parameter_identification_final"
)

fitted_parameters_filename = (
    data_folder / "covid_multi_variant_fitted_parameters.csv"
)


# =============================================================================
# Configuration
# =============================================================================

filename_incidence = (
    "D:/Projects/Dissertation/Assignment of Cases to Variants/"
    "data/moving_average_per_variant.csv"
)

start_date = "2021-11-01"

variants = [
    "Delta",
    "Omicron BA.1",
    "Omicron BA.2",
]

t0 = 0.0
tf = 210.0
population = 83_497_147
b = 3.3e-5

fraction_antibodies = 0.894

antibody_effectiveness = np.array(
    [
        0.80,
        0.30,
        0.30,
    ]
)

beta_initial = np.array(
    [
        0.50,
        0.50,
        0.50,
    ]
)

gamma_initial = np.array(
    [
        0.15,
        0.15,
        0.15,
    ]
)

infectious_initial = np.array(
    [
        1e-3,
        1e-8,
        1e-15,
    ]
)

# Fixed cross-immunity parameters:
# sigma_12 = Delta <-> Omicron BA.1
# sigma_13 = Delta <-> Omicron BA.2
# sigma_23 = Omicron BA.1 <-> Omicron BA.2
sigma_upper_fixed = np.array(
    [
        0.571,
        0.571,
        0.85,
    ]
)

n_variants = len(variants)

sigma_fixed = params_to_sigma(
    sigma_upper_fixed,
    n_variants,
)


# =============================================================================
# Read reported data
# =============================================================================

df_reported = read_data(filename_incidence)

start = pd.Timestamp(start_date)
end_date = start + pd.Timedelta(days=int(tf) - 1)

df_reported_slice = df_reported.loc[
    start:end_date,
    variants,
]

reported_incidence = cases_to_incidence(
    df_reported_slice.values,
    population,
)


# =============================================================================
# Initial model run and plot
# =============================================================================

incidence_initial = run_model(
    beta=beta_initial,
    gamma=gamma_initial,
    sigma=sigma_fixed,
    infectious_at_start=infectious_initial,
    fraction_antibodies=fraction_antibodies,
    antibody_effectiveness=antibody_effectiveness,
    t0=t0,
    tf=tf,
    population=population,
    b=b,
)

dates_initial = pd.date_range(
    start=start,
    periods=len(incidence_initial),
    freq="D",
)

plot_initial_model(
    dates_reported=df_reported_slice.index,
    reported_incidence=reported_incidence,
    dates_model=dates_initial,
    incidence_model=incidence_initial,
    variants=variants,
    colors=colors,
    fontsize_text=fontsize_text,
    fontsize_ticks=fontsize_ticks,
    alpha_grid=alpha_grid,
    dpi=dpi,
    filename=initial_plot_filename,
    fileformat=fileformat,
)


# =============================================================================
# Stage 1: Fit Delta
# =============================================================================

(
    beta_stage_1,
    gamma_stage_1,
    infectious_stage_1,
    detection_stage_1,
) = fit_stage(
    active_variant_indices=[0],
    beta_start=beta_initial[[0]],
    gamma_start=gamma_initial[[0]],
    infectious_start=infectious_initial[[0]],
    detection_start=0.50,
    beta_initial=beta_initial,
    gamma_initial=gamma_initial,
    infectious_initial=infectious_initial,
    sigma_fixed=sigma_fixed,
    fraction_antibodies=fraction_antibodies,
    antibody_effectiveness=antibody_effectiveness,
    t0=t0,
    tf=tf,
    population=population,
    b=b,
    reported_incidence=reported_incidence,
    variants=variants,
    stage_name="Stage 1: Delta",
)


# =============================================================================
# Stage 2: Re-fit Delta and fit Omicron BA.1
# =============================================================================

(
    beta_stage_2,
    gamma_stage_2,
    infectious_stage_2,
    detection_stage_2,
) = fit_stage(
    active_variant_indices=[0, 1],
    beta_start=np.array(
        [
            beta_stage_1[0],
            beta_initial[1],
        ]
    ),
    gamma_start=np.array(
        [
            gamma_stage_1[0],
            gamma_initial[1],
        ]
    ),
    infectious_start=np.array(
        [
            infectious_stage_1[0],
            infectious_initial[1],
        ]
    ),
    detection_start=detection_stage_1,
    beta_initial=beta_initial,
    gamma_initial=gamma_initial,
    infectious_initial=infectious_initial,
    sigma_fixed=sigma_fixed,
    fraction_antibodies=fraction_antibodies,
    antibody_effectiveness=antibody_effectiveness,
    t0=t0,
    tf=tf,
    population=population,
    b=b,
    reported_incidence=reported_incidence,
    variants=variants,
    stage_name="Stage 2: Delta + Omicron BA.1",
)


# =============================================================================
# Stage 3: Re-fit all variants including Omicron BA.2
# =============================================================================

(
    beta_final,
    gamma_final,
    infectious_final,
    detection_rate_final,
) = fit_stage(
    active_variant_indices=[0, 1, 2],
    beta_start=np.array(
        [
            beta_stage_2[0],
            beta_stage_2[1],
            beta_initial[2],
        ]
    ),
    gamma_start=np.array(
        [
            gamma_stage_2[0],
            gamma_stage_2[1],
            gamma_initial[2],
        ]
    ),
    infectious_start=np.array(
        [
            infectious_stage_2[0],
            infectious_stage_2[1],
            infectious_initial[2],
        ]
    ),
    detection_start=detection_stage_2,
    beta_initial=beta_initial,
    gamma_initial=gamma_initial,
    infectious_initial=infectious_initial,
    sigma_fixed=sigma_fixed,
    fraction_antibodies=fraction_antibodies,
    antibody_effectiveness=antibody_effectiveness,
    t0=t0,
    tf=tf,
    population=population,
    b=b,
    reported_incidence=reported_incidence,
    variants=variants,
    stage_name="Stage 3: Delta + Omicron BA.1 + Omicron BA.2",
)


# =============================================================================
# Print final parameter estimates
# =============================================================================

print()
print("=" * 80)
print("Final fitted parameters")
print("=" * 80)

for i, variant in enumerate(variants):
    print(f"{variant}:")
    print(f"  beta = {beta_final[i]:.6f}")
    print(f"  gamma = {gamma_final[i]:.6f}")
    print(f"  infectious_at_start = {infectious_final[i]:.6e}")

print()
print(f"Shared detection rate = {detection_rate_final:.6f}")

print()
print("Fixed sigma matrix:")
print(sigma_fixed)


# =============================================================================
# Final model run and plot
# =============================================================================

incidence_final = run_model(
    beta=beta_final,
    gamma=gamma_final,
    sigma=sigma_fixed,
    infectious_at_start=infectious_final,
    fraction_antibodies=fraction_antibodies,
    antibody_effectiveness=antibody_effectiveness,
    t0=t0,
    tf=tf,
    population=population,
    b=b,
)

detected_incidence_final = detection_rate_final * incidence_final

dates_final = pd.date_range(
    start=start,
    periods=len(detected_incidence_final),
    freq="D",
)

plot_final_model(
    dates_reported=df_reported_slice.index,
    reported_incidence=reported_incidence,
    dates_model=dates_final,
    detected_incidence=detected_incidence_final,
    variants=variants,
    colors=colors,
    fontsize_text=fontsize_text,
    fontsize_ticks=fontsize_ticks,
    alpha_grid=alpha_grid,
    dpi=dpi,
    filename=final_plot_filename,
    fileformat=fileformat,
)


# =============================================================================
# Save final fitted parameters
# =============================================================================

save_fitted_parameters(
    filename=fitted_parameters_filename,
    variants=variants,
    beta=beta_final,
    gamma=gamma_final,
    infectious_at_start=infectious_final,
    sigma=sigma_fixed,
    detection_rate=detection_rate_final,
)

print(f"\nFitted parameters saved to: {fitted_parameters_filename}")
