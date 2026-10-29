"""
Fit a sequential multi-variant cross-immunity model to variant-specific
COVID-19 incidence data.

The fitting procedure is sequential:

1. Fit Delta.
2. Re-fit Delta and fit Omicron BA.1.
3. Re-fit all variants including Omicron BA.2.

Fitted parameters
------------------
For every active variant:
- beta
- gamma
- infectious_at_start

Shared across all active variants:
- detection_rate

The detection_rate scales modelled infection events to reported COVID-19
incidence.

Fixed parameters
----------------
- fraction_antibodies
- antibody_effectiveness
- cross-immunity matrix sigma
- birth/death rate b

Input data are daily variant-specific COVID-19 case estimates. Case estimates
are converted to incidence per 100,000 inhabitants. Modelled cumulative
infection counts are converted to daily infections and smoothed using a
seven-day moving average before fitting.
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
    Reads variant-specific COVID-19 incidence data from a CSV file.

    The input file must contain a column named ``"Date"``. This column is
    parsed as dates and used as the data frame index. All remaining columns
    are retained unchanged.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the CSV file containing incidence data.

    Returns
    -------
    pd.DataFrame
        Data frame indexed by the parsed ``"Date"`` column.

    Raises
    ------
    FileNotFoundError
        If ``filename`` does not exist.
    ValueError
        If the input file does not contain a ``"Date"`` column or dates cannot
        be parsed.
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
    Constructs a symmetric cross-immunity matrix from a flat vector of
    upper-triangular off-diagonal entries.

    The parameter order follows the upper triangle of the matrix row by row.
    For three variants, the order is ``sigma_12``, ``sigma_13``, and
    ``sigma_23``.

    Parameters
    ----------
    sigma_params : np.ndarray or list of float
        Cross-immunity values for the upper-triangular off-diagonal entries.
        Must contain ``n_variants * (n_variants - 1) // 2`` values.
    n_variants : int
        Number of COVID-19 variants and therefore the dimension of the
        resulting cross-immunity matrix.

    Returns
    -------
    sigma : np.ndarray
        Symmetric cross-immunity matrix of shape
        ``(n_variants, n_variants)``. Diagonal entries are equal to 1.

    Raises
    ------
    ValueError
        If the number of supplied cross-immunity parameters is incorrect.
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
    Simulates the multi-variant model and returns the modelled incidence.

    The model is integrated with daily output. Cumulative infection counters
    are transformed into daily infections, smoothed using a seven-day moving
    average, and converted to incidence per 100,000 inhabitants.

    Parameters
    ----------
    beta : np.ndarray or list of float
        Transmission rates, with one entry per variant.
    gamma : np.ndarray or list of float
        Recovery rates, with one entry per variant. Must have the same length
        as ``beta``.
    sigma : np.ndarray
        Cross-immunity matrix of shape ``(n_variants, n_variants)``.
    infectious_at_start : np.ndarray or list of float
        Initial infectious fractions, with one entry per variant. Must have
        the same length as ``beta``.
    fraction_antibodies : float
        Fraction of the population with antibodies at the start of the
        simulation.
    antibody_effectiveness : np.ndarray or list of float
        Protection probabilities against each variant for individuals with
        antibodies. Must have one entry per variant.
    t0 : float
        Start time of the simulation in days.
    tf : float
        End time of the simulation in days. Daily model output is evaluated
        for times from ``t0`` up to, but excluding, ``tf``.
    population : int
        Total population size used to convert modelled fractions into
        incidence per 100,000 inhabitants.
    b : float
        Natural birth and death rate of the model.

    Returns
    -------
    incidence : np.ndarray
        Modelled seven-day average incidence per 100,000 inhabitants. Rows
        correspond to time points and columns correspond to variants.

    Raises
    ------
    RuntimeError
        If numerical integration of the ODE system fails.
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
    Computes the sum of squared incidence residuals for one sequential fitting
    stage.

    The parameter vector is interpreted in the following order:

    1. Transmission rates ``beta`` of active variants.
    2. Recovery rates ``gamma`` of active variants.
    3. Base-10 logarithms of the initial infectious fractions of active
       variants.
    4. One shared detection rate.

    Parameters of inactive variants remain fixed at their supplied initial
    values. The cross-immunity matrix and antibody effectiveness values remain
    fixed throughout the optimization.

    Parameters
    ----------
    params : np.ndarray
        Flat optimization parameter vector containing the parameters of active
        variants and the shared detection rate.
    active_variant_indices : list of int
        Indices of variants whose ``beta``, ``gamma``, and initial infectious
        fractions are optimized in the current fitting stage.
    beta_initial : np.ndarray
        Full vector of baseline transmission rates. Entries of inactive
        variants remain unchanged.
    gamma_initial : np.ndarray
        Full vector of baseline recovery rates. Entries of inactive variants
        remain unchanged.
    infectious_initial : np.ndarray
        Full vector of baseline initial infectious fractions. Entries of
        inactive variants remain unchanged.
    sigma_fixed : np.ndarray
        Fixed cross-immunity matrix used for all model evaluations.
    fraction_antibodies : float
        Fraction of the population with antibodies at the start of the
        simulation.
    antibody_effectiveness : np.ndarray
        Fixed protection probabilities against each variant for individuals
        with antibodies.
    t0 : float
        Start time of the fitting simulation in days.
    tf : float
        End time of the fitting simulation in days.
    population : int
        Total population size used to calculate incidence.
    b : float
        Natural birth and death rate of the model.
    reported_incidence : np.ndarray
        Observed variant-specific incidence values in cases per 100,000
        inhabitants. Columns must follow the model variant ordering.

    Returns
    -------
    float
        Sum of squared residuals between detected modelled incidence and
        observed incidence for active variants. Returns ``1e12`` if model
        evaluation fails or produces non-finite values.
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
    Fits the parameters of active variants in one sequential optimization
    stage.

    The Nelder-Mead optimizer minimizes the sum of squared incidence
    residuals returned by ``objective_active_variants``. Transmission rates,
    recovery rates, and initial infectious fractions are fitted for active
    variants. A single detection rate is fitted jointly for all active
    variants.

    Parameters
    ----------
    active_variant_indices : list of int
        Indices of variants whose parameters are fitted in this stage.
    beta_start : np.ndarray
        Initial optimization values for transmission rates of active variants.
    gamma_start : np.ndarray
        Initial optimization values for recovery rates of active variants.
    infectious_start : np.ndarray
        Initial optimization values for infectious fractions of active
        variants.
    detection_start : float
        Initial optimization value for the shared detection rate.
    beta_initial : np.ndarray
        Full vector of transmission rates used as fixed values for inactive
        variants.
    gamma_initial : np.ndarray
        Full vector of recovery rates used as fixed values for inactive
        variants.
    infectious_initial : np.ndarray
        Full vector of initial infectious fractions used as fixed values for
        inactive variants.
    sigma_fixed : np.ndarray
        Fixed cross-immunity matrix.
    fraction_antibodies : float
        Fraction of the population with antibodies at the start of the
        simulation.
    antibody_effectiveness : np.ndarray
        Fixed protection probabilities against each variant for individuals
        with antibodies.
    t0 : float
        Start time of the fitting simulation in days.
    tf : float
        End time of the fitting simulation in days.
    population : int
        Total population size used to calculate incidence.
    b : float
        Natural birth and death rate of the model.
    reported_incidence : np.ndarray
        Observed variant-specific incidence values. Columns must correspond to
        the model variant ordering.
    variants : list of str
        Variant names used when printing fitted parameter values.
    stage_name : str
        Descriptive name of the current optimization stage.

    Returns
    -------
    beta_opt : np.ndarray
        Fitted transmission rates for active variants.
    gamma_opt : np.ndarray
        Fitted recovery rates for active variants.
    infectious_opt : np.ndarray
        Fitted initial infectious fractions for active variants.
    detection_rate_opt : float
        Fitted detection rate shared by all active variants.
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
    """
    Plots observed incidence and incidence from the initial parameter values.

    Points represent reported variant-specific incidence values. Dashed lines
    represent modelled incidence before parameter optimization.

    Parameters
    ----------
    dates_reported : pd.DatetimeIndex
        Dates corresponding to the reported incidence data.
    reported_incidence : np.ndarray
        Observed incidence values of shape ``(n_times, n_variants)`` in cases
        per 100,000 inhabitants.
    dates_model : pd.DatetimeIndex
        Dates corresponding to the modelled incidence values.
    incidence_model : np.ndarray
        Initial modelled incidence of shape ``(n_times, n_variants)`` in cases
        per 100,000 inhabitants.
    variants : list of str
        Variant names in the column order of the incidence arrays.
    colors : list of str
        Matplotlib color specifications, with one entry per variant.
    fontsize_text : int
        Font size used for axis labels.
    fontsize_ticks : int
        Font size used for tick labels and the legend.
    alpha_grid : float
        Transparency of the plot grid.
    dpi : int
        Resolution of the saved figure in dots per inch.
    filename : str or pathlib.Path or None, optional
        Output filename without or with extension. If None, the figure is
        shown but not saved. The default is None.
    fileformat : str, optional
        File format used when saving the figure. The filename suffix is
        replaced by this format. The default is ``"pdf"``.

    Returns
    -------
    None.
    """
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
    """
    Plots observed incidence and the final fitted model.

    Points represent reported variant-specific incidence values. Solid lines
    represent modelled incidence after multiplication by the fitted detection
    rate.

    Parameters
    ----------
    dates_reported : pd.DatetimeIndex
        Dates corresponding to the reported incidence data.
    reported_incidence : np.ndarray
        Observed incidence values of shape ``(n_times, n_variants)`` in cases
        per 100,000 inhabitants.
    dates_model : pd.DatetimeIndex
        Dates corresponding to the modelled incidence values.
    detected_incidence : np.ndarray
        Modelled incidence after scaling by the fitted detection rate. Must
        have shape ``(n_times, n_variants)``.
    variants : list of str
        Variant names in the column order of the incidence arrays.
    colors : list of str
        Matplotlib color specifications, with one entry per variant.
    fontsize_text : int
        Font size used for axis labels.
    fontsize_ticks : int
        Font size used for tick labels and the legend.
    alpha_grid : float
        Transparency of the plot grid.
    dpi : int
        Resolution of the saved figure in dots per inch.
    filename : str or pathlib.Path
        Output filename without or with extension. Parent directories are
        created if necessary.
    fileformat : str
        File format used when saving the figure. The filename suffix is
        replaced by this format.

    Returns
    -------
    None.
    """
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
    """
    Saves fitted and fixed COVID-19 model parameters to a CSV file.

    The output contains one row per variant. The shared detection rate and all
    upper-triangular cross-immunity values are repeated in every row.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path of the CSV file to create. Parent directories are created if
        necessary.
    variants : list of str
        Variant names in the ordering used by all parameter arrays.
    beta : np.ndarray
        Fitted transmission rates, with one entry per variant.
    gamma : np.ndarray
        Fitted recovery rates, with one entry per variant.
    infectious_at_start : np.ndarray
        Fitted initial infectious fractions, with one entry per variant.
    sigma : np.ndarray
        Fixed or fitted cross-immunity matrix of shape
        ``(n_variants, n_variants)``.
    detection_rate : float
        Shared fitted proportion of modelled infections interpreted as detected
        reported cases.

    Returns
    -------
    None.
    """
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


# Plot settings
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

# Configuration
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

# Initial values for optimization
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
detection_initial = 0.5

# Read reported data
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


# Initial model run and plot
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

# Stage 1: Fit Delta
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
    detection_start=detection_initial,
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

# Stage 2: Re-fit Delta and fit Omicron BA.1
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

# Stage 3: Re-fit all variants including Omicron BA.2
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

# Print final parameter estimates
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

# Final model run and plot
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

# Save final fitted parameters
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
