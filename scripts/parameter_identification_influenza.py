"""
Fit a sequential multi-variant cross-immunity model to strain-specific
influenza case estimates.

The fitting procedure is sequential:

1. Fit A(H1N1).
2. Re-fit A(H1N1) and fit A(H3).
3. Re-fit A(H1N1), A(H3), and fit B/Victoria.
4. Re-fit all variants including B/Yamagata.

Fitted parameters
------------------
For every active variant:
- beta
- gamma
- infectious_at_start
- antibody_effectiveness

Shared across all active variants:
- symptomatic_rate

The symptomatic_rate is analogous to the detection_rate in the COVID fitting
script. It scales modelled infection events to estimated symptomatic influenza
cases.

Fixed parameters
----------------
- fraction_antibodies
- cross-immunity matrix sigma
- birth/death rate b

Input data are weekly estimated influenza cases per strain. The fitting period
is restricted to November through May.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from cross_immunity_model import (
    construct_matrices,
    create_initial_conditions_from_seroprevalence,
    multi_variant_model_with_matrices,
)


### Data handling


def epidemiological_week_date(year: int, week: int) -> pd.Timestamp:
    """
    Returns the Monday corresponding to an ISO epidemiological week.

    Parameters
    ----------
    year : int
        ISO week-numbering year.
    week : int
        ISO week number in the range from 1 to 53.

    Returns
    -------
    pd.Timestamp
        Timestamp of the Monday of the specified ISO epidemiological week.
    """
    return pd.Timestamp(date.fromisocalendar(int(year), int(week), 1))


def read_and_filter_weekly_cases(
    filename: str | Path,
    variants: list[str],
    flu_season: str,
) -> pd.DataFrame:
    """
    Reads strain-specific weekly influenza case estimates and restricts them
    to the period from November through May.

    The input file must contain the columns ``year``, ``week``,
    ``calendar_week``, ``estimated_cases``, and one column for every variant
    in ``variants``.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the CSV file containing weekly influenza case estimates.
    variants : list of str
        Names of influenza variants. Each name must correspond to a column in
        the input file containing strain-specific estimated case counts.
    flu_season : str
        Influenza season in the format ``"YYYY-YYYY"``, for example
        ``"2017-2018"``. The first year determines the start of the fitting
        period in November.

    Returns
    -------
    pd.DataFrame
        Weekly case data sorted by ISO year and week, with an additional
        ``date`` column containing the Monday of each epidemiological week.
        Only observations between 1 November of the first season year and
        31 May of the following year are retained.

    Raises
    ------
    ValueError
        If required columns are missing, strain-specific case counts contain
        missing or negative values, or no observations remain after filtering.
    """
    data = pd.read_csv(filename)

    required_columns = {
        "year",
        "week",
        "calendar_week",
        "estimated_cases",
        *variants,
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Input file is missing columns: {sorted(missing_columns)}"
        )

    data = data.sort_values(["year", "week"]).reset_index(drop=True)

    if data[variants].isna().any().any():
        raise ValueError("Strain-specific case data contain missing values.")

    if (data[variants] < 0).any().any():
        raise ValueError("Strain-specific case data contain negative values.")

    data["date"] = pd.DatetimeIndex(
        [
            epidemiological_week_date(year, week)
            for year, week in zip(data["year"], data["week"])
        ]
    )

    season_start_year = int(flu_season.split("-")[0])

    fit_start_date = pd.Timestamp(
        year=season_start_year,
        month=11,
        day=1,
    )

    fit_end_date = pd.Timestamp(
        year=season_start_year + 1,
        month=5,
        day=31,
    )

    data = data.loc[data["date"].between(fit_start_date, fit_end_date)].copy()

    if data.empty:
        raise ValueError(
            "No observations remain after restricting the fitting period to "
            f"{fit_start_date.date()} through {fit_end_date.date()}."
        )

    return data


### Cross-immunity matrix from flat parameter vector


def params_to_sigma(
    sigma_params: np.ndarray | list[float],
    n_variants: int,
) -> np.ndarray:
    """
    Constructs a symmetric cross-immunity matrix from a flat vector of
    upper-triangular off-diagonal entries.

    The parameter order follows the upper triangle of the matrix row by row.
    For four variants, the order is ``sigma_12``, ``sigma_13``,
    ``sigma_14``, ``sigma_23``, ``sigma_24``, and ``sigma_34``.

    Parameters
    ----------
    sigma_params : np.ndarray or list of float
        Cross-immunity values for the upper-triangular off-diagonal entries.
        Must contain ``n_variants * (n_variants - 1) // 2`` values, each in
        the interval [0, 1].
    n_variants : int
        Number of influenza variants and therefore the dimension of the
        resulting cross-immunity matrix.

    Returns
    -------
    sigma : np.ndarray
        Symmetric cross-immunity matrix of shape
        ``(n_variants, n_variants)``. Diagonal entries are equal to 1.

    Raises
    ------
    ValueError
        If the number of supplied cross-immunity parameters is incorrect or
        any parameter lies outside the interval [0, 1].
    """
    sigma_params = np.asarray(sigma_params, dtype=float)

    n_sigma_params = n_variants * (n_variants - 1) // 2

    if len(sigma_params) != n_sigma_params:
        raise ValueError(
            f"Expected {n_sigma_params} sigma parameters, "
            f"but received {len(sigma_params)}."
        )

    if np.any((sigma_params < 0.0) | (sigma_params > 1.0)):
        raise ValueError("All cross-immunity parameters must lie in [0, 1].")

    sigma = np.eye(n_variants, dtype=float)

    parameter_index = 0

    for first_index in range(n_variants):
        for second_index in range(first_index + 1, n_variants):
            sigma[first_index, second_index] = sigma_params[parameter_index]
            sigma[second_index, first_index] = sigma_params[parameter_index]
            parameter_index += 1

    return sigma


### Model simulation


def run_model_weekly_cases(
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    infectious_at_start: np.ndarray,
    fraction_antibodies: float,
    antibody_effectiveness: np.ndarray,
    symptomatic_rate: float,
    n_weeks: int,
    population: int,
    b: float,
) -> np.ndarray:
    """
    Simulates the multi-variant model and returns weekly symptomatic case
    estimates for each variant.

    The model is integrated with daily output. Cumulative infection counters
    are sampled at weekly intervals, converted to absolute infection counts,
    and multiplied by ``symptomatic_rate``.

    Parameters
    ----------
    beta : np.ndarray
        Vector of transmission rates, with one entry per variant.
    gamma : np.ndarray
        Vector of recovery rates, with one entry per variant. Must have the
        same length as ``beta``.
    sigma : np.ndarray
        Cross-immunity matrix of shape ``(n_variants, n_variants)``.
    infectious_at_start : np.ndarray
        Initial infectious fractions for all variants. Must have the same
        length as ``beta`` and contain no negative values.
    fraction_antibodies : float
        Fraction of the population with antibodies at the start of the
        simulation. Must lie in the interval [0, 1].
    antibody_effectiveness : np.ndarray
        Protection probabilities against each variant for individuals with
        antibodies. Must have the same length as ``beta``; all entries must
        lie in the interval [0, 1].
    symptomatic_rate : float
        Shared proportion of modelled infection events that are interpreted
        as symptomatic cases. Must lie in [0, 1].
    n_weeks : int
        Number of weeks to simulate.
    population : int
        Total population size used to convert population fractions into
        absolute numbers of infections.
    b : float
        Natural birth and death rate of the model.

    Returns
    -------
    weekly_symptomatic_cases : np.ndarray
        Array of shape ``(n_weeks, n_variants)`` containing modelled weekly
        symptomatic case counts for every variant.

    Raises
    ------
    ValueError
        If parameter dimensions are inconsistent or parameter values violate
        their allowed ranges.
    RuntimeError
        If numerical integration of the ODE system fails.
    """
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    infectious_at_start = np.asarray(infectious_at_start, dtype=float)
    antibody_effectiveness = np.asarray(
        antibody_effectiveness,
        dtype=float,
    )

    n_variants = len(beta)

    if len(gamma) != n_variants:
        raise ValueError("beta and gamma must have identical lengths.")

    if len(infectious_at_start) != n_variants:
        raise ValueError(
            "infectious_at_start must have the same length as beta."
        )

    if len(antibody_effectiveness) != n_variants:
        raise ValueError(
            "antibody_effectiveness must have the same length as beta."
        )

    if sigma.shape != (n_variants, n_variants):
        raise ValueError("sigma must have shape (n_variants, n_variants).")

    if not 0.0 <= fraction_antibodies <= 1.0:
        raise ValueError("fraction_antibodies must lie in [0, 1].")

    if np.any((antibody_effectiveness < 0.0) | (antibody_effectiveness > 1.0)):
        raise ValueError("antibody_effectiveness entries must lie in [0, 1].")

    if np.any(infectious_at_start < 0.0):
        raise ValueError(
            "infectious_at_start must not contain negative values."
        )

    if not 0.0 <= symptomatic_rate <= 1.0:
        raise ValueError("symptomatic_rate must lie in [0, 1].")

    x0 = create_initial_conditions_from_seroprevalence(
        fraction_with_antibodies=fraction_antibodies,
        protection_against_variant=antibody_effectiveness,
        infectious_with_variant=infectious_at_start,
    )

    # The model appends one cumulative infection counter per variant.
    cumulative_cases_start = np.zeros(n_variants, dtype=float)
    y0 = np.concatenate([x0, cumulative_cases_start])

    V, G, T, S = construct_matrices(gamma, sigma)

    days_per_week = 7
    final_time = n_weeks * days_per_week

    # Daily output is used so cumulative case counters can be sampled exactly
    # at weekly intervals.
    t_eval = np.arange(
        0,
        final_time + 1,
        1,
        dtype=float,
    )

    solution = solve_ivp(
        multi_variant_model_with_matrices,
        t_span=(0.0, float(final_time)),
        y0=y0,
        t_eval=t_eval,
        args=(
            beta,
            gamma,
            sigma,
            V,
            G,
            T,
            S,
            b,
            True,
        ),
        rtol=1e-6,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")

    cumulative_infections_daily = solution.y[-n_variants:, :].T

    weekly_cumulative_infections = cumulative_infections_daily[
        ::days_per_week,
        :,
    ]

    weekly_infections = (
        np.diff(
            weekly_cumulative_infections,
            axis=0,
        )
        * population
    )

    weekly_symptomatic_cases = symptomatic_rate * weekly_infections

    return np.maximum(weekly_symptomatic_cases, 0.0)


# Sequential objective function and optimization


def objective_active_variants(
    params: np.ndarray,
    active_variant_indices: list[int],
    beta_initial: np.ndarray,
    gamma_initial: np.ndarray,
    infectious_initial: np.ndarray,
    antibody_effectiveness_initial: np.ndarray,
    sigma_fixed: np.ndarray,
    fraction_antibodies: float,
    n_weeks: int,
    population: int,
    b: float,
    reported_cases: np.ndarray,
) -> float:
    """
    Computes the sum of squared log-scale residuals for one sequential fitting
    stage.

    The parameter vector is interpreted in the following order:

    1. Transmission rates ``beta`` of active variants.
    2. Recovery rates ``gamma`` of active variants.
    3. Base-10 logarithms of the initial infectious fractions of active
       variants.
    4. Antibody effectiveness values of active variants.
    5. One shared symptomatic rate.

    Parameters of inactive variants remain fixed at their supplied initial
    values.

    Parameters
    ----------
    params : np.ndarray
        Flat optimization parameter vector containing the parameters of active
        variants and the shared symptomatic rate.
    active_variant_indices : list of int
        Indices of the variants whose parameters are optimized in the current
        fitting stage.
    beta_initial : np.ndarray
        Full vector of baseline transmission rates. Entries of inactive
        variants remain unchanged.
    gamma_initial : np.ndarray
        Full vector of baseline recovery rates. Entries of inactive variants
        remain unchanged.
    infectious_initial : np.ndarray
        Full vector of baseline initial infectious fractions. Entries of
        inactive variants remain unchanged.
    antibody_effectiveness_initial : np.ndarray
        Full vector of baseline antibody effectiveness values. Entries of
        inactive variants remain unchanged.
    sigma_fixed : np.ndarray
        Fixed cross-immunity matrix used for all model evaluations.
    fraction_antibodies : float
        Fraction of the population with antibodies at the start of the
        simulation.
    n_weeks : int
        Number of weeks for which the model is simulated.
    population : int
        Total population size used to convert modelled fractions into case
        counts.
    b : float
        Natural birth and death rate of the model.
    reported_cases : np.ndarray
        Observed weekly strain-specific case estimates. Columns must correspond
        to the variant ordering used by the model.

    Returns
    -------
    float
        Sum of squared residuals between ``log1p``-transformed modelled and
        observed cases. Returns ``1e12`` if model evaluation fails or produces
        non-finite values.
    """
    n_active = len(active_variant_indices)

    beta_active = params[:n_active]
    gamma_active = params[n_active : 2 * n_active]
    log_infectious_active = params[2 * n_active : 3 * n_active]
    antibody_effectiveness_active = params[3 * n_active : 4 * n_active]
    symptomatic_rate = params[-1]

    beta_model = beta_initial.copy()
    gamma_model = gamma_initial.copy()
    infectious_model = infectious_initial.copy()
    antibody_effectiveness_model = antibody_effectiveness_initial.copy()

    for local_index, variant_index in enumerate(active_variant_indices):
        beta_model[variant_index] = beta_active[local_index]
        gamma_model[variant_index] = gamma_active[local_index]
        infectious_model[variant_index] = (
            10.0 ** log_infectious_active[local_index]
        )
        antibody_effectiveness_model[variant_index] = (
            antibody_effectiveness_active[local_index]
        )

    try:
        modelled_cases = run_model_weekly_cases(
            beta=beta_model,
            gamma=gamma_model,
            sigma=sigma_fixed,
            infectious_at_start=infectious_model,
            fraction_antibodies=fraction_antibodies,
            antibody_effectiveness=antibody_effectiveness_model,
            symptomatic_rate=symptomatic_rate,
            n_weeks=n_weeks,
            population=population,
            b=b,
        )
    except (FloatingPointError, RuntimeError, ValueError, OverflowError):
        return 1e12

    if not np.all(np.isfinite(modelled_cases)):
        return 1e12

    n_points = min(
        len(modelled_cases),
        len(reported_cases),
    )

    modelled_active = modelled_cases[
        :n_points,
        active_variant_indices,
    ]

    reported_active = reported_cases[
        :n_points,
        active_variant_indices,
    ]

    # Log-scale residuals prevent high-incidence weeks from dominating the fit.
    residuals = np.log1p(modelled_active) - np.log1p(reported_active)

    return float(np.nansum(residuals**2))


def fit_stage(
    active_variant_indices: list[int],
    beta_start: np.ndarray,
    gamma_start: np.ndarray,
    infectious_start: np.ndarray,
    antibody_effectiveness_start: np.ndarray,
    symptomatic_rate_start: float,
    beta_initial: np.ndarray,
    gamma_initial: np.ndarray,
    infectious_initial: np.ndarray,
    antibody_effectiveness_initial: np.ndarray,
    sigma_fixed: np.ndarray,
    fraction_antibodies: float,
    n_weeks: int,
    population: int,
    b: float,
    reported_cases: np.ndarray,
    variants: list[str],
    stage_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fits the parameters of active variants in one sequential optimization stage.

    The Nelder-Mead optimizer minimizes the log-scale sum of squared residuals
    returned by ``objective_active_variants``. Parameters and initial values of
    inactive variants remain fixed.

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
    antibody_effectiveness_start : np.ndarray
        Initial optimization values for antibody effectiveness of active
        variants.
    symptomatic_rate_start : float
        Initial optimization value for the shared symptomatic rate.
    beta_initial : np.ndarray
        Full vector of transmission rates used as fixed values for inactive
        variants.
    gamma_initial : np.ndarray
        Full vector of recovery rates used as fixed values for inactive
        variants.
    infectious_initial : np.ndarray
        Full vector of initial infectious fractions used as fixed values for
        inactive variants.
    antibody_effectiveness_initial : np.ndarray
        Full vector of antibody effectiveness values used as fixed values for
        inactive variants.
    sigma_fixed : np.ndarray
        Fixed cross-immunity matrix.
    fraction_antibodies : float
        Fraction of the population with antibodies at the start of the
        simulation.
    n_weeks : int
        Number of weeks in the fitting period.
    population : int
        Total population size.
    b : float
        Natural birth and death rate of the model.
    reported_cases : np.ndarray
        Observed weekly strain-specific case estimates.
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
    antibody_effectiveness_opt : np.ndarray
        Fitted antibody effectiveness values for active variants.
    symptomatic_rate_opt : float
        Fitted symptomatic rate shared by all active variants.
    """
    n_active = len(active_variant_indices)

    p0 = np.concatenate(
        [
            np.asarray(beta_start, dtype=float),
            np.asarray(gamma_start, dtype=float),
            np.log10(np.asarray(infectious_start, dtype=float)),
            np.asarray(antibody_effectiveness_start, dtype=float),
            [symptomatic_rate_start],
        ]
    )

    bounds = (
        [(minimum_beta, maximum_beta)] * n_active
        + [(minimum_gamma, maximum_gamma)] * n_active
        + [
            (
                minimum_log10_infectious_at_start,
                maximum_log10_infectious_at_start,
            )
        ]
        * n_active
        + [
            (
                minimum_antibody_effectiveness,
                maximum_antibody_effectiveness,
            )
        ]
        * n_active
        + [(minimum_symptomatic_rate, maximum_symptomatic_rate)]
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
            antibody_effectiveness_initial,
            sigma_fixed,
            fraction_antibodies,
            n_weeks,
            population,
            b,
            reported_cases,
        ),
        method="Nelder-Mead",
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "xatol": optimization_tolerance,
            "fatol": optimization_tolerance,
            "disp": True,
        },
    )

    beta_opt = result.x[:n_active]
    gamma_opt = result.x[n_active : 2 * n_active]

    log_infectious_opt = result.x[2 * n_active : 3 * n_active]
    infectious_opt = 10.0**log_infectious_opt

    antibody_effectiveness_opt = result.x[3 * n_active : 4 * n_active]

    symptomatic_rate_opt = float(result.x[-1])

    print()
    print(f"{stage_name} results:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Final SSE (log scale): {result.fun:.6f}")
    print(f"  Shared symptomatic rate: {symptomatic_rate_opt:.6f}")

    for local_index, variant_index in enumerate(active_variant_indices):
        print(f"  {variants[variant_index]}:")
        print(f"    beta = {beta_opt[local_index]:.6f}")
        print(f"    gamma = {gamma_opt[local_index]:.6f}")
        print(
            "    infectious_at_start = " f"{infectious_opt[local_index]:.6e}"
        )
        print(
            "    antibody_effectiveness = "
            f"{antibody_effectiveness_opt[local_index]:.6f}"
        )

    return (
        beta_opt,
        gamma_opt,
        infectious_opt,
        antibody_effectiveness_opt,
        symptomatic_rate_opt,
    )


### Output functions


def plot_model_fit(
    dates: pd.DatetimeIndex,
    reported_cases: np.ndarray,
    fitted_cases: np.ndarray,
    variants: list[str],
    colors: list[str],
    filename: str | Path,
) -> None:
    """
    Plots observed and fitted weekly influenza cases for all variants.

    Points represent observed strain-specific case estimates and lines
    represent case counts produced by the fitted multi-variant model.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Dates corresponding to the weekly observations and model outputs.
    reported_cases : np.ndarray
        Observed weekly case estimates of shape ``(n_weeks, n_variants)``.
    fitted_cases : np.ndarray
        Modelled weekly symptomatic case estimates of shape
        ``(n_weeks, n_variants)``.
    variants : list of str
        Names of the influenza variants in the column order of the case
        arrays.
    colors : list of str
        Matplotlib color specifications, with one entry per variant.
    filename : str or pathlib.Path
        Output filename without or with extension. The extension is replaced
        by the globally configured ``fileformat``.

    Returns
    -------
    None.
    """
    fig = plt.figure()

    for variant_index, variant in enumerate(variants):
        plt.plot(
            dates,
            reported_cases[:, variant_index],
            color=colors[variant_index],
            marker="o",
            markersize=2,
            linestyle="none",
            label=f"{variant} (Data)",
        )

        plt.plot(
            dates,
            fitted_cases[:, variant_index],
            color=colors[variant_index],
            linewidth=2,
            linestyle="-",
            label=f"{variant} (Model)",
        )

    plt.xlabel("Date", fontsize=fontsize_text)

    plt.ylabel(
        "Symptomatic cases\nper week",
        fontsize=fontsize_text,
    )

    plt.xlim(dates[0], dates[-1])
    plt.ylim(bottom=0)

    plt.xticks(fontsize=fontsize_ticks, rotation=90)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid(alpha=alpha_grid)

    fig.legend(
        fontsize=fontsize_ticks,
        ncol=2,
        bbox_to_anchor=(0.50, 0.05),
        loc="upper center",
    )

    plt.tight_layout()

    filename = Path(filename).with_suffix(f".{fileformat}")
    filename.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(filename, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close()


def save_fitted_parameters(
    filename: str | Path,
    variants: list[str],
    beta: np.ndarray,
    gamma: np.ndarray,
    infectious_at_start: np.ndarray,
    antibody_effectiveness: np.ndarray,
    sigma: np.ndarray,
    symptomatic_rate: float,
) -> None:
    """
    Saves fitted and fixed influenza model parameters to a CSV file.

    The output contains one row per variant. Shared parameters, including the
    symptomatic rate and cross-immunity values, are repeated in every row.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path of the CSV file to create. Parent directories are created if
        necessary.
    variants : list of str
        Names of variants in the ordering used by all parameter arrays.
    beta : np.ndarray
        Fitted transmission rates, with one entry per variant.
    gamma : np.ndarray
        Fitted recovery rates, with one entry per variant.
    infectious_at_start : np.ndarray
        Fitted initial infectious fractions, with one entry per variant.
    antibody_effectiveness : np.ndarray
        Fitted antibody effectiveness values, with one entry per variant.
    sigma : np.ndarray
        Cross-immunity matrix of shape ``(n_variants, n_variants)``.
    symptomatic_rate : float
        Shared fitted proportion of infections interpreted as symptomatic
        cases.

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
        "mean_infectious_duration_days",
        "infectious_at_start",
        "initial_infectious_persons",
        "antibody_effectiveness",
        "symptomatic_rate",
    ]

    for first_index in range(n_variants):
        for second_index in range(first_index + 1, n_variants):
            fieldnames.append(f"sigma_{first_index + 1}{second_index + 1}")

    with filename.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for variant_index, variant in enumerate(variants):
            row = {
                "variant": variant,
                "beta": beta[variant_index],
                "gamma": gamma[variant_index],
                "mean_infectious_duration_days": (1.0 / gamma[variant_index]),
                "infectious_at_start": infectious_at_start[variant_index],
                "initial_infectious_persons": (
                    infectious_at_start[variant_index] * population
                ),
                "antibody_effectiveness": (
                    antibody_effectiveness[variant_index]
                ),
                "symptomatic_rate": symptomatic_rate,
            }

            for first_index in range(n_variants):
                for second_index in range(first_index + 1, n_variants):
                    row[f"sigma_{first_index + 1}{second_index + 1}"] = sigma[
                        first_index, second_index
                    ]

            writer.writerow(row)


def save_weekly_cases(
    filename: str | Path,
    weekly_data: pd.DataFrame,
    variants: list[str],
    reported_cases: np.ndarray,
    fitted_cases: np.ndarray,
) -> None:
    """
    Saves observed and modelled weekly strain-specific influenza cases to a
    CSV file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path of the CSV file to create. Parent directories are created if
        necessary.
    weekly_data : pd.DataFrame
        Filtered input data containing at least the columns ``year``, ``week``,
        ``calendar_week``, ``date``, and ``estimated_cases``.
    variants : list of str
        Names of variants in the column order of the observed and modelled
        case arrays.
    reported_cases : np.ndarray
        Observed weekly strain-specific case estimates of shape
        ``(n_weeks, n_variants)``.
    fitted_cases : np.ndarray
        Modelled weekly strain-specific case estimates of shape
        ``(n_weeks, n_variants)``.

    Returns
    -------
    None.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    output_data = weekly_data[
        [
            "year",
            "week",
            "calendar_week",
            "date",
            "estimated_cases",
        ]
    ].copy()

    for variant_index, variant in enumerate(variants):
        output_data[f"{variant}_observed"] = reported_cases[
            :,
            variant_index,
        ]

        output_data[f"{variant}_model"] = fitted_cases[
            :,
            variant_index,
        ]

    output_data.to_csv(
        filename,
        index=False,
        float_format="%.8f",
    )


### Read precomputed parameters


def read_fitted_parameters(
    filename: str | Path,
    variants: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Reads previously fitted influenza model parameters from a CSV file.

    The file must have been created by ``save_fitted_parameters`` and must
    contain one row for every requested variant. The stored symptomatic rate
    must be identical for all rows.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the CSV file containing fitted parameters.
    variants : list of str
        Variant names and desired ordering of the returned parameter arrays.

    Returns
    -------
    beta : np.ndarray
        Fitted transmission rates in the order specified by ``variants``.
    gamma : np.ndarray
        Fitted recovery rates in the order specified by ``variants``.
    infectious_at_start : np.ndarray
        Fitted initial infectious fractions in the order specified by
        ``variants``.
    antibody_effectiveness : np.ndarray
        Fitted antibody effectiveness values in the order specified by
        ``variants``.
    sigma : np.ndarray
        Symmetric cross-immunity matrix reconstructed from the stored
        upper-triangular values.
    symptomatic_rate : float
        Shared fitted symptomatic rate.

    Raises
    ------
    FileNotFoundError
        If ``filename`` does not exist.
    ValueError
        If required columns or requested variants are missing, or if the
        symptomatic rate differs between variants in the parameter file.
    """
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            "The parameter file does not exist. Set recompute=True, to create "
            f"it first: {filename}"
        )

    parameters = pd.read_csv(filename)

    n_variants = len(variants)

    required_columns = {
        "variant",
        "beta",
        "gamma",
        "infectious_at_start",
        "antibody_effectiveness",
        "symptomatic_rate",
    }

    for first_index in range(n_variants):
        for second_index in range(first_index + 1, n_variants):
            required_columns.add(f"sigma_{first_index + 1}{second_index + 1}")

    missing_columns = required_columns - set(parameters.columns)

    if missing_columns:
        raise ValueError(
            "The parameter file does not contain all necessary columns: "
            f"{sorted(missing_columns)}"
        )

    parameters = parameters.set_index("variant")

    missing_variants = set(variants) - set(parameters.index)

    if missing_variants:
        raise ValueError(
            "The parameter file does not contain all strains: "
            f"{sorted(missing_variants)}"
        )

    parameters = parameters.loc[variants]

    beta = parameters["beta"].to_numpy(dtype=float)
    gamma = parameters["gamma"].to_numpy(dtype=float)

    infectious_at_start = parameters["infectious_at_start"].to_numpy(
        dtype=float
    )

    antibody_effectiveness = parameters["antibody_effectiveness"].to_numpy(
        dtype=float
    )

    symptomatic_rates = parameters["symptomatic_rate"].to_numpy(dtype=float)

    if not np.allclose(
        symptomatic_rates,
        symptomatic_rates[0],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "The stored symptomatic_rate is not identical for all strains."
        )

    symptomatic_rate = float(symptomatic_rates[0])

    sigma = np.eye(n_variants, dtype=float)

    first_row = parameters.iloc[0]

    for first_index in range(n_variants):
        for second_index in range(first_index + 1, n_variants):
            column = f"sigma_{first_index + 1}{second_index + 1}"

            sigma_value = float(first_row[column])

            sigma[first_index, second_index] = sigma_value
            sigma[second_index, first_index] = sigma_value

    return (
        beta,
        gamma,
        infectious_at_start,
        antibody_effectiveness,
        sigma,
        symptomatic_rate,
    )


# Plot and output configuration
input_folder = Path("../../Influenza Case Estimation/data")
data_folder = Path("../data")
plot_folder = Path("../plots")

fileformat = "pdf"
dpi = 1200

fontsize_text = 16
fontsize_ticks = 14
alpha_grid = 0.7

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:purple",
]


# Model and optimization configuration
flu_season = "2017-2018"
fit_period = "november_to_may"
recompute = False

variants = [
    "A(H1N1)",
    "A(H3)",
    "B/Victoria",
    "B/Yamagata",
]

input_file = input_folder / f"weekly_cases_per_strain_{flu_season}.csv"

fitted_parameters_filename = (
    data_folder / f"influenza_multi_variant_fitted_parameters_"
    f"{flu_season}_{fit_period}.csv"
)

fitted_weekly_cases_filename = (
    data_folder / f"influenza_multi_variant_fitted_weekly_cases_"
    f"{flu_season}_{fit_period}.csv"
)

final_plot_filename = (
    plot_folder / f"influenza_multi_variant_parameter_identification_"
    f"{flu_season}_{fit_period}"
)

population = 325_000_000
b = 2.4e-5
fraction_antibodies = 1.0

# Parameter bounds.
minimum_beta = 0.05
maximum_beta = 5.00
minimum_gamma = 1.0 / 7.0
maximum_gamma = 1.0 / 1.0
minimum_log10_infectious_at_start = -12.0
maximum_log10_infectious_at_start = -3.0
minimum_antibody_effectiveness = 1e-6
maximum_antibody_effectiveness = 1.0
minimum_symptomatic_rate = 0.01
maximum_symptomatic_rate = 1.0

# Nelder-Mead settings.
maxiter = 50_000
optimization_tolerance = 1e-7

# Initial parameter values.
beta_initial = np.array(
    [
        0.40,  # A(H1N1)
        0.40,  # A(H3)
        0.40,  # B/Victoria
        0.40,  # B/Yamagata
    ],
    dtype=float,
)
gamma_initial = np.array(
    [
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
    ],
    dtype=float,
)
infectious_initial = np.array(
    [
        1e-7,  # A(H1N1)
        1e-8,  # A(H3)
        1e-9,  # B/Victoria
        1e-10,  # B/Yamagata
    ],
    dtype=float,
)
antibody_effectiveness_initial = np.array(
    [
        0.62,  # A(H1N1)
        0.22,  # A(H3)
        0.50,  # B/Victoria
        0.50,  # B/Yamagata
    ],
    dtype=float,
)
symptomatic_rate_initial = 0.50

# Fixed symmetric cross-immunity values.
#
# Parameter order:
# sigma_12 = A(H1N1) <-> A(H3)
# sigma_13 = A(H1N1) <-> B/Victoria
# sigma_14 = A(H1N1) <-> B/Yamagata
# sigma_23 = A(H3) <-> B/Victoria
# sigma_24 = A(H3) <-> B/Yamagata
# sigma_34 = B/Victoria <-> B/Yamagata
sigma_upper_fixed = np.array(
    [
        0.20,
        0.00,
        0.00,
        0.00,
        0.00,
        0.20,
    ],
    dtype=float,
)
n_variants = len(variants)
sigma_fixed = params_to_sigma(
    sigma_upper_fixed,
    n_variants,
)


# Read influenza data
weekly_data = read_and_filter_weekly_cases(
    filename=input_file,
    variants=variants,
    flu_season=flu_season,
)

dates = pd.DatetimeIndex(weekly_data["date"])
reported_cases = weekly_data[variants].to_numpy(dtype=float)
n_weeks = len(weekly_data)

print("=" * 80)
print(f"Sequential multi-variant influenza fit: {flu_season}")
print("=" * 80)
print(f"Input file: {input_file}")
print(f"Fitting period: {dates[0].date()} through {dates[-1].date()}")
print(f"Number of fitted weeks: {n_weeks}")
print(f"Population: {population:,}")
print(f"Fraction with antibodies: {fraction_antibodies:.6f}")
print(f"Total estimated cases: {reported_cases.sum():,.0f}")
print(f"Recompute parameters: {recompute}")
print()

if recompute:
    print("Fixed cross-immunity matrix:")
    print(sigma_fixed)
    print()

    # Stage 1: Fit A(H1N1)
    (
        beta_stage_1,
        gamma_stage_1,
        infectious_stage_1,
        antibody_effectiveness_stage_1,
        symptomatic_rate_stage_1,
    ) = fit_stage(
        active_variant_indices=[0],
        beta_start=beta_initial[[0]],
        gamma_start=gamma_initial[[0]],
        infectious_start=infectious_initial[[0]],
        antibody_effectiveness_start=antibody_effectiveness_initial[[0]],
        symptomatic_rate_start=symptomatic_rate_initial,
        beta_initial=beta_initial,
        gamma_initial=gamma_initial,
        infectious_initial=np.array(
            [
                infectious_initial[0],
                0.0,
                0.0,
                0.0,
            ]
        ),
        antibody_effectiveness_initial=antibody_effectiveness_initial,
        sigma_fixed=sigma_fixed,
        fraction_antibodies=fraction_antibodies,
        n_weeks=n_weeks,
        population=population,
        b=b,
        reported_cases=reported_cases,
        variants=variants,
        stage_name="Stage 1: A(H1N1)",
    )

    # Stage 2: Re-fit A(H1N1) and fit A(H3)
    (
        beta_stage_2,
        gamma_stage_2,
        infectious_stage_2,
        antibody_effectiveness_stage_2,
        symptomatic_rate_stage_2,
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
        antibody_effectiveness_start=np.array(
            [
                antibody_effectiveness_stage_1[0],
                antibody_effectiveness_initial[1],
            ]
        ),
        symptomatic_rate_start=symptomatic_rate_stage_1,
        beta_initial=beta_initial,
        gamma_initial=gamma_initial,
        infectious_initial=np.array(
            [
                infectious_stage_1[0],
                infectious_initial[1],
                0.0,
                0.0,
            ]
        ),
        antibody_effectiveness_initial=antibody_effectiveness_initial,
        sigma_fixed=sigma_fixed,
        fraction_antibodies=fraction_antibodies,
        n_weeks=n_weeks,
        population=population,
        b=b,
        reported_cases=reported_cases,
        variants=variants,
        stage_name="Stage 2: A(H1N1) + A(H3)",
    )

    # Stage 3: Re-fit A(H1N1), A(H3), and fit B/Victoria
    (
        beta_stage_3,
        gamma_stage_3,
        infectious_stage_3,
        antibody_effectiveness_stage_3,
        symptomatic_rate_stage_3,
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
        antibody_effectiveness_start=np.array(
            [
                antibody_effectiveness_stage_2[0],
                antibody_effectiveness_stage_2[1],
                antibody_effectiveness_initial[2],
            ]
        ),
        symptomatic_rate_start=symptomatic_rate_stage_2,
        beta_initial=beta_initial,
        gamma_initial=gamma_initial,
        infectious_initial=np.array(
            [
                infectious_stage_2[0],
                infectious_stage_2[1],
                infectious_initial[2],
                0.0,
            ]
        ),
        antibody_effectiveness_initial=antibody_effectiveness_initial,
        sigma_fixed=sigma_fixed,
        fraction_antibodies=fraction_antibodies,
        n_weeks=n_weeks,
        population=population,
        b=b,
        reported_cases=reported_cases,
        variants=variants,
        stage_name="Stage 3: A(H1N1) + A(H3) + B/Victoria",
    )

    # Stage 4: Re-fit all variants including B/Yamagata
    (
        beta_final,
        gamma_final,
        infectious_final,
        antibody_effectiveness_final,
        symptomatic_rate_final,
    ) = fit_stage(
        active_variant_indices=[0, 1, 2, 3],
        beta_start=np.array(
            [
                beta_stage_3[0],
                beta_stage_3[1],
                beta_stage_3[2],
                beta_initial[3],
            ]
        ),
        gamma_start=np.array(
            [
                gamma_stage_3[0],
                gamma_stage_3[1],
                gamma_stage_3[2],
                gamma_initial[3],
            ]
        ),
        infectious_start=np.array(
            [
                infectious_stage_3[0],
                infectious_stage_3[1],
                infectious_stage_3[2],
                infectious_initial[3],
            ]
        ),
        antibody_effectiveness_start=np.array(
            [
                antibody_effectiveness_stage_3[0],
                antibody_effectiveness_stage_3[1],
                antibody_effectiveness_stage_3[2],
                antibody_effectiveness_initial[3],
            ]
        ),
        symptomatic_rate_start=symptomatic_rate_stage_3,
        beta_initial=beta_initial,
        gamma_initial=gamma_initial,
        infectious_initial=np.array(
            [
                infectious_stage_3[0],
                infectious_stage_3[1],
                infectious_stage_3[2],
                infectious_initial[3],
            ]
        ),
        antibody_effectiveness_initial=antibody_effectiveness_initial,
        sigma_fixed=sigma_fixed,
        fraction_antibodies=fraction_antibodies,
        n_weeks=n_weeks,
        population=population,
        b=b,
        reported_cases=reported_cases,
        variants=variants,
        stage_name=("Stage 4: A(H1N1) + A(H3) + B/Victoria + B/Yamagata"),
    )

else:
    print(
        "Loading saved fitted parameters without re-optimization: "
        f"{fitted_parameters_filename}"
    )

    (
        beta_final,
        gamma_final,
        infectious_final,
        antibody_effectiveness_final,
        sigma_final,
        symptomatic_rate_final,
    ) = read_fitted_parameters(
        filename=fitted_parameters_filename,
        variants=variants,
    )

    print()
    print("Cross-immunity matrix loaded from parameter file:")
    print(sigma_final)


# Print final estimates
print()
print("=" * 80)
print("Final fitted influenza parameters")
print("=" * 80)

for variant_index, variant in enumerate(variants):
    print()
    print(f"{variant}:")
    print(f"  beta = {beta_final[variant_index]:.8f}")
    print(f"  gamma = {gamma_final[variant_index]:.8f}")
    print(
        "  mean infectious duration = "
        f"{1.0 / gamma_final[variant_index]:.4f} days"
    )
    print("  infectious_at_start = " f"{infectious_final[variant_index]:.8e}")
    print(
        "  initial infectious persons = "
        f"{infectious_final[variant_index] * population:,.2f}"
    )
    print(
        "  antibody_effectiveness = "
        f"{antibody_effectiveness_final[variant_index]:.8f}"
    )

print()
print("Shared symptomatic rate = " f"{symptomatic_rate_final:.8f}")

print()
print("Fixed sigma matrix:")
print(sigma_fixed)


# Final model run
fitted_cases_final = run_model_weekly_cases(
    beta=beta_final,
    gamma=gamma_final,
    sigma=sigma_fixed,
    infectious_at_start=infectious_final,
    fraction_antibodies=fraction_antibodies,
    antibody_effectiveness=antibody_effectiveness_final,
    symptomatic_rate=symptomatic_rate_final,
    n_weeks=n_weeks,
    population=population,
    b=b,
)

if len(fitted_cases_final) != len(reported_cases):
    raise RuntimeError(
        "The number of modelled and observed weeks does not match: "
        f"{len(fitted_cases_final)} modelled versus "
        f"{len(reported_cases)} observed."
    )


# Plot and save outputs
plot_model_fit(
    dates=dates,
    reported_cases=reported_cases,
    fitted_cases=fitted_cases_final,
    variants=variants,
    colors=colors,
    filename=final_plot_filename,
)

save_fitted_parameters(
    filename=fitted_parameters_filename,
    variants=variants,
    beta=beta_final,
    gamma=gamma_final,
    infectious_at_start=infectious_final,
    antibody_effectiveness=antibody_effectiveness_final,
    sigma=sigma_fixed,
    symptomatic_rate=symptomatic_rate_final,
)

save_weekly_cases(
    filename=fitted_weekly_cases_filename,
    weekly_data=weekly_data,
    variants=variants,
    reported_cases=reported_cases,
    fitted_cases=fitted_cases_final,
)

print()
print("=" * 80)
print("Output")
print("=" * 80)
print(f"Fitted parameters saved to: {fitted_parameters_filename}")
print(f"Weekly fitted cases saved to: {fitted_weekly_cases_filename}")
print(
    "Final model plot saved to: "
    f"{Path(final_plot_filename).with_suffix(f'.{fileformat}')}"
)
