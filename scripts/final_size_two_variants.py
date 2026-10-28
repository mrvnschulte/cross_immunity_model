"""
Numerically investigates the final epidemic size of a two-variant
cross-immunity model.

The model is simulated over a sufficiently long time interval for multiple
values of:

1. The cross-immunity factor sigma_12.
2. The transmission rate beta_1.
3. The transmission rate beta_2.

Recovery rates are fixed at gamma_1 = gamma_2 = 0.1 and the demographic rate
is fixed at b = 0.

For every simulation, cumulative infection events are integrated separately
for both variants using ``compute_dn_variant=True`` in
``multi_variant_model_with_matrices``. The cumulative final size is defined as
the sum of the final sizes of both variants.

Important
---------
The cumulative final size counts infection events. Therefore, an individual
infected with both variants contributes once to each variant-specific final
size and twice to the cumulative final size.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from cross_immunity_model import (
    construct_matrices,
    critical_r01,
    critical_r02,
    critical_sigma,
    infectious_per_variant,
    multi_variant_model_with_matrices,
)


def simulate_final_size(
    beta_1: float,
    beta_2: float,
    sigma_12: float,
    gamma_1: float = 0.1,
    gamma_2: float = 0.1,
    b: float = 0.0,
    initial_infectious_1: float = 1e-5,
    initial_infectious_2: float = 1e-5,
    tf: float = 1000.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Simulate the two-variant model and compute final epidemic sizes.

    The state vector is extended by two cumulative variables. These variables
    contain the total number of infection events caused by variant 1 and
    variant 2, respectively.

    Parameters
    ----------
    beta_1 : float
        Transmission rate of variant 1.
    beta_2 : float
        Transmission rate of variant 2.
    sigma_12 : float
        Symmetric cross-immunity factor between variants 1 and 2.
    gamma_1 : float, optional
        Recovery rate of variant 1. The default is 0.1.
    gamma_2 : float, optional
        Recovery rate of variant 2. The default is 0.1.
    b : float, optional
        Natural birth and death rate. The default is 0.0.
    initial_infectious_1 : float, optional
        Initial infectious fraction with variant 1. The default is 1e-5.
    initial_infectious_2 : float, optional
        Initial infectious fraction with variant 2. The default is 1e-5.
    tf : float, optional
        Final simulation time. It should be sufficiently large such that the
        epidemic has ended. The default is 1000.0.
    rtol : float, optional
        Relative tolerance of the ODE solver. The default is 1e-8.
    atol : float, optional
        Absolute tolerance of the ODE solver. The default is 1e-10.

    Returns
    -------
    final_size_total : float
        Total cumulative number of infection events from both variants.
    final_size_per_variant : np.ndarray
        Cumulative infection events for variants 1 and 2.
    final_infectious_per_variant : np.ndarray
        Infectious fractions of variants 1 and 2 at time ``tf``.

    Raises
    ------
    RuntimeError
        If the numerical ODE integration fails.
    """
    beta = np.array([beta_1, beta_2], dtype=float)
    gamma = np.array([gamma_1, gamma_2], dtype=float)
    sigma = np.array(
        [
            [1.0, sigma_12],
            [sigma_12, 1.0],
        ]
    )

    V, G, T, S = construct_matrices(gamma, sigma)

    # State ordering for two variants:
    #
    # [S, R_1, I_1, R_2, I_2, R_12, I_1,12, I_2,12, I_12]
    #
    # Two additional entries are appended for cumulative infections with
    # variants 1 and 2.
    x0 = np.zeros(3**2 + 2)

    x0[2] = initial_infectious_1
    x0[4] = initial_infectious_2
    x0[0] = 1.0 - initial_infectious_1 - initial_infectious_2

    # The cumulative counters include the initially infectious individuals.
    x0[-2:] = infectious_per_variant(x0[: 3**2], V)

    solution = solve_ivp(
        multi_variant_model_with_matrices,
        (0.0, tf),
        x0,
        t_eval=[tf],
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
        rtol=rtol,
        atol=atol,
    )

    if not solution.success:
        raise RuntimeError(
            "Simulation failed for "
            f"beta_1={beta_1}, beta_2={beta_2}, sigma_12={sigma_12}: "
            f"{solution.message}"
        )

    final_state = solution.y[: 3**2, -1]
    final_size_per_variant = solution.y[-2:, -1]
    final_size_total = float(np.sum(final_size_per_variant))

    final_infectious_per_variant = infectious_per_variant(final_state, V)

    return (
        final_size_total,
        final_size_per_variant,
        final_infectious_per_variant,
    )


def run_parameter_sweep(
    parameter_name: str,
    parameter_values: np.ndarray,
    beta_1: float,
    beta_2: float,
    sigma_12: float,
    gamma_1: float,
    gamma_2: float,
    b: float,
    initial_infectious_1: float,
    initial_infectious_2: float,
    tf: float,
    r0_2_target: float | None = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Simulate final sizes for a range of one parameter.

    If ``parameter_name`` is ``"gamma_2"`` and ``r0_2_target`` is given,
    beta_2 is adjusted for every parameter value such that

        R_0,2 = beta_2 / (gamma_2 + b)

    remains constant.

    Parameters
    ----------
    parameter_name : str
        Name of the varied parameter. Must be one of ``"sigma_12"``,
        ``"beta_2"``, or ``"gamma_2"``.
    parameter_values : np.ndarray
        Values of the parameter to be varied.
    beta_1 : float
        Fixed transmission rate of variant 1.
    beta_2 : float
        Baseline transmission rate of variant 2.
    sigma_12 : float
        Baseline cross-immunity factor.
    gamma_1 : float
        Fixed recovery rate of variant 1.
    gamma_2 : float
        Baseline recovery rate of variant 2.
    b : float
        Birth and death rate.
    initial_infectious_1 : float
        Initial infectious fraction with variant 1.
    initial_infectious_2 : float
        Initial infectious fraction with variant 2.
    tf : float
        Final simulation time.
    r0_2_target : float or None, optional
        Target basic reproduction number of variant 2. This is only used
        when ``parameter_name="gamma_2"``. The default is None.
    verbose : bool, optional
        If True, print progress messages. The default is True.

    Returns
    -------
    results : dict[str, np.ndarray]
        Dictionary containing parameter values, actual parameter values,
        final sizes, and residual infectious fractions.
    """
    allowed_parameters = {"sigma_12", "beta_2", "gamma_2"}

    if parameter_name not in allowed_parameters:
        raise ValueError(
            f"parameter_name must be one of {sorted(allowed_parameters)}."
        )

    if parameter_name == "gamma_2" and r0_2_target is None:
        raise ValueError(
            "r0_2_target must be specified when gamma_2 is varied."
        )

    n_values = len(parameter_values)

    final_size_total = np.zeros(n_values)
    final_size_variant_1 = np.zeros(n_values)
    final_size_variant_2 = np.zeros(n_values)
    final_infectious_1 = np.zeros(n_values)
    final_infectious_2 = np.zeros(n_values)

    actual_beta_1 = np.zeros(n_values)
    actual_beta_2 = np.zeros(n_values)
    actual_gamma_1 = np.zeros(n_values)
    actual_gamma_2 = np.zeros(n_values)
    actual_sigma_12 = np.zeros(n_values)
    actual_r0_2 = np.zeros(n_values)

    for index, parameter_value in enumerate(parameter_values):
        current_beta_1 = beta_1
        current_beta_2 = beta_2
        current_gamma_1 = gamma_1
        current_gamma_2 = gamma_2
        current_sigma_12 = sigma_12

        if parameter_name == "beta_2":
            current_beta_2 = parameter_value

        elif parameter_name == "gamma_2":
            current_gamma_2 = parameter_value
            current_beta_2 = r0_2_target * (current_gamma_2 + b)

        elif parameter_name == "sigma_12":
            current_sigma_12 = parameter_value

        (
            final_size_total[index],
            final_size_per_variant,
            final_infectious_per_variant,
        ) = simulate_final_size(
            beta_1=current_beta_1,
            beta_2=current_beta_2,
            sigma_12=current_sigma_12,
            gamma_1=current_gamma_1,
            gamma_2=current_gamma_2,
            b=b,
            initial_infectious_1=initial_infectious_1,
            initial_infectious_2=initial_infectious_2,
            tf=tf,
        )

        final_size_variant_1[index] = final_size_per_variant[0]
        final_size_variant_2[index] = final_size_per_variant[1]
        final_infectious_1[index] = final_infectious_per_variant[0]
        final_infectious_2[index] = final_infectious_per_variant[1]

        actual_beta_1[index] = current_beta_1
        actual_beta_2[index] = current_beta_2
        actual_gamma_1[index] = current_gamma_1
        actual_gamma_2[index] = current_gamma_2
        actual_sigma_12[index] = current_sigma_12
        actual_r0_2[index] = current_beta_2 / (current_gamma_2 + b)

        if verbose:
            print(
                f"{parameter_name}: {index + 1}/{n_values} "
                f"({parameter_value:.4f})"
            )

    return {
        "parameter_values": parameter_values,
        "beta_1": actual_beta_1,
        "beta_2": actual_beta_2,
        "gamma_1": actual_gamma_1,
        "gamma_2": actual_gamma_2,
        "sigma_12": actual_sigma_12,
        "r0_2": actual_r0_2,
        "final_size_total": final_size_total,
        "final_size_variant_1": final_size_variant_1,
        "final_size_variant_2": final_size_variant_2,
        "final_infectious_1": final_infectious_1,
        "final_infectious_2": final_infectious_2,
    }


def save_final_size_data(
    csv_filename: str,
    sweep_name: str,
    results: dict[str, np.ndarray],
    b: float,
) -> None:
    """
    Save final-size results from one parameter sweep to a CSV file.

    The CSV file stores the actual parameter values used for every simulation.
    This is important for sweeps where beta_2 is adjusted to maintain a
    constant R_0,2.

    Parameters
    ----------
    csv_filename : str
        Filename of the csv file to be stored.
    sweep_name : str
        Name of the parameter that is varied.
    results : dict[str, np.ndarray]
        Results returned by ``run_parameter_sweep``.
    b : float
        Birth and death rate.
    """
    csv_path = Path(csv_filename)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "sweep_parameter",
                "parameter_value",
                "beta_1",
                "beta_2",
                "gamma_1",
                "gamma_2",
                "sigma_12",
                "b",
                "r0_2",
                "final_size_total",
                "final_size_variant_1",
                "final_size_variant_2",
                "final_infectious_1",
                "final_infectious_2",
            ],
        )
        writer.writeheader()

        for index, parameter_value in enumerate(results["parameter_values"]):
            writer.writerow(
                {
                    "sweep_parameter": sweep_name,
                    "parameter_value": parameter_value,
                    "beta_1": results["beta_1"][index],
                    "beta_2": results["beta_2"][index],
                    "gamma_1": results["gamma_1"][index],
                    "gamma_2": results["gamma_2"][index],
                    "sigma_12": results["sigma_12"][index],
                    "b": b,
                    "r0_2": results["r0_2"][index],
                    "final_size_total": results["final_size_total"][index],
                    "final_size_variant_1": results["final_size_variant_1"][
                        index
                    ],
                    "final_size_variant_2": results["final_size_variant_2"][
                        index
                    ],
                    "final_infectious_1": results["final_infectious_1"][index],
                    "final_infectious_2": results["final_infectious_2"][index],
                }
            )


def plot_final_sizes(
    parameter_values: np.ndarray,
    results: dict[str, np.ndarray],
    xlabel: str,
    critical_values: list[tuple[float, str]] | None = None,
    filename: str | None = None,
) -> None:
    """
    Plot cumulative and variant-specific final epidemic sizes.

    Optionally adds vertical dashed lines for critical parameter values.

    Parameters
    ----------
    parameter_values : np.ndarray
        Values of the parameter shown on the x-axis.
    results : dict[str, np.ndarray]
        Results returned by ``run_parameter_sweep``.
    xlabel : str
        Label of the x-axis.
    critical_values : list of tuple[float, str] or None, optional
        Critical values to mark by vertical dashed lines. Each tuple contains
        the critical value and its legend label. Values outside the plotted
        parameter range are ignored. The default is None.
    filename : str or None, optional
        Path at which to save the figure. If None, the figure is not saved.
        The default is None.
    """
    fig, ax = plt.subplots()

    ax.plot(
        parameter_values,
        results["final_size_variant_1"],
        label="Strain 1",
    )
    ax.plot(
        parameter_values,
        results["final_size_variant_2"],
        label="Strain 2",
    )
    ax.plot(
        parameter_values,
        results["final_size_total"],
        label="Cumulative",
    )

    if critical_values is not None:
        for critical_value, label in critical_values:
            is_finite = np.isfinite(critical_value)
            is_in_range = (
                parameter_values[0] <= critical_value <= parameter_values[-1]
            )

            if is_finite and is_in_range:
                ax.axvline(
                    critical_value,
                    color="grey",
                    linestyle="dashed",
                    linewidth=1.5,
                    label=label,
                )

    ax.set_xlabel(xlabel, fontsize=fontsize_text)
    ax.set_ylabel("Final number\nof infections", fontsize=fontsize_text)
    ax.set_xlim(parameter_values[0], parameter_values[-1])
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=fontsize_ticks)
    ax.grid(alpha=alpha_grid)

    fig.legend(
        fontsize=fontsize_ticks,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.02),
        ncol=2,
    )

    plt.tight_layout()

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
data_folder = "../data"
fileformat = "pdf"

verbose = True


# Fixed model parameters
gamma_1 = 0.1
gamma_2 = 0.1
b = 0.0

beta_1 = 0.3
beta_2 = 0.35
sigma_12 = 0.5

initial_infectious_1 = 1e-4
initial_infectious_2 = 1e-4

tf = 1000.0

# Critical parameter values
r_01 = beta_1 / gamma_1
r_02 = beta_2 / gamma_2

# Basic epidemic thresholds: R_0,i = 1
critical_beta_1_epidemic = gamma_1
critical_beta_2_epidemic = gamma_2

# Variant 1 can invade the variant-2 equilibrium above this threshold
critical_beta_1_invasion = gamma_1 * critical_r02(r_02, sigma_12)
critical_beta_1_extinction = gamma_1 * critical_r01(r_02, sigma_12)

# Variant 2 can invade the variant-1 equilibrium above this threshold
critical_beta_2_invasion = gamma_2 * critical_r02(r_01, sigma_12)
critical_beta_2_extinction = gamma_2 * critical_r01(r_01, sigma_12)


# Cross-immunity threshold
if r_01 >= r_02:
    critical_sigma_12 = critical_sigma(r_01, r_02)
    sigma_critical_label = r"$\sigma_{2 \to 1}$"
else:
    critical_sigma_12 = critical_sigma(r_02, r_01)
    sigma_critical_label = r"$\sigma_{1 \to 2}$"


### STEP 1: Vary sigma_12

sigma_12_values = np.linspace(0.0, 1.0, 101)

results_sigma_12 = run_parameter_sweep(
    parameter_name="sigma_12",
    parameter_values=sigma_12_values,
    beta_1=beta_1,
    beta_2=beta_2,
    sigma_12=sigma_12,
    gamma_1=gamma_1,
    gamma_2=gamma_2,
    b=b,
    initial_infectious_1=initial_infectious_1,
    initial_infectious_2=initial_infectious_2,
    tf=tf,
    verbose=verbose,
)

save_final_size_data(
    f"{data_folder}/final_size_sigma_12.csv",
    "sigma_12",
    results_sigma_12,
    b,
)

plot_final_sizes(
    sigma_12_values,
    results_sigma_12,
    xlabel=r"Cross-immunity $\sigma_{12}$",
    critical_values=[(critical_sigma_12, sigma_critical_label)],
    filename=f"{plot_folder}/final_size_sigma_12.{fileformat}",
)


### STEP 2: Vary beta_2

sigma_12_values = [0.2, 0.5, 0.8]
beta_2_values = np.linspace(0.1, 0.5, 101)

for sigma_12 in sigma_12_values:
    r_01 = beta_1 / (gamma_1 + b)

    critical_beta_2_invasion = (gamma_2 + b) * critical_r02(r_01, sigma_12)
    critical_beta_2_extinction = (gamma_2 + b) * critical_r01(r_01, sigma_12)

    results_beta_2 = run_parameter_sweep(
        parameter_name="beta_2",
        parameter_values=beta_2_values,
        beta_1=beta_1,
        beta_2=beta_2,
        sigma_12=sigma_12,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        b=b,
        initial_infectious_1=initial_infectious_1,
        initial_infectious_2=initial_infectious_2,
        tf=tf,
        verbose=verbose,
    )

    save_final_size_data(
        f"{data_folder}/final_size_beta_2_sigma_{sigma_12}.csv",
        "beta_2",
        results_beta_2,
        b,
    )

    plot_final_sizes(
        beta_2_values,
        results_beta_2,
        xlabel=r"Transmission rate $\beta_2$",
        critical_values=[
            (critical_beta_2_invasion, r"$R_{0,2}^{2 \to 1}$"),
            (critical_beta_2_extinction, r"$R_{0,2}^{1 \to 2}$"),
        ],
        filename=(
            f"{plot_folder}/final_size_beta_2_sigma_{sigma_12}."
            f"{fileformat}"
        ),
    )

### STEP 3: Vary gamma_2 while keeping R_0,2 = 3.5 constant

r0_2_target = 3.5
gamma_2_values = np.linspace(0.025, 0.25, 101)
sigma_12_values = [0.2, 0.5, 0.8]

for sigma_12 in sigma_12_values:
    results_gamma_2 = run_parameter_sweep(
        parameter_name="gamma_2",
        parameter_values=gamma_2_values,
        beta_1=beta_1,
        beta_2=beta_2,
        sigma_12=sigma_12,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        b=b,
        initial_infectious_1=initial_infectious_1,
        initial_infectious_2=initial_infectious_2,
        tf=tf,
        r0_2_target=r0_2_target,
        verbose=verbose,
    )

    save_final_size_data(
        f"{data_folder}/final_size_gamma_2_r0_2_{r0_2_target}_"
        f"sigma_{sigma_12}.csv",
        "gamma_2",
        results_gamma_2,
        b,
    )

    plot_final_sizes(
        gamma_2_values,
        results_gamma_2,
        xlabel=r"Recovery rate $\gamma_2$",
        filename=(
            f"{plot_folder}/final_size_gamma_2_r0_2_{r0_2_target}_"
            f"sigma_{sigma_12}.{fileformat}"
        ),
        critical_values=[(beta_1 / r0_2_target, r"$\beta_1=\beta_2$")],
    )
