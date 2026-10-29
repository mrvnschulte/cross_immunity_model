"""
Checks the stability of the two-variant model numerically in two steps. First,
for given parameters, the expected equilibrium is computed and compared to the
state of the system after long-time simulation for multiple random initial
values (checking global stability). Second, random parameters are sampled and
it is checked if they admit a two-variant equilibrium. If so, then the
eigenvalues of the linearized system at the equilibrium are computed
numerically (checking local stability) and also the states after a long-time
simulation for random initial values are compared to the expected equilibrium
(checking global stability).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from cross_immunity_model import (
    basic_reproduction_number,
    check_validity,
    eigenvalues_linearized_system,
    endemic_equilibrium_equation,
    invasion_reproduction_number,
    random_simplex_points,
    sample_random_parameters,
    two_variant_model,
    x_bar,
)


def check_initial_values(
    beta_1: float,
    beta_2: float,
    gamma_1: float,
    gamma_2: float,
    sigma: float,
    b: float,
    N_initial_values: int = 100,
    tf: float = 1000,
    filename: str | None = None,
    verbose: bool = False,
    verbose_step: int = 10,
    seed: int = 42,
    dpi: int = 500,
) -> None:
    """
    Checks the stability of the two-variant model for given parameters and
    multiple random initial values.

    Parameters
    ----------
    beta_1 : float
        Transmission rate for variant 1.
    beta_2 : float
        Transmission rate for variant 2.
    gamma_1 : float
        Reocvery rate for variant 1.
    gamma_2 : float
        Recovery rate for variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.
    N_initial_values : int, optional
        Number of initial values to be checked. The default is 100.
    tf : float, optional
        Time until which to simulate. The default is 1000.
    filename : str or None, optional
        Filename to store plots. If None is given, the plots are not stored.
        The default is None.
    verbose : bool, optional
        Verbose mode. The default is False.
    seed : int, optional
        Seed of random number generator. The default is 42.
    dpi : int, optional
        Dots per inch for saving plots. The default is 500.

    Returns
    -------
    None.
    """
    rng = np.random.default_rng(seed=seed)
    x0_list = random_simplex_points(N_initial_values, dim=9, rng=rng)

    # Compute two-variant endemic equilibrium
    sol = root(
        endemic_equilibrium_equation,
        [1, 1],
        args=(beta_1, beta_2, gamma_1, gamma_2, sigma, b),
        method="hybr",
    )
    x_star = np.array(x_bar(sol.x, beta_1, beta_2, gamma_1, gamma_2, sigma, b))

    # For each initial value, solve the system until tf and compare the
    # distance to the computed equilibrium
    distances = []
    for i in range(N_initial_values):
        x0 = x0_list[i]
        sol_ode = solve_ivp(
            two_variant_model,
            (0, tf),
            x0,
            args=(beta_1, beta_2, gamma_1, gamma_2, sigma, b),
            vectorized=False,
        )
        final_x = sol_ode.y.T[-1]
        dist = np.linalg.norm(final_x - x_star)
        distances.append(dist)
        if verbose:
            if (i + 1) % verbose_step == 0:
                print(f"Computed {i+1} initial values")
    distances = np.array(distances)

    print("Mean distance to x_star:", distances.mean())
    print("Maximal distance to x_star:", max(distances))

    # Plot distance between long-term simulation and computed equilibrium
    plt.figure()
    plt.scatter(range(N_initial_values), distances)
    plt.xlabel("Index of initial value", fontsize=fontsize_text)
    plt.ylabel(r"Distance $||x(1000) - \hat x||$", fontsize=fontsize_text)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )
    plt.gca().yaxis.get_offset_text().set_fontsize(fontsize_ticks)
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()


def check_parameter_sets(
    beta_mean: float,
    beta_std: float,
    gamma_mean: float,
    gamma_std: float,
    sigma_alpha: float,
    sigma_beta: float,
    b_val: float,
    N_target_sets: int = 100,
    N_initial_values: int = 10,
    tf: float = 5000,
    threshold_r0: float = 1.0,
    filename: str | None = None,
    verbose: bool = False,
    dpi: int = 500,
) -> None:
    """
    Checks stability of the two-variant equilibrium for random parameters.

    Parameters
    ----------
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
    b_val : float
        Numerical value of birth and death rate.
    N_target_sets : int, optional
        Number of parameter sets to check. The default is 100.
    N_initial_values : int, optional
        Number of initial values per parameter set to check. The default is 10.
    tf : float, optional
        Time until which to simulate. The default is 5000.
    threshold_r0 : float, optional
        Threshold value of the invasion reproduction number which has to be
        exceeded. The default is 1.0.
    filename : str or None, optional
        Filename to store plots. If None is given, the plots are not stored.
        The default is None.
    verbose : bool, optional
        Verbose mode. The default is False.
    dpi : int, optional
        Dots per inch for saving plots. The default is 500.

    Returns
    -------
    None.
    """
    # Initialize variables
    valid_stats = []
    cycles = 0

    while len(valid_stats) < N_target_sets:
        # Sample random parameters for the model
        rng = np.random.default_rng(seed=cycles)
        cycles += 1
        beta, gamma, sigma_matrix = sample_random_parameters(
            2,
            beta_mean,
            beta_std,
            gamma_mean,
            gamma_std,
            sigma_alpha,
            sigma_beta,
            rng,
        )
        beta_1 = beta[0]
        beta_2 = beta[1]
        gamma_1 = gamma[0]
        gamma_2 = gamma[1]
        sigma_val = sigma_matrix[0, 1]

        # Compute (invasion) reproduction numbers and check if both are greater
        # than 1
        r_01, r_02 = basic_reproduction_number(beta, gamma, b_val)
        r_021 = invasion_reproduction_number(r_01, r_02, sigma_val)
        r_012 = invasion_reproduction_number(r_02, r_01, sigma_val)
        if not (r_012 > threshold_r0 and r_021 > threshold_r0):
            continue

        # Solve the system for the endemic equilibrium and check if the
        # two-variant equilibrium was found
        sol = root(
            endemic_equilibrium_equation,
            [1, 1],
            args=(beta_1, beta_2, gamma_1, gamma_2, sigma_val, b_val),
            method="hybr",
        )
        if min(sol.x) < 1e-3:
            continue
        x_star = np.array(
            x_bar(sol.x, beta_1, beta_2, gamma_1, gamma_2, sigma_val, b_val)
        )
        x_eq = x_star[[5, 6, 0, 7, 3, 8, 1, 4, 2]]

        # Compute the maximal real part of the eigenvalues of the linearized
        # system
        _, max_real_part, _ = eigenvalues_linearized_system(
            x_eq, beta, gamma, sigma_matrix, b_val
        )

        # Create list of random initial values
        x0_list = random_simplex_points(N_initial_values, dim=9, rng=rng)
        final_states = []
        dist_to_xstar = []

        # For each initial value, solve the system until tf and compare the
        # distance to the computed equilibrium
        for x0 in x0_list:
            sol_ode = solve_ivp(
                two_variant_model,
                (0, tf),
                x0,
                args=(beta_1, beta_2, gamma_1, gamma_2, sigma_val, b_val),
                vectorized=False,
            )
            final_x = sol_ode.y.T[-1]
            if not check_validity(final_x):
                continue
            final_states.append(final_x)
            dist_to_xstar.append(np.linalg.norm(final_x - x_star))

        # Convert to arrays
        if len(final_states) == 0:
            continue
        final_states = np.array(final_states)
        dist_to_xstar = np.array(dist_to_xstar)

        # Store statistics in a dictionary
        stats = {
            "set": len(valid_stats),
            "params": [beta, gamma, sigma_val, b_val],
            "mean_dist": dist_to_xstar.mean(),
            "std_dist": dist_to_xstar.std(),
            "max_dist": dist_to_xstar.max(),
            "min_dist": dist_to_xstar.min(),
            "mean_final": final_states.mean(axis=0),
            "r_01": r_01,
            "r_02": r_02,
            "r_012": r_012,
            "r_021": r_021,
            "max_eigval": max_real_part,
        }
        valid_stats.append(stats)

        if verbose:
            print(
                f"Valid Set {stats['set']}: "
                f"R_01={r_01:.3f}, R_02={r_02:.3f}, R_012={r_012:.3f}, R_021={r_021:.3f}, "
                f"mean_dist={stats['mean_dist']:.4e}, max_dist={stats['max_dist']:.4e}, cycles={cycles}"
            )

    # Plot maximal distance between long-term simulation and computed
    # equilibrium
    plt.figure()
    plt.scatter(
        [s["set"] for s in valid_stats], [s["max_dist"] for s in valid_stats]
    )
    plt.xlabel("Index of parameter set", fontsize=fontsize_text)
    plt.ylabel(
        f"Maximal distance\n$||x({tf}) - \\hat x||$", fontsize=fontsize_text
    )
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )
    plt.gca().yaxis.get_offset_text().set_fontsize(fontsize_ticks)
    if filename is not None:
        plt.savefig(
            f"{filename}_distance.{fileformat}", dpi=dpi, bbox_inches="tight"
        )
    plt.show()
    plt.close()

    # Plot maximal real part of eigenvalues at computed equilibria
    plt.figure()
    plt.scatter(
        [s["set"] for s in valid_stats], [s["max_eigval"] for s in valid_stats]
    )
    plt.axhline(color="grey", linestyle="dotted")
    plt.xlabel("Index of parameter set", fontsize=fontsize_text)
    plt.ylabel("Maximal real part of\neigenvalues", fontsize=fontsize_text)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )
    plt.gca().yaxis.get_offset_text().set_fontsize(fontsize_ticks)
    if filename is not None:
        plt.savefig(
            f"{filename}_eigval.{fileformat}", dpi=dpi, bbox_inches="tight"
        )
    plt.show()
    plt.close()

    # Plot statistics of invasion reproduction numbers
    plt.figure()
    max_val = max([min(s["r_012"], s["r_021"]) for s in valid_stats])
    bins = np.arange(1.0, max_val + 0.1, 0.1)
    plt.hist([min(s["r_012"], s["r_021"]) for s in valid_stats], bins=bins)
    plt.xlabel(r"$\min\{R_{0,1\to 2},R_{0,2\to 1}\}$", fontsize=fontsize_text)
    plt.ylabel("Number of\nparameter sets", fontsize=fontsize_text)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    if filename is not None:
        plt.savefig(
            f"{filename}_stats.{fileformat}", dpi=dpi, bbox_inches="tight"
        )
    plt.show()
    plt.close()

    total_max_real_part = max([s["max_eigval"] for s in valid_stats])
    print(f"The maximal real part of any eigenvalue is: {total_max_real_part}")


dpi = 1200
fontsize_text = 20
fontsize_ticks = 16
folder = "../plots"
fileformat = "pdf"

### STEP 1: Check for given parameters and multiple initial values

# Set parameters
sigma = 0.8
beta_1 = 0.3
beta_2 = 0.35
gamma_1 = 0.1
gamma_2 = 0.1
b = 0.01
N_initial_values = 100000
tf = 1000

# Set plot options and seed
seed = 42
filename = f"{folder}/different_initial_values_distance_{N_initial_values}.{fileformat}"
verbose_step = N_initial_values / 100

check_initial_values(
    sigma,
    beta_1,
    beta_2,
    gamma_1,
    gamma_2,
    b,
    N_initial_values=N_initial_values,
    tf=tf,
    filename=filename,
    verbose=True,
    verbose_step=verbose_step,
    seed=seed,
    dpi=dpi,
)


### STEP 2: Check for multiple parameter sets and initial values

beta_mean = 0.3
beta_std = 0.025
gamma_mean = 0.1
gamma_std = 0
sigma_alpha = 9
sigma_beta = 1
b = 0.01
N_target_sets = 1000
N_initial_values = 100
threshold_r0 = 1
filename = f"{folder}/different_parameters_new"
tf = 20000

check_parameter_sets(
    beta_mean,
    beta_std,
    gamma_mean,
    gamma_std,
    sigma_alpha,
    sigma_beta,
    b,
    N_target_sets=N_target_sets,
    N_initial_values=N_initial_values,
    threshold_r0=threshold_r0,
    filename=filename,
    tf=tf,
    verbose=True,
    dpi=dpi,
)
