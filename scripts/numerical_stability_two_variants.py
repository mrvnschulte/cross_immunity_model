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
import sympy as sp

from cross_immunity_model import (
    basic_reproduction_number,
    check_validity,
    endemic_equilibrium_equation,
    invasion_reproduction_number,
    random_simplex_points,
    two_variant_model,
    x_bar,
)


def sample_parameters(rng: np.random.Generator) -> dict:
    """
    Sample random parameters for the model.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    dict
        Dictionary of model parameters.
    """
    sigma = rng.uniform(0.0, 1.0)
    beta1 = rng.uniform(0.1, 2.0)
    beta2 = rng.uniform(0.1, 2.0)
    gamma1 = rng.uniform(0.1, beta1)
    gamma2 = rng.uniform(0.1, beta2)
    b = rng.uniform(0.01, 0.1)
    return dict(
        sigma=sigma,
        beta1=beta1,
        beta2=beta2,
        gamma1=gamma1,
        gamma2=gamma2,
        b=b,
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
    N_target_sets: int = 100,
    N_initial_values: int = 10,
    tf: float = 5000,
    threshold_r0: float = 1.0,
    filename: str | None = None,
    verbose: bool = False,
    seed: int = 42,
    dpi: int = 500,
) -> None:
    """
    Checks stability of the two-variant equilibrium for random parameters.

    Parameters
    ----------
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
    seed : int, optional
        Seed of random number generator. The default is 42.
    dpi : int, optional
        Dots per inch for saving plots. The default is 500.

    Returns
    -------
    None.
    """
    # Initialize variables
    rng = np.random.default_rng(seed=seed)
    valid_stats = []

    x1, x2, x3, x4, x5, x6, x7, x8, x9 = sp.symbols(
        "x1 x2 x3 x4 x5 x6 x7 x8 x9"
    )
    xs = [x1, x2, x3, x4, x5, x6, x7, x8, x9]
    gamma1, gamma2, beta1, beta2, sigma, b = sp.symbols(
        "gamma1 gamma2 beta1 beta2 sigma b", positive=True
    )

    # Linearize the model symbolically
    F = two_variant_model(0, xs, beta1, beta2, gamma1, gamma2, sigma, b)
    J = sp.Matrix(F).jacobian(xs)

    cycles = 0
    while len(valid_stats) < N_target_sets:
        # Sample random parameters for the model
        rng = np.random.default_rng(seed=cycles)
        cycles += 1
        params = sample_parameters(rng)
        beta_1 = params["beta1"]
        beta_2 = params["beta2"]
        gamma_1 = params["gamma1"]
        gamma_2 = params["gamma2"]
        sigma_val = params["sigma"]
        b_val = params["b"]

        # Compute (invasion) reproduction numbers and check if both are greater
        # than 1
        r_01, r_02 = basic_reproduction_number(
            [beta_1, beta_2], [gamma_1, gamma_2], b_val
        )
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

        # Replace the symbolic variables by its numerical values and compute
        # the maximal real part of the eigenvalues of the linearized system
        x_star_dict = {
            x1: x_star[0],
            x2: x_star[1],
            x3: x_star[2],
            x4: x_star[3],
            x5: x_star[4],
            x6: x_star[5],
            x7: x_star[6],
            x8: x_star[7],
            x9: x_star[8],
            b: b_val,
            beta1: beta_1,
            beta2: beta_2,
            gamma1: gamma_1,
            gamma2: gamma_2,
            sigma: sigma_val,
        }
        J_num = J.subs(x_star_dict)
        J_num = np.array(J_num.evalf().tolist(), dtype=float)
        eigvals = np.linalg.eigvals(J_num)
        max_real_part = np.max(eigvals.real)

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
            "params": params,
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
fileformat = "jpg"

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
filename = f"{folder}/different_initial_values_distance.{fileformat}"
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

N_target_sets = 1000
N_initial_values = 100
threshold_r0 = 1
filename = f"{folder}/different_parameters"
tf = 5000
seed = 42

check_parameter_sets(
    N_target_sets=N_target_sets,
    N_initial_values=N_initial_values,
    threshold_r0=threshold_r0,
    filename=filename,
    tf=tf,
    verbose=True,
    seed=seed,
    dpi=dpi,
)
