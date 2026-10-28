"""
This script compares the two-variant model with co-infection with an otherwise
equal model without the possibility of co-infection and plots both in one plot.
"""

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from cross_immunity_model import two_variant_model


def two_variant_model_without_coinfection(
    t: float,
    x: ArrayLike,
    beta1: float,
    beta2: float,
    gamma1: float,
    gamma2: float,
    sigma: float,
    b: float = 0.0,
) -> list:
    """
    Two-variant cross-immmunity model with demographics and no co-infection.

    Parameters
    ----------
    t : float
        Time (not used).
    x : array-like
        State of the system.
    beta1 : float
        Transmission rate of variant 1.
    beta2 : float
        Transmission rate of variant 2.
    gamma1 : float
        Recovery rate of variant 1.
    gamma2 : float
        Recovery rate of variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.

    Returns
    -------
    f : list
        Derivatives of the state.
    """

    x1, x2, x3, x4, x5, x6, x7, x8 = x

    f1 = -gamma1 * x1 + (1 - sigma) * beta1 * x5 * (x1 + x2) - b * x1
    f2 = (
        -gamma1 * x2
        + sigma * beta1 * x5 * (x1 + x2)
        + beta1 * x7 * (x1 + x2)
        - b * x2
    )
    f3 = -gamma2 * x3 + (1 - sigma) * beta2 * x5 * (x3 + x4) - b * x3
    f4 = (
        -gamma2 * x4
        + sigma * beta2 * x5 * (x3 + x4)
        + beta2 * x6 * (x3 + x4)
        - b * x4
    )
    f5 = -beta1 * x5 * (x1 + x2) - beta2 * x5 * (x3 + x4) + b - b * x5
    f6 = -beta2 * x6 * (x3 + x4) + gamma1 * x1 - b * x6
    f7 = -beta1 * x7 * (x1 + x2) + gamma2 * x3 - b * x7
    f8 = gamma1 * x2 + gamma2 * x4 - b * x8

    f = [f1, f2, f3, f4, f5, f6, f7, f8]

    return f


dpi = 1200
fileformat = "pdf"
folder = "../plots"
fontsize_text = 16
fontsize_ticks = 14
alpha_grid = 0.7

# Set model parameters
infectious_at_start = 0.001
t0 = 0
tf = 250
beta1 = 0.3
beta2 = 0.2
gamma1 = 0.1
gamma2 = 0.1
sigma = 0
b = 0

# Construct initial conditions from parameter
x0_coinfection = [0.0 for i in range(9)]
x0_no_coinfection = [0.0 for i in range(8)]

x0_coinfection[0] = infectious_at_start
x0_coinfection[3] = infectious_at_start
x0_no_coinfection[0] = infectious_at_start
x0_no_coinfection[2] = infectious_at_start

x0_coinfection[5] = 1 - sum(x0_coinfection)
x0_no_coinfection[4] = 1 - sum(x0_no_coinfection)

# Solve both models
t_eval = np.arange(t0, tf)
sol_coinfection = solve_ivp(
    two_variant_model,
    (t0, tf),
    x0_coinfection,
    t_eval=t_eval,
    args=(beta1, beta2, gamma1, gamma2, sigma, b),
)
sol_no_coinfection = solve_ivp(
    two_variant_model_without_coinfection,
    (t0, tf),
    x0_no_coinfection,
    t_eval=t_eval,
    args=(beta1, beta2, gamma1, gamma2, sigma, b),
)

# Compute infectious fraction for both strains and models
infectious_with_1_coinfection = (
    sol_coinfection.y[0] + sol_coinfection.y[1] + sol_coinfection.y[2]
)
infectious_with_2_coinfection = (
    sol_coinfection.y[2] + sol_coinfection.y[3] + sol_coinfection.y[4]
)
infectious_with_1_no_coinfection = (
    sol_no_coinfection.y[0] + sol_no_coinfection.y[1]
)
infectious_with_2_no_coinfection = (
    sol_no_coinfection.y[2] + sol_no_coinfection.y[3]
)

# Plot the results and store them
fig = plt.figure()
plt.plot(
    sol_coinfection.t,
    infectious_with_1_coinfection,
    linestyle="-",
    color="tab:blue",
)
plt.plot(
    sol_coinfection.t,
    infectious_with_2_coinfection,
    linestyle="-",
    color="tab:orange",
)
plt.plot(
    sol_coinfection.t, sol_coinfection.y[2], linestyle="-", color="tab:green"
)
plt.plot(
    sol_coinfection.t,
    infectious_with_1_no_coinfection,
    linestyle="--",
    color="tab:blue",
)
plt.plot(
    sol_coinfection.t,
    infectious_with_2_no_coinfection,
    linestyle="--",
    color="tab:orange",
)
blue_line = mlines.Line2D([], [], color="tab:blue", label="strain 1")
orange_line = mlines.Line2D([], [], color="tab:orange", label="strain 2")
green_line = mlines.Line2D([], [], color="tab:green", label="both strains")
solid_line = mlines.Line2D(
    [], [], color="black", linestyle="-", label="with co-infection"
)
dashed_line = mlines.Line2D(
    [], [], color="black", linestyle="--", label="without co-infection"
)
legend_infectious = fig.legend(
    handles=[blue_line, orange_line, green_line],
    title="Infectious with",
    title_fontproperties={"weight": "bold", "size": fontsize_text},
    loc="upper left",
    bbox_to_anchor=(0.03, 0.025),
    fontsize=fontsize_text,
    frameon=True,
)

legend_model = fig.legend(
    handles=[solid_line, dashed_line],
    title="Model",
    title_fontproperties={"weight": "bold", "size": fontsize_text},
    loc="upper right",
    bbox_to_anchor=(0.97, 0.025),
    fontsize=fontsize_text,
    frameon=True,
)
plt.xlabel(r"Time $t$", fontsize=fontsize_text)
plt.ylabel("Fraction of the population", fontsize=fontsize_text)
plt.xlim(t0, tf)
plt.ylim(0, None)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.grid(alpha=alpha_grid)
plt.tight_layout()
plt.savefig(
    f"{folder}/comparison_independent_transmission.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
    bbox_extra_artists=(legend_infectious, legend_model),
)
plt.show()

fig = plt.figure()
plt.plot(
    sol_coinfection.t, sol_coinfection.y[6], linestyle="-", color="tab:blue"
)
plt.plot(
    sol_coinfection.t, sol_coinfection.y[7], linestyle="-", color="tab:orange"
)
plt.plot(
    sol_coinfection.t, sol_coinfection.y[8], linestyle="-", color="tab:green"
)
plt.plot(
    sol_coinfection.t,
    sol_coinfection.y[6] + sol_coinfection.y[7] + sol_coinfection.y[8],
    linestyle="-",
    color="tab:red",
)

plt.plot(
    sol_coinfection.t,
    sol_no_coinfection.y[5],
    linestyle="--",
    color="tab:blue",
)
plt.plot(
    sol_coinfection.t,
    sol_no_coinfection.y[6],
    linestyle="--",
    color="tab:orange",
)
plt.plot(
    sol_coinfection.t,
    sol_no_coinfection.y[7],
    linestyle="--",
    color="tab:green",
)
plt.plot(
    sol_coinfection.t,
    sol_no_coinfection.y[5]
    + sol_no_coinfection.y[6]
    + sol_no_coinfection.y[7],
    linestyle="--",
    color="tab:red",
)
blue_line = mlines.Line2D([], [], color="tab:blue", label="strain 1")
orange_line = mlines.Line2D([], [], color="tab:orange", label="strain 2")
green_line = mlines.Line2D([], [], color="tab:green", label="both strains")
red_line = mlines.Line2D([], [], color="tab:red", label="any strain")
solid_line = mlines.Line2D(
    [], [], color="black", linestyle="-", label="with co-infection"
)
dashed_line = mlines.Line2D(
    [], [], color="black", linestyle="--", label="without co-infection"
)
legend_infectious = fig.legend(
    handles=[blue_line, orange_line, green_line, red_line],
    title="Recovered from",
    title_fontproperties={"weight": "bold", "size": fontsize_text},
    loc="upper left",
    bbox_to_anchor=(0.03, 0.025),
    fontsize=fontsize_text,
    frameon=True,
)

legend_model = fig.legend(
    handles=[solid_line, dashed_line],
    title="Model",
    title_fontproperties={"weight": "bold", "size": fontsize_text},
    loc="upper right",
    bbox_to_anchor=(0.97, 0.025),
    fontsize=fontsize_text,
    frameon=True,
)
plt.xlabel(r"Time $t$", fontsize=fontsize_text)
plt.ylabel("Fraction of the population", fontsize=fontsize_text)
plt.xlim(t0, tf)
plt.ylim(0, 1)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.grid(alpha=alpha_grid)
plt.tight_layout()
plt.savefig(
    f"{folder}/comparison_independent_transmission_recovered.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
    bbox_extra_artists=(legend_infectious, legend_model),
)
plt.show()
