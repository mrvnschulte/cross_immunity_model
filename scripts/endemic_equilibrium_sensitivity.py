"""
Plots the sensitivity of the two-variant equilibrium for multiple
configurations.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

from cross_immunity_model import (
    critical_r01,
    critical_r02,
    critical_sigma,
    endemic_equilibrium_equation,
    protected_1,
    protected_12,
    protected_2,
    susceptibles,
    x_bar,
)

# Variables for storing plots
folder = "../plots"
fileformat = "pdf"
dpi = 1200
fontsize_text = 20
fontsize_ticks = 16
alpha_grid = 0.7

# Variables for base scenario
beta_1 = 0.3
beta_2 = 0.35
gamma_1 = 0.1
gamma_2 = 0.1
sigma = 0.8
b = 0.01
r_01 = beta_1 / (gamma_1 + b)
r_02 = beta_2 / (gamma_2 + b)

compartment_names = [
    r"$I_{1,1}$",
    r"$I_{1,12}$",
    r"$I_{12,12}$",
    r"$I_{2,2}$",
    r"$I_{2,12}$",
    r"$I_{\emptyset,\emptyset}$",
    r"$I_{\emptyset,1}$",
    r"$I_{\emptyset,2}$",
    r"$I_{\emptyset,12}$",
]


### STEP 1: Plot sensitivity of i_1, i_2 on sigma

sigmas = np.linspace(0, 0.9, 10)
fig = plt.figure(figsize=(5, 5))

# Compute and plot variant 1 equilibrium
sol = root(
    endemic_equilibrium_equation,
    [1, 0],
    args=(beta_1, beta_2, gamma_1, gamma_2, 1, b),
    method="hybr",
)
plt.plot(sol.x[0], sol.x[1], "s")  # , label="Variant 1 equilibrium")
plt.axvline(sol.x[0], linestyle="dashed", color="black", alpha=0.4)

plt.annotate(
    "Strain 1 equilibrium",
    xy=(sol.x[0], sol.x[1]),
    xytext=(-6, 9),
    textcoords="offset points",
    fontsize=fontsize_ticks,
    ha="right",
)


# Compute and plot variant 2 equilibrium
sol = root(
    endemic_equilibrium_equation,
    [0, 1],
    args=(beta_1, beta_2, gamma_1, gamma_2, 1, b),
    method="hybr",
)
plt.plot(sol.x[0], sol.x[1], "s")  # , label="Variant 2 equilibrium")
plt.axhline(sol.x[1], linestyle="dashed", color="black", alpha=0.4)

plt.annotate(
    "Strain 2 equilibrium",
    xy=(sol.x[0], sol.x[1]),
    xytext=(5, 9),
    textcoords="offset points",
    fontsize=fontsize_ticks,
    ha="left",
)
plt.gca().set_prop_cycle(None)

# Compute and plot two-variant equilibria for multiple sigma (points)
for s in sigmas:
    sol = root(
        endemic_equilibrium_equation,
        [1, 1],
        args=(beta_1, beta_2, gamma_1, gamma_2, s, b),
        method="hybr",
    )
    plt.plot(sol.x[0], sol.x[1], "o", label=rf"${round(s, 1)}$")

# Compute and plot two-variant equilibria for multiple sigma (line)
sigmas_line = np.linspace(0, 1, 101)
i_solutions = []
x_solutions = []
for s in sigmas_line:
    sol = root(
        endemic_equilibrium_equation,
        [1, 1],
        args=(beta_1, beta_2, gamma_1, gamma_2, s, b),
        method="hybr",
    )
    i_solutions.append(sol.x)
    x_solutions.append(x_bar(sol.x, beta_1, beta_2, gamma_1, gamma_2, s, b))
i_solutions = np.array(i_solutions).T
x_solutions = np.array(x_solutions)
plt.plot(i_solutions[0], i_solutions[1], linestyle="dashed", color="black")

# Label plot
plt.xlim((-0.002, 0.065))
plt.ylim((-0.002, 0.07))
plt.xlabel(r"$I_1$", fontsize=fontsize_text)
plt.ylabel(r"$I_2$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
fig.legend(
    fontsize=fontsize_ticks,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    title=r"Cross-immunity $\mathbf{\sigma}$",
    title_fontproperties={
        "size": fontsize_ticks,
        "weight": "bold",
    },
)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/endemic_equilibrium_sensitivity_sigma.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close()


### STEP 2: Plot sensitivity of x on sigma

threshold_sigma = critical_sigma(r_02, r_01)

fig = plt.figure()
plt.semilogy(
    sigmas_line, x_solutions, label=[compartment_names[i] for i in range(9)]
)
plt.axvline(
    threshold_sigma,
    color="grey",
    linestyle="dashed",
    # label=r"$\sigma^{1\to 2}$",
)
plt.annotate(
    r"$\sigma^{1\to 2}$",
    xy=(threshold_sigma, 0),
    xycoords=("data", "axes fraction"),
    xytext=(-25, 20),
    textcoords="offset points",
    ha="center",
    va="top",
    fontsize=fontsize_ticks,
)

plt.xlabel(r"Cross-immunity $\sigma$", fontsize=fontsize_text)
plt.ylabel("Compartment size", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.xlim(sigmas_line[0], sigmas_line[-1])
plt.ylim(bottom=1e-5, top=1)
fig.legend(
    fontsize=fontsize_ticks,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=3,
)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/endemic_equilibrium_x_sensitivity_sigma.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close()


### STEP 3: Plot sensitivity of protection states on sigma

fig = plt.figure()
plt.plot(
    sigmas_line,
    [
        [susceptibles(x), protected_1(x), protected_2(x), protected_12(x)]
        for x in x_solutions
    ],
    label=["no strain", r"strain 1", r"strain 2", r"both strains"],
)
plt.axvline(
    threshold_sigma,
    color="grey",
    linestyle="dashed",
    # label=r"$\sigma_{1\to 2}^*$",
)
plt.annotate(
    r"$\sigma^{1\to 2}$",
    xy=(threshold_sigma, 0.67),
    xycoords=("data", "axes fraction"),
    xytext=(-25, 0),
    textcoords="offset points",
    ha="center",
    va="top",
    fontsize=fontsize_ticks,
)
plt.xlabel(r"Cross-immunity $\sigma$", fontsize=fontsize_text)
plt.ylabel("Compartment size", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.xlim(sigmas_line[0], sigmas_line[-1])
fig.legend(
    fontsize=fontsize_ticks,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=2,
    title="Protected against",
    title_fontproperties={
        "size": fontsize_ticks,
        "weight": "bold",
    },
)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/endemic_equilibrium_p_sensitivity_sigma.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close()


### STEP 4: Plot sensitivity of i_1, i_2 on sigma

fig = plt.figure()
plt.plot(
    sigmas_line,
    i_solutions.T,
    label=[r"$I_1$", r"$I_2$"],
)
plt.axvline(
    threshold_sigma,
    color="grey",
    linestyle="dashed",
    # label=r"$\sigma^{1\to 2}$",
)
plt.annotate(
    r"$\sigma^{1\to 2}$",
    xy=(threshold_sigma, 0.58),
    xycoords=("data", "axes fraction"),
    xytext=(-25, 0),
    textcoords="offset points",
    ha="center",
    va="top",
    fontsize=fontsize_ticks,
)
plt.xlabel(r"Cross-immunity $\sigma$", fontsize=fontsize_text)
plt.ylabel("Compartment size", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.xlim(sigmas_line[0], sigmas_line[-1])
plt.legend(fontsize=fontsize_ticks)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/endemic_equilibrium_i_sensitivity_sigma.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close()


# ### STEP 5: Plot sensitivity of i_1, i_2 on beta_2

beta_2s = np.linspace(0.2, 0.55, 8)
fig = plt.figure(figsize=(5, 5))

# Compute and plot two-variant equilibrium for multiple beta_1 (points)
for b_2 in beta_2s:
    sol = root(
        endemic_equilibrium_equation,
        [1, 1],
        args=(beta_1, b_2, gamma_1, gamma_2, sigma, b),
        method="hybr",
    )
    plt.plot(sol.x[0], sol.x[1], "o", label=rf"${round(b_2, 2)}$")

# Compute and plot two-variant equilibrium for multiple beta_1 (line)
beta_2s_line = np.linspace(0.2, 0.55, 101)
i_solutions = []
x_solutions = []
for b_2 in beta_2s_line:
    sol = root(
        endemic_equilibrium_equation,
        [1, 1],
        args=(beta_1, b_2, gamma_1, gamma_2, sigma, b),
        method="hybr",
    )
    i_solutions.append(sol.x)
    x_solutions.append(
        x_bar(sol.x, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
    )
i_solutions = np.array(i_solutions).T
x_solutions = np.array(x_solutions)
plt.plot(i_solutions[0], i_solutions[1], linestyle="dashed", color="black")

# Label plot
plt.xlabel(r"$I_1$", fontsize=fontsize_text)
plt.ylabel(r"$I_2$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
fig.legend(
    fontsize=fontsize_ticks,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    title=r"Transmission rate $\mathbf{\beta_2}$",
    title_fontproperties={
        "size": fontsize_ticks,
        "weight": "bold",
    },
)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/endemic_equilibrium_sensitivity_r02.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close()

### STEP 6: Plot sensitivity of x on beta_2

lower_threshold_beta2 = (gamma_2 + b) * critical_r02(r_01, sigma)
upper_threshold_beta2 = (gamma_2 + b) * critical_r01(r_01, sigma)

fig = plt.figure()
plt.semilogy(
    beta_2s_line, x_solutions, label=[compartment_names[i] for i in range(9)]
)
plt.axvline(
    lower_threshold_beta2,
    color="grey",
    linestyle="dashed",
    # label=r"$R_{0,2}^{2\to1}$",
)
plt.annotate(
    r"$R_{0,2}^{2\to1}$",
    xy=(lower_threshold_beta2, 0.15),
    xycoords=("data", "axes fraction"),
    xytext=(5, 0),
    textcoords="offset points",
    ha="left",
    va="top",
    fontsize=fontsize_ticks,
)
plt.axvline(
    upper_threshold_beta2,
    color="grey",
    linestyle="dotted",
    # label=r"$R_{0,2}^{1\to2}$",
)
plt.annotate(
    r"$R_{0,2}^{1\to2}$",
    xy=(upper_threshold_beta2, 0.15),
    xycoords=("data", "axes fraction"),
    xytext=(2, 0),
    textcoords="offset points",
    ha="left",
    va="top",
    fontsize=fontsize_ticks,
)
plt.xlabel(r"Transmission rate $\beta_2$", fontsize=fontsize_text)
plt.ylabel("Compartment size", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
fig.legend(
    fontsize=fontsize_ticks,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
)
plt.xlim(beta_2s_line[0], beta_2s_line[-1])
plt.ylim(bottom=1e-5, top=1)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/endemic_equilibrium_x_sensitivity_r02.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close()


### STEP 7: Plot sensitivity of protection states on beta_2

fig = plt.figure()
plt.plot(
    beta_2s_line,
    [
        [susceptibles(x), protected_1(x), protected_2(x), protected_12(x)]
        for x in x_solutions
    ],
    label=["no strain", r"strain 1", r"strain 2", r"both strains"],
)
plt.axvline(
    lower_threshold_beta2,
    color="grey",
    linestyle="dashed",
    # label=r"$R_{0,2}^{2\to1}$",
)
plt.annotate(
    r"$R_{0,2}^{2\to1}$",
    xy=(lower_threshold_beta2, 0.75),
    xycoords=("data", "axes fraction"),
    xytext=(25, 0),
    textcoords="offset points",
    ha="center",
    va="top",
    fontsize=fontsize_ticks,
)
plt.axvline(
    upper_threshold_beta2,
    color="grey",
    linestyle="dotted",
    # label=r"$R_{0,2}^{1\to2}$",
)
plt.annotate(
    r"$R_{0,2}^{1\to2}$",
    xy=(upper_threshold_beta2, 0.75),
    xycoords=("data", "axes fraction"),
    xytext=(-25, 0),
    textcoords="offset points",
    ha="center",
    va="top",
    fontsize=fontsize_ticks,
)
plt.xlabel(r"Transmission rate $\beta_2$", fontsize=fontsize_text)
plt.ylabel("Compartment size", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.xlim(beta_2s_line[0], beta_2s_line[-1])
fig.legend(
    fontsize=fontsize_ticks,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=2,
    title="Protected against",
    title_fontproperties={
        "size": fontsize_ticks,
        "weight": "bold",
    },
)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/endemic_equilibrium_p_sensitivity_r02.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close()


# ### STEP 8: Plot sensitivity of i_1, i_2 on beta_2

fig = plt.figure()
plt.plot(
    beta_2s_line,
    i_solutions.T,
    label=[r"$I_1$", r"$I_2$"],
)
plt.axvline(
    lower_threshold_beta2,
    color="grey",
    linestyle="dashed",
    # label=r"$R_{0,2}^{2\to1}$",
)
plt.annotate(
    r"$R_{0,2}^{2\to1}$",
    xy=(lower_threshold_beta2, 0.9),
    xycoords=("data", "axes fraction"),
    xytext=(25, 0),
    textcoords="offset points",
    ha="center",
    va="top",
    fontsize=fontsize_ticks,
)

plt.axvline(
    upper_threshold_beta2,
    color="grey",
    linestyle="dotted",
    # label=r"$R_{0,2}^{1\to2}$",
)
plt.annotate(
    r"$R_{0,2}^{1\to2}$",
    xy=(upper_threshold_beta2, 0.28),
    xycoords=("data", "axes fraction"),
    xytext=(-25, 0),
    textcoords="offset points",
    ha="center",
    va="top",
    fontsize=fontsize_ticks,
)
plt.xlabel(r"Transmission rate $\beta_2$", fontsize=fontsize_text)
plt.ylabel("Compartment size", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.xlim(beta_2s_line[0], beta_2s_line[-1])
plt.legend(fontsize=fontsize_ticks)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/endemic_equilibrium_i_sensitivity_r02.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close()


### STEP 9: Plot sensitivity of i_1, i_2 on sigma for multiple beta_2

sigmas = np.linspace(0, 1, 11)
beta_2s = [0.2, 0.25, 0.3, 0.35, 0.45, 0.55]
fig = plt.figure(figsize=(5, 5))

# Compute and plot variant 1 equilibrium
sol = root(
    endemic_equilibrium_equation,
    [1, 0],
    args=(beta_1, beta_2, gamma_1, gamma_2, 1, b),
    method="hybr",
)
plt.axvline(sol.x[0], linestyle="dashed", color="black", alpha=0.4)

# Plot i_1-i_2 line
i1_line = np.linspace(0, sol.x[0], 101)
i1_i2_line = sol.x[0] - i1_line
plt.plot(i1_line, i1_i2_line, linestyle="dotted", color="black", alpha=0.4)

for b_2 in beta_2s:
    # Plot variant 2 equilibrium
    sol = root(
        endemic_equilibrium_equation,
        [0, 1],
        args=(beta_1, b_2, gamma_1, gamma_2, 1, b),
        method="hybr",
    )
    plt.axhline(sol.x[1], linestyle="dashed", color="black", alpha=0.4)
    plt.gca().set_prop_cycle(None)

    # Compute and plot two-variant equilibria for multiple sigma (points)
    for s in sigmas:
        sol = root(
            endemic_equilibrium_equation,
            [1, 1],
            args=(beta_1, b_2, gamma_1, gamma_2, s, b),
            method="hybr",
        )
        plt.plot(
            sol.x[0],
            sol.x[1],
            "o",
            label=(rf"${round(s, 1)}$" if b_2 == 0.2 else None),
        )
        plt.text(
            sol.x[0] + 0.004,
            sol.x[1] - 0.0007,
            rf"$\beta_2={round(b_2, 2)}$" if s == 0 else "",
            fontsize=fontsize_ticks,
        )

    # Compute and plot two-variant equilibria for multiple sigma (line)
    sigmas_line = np.linspace(0, 1, 101)
    i_solutions = []
    x_solutions = []
    for s in sigmas_line:
        sol = root(
            endemic_equilibrium_equation,
            [1, 1],
            args=(beta_1, b_2, gamma_1, gamma_2, s, b),
            method="hybr",
        )
        i_solutions.append(sol.x)
        x_solutions.append(x_bar(sol.x, beta_1, b_2, gamma_1, gamma_2, s, b))

    i_solutions = np.array(i_solutions).T
    x_solutions = np.array(x_solutions)
    plt.plot(i_solutions[0], i_solutions[1], linestyle="dashed", color="black")

# Label plot
plt.text(
    0.027,
    0.027,
    r"$I_1+I_2=I_1^*$",
    fontsize=fontsize_ticks,
    ha="right",
    va="top",
)
plt.xlabel(r"$I_1$", fontsize=fontsize_text)
plt.ylabel(r"$I_2$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
fig.legend(
    fontsize=fontsize_ticks,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    title=r"Cross-immunity $\mathbf{\sigma}$",
    title_fontproperties={
        "size": fontsize_ticks,
        "weight": "bold",
    },
)
plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/endemic_equilibrium_multisensitivity_sigma.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close()


### STEP 10: Plot sensitivity of i_1, i_2 on sigma for multiple beta_2

sigmas = [0.7, 0.8, 0.9]
beta_2s = np.linspace(0.2, 0.6, 9)
fig = plt.figure(figsize=(5, 5))

for s in sigmas:
    upper_threshold_beta2 = (gamma_2 + b) * critical_r01(r_01, s)

    # Compute and plot two-variant equilibria for multiple beta_2 (points)
    plt.gca().set_prop_cycle(None)
    for b_2 in beta_2s:
        if b_2 > upper_threshold_beta2:
            continue
        sol = root(
            endemic_equilibrium_equation,
            [1, 1],
            args=(beta_1, b_2, gamma_1, gamma_2, s, b),
            method="hybr",
        )
        plt.plot(
            sol.x[0],
            sol.x[1],
            "o",
            label=(rf"${round(b_2, 2)}$" if s == 0.7 else None),
        )

    # Compute and plot two-variant equilibria for multiple beta_2 (line)
    beta_2s_line = np.linspace(0.2, min(0.6, upper_threshold_beta2), 101)
    i_solutions = []
    x_solutions = []
    for b_2 in beta_2s_line:
        sol = root(
            endemic_equilibrium_equation,
            [1, 1],
            args=(beta_1, b_2, gamma_1, gamma_2, s, b),
            method="hybr",
        )
        i_solutions.append(sol.x)
        x_solutions.append(x_bar(sol.x, beta_1, b_2, gamma_1, gamma_2, s, b))

    i_solutions = np.array(i_solutions).T
    x_solutions = np.array(x_solutions)
    plt.plot(i_solutions[0], i_solutions[1], linestyle="dashed", color="black")

# Label plot
plt.text(0.0005, 0.0450, r"$\sigma=0.9$", fontsize=fontsize_ticks)
plt.text(-0.006, 0.072, r"$\sigma=0.8$", fontsize=fontsize_ticks)
plt.text(0.025, 0.0525, r"$\sigma=0.7$", fontsize=fontsize_ticks)
plt.text(0.010, 0.022, r"$I_1=I_2$", fontsize=fontsize_ticks)

plt.plot(i1_line, i1_line, color="black", linestyle="dashed", alpha=0.4)
plt.xlabel(r"$I_1$", fontsize=fontsize_text)
plt.ylabel(r"$I_2$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
fig.legend(
    fontsize=fontsize_ticks,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    title=r"Transmission rate $\mathbf{\beta_2}$",
    title_fontproperties={
        "size": fontsize_ticks,
        "weight": "bold",
    },
)

plt.grid(alpha=alpha_grid)
plt.savefig(
    f"{folder}/endemic_equilibrium_multisensitivity_r02.{fileformat}",
    dpi=dpi,
    bbox_inches="tight",
)
plt.show()
plt.close()
