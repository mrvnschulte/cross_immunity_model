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

# Variables for base scenario
beta_1 = 0.3
beta_2 = 0.35
gamma_1 = 0.1
gamma_2 = 0.1
sigma = 0.8
b = 0.01

# Variables for storing plots
folder = "../plots"
fileformat = "jpg"
dpi = 500

### STEP 1: Plot sensitivity of i_1, i_2 on sigma

sigmas = np.linspace(0, 0.9, 10)
plt.figure(figsize=(5, 5))

# Compute and plot variant 1 equilibrium
sol = root(
    endemic_equilibrium_equation,
    [1, 0],
    args=(beta_1, beta_2, gamma_1, gamma_2, 1, b),
    method="hybr",
)
plt.plot(sol.x[0], sol.x[1], "s", label="Variant 1 equilibrium")
plt.axvline(sol.x[0], linestyle="dashed", color="black", alpha=0.4)

# Compute and plot variant 2 equilibrium
sol = root(
    endemic_equilibrium_equation,
    [0, 1],
    args=(beta_1, beta_2, gamma_1, gamma_2, 1, b),
    method="hybr",
)
plt.plot(sol.x[0], sol.x[1], "s", label="Variant 2 equilibrium")
plt.axhline(sol.x[1], linestyle="dashed", color="black", alpha=0.4)
plt.gca().set_prop_cycle(None)

# Compute and plot two-variant equilibria for multiple sigma (points)
for s in sigmas:
    sol = root(
        endemic_equilibrium_equation,
        [1, 1],
        args=(beta_1, beta_2, gamma_1, gamma_2, s, b),
        method="hybr",
    )
    plt.plot(sol.x[0], sol.x[1], "o", label=rf"$\sigma={round(s, 1)}$")

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
plt.ylim((-0.002, 0.065))
plt.xlabel("Infectious with variant 1")
plt.ylabel("Infectious with variant 2")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    f"{folder}/Endemic_equilibrium_sensitivity_sigma.{fileformat}", dpi=dpi
)
plt.show()


### STEP 2: Plot sensitivity of x on sigma

threshold_sigma = critical_sigma(
    beta_1 / (gamma_1 + b), beta_2 / (gamma_2 + b)
)

plt.semilogy(sigmas_line, x_solutions, label=[rf"$x_{i+1}$" for i in range(9)])
plt.axvline(
    threshold_sigma,
    color="grey",
    linestyle="dashed",
    label=r"$\sigma_{1\to 2}^*$",
)
plt.xlabel(r"Cross-immunity $\sigma$")
plt.ylabel("Compartment size")
plt.ylim(bottom=1e-5, top=1)
plt.legend(loc="lower left")
plt.grid()
plt.tight_layout()
plt.savefig(
    f"{folder}/Endemic_equilibrium_x_sensitivity_sigma.{fileformat}", dpi=dpi
)
plt.show()


### STEP 3: Plot sensitivity of protection states on sigma

plt.plot(
    sigmas_line,
    [
        [susceptibles(x), protected_1(x), protected_2(x), protected_12(x)]
        for x in x_solutions
    ],
    label=["s", r"$p_1$", r"$p_2$", r"$p_{12}$"],
)
plt.axvline(
    threshold_sigma,
    color="grey",
    linestyle="dashed",
    label=r"$\sigma_{1\to 2}^*$",
)
plt.xlabel(r"Cross-immunity $\sigma$")
plt.ylabel("Compartment size")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    f"{folder}/Endemic_equilibrium_p_sensitivity_sigma.{fileformat}", dpi=dpi
)
plt.show()


### STEP 4: Plot sensitivity of i_1, i_2 on sigma

plt.plot(
    sigmas_line,
    i_solutions.T,
    label=[r"$i_1$", r"$i_2$"],
)
plt.axvline(
    threshold_sigma,
    color="grey",
    linestyle="dashed",
    label=r"$\sigma_{1\to 2}^*$",
)
plt.xlabel(r"Cross-immunity $\sigma$")
plt.ylabel("Compartment size")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    f"{folder}/Endemic_equilibrium_i_sensitivity_sigma.{fileformat}", dpi=dpi
)
plt.show()


### STEP 5: Plot sensitivity of i_1, i_2 on beta_2

beta_2s = np.linspace(0.2, 0.55, 8)
plt.figure(figsize=(5, 5))

# Compute and plot two-variant equilibrium for multiple beta_1 (points)
for b_2 in beta_2s:
    sol = root(
        endemic_equilibrium_equation,
        [1, 1],
        args=(beta_1, b_2, gamma_1, gamma_2, sigma, b),
        method="hybr",
    )
    plt.plot(sol.x[0], sol.x[1], "o", label=rf"$\beta_2={round(beta_2, 2)}$")

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
plt.xlabel("Infectious with variant 1")
plt.ylabel("Infectious with variant 2")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    f"{folder}/Endemic_equilibrium_sensitivity_r02.{fileformat}", dpi=dpi
)
plt.show()


### STEP 6: Plot sensitivity of x on beta_2

lower_threshold_beta2 = (gamma_2 + b) * critical_r01(
    beta_1 / (gamma_1 + b), sigma
)
upper_threshold_beta2 = (gamma_2 + b) * critical_r02(
    beta_1 / (gamma_1 + b), sigma
)

plt.semilogy(
    beta_2s_line, x_solutions, label=[rf"$x_{i+1}$" for i in range(9)]
)
plt.axvline(
    lower_threshold_beta2,
    color="grey",
    linestyle="dashed",
    label=r"$R_{0,2}^{2\to1}$",
)
plt.axvline(
    upper_threshold_beta2,
    color="grey",
    linestyle="dotted",
    label=r"$R_{0,2}^{1\to2}$",
)
plt.xlabel(r"Transmission rate $\beta_2$")
plt.ylabel("Compartment size")
plt.ylim(bottom=1e-5, top=1)
plt.legend(loc=(1.02, 1e-3))
plt.grid()
plt.tight_layout()
plt.savefig(
    f"{folder}/Endemic_equilibrium_x_sensitivity_r02.{fileformat}", dpi=dpi
)
plt.show()


### STEP 7: Plot sensitivity of protection states on beta_2

plt.plot(
    beta_2s_line,
    [
        [susceptibles(x), protected_1(x), protected_2(x), protected_12(x)]
        for x in x_solutions
    ],
    label=["s", r"$p_1$", r"$p_2$", r"$p_{12}$"],
)
plt.axvline(
    lower_threshold_beta2,
    color="grey",
    linestyle="dashed",
    label=r"$R_{0,2}^{2\to1}$",
)
plt.axvline(
    upper_threshold_beta2,
    color="grey",
    linestyle="dotted",
    label=r"$R_{0,2}^{1\to2}$",
)
plt.xlabel(r"Transmission rate $\beta_2$")
plt.ylabel("Compartment size")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    f"{folder}/Endemic_equilibrium_p_sensitivity_r02.{fileformat}", dpi=dpi
)
plt.show()


### STEP 8: Plot sensitivity of i_1, i_2 on beta_2

plt.plot(
    beta_2s_line,
    i_solutions.T,
    label=[r"$i_1$", r"$i_2$"],
)
plt.axvline(
    lower_threshold_beta2,
    color="grey",
    linestyle="dashed",
    label=r"$R_{0,2}^{2\to1}$",
)
plt.axvline(
    upper_threshold_beta2,
    color="grey",
    linestyle="dotted",
    label=r"$R_{0,2}^{1\to2}$",
)
plt.xlabel(r"Transmission rate $\beta_2$")
plt.ylabel("Compartment size")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    f"{folder}/Endemic_equilibrium_i_sensitivity_r02.{fileformat}", dpi=dpi
)
plt.show()


### STEP 9: Plot sensitivity of i_1, i_2 on sigma for multiple beta_2

sigmas = np.linspace(0, 1, 11)
beta_2s = [0.2, 0.25, 0.3, 0.35, 0.45, 0.55]
plt.figure(figsize=(5, 5))

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
            label=(rf"$\sigma={round(s, 1)}$" if b_2 == 0.2 else None),
        )
        plt.text(
            sol.x[0] + 0.004,
            sol.x[1] - 0.0007,
            rf"$\beta_2={round(b_2, 2)}$" if s == 0 else "",
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
plt.ylim(bottom=-0.002)
plt.xlabel("Infectious with variant 1")
plt.ylabel("Infectious with variant 2")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    f"{folder}/Endemic_equilibrium_multisensitivity_sigma.{fileformat}",
    dpi=dpi,
)
plt.show()


### STEP 10: Plot sensitivity of i_1, i_2 on sigma for multiple beta_2

sigmas = [0.7, 0.8, 0.9]
beta_2s = np.linspace(0.2, 0.6, 9)
plt.figure(figsize=(5, 5))

for s in sigmas:
    upper_threshold_beta2 = (gamma_2 + b) * critical_r02(
        beta_1 / (gamma_1 + b), s
    )

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
            label=(rf"$\beta_2={round(b_2, 2)}$" if s == 0.7 else None),
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
plt.text(0.001, 0.0525, r"$\sigma=0.9$")
plt.text(-0.005, 0.0725, r"$\sigma=0.8$")
plt.text(0.025, 0.0525, r"$\sigma=0.7$")

plt.plot(i1_line, i1_line, color="black", linestyle="dashed", alpha=0.4)
plt.xlabel("Infectious with variant 1")
plt.ylabel("Infectious with variant 2")
plt.legend(loc="upper right")
plt.grid()
plt.tight_layout()
plt.savefig(
    f"{folder}/Endemic_equilibrium_multisensitivity_r02.{fileformat}", dpi=dpi
)
plt.show()
