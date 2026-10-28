"""
For two variants which can co-exist, this code checks whether a third variant
can invade the more transmissible variant and the combined endemic equilibrium
and plots the invasion ability depending on the cross-immunity and the
transmission rate.
"""

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from cross_immunity_model import (
    basic_reproduction_number,
    construct_matrices,
    find_multi_variant_equilibrium,
    invasion_reproduction_number,
    invasion_reproduction_number_multiple_variants,
)


resolution = 1001
beta_3 = np.linspace(0.1, 0.35, resolution)
sigma_12 = 0.5
sigma_23 = np.linspace(0, 1, resolution)
gamma = np.array([0.1, 0.1, 0.1])
b = 0.01
atol = 0
filename = "../plots/invasion_ability_third_variant.pdf"
dpi = 1200
fontsize_text = 20
fontsize_ticks = 16
plot_single_r0 = False
verbose = True
verbose_step = 10

r0_3to12 = np.zeros((len(beta_3), len(sigma_23)))
r0_3to2 = np.zeros((len(beta_3), len(sigma_23)))
r0_3to1 = np.zeros((len(beta_3), len(sigma_23)))

for i, b3 in enumerate(beta_3):
    for k, s in enumerate(sigma_23):
        beta = np.array([0.3, 0.35, b3])
        r0 = basic_reproduction_number(beta, gamma, b)
        sigma = np.array(
            [
                [1, sigma_12, sigma_12 * s],
                [sigma_12, 1, s],
                [sigma_12 * s, s, 1],
            ]
        )
        V, G, T, S = construct_matrices(gamma, sigma)
        I_variant_guess = [0.2, 0.2, 0.2]

        I_variant, I_eq, x_eq = find_multi_variant_equilibrium(
            beta,
            gamma,
            sigma,
            V,
            G,
            T,
            S,
            b,
            I_variant_guess,
            free_indices=[0, 1],
        )
        r0_3to12[i, k] = invasion_reproduction_number_multiple_variants(
            x_eq, beta, gamma, b, V
        )[2]
        r0_3to1[i, k] = invasion_reproduction_number(r0[0], r0[2], sigma[2, 0])
        r0_3to2[i, k] = invasion_reproduction_number(r0[1], r0[2], sigma[2, 1])

    if plot_single_r0:
        plt.plot(sigma_23, r0_3to12[i], label=r"$R_{0,3\to12}$")
        plt.plot(sigma_23, r0_3to1[i], label=r"$R_{0,3\to1}$")
        plt.plot(sigma_23, r0_3to2[i], label=r"$R_{0,3\to2}$")
        plt.axhline(1, linestyle="dashed", color="gray")
        plt.legend()
        plt.show()
    if verbose:
        if (i + 1) % verbose_step == 0:
            print(f"Computed invasion ability for beta_3={b3} (Number {i+1})")

colors = np.zeros((len(beta_3), len(sigma_23)), dtype=int)

for i in range(len(beta_3)):
    for k in range(len(sigma_23)):
        cond12 = r0_3to12[i, k] > 1 + atol
        cond2 = r0_3to2[i, k] > 1 + atol

        if cond2 and cond12:
            colors[i, k] = 0
        elif cond2 and not cond12:
            colors[i, k] = 1
        elif (not cond2) and cond12:
            colors[i, k] = 2
        else:
            colors[i, k] = 3

cmap = ListedColormap(["white", "blue", "green", "red"])
legend_handles = [
    Patch(
        facecolor="white",
        edgecolor="gray",
        alpha=0.3,
        linewidth=1.0,
        label="Can invade {2} and {1, 2}",
    ),
    Patch(color="blue", label="Can invade {2}, but not {1, 2}"),
    Patch(color="green", label="Can invade {1, 2}, but not {2}"),
    Patch(color="red", label="Can invade neither"),
]

plt.figure()
plt.imshow(
    colors,
    origin="lower",
    aspect="auto",
    extent=[min(sigma_23), max(sigma_23), min(beta_3), max(beta_3)],
    cmap=cmap,
    vmin=0,
    vmax=3,
)
plt.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.52, -0.2),
    ncol=1,
    fontsize=fontsize_ticks,
)
plt.xlabel(r"Cross-immunity $\sigma_{23}$", fontsize=fontsize_text)
plt.ylabel(r"Transmission rate $\beta_3$", fontsize=fontsize_text)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
if filename is not None:
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
plt.show()
