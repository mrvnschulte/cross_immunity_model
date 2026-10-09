"""
Solves an example multi-variant system and plots it.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from cross_immunity_model import (
    construct_matrices,
    multi_variant_model_with_matrices,
)

folder = "../plots"
fileformat = "pdf"
dpi = 1200
fontsize_text = 16
fontsize_ticks = 14

# Solve a two-variant model
t0 = 0
tf = 7500
x0 = np.array([0.998, 0, 0.001, 0, 0.001, 0, 0, 0, 0])
beta = [0.3, 0.2]
gamma = [0.1, 0.1]
sigma = np.array([[1, 0.5], [0.5, 1]])
b = 0.001

V, G, T, S = construct_matrices(gamma, sigma)

res12 = solve_ivp(
    multi_variant_model_with_matrices,
    (t0, tf),
    x0,
    t_eval=np.arange(tf),
    args=(beta, gamma, sigma, V, G, T, S, b),
)

plt.plot(
    res12.t,
    res12.y.T,
    label=[
        r"$I_{\emptyset,\emptyset}$",
        r"$I_{\emptyset,1}$",
        r"$I_{1,1}$",
        r"$I_{\emptyset,2}$",
        r"$I_{2,2}$",
        r"$I_{\emptyset,12}$",
        r"$I_{1,12}$",
        r"$I_{2,12}$",
        r"$I_{12,12}$",
    ],
)
plt.xlabel(r"Time $t$", fontsize=fontsize_text)
plt.ylabel("Fraction of the population", fontsize=fontsize_text)
plt.grid(alpha=0.7)
plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.05),
    title="Compartment",
    title_fontproperties={"weight": "bold", "size": fontsize_text},
    fontsize=fontsize_text,
)
plt.xlim(t0, tf)
plt.ylim(0, 1)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.tight_layout()
plt.savefig(f"{folder}/example_two_variant_model.{fileformat}", dpi=dpi)
plt.show()


# Solves the same system on a smaller timescale (initial outbreak)
tf = 200

V, G, T, S = construct_matrices(gamma, sigma)

res12 = solve_ivp(
    multi_variant_model_with_matrices,
    (t0, tf),
    x0,
    t_eval=np.arange(tf),
    args=(beta, gamma, sigma, V, G, T, S, b),
)

i_1 = res12.y[2] + res12.y[6]
i_2 = res12.y[4] + res12.y[7]
i_12 = res12.y[8]

plt.plot(
    res12.t,
    np.array([i_1, i_2, i_12]).T,
    label=["strain 1", "strain 2", "both strains"],
)
plt.xlabel(r"Time $t$", fontsize=fontsize_text)
plt.ylabel("Fraction of the population", fontsize=fontsize_text)
plt.grid(alpha=0.7)
plt.legend(
    loc="upper right",
    title="Infected with",
    title_fontproperties={"weight": "bold", "size": fontsize_text},
    fontsize=fontsize_text,
)
plt.xlim(t0, tf)
plt.ylim(0, None)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.tight_layout()
plt.savefig(
    f"{folder}/example_two_variant_model_initial_outbreak.{fileformat}",
    dpi=dpi,
)
plt.show()
