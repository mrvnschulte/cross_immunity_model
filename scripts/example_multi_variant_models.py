"""
Solves some example multi-variant systems and plots them.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from cross_immunity_model import (
    construct_matrices,
    multi_variant_model,
    multi_variant_model_for_loops,
    multi_variant_model_with_matrices,
    invasion_reproduction_number_multiple_variants,
)

# Solve a two-variant model
t0 = 0
tf = 1000
x0 = np.array([0.998, 0, 0.001, 0, 0.001, 0, 0, 0, 0])
beta = [0.3, 0.35]
gamma = [0.1, 0.1]
sigma = np.array([[1, 0.8], [0.8, 1]])
b = 0.01

res12 = solve_ivp(
    multi_variant_model,
    (t0, tf),
    x0,
    t_eval=np.arange(tf),
    args=(beta, gamma, sigma, b),
)

plt.plot(
    res12.t,
    res12.y.T,
    label=["s", "r1", "i1", "r2", "i2", "r12", "i1,12", "i2,12", "i12"],
)
inv_r0 = [
    invasion_reproduction_number_multiple_variants(x, beta, gamma, b)
    for x in res12.y.T
]
plt.plot(res12.t, inv_r0, label=["inv_r01", "inv_r02"])
plt.xlabel("Time")
plt.ylabel("Compartment size")
plt.legend()
plt.show()

# Solve a 1-variant SIR model using multi-variant model
x0 = np.array([0.999, 0, 0.001])
beta = beta[:1]
gamma = gamma[:1]
sigma = np.array([[1]])

res1 = solve_ivp(
    multi_variant_model_for_loops,
    (t0, tf),
    x0,
    t_eval=np.arange(tf),
    args=(beta, gamma, sigma),
)

plt.plot(res1.t, res1.y.T, label=["s", "r1", "i1"])
plt.xlabel("Time")
plt.ylabel("Compartment size")
plt.legend()
plt.show()


# Solve three-variant model
t0 = 0
tf = 5000
beta_3 = 0.14
sigma_23 = 0.3
beta = [0.3, 0.35, beta_3]
gamma = [0.1, 0.1, 0.1]
b = 0.01
sigma_12 = 0.5
sigma = np.array(
    [
        [1, sigma_12, sigma_12 * sigma_23],
        [sigma_12, 1, sigma_23],
        [sigma_12 * sigma_23, sigma_23, 1],
    ]
)
n_variants = len(beta)
size = 2**n_variants
x0 = 0.001 * np.ones(3**n_variants)
x0[0] = 1 - np.sum(x0[1:])

res12 = solve_ivp(
    multi_variant_model,
    (t0, tf),
    x0,
    t_eval=np.arange(tf),
    args=(beta, gamma, sigma, b),
)

plt.plot(
    res12.t,
    res12.y.T,
)
plt.show()

# Solve two-variant model with matrices
beta = beta[1:]
gamma = gamma[1:]
sigma = np.array([[1, sigma_23], [sigma_23, 1]])
n_variants = len(beta)
size = 2**n_variants
V, G, T, S = construct_matrices(gamma, sigma)
x0 = 0.001 * np.ones(3**n_variants)
x0[0] = 1 - np.sum(x0[1:])

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
)
plt.show()
