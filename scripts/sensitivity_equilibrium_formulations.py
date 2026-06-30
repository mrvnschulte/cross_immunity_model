"""
This script examines the sensitivity of the equilibrium equation for the two
possible formulations, either the direct one or the one with the invasion
reproduction number. For illustration purposes, a one-variant model is
considered and the difference to 0 in both formulations as well as its gradient
is shown.
"""

import matplotlib.pyplot as plt
import numpy as np

from cross_immunity_model import (
    construct_matrices,
    multi_variant_equilibrium_equation,
)

# Parameters
beta = [0.3]
gamma = [0.1]
sigma = [[1]]
b = 0.01
dpi = 500

# Values to compute
S_eq = (gamma[0] + b) / beta[0]
I_variant_eq = (b - b * S_eq) / (gamma[0] + b)
V, G, T, S = construct_matrices(gamma, sigma)

# Difference of both formulations
I_variant_all = np.linspace(0, 0.1, 5001)
diff_inv_formulation = [
    multi_variant_equilibrium_equation(
        np.array([I_variant]),
        beta,
        gamma,
        sigma,
        V,
        G,
        T,
        S,
        b,
        use_inv_formulation=True,
    )
    for I_variant in I_variant_all
]
diff_direct_formulation = [
    multi_variant_equilibrium_equation(
        np.array([I_variant]),
        beta,
        gamma,
        sigma,
        V,
        G,
        T,
        S,
        b,
        use_inv_formulation=False,
    )
    for I_variant in I_variant_all
]

# Gradient of both formulations
diff_inv = np.array(diff_inv_formulation).flatten()
diff_direct = np.array(diff_direct_formulation).flatten()

grad_inv = np.gradient(diff_inv, I_variant_all)
grad_direct = np.gradient(diff_direct, I_variant_all)

idx_eq = int(np.argmin(np.abs(I_variant_all - I_variant_eq)))

grad_inv_at_eq = grad_inv[idx_eq]
grad_direct_at_eq = grad_direct[idx_eq]

print(
    f"I_variant_eq = {I_variant_eq:.6f} (Grid point: {I_variant_all[idx_eq]:.6f})"
)
print(f"Gradient     inv-formulation at I_eq: {grad_inv_at_eq:.6f}")
print(f"Gradient non-inv-formulation at I_eq: {grad_direct_at_eq:.6f}")

# Plot difference
plt.figure()
plt.plot(I_variant_all, diff_inv, label="Inv. R0")
plt.plot(I_variant_all, diff_direct, label="Direct")
plt.axhline(0, linestyle="--", color="gray")
plt.axvline(I_variant_eq, linestyle=":", color="black")
plt.xlabel("I")
plt.ylabel("Difference")
plt.legend(title="Formulation")
plt.grid(True)
plt.tight_layout()
plt.savefig("../plots/sensitivity_equilibrium_formulations.jpg", dpi=dpi)
plt.show()

# PLot gradient
plt.figure()
plt.plot(I_variant_all, grad_inv, label="Inv. R0")
plt.plot(I_variant_all, grad_direct, label="Direct")
plt.axhline(0, linestyle="--", color="gray")
plt.axvline(I_variant_eq, linestyle=":", color="black")
plt.xlabel("I")
plt.ylabel("Gradient of the difference")
plt.legend(title="Formulation")
plt.grid(True)
plt.tight_layout()
plt.savefig("../plots/gradient_equilibrium_formulations.jpg", dpi=dpi)
plt.show()
