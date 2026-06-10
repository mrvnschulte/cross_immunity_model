"""
Computes the linearized system of the two-variant model symbolically.
"""

import sympy as sp

from cross_immunity_model import two_variant_model


x1, x2, x3, x4, x5, x6, x7, x8, x9 = sp.symbols("x1 x2 x3 x4 x5 x6 x7 x8 x9")
gamma1, gamma2, beta1, beta2, sigma, b = sp.symbols(
    "gamma1 gamma2 beta1 beta2 sigma b", positive=True
)

xs = [x1, x2, x3, x4, x5, x6, x7, x8, x9]
F = two_variant_model(0, xs, beta1, beta2, gamma1, gamma2, sigma, b)

J = sp.Matrix(F).jacobian(xs)

print(J)
sp.print_latex(J)
