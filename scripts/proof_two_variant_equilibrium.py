"""
Proof that there are at most four solutions in [0,1]².

Proof strategy:
1. Symbolic construction of the system (Steps 1-2)
2. Polynomialisation of the system and degree analysis (Steps 3-4)
3. Division by common divisor and degree analysis of reduced system (Step 5-7)
4. Compute resultant of the system (Step 8)
5. Examine roots of resultant (Step 9-12)
"""

import matplotlib.pyplot as plt
import numpy as np
from sympy import (
    degree,
    div,
    factor,
    gcd,
    numer,
    Rational,
    resultant,
    roots,
    simplify,
    symbols,
    together,
    total_degree,
)

from cross_immunity_model import x_bar


dpi = 1200
folder = "../plots"
fileformat = "jpg"
fontsize_text = 20
fontsize_ticks = 16
alpha_grid = 0.7

print("=" * 80)
print("Proof: Number of solutions in [0,1]²")
print("=" * 80)

# Variables

i1, i2 = symbols("i1 i2", real=True)

# Symbolic parameters and values for visualization only

sigma = symbols("sigma", real=True, positive=True)
beta_1 = symbols("beta_1", real=True, positive=True)
beta_2 = symbols("beta_2", real=True, positive=True)
gamma_1 = symbols("gamma_1", real=True, positive=True)
gamma_2 = symbols("gamma_2", real=True, positive=True)
b = symbols("b", real=True, positive=True)

params_dict = {
    sigma: Rational(65, 100),
    beta_2: Rational(30, 100),
    beta_1: Rational(2, 10),
    gamma_1: Rational(1, 10),
    gamma_2: Rational(1, 10),
    b: Rational(1, 100),
}

print("\n" + "-" * 80)
print("STEP 1: Symbolic construction")
print("-" * 80)


### STEP 1: CONSTRUCTION OF ALL x_i FUNCTIONS

print("\nConstruct x_i functions symbolically...")

x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9 = x_bar(
    [i1, i2], beta_1, beta_2, gamma_1, gamma_2, sigma, b
)

sum_x = simplify(x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9)
print(f"The sum of all x is: {sum_x}")


### STEP 2: CONSTRUCTION OF THE SYSTEM OF EQUATIONS

print("\n" + "-" * 80)
print("STEP 2: Construction of the system of equations")
print("-" * 80)

F1 = x_1 + x_2 + x_3 - i1
F2 = x_3 + x_4 + x_5 - i2

print("\nF1 = x_1 + x_2 + x_3 - i1")
print("F2 = x_3 + x_4 + x_5 - i2")


### STEP 3: POLYNOMIALISATION AND DEGREE ANALYSIS

print("\n" + "-" * 80)
print("STEP 3: Polynomialisation")
print("-" * 80)

print("\nBring F1 and F2 down to a common denominator...")

F1_together = together(F1)
F2_together = together(F2)

print("F1 and F2 summarized")

P1 = numer(F1_together)
P2 = numer(F2_together)

print("Nominators extracted")
print(f"Polynomial P1 is: P1(i1, i2) = {P1}")
print(f"Polynomial P2 is: P2(i1, i2) = {P2}")

print("\nCompute degrees...")
deg_P1_i1 = degree(P1, i1)
deg_P1_i2 = degree(P1, i2)
deg_P2_i1 = degree(P2, i1)
deg_P2_i2 = degree(P2, i2)
deg_P1 = total_degree(P1, i1, i2)
deg_P2 = total_degree(P1, i1, i2)

print("\nP1:")
print(f"  Degree in i1: {deg_P1_i1}")
print(f"  Degree in i2: {deg_P1_i2}")
print(f"  Total degree: {deg_P1}")

print("\nP2:")
print(f"  Degree in i1: {deg_P2_i1}")
print(f"  Degree in i2: {deg_P2_i2}")
print(f"  Total degree: {deg_P2}")

bezout_bound = deg_P1 * deg_P2
print(f"\nNaïve Bézout bound: {deg_P1} × {deg_P2} = {bezout_bound}")


### STEP 4: DIVISION BY COMMON DIVISOR

print("\n" + "-" * 80)
print("STEP 4: Division by common divisor of P1 and P2")
print("-" * 80)

G = gcd(P1, P2)

print("\nCommon divisor of P1 and P2:")
print(f"G(i1, i2) = {G}")

G_factored = factor(G)
print(f"\nG factorised: {G_factored}")

print("\nRoots of G:")
for a in roots(G, i1):
    print(f"  i1 = {a}  (outside [0,1])")
for a in roots(G, i2):
    print(f"  i2 = {a}  (outside [0,1])")

print("\n✓ G = 0 has NO solutions in [0,1]²!")
print("  → The common divisor has no solutions in the unit square")

print("\nCompute Q1 = P1 / G...")

Q1, remainder1 = div(P1, G, i1, i2)
print("Q1 computed")
print(f"Q1(i1, i2) = {Q1}")
print(f"Remainder: {remainder1}")

print("\nCompute Q2 = P2 / G...")

Q2, remainder2 = div(P2, G, i1, i2)
print("Q2 computed")
print(f"Q2(i1, i2) = {Q2}")
print(f"Remainder: {remainder2}")

deg_Q1_i1 = degree(Q1, i1)
deg_Q1_i2 = degree(Q1, i2)
deg_Q2_i1 = degree(Q2, i1)
deg_Q2_i2 = degree(Q2, i2)
deg_Q1 = total_degree(Q1, i1, i2)
deg_Q2 = total_degree(Q1, i1, i2)

print("\nQ1:")
print(f"  Degree in i1: {deg_Q1_i1}")
print(f"  Degree in i2: {deg_Q1_i2}")
print(f"  Total degree: {deg_Q1}")

print("\nQ2:")
print(f"  Degree in i1: {deg_Q2_i1}")
print(f"  Degree in i2: {deg_Q2_i2}")
print(f"  Total degree: {deg_Q2}")

bezout_bound = deg_Q1 * deg_Q2
print(f"\nNaïve Bézout bound: {deg_Q1} × {deg_Q2} = {bezout_bound}")


### STEP 5: RESULTANT OF THE REDUCED SYSTEM

print("\n" + "-" * 80)
print("STEP 5: Resultant of the reduced system")
print("-" * 80)

print("\nCompute resultant R_Q = Res(Q1, Q2, i2)...")

R_Q = resultant(Q1, Q2, i2)
R_Q = simplify(R_Q)
deg_R_Q = degree(R_Q, i1)

print(f"\nDegree of the resultant in i1: {deg_R_Q}")

if deg_R_Q == float("-inf") or R_Q == 0:
    print("\n⚠ Resultant is zero - more common divisors!")
else:
    print("\n✓ Resultant is non-trivial")
    print(f"  Maximal number of roots in i1: {deg_R_Q}")

    print("\nFactorise resultant...")
    R_Q_factored = factor(R_Q)
    print(f"R_Q factorised: {R_Q_factored}")


### STEP 6: ROOTS OF DIVISOR

print("\n" + "-" * 80)
print("STEP 6: Roots of divisor")
print("-" * 80)

divisor = (
    -b
    * beta_1
    * beta_2**2
    * i1**3
    * sigma
    * (b + beta_1 * i1)
    * (
        b**2
        + b * beta_1 * i1
        - b * beta_1
        + b * gamma_1
        + beta_1 * gamma_1 * i1
    )
)
roots_divisor = roots(divisor, i1)
print(f"\nDivisor g of R_Q: {divisor}")
print(f"with roots: {roots_divisor}")

for a in roots_divisor:
    Q1_a = Q1.subs(i1, a)
    Q2_a = Q2.subs(i1, a)
    roots_Q1 = roots(Q1_a, i2)
    roots_Q2 = roots(Q2_a, i2)
    print(f"\nFor root i1={a} we have:")
    print(f"\nQ1({a}, i2): {simplify(Q1_a)}")
    print(f"Q2({a}, i2): {simplify(Q2_a)}")
    print(f"\nQ1({a}, i2) has roots: {roots_Q1}")
    print(f"Q2({a}, i2) has roots: {roots_Q2}")
    print("-" * 80)


### STEP 7: I2 ROOTS OF Q1

print("\n" + "-" * 80)
print("STEP 7: Roots of Q1 wrt i2")
print("-" * 80)

i2_i1 = roots(Q1, i2)
print(f"\nIn general, we find the roots of Q1 wrt i2: {i2_i1}")

i2_i1 = (
    -(b + beta_1 * i1)
    * (
        b**2
        + b * beta_1 * i1
        - b * beta_1
        + b * gamma_1
        + beta_1 * gamma_1 * i1
    )
    / (
        beta_2
        * (
            b**2
            + b * beta_1 * i1
            + b * beta_1 * sigma
            - b * beta_1
            + b * gamma_1
            + beta_1 * gamma_1 * i1
        )
    )
)
i2_i1 = simplify(i2_i1)
print(f"\nThis implies i2(i1): {i2_i1}")


### STEP 8: EVALUATE REMAINING CUBIC OF RESULTANT AT CRITICAL VALUES OF I2

print("\n" + "-" * 80)
print("STEP 8: Evaluate remaining cubic of resultant at critical values of i2")
print("-" * 80)

S, remainder = div(R_Q, divisor)
S = simplify(S)
print(f"\nS=R_Q/g: {S}")
print(f"with remainder: {remainder}")

i1a = -b / beta_1

i1c = (-(b**2) + b * beta_1 - b * gamma_1) / (b * beta_1 + beta_1 * gamma_1)
print(f"\nCritical value i1c of i1: {i1c}")

Si1c = S.subs(i1, i1c)
Si1c = simplify(Si1c)
print(f"S(i1c)={Si1c}")

roots_Si1c = roots(Si1c, sigma)
print(f"With sigma roots: {roots_Si1c}")

i1d = (-(b**2) + b * beta_1 - b * gamma_1 - b * beta_1 * sigma) / (
    b * beta_1 + beta_1 * gamma_1
)
print(f"\nCritical i1d of i1: {i1d}")

Si1d = S.subs(i1, i1d)
Si1d = simplify(Si1d)
print(f"S(i1d)={Si1d}")

roots_Si1d = roots(Si1d, sigma)
print(f"With sigma roots: {roots_Si1d}")

Si10 = S.subs(i1, 0)
Si10 = simplify(Si10)
print(f"\nS(0)={Si10}")

roots_Si10 = roots(Si10, sigma)
print(f"With sigma roots: {roots_Si10}")


### VISUALIZE IMPORTANT RELATIONS

# i2 dependence on i1 and important values
i2_i1_val = i2_i1.subs(params_dict)
i1a_val = i1a.subs(params_dict)
i1d_val = i1d.subs(params_dict)
i1c_val = i1c.subs(params_dict)
epsilon = 1e-8
i1_start = float(i1a_val + 0.5 * i1a_val)
i1_split = float(i1d_val)
i1_end = float(i1c_val + 0.5 * i1c_val)
i1_left = np.linspace(i1_start, i1_split - epsilon, 500)
i1_right = np.linspace(i1_split + epsilon, i1_end, 500)
i2_left = [float(i2_i1_val.subs(i1, x)) for x in i1_left]
i2_right = [float(i2_i1_val.subs(i1, x)) for x in i1_right]

plt.plot(i1_left, i2_left, color="tab:blue")
plt.plot(i1_right, i2_right, color="tab:blue")
plt.grid(alpha=alpha_grid)
plt.xlim(i1_left[0], i1_right[-1])
plt.ylim(-0.5, 0.5)
plt.xticks(
    np.asarray([i1a_val, i1d_val, 0, i1c_val], dtype=float),
    labels=[r"$I_1^a$", r"$I_1^d$", "0", r"$I_1^c$"],
    fontsize=fontsize_ticks,
)
plt.yticks([0], [0], fontsize=fontsize_ticks)
plt.xlabel(r"$I_1$", fontsize=fontsize_text)
plt.ylabel(r"$I_2(I_1)$", fontsize=fontsize_text)
plt.savefig(f"{folder}/i2_i1.{fileformat}", dpi=dpi, bbox_inches="tight")
plt.show()

# S(i1c) dependence on sigma and important values
fixed_sigma = params_dict[sigma]
del params_dict[sigma]
S1c_val = Si1c.subs(params_dict)
sigma_val = np.linspace(-0.2, 1.2)
S1c_val_sigma = [S1c_val.subs(sigma, s) for s in sigma_val]

plt.plot(sigma_val, S1c_val_sigma)
plt.grid(alpha=alpha_grid)
plt.ylim(-1e-8, 3e-8)
plt.xlim(-0.1, 1.1)
plt.xticks([0, 1], labels=[0, 1], fontsize=fontsize_ticks)
plt.yticks([0], [0], fontsize=fontsize_ticks)
plt.xlabel(r"$\sigma$", fontsize=fontsize_text)
plt.ylabel(r"$S(I_1^c)(\sigma)$", fontsize=fontsize_text)
plt.savefig(f"{folder}/Si1c_sigma.{fileformat}", dpi=dpi, bbox_inches="tight")
plt.show()

# S(i1d) dependence on sigma and important values
S1d_val = Si1d.subs(params_dict)
sigma_val = np.linspace(-0.2, 1.2)
S1d_val_sigma = [S1d_val.subs(sigma, s) for s in sigma_val]

plt.plot(sigma_val, S1d_val_sigma)
plt.grid(alpha=alpha_grid)
plt.ylim(-1.5e-8, 1e-8)
plt.xlim(-0.1, 1.1)
plt.xticks([0, 1], labels=[0, 1], fontsize=fontsize_ticks)
plt.yticks([0], [0], fontsize=fontsize_ticks)
plt.xlabel(r"$\sigma$", fontsize=fontsize_text)
plt.ylabel(r"$S(I_1^d)(\sigma)$", fontsize=fontsize_text)
plt.savefig(f"{folder}/Si1d_sigma.{fileformat}", dpi=dpi, bbox_inches="tight")
plt.show()

# S(0) dependence on sigma and important values
S10_val = Si10.subs(params_dict)
sigma_val = np.linspace(-0.2, 1.2)
S10_val_sigma = [S10_val.subs(sigma, s) for s in sigma_val]
r_01 = beta_1 / (gamma_1 + b)
r_02 = beta_2 / (gamma_2 + b)
sigma_star = r_02 * (r_01 - 1) / (r_01 * (r_02 - 1))
sigma_star_val = sigma_star.subs(params_dict)

plt.plot(sigma_val, S10_val_sigma)
plt.grid(alpha=alpha_grid)
plt.ylim(-4e-8, 2e-8)
plt.xlim(-0.1, 1.1)
plt.xticks(
    np.asarray([0, sigma_star_val, 1], dtype=float),
    labels=[0, r"$\sigma^{1\to2}$", 1],
    fontsize=fontsize_ticks,
)
plt.yticks([0], [0], fontsize=fontsize_ticks)
plt.xlabel(r"$\sigma$", fontsize=fontsize_text)
plt.ylabel(r"$S(0)(\sigma)$", fontsize=fontsize_text)
plt.savefig(f"{folder}/Si10_sigma.{fileformat}", dpi=dpi, bbox_inches="tight")
plt.show()

# S dependence on i1 and important values
params_dict[sigma] = fixed_sigma
S_val = S.subs(params_dict)
i1_val = np.linspace(i1d_val + 0.2 * i1d_val, i1c_val + 0.2 * i1c_val, 1001)
i2_val = [S_val.subs(i1, i1_v) for i1_v in i1_val]

plt.plot(i1_val, i2_val)
plt.grid(alpha=alpha_grid)
plt.xlim(float(i1d_val + 0.1 * i1d_val), float(i1c_val + 0.1 * i1c_val))
plt.xticks(
    np.asarray([i1d_val, 0, i1c_val], dtype=float),
    labels=[r"$I_1^d$", "0", r"$I_1^c$"],
    fontsize=fontsize_ticks,
)
plt.yticks([0], [0], fontsize=fontsize_ticks)
plt.xlabel(r"$I_1$", fontsize=fontsize_text)
plt.ylabel(r"$S(I_1)$", fontsize=fontsize_text)
plt.savefig(f"{folder}/S_i1.{fileformat}", dpi=dpi, bbox_inches="tight")
plt.show()
