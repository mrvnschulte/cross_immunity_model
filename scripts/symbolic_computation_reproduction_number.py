"""
Computes the eigenvalues and eigenvectors of the 2-variant transmission matrix,
the ones for one block of the 3-variant model and the invasion reproduction
number for the 2-variant model.
"""

from sympy import Matrix, symbols


### STEP 1: Compute the eigenvalues of the two-variant model
print("### STEP 1: Compute the eigenvalues of the two-variant model\n")

sigma = symbols("sigma")
r01 = symbols("r01")
r02 = symbols("r02")

FV = Matrix(
    [
        [(1 - sigma) * r01, 0, (1 - sigma) * r01, 0],
        [0, (1 - sigma) * r02, 0, (1 - sigma) * r02],
        [sigma * r01, 0, sigma * r01, 0],
        [0, sigma * r02, 0, sigma * r02],
    ]
)
eigenvectors = FV.eigenvects()

for eigenvalue, multiplicity, vects in eigenvectors:
    print(
        f"Eigenvalue {eigenvalue} has multiplicity {multiplicity} with "
        f"eigenvectors {vects}"
    )


### STEP 2: Compute the eigenvalues of a block of the three-variant model
print(
    "\n\n\n### STEP 2: Compute the eigenvalues of a block of the "
    "three-variant model\n"
)

sigma_12 = symbols("sigma_12")
sigma_13 = symbols("sigma_13")

FV = Matrix(
    [
        [
            (1 - sigma_12) * (1 - sigma_13) * r01,
            (1 - sigma_12) * (1 - sigma_13) * r01,
            (1 - sigma_12) * (1 - sigma_13) * r01,
            (1 - sigma_12) * (1 - sigma_13) * r01,
        ],
        [
            sigma_12 * (1 - sigma_13) * r01,
            sigma_12 * (1 - sigma_13) * r01,
            sigma_12 * (1 - sigma_13) * r01,
            sigma_12 * (1 - sigma_13) * r01,
        ],
        [
            sigma_13 * (1 - sigma_12) * r01,
            sigma_13 * (1 - sigma_12) * r01,
            sigma_13 * (1 - sigma_12) * r01,
            sigma_13 * (1 - sigma_12) * r01,
        ],
        [
            sigma_13 * sigma_12 * r01,
            sigma_13 * sigma_12 * r01,
            sigma_13 * sigma_12 * r01,
            sigma_13 * sigma_12 * r01,
        ],
    ]
)
eigenvectors = FV.eigenvects()

for eigenvalue, multiplicity, vects in eigenvectors:
    print(
        f"Eigenvalue {eigenvalue} has multiplicity {multiplicity} with "
        f"eigenvectors {vects}"
    )


### STEP 3: Compute the eigenvalues of the two-variant model for variant 1
### invading variant 2 equilibrium
print(
    "\n\n\n### STEP 3: Compute the eigenvalues of the two-variant model for "
    "variant 1 invading variant 2 equilibrium\n"
)

# Define symbolic variables
beta_1 = symbols("beta_1")
beta_2 = symbols("beta_2")
gamma_1 = symbols("gamma_1")
gamma_2 = symbols("gamma_2")
sigma = symbols("sigma")
b = symbols("b")

# Define abbreviations
R_01 = beta_1 / (gamma_1 + b)
I_11 = (1 - sigma) * b / beta_1 * (R_01 - 1)
I_112 = sigma * b / beta_1 * (R_01 - 1)
I_00 = 1 / R_01
I_01 = (1 - sigma) * gamma_1 / beta_1 * (R_01 - 1)
I_1 = I_11 + I_112

p = (
    1 / (beta_1 * I_1 + gamma_2 + b)
    + (gamma_1 * beta_1 * I_1)
    / ((beta_1 * I_1 + gamma_2 + b) * (gamma_2 + b) * (gamma_1 + gamma_2 + b))
    + (beta_1 * I_1) / ((beta_1 * I_1 + gamma_2 + b) * (gamma_1 + gamma_2 + b))
)
q = gamma_1 / ((gamma_2 + b) * (gamma_1 + gamma_2 + b)) + (
    1 / (gamma_1 + gamma_2 + b)
)

# Define next-generation matrix
FV_invasion = Matrix(
    [
        [
            (1 - sigma) * beta_2 * I_00 * p,
            (1 - sigma) * beta_2 * I_00 / (gamma_2 + b),
            (1 - sigma) * beta_2 * I_00 * q,
        ],
        [
            (sigma * beta_2 * I_00 + beta_2 * I_01) * p,
            (sigma * beta_2 * I_00 + beta_2 * I_01) / (gamma_2 + b),
            (sigma * beta_2 * I_00 + beta_2 * I_01) * q,
        ],
        [
            beta_2 * I_11 * p,
            beta_2 * I_11 / (gamma_2 + b),
            beta_2 * I_11 * q,
        ],
    ]
)
eigvals = FV_invasion.eigenvals(simplify=True)

print(f"The eigenvalues of the next-generation matrix are: {eigvals}")
