"""
Computes the eigenvalues and eigenvectors of the 2-variant transmission matrix,
the ones for one block of the 3-variant model and the invasion reproduction
number for the 2-variant model.
"""

from sympy import Matrix, symbols


### STEP 1: Compute the eigenvalues of the two-variant model
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

print("### STEP 1: Compute the eigenvalues of the two-variant model\n")
for eigenvalue, multiplicity, vects in eigenvectors:
    print(
        f"Eigenvalue {eigenvalue} has multiplicity {multiplicity} with "
        f"eigenvectors {vects}"
    )


### STEP 2: Compute the eigenvalues of a block of the three-variant model
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

print(
    "\n\n\n### STEP 2: Compute the eigenvalues of a block of the "
    "three-variant model\n"
)
for eigenvalue, multiplicity, vects in eigenvectors:
    print(
        f"Eigenvalue {eigenvalue} has multiplicity {multiplicity} with "
        f"eigenvectors {vects}"
    )


### STEP 3: Compute the eigenvalues of the two-variant model for variant 1
### invading variant 2 equilibrium

# Define symbolic variables
beta_1 = symbols("beta_1")
beta_2 = symbols("beta_2")
gamma_1 = symbols("gamma_1")
gamma_2 = symbols("gamma_2")
sigma = symbols("sigma")
b = symbols("b")

# Define abbreviations
x_4 = (1 - sigma) * b * (1 / (gamma_2 + b) - 1 / beta_2)
x_5 = sigma * b * (1 / (gamma_2 + b) - 1 / beta_2)
x_6 = (gamma_2 + b) / beta_2
x_8 = gamma_2 / b * (1 - sigma) * b * (1 / (gamma_2 + b) - 1 / beta_2)

first_sum = (
    1 / (gamma_1 + beta_2 * (x_4 + x_5) + b)
    + gamma_2
    * beta_2
    * (x_4 + x_5)
    / (
        (gamma_1 + b)
        * (gamma_1 + beta_2 * (x_4 + x_5) + b)
        * (gamma_1 + gamma_2 + b)
    )
    + beta_2
    * (x_4 + x_5)
    / ((gamma_1 + beta_2 * (x_4 + x_5) + b) * (gamma_1 + gamma_2 + b))
)
second_sum = gamma_2 / ((gamma_1 + b) * (gamma_1 + gamma_2 + b)) + 1 / (
    (gamma_1 + gamma_2 + b)
)

# Define next-generation matrix
FV_invasion = Matrix(
    [
        [
            (1 - sigma) * beta_1 * x_6 * first_sum,
            (sigma * beta_1 * x_6 + beta_1 * x_8) * first_sum,
            beta_1 * x_4 * first_sum,
        ],
        [
            (1 - sigma) * beta_1 * x_6 / (gamma_1 + b),
            (sigma * beta_1 * x_6 + beta_1 * x_8) / (gamma_1 + b),
            beta_1 * x_4 / (gamma_1 + b),
        ],
        [
            (1 - sigma) * beta_1 * x_6 * second_sum,
            (sigma * beta_1 * x_6 + beta_1 * x_8) * second_sum,
            beta_1 * x_4 * second_sum,
        ],
    ]
).transpose()
eigvals = FV_invasion.eigenvals(simplify=True)

print(
    "\n\n\n### STEP 3: Compute the eigenvalues of the two-variant model for "
    "variant 1 invading variant 2 equilibrium\n"
)
print(f"The eigenvalues of the next-generation matrix are: {eigvals}")
