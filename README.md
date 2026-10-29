[![Tests](https://gitlab.itwm.fraunhofer.de/schulte/cross-immunity-model/badges/master/pipeline.svg?key_text=Tests&key_width=40)](https://gitlab.itwm.fraunhofer.de/schulte/cross-immunity-model/-/pipelines)
[![Coverage](https://gitlab.itwm.fraunhofer.de/schulte/cross-immunity-model/badges/master/coverage.svg)](https://gitlab.itwm.fraunhofer.de/schulte/cross-immunity-model/-/commits/master)


# Cross Immunity Model

A generic multi-variant SIR-like model for infectious diseases with partial cross-immunity.

## Overview

This package implements a mathematical framework for modelling the dynamics of multiple
co-circulating infectious disease variants with partial cross-immunity. It provides:

- A general **multi-variant model** formulated with naive for-loops, dense
  matrix-vector operations, precomputed dense matrices, and sparse operators
- Tools for computing **endemic equilibria** and their **stability**
- Analysis of **invasion reproduction numbers** and invasion conditions
- Utilities for **initial condition creation** from seroprevalence data
- Utilities for transforming cumulative case data into daily cases and incidence
- Sequential parameter-identification workflows for variant-specific COVID-19
  incidence data and strain-specific influenza case estimates
- Symbolic computation tools for analytical results

## Installation

For development:
```bash
git clone git@gitlab.itwm.fraunhofer.de:schulte/cross-immunity-model.git
cd cross-immunity-model
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.9
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0
- pandas >= 1.3.0
- sympy >= 1.9.0

## Model Description

The model tracks the population distributed across **infectious states** and
**protection states** using a matrix formulation. For `n` variants, the state
is represented as a `(2^n × 2^n)` matrix `I`, where entry `I[i, j]` denotes
the fraction of the population:

- currently infectious with the variants in set `i`
- protected against the variants in set `j`

Cross-immunity between variants is encoded in an `(n × n)` matrix `σ`, where
`σ[k, m]` is the cross-immunity factor conferred by previous infection with
variant `m` against variant `k`.

The state can equivalently be represented as a vector `x` of length `3^n`,
obtained by flattening the upper triangular part of `I`.

## Project Structure

```
cross_immunity_model/
├── multi_variant_model.py                    # Core ODEs: loop, dense, and sparse formulations
├── two_variant_model.py                      # Explicit two-variant model
├── matrices.py                               # Dense and sparse model-operator construction
├── model_transformations.py                  # State transformations and projections
├── multi_variant_equilibrium_equations.py    # Equilibrium computation and stability analysis
├── two_variant_equilibrium_equations.py      # Two-variant equilibrium equations
├── invasion_reproduction_number.py           # Invasion analysis
├── create_initial_conditions.py              # Initial conditions from seroprevalence
├── random_parameters.py                      # Random parameter generation
├── symbolic_system.py                        # Symbolic model and Jacobian
├── case_transformations.py                   # Case-count and incidence utilities
└── utils.py                                  # General utility functions
```

## Usage

### Simulating a Two-Variant Model

```python
import numpy as np
from scipy.integrate import solve_ivp
from cross_immunity_model import multi_variant_model

### Parameters

beta = [0.3, 0.35]
gamma = [0.1, 0.1]
sigma = np.array([[1.0, 0.8],
                  [0.8, 1.0]])
b = 0.01

### Initial conditions: [s, r1, i1, r2, i2, r12, i1_12, i2_12, i12]

x0 = np.array([0.998, 0, 0.001, 0, 0.001, 0, 0, 0, 0])

### Solve

sol = solve_ivp(
    multi_variant_model,
    (0, 500),
    x0,
    args=(beta, gamma, sigma, b)
)
```

### Using Precomputed Matrices (Faster)

```python
from cross_immunity_model import construct_matrices, multi_variant_model_with_matrices

V, G, T, S = construct_matrices(gamma, sigma)

sol = solve_ivp(
    multi_variant_model_with_matrices,
    (0, 500),
    x0,
    args=(beta, gamma, sigma, V, G, T, S, b)
)
```

### Using Sparse Operators

For larger numbers of variants, the sparse formulation avoids storing dense
transition matrices.

```python
from scipy.integrate import solve_ivp
from cross_immunity_model import (
    construct_sparse_operators,
    multi_variant_model_sparse,
)

operators = construct_sparse_operators(gamma, sigma)

sol = solve_ivp(
    multi_variant_model_sparse,
    (0, 500),
    x0,
    args=(beta, operators, b),
)
```

The sparse operators contain the recovery rates and cross-immunity transition
matrices. Therefore, `gamma` and `sigma` must remain fixed during integration.


### Computing the Invasion Reproduction Number

```python
from cross_immunity_model import invasion_reproduction_number

r0_1 = 3.5   # Basic reproduction number of variant 1
r0_2 = 3.0   # Basic reproduction number of variant 2
sigma = 0.8  # Cross-immunity

### Can variant 2 invade the variant 1 equilibrium?

r_inv = invasion_reproduction_number(r0_1, r0_2, sigma)
print(f"Invasion R0 = {r_inv:.3f}")  # > 1 means invasion is possible
```

### Finding the Stable Equilibrium

```python
import numpy as np
from cross_immunity_model import find_stable_multi_variant_equilibrium

beta = [0.35, 0.30, 0.15]
gamma = [0.1, 0.1, 0.1]
sigma = np.array([[1.0, 0.5, 0.25],
                  [0.5, 1.0, 0.42],
                  [0.25, 0.42, 1.0]])
b = 0.01

coexisting, I_variant_eq, x_eq, convergence = find_stable_multi_variant_equilibrium(
    beta, gamma, sigma, b
)
print(f"Coexisting variants: {coexisting}")
```

### Creating Initial Conditions from Seroprevalence

```python
from cross_immunity_model import create_initial_conditions_from_seroprevalence

x0 = create_initial_conditions_from_seroprevalence(
    fraction_with_antibodies=0.6,
    protection_against_variant=[0.8, 0.7],
    infectious_with_variant=[0.01, 0.005]
)
```

### Computing Critical Values for Invasion

```python
from cross_immunity_model import critical_sigma, critical_r01, critical_r02

r0_1 = 3.5
r0_2 = 3.0

### Critical cross-immunity below which variant 2 can invade variant 1

sigma_crit = critical_sigma(r0_1, r0_2)
print(f"Critical sigma: {sigma_crit:.3f}")
```

### Transforming Case Data

```python
from cross_immunity_model import cumulative_cases_to_incidence

cumulative_cases = [0, 10, 25, 50, 90, 150]
population = 1_000_000

incidence = cumulative_cases_to_incidence(cumulative_cases, population, window=7)
```

## Example Scripts

| Script | Description |
|--------|-------------|
| `comparison_independent_transmission.py` | Comparison of models with and without co-infection |
| `endemic_equilibrium_sensitivity.py` | Sensitivity analysis of the two-variant endemic equilibrium |
| `example_multi_variant_model.py` | Simulation of a two-variant system |
| `example_stable_equilibrium_without_most_transmissible_variant.py` | Example of a stable three-variant equilibrium excluding the most transmissible variant |
| `final_size_two_variants.py` | Final epidemic-size sweeps over cross-immunity, transmission, and recovery rates |
| `invasion_ability_third_variant.py` | Invasion analysis for a third variant |
| `non_transitivity_example.py` | Demonstration that invasion ability is not transitive |
| `numerical_stability_multiple_variants.py` | Numerical stability checks for multi-variant equilibria |
| `numerical_stability_two_variants.py` | Numerical stability checks for the two-variant model |
| `parameter_identification_covid.py` | Sequential fit to Delta, Omicron BA.1, and Omicron BA.2 incidence data |
| `parameter_identification_influenza.py` | Sequential fit to strain-specific influenza case estimates |
| `plot_invasion_reproduction_number.py` | Visualisation of invasion regions and critical values |
| `proof_two_variant_equilibrium.py` | Symbolic investigation of the number of feasible two-variant equilibria |
| `runtime_model_formulations.py` | Runtime comparison of loop, dense, precomputed, and sparse ODE formulations |
| `runtime_stable_equilibrium.py` | Runtime and accuracy comparison of equilibrium-search methods |
| `symbolic_computation_reproduction_number.py` | Symbolic computation of transmission-matrix eigenvalues and invasion quantities |

## API Reference

### Core Model

| Function | Description |
|----------|-------------|
| `multi_variant_model` | ODE right-hand side with dense matrices constructed during each evaluation |
| `multi_variant_model_for_loops` | ODE right-hand side using naive for-loop formulation |
| `multi_variant_model_with_matrices` | ODE right-hand side using precomputed dense matrices |
| `multi_variant_model_sparse` | ODE right-hand side using precomputed sparse operators |
| `two_variant_model` | Explicit two-variant ODE with demographics |

### Matrix Construction

| Function | Description |
|----------|-------------|
| `construct_matrices` | Construct all dense matrices (V, G, T, S) |
| `construct_sparse_operators` | Construct sparse transition operators for the sparse model formulation |
| `SparseModelOperators` | Container holding sparse transition matrices and associated indices |
| `variant_in_set_matrix` | Variant-in-set matrix V |
| `compute_gamma_matrix` | Recovery matrix G |
| `construct_transition_matrix` | Dense cross-immunity transition matrix T |
| `shift_matrix` | Shift matrix S |

### State Transformations

| Function | Description |
|----------|-------------|
| `map_x_to_I` | Map vector state x to matrix state I |
| `map_I_to_x` | Map matrix state I to vector state x |
| `susceptibles_per_variant` | Compute susceptible fraction per variant |
| `infectious_per_variant` | Compute infectious fraction per variant |
| `project_n_variants` | Project reduced state onto full n-variant state |
| `reduce_variants` | Reduce full state to subset of variants |

### Equilibria

| Function | Description |
|----------|-------------|
| `find_stable_multi_variant_equilibrium` | Find the stable endemic equilibrium |
| `find_multi_variant_equilibrium` | Find equilibrium for a given set of coexisting variants |
| `check_stability_of_variants` | Check stability of a given variant set |
| `multi_variant_equilibrium_compartments` | Compute compartment sizes at equilibrium |
| `eigenvalues_linearized_system` | Eigenvalues of the linearized system at equilibrium |

### Two-Variant Equilibrium

| Function | Description |
|----------|-------------|
| `endemic_equilibrium_equation` | Characteristic equation of the two-variant endemic equilibrium |
| `x_bar` | Compute two-variant endemic equilibrium compartments |
| `susceptibles` | Fully susceptible fraction |
| `protected_1` | Protected only against variant 1 fraction |
| `protected_2` | Protected only against variant 2 fraction |
| `protected_12` | Protected against both variants fraction |

### Invasion Analysis

| Function | Description |
|----------|-------------|
| `invasion_reproduction_number` | Two-variant invasion reproduction number |
| `invasion_reproduction_number_multiple_variants` | Invasion reproduction number at a given equilibrium |
| `critical_sigma` | Critical cross-immunity for variant 2 to invade variant 1 |
| `critical_r01` | Critical R0 of variant 1 for invasion of variant 2 into variant 1 |
| `critical_r02` | Critical R0 of variant 2 for invasion of variant 2 into variant 1 |

### Initial Conditions

| Function | Description |
|----------|-------------|
| `create_initial_conditions_from_seroprevalence` | Full initial state from seroprevalence data |
| `create_protection_state_from_seroprevalence` | Protection state from seroprevalence data |
| `distribute_infectious_fraction` | Distribute infectious individuals across protection states |

### Random Parameters

| Function | Description |
|----------|-------------|
| `sample_symmetric_beta_matrix` | Sample a symmetric cross-immunity matrix |
| `sample_random_parameters` | Sample random model parameters |
| `random_simplex_points` | Generate random points on the standard simplex |

### Symbolic System

| Function | Description |
|----------|-------------|
| `build_symbolic_system` | Build the multi-variant model symbolically |
| `build_jacobian_function` | Build and lambdify the Jacobian symbolically |

### Case Transformations

| Function | Description |
|----------|-------------|
| `cumulative_to_daily_cases` | Convert cumulative to daily case counts |
| `relative_to_absolute_cases` | Convert relative to absolute case counts |
| `moving_average` | Compute moving average of case counts |
| `cases_to_incidence` | Convert case counts to incidence per 100,000 |
| `cumulative_cases_to_incidence` | Convert cumulative cases to incidence |

### Utilities

| Function | Description |
|----------|-------------|
| `basic_reproduction_number` | Compute basic reproduction numbers |
| `check_validity` | Check biological feasibility of a state |
| `hamming_weight_iterator` | Iterate over integers ordered by Hamming weight |
| `is_subset` | Check subset relation via binary representation |
| `number_of_variants` | Determine number of variants from state size |
| `subsets_of` | Return all subsets of a given index set |
| `variant_in_set` | Check whether variant k is in set i |

## Documentation

The API documentation is generated automatically from the NumPy-style
docstrings using Sphinx. It includes the public package modules.

From the project root, generate the API pages and build the HTML documentation:

```powershell
sphinx-apidoc --force --separate --module-first `
    -o docs/source/api `
    cross_immunity_model

python -m sphinx -b html docs/source docs/build/html
```

Open the generated documentation with:

```powershell
Start-Process docs/build/html/index.html
```

## Citation

If you use this software in your research, please cite:
[Citation will be added upon publication]
For a mathematical analysis of the underlying model, see:
[Citation will be added upon publication]

## License

tba

## Author

Marvin Schulte - [marvin.schulte@itwm.fraunhofer.de](mailto:marvin.schulte@itwm.fraunhofer.de)

Division Processes and Materials, Department Flow Processes, Fraunhofer Institute for Industrial Mathematics ITWM
Department of Mathematics, RPTU University of Kaiserslautern-Landau
