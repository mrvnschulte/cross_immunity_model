"""
Provides a multi-variant epidemiological model with cross-immunity.

The package contains functions for constructing and simulating multi-variant
cross-immunity models, computing equilibria and invasion reproduction numbers,
generating initial conditions, and transforming model states and case data.
"""

from .case_transformations import (
    cases_to_incidence,
    cumulative_cases_to_incidence,
    cumulative_to_daily_cases,
    moving_average,
    relative_to_absolute_cases,
)
from .create_initial_conditions import (
    create_initial_conditions_from_seroprevalence,
)
from .invasion_reproduction_number import (
    critical_r01,
    critical_r02,
    critical_sigma,
    invasion_reproduction_number,
    invasion_reproduction_number_multiple_variants,
)
from .matrices import (
    compute_gamma_matrix,
    construct_matrices,
    construct_sparse_operators,
    construct_transition_matrix,
    shift_matrix,
    variant_in_set_matrix,
)
from .model_transformations import (
    infectious_per_variant,
    map_I_to_x,
    map_x_to_I,
    susceptibles_per_variant,
)
from .multi_variant_equilibrium_equations import (
    eigenvalues_linearized_system,
    find_multi_variant_equilibrium,
    find_stable_multi_variant_equilibrium,
    multi_variant_equilibrium_compartments,
    multi_variant_equilibrium_equation,
    solve_multi_variant_equilibrium_equation,
)
from .multi_variant_model import (
    multi_variant_model,
    multi_variant_model_for_loops,
    multi_variant_model_sparse,
    multi_variant_model_with_matrices,
)
from .random_parameters import sample_random_parameters, random_simplex_points
from .symbolic_system import build_symbolic_system
from .two_variant_equilibrium_equations import (
    endemic_equilibrium_equation,
    x_bar,
)
from .two_variant_model import (
    protected_1,
    protected_12,
    protected_2,
    susceptibles,
    two_variant_model,
)
from .utils import (
    basic_reproduction_number,
    check_validity,
    is_subset,
    number_of_variants,
    variant_in_set,
)

__all__ = [
    "basic_reproduction_number",
    "build_symbolic_system",
    "cases_to_incidence",
    "check_validity",
    "compute_gamma_matrix",
    "construct_matrices",
    "construct_sparse_operators",
    "construct_transition_matrix",
    "create_initial_conditions_from_seroprevalence",
    "critical_r01",
    "critical_r02",
    "critical_sigma",
    "cumulative_cases_to_incidence",
    "cumulative_to_daily_cases",
    "eigenvalues_linearized_system",
    "endemic_equilibrium_equation",
    "find_multi_variant_equilibrium",
    "find_stable_multi_variant_equilibrium",
    "infectious_per_variant",
    "invasion_reproduction_number",
    "invasion_reproduction_number_multiple_variants",
    "is_subset",
    "map_I_to_x",
    "map_x_to_I",
    "moving_average",
    "multi_variant_equilibrium_compartments",
    "multi_variant_equilibrium_equation",
    "multi_variant_model",
    "multi_variant_model_for_loops",
    "multi_variant_model_sparse",
    "multi_variant_model_with_matrices",
    "number_of_variants",
    "protected_1",
    "protected_12",
    "protected_2",
    "random_simplex_points",
    "relative_to_absolute_cases",
    "sample_random_parameters",
    "shift_matrix",
    "solve_multi_variant_equilibrium_equation",
    "susceptibles",
    "susceptibles_per_variant",
    "two_variant_model",
    "variant_in_set",
    "variant_in_set_matrix",
    "x_bar",
]
