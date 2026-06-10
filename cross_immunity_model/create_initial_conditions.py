"""
Provides utility functions to create initial conditions for multi-variant model
"""

import numpy as np
from numpy.typing import ArrayLike

from .model_transformations import map_I_to_x, susceptibles_per_variant
from .utils import variant_in_set


def create_protection_state_from_seroprevalence(
    fraction_with_antibodies: float,
    protection_against_variant: ArrayLike,
    matrix_form: bool = True,
) -> np.ndarray:
    """
    Creates a protection state matrix from seroprevalence data.

    Constructs a (2^n x 2^n) matrix I where entry I[i, j] represents the
    fraction of the population in infectious state i and protection state j.
    Individuals without antibodies are placed in state I[0, 0]. Individuals
    with antibodies are distributed across protection states j according to
    their probability of being protected against each variant. All are set to
    the infectious state of no infectiousness.

    Parameters
    ----------
    fraction_with_antibodies : float
        Fraction of the population with antibodies, i.e. seroprevalence.
        Must be in [0, 1].
    protection_against_variant : array-like of float
        Protection probability against each variant for individuals with
        antibodies. Each entry must be in [0, 1]. The length n determines
        the number of variants.
    matrix_form : bool, optional
        If True, returns the (2^n x 2^n) matrix I directly.
        If False, returns the flattened state vector x via map_I_to_x.
        The default is True.

    Returns
    -------
    x : np.ndarray
        Initial protection state, either as (2^n x 2^n) matrix (if
        matrix_form=True) or as flattened state vector (if matrix_form=False).
    """
    n_variants = len(protection_against_variant)
    I = np.zeros((2**n_variants, 2**n_variants))

    I[0, 0] = 1 - fraction_with_antibodies
    for j in range(2**n_variants):
        variants_of_set = [
            variant_in_set(k, j) for k in range(1, n_variants + 1)
        ]
        factors = np.where(
            variants_of_set,
            protection_against_variant,
            1 - np.asarray(protection_against_variant),
        )
        I[0, j] += fraction_with_antibodies * np.prod(factors)

    if matrix_form:
        x = I
    else:
        x = map_I_to_x(I)

    return x


def distribute_infectious_fraction(
    I: np.ndarray, infectious_with_variant: ArrayLike
) -> np.ndarray:
    """
    Distributes infectious individuals across protection states.

    For each variant k, moves a fraction of susceptible individuals into
    the corresponding infectious state. The infectious fraction is computed
    relative to the population that is not protected against variant k, and
    is applied proportionally across all protection states.

    Parameters
    ----------
    I : np.ndarray
        Protection state matrix of shape (2^n x 2^n), as returned by
        create_protection_state_from_seroprevalence with matrix_form=True.
        Modified in-place and returned.
    infectious_with_variant : array-like of float
        Fraction of the total population currently infectious with each
        variant. Each entry must be in [0, 1].

    Returns
    -------
    I : np.ndarray
        Updated protection state matrix of shape (2^n x 2^n) with
        infectious individuals distributed into their respective states.
    """
    protected_against_variant = 1 - susceptibles_per_variant(
        I, matrix_form=True
    )
    n_variants = len(infectious_with_variant)
    for k in range(n_variants):
        protection_sets = [
            variant_in_set(k + 1, i) for i in range(2**n_variants)
        ]
        infectious_fraction = (
            infectious_with_variant[k] / protected_against_variant[k]
        )
        I[2**k : 2 ** (k + 1), protection_sets] = (
            infectious_fraction * I[: 2**k, protection_sets]
        )
        I[: 2**k, protection_sets] = (1 - infectious_fraction) * I[
            : 2**k, protection_sets
        ]

    return I


def create_initial_conditions_from_seroprevalence(
    fraction_with_antibodies: float,
    protection_against_variant: ArrayLike,
    infectious_with_variant: ArrayLike,
) -> np.ndarray:
    """
    Creates initial conditions for the multi-variant model from seroprevalence
    data.

    Combines create_protection_state_from_seroprevalence and
    distribute_infectious_fraction to build a complete initial state vector.
    First distributes the population across protection states based on
    seroprevalence, then moves infectious individuals into the appropriate
    infectious states, and finally maps the result to the flattened state
    vector x.

    Parameters
    ----------
    fraction_with_antibodies : float
        Fraction of the population with antibodies, i.e. seroprevalence.
        Must be in [0, 1].
    protection_against_variant : array-like of float
        Protection probability against each variant for individuals with
        antibodies. Each entry must be in [0, 1]. The length n determines
        the number of variants and must match infectious_with_variant.
    infectious_with_variant : array-like of float
        Fraction of the total population currently infectious with each
        variant. Each entry must be in [0, 1]. Must have the same length
        as protection_against_variant.

    Returns
    -------
    x : np.ndarray
        Flattened initial state vector of length 2^(2n), representing the
        full distribution of the population across infectious and protection
        states.

    Raises
    ------
    ValueError
        If protection_against_variant and infectious_with_variant do not
        have the same length.
    """
    n_variants = len(protection_against_variant)
    if not n_variants == len(infectious_with_variant):
        raise ValueError(
            "protection_against_variant and infectious_with_variant must be of same length"
        )
    I = create_protection_state_from_seroprevalence(
        fraction_with_antibodies, protection_against_variant, matrix_form=True
    )
    I = distribute_infectious_fraction(I, infectious_with_variant)
    x = map_I_to_x(I)

    return x
