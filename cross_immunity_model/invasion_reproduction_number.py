"""
Defines functions to compute the invasion reproduction number as well as
critical values for the invasion ability.
"""

import numpy as np
from numpy.typing import ArrayLike

from .model_transformations import susceptibles_per_variant
from .utils import basic_reproduction_number


def invasion_reproduction_number(
    r01: float, r02: float, sigma: float
) -> float:
    """
    Computes the invasion reproduction number of variant 1 invading variant 2
    equilibrium.

    Parameters
    ----------
    r01 : float
        Basic reproduction number of variant 1.
    r02 : float
        Basic reproduction number of variant 2.
    sigma : float
        Cross-immunity between the two variants.

    Returns
    -------
    invasion_r0 : float
        Invasion reproduction number of variant 1 invading the variant 2
        equilibrium.
    """
    invasion_r0 = r01 - sigma * (r01 / r02) * (r02 - 1)
    return invasion_r0


def invasion_reproduction_number_multiple_variants(
    x_eq: ArrayLike,
    beta: ArrayLike,
    gamma: ArrayLike,
    b: float,
    V: np.ndarray | None = None,
) -> np.ndarray:
    """
    Returns the invasion reproduction number for all variants at given
    equilibrium.

    Parameters
    ----------
    x_eq : array-like
        State at equilibrium.
    beta : array-like
        Vector of transmission rates of the variants.
    gamma : array-like
        Vector of recovery rates of the variants.
    b : float
        Natural birth and death rate.
    V : np.ndarray | None, optional
        Variant-in-set matrix. If None is given, it is computed.

    Returns
    -------
    inv_r0 : np.ndarray
        Vector of invasion reproduction numbers.
    """
    S_variant = susceptibles_per_variant(x_eq, V)
    r0 = basic_reproduction_number(beta, gamma, b)
    inv_r0 = S_variant * r0
    return inv_r0


def critical_sigma(r01: float, r02: float | np.ndarray) -> float | np.ndarray:
    """
    Computes the critical sigma, under which variant 1 can invade variant 2.

    Parameters
    ----------
    r01 : float
        Basic reproduction number of variant 1.
    r02 : float, or np.ndarray
        Basic reproduction number of variant 2.

    Returns
    -------
    sigma : float, or np.ndarray
        Critical sigma, under which variant 1 can invade variant 2.
    """
    r02 = np.asarray(r02, dtype=float)
    sigma = np.asarray(np.ones_like(r02) * np.inf)
    mask = r02 >= r01
    sigma[mask] = r02[mask] * (r01 - 1) / (r01 * (r02[mask] - 1))
    if sigma.ndim == 0:
        sigma = float(sigma)
    return sigma


def critical_r01(r02: float | np.ndarray, sigma: float) -> float | np.ndarray:
    """
    Computes the critical r01, over which variant 1 can invade variant 2.

    Parameters
    ----------
    r02 : float, or np.ndarray
        Basic reproduction number of variant 2.
    sigma : float
        Cross-immunity between the two variants.

    Returns
    -------
    r01 : float, or np.ndarray
        Critical basic reproduction number of variant 1, over which variant 1
        can invade variant 2.
    """
    r02 = np.asarray(r02, dtype=float)
    r01 = np.ones_like(r02)
    mask = r02 > 1
    r01[mask] = 1.0 / (1.0 - sigma / r02[mask] * (r02[mask] - 1.0))
    if r01.ndim == 0:
        r01 = float(r01)
    return r01


def critical_r02(r01: float | np.ndarray, sigma: float) -> float | np.ndarray:
    """
    Computes the critical r02, under which variant 1 can invade variant 2.

    Parameters
    ----------
    r01 : float, or np.ndarray
        Basic reproduction number of variant 1.
    sigma : float
        Cross-immunity between the two variants.

    Returns
    -------
    r01 : float, or np.ndarray
        Critical basic reproduction number of variant 2, under which variant 1
        can invade variant 2.
    """
    r01 = np.asarray(r01, dtype=float)
    r02 = np.ones_like(r01)
    mask = r01 >= 1
    r02_star = sigma * 1 / (sigma + 1 / r01 - 1)
    r02[mask] = r02_star[mask]
    r02[mask & (r02_star <= 0)] = np.inf
    if r02.ndim == 0:
        r02 = float(r02)
    return r02
