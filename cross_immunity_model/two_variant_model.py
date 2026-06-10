"""
Defines the two-variant model with the state in the order as for the
computation of the invasion reproduction number and functions returning summed
compartments.
"""

from numpy.typing import ArrayLike


def two_variant_model(
    t: float,
    x: ArrayLike,
    beta1: float,
    beta2: float,
    gamma1: float,
    gamma2: float,
    sigma: float,
    b: float = 0.0,
) -> list:
    """
    Full two-variant cross-immmunity model with demographics.

    Parameters
    ----------
    t : float
        Time (not used).
    x : array-like
        State of the system.
    beta1 : float
        Transmission rate of variant 1.
    beta2 : float
        Transmission rate of variant 2.
    gamma1 : float
        Recovery rate of variant 1.
    gamma2 : float
        Recovery rate of variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.

    Returns
    -------
    f : list
        Derivatives of the state.
    """

    x1, x2, x3, x4, x5, x6, x7, x8, x9 = x

    f1 = (
        -gamma1 * x1
        - beta2 * x1 * (x3 + x4 + x5)
        + (1 - sigma) * beta1 * x6 * (x1 + x2 + x3)
        - b * x1
    )
    f2 = (
        -gamma1 * x2
        + sigma * beta1 * x6 * (x1 + x2 + x3)
        + beta1 * x8 * (x1 + x2 + x3)
        + gamma2 * x3
        - b * x2
    )
    f3 = (
        -gamma1 * x3
        - gamma2 * x3
        + beta1 * x4 * (x1 + x2 + x3)
        + beta2 * x1 * (x3 + x4 + x5)
        - b * x3
    )
    f4 = (
        -gamma2 * x4
        - beta1 * x4 * (x1 + x2 + x3)
        + (1 - sigma) * beta2 * x6 * (x3 + x4 + x5)
        - b * x4
    )
    f5 = (
        -gamma2 * x5
        + sigma * beta2 * x6 * (x3 + x4 + x5)
        + beta2 * x7 * (x3 + x4 + x5)
        + gamma1 * x3
        - b * x5
    )
    f6 = (
        -beta1 * x6 * (x1 + x2 + x3) - beta2 * x6 * (x3 + x4 + x5) + b - b * x6
    )
    f7 = -beta2 * x7 * (x3 + x4 + x5) + gamma1 * x1 - b * x7
    f8 = -beta1 * x8 * (x1 + x2 + x3) + gamma2 * x4 - b * x8
    f9 = gamma1 * x2 + gamma2 * x5 - b * x9

    f = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

    return f


def susceptibles(x: ArrayLike) -> float:
    """
    Computes the fully susceptible fraction of state x.

    Parameters
    ----------
    x : array-like
        State of the system.

    Returns
    -------
    s : float
        The fully susceptible fraction of state x.
    """
    s = x[5]
    return s


def protected_1(x: ArrayLike) -> float:
    """
    Computes the protected only against variant 1 fraction of state x.

    Parameters
    ----------
    x : array-like
        State of the system.

    Returns
    -------
    p1 : float
        The protected only against variant 1 fraction of state x.
    """
    p1 = x[0] + x[6]
    return p1


def protected_2(x: ArrayLike) -> float:
    """
    Computes the protected only against variant 2 fraction of state x.

    Parameters
    ----------
    x : array-like
        State of the system.

    Returns
    -------
    p2 : float
        The protected only against variant 2 fraction of state x.
    """
    p2 = x[3] + x[7]
    return p2


def protected_12(x: ArrayLike) -> float:
    """
    Computes the protected against both variants fraction of state x.

    Parameters
    ----------
    x : array-like
        State of the system.

    Returns
    -------
    p12 : float
        The protected against both variants fraction of state x.
    """
    p12 = x[1] + x[2] + x[4] + x[8]
    return p12
