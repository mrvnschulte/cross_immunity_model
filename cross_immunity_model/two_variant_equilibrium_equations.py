"""
Defines the equilibrium equations for the two-variant model.
"""

from numpy.typing import ArrayLike


def x_6(
    i_1: float,
    i_2: float,
    beta_1: float,
    beta_2: float,
    gamma_1: float,
    gamma_2: float,
    sigma: float,
    b: float,
) -> float:
    """
    Endemic equilibrium of compartment x_6.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    beta_1 : float
        Transmission rate for variant 1.
    beta_2 : float
        Transmission rate for variant 2.
    gamma_1 : float
        Reocvery rate for variant 1.
    gamma_2 : float
        Recovery rate for variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.

    Returns
    -------
    float
        Value of x_6 at two-strain endemic equilibrium.
    """
    return b / (beta_1 * i_1 + beta_2 * i_2 + b)


def x_1(
    i_1: float,
    i_2: float,
    beta_1: float,
    beta_2: float,
    gamma_1: float,
    gamma_2: float,
    sigma: float,
    b: float,
) -> float:
    """
    Endemic equilibrium of compartment x_1.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    beta_1 : float
        Transmission rate for variant 1.
    beta_2 : float
        Transmission rate for variant 2.
    gamma_1 : float
        Reocvery rate for variant 1.
    gamma_2 : float
        Recovery rate for variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.

    Returns
    -------
    float
        Value of x_1 at two-strain endemic equilibrium.
    """
    return (
        (1 - sigma)
        * beta_1
        * x_6(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        * i_1
    ) / (gamma_1 + beta_2 * i_2 + b)


def x_4(
    i_1: float,
    i_2: float,
    beta_1: float,
    beta_2: float,
    gamma_1: float,
    gamma_2: float,
    sigma: float,
    b: float,
) -> float:
    """
    Endemic equilibrium of compartment x_4.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    beta_1 : float
        Transmission rate for variant 1.
    beta_2 : float
        Transmission rate for variant 2.
    gamma_1 : float
        Reocvery rate for variant 1.
    gamma_2 : float
        Recovery rate for variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.

    Returns
    -------
    float
        Value of x_4 at two-strain endemic equilibrium.
    """
    return (
        (1 - sigma)
        * beta_2
        * x_6(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        * i_2
    ) / (gamma_2 + beta_1 * i_1 + b)


def x_7(
    i_1: float,
    i_2: float,
    beta_1: float,
    beta_2: float,
    gamma_1: float,
    gamma_2: float,
    sigma: float,
    b: float,
) -> float:
    """
    Endemic equilibrium of compartment x_7.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    beta_1 : float
        Transmission rate for variant 1.
    beta_2 : float
        Transmission rate for variant 2.
    gamma_1 : float
        Reocvery rate for variant 1.
    gamma_2 : float
        Recovery rate for variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.

    Returns
    -------
    float
        Value of x_7 at two-strain endemic equilibrium.
    """
    return (
        gamma_1 * x_1(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
    ) / (beta_2 * i_2 + b)


def x_8(
    i_1: float,
    i_2: float,
    beta_1: float,
    beta_2: float,
    gamma_1: float,
    gamma_2: float,
    sigma: float,
    b: float,
) -> float:
    """
    Endemic equilibrium of compartment x_8.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    beta_1 : float
        Transmission rate for variant 1.
    beta_2 : float
        Transmission rate for variant 2.
    gamma_1 : float
        Reocvery rate for variant 1.
    gamma_2 : float
        Recovery rate for variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.

    Returns
    -------
    float
        Value of x_8 at two-strain endemic equilibrium.
    """
    return (
        gamma_2 * x_4(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
    ) / (beta_1 * i_1 + b)


def x_3(
    i_1: float,
    i_2: float,
    beta_1: float,
    beta_2: float,
    gamma_1: float,
    gamma_2: float,
    sigma: float,
    b: float,
) -> float:
    """
    Endemic equilibrium of compartment x_3.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    beta_1 : float
        Transmission rate for variant 1.
    beta_2 : float
        Transmission rate for variant 2.
    gamma_1 : float
        Reocvery rate for variant 1.
    gamma_2 : float
        Recovery rate for variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.

    Returns
    -------
    float
        Value of x_3 at two-strain endemic equilibrium.
    """
    return (
        beta_1
        * x_4(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        * i_1
        + beta_2
        * x_1(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        * i_2
    ) / (gamma_1 + gamma_2 + b)


def x_2(
    i_1: float,
    i_2: float,
    beta_1: float,
    beta_2: float,
    gamma_1: float,
    gamma_2: float,
    sigma: float,
    b: float,
) -> float:
    """
    Endemic equilibrium of compartment x_2.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    beta_1 : float
        Transmission rate for variant 1.
    beta_2 : float
        Transmission rate for variant 2.
    gamma_1 : float
        Reocvery rate for variant 1.
    gamma_2 : float
        Recovery rate for variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.

    Returns
    -------
    float
        Value of x_2 at two-strain endemic equilibrium.
    """
    return (
        sigma
        * beta_1
        * x_6(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        * i_1
        + beta_1
        * x_8(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        * i_1
        + gamma_2 * x_3(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
    ) / (gamma_1 + b)


def x_5(
    i_1: float,
    i_2: float,
    beta_1: float,
    beta_2: float,
    gamma_1: float,
    gamma_2: float,
    sigma: float,
    b: float,
) -> float:
    """
    Endemic equilibrium of compartment x_5.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    beta_1 : float
        Transmission rate for variant 1.
    beta_2 : float
        Transmission rate for variant 2.
    gamma_1 : float
        Reocvery rate for variant 1.
    gamma_2 : float
        Recovery rate for variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.

    Returns
    -------
    float
        Value of x_5 at two-strain endemic equilibrium.
    """
    return (
        sigma
        * beta_2
        * x_6(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        * i_2
        + beta_2
        * x_7(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        * i_2
        + gamma_1 * x_3(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
    ) / (gamma_2 + b)


def x_9(
    i_1: float,
    i_2: float,
    beta_1: float,
    beta_2: float,
    gamma_1: float,
    gamma_2: float,
    sigma: float,
    b: float,
) -> float:
    """
    Endemic equilibrium of compartment x_9.

    Parameters
    ----------
    i_1 : float
        Infectious fraction with variant 1.
    i_2 : float
        Infectious fraction with variant 2.
    beta_1 : float
        Transmission rate for variant 1.
    beta_2 : float
        Transmission rate for variant 2.
    gamma_1 : float
        Reocvery rate for variant 1.
    gamma_2 : float
        Recovery rate for variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.

    Returns
    -------
    float
        Value of x_9 at two-strain endemic equilibrium.
    """
    return (
        gamma_1 * x_2(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        + gamma_2 * x_5(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
    ) / b


def endemic_equilibrium_equation(
    i: ArrayLike,
    beta_1: float,
    beta_2: float,
    gamma_1: float,
    gamma_2: float,
    sigma: float,
    b: float,
) -> list:
    """
    Defines the characteristic equation of the endemic equilibrium for i.

    Parameters
    ----------
    i : array-like
        Infectious fraction with each variant as list.
    beta_1 : float
        Transmission rate for variant 1.
    beta_2 : float
        Transmission rate for variant 2.
    gamma_1 : float
        Reocvery rate for variant 1.
    gamma_2 : float
        Recovery rate for variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.

    Returns
    -------
    list
        Difference between infectious fraction given and as computed for the
        two-variant endemic equilibrium.
    """
    i_1, i_2 = i
    return [
        x_1(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        + x_2(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        + x_3(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        - i_1,
        x_3(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        + x_4(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        + x_5(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b)
        - i_2,
    ]


def x_bar(
    i: ArrayLike,
    beta_1: float,
    beta_2: float,
    gamma_1: float,
    gamma_2: float,
    sigma: float,
    b: float,
) -> list:
    """
    Computes the two-variant endemic equilibrium compartments for given i.

    Parameters
    ----------
    i : array-like
        Infectious fraction with each variant as list.
    beta_1 : float
        Transmission rate for variant 1.
    beta_2 : float
        Transmission rate for variant 2.
    gamma_1 : float
        Reocvery rate for variant 1.
    gamma_2 : float
        Recovery rate for variant 2.
    sigma : float
        Cross-immunity factor.
    b : float
        Birth and death rate.

    Returns
    -------
    list
        Two-strain endemic equilibrium in original compartments.
    """

    i_1, i_2 = i
    return [
        x_1(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b),
        x_2(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b),
        x_3(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b),
        x_4(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b),
        x_5(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b),
        x_6(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b),
        x_7(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b),
        x_8(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b),
        x_9(i_1, i_2, beta_1, beta_2, gamma_1, gamma_2, sigma, b),
    ]
