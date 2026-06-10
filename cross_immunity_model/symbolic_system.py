"""
Defines the multi-variant model symbolically.
"""

from functools import lru_cache
from typing import Callable

import numpy as np
from sympy import lambdify, Matrix, Symbol, symbols

from .multi_variant_model import multi_variant_model


def build_symbolic_system(
    n: int,
) -> tuple[np.ndarray, list, np.ndarray, np.ndarray, np.ndarray, Symbol]:
    """
    Builds the multi-variant model for given number of variants symbolically.

    Parameters
    ----------
    n : int
        Number of variants.

    Returns
    -------
    dx : np.ndarray
        Vector of derivatives of the state.
    x : list
        Vector of symbolic compartments.
    beta : np.ndarray
        Vector of symbolic transmission rates.
    gamma : np.ndarray
        Vector of symbolic recovery rates.
    sigma : np.ndarray
        Matrix of symbolic cross-immunity factors.
    b : Symbol
        Symbolic natural birth and death rate.

    """
    N = frozenset(range(1, n + 1))

    x = [symbols(f"x_{k}") for k in range(3**n)]
    gamma = np.array([symbols(f"gamma_{k}") for k in N], dtype=object)
    beta = np.array([symbols(f"beta_{k}") for k in N], dtype=object)
    sigma = np.array(
        [[symbols(f"sigma_{k}{m}") for m in N] for k in N],
        dtype=object,
    )
    b = symbols("b")

    dx = multi_variant_model(0.0, x, beta, gamma, sigma, b, dtype=object)

    return dx, x, beta, gamma, sigma, b


@lru_cache(maxsize=8)
def build_jacobian_function(n_variants: int) -> Callable[..., np.ndarray]:
    """
    Build the jacobian of the system symbolically and lambdify it.

    Parameters
    ----------
    n_variants : int
        Number of variants.

    Returns
    -------
    callable
        Callable function of jacobian of the system.
    """
    dx, x, beta_symb, gamma_symb, sigma_symb, b_symb = build_symbolic_system(
        n_variants
    )
    J_symb = Matrix(dx).jacobian(x)
    all_symbols = (
        list(x)
        + list(beta_symb)
        + list(gamma_symb)
        + list(sigma_symb.flatten())
        + [b_symb]
    )
    return lambdify(all_symbols, J_symb, modules="numpy")
