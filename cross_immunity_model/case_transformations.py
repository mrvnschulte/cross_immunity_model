"""
Defines utility functions to transform case counts, namely by converting
cumulative to daily, relative to absolute and cumulative to incidence cases
and computing the moving average.
"""

import numpy as np
from numpy.typing import ArrayLike


def cumulative_to_daily_cases(cumulative_cases: ArrayLike) -> np.ndarray:
    """
    Transforms cumulative cases (given daily) to daily number of cases.

    Parameters
    ----------
    cumulative_cases : array-like
        Cumulative number of cases, given once per day.

    Returns
    -------
    daily_cases : np.ndarray
        Daily number of cases.
    """
    daily_cases = np.diff(cumulative_cases, axis=0)
    return daily_cases


def relative_to_absolute_cases(
    relative_cases: ArrayLike, population: int
) -> np.ndarray:
    """
    Transforms relative number of cases to absolute number of cases.

    Parameters
    ----------
    relative_cases : array-like
        Relative number of cases.
    population : int
        Population size.

    Returns
    -------
    absolute_cases : np.ndarray
        Absolute number of cases.
    """
    absolute_cases = np.array(relative_cases) * population
    return absolute_cases


def moving_average(cases: ArrayLike, window: int = 7) -> np.ndarray:
    """
    Computes the moving average of cases in given window.

    Parameters
    ----------
    cases : array-like
        Daily number of cases.
    window : int, optional
        Window of moving average. The default is 7.

    Returns
    -------
    np.ndarray
        Moving average of cases in window.
    """
    cases = np.asarray(cases)
    kernel = np.ones(window) / window

    if cases.ndim == 1:
        return np.convolve(cases, kernel, mode="valid")

    return np.apply_along_axis(
        func1d=lambda col: np.convolve(col, kernel, mode="valid"),
        axis=0,
        arr=cases,
    )


def cases_to_incidence(cases: ArrayLike, population: int) -> np.ndarray:
    """
    Transforms case numbers into incidence (cases per 100,000 inhabitants).

    Parameters
    ----------
    cases : ArrayLike
        Absolute number of cases.
    population : int
        Population size.

    Returns
    -------
    incidence : np.ndarray
        Moving average of incidence over given window.
    """
    incidence = np.array(cases) * 1e5 / population
    return incidence


def cumulative_cases_to_incidence(
    cumulative_cases: ArrayLike, population: int, window: int = 7
) -> np.ndarray:
    """
    Transforms cumulative cases, given daily, into the moving average incidence
    (cases per 100,000 inhabitants) over given window.

    Parameters
    ----------
    cumulative_cases : ArrayLike
        Cumulative number of cases, given once per day.
    population : int
        Population size.
    window : int, optional
        Window of moving average. The default is 7.

    Returns
    -------
    incidence : np.ndarray
        Moving average of incidence over given window.
    """
    daily_cases = cumulative_to_daily_cases(cumulative_cases)
    absolute_cases = relative_to_absolute_cases(daily_cases, population)
    average_cases = moving_average(absolute_cases, window)
    incidence = cases_to_incidence(average_cases, population)
    return incidence
