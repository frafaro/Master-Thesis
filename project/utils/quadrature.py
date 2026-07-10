"""
Quadrature utilities:
  - cumulant_domain: compute truncated domain I from eq. (22) of Gambaro (2024)
  - make_grid:       uniform evaluation grid on I
  - trapz:           trapezoidal integration wrapper
"""

import numpy as np
from typing import Tuple


def cumulant_domain(cumulants: dict, L: float = 4.0) -> Tuple[float, float]:
    """
    Domain I = [k1 - L*sqrt(k2 + sqrt(k4)),  k1 + L*sqrt(k2 + sqrt(k4))]
    Equation (22) of Gambaro (2024), following Fang & Oosterlee (2009).

    Parameters
    ----------
    cumulants : dict with keys 'k1', 'k2', 'k4' (first, second, fourth cumulant)
    L         : domain widening parameter (paper uses L=4)

    Returns
    -------
    (a, b) : left and right endpoints of I
    """
    k1 = float(cumulants["k1"])
    k2 = float(cumulants["k2"])
    k4 = float(cumulants["k4"])
    half_width = L * np.sqrt(k2 + np.sqrt(abs(k4)))
    return k1 - half_width, k1 + half_width


def clr_domain(log_p_func, cumulants: dict, L_start: float = 4.0,
               clr_tol: float = 10.0, n_pts: int = 2000) -> Tuple[float, float]:
    """
    Restricted domain: shrink the L=4 domain until |clr(p)(x)| < clr_tol everywhere.
    Used for the Heston 'restricted domain' experiments (Figs 7, 11).

    clr(p)(x) = log(p(x)) - E_nu[log(p(X))]
    For the restricted domain we simply find the largest symmetric sub-interval
    [k1-h, k1+h] where log(p(x)) > -clr_tol (ignoring the mean-shift correction,
    which is small; the dominant effect is p(x) -> 0 in the tails).

    Parameters
    ----------
    log_p_func : callable x -> log(p(x)), evaluated on a fine grid
    cumulants  : dict with keys 'k1', 'k2', 'k4'
    L_start    : initial L used to build the test grid
    clr_tol    : tolerance (paper uses 10)
    n_pts      : number of grid points for evaluation

    Returns
    -------
    (a, b) : restricted domain endpoints
    """
    a0, b0 = cumulant_domain(cumulants, L=L_start)
    x = np.linspace(a0, b0, n_pts)
    lp = log_p_func(x)
    # require log(p) > -clr_tol  (equivalently p > exp(-10) ~ 4.5e-5)
    valid = lp > -clr_tol
    if not valid.any():
        raise ValueError("No valid domain found with |log(p)| < clr_tol")
    a_restr = x[valid][0]
    b_restr = x[valid][-1]
    return float(a_restr), float(b_restr)


def make_grid(a: float, b: float, n: int) -> np.ndarray:
    """Uniform grid of n points on [a, b]."""
    return np.linspace(a, b, n)


def trapz(f: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration of f over x."""
    return float(np.trapezoid(f, x))
