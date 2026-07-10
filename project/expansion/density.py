"""
Exponential expansion density  (eq. 16, Gambaro 2024):
    p^_N(x) = C^_0 * exp( sum_{j=1}^{N} c^_j * phi_j(x*) )

where x* = (x - m1) / sigma and phi_j is the chosen orthonormal basis.

C^_0 = 1 / (sigma * integral_{I*} exp(sum c^_j phi_j(t)) dt)

The normalization constant is computed via numerical quadrature on the
(standardized) truncated domain I* = (I - m1)/sigma.
"""

import numpy as np
from scipy.integrate import quad
from typing import Callable, Tuple


def exponent_func(x_std: np.ndarray,
                  c_hat: np.ndarray,
                  eval_basis: Callable) -> np.ndarray:
    """
    f(x*) = sum_{j=1}^{N} c^_j * phi_j(x*)

    Parameters
    ----------
    x_std      : standardized grid points x* (1-D array)
    c_hat      : coefficient array of length N  (c^_1,...,c^_N, 0-indexed: c[0]=c^_1)
    eval_basis : callable(x*) → (N+1, len(x*)) array of basis polynomials

    Returns
    -------
    f : 1-D array of shape (len(x_std),)
    """
    P = eval_basis(x_std)   # shape (N+1, len(x*)), rows 0..N
    N = len(c_hat)
    # sum_{j=1}^{N} c[j-1] * P[j, :]
    return np.einsum("j,j...->...", c_hat, P[1:N + 1])


def compute_C0(c_hat: np.ndarray,
               eval_basis: Callable,
               a: float, b: float,
               m1: float, sigma: float,
               n_pts: int = 20_000) -> float:
    """
    Compute the normalization constant:
      C^_0 = 1 / (sigma * integral_{I*} exp(f(t)) dt)
    where I* = [(a-m1)/sigma, (b-m1)/sigma] is the standardized domain.

    Uses the trapezoidal rule on a fine grid of n_pts points.
    For robustness: subtract the maximum of f to avoid overflow, then
    compensate at the end.

    Parameters
    ----------
    c_hat      : coefficient vector c^_1,...,c^_N (0-indexed)
    eval_basis : callable(x*) → basis polynomial array
    a, b       : domain I = [a, b] (original scale)
    m1, sigma  : standardization parameters
    n_pts      : quadrature grid size

    Returns
    -------
    C0 : normalization constant (float)
    """
    a_std = (a - m1) / sigma
    b_std = (b - m1) / sigma
    t = np.linspace(a_std, b_std, n_pts)
    f = exponent_func(t, c_hat, eval_basis)
    # subtract max for numerical stability
    f_max = f.max()
    exp_f = np.exp(f - f_max)
    integral_std = np.trapezoid(exp_f, t)
    # integral over x = sigma * integral over x*
    integral_x = sigma * integral_std * np.exp(f_max)
    return 1.0 / integral_x


def eval_density(x: np.ndarray,
                 c_hat: np.ndarray,
                 C0: float,
                 eval_basis: Callable,
                 m1: float, sigma: float) -> np.ndarray:
    """
    Evaluate  p^_N(x) = C^_0 * exp( sum_j c^_j * phi_j(x*) )
    at the given x values.

    Parameters
    ----------
    x          : evaluation points (original scale)
    c_hat      : coefficient vector
    C0         : normalization constant
    eval_basis : callable(x*) → basis array
    m1, sigma  : standardization parameters

    Returns
    -------
    p : density values at x, shape (len(x),)
    """
    x_std = (np.asarray(x) - m1) / sigma
    f = exponent_func(x_std, c_hat, eval_basis)
    return C0 * np.exp(f)


def verify_normalization(x: np.ndarray, p: np.ndarray,
                         tol: float = 1e-4) -> bool:
    """Check that integral of p over x ≈ 1."""
    integral = np.trapezoid(p, x)
    ok = abs(integral - 1.0) < tol
    if not ok:
        print(f"  Normalization check: integral = {integral:.6f}  (tol={tol:.1e})")
    return ok
