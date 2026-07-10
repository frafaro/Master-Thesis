"""
Logistic polynomial basis  (Heston & Rossi 2016).

Polynomials L_0,...,L_N orthonormal w.r.t. the standard logistic weight:
    nu_L(x) = exp(-x) / (1 + exp(-x))^2  (mean=0, variance=pi^2/3)

Algorithm:
  1. Stieltjes/Lanczos algorithm on a Gauss-Legendre quadrature grid to
     compute the three-term recurrence coefficients (alpha_k, beta_k):
       L_{k+1}(x) = (x - alpha_k)*L_k(x) - beta_k*L_{k-1}(x)
  2. Evaluation at any x via the recurrence (stable for all N).

The change-of-basis matrix Q_n[k,j] = <L_k, h_j>_Gaussian is computed
separately in matrices/change_of_basis.py via numerical inner products,
bypassing the numerically unstable monomial coefficient representation.

Note: For symmetric nu_L, alpha_k = 0 for all k (verified numerically).
"""

import numpy as np
from math import sqrt
from typing import Tuple

_INTEG_LIM = 30.0
_N_QUAD    = 3000    # Gauss-Legendre points


def logistic_weight(x: np.ndarray) -> np.ndarray:
    """nu_L(x) = sech^2(x/2) / 4."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (4.0 * np.cosh(x / 2.0)**2)


def _make_quad_grid(n_quad: int = _N_QUAD, lim: float = _INTEG_LIM):
    """Gauss-Legendre nodes on [-lim, lim] with logistic weights."""
    from numpy.polynomial.legendre import leggauss
    t, w_gl = leggauss(n_quad)
    x = lim * t
    w = w_gl * lim * logistic_weight(x)
    return x, w


_QUAD_CACHE = {}

def _get_quad(n_quad: int = _N_QUAD):
    if n_quad not in _QUAD_CACHE:
        _QUAD_CACHE[n_quad] = _make_quad_grid(n_quad)
    return _QUAD_CACHE[n_quad]


# ── Stieltjes recurrence coefficients ────────────────────────────────────────

def stieltjes_recurrence(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute three-term recurrence coefficients (alpha, beta) for the
    logistic orthogonal polynomials via the Stieltjes algorithm:
        L_{k+1}(x) = (x - alpha[k])*L_k(x) - beta[k]*L_{k-1}(x)

    For symmetric nu_L: alpha[k] ≡ 0.

    Returns
    -------
    alpha : (N+1,) array  (all ≈ 0 for logistic)
    beta  : (N+1,) array  (> 0)
    """
    x, w = _get_quad()

    # Initialize the recurrence
    pi_km1 = np.zeros_like(x)            # pi_{-1} = 0
    pi_k   = np.ones_like(x)             # pi_0 = 1 (monic)
    norm_km1 = 1.0                        # sentinel
    norm_k   = float(w @ pi_k**2)        # ||pi_0||^2 = 1

    alpha = np.zeros(N + 1)
    beta  = np.zeros(N + 1)

    for k in range(N):
        alpha_k = float(w @ (x * pi_k**2)) / norm_k
        alpha[k] = alpha_k
        beta_k   = norm_k / norm_km1 if k > 0 else norm_k  # beta_0 = ||pi_0||^2
        beta[k]  = beta_k

        pi_kp1 = (x - alpha_k) * pi_k - (norm_k / norm_km1 if k > 0 else 0.0) * pi_km1
        norm_kp1 = float(w @ pi_kp1**2)

        pi_km1   = pi_k.copy()
        pi_k     = pi_kp1
        norm_km1 = norm_k
        norm_k   = norm_kp1

    alpha[N] = float(w @ (x * pi_k**2)) / norm_k
    beta[N]  = norm_k / norm_km1

    return alpha, beta


def eval_logistic_recurrence(x: np.ndarray, N: int,
                              alpha: np.ndarray,
                              beta: np.ndarray) -> np.ndarray:
    """
    Evaluate the NORMALIZED logistic orthogonal polynomials L_0,...,L_N
    at each point in x using the three-term recurrence.

    The monic polynomials satisfy:
       pi_{k+1} = (x - alpha[k])*pi_k - (||pi_k||/||pi_{k-1}||)*pi_{k-1}

    The normalized polynomials are L_k = pi_k / ||pi_k||.

    To avoid carrying the norm explicitly, we use the equivalent recurrence
    for normalized polynomials.  Let L_k = pi_k / gamma_k where gamma_k = ||pi_k||.
    Then:
      L_{k+1} = [(x - alpha_k)*pi_k - beta_k*pi_{k-1}] / gamma_{k+1}
              = [(x - alpha_k)*gamma_k*L_k - beta_k*gamma_{k-1}*L_{k-1}] / gamma_{k+1}

    Since gamma_k^2 = beta_k * gamma_{k-1}^2 (from the Stieltjes algorithm:
    norm_k = beta_k * norm_{k-1}), we have gamma_{k+1} = sqrt(beta_{k+1}) * gamma_k.
    The normalized recurrence is:
      L_{k+1} = (x - alpha_k)/sqrt(beta_{k+1}) * L_k
                - sqrt(beta_k/beta_{k+1}) * L_{k-1}

    This is the standard form for normalized orthogonal polynomials.

    Returns
    -------
    P : (N+1, len(x)) array
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    P = np.zeros((N + 1, n))

    # L_0 = 1 / sqrt(beta[0])  where beta[0] = ||pi_0||^2 = 1 (since pi_0=1, integral nu_L=1)
    # But from stieltjes_recurrence: beta[0] = norm_0 = 1 (for normalized nu_L)
    b0 = beta[0]
    P[0] = 1.0 / sqrt(b0)      # L_0 = 1 / sqrt(||pi_0||^2) = 1 (since b0=1)

    if N == 0:
        return P

    # L_1: pi_1 = x - alpha_0, ||pi_1||^2 = beta[1]
    P[1] = (x - alpha[0]) * P[0] / sqrt(beta[1])

    for k in range(1, N):
        # L_{k+1} = [(x - alpha_k)*L_k - sqrt(beta_k)*L_{k-1}] / sqrt(beta_{k+1})
        P[k + 1] = ((x - alpha[k]) * P[k] - sqrt(beta[k]) * P[k - 1]) / sqrt(beta[k + 1])

    return P


# Alias for backward compatibility
def eval_logistic(x: np.ndarray, N: int,
                  B_L_or_alpha,
                  beta: np.ndarray = None) -> np.ndarray:
    """
    Evaluate logistic polynomials.
    Accepts either:
      - eval_logistic(x, N, alpha, beta)   → recurrence mode (recommended)
      - eval_logistic(x, N, B_L)           → Horner mode (may be unstable for N>10)
    """
    if beta is not None:
        return eval_logistic_recurrence(x, N, B_L_or_alpha, beta)
    else:
        # Horner mode (legacy): B_L_or_alpha is the coefficient matrix
        B_L = B_L_or_alpha
        x = np.asarray(x, dtype=float)
        n_pts = len(x)
        P = np.zeros((N + 1, n_pts))
        for k in range(N + 1):
            coeffs = B_L[k]
            val = np.zeros(n_pts)
            for j in range(N, -1, -1):
                val = val * x + coeffs[j]
            P[k] = val
        return P


def logistic_raw_moment(k: int) -> float:
    """E_L[X^k] via Gauss-Legendre quadrature."""
    x, w = _get_quad()
    return float(w @ x**k)
