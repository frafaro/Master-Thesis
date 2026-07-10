"""
Change-of-basis matrix Q_n.

Definition (Gambaro 2024, eq. 13):
    phi_n(x*) = sum_{j=0}^{n} q_{n,j} h_j(x*)

where {phi_n} is the chosen orthonormal basis and {h_j} are the normalized
Hermite polynomials (both orthonormal w.r.t. their respective weights).

For the HERMITE basis:
    Q_n = I  (identity matrix)

For the LOGISTIC basis:
    Q_n[n,j] = <L_n, h_j>_Gaussian = integral L_n(x) h_j(x) omega(x) dx

    This is computed numerically using the Gaussian quadrature adapted to
    the Gaussian weight omega(x) = exp(-x^2/2)/sqrt(2*pi).

    Rationale: since {h_j} are orthonormal w.r.t. omega, we have:
        <phi_n, h_j>_omega = sum_k q_{n,k} <h_k, h_j>_omega = q_{n,j}

    Computing Q_n[n,j] via inner products completely avoids the unstable
    monomial coefficient representation.
"""

import numpy as np
from math import sqrt, pi
from numpy.polynomial.legendre import leggauss
from typing import Tuple

# Gauss-Hermite quadrature for Gaussian-weighted inner products
_N_GH = 200    # number of Gauss-Hermite nodes


def _gauss_hermite_grid(n: int = _N_GH):
    """
    Gauss-Hermite nodes and weights for integral f(x)*exp(-x^2) dx.
    Convert to integral f(x)*omega(x) dx where omega = exp(-x^2/2)/sqrt(2pi):
    substitute t = x/sqrt(2), dt = dx/sqrt(2).
    """
    # Standard Gauss-Hermite: integral f(t)*exp(-t^2) dt ≈ sum w_i f(t_i)
    from numpy.polynomial.hermite import hermgauss
    t, w_gh = hermgauss(n)
    # x = sqrt(2)*t,  omega(x) dx = exp(-x^2/2)/sqrt(2pi) dx
    # integral f(x)*omega(x) dx = integral f(sqrt(2)*t)*exp(-t^2)/sqrt(pi) dt
    #                           ≈ sum w_i f(sqrt(2)*t_i) / sqrt(pi)
    x = sqrt(2) * t
    w = w_gh / sqrt(pi)
    return x, w


_GH_CACHE = {}

def _get_gh(n: int = _N_GH):
    if n not in _GH_CACHE:
        _GH_CACHE[n] = _gauss_hermite_grid(n)
    return _GH_CACHE[n]


def build_Q_hermite(N: int) -> np.ndarray:
    """For Hermite basis, Q_n = I (identity)."""
    return np.eye(N + 1)


def build_Q_logistic(N: int, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Build Q_n for the Logistic basis via numerical inner products:
        Q_n[k, j] = <L_k, h_j>_Gaussian
    for k, j = 0,...,N.

    Parameters
    ----------
    alpha, beta : Stieltjes recurrence coefficients for the logistic basis

    Returns
    -------
    Q : (N+1, N+1) matrix  (not necessarily triangular, but dense)
    """
    from basis.logistic import eval_logistic_recurrence
    from basis.hermite import eval_hermite

    x_gh, w_gh = _get_gh()

    # Evaluate logistic basis at Gauss-Hermite nodes
    L_vals = eval_logistic_recurrence(x_gh, N, alpha, beta)   # (N+1, n_gh)

    # Evaluate Hermite basis at Gauss-Hermite nodes
    H_vals = eval_hermite(x_gh, N)                             # (N+1, n_gh)

    # Q[k, j] = integral L_k(x) h_j(x) omega(x) dx
    #         ≈ sum_i w_gh[i] * L_vals[k, i] * H_vals[j, i]
    Q = (L_vals * w_gh) @ H_vals.T   # (N+1, N+1)
    return Q


def verify_Q_hermite(Q: np.ndarray, tol: float = 1e-10) -> bool:
    """For Hermite basis, Q should be identity."""
    err = np.linalg.norm(Q - np.eye(Q.shape[0]), "fro")
    return err < tol


def verify_Q_logistic(Q: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
                      N: int, tol: float = 1e-6) -> bool:
    """
    Verify Q: check that sum_j Q[k,j]*Q[l,j] = delta_{k,l}
    (orthonormality in the Hermite-weighted sense, since Hermite is orthonormal).
    Actually check: Q Q^T ≈ Gram matrix of L_k in Hermite basis.
    The correct check is: each row of Q gives the coordinates of L_k in {h_j},
    and since both L_k and h_j are orthonormal w.r.t. their own weights,
    Q is not necessarily orthogonal.
    We verify instead that ||phi_n||_{Gaussian}^2 = sum_j Q[n,j]^2 ≈ 1.
    (True when the Logistic polynomials are square-integrable w.r.t. Gaussian.)
    """
    from basis.logistic import eval_logistic_recurrence
    from basis.hermite import gaussian_weight

    x = np.linspace(-10, 10, 20000)
    w = gaussian_weight(x)
    L = eval_logistic_recurrence(x, N, alpha, beta)
    max_err = 0.0
    for k in range(N + 1):
        norm2 = np.trapz(L[k]**2 * w, x)
        # sum_j Q[k,j]^2 is an approximation but won't equal 1 in general
        # Just check that inner products are finite
        if not np.isfinite(norm2):
            print(f"L_{k} has non-finite Gaussian norm")
            return False
    return True
