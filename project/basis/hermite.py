"""
Hermite polynomial basis (probabilist's convention).

Normalized Hermite polynomials:
    h_j(x) = He_j(x) / sqrt(j!)

where He_j satisfies:
    He_0 = 1,  He_1 = x,  He_{n+1} = x*He_n - n*He_{n-1}

Orthonormality:
    integral h_m(x) h_n(x) omega(x) dx = delta_{m,n}
where omega(x) = (1/sqrt(2*pi))*exp(-x^2/2) is the standard Gaussian weight.

Derivative property:
    h'_j(x) = sqrt(j) * h_{j-1}(x)
"""

import numpy as np
from math import factorial, sqrt


def He_coeffs(n: int) -> np.ndarray:
    """
    Return the monomial coefficient vector of He_n(x).
    coeffs[k] is the coefficient of x^k in He_n(x).
    Uses the three-term recurrence on polynomial coefficient arrays.
    """
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 1.0])
    c_prev = np.array([1.0])
    c_curr = np.array([0.0, 1.0])
    for k in range(1, n):
        # He_{k+1}(x) = x * He_k(x) - k * He_{k-1}(x)
        # x * He_k  →  shift coefficients right by 1
        shifted = np.zeros(len(c_curr) + 1)
        shifted[1:] = c_curr
        c_next = shifted - k * np.pad(c_prev, (0, len(shifted) - len(c_prev)))
        c_prev = c_curr
        c_curr = c_next
    return c_curr


def h_coeffs(n: int) -> np.ndarray:
    """
    Monomial coefficient vector of the normalized Hermite polynomial h_n(x).
    h_n = He_n / sqrt(n!).
    """
    return He_coeffs(n) / sqrt(factorial(n))


def hermite_basis_matrix(N: int) -> np.ndarray:
    """
    Build the (N+1) x (N+1) upper-triangular matrix H_N whose row i
    contains the monomial coefficients of h_{i}(x), zero-padded.

    H_N[i, j] = coefficient of x^j in h_i(x)   for i,j = 0,...,N.

    This is the H_n matrix of the paper (with 1-based indexing shifted).
    """
    H = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        c = h_coeffs(i)
        H[i, :len(c)] = c
    return H


def eval_hermite(x: np.ndarray, N: int) -> np.ndarray:
    """
    Evaluate all normalized Hermite polynomials h_0,...,h_N at each point in x.
    Returns array of shape (N+1, len(x)).
    Uses the three-term recurrence on values (stable).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    P = np.zeros((N + 1, n))
    P[0] = 1.0
    if N >= 1:
        P[1] = x
    for k in range(1, N):
        # He_{k+1} = x * He_k - k * He_{k-1}
        # h_{k+1}  = (x * h_k * sqrt(k!) - k * h_{k-1} * sqrt((k-1)!)) / sqrt((k+1)!)
        # Equivalently via normalized recurrence:
        #   h_{k+1} = (x * h_k - sqrt(k) * h_{k-1}) / sqrt(k+1)
        P[k + 1] = (x * P[k] - sqrt(k) * P[k - 1]) / sqrt(k + 1)
    return P


def gaussian_weight(x: np.ndarray) -> np.ndarray:
    """Standard Gaussian weight omega(x) = exp(-x^2/2)/sqrt(2*pi)."""
    return np.exp(-x**2 / 2.0) / sqrt(2.0 * np.pi)
