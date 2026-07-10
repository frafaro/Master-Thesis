"""
COS method benchmark  (Fang & Oosterlee 2009).

PDF approximation:
    p(x) ≈ (2/(b-a)) * sum_{k=0}^{N-1} ' Re[ phi(k*pi/(b-a)) * exp(-i*k*pi*a/(b-a)) ]
                                          * cos(k*pi*(x-a)/(b-a))
where ' means the k=0 term is halved.

References
----------
Fang, F., Oosterlee, C.W. (2009). "A novel pricing method for European options
based on Fourier-cosine series expansions." SIAM J. Sci. Comput. 31(2), 826–848.

The Heston CF uses the Gatheral form (models/heston.py) which avoids branch
cut issues for large |u|.  See also Lord & Kahl (2010) for branch-cut analysis.
"""

import numpy as np
from typing import Callable, Tuple


def cos_density(x: np.ndarray,
                cf: Callable,
                a: float, b: float,
                N_cos: int = 4096) -> np.ndarray:
    """
    Evaluate the COS density approximation at points x.

    Parameters
    ----------
    x     : evaluation points, shape (M,)
    cf    : characteristic function phi(u), callable on complex array
    a, b  : domain [a, b]
    N_cos : number of cosine terms (paper uses 2^12 = 4096)

    Returns
    -------
    p : density values at x, shape (M,)
    """
    x = np.asarray(x, dtype=float)
    k = np.arange(N_cos)
    u_k = k * np.pi / (b - a)

    # COS coefficients: F_k = (2/(b-a)) * Re[ phi(u_k) * exp(-i*u_k*a) ]
    phi_vals = cf(u_k)
    F_k = np.real(phi_vals * np.exp(-1j * u_k * a))
    F_k[0] *= 0.5  # half weight for k=0

    # Evaluate sum_k F_k * cos(u_k * (x - a))
    # x: (M,),  u_k: (N_cos,)  → broadcast
    cos_mat = np.cos(np.outer(u_k, x - a))  # (N_cos, M)
    p = (2.0 / (b - a)) * F_k @ cos_mat     # (M,)

    # Clip negative values (numerical artifact for large domain or fat tails)
    p = np.maximum(p, 0.0)
    return p


def benchmark_fourier_coeffs(cos_log_p: np.ndarray,
                              x: np.ndarray,
                              eval_basis: Callable,
                              nu_weight: np.ndarray,
                              m1: float, sigma: float,
                              N: int) -> np.ndarray:
    """
    Estimate the exact Fourier coefficients c_j = <clr(p), phi_j>_nu  (eq. 9)
    using the COS-estimated density.

    c_j = integral_I clr(p)(x) * phi_j(x*) * nu(x*)/sigma dx

    where clr(p)(x) = log(p(x)) - E_nu[log(p(X))]
    and nu_weight is evaluated at x (already on the integration grid, original scale).

    Parameters
    ----------
    cos_log_p  : log of COS density at grid points, shape (M,)
    x          : integration grid (original scale), shape (M,)
    eval_basis : callable(x*) → (N+1, M) basis polynomial array
    nu_weight  : weight function nu evaluated at original x, shape (M,)
                 (already includes the 1/sigma Jacobian if necessary)
    m1, sigma  : standardization parameters
    N          : number of Fourier coefficients to compute

    Returns
    -------
    c : array of length N,  c[j-1] = c_j  (j=1,...,N)
    """
    x_std = (x - m1) / sigma
    P = eval_basis(x_std)  # (N+1, M)

    # E_nu[log p] = integral log(p(x)) * nu(x*)/sigma dx
    # nu_weight is already nu(x*)/sigma evaluated at x
    E_log_p = np.trapz(cos_log_p * nu_weight, x)

    # clr(p)(x) = log(p(x)) - E_nu[log p]
    clr_p = cos_log_p - E_log_p

    # c_j = integral clr_p(x) * phi_j(x*) * nu(x*)/sigma dx
    c = np.zeros(N)
    for j in range(1, N + 1):
        c[j - 1] = np.trapz(clr_p * P[j] * nu_weight, x)
    return c


def verify_cos_density(x: np.ndarray, p: np.ndarray,
                       raw_moments_true: np.ndarray,
                       tol: float = 1e-4) -> bool:
    """
    Verify the COS density by checking:
      1. integral p(x) dx ≈ 1
      2. integral x * p(x) dx ≈ mu_1
      3. integral x^2 * p(x) dx ≈ mu_2
    """
    ok = True
    integ = np.trapz(p, x)
    if abs(integ - 1.0) > tol:
        print(f"  COS density normalization: {integ:.6f}")
        ok = False
    mu1_num = np.trapz(x * p, x)
    if abs(mu1_num - raw_moments_true[1]) > tol:
        print(f"  COS mean: {mu1_num:.6f} vs {raw_moments_true[1]:.6f}")
        ok = False
    mu2_num = np.trapz(x**2 * p, x)
    if abs(mu2_num - raw_moments_true[2]) > 10 * tol:
        print(f"  COS E[X^2]: {mu2_num:.6f} vs {raw_moments_true[2]:.6f}")
        ok = False
    return ok
