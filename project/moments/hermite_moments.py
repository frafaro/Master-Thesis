"""
Hermite moments computation.

Definition (Gambaro 2024, eq. before (15)):
    m^h_k = E_p[h_k(X*)]  =  E_p[He_k(X*)] / sqrt(k!)

where X* = (X - m1) / sigma,  m1 = E[X],  sigma = sqrt(Var(X)).

Method (Rompolis & Tzavalis 2008, eqs. (2)-(3)):
    sqrt(k!) * m^h_k = E[He_k(X*)]
                     = sum_{j=0}^{k} c_j^{(k)} * mu*_j

where c_j^{(k)} are the monomial coefficients of He_k and mu*_j = E[(X*)^j]
are the raw moments of the standardized variable X*.

This is the most direct and numerically stable approach:
1. Compute raw moments mu_j = E[X^j] analytically from the model.
2. Standardize: mu*_j = E[(X*)^j] = E[((X-m1)/sigma)^j]
   via binomial expansion of (X - m1)^j / sigma^j.
3. Evaluate E[He_k(X*)] by applying the polynomial He_k to the moment sequence.
4. Normalize: m^h_k = E[He_k(X*)] / sqrt(k!).
"""

import numpy as np
from math import factorial, sqrt, comb
from basis.hermite import He_coeffs


def standardize_moments(raw_mu: np.ndarray) -> np.ndarray:
    """
    Given raw moments mu[k] = E[X^k] (k=0,...,K),
    compute the raw moments of the standardized variable X* = (X - m1)/sigma:
        mu*[k] = E[(X*)^k] = E[((X - m1)/sigma)^k]
               = (1/sigma^k) * sum_{j=0}^{k} C(k,j) * (-m1)^{k-j} * E[X^j]

    Parameters
    ----------
    raw_mu : array of length K+1, raw_mu[k] = E[X^k]

    Returns
    -------
    std_mu : array of length K+1, std_mu[k] = E[(X*)^k]
    """
    m1    = raw_mu[1]
    m2    = raw_mu[2]
    sigma = sqrt(m2 - m1**2)

    K = len(raw_mu) - 1
    # Central moments of X: bar_mu[k] = E[(X - m1)^k]
    bar_mu = np.zeros(K + 1)
    for k in range(K + 1):
        s = 0.0
        for j in range(k + 1):
            s += comb(k, j) * (-m1)**(k - j) * raw_mu[j]
        bar_mu[k] = s

    # Standardized moments: mu*[k] = bar_mu[k] / sigma^k
    std_mu = np.zeros(K + 1)
    for k in range(K + 1):
        std_mu[k] = bar_mu[k] / sigma**k if k > 0 else 1.0

    return std_mu, m1, sigma


def hermite_moments_from_raw(raw_mu: np.ndarray, K_max: int = None) -> np.ndarray:
    """
    Compute normalized Hermite moments m^h_k for k=0,...,K_max.

    m^h_k = (1/sqrt(k!)) * sum_{j=0}^{k} He_k_coeffs[j] * mu*_j

    where He_k_coeffs[j] is the coefficient of x^j in He_k(x).

    Parameters
    ----------
    raw_mu : array, raw_mu[k] = E[X^k], length >= K_max + 1
    K_max  : maximum order; defaults to len(raw_mu) - 1

    Returns
    -------
    mh : array of length K_max+1:
         mh[k] = m^h_k = E[h_k(X*)]
    Also returns (m1, sigma) for use in standardization.
    """
    if K_max is None:
        K_max = len(raw_mu) - 1

    std_mu, m1, sigma = standardize_moments(raw_mu)

    mh = np.zeros(K_max + 1)
    mh[0] = 1.0  # h_0 = 1, E[1] = 1

    for k in range(1, K_max + 1):
        coeffs = He_coeffs(k)  # monomial coefficients of He_k
        # E[He_k(X*)] = sum_j coeffs[j] * mu*[j]
        he_expect = sum(coeffs[j] * std_mu[j] for j in range(len(coeffs)))
        mh[k] = he_expect / sqrt(factorial(k))

    return mh, m1, sigma


def verify_hermite_moments(mh: np.ndarray) -> None:
    """
    Sanity checks on Hermite moments of a standardized distribution:
      mh[0] = 1      (trivially true)
      mh[1] = 0      (mean of X* is 0)
      mh[2] = 0      (variance of X* is 1, so E[He_2(X*)] = E[X*^2 - 1] = 1-1 = 0)
    """
    tol = 1e-8
    assert abs(mh[0] - 1.0) < tol, f"mh[0] = {mh[0]} ≠ 1"
    if len(mh) > 1:
        assert abs(mh[1]) < tol,         f"mh[1] = {mh[1]} ≠ 0 (mean not zero)"
    if len(mh) > 2:
        assert abs(mh[2]) < tol,         f"mh[2] = {mh[2]} ≠ 0 (variance not 1)"
