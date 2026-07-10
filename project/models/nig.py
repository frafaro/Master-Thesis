"""
Normal Inverse Gaussian (NIG) model.

Characteristic function (Appendix A of Gambaro 2024):
    phi(u) = exp(i*u*mu*dt + (dt/kappa)*(1 - sqrt(1 + u^2*sigma^2*kappa - 2*i*u*theta*kappa)))

Cumulant generating function:
    K(s) = mu*s + (1/kappa)*(1 - sqrt(1 - 2*theta*kappa*s - sigma^2*kappa*s^2))

Parameters (Appendix A of Gambaro 2024):
    mu=0, theta=0.05, sigma=0.2, kappa=0.3, dt=1
"""

import numpy as np
from math import comb
from typing import Dict


def characteristic_function(u: np.ndarray, params: Dict) -> np.ndarray:
    """
    NIG characteristic function at array u.
    phi(u) = exp(i*u*mu*dt + (dt/kappa)*(1 - sqrt(1 + u^2*sigma^2*kappa - 2*i*u*theta*kappa)))
    """
    mu    = params["mu"]
    theta = params["theta"]
    sigma = params["sigma"]
    kappa = params["kappa"]
    dt    = params.get("dt", 1.0)
    u = np.asarray(u, dtype=complex)
    inner = 1.0 + u**2 * sigma**2 * kappa - 2j * u * theta * kappa
    # use principal branch of sqrt
    return np.exp(1j * u * mu * dt + (dt / kappa) * (1.0 - np.sqrt(inner)))


def log_cgf_deriv(s: float, params: Dict, order: int) -> float:
    """
    k-th derivative of K(s) = mu*s + (1/kappa)*(1 - sqrt(1 - 2*theta*kappa*s - sigma^2*kappa*s^2))
    evaluated at s=0, via the same recurrence as the VG case.
    """
    mu    = params["mu"]
    theta = params["theta"]
    sigma = params["sigma"]
    kappa = params["kappa"]

    # g(s) = 1 - 2*theta*kappa*s - sigma^2*kappa*s^2
    # K(s) = mu*s + (1/kappa)*(1 - sqrt(g(s)))
    # K^{(k)}(0) = mu*(k==1) + (1/kappa) * d^k/ds^k (1 - sqrt(g)) |_0
    #            = mu*(k==1) - (1/kappa) * d^k/ds^k sqrt(g) |_0

    g0   =  1.0
    gp   = -2.0 * theta * kappa       # g'(0)
    gpp  = -2.0 * sigma**2 * kappa    # g''(0)

    def g_deriv(k):
        if k == 0: return g0
        if k == 1: return gp
        if k == 2: return gpp
        return 0.0

    # Compute derivatives of sqrt(g) at s=0.
    # Let h = sqrt(g), so h^2 = g.
    # Differentiate: 2*h*h' = g',  then recursively for higher orders.
    # d^k (h^2)/ds^k = g^{(k)}
    # Using Leibniz rule on h*h:
    #   sum_{j=0}^{k} C(k,j) H[j]*H[k-j] = g_deriv(k)
    #   2*H[0]*H[k] + sum_{j=1}^{k-1} C(k,j)*H[j]*H[k-j] = g_deriv(k)
    #   H[k] = (g_deriv(k) - sum_{j=1}^{k-1} C(k,j)*H[j]*H[k-j]) / (2*H[0])

    H = np.zeros(order + 1)
    H[0] = np.sqrt(g0)  # = 1
    for k in range(1, order + 1):
        rhs = g_deriv(k)
        for j in range(1, k):
            rhs -= comb(k, j) * H[j] * H[k - j]
        H[k] = rhs / (2.0 * H[0])

    K_k = (mu if order == 1 else 0.0) - (1.0 / kappa) * H[order]
    return K_k


def cumulants(params: Dict, max_order: int = 30) -> np.ndarray:
    """Return array kappas[k] = k-th cumulant of NIG for k=1,...,max_order."""
    kappas = np.zeros(max_order + 1)
    for k in range(1, max_order + 1):
        kappas[k] = log_cgf_deriv(0.0, params, k)
    return kappas


def raw_moments(params: Dict, max_order: int = 30) -> np.ndarray:
    """Raw moments E[X^k] for k=0,...,max_order via moment-cumulant relation."""
    kappas = cumulants(params, max_order)
    return _cumulants_to_raw_moments(kappas, max_order)


def _cumulants_to_raw_moments(kappas: np.ndarray, max_order: int) -> np.ndarray:
    mu = np.zeros(max_order + 1)
    mu[0] = 1.0
    for n in range(1, max_order + 1):
        s = 0.0
        for k in range(1, n + 1):
            s += comb(n - 1, k - 1) * kappas[k] * mu[n - k]
        mu[n] = s
    return mu
