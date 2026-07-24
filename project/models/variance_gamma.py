"""
Variance Gamma (VG) model.

Characteristic function (Carr-Madan parametrization):
    phi(u) = exp(i*u*mu) * (1 / (1 - i*u*theta*nu + u^2*sigma^2*nu/2))^(1/nu)

Cumulant generating function (CGF):
    K(s) = mu*s - (1/nu)*log(1 - theta*nu*s - sigma^2*nu*s^2/2)

Cumulants derived analytically from CGF.

Parameters from Heston & Rossi (2016):
    mu=0, sigma=0.2, nu=2/3, theta=0
    → skewness=0, excess kurtosis = 3*nu = 2
"""

import numpy as np
from math import factorial
from typing import Dict


def characteristic_function(u: np.ndarray, params: Dict) -> np.ndarray:
    """
    VG characteristic function evaluated at array u.

    phi(u) = exp(i*u*mu) * (1 / (1 - i*u*theta*nu + u^2*sigma^2*nu/2))^(1/nu)
    """
    mu    = params["mu"]
    sigma = params["sigma"]
    nu    = params["nu"]
    theta = params["theta"]
    u = np.asarray(u, dtype=complex) #converte u in un vettore di numeri complessi
    return (np.exp(1j * u * mu) #calcola la funzione caratteristica per ogni punto della griglia u
            * (1.0 / (1.0 - 1j * u * theta * nu + u**2 * sigma**2 * nu / 2.0))**(1.0 / nu))


def log_cgf(s: float, params: Dict) -> float:
    """
    Log of the moment generating function  E[exp(s*X)] = exp(K(s)).
    K(s) = mu*s - (1/nu)*log(1 - theta*nu*s - sigma^2*nu*s^2/2) -> dimostrazione su appunti
    Valid for s in the strip where the argument of log is positive.
    """
    mu    = params["mu"]
    sigma = params["sigma"]
    nu    = params["nu"]
    theta = params["theta"]
    arg = 1.0 - theta * nu * s - (sigma**2 * nu * s**2) / 2.0 #argomento della funzione logaritmica più facile da valutare e trovare errori se negativo
    if arg <= 0:
        raise ValueError(f"CGF argument non-positive at s={s}")
    return mu * s - (1.0 / nu) * np.log(arg) #K(s)


def cumulants(params: Dict, max_order: int = 30) -> np.ndarray: #calcola i cumulanti fino al 30esimo cumulante
    """
    Compute VG cumulants kappa_k for k=1,...,max_order analytically.

    From the CGF  K(s) = mu*s - (1/nu)*log(1 - nu*theta*s - nu*sigma^2*s^2/2):
    Let  f(s) = 1 - nu*theta*s - nu*sigma^2*s^2/2 è ciò che prima abbiamo definito arg in log_cgf

    The k-th cumulant is  K^{(k)}(0).

    We use the Faà di Bruno / iterated differentiation of log(f) approach:
    d^k/ds^k [ -(1/nu)*log(f(s)) ] evaluated at s=0.

    Since f is a degree-2 polynomial, d^n f/ds^n = 0 for n >= 3.
    The derivatives of log(f) can be computed via the recursion for
    log-derivatives of a polynomial (Bell polynomial approach).
    """
    mu    = params["mu"]
    sigma = params["sigma"]
    nu    = params["nu"]
    theta = params["theta"]

    # f(s) = 1 - nu*theta*s - nu*sigma^2/2 * s^2
    # f'(0)  = -nu*theta = 0 pk theta = 0 per definizione
    # f''(0) = -nu*sigma^2 = 0.0266 pk sigma = 0.2 per definizione
    # f^{(k)}(0) = 0  for k >= 3
    f0  =  1.0
    fp  = -nu * theta         # f'(0)
    fpp = -nu * sigma**2      # f''(0)

    # Derivatives of log(f) at s=0: D_k = d^k log(f)/ds^k |_{s=0}
    # Use recurrence:  D_1 = f'/f,  then D_{k+1} = (f^{(k+1)} - sum_{j=1}^{k} C(k,j-1)*D_j*f^{(k+1-j)})/f
    # More directly, from the identity:  (log f)' * f = f'
    # Differentiating k times:  sum_{j=0}^{k} C(k,j) D_{j+1} f^{(k-j)} = f^{(k+1)}
    # where D_0 = log(f)|_0 (not needed) and we define D_j = d^j/ds^j log(f) |_0

    D = np.zeros(max_order + 1)  # D[k] = d^k log(f)/ds^k |_{s=0}

    def f_deriv(k):
        """k-th derivative of f at 0."""
        if k == 0:
            return f0
        elif k == 1:
            return fp
        elif k == 2:
            return fpp
        else:
            return 0.0

    # Recurrence: D[1]*f_deriv(0) = f_deriv(1)  → D[1] = fp
    # For k>=1:  D[k]*f0 + sum_{j=0}^{k-1} C(k-1,j) * D[j+1] * f_deriv(k-1-j) for j>=1
    # i.e. differentiate the equation (log f)' * f = f':
    # d^{k-1}/ds^{k-1} [ (log f)' * f ] = f^{(k)}
    # => sum_{j=0}^{k-1} C(k-1,j) * (log f)^{(j+1)} * f^{(k-1-j)} = f^{(k)}
    # => D[k] * f0 = f^{(k)} - sum_{j=0}^{k-2} C(k-1,j) * D[j+1] * f^{(k-1-j)}

    from math import comb
    for k in range(1, max_order + 1):
        rhs = f_deriv(k) #f^{(k)}(0) ovvero la derivata k-esima di f in s=0
        for j in range(0, k - 1):
            rhs -= comb(k - 1, j) * D[j + 1] * f_deriv(k - 1 - j) #conosce già D[j+1] pk D(1) è fp e dopo se lo ricorda e lo usa per calcolare il nuovo valore di rhs aggiornato e così via
        D[k] = rhs / f0 #D[k] è la derivata k-esima di log(f) in s=0 

    # K(s) = mu*s - (1/nu)*log(f(s))
    # K^{(k)}(0) = mu * (k==1) - (1/nu) * D[k]
    kappas = np.zeros(max_order + 1)
    for k in range(1, max_order + 1):
        kappas[k] = (mu if k == 1 else 0.0) - (1.0 / nu) * D[k] #formula del cumulante dove D[k] è la derivata k-esima di log(f) in s=0, dimostrazione su appunti

    return kappas  # kappas[k] = k-th cumulant, kappas[0] unused

def _cumulants_to_raw_moments(kappas: np.ndarray, max_order: int) -> np.ndarray:
    """
    Convert cumulants to raw moments via the recursive formula:
      mu_n = sum_{k=1}^{n} C(n-1, k-1) * kappa_k * mu_{n-k}
    where mu_0 = 1.
    """
    from math import comb
    mu = np.zeros(max_order + 1)
    mu[0] = 1.0
    for n in range(1, max_order + 1):
        s = 0.0
        for k in range(1, n + 1):
            s += comb(n - 1, k - 1) * kappas[k] * mu[n - k]
        mu[n] = s
    return mu

def raw_moments(params: Dict, max_order: int = 30) -> np.ndarray: #funzione finale che restituisce i momenti raw E[X^k] per k=0,...,max_order dai cumulanti
    """
    Compute raw moments E[X^k] for k=0,...,max_order from cumulants.
    Uses the moment-cumulant relation (recursive).
    Returns array mu[k] = E[X^k], mu[0]=1.
    """
    kappas = cumulants(params, max_order)
    return _cumulants_to_raw_moments(kappas, max_order)
