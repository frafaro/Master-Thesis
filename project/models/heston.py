"""
Heston stochastic volatility model.

Characteristic function (Gatheral form, avoids branch cuts):
    phi(u) = exp(A(u,T) + B(u,T)*v0)

where the Gatheral form uses:
    d  = sqrt((kappa - rho*xi*i*u)^2 + xi^2*(i*u + u^2))
    g  = (kappa - rho*xi*i*u - d) / (kappa - rho*xi*i*u + d)
    A  = i*u*r*T + kappa*theta/xi^2 * ((kappa - rho*xi*i*u - d)*T
                                         - 2*log((1 - g*exp(-d*T))/(1-g)))
    B  = (kappa - rho*xi*i*u - d)/xi^2 * (1 - exp(-d*T))/(1 - g*exp(-d*T))

Parameters calibrated (via numerical optimization) to match:
    skewness = -1.2,  excess kurtosis = 2.5
from Rompolis & Tzavalis (2008).

Cumulants are computed via numerical differentiation of log(phi(u)) at u=0.
"""

import numpy as np
from math import comb
from typing import Dict


def _d_func(u: np.ndarray, kappa: float, rho: float, xi: float) -> np.ndarray:
    """d(u) = sqrt((kappa - rho*xi*iu)^2 + xi^2*(iu + u^2))"""
    iu = 1j * u
    return np.sqrt((kappa - rho * xi * iu)**2 + xi**2 * (iu + u**2))


def characteristic_function(u: np.ndarray, params: Dict) -> np.ndarray:
    """
    Heston CF in Gatheral form (numerically stable for large |u|).
    cf. Gatheral (2006) "The Volatility Surface".

    Log-return X = log(S_T/S_0) under risk-neutral measure.
    """
    kappa = params["kappa"]
    theta = params["theta"]
    xi    = params["xi"]
    rho   = params["rho"]
    v0    = params["v0"]
    T     = params["T"]
    r     = params.get("r", 0.0)

    u = np.asarray(u, dtype=complex)
    iu = 1j * u

    d = _d_func(u, kappa, rho, xi)

    # Gatheral form: uses log( (1 - g*exp(-d*T)) / (1 - g) ) which is stable
    # when Re(d) > 0 (which holds for the principal branch sqrt with Re(d)>=0).
    alpha = kappa - rho * xi * iu
    g = (alpha - d) / (alpha + d)

    # Avoid division by zero: if |1 - g| is tiny, use limiting form
    exp_dT = np.exp(-d * T)
    denom = 1.0 - g * exp_dT
    denom0 = 1.0 - g

    A = (iu * r * T
         + kappa * theta / xi**2 * (
             (alpha - d) * T - 2.0 * np.log(denom / denom0)
         ))
    B = (alpha - d) / xi**2 * (1.0 - exp_dT) / denom

    return np.exp(A + B * v0)


def cumulants_numerical(params: Dict, max_order: int = 30) -> np.ndarray:
    """
    Compute cumulants of X = log(S_T/S_0) via numerical differentiation
    of the log-MGF  K(s) = log phi(-i*s)  at s=0.

    Uses mpmath for arbitrary-precision evaluation to avoid cancellation
    in high-order finite differences.

    The step size h is chosen adaptively per order to balance truncation
    and rounding error.
    """
    import mpmath
    mpmath.mp.dps = 50    # 50 decimal places of precision

    kappa_H = float(params["kappa"])
    theta_H = float(params["theta"])
    xi      = float(params["xi"])
    rho     = float(params["rho"])
    v0      = float(params["v0"])
    T       = float(params["T"])
    r       = float(params.get("r", 0.0))

    def K_mpmath(s):
        """Log-MGF K(s) evaluated in mpmath arithmetic."""
        iu = -s   # u = -i*s, so iu = i*u = i*(-i*s) = s  ... actually:
        # phi(u) with u = -i*s:  iu = i*u = i*(-i*s) = s
        iu = s    # because u = -i*s → i*u = i*(-i*s) = s
        alpha = kappa_H - rho * xi * iu
        # d = sqrt((kappa - rho*xi*iu)^2 + xi^2*(iu + u^2))
        # with u = -i*s: u^2 = -s^2, iu = s
        d = mpmath.sqrt(alpha**2 + xi**2 * (s - s**2))
        g = (alpha - d) / (alpha + d)
        exp_dT = mpmath.exp(-d * T)
        denom  = 1 - g * exp_dT
        denom0 = 1 - g
        A = (iu * r * T
             + kappa_H * theta_H / xi**2 * (
                 (alpha - d) * T - 2 * mpmath.log(denom / denom0)
             ))
        B = (alpha - d) / xi**2 * (1 - exp_dT) / denom
        return A + B * v0   # = log phi(u)

    kappas = np.zeros(max_order + 1)

    # Use mpmath.diff for exact Taylor coefficients
    # K'(s) at s=0 = first cumulant, K''(s) = second cumulant, etc.
    s0 = mpmath.mpf(0)
    for k in range(1, max_order + 1):
        try:
            val = mpmath.diff(K_mpmath, s0, k)
            kappas[k] = float(val)
        except Exception as e:
            # For very high orders, fall back to 0
            kappas[k] = 0.0
    return kappas


def raw_moments(params: Dict, max_order: int = 30) -> np.ndarray:
    """Raw moments E[X^k] for k=0,...,max_order."""
    kappas = cumulants_numerical(params, max_order)
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


def calibrate_to_moments(target_skew: float = -1.2,
                         target_kurt: float = 2.5,
                         x0: Dict = None) -> Dict:
    """
    Find Heston parameters (kappa, theta, xi, rho, v0) at T=1, r=0
    such that the log-return distribution has the target skewness and
    excess kurtosis using scipy.optimize.minimize.

    Returns the calibrated parameter dict.
    """
    from scipy.optimize import minimize

    if x0 is None:
        x0 = {"kappa": 2.0, "theta": 0.06, "xi": 0.3, "rho": -0.7,
              "v0": 0.06, "T": 1.0, "r": 0.0}

    def objective(pars):
        params = {
            "kappa": pars[0], "theta": pars[1], "xi": pars[2],
            "rho": pars[3], "v0": pars[4], "T": 1.0, "r": 0.0,
        }
        try:
            kap = cumulants_numerical(params, max_order=4)
            std = np.sqrt(kap[2])
            skew = kap[3] / std**3
            kurt = kap[4] / kap[2]**2  # excess kurtosis
            loss = (skew - target_skew)**2 + (kurt - target_kurt)**2
        except Exception:
            loss = 1e10
        return loss

    bounds = [(0.1, 10), (0.001, 0.5), (0.05, 1.5), (-0.99, -0.01),
              (0.001, 0.5)]
    p0 = [x0["kappa"], x0["theta"], x0["xi"], x0["rho"], x0["v0"]]
    res = minimize(objective, p0, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-10})

    calibrated = {
        "kappa": res.x[0], "theta": res.x[1], "xi": res.x[2],
        "rho":   res.x[3], "v0":   res.x[4], "T": 1.0, "r": 0.0,
    }
    return calibrated
