"""
Distance metrics between the approximated density p^_N and the true density p.

Equations from Gambaro (2024):
  (17) d2_coeff: sum_{j=1}^{N} (c^_j - c_j)^2
  (18) d_A:      sqrt( integral (clr(p^_N) - clr(p))^2 * nu dx )
  (19) d2_log:   sqrt( integral (log(p^_N) - log(p))^2 dx )
  (20) d1:       integral |p^_N - p| dx
  (21) d2:       sqrt( integral (p^_N - p)^2 dx )

All integrals are computed via the trapezoidal rule on the grid x.
"""

import numpy as np
from typing import Optional


def d2_coeff_estim(c_hat: np.ndarray, c_exact: np.ndarray, #la f. sta ricevendo i coef stimati, i coef esatti e il max_j opzionale  per poter calcolare la distanza solo tra i primi max_j coefficienti
                   max_j: Optional[int] = None) -> float:
    """
    Estimation-only coefficient distance (no truncation error), for Figures 1-3.

    sqrt( Σ_{j=1}^{K} (ĉj - cj)² )   where K = min(max_j, N) if max_j given, else N.

    This is what the paper plots on the y-axis ("square root distance"):
      - "all coefficients":   max_j=None  → uses all N components
      - "first 6 coefficients": max_j=6  → uses only j=1..min(6,N) components
    """
    N = len(c_hat) #numero di coefficienti stimati
    K = min(max_j, N) if max_j is not None else N #numero di coef da usare per il calcolo della distanza
    diff = c_hat[:K] - c_exact[:K] #differenza tra i coef stimati e i coef esatti
    return float(np.sqrt(np.sum(diff**2))) #calcolo della distanza 


def d2_coeff(c_hat: np.ndarray, c_exact: np.ndarray,
             c_exact_full: Optional[np.ndarray] = None) -> float:
    """
    Full coefficient distance (eq. 17):
        d₂(ĉN, c) = sqrt( Σ_{j=1}^N (ĉj - cj)² + Σ_{j>N} cj² )

    where ĉN_j = ĉj for j≤N, ĉN_j = 0 for j>N. #se il vettore vero è più lungo dei coef stimati, si pone a 0 i coef che superano N

    Parameters
    ----------
    c_hat        : estimated coefficients, length N (0-indexed, j=1..N)
    c_exact      : benchmark coefficients for j=1..N (may be shorter, zero-padded)
    c_exact_full : all benchmark coefficients j=1..N_MAX (for truncation error);
                   if None, only the estimation component is included.

    Returns
    -------
    d2 : scalar (the distance, already square-rooted)
    """
    N = len(c_hat) #coef stimati
    N_ex = len(c_exact) #coef esatti
    c_ex = np.zeros(N) #vettore di zeri con lunghezza N per poter confrontare con lo stesso numero di coef nei due vettori sennò non si può fare la differenza
    c_ex[:min(N, N_ex)] = c_exact[:min(N, N_ex)] #assegna i coef esatti ai coef stimati fino al minimo tra N e N_ex
    estimation_sq = np.sum((c_hat - c_ex)**2)

    truncation_sq = 0.0 
    if c_exact_full is not None and len(c_exact_full) > N: #se i coef esatti risultano maggiori di N allora si calcola la distanza tra i coef stimati e i coef esatti più lunghi ponendo a 0 i coef che superano N
        truncation_sq = float(np.sum(c_exact_full[N:]**2)) #prende i coef a partire da N+1 ovvero quelli che superano N

    return float(np.sqrt(estimation_sq + truncation_sq)) #calcolo della distanza con errore di troncamento


def d_aitchison(x: np.ndarray,
                log_p_hat: np.ndarray,
                log_p: np.ndarray,
                nu_weight: np.ndarray) -> float:
    """
    Aitchison distance (eq. 18):
    d_A = sqrt( integral (clr(p^_N) - clr(p))^2 * nu dx )

    Since clr(f) = log(f) - E_nu[log(f)], we have:
    clr(p^_N) - clr(p) = (log(p^_N) - log(p)) - E_nu[log(p^_N) - log(p)]

    The E_nu term shifts the difference by a constant, which integrates to
    a constant-squared contribution. We compute it explicitly.

    Parameters
    ----------
    x          : integration grid (original scale), shape (M,)
    log_p_hat  : log(p^_N(x)), shape (M,)
    log_p      : log(p(x)) from COS, shape (M,)
    nu_weight  : nu(x*)/sigma evaluated at x, shape (M,)  (normalized: integral = 1)

    Returns
    -------
    d_A : scalar
    """
    diff_log = log_p_hat - log_p #differenza tra la densità stimata e la densità esatta. 
    E_diff = np.trapz(diff_log * nu_weight, x) #integrale della differenza tra la densità stimata e la densità esatta pesata per il peso nu(x*)/sigma.
    clr_diff = diff_log - E_diff
    return float(np.sqrt(np.trapz(clr_diff**2 * nu_weight, x))) #calcolo della distanza usando la formula di trapezio.


def d2_log(x: np.ndarray,
           log_p_hat: np.ndarray,
           log_p: np.ndarray) -> float:
    """
    L2 distance between log-densities (eq. 19):
    d2_log = sqrt( integral (log(p^_N) - log(p))^2 dx )
    Not weighted by nu; allows comparison across bases.
    """
    return float(np.sqrt(np.trapz((log_p_hat - log_p)**2, x)))


def d1(x: np.ndarray, p_hat: np.ndarray, p: np.ndarray) -> float:
    """L1 distance (eq. 20): integral |p^_N - p| dx."""
    return float(np.trapz(np.abs(p_hat - p), x))


def d2(x: np.ndarray, p_hat: np.ndarray, p: np.ndarray) -> float:
    """L2 distance (eq. 21): sqrt( integral (p^_N - p)^2 dx )."""
    return float(np.sqrt(np.trapz((p_hat - p)**2, x)))


def all_distances(x: np.ndarray,
                  p_hat: np.ndarray,
                  p: np.ndarray,
                  nu_weight: np.ndarray) -> dict:
    """
    Compute all four density distances at once.

    Parameters
    ----------
    x          : grid (original scale)
    p_hat      : approximated density on x
    p          : true (COS) density on x
    nu_weight  : weight nu(x*)/sigma on x (integrates to 1)

    Returns
    -------
    dict with keys: 'aitchison', 'log_l2', 'l1', 'l2'
    """
    # Clip to avoid log(0)
    eps = 1e-300
    lp_hat = np.log(np.maximum(p_hat, eps))
    lp     = np.log(np.maximum(p, eps))
    return {
        "aitchison": d_aitchison(x, lp_hat, lp, nu_weight),
        "log_l2":    d2_log(x, lp_hat, lp),
        "l1":        d1(x, p_hat, p),
        "l2":        d2(x, p_hat, p),
    }
