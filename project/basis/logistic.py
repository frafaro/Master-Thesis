"""
Logistic polynomial basis  (Heston & Rossi 2016).

Polynomials L_0,...,L_N orthonormal w.r.t. the standard logistic weight:
    nu_L(x) = exp(-x) / (1 + exp(-x))^2  (mean=0, variance=pi^2/3)

Algorithm:
  1. Stieltjes/Lanczos algorithm on a Gauss-Legendre quadrature grid to
     compute the three-term recurrence coefficients (alpha_k, beta_k)
     for the MONIC polynomials pi_k:
       pi_{k+1}(x) = (x - alpha_k)*pi_k(x) - beta_k*pi_{k-1}(x)
  2. The NORMALIZED polynomials L_k = pi_k / ||pi_k|| satisfy:
       L_{k+1}(x) = [(x - alpha_k)*L_k(x) - sqrt(beta_k)*L_{k-1}(x)] / sqrt(beta_{k+1})
     where the sqrt(beta_{k+1}) denominator normalizes L_{k+1} and
     sqrt(beta_k) accounts for the norm ratio ||pi_k|| / ||pi_{k-1}||.
  3. Evaluation at any x via the recurrence (stable for all N).

The change-of-basis matrix Q_n[k,j] = <L_k, h_j>_Gaussian is computed
separately in matrices/change_of_basis.py via numerical inner products,
bypassing the numerically unstable monomial coefficient representation.

Note: For symmetric nu_L, alpha_k = 0 for all k (verified numerically).
"""

import numpy as np
from math import sqrt
from typing import Tuple

"""
la quadratura serve per calcolare le norme ||pi_k||^2 = integral pi_k(x)^2 nu_L(x) dx, da questi si riva beta_k = ||pi_k||^2 / ||pi_{k-1}||^2
questa formula sostituisce l'integrale analitico con una sommatoria pesata con i pesi di Gauss-Legendre.
i pesi di Gauss-Legendre sono i pesi w_gl, i nodi sono i punti t.
"""
_INTEG_LIM = 30.0    #limite di integrazione per la quadratura 
_N_QUAD    = 3000    # Gauss-Legendre points


def logistic_weight(x: np.ndarray) -> np.ndarray:
    """
    Restituisce il peso logistico standard nu_L(x) = sech^2(x/2) / 4.
    la riscrivo in funzione di cosh(x/2) infatti vale la relazione sech^2(t) = 1/cosh^2(t)
    """
    x = np.asarray(x, dtype=float) #x è un vettore contenente i punti nei quali si vuole valutare la funzione peso.
    return 1.0 / (4.0 * np.cosh(x / 2.0)**2) #restituisce il valore di nu_L(x) per ogni punto della griglia x


def _make_quad_grid(n_quad: int = _N_QUAD, lim: float = _INTEG_LIM): #n_quad: numero di punti della griglia, lim: limite di integrazione
    """
    crea la griglia di punti e pesi che useremo per approssimare: ||pi_k||^2 = integral pi_k(x)^2 nu_L(x) dx
    
    La quadratura di Gauss-Legendre originale permette di calcolare un integrale su [-1,1]: ∫[-1,1] f(t) dt ≈ Σ_i w_i^GL f(t_i)
    
    Noi però dobbiamo calcolare un integrale sulla variabile x nell'intervallo [-lim, lim] con il peso logistico: ∫[-lim,lim] f(x) nu_L(x) dx
    dove: nu_L(x) = e^(-x)/(1+e^(-x))² = 1/4 sech²(x/2)

    Per passare dall'intervallo [-1,1] all'intervallo [-lim,lim] facciamo il cambio di variabile:x = lim * t
    quindi: dx = lim * dt

    sostituendo nell'integrale otteniamo:
    ∫[-lim,lim] f(x)nu_L(x)dx = ∫[-1,1] f(lim*t)nu_L(lim*t) lim dt

    Ora possiamo applicare Gauss-Legendre:
    ≈ Σ_i w_i^GL * lim * f(x_i) * nu_L(x_i)

    Quindi: w_i = w_i^GL * lim * nu_L(x_i)

    """
    from numpy.polynomial.legendre import leggauss #Importiamo la funzione che calcola i nodi e pesi della quadratura di Gauss-Legendre.
    t, w_gl = leggauss(n_quad)          #calcola i nodi e pesi della quadratura di Gauss-Legendre.
    x = lim * t                         #moltiplica i nodi per il limite di integrazione per ottenere la griglia di punti.
    w = w_gl * lim * logistic_weight(x) #moltiplica i pesi per il limite di integrazione e la funzione peso per ottenere i pesi della griglia.
    return x, w


_QUAD_CACHE = {} #Serve per evitare di ricalcolare sempre la stessa griglia.

def _get_quad(n_quad: int = _N_QUAD):
    if n_quad not in _QUAD_CACHE:
        _QUAD_CACHE[n_quad] = _make_quad_grid(n_quad)
    return _QUAD_CACHE[n_quad] #restituisce la griglia di punti e pesi calcolata


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
    x, w = _get_quad() #restituisce la griglia di punti e pesi calcolata prima

    # Initialize the recurrence
    pi_km1 = np.zeros_like(x)            # pi_{-1} = 0 (polinomio di grado -1)
    pi_k   = np.ones_like(x)             # pi_0 = 1 (monic)
    norm_km1 = 1.0                       # valore di partenza per evitare di dividere per zero in beta_0 = ||pi_1||^2 / ||pi_{-1}||^2
    norm_k   = float(w @ pi_k**2)        # ||pi_0||^2 = 1 , 

    alpha = np.zeros(N + 1)
    beta  = np.zeros(N + 1)

    for k in range(N): #inizia il ciclo for e si calcola alpha_k e beta_k per ogni grado k del polinomio
        alpha_k = float(w @ (x * pi_k**2)) / norm_k
        alpha[k] = alpha_k
        beta_k   = norm_k / norm_km1 if k > 0 else norm_k  # beta_0 = ||pi_0||^2
        beta[k]  = beta_k

        pi_kp1 = (x - alpha_k) * pi_k - (norm_k / norm_km1 if k > 0 else 0.0) * pi_km1 #calcola il polinomio di grado k+1 non normalizzato
        norm_kp1 = float(w @ pi_kp1**2) #calcola la norma del polinomio di grado k+1 non normalizzato così lo possiamo utilizzare per il calcolo di beta_k+1

        pi_km1   = pi_k.copy() #copia il polinomio di grado k in pi_km1
        pi_k     = pi_kp1     #assegna il polinomio di grado k+1 a pi_k
        norm_km1 = norm_k     #assegna la norma del polinomio di grado k a norm_km1
        norm_k   = norm_kp1   #assegna la norma del polinomio di grado k+1 a norm_k

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
    P = np.zeros((N + 1, n)) #come per eval_hermite, crea una matrice di zeri con dimensione (N+1) x n (righe x colonne), dove le righe sono i polinomi di grado 0, 1, 2, ..., N e le colonne sono i punti della griglia x.

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
    """
    E_L[X^k] via Gauss-Legendre quadrature.
    
    """
    x, w = _get_quad()
    return float(w @ x**k) #calcola il valore atteso ovvero i momenti della variabile casuale X^k
