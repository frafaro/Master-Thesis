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
    x     : punti nei quali valutiamo la PDF
    cf    : characteristic function phi(u) da richiamare
    a, b  : domain [a, b] dove viene valutata la PDF
    N_cos : numero di termini della serie 2^12 = 4096

    Returns
    -------
    p : density values at x, shape (M,)
    """
    x = np.asarray(x, dtype=float)
    k = np.arange(N_cos) #crea un vettore di interi da 0 a N_cos-1
    u_k = k * np.pi / (b - a) #valori nei quali va valutata la funzione caratteristica cf

    # COS coefficients: F_k = (2/(b-a)) * Re[ phi(u_k) * exp(-i*u_k*a) ]
    phi_vals = cf(u_k) #cf viene da main.py (160-164) e viene richiamata per valutare la funzione caratteristica cf(u_k) in ogni punto di u_k quindi restituisce un vettore di 4096 valori complessi.
    F_k = np.real(phi_vals * np.exp(-1j * u_k * a)) #np.real restituisce la parte reale del prodotto tra phi_vals e np.exp(-1j * u_k * a) (implemento in seguito 2/(a-b))
    F_k[0] *= 0.5  # half weight for k=0, spiegazione su appunti in step 4

    # Evaluate sum_k F_k * cos(u_k * (x - a))
    # x: (M,),  u_k: (N_cos,)  → broadcast
    cos_mat = np.cos(np.outer(u_k, x - a))  # (N_cos, M)
    """
    costruzione della matrice dei coseni, si hanno due vettori: u_k di dimensione N_cos (4096) e x-a di dimensione M ovvero il numero di punti x nei quali valutiamo la PDF.
    con np.outer si costruisce la matrice (N_cos, M), mentre con np.cos viene calcolato il coseno in ogni piunto della matrice
    """
    p = (2.0 / (b - a)) * F_k @ cos_mat 

    """
    prodotto matriciale tra F_k di dimensione (1, N_cos) vedi riga 45, e cos_mat di dimensione (N_cos, M) quindi restituisce un vettore di dimensione M ovvero il numero di punti x nei quali valutiamo la PDF.
    """
    p = np.maximum(p, 0.0) #la densità non può essere negativa quindi si pone a 0 se negativo
    return p


def benchmark_fourier_coeffs(cos_log_p: np.ndarray, #definito in main.py (164-169) con il nome di log_p_cos_full
                              x: np.ndarray, #definito in main.py (164-169) con il nome di x_full
                              eval_basis: Callable,
                              nu_weight: np.ndarray, 
                              m1: float, sigma: float,
                              N: int) -> np.ndarray:
    """
    Estimate the exact Fourier coefficients (eq. 9)
    using the COS-estimated density.

    c_j = integral_I clr(p)(x) * phi_j(x*) * nu(x*)/sigma dx derivation on appunti in step 4

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
    #calcolo delle basi ortogonali di hermite e logistic.
    P = eval_basis(x_std)  # (N+1, M) #matrice delle basi polinomiali eval_basis valuta sia hermite che logistic (hermite_basis e logistic_basis). 

    # E_nu[log p] = integral log(p(x)) * nu(x*)/sigma dx
    # nu_weight is already nu(x*)/sigma evaluated at x
    E_log_p = np.trapz(cos_log_p * nu_weight, x) 

    # clr(p)(x) = log(p(x)) - E_nu[log p]
    clr_p = cos_log_p - E_log_p

    #calcolo Fourier coefficients
    # c_j = integral clr_p(x) * phi_j(x*) * nu(x*)/sigma dx
    c = np.zeros(N) #vettore di zeri con lunghezza N per poter confrontare con lo stesso numero di coef nei due vettori sennò non si può fare la differenza
    for j in range(1, N + 1):
        c[j - 1] = np.trapz(clr_p * P[j] * nu_weight, x) #integrale del prodotto tra clr_p, la j-esima base polinomiale e il peso nu_weight sui punti x eq(9)
    return c


def verify_cos_density(x: np.ndarray, p: np.ndarray,
                       raw_moments_true: np.ndarray, #
                       tol: float = 1e-4) -> bool:
    """
    Verify the COS density by checking:
      1. integral p(x) dx ≈ 1
      2. integral x * p(x) dx ≈ mu_1
      3. integral x^2 * p(x) dx ≈ mu_2
    """
    ok = True
    integ = np.trapz(p, x) #integrale della densità sui punti x
    if abs(integ - 1.0) > tol: #la tolleranza è fissata a 0,0001 quindi se l'integrale non è 1 con una tolleranza di 0,0001 allora non è verificata la normalizzazione
        print(f"  COS density normalization: {integ:.6f}")
        ok = False
    mu1_num = np.trapz(x * p, x)
    if abs(mu1_num - raw_moments_true[1]) > tol: #i raw_moments_true sono quelli delle distribuzioni vg, nig e heston quindi m1 è il primo momento e m2 è il secondo momento.
        print(f"  COS mean: {mu1_num:.6f} vs {raw_moments_true[1]:.6f}")
        ok = False
    mu2_num = np.trapz(x**2 * p, x)
    if abs(mu2_num - raw_moments_true[2]) > 10 * tol: #tolleranza è 0,001
        print(f"  COS E[X^2]: {mu2_num:.6f} vs {raw_moments_true[2]:.6f}")
        ok = False
    return ok
