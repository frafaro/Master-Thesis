"""
Build and solve the linear system A_N * c_hat = b_N  (eq. 15, Gambaro 2024).

Components:
  - Delta_{p,q,r}:  triple integral of UNNORMALIZED Hermite polynomials
                    = integral He_p He_q He_r omega dx
                    = p! q! r! / (b-p)!(b-q)!(b-r)!   (b = (p+q+r)/2)
  - A~_N[i,j]:  = sum_{k=0}^{i+j-2}  Delta_{i-1,j-1,k} / sqrt((i-1)!(j-1)!k!)  *  mh[k]
                  = E_p[ h_{i-1}(X*) * h_{j-1}(X*) ]
                  using NORMALIZED moments  mh[k] = E_p[h_k(X*)] = E_p[He_k(X*)] / sqrt(k!)
  - A_N[i,n]:   = sum_{j=1}^{n} sqrt(j) * Q[n,j] * A~_N[i,j]
  - b_N[i]:     = -sqrt(i-1) * mh[i-2]   (1-based index i=1,...,N)

Normalization convention (consistent across the entire code-base):
  h_k = He_k / sqrt(k!)           — NORMALIZED Hermite polynomial
  mh[k] = E_p[h_k(X*)]            — NORMALIZED Hermite moment  (stored in mh array)
  Delta_{p,q,r}                   — uses UNNORMALIZED He polynomials
  A~_N[i,j] factor = 1/sqrt(p!q!r!) — converts Delta back to normalized inner product

A~_N reduces to the identity matrix when p = omega (Gaussian), because
  Delta(p,p,0)/sqrt(p!p!) = p!/p! = 1  (only k=0 contributes for Gaussian mh).

Reference for Delta: Erdelyi et al. (1953) / Grad (1949):
  integral He_p He_q He_r omega dx = p! q! r! / (b-p)!(b-q)!(b-r)!
  with b = (p+q+r)/2, if p+q+r even and b >= max(p,q,r); else 0.
"""

import numpy as np
from math import factorial, comb, sqrt, lgamma
from scipy.linalg import solve
from typing import Tuple


def delta_coeff(p: int, q: int, r: int) -> float:
    """
    Delta_{p,q,r}: triple Hermite integral coefficient.
    = p!*q!*r! / ((b-p)!*(b-q)!*(b-r)!)  if p+q+r even, b=(p+q+r)/2 >= p,q,r
    = 0  otherwise.

    Computed in log-space to avoid overflow for large indices.
    """
    s = p + q + r
    if s % 2 != 0: #controlla se la somma è pari altrimenti delta vale 0
        return 0.0
    b = s // 2 #calcola il valore di b
    if b < p or b < q or b < r: #controlla se b è minore di p, q o r altrimenti delta vale 0
        return 0.0
    # log(p!*q!*r! / (b-p)!*(b-q)!*(b-r)!)
    log_val = (lgamma(p + 1) + lgamma(q + 1) + lgamma(r + 1)
               - lgamma(b - p + 1) - lgamma(b - q + 1) - lgamma(b - r + 1)) #uso proprietà dei logaritmi: log(a*b) = log(a) + log(b) e log(a/b) = log(a) - log(b)
    return np.exp(log_val) #restituisce il valore di delta facendo l'esponenziale


def build_A_tilde(N: int, mh: np.ndarray) -> np.ndarray: #dimostrazione su appunti step8
    """
    Build the N×N matrix A~_N.

    Correct formula (uses NORMALIZED moments mh[k] = E_p[h_k(X*)]):
        A~_N[i,j] = sum_{k=0}^{i+j-2}  Delta_{i-1,j-1,k} / sqrt((i-1)! (j-1)! k!)  *  mh[k]
                  = E_p[ h_{i-1}(X*) * h_{j-1}(X*) ]

    The denominator sqrt(p! q! r!) converts the unnormalized Delta integral
    (for He_p He_q He_r) into the normalized inner product (for h_p h_q h_r = He_p/sqrt(p!) …).

    where i,j = 1,...,N  (Python: 0,...,N-1 via offset i_py = i-1, j_py = j-1).
    """
    At = np.zeros((N, N))
    for i in range(1, N + 1): #ciclo per le righe
        for j in range(1, N + 1): #ciclo per le colonne
            p_idx = i - 1 #indice della riga
            q_idx = j - 1 #indice della colonna
            s = 0.0
            for k in range(0, i + j - 1):   # k = 0,...,i+j-2
                if k >= len(mh):
                    break
                d = delta_coeff(p_idx, q_idx, k) #calcola il valore di delta usando la funzzione precedente con gli indici della riga e della colonna
                if d == 0.0:
                    continue
                norm = sqrt(factorial(p_idx) * factorial(q_idx) * factorial(k)) #denominaore della formula
                s += (d / norm) * mh[k] #somma dei prodotti tra delta normalizzato e momento centrato, mh viene da hermite_moments.py
            At[i - 1, j - 1] = s #salva il valore di s nella matrice At con gli indici della riga e della colonna
    return At


def build_A(N: int, Q: np.ndarray, At: np.ndarray) -> np.ndarray:
    """
    Build the full N×N system matrix A_N.
    A_N[i,n] = sum_{j=0}^{n} sqrt(j) * Q[n,j] * A~_N[i,j]
    with paper indices i,n = 1,...,N (Python: i_py=i-1, n_py=n-1).
    Q is the (N+1)×(N+1) change-of-basis matrix (0-indexed).
    At is the N×N A~_N matrix (0-indexed, rows/cols 0,...,N-1 ↔ paper 1,...,N).
    """
    A = np.zeros((N, N))
    for i in range(1, N + 1): #ciclo per le righe
        for n in range(1, N + 1): #ciclo per le colonne
            s = 0.0
            for j in range(1, n + 1):   # j=1,...,n  (j=0 contributes sqrt(0)=0) 
                s += sqrt(j) * Q[n, j] * At[i - 1, j - 1] #somma dei prodotti tra la radice quadrata di j, la matrice Q e la matrice At con gli indici della riga e della colonna
            A[i - 1, n - 1] = s #salva il valore di s nella matrice A con gli indici della riga e della colonna
    return A


def build_b(N: int, mh: np.ndarray) -> np.ndarray:
    """
    Build the N-vector b_N.

    From Gambaro (2024) eq. (15) / user derivation (image 1):
        b_N[i] = -sqrt(i-1) * mh[i-2]   for i=1,...,N  (paper 1-based)

    The index is i-2 (not i-1): this is the corrected formula that recovers
    the exact Fourier coefficients for the Gaussian reference distribution
    (c_2 = -1/sqrt(2), c_j=0 for j != 2) and gives non-zero ĉ for even-indexed
    components in symmetric distributions.

    Special case i=1: sqrt(i-1)=0, so b[1]=0 regardless.

    Python mapping: b[i_py] = -sqrt(i_py) * mh[i_py - 1]
    with b[0] = 0 (i_py=0 → sqrt(0)=0).
    """
    b = np.zeros(N)
    for i_py in range(N):
        i = i_py + 1          # paper 1-based index
        if i <= 1: #controlla se i è minore o uguale a 1 altrimenti b vale 0
            b[i_py] = 0.0     # sqrt(i-1)=0, index mh[-1] undefined
        else: #se i è maggiore di 1 allora b vale la formula
            b[i_py] = -sqrt(i - 1) * mh[i - 2]   # mh[i-2], not mh[i-1], il vettore mh viene da hermite_moments.py
    return b


def solve_system(A: np.ndarray, b: np.ndarray,
                 cond_threshold: float = 1e14) -> Tuple[np.ndarray, float]:
    """
    Solve A @ c = b.
    Uses lstsq if the system is ill-conditioned (cond > cond_threshold)
    to avoid divergence; otherwise uses LU decomposition.
    Returns (c_hat, cond_number).
    """
    import warnings
    cond = np.linalg.cond(A) #calcolo il numero di condizionamento della matrice A, 
    if cond > cond_threshold: #se il numero di condizionamento è maggiore della tolleranza allora si usa il metodo dei minimi quadrati, succede principalmente per la base logistic che ha un numero di condizionamento elevato
        c_hat, _, _, _ = np.linalg.lstsq(A, b, rcond=None) #risolve il sistema lineare usando il metodo dei minimi quadrati. c_hat = arg min ||A @ c - b||_2
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c_hat = solve(A, b) #risolve il sistema lineare usando la decomposizione LU. c_hat = A^-1 @ b quando il sistema è ben condizionato
    return c_hat, cond


def verify_residual(A: np.ndarray, c_hat: np.ndarray, b: np.ndarray, 
                    tol: float = 1e-8) -> bool:
    """Check ||A @ c_hat - b|| / ||b|| < tol.
       guardo se c_hat soddisfa davvero il sistema lineare. idealmente si avrebbe A @ c_hat - b = 0 

    """
    res = np.linalg.norm(A @ c_hat - b) #calcolo la norma del residuo A @ c_hat - b
    nb  = np.linalg.norm(b) #calcolo la norma del vettore b
    rel = res / nb if nb > 1e-15 else res #calcolo il residuo relativo
    if rel > tol: #se il residuo relativo è maggiore della tolleranza allora il sistema non è soddisfatto
        print(f"  Linear system residual: {rel:.2e}  (tol={tol:.1e})")
        return False
    return True
