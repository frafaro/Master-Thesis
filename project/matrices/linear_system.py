"""
Build and solve the linear system A_N * c_hat = b_N  (eq. 15, Gambaro 2024).

Components:
  - Delta_{p,q,r}:  combinatorial triple-Hermite-integral coefficient
  - A~_N[i,j]:  = sum_{k=0}^{i+j-2} (1/k!) * Delta_{i-1,j-1,k} * mh[k]
  - A_N[i,n]:   = sum_{j=0}^{n} sqrt(j) * Q[n,j] * A~_N[i,j]
  - b_N[i]:     = -sqrt(i-1) * mh[i-1]   (1-based index i=1,...,N)

All arrays use 0-based Python indexing internally; the paper's 1-based
formulas are translated as: paper index i  →  Python index i-1.

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
    if s % 2 != 0:
        return 0.0
    b = s // 2
    if b < p or b < q or b < r:
        return 0.0
    # log(p!*q!*r! / (b-p)!*(b-q)!*(b-r)!)
    log_val = (lgamma(p + 1) + lgamma(q + 1) + lgamma(r + 1)
               - lgamma(b - p + 1) - lgamma(b - q + 1) - lgamma(b - r + 1))
    return np.exp(log_val)


def build_A_tilde(N: int, mh: np.ndarray) -> np.ndarray:
    """
    Build the N×N matrix A~_N.
    A~_N[i,j] = sum_{k=0}^{i+j-2} (1/k!) * Delta_{i-1,j-1,k} * mh[k]
    where i,j = 1,...,N  (Python: 0,...,N-1 via offset i_py = i-1, j_py = j-1).
    The paper's mh[k] = m^h_k is provided as array mh (0-indexed: mh[k]).
    """
    At = np.zeros((N, N))
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            s = 0.0
            for k in range(0, i + j - 1):   # k = 0,...,i+j-2
                if k >= len(mh):
                    break
                fk = float(factorial(k))
                d = delta_coeff(i - 1, j - 1, k)
                if d == 0.0:
                    continue
                s += (1.0 / fk) * d * mh[k]
            At[i - 1, j - 1] = s
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
    for i in range(1, N + 1):
        for n in range(1, N + 1):
            s = 0.0
            for j in range(1, n + 1):   # j=1,...,n  (j=0 contributes sqrt(0)=0)
                s += sqrt(j) * Q[n, j] * At[i - 1, j - 1]
            A[i - 1, n - 1] = s
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
        if i <= 1:
            b[i_py] = 0.0     # sqrt(i-1)=0, index mh[-1] undefined
        else:
            b[i_py] = -sqrt(i - 1) * mh[i - 2]   # mh[i-2], not mh[i-1]
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
    cond = np.linalg.cond(A)
    if cond > cond_threshold:
        c_hat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c_hat = solve(A, b)
    return c_hat, cond


def verify_residual(A: np.ndarray, c_hat: np.ndarray, b: np.ndarray,
                    tol: float = 1e-8) -> bool:
    """Check ||A @ c_hat - b|| / ||b|| < tol."""
    res = np.linalg.norm(A @ c_hat - b)
    nb  = np.linalg.norm(b)
    rel = res / nb if nb > 1e-15 else res
    if rel > tol:
        print(f"  Linear system residual: {rel:.2e}  (tol={tol:.1e})")
        return False
    return True
