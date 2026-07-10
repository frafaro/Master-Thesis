"""
Build basis matrices and change-of-basis matrices.

For Hermite basis:
    H_N[i,j] = coefficient of x^j in h_i(x)   (used for coefficient extraction)
    Q_hermite = I

For Logistic basis:
    alpha, beta = Stieltjes recurrence coefficients
    Q_logistic[k,j] = <L_k, h_j>_Gaussian   (computed via inner products)
"""

import numpy as np
from basis.hermite import hermite_basis_matrix
from basis.logistic import stieltjes_recurrence
from matrices.change_of_basis import build_Q_hermite, build_Q_logistic


def build_H(N: int) -> np.ndarray:
    """(N+1)×(N+1) Hermite monomial coefficient matrix (upper triangular)."""
    return hermite_basis_matrix(N)


def build_logistic_and_Q(N: int):
    """
    Build Stieltjes recurrence for logistic basis and compute Q_logistic.

    Returns
    -------
    alpha      : (N+1,) Stieltjes alpha coefficients
    beta       : (N+1,) Stieltjes beta coefficients
    Q_logistic : (N+1,N+1) change-of-basis matrix
    """
    alpha, beta = stieltjes_recurrence(N)
    Q = build_Q_logistic(N, alpha, beta)
    return alpha, beta, Q
