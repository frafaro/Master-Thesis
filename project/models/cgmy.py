"""
Modello CGMY (Carr, Geman, Madan & Yor 2002).

Processo di Lévy puro-salto, senza diffusione (σ=0) e senza drift
deterministico extra. Stesso oggetto di VG/NIG in questa pipeline:
la legge dell'incremento X_t, non il prezzo risk-neutral S_t.

Densità di Lévy [Carr et al. 2002, "The Fine Structure of Asset Returns"]:
    k(x) = C e^{-M x} / x^{1+Y}           se x > 0
    k(x) = C e^{-G |x|} / |x|^{1+Y}       se x < 0

C > 0  attività (scala il numero di salti)
G > 0  decrescita della coda sinistra
M > 0  decrescita della coda destra
Y ∈ (0, 2) \\ {1}  indice di attività (Blumenthal–Getoor)
    0 < Y < 1 : attività infinita, variazione finita
    1 < Y < 2 : attività infinita, variazione infinita

G < M  ⇒  coda sinistra più grassa  ⇒  skewness negativa.

Funzione caratteristica (screenshot / Carr et al. 2002), t = orizzonte:
    φ(u) = exp( t C Γ(-Y) [ (M − i u)^Y − M^Y + (G + i u)^Y − G^Y ] )

Non compaiono r né q: servono solo se si studia R = log(S_t/S_0) sotto
la misura martingala, con compensatore
    ω = −C Γ(-Y) [(M−1)^Y − M^Y + (G+1)^Y − G^Y]
e fattore e^{i u (r−q+ω) t}. Qui, come per VG e NIG, quel pezzo non entra.

CGF  K(s) = log E[e^{s X}] = log φ(−i s)
    K(s) = t C Γ(-Y) [ (M − s)^Y − M^Y + (G + s)^Y − G^Y ]

Cumulanti (n ≥ 1), usando Γ(-Y) Y(Y−1)…(Y−n+1) = (−1)^n Γ(n−Y):
    κ_n = t C Γ(n − Y) [ M^{Y−n} + (−1)^n G^{Y−n} ]
"""

import numpy as np
from math import comb, gamma
from typing import Dict


def C_from_variance(G: float, M: float, Y: float, t: float,
                    target_var: float) -> float:
    """
    Risolve κ_2 = target_var rispetto a C, con (G, M, Y, t) fissati.

    κ_2 = t C Γ(2−Y) (M^{Y−2} + G^{Y−2})
    ⇒  C = target_var / [ t Γ(2−Y) (M^{Y−2} + G^{Y−2}) ]

    Con (G,M,Y) fissati, C determina anche skew e kurtosi
    (skew ∝ 1/√(tC), exkurt ∝ 1/(tC)): non è un puro parametro di scala.
    """
    a2 = gamma(2.0 - Y) * (M**(Y - 2.0) + G**(Y - 2.0))
    return target_var / (t * a2)


def characteristic_function(u: np.ndarray, params: Dict) -> np.ndarray:
    """
    φ(u) = exp( t C Γ(-Y) [ (M − i u)^Y − M^Y + (G + i u)^Y − G^Y ] )

    Per u reale, Re(M − i u) = M > 0 e Re(G + i u) = G > 0, quindi
    la potenza principale di numpy è ben definita (niente taglio).
    """
    C = params["C"]
    G = params["G"]
    M = params["M"]
    Y = params["Y"]
    t = params.get("t", 1.0)
    u = np.asarray(u, dtype=complex)
    # Γ(-Y) è un numero reale (Y=0.6 ⇒ Γ(-0.6) < 0, tra -1 e 0)
    prefactor = t * C * gamma(-Y)
    # (M − i u)^Y + (G + i u)^Y meno i termini di centro
    jump = ((M - 1j * u)**Y - M**Y
            + (G + 1j * u)**Y - G**Y)
    return np.exp(prefactor * jump)


def cumulant(n: int, params: Dict) -> float:
    """
    κ_n = t C Γ(n−Y) [ M^{Y−n} + (−1)^n G^{Y−n} ]

    n=1 media, n=2 varianza, n=3 → skew, n=4 → kurtosi di eccesso.
    """
    if n < 1:
        raise ValueError("i cumulanti partono da n=1")
    C = params["C"]
    G = params["G"]
    M = params["M"]
    Y = params["Y"]
    t = params.get("t", 1.0)
    return t * C * gamma(n - Y) * (M**(Y - n) + ((-1)**n) * G**(Y - n))


def cumulants(params: Dict, max_order: int = 30) -> np.ndarray:
    """Vettore κ[0 unused], κ[1], …, κ[max_order]."""
    kappas = np.zeros(max_order + 1)
    for k in range(1, max_order + 1):
        kappas[k] = cumulant(k, params)
    return kappas


def raw_moments(params: Dict, max_order: int = 30) -> np.ndarray:
    """
    Momenti grezzi E[X^k] dai cumulanti, stessa ricorrenza di VG/NIG:
        μ_0 = 1
        μ_n = Σ_{k=1}^n C(n−1, k−1) κ_k μ_{n−k}
    """
    kappas = cumulants(params, max_order)
    mu = np.zeros(max_order + 1)
    mu[0] = 1.0
    for n in range(1, max_order + 1):
        s = 0.0
        for k in range(1, n + 1):
            s += comb(n - 1, k - 1) * kappas[k] * mu[n - k]
        mu[n] = s
    return mu


def summarize_moments(params: Dict) -> Dict[str, float]:
    """Media, varianza, skewness, kurtosi di eccesso (per log / README)."""
    k1 = cumulant(1, params)
    k2 = cumulant(2, params)
    k3 = cumulant(3, params)
    k4 = cumulant(4, params)
    return {
        "mean": k1,
        "var": k2,
        "sigma": float(np.sqrt(k2)),
        "skew": k3 / k2**1.5,
        "exkurt": k4 / k2**2,
    }
