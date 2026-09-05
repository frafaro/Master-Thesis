"""
Global configuration for Gambaro (2024) replication.

VG parameters:   symmetric VG from Heston & Rossi (2016)
                 theta=0  →  skewness=0
                 nu=2/3   →  excess kurtosis = 3*nu = 2
                 sigma=0.2 (typical annual vol), mu=0

NIG parameters:  explicitly stated in Appendix A of Gambaro (2024)
                 mu=0, theta=0.05, sigma=0.2, kappa=0.3, dt=1

Heston parameters: calibrated to match Rompolis & Tzavalis (2008) targets
                   skewness=-1.2, excess kurtosis=2.5
                   T=1 (one year log-return)

CGMY parameters:   Carr, Geman, Madan & Yor (2002), pure jump (no Brownian).
                   G=7, M=12, Y=0.6, t=1; C risolto su κ₂=0.04 (vol 20%).
                   Nessun r, q: come VG/NIG, è la legge dell'incremento X_t.
                   Momenti effettivi: skew≈−0.494, exkurt≈1.353
                   (i valori −0.5 e 1.36 sono arrotondamenti, non vincoli).
"""

from math import gamma as _gamma
import numpy as np

# ── Variance Gamma ──────────────────────────────────────────────────────────
VG_PARAMS = {
    "mu":    0.0,
    "sigma": 0.2,
    "nu":    2.0 / 3.0,   # excess kurtosis = 3*nu = 2
    "theta": 0.0,          # zero skewness
}

# ── Normal Inverse Gaussian  ─────────────────────────────────────────────────
# Source: Appendix A of Gambaro (2024)
NIG_PARAMS = {
    "mu":    0.0,
    "theta": 0.05,
    "sigma": 0.2,
    "kappa": 0.3,
    "dt":    1.0,
}

# ── Heston model ─────────────────────────────────────────────────────────────
# Target moments (Table 1 of Gambaro 2024, from Rompolis & Tzavalis 2008):
#   skewness = -1.2,  excess kurtosis = 2.5
# Parameters calibrated numerically to match these targets.
# Starting point from Rompolis & Tzavalis (2008) simulation study.
HESTON_PARAMS = {
    # Calibrated to match Rompolis & Tzavalis (2008) targets:
    # skewness = -1.2,  excess kurtosis = 2.5
    "kappa": 2.0015,    # mean-reversion speed
    "theta": 0.04785,   # long-term variance
    "xi":    0.40299,   # vol-of-vol
    "rho":  -0.76635,   # correlation
    "v0":    0.05879,   # initial variance
    "T":     1.0,       # time horizon (years)
    "r":     0.0,       # risk-free rate (set to 0 for density of log-returns)
}

# ── CGMY (Carr–Geman–Madan–Yor 2002) ────────────────────────────────────────
# Lévy puro-salto. G < M ⇒ skew negativa. Y=0.6 ∈ (0,1): variazione finita.
# C = 0.04 / [t Γ(2−Y)(M^{Y−2}+G^{Y−2})]  così κ₂=0.04 come il VG.
# Non usiamo r né q (non entrano nella CF dell'incremento).
_CGMY_G, _CGMY_M, _CGMY_Y, _CGMY_T = 7.0, 12.0, 0.6, 1.0
_CGMY_VAR = 0.04
_CGMY_C = _CGMY_VAR / (
    _CGMY_T * _gamma(2.0 - _CGMY_Y)
    * (_CGMY_M ** (_CGMY_Y - 2.0) + _CGMY_G ** (_CGMY_Y - 2.0))
)

CGMY_PARAMS = {
    "C": _CGMY_C,       # ≈ 0.467485  (non 0.4692: quello dava Var=0.04015)
    "G": _CGMY_G,
    "M": _CGMY_M,
    "Y": _CGMY_Y,
    "t": _CGMY_T,
}

# ── Numerical settings ───────────────────────────────────────────────────────
N_MAX   = 20          # maximum expansion order for convergence plots
N_FIXED = [6, 16]     # orders used for density comparison figures (Figs 9-12)
N_COS   = 2**12       # COS grid size  = 4096

L       = 4.0         # domain parameter: I = [k1 - L*sqrt(k2+sqrt(k4)),
                      #                        k1 + L*sqrt(k2+sqrt(k4))]

GRID_SIZE = 10_000    # integration grid for distances / density evaluation

# ── CLR tolerance (from paper p.13) ─────────────────────────────────────────
CLR_TOL = 10.0        # |clr(p)(x)| must be < CLR_TOL on domain I
