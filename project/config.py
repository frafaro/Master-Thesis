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
"""

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

# ── Numerical settings ───────────────────────────────────────────────────────
N_MAX   = 20          # maximum expansion order for convergence plots
N_FIXED = [6, 16]     # orders used for density comparison figures (Figs 9-12)
N_COS   = 2**12       # COS grid size  = 4096

L       = 4.0         # domain parameter: I = [k1 - L*sqrt(k2+sqrt(k4)),
                      #                        k1 + L*sqrt(k2+sqrt(k4))]

GRID_SIZE = 10_000    # integration grid for distances / density evaluation

# ── CLR tolerance (from paper p.13) ─────────────────────────────────────────
CLR_TOL = 10.0        # |clr(p)(x)| must be < CLR_TOL on domain I
