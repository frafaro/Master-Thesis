"""
Diagnostica Heston: i momenti mpmath sono accurati? Spiega il picco ~100
della Fig. 3 e la Hermite piatta della Fig. 11.

 1. Cumulanti mpmath dps=50 (pipeline) vs dps=200 vs quadratura COS
    su dominio largo (L=10): dove iniziano a divergere?
 2. Verifica centratura: mh[1], mh[2] per ogni variante.
 3. Sistema eq. 15 ricostruito con ciascuna variante di momenti
    (con la regola lstsq per cond>1e14, come le figure committate):
    d2(c_hat, c_Fourier) per N=4..16 — il picco ~100 sparisce con
    momenti migliori?
 4. Fig. 11: norma di c_hat lstsq a N=16 Hermite e range dell'esponente
    f sul dominio ristretto (perche' la densita' e' piatta).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
import numpy as np
from math import sqrt, comb

import mpmath
import config as CFG
import models.heston as heston_mod
from models.heston import _cumulants_to_raw_moments
from utils.quadrature import cumulant_domain, clr_domain, make_grid
from basis.hermite import eval_hermite
from basis.logistic import logistic_weight, eval_logistic_recurrence
from moments.hermite_moments import hermite_moments_from_raw
from matrices.basis_matrices import build_logistic_and_Q
from matrices.linear_system import build_A_tilde, build_A, build_b, solve_system
from expansion.density import compute_C0, eval_density, exponent_func
from cos.cos_method import cos_density, benchmark_fourier_coeffs

params = CFG.HESTON_PARAMS
N_MAX = 16
MAXORD = 2 * N_MAX + 2   # 34: basta per N<=16 (A~ usa mh fino a 2N-2=30)

# ── cumulanti mpmath a dps variabile, con eccezioni visibili ────────────────
def cumulants_dps(dps, max_order):
    mpmath.mp.dps = dps
    kappa_H = float(params["kappa"]); theta_H = float(params["theta"])
    xi = float(params["xi"]); rho = float(params["rho"])
    v0 = float(params["v0"]); T = float(params["T"])
    def K(s):
        alpha = kappa_H - rho * xi * s
        d = mpmath.sqrt(alpha**2 + xi**2 * (s - s**2))
        g = (alpha - d) / (alpha + d)
        exp_dT = mpmath.exp(-d * T)
        A = kappa_H * theta_H / xi**2 * ((alpha - d) * T
            - 2 * mpmath.log((1 - g * exp_dT) / (1 - g)))
        B = (alpha - d) / xi**2 * (1 - exp_dT) / (1 - g * exp_dT)
        return A + B * v0
    kap = np.zeros(max_order + 1)
    errs = []
    for k in range(1, max_order + 1):
        try:
            kap[k] = float(mpmath.diff(K, mpmath.mpf(0), k))
        except Exception as e:
            errs.append((k, repr(e)))
            kap[k] = 0.0
    return kap, errs

print("Calcolo cumulanti dps=50 e dps=200 ...")
kap50, err50 = cumulants_dps(50, MAXORD)
kap200, err200 = cumulants_dps(200, MAXORD)
print(f"Eccezioni silenziate a dps=50:  {err50 if err50 else 'nessuna'}")
print(f"Eccezioni silenziate a dps=200: {err200 if err200 else 'nessuna'}")
print(f"skew={kap50[3]/kap50[2]**1.5:.4f}  exkurt={kap50[4]/kap50[2]**2:.4f}")

print("\n k | kappa_k dps=50        | kappa_k dps=200       | rel.diff")
for k in [2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 34]:
    d = abs(kap50[k] - kap200[k]) / max(abs(kap200[k]), 1e-300)
    print(f"{k:3d}| {kap50[k]: .12e} | {kap200[k]: .12e} | {d:.2e}")

# ── quadratura COS su dominio largo ─────────────────────────────────────────
mu50 = _cumulants_to_raw_moments(kap50, MAXORD)
mu200 = _cumulants_to_raw_moments(kap200, MAXORD)
kdict = {"k1": kap200[1], "k2": kap200[2], "k4": kap200[4]}
aW, bW = cumulant_domain(kdict, L=10.0)
xW = make_grid(aW, bW, 60_000)
cf = lambda u: heston_mod.characteristic_function(u, params)
pW = cos_density(xW, cf, aW, bW, N_cos=2**14)
print(f"\nDominio largo L=10: [{aW:.3f}, {bW:.3f}], integrale p = "
      f"{np.trapz(pW, xW):.8f}")
mu_quad = np.array([np.trapz(xW**k * pW, xW) for k in range(MAXORD + 1)])

print("\n k | mu_k dps=50           | mu_k quadratura L=10  | rel.diff (50 vs quad)")
for k in [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]:
    d = abs(mu50[k] - mu_quad[k]) / max(abs(mu_quad[k]), 1e-300)
    print(f"{k:3d}| {mu50[k]: .12e} | {mu_quad[k]: .12e} | {d:.2e}")

# ── momenti di Hermite per le tre varianti ──────────────────────────────────
mh50, m1_50, sg50 = hermite_moments_from_raw(mu50, K_max=MAXORD - 2)
mh200, m1_200, sg200 = hermite_moments_from_raw(mu200, K_max=MAXORD - 2)
mhq, m1q, sgq = hermite_moments_from_raw(mu_quad, K_max=MAXORD - 2)
print(f"\nCentratura: mh50[1]={mh50[1]:.2e} mh50[2]={mh50[2]:.2e} | "
      f"mhq[1]={mhq[1]:.2e} mhq[2]={mhq[2]:.2e}")
print("\n k | mh dps=50     | mh dps=200    | mh quadratura")
for k in [3, 4, 6, 8, 10, 14, 18, 22, 26, 30]:
    print(f"{k:3d}| {mh50[k]: .6e} | {mh200[k]: .6e} | {mhq[k]: .6e}")

# ── sistema eq.15 con le tre varianti: d2 per N=4..16 ───────────────────────
a4, b4 = cumulant_domain(kdict, L=CFG.L)
x4 = make_grid(a4, b4, CFG.GRID_SIZE)
p4 = cos_density(x4, cf, a4, b4, N_cos=CFG.N_COS)
lp4 = np.log(np.maximum(p4, 1e-300))
alpha_L, beta_L, Q_log = build_logistic_and_Q(N_MAX)
Q_her = np.eye(N_MAX + 1)

def d2_curve(mh, m1, sg, Q, basis):
    xs = (x4 - m1) / sg
    if basis == "H":
        nu = np.exp(-xs**2 / 2) / (np.sqrt(2 * np.pi) * sg)
        cF = benchmark_fourier_coeffs(lp4, x4, lambda z: eval_hermite(z, N_MAX),
                                      nu, m1, sg, N_MAX)
    else:
        nu = logistic_weight(xs) / sg
        cF = benchmark_fourier_coeffs(
            lp4, x4, lambda z: eval_logistic_recurrence(z, N_MAX, alpha_L, beta_L),
            nu, m1, sg, N_MAX)
    out = []
    for N in range(4, N_MAX + 1):
        At = build_A_tilde(N, mh); bN = build_b(N, mh)
        A = build_A(N, Q, At)
        c, cond = solve_system(A, bN, cond_threshold=1e14)  # regola delle figure
        out.append((N, np.linalg.norm(c - cF[:N]), cond))
    return out

print("\n== d2(c_hat, c_Fourier), regola lstsq cond>1e14 (come Fig. 3) ==")
for basis, Q in [("H", Q_her), ("L", Q_log)]:
    print(f"\n base {basis}:  N | dps=50 (pipeline) | dps=200 | quadratura")
    r50 = d2_curve(mh50, m1_50, sg50, Q, basis)
    r200 = d2_curve(mh200, m1_200, sg200, Q, basis)
    rq = d2_curve(mhq, m1q, sgq, Q, basis)
    for (N, d1, c1), (_, d2_, _), (_, d3, c3) in zip(r50, r200, rq):
        print(f"  {N:2d} | {d1:12.4f} | {d2_:12.4f} | {d3:12.4f}"
              f"   (cond50={c1:.1e}, condq={c3:.1e})")

# ── Fig. 11: perche' Hermite N=16 e' piatta ─────────────────────────────────
print("\n== Fig. 11: Hermite N=16 sul dominio ristretto ==")
a_r, b_r = clr_domain(
    lambda xx: np.log(np.maximum(cos_density(xx, cf, a4, b4, CFG.N_COS), 1e-300)),
    kdict, L_start=CFG.L, clr_tol=CFG.CLR_TOL)
x_r = make_grid(a_r, b_r, CFG.GRID_SIZE)
At16 = build_A_tilde(16, mh50); b16 = build_b(16, mh50)
A16 = build_A(16, Q_her, At16)
c16, cond16 = solve_system(A16, b16, cond_threshold=1e14)
print(f"cond(A_H,16) = {cond16:.2e}  -> lstsq. ||c_hat|| = {np.linalg.norm(c16):.4f}")
print(f"residuo lstsq = {np.linalg.norm(A16 @ c16 - b16):.3e}  (||b|| = {np.linalg.norm(b16):.3f})")
fn16 = lambda z: eval_hermite(np.asarray(z), 16)
xs_r = (x_r - m1_50) / sg50
f_r = exponent_func(xs_r, c16, fn16)
print(f"range di f su I_restr: [{f_r.min():.3f}, {f_r.max():.3f}]  "
      f"(escursione {f_r.max()-f_r.min():.3f})")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    C0 = compute_C0(c16, fn16, a_r, b_r, m1_50, sg50)
    p16 = eval_density(x_r, c16, C0, fn16, m1_50, sg50)
print(f"p_hat N=16: min={p16.min():.4f} max={p16.max():.4f} "
      f"(COS max = {cos_density(x_r, cf, a4, b4, CFG.N_COS).max():.4f})")
