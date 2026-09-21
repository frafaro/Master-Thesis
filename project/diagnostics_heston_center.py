"""
Heston: momenti grezzi+binomio vs cumulanti gia' standardizzati (come VG, m1=0).

Per VG, E[X]=0 quindi mu_k ~ momenti centrali. Per Heston m1=-0.026:
hermite_moments_from_raw fa
    mu*_k = sigma^{-k} sum_j C(k,j) (-m1)^{k-j} mu_j
che a k~20-30 e' un binomio alternato su numeri enormi.

Equivalente esatto (se l'aritmetica fosse infinita):
    kappa*_1 = 0,  kappa*_n = kappa_n / sigma^n   (n>=2)
    mu* dai cumulanti kappa*   (= E[(X*)^k] direttamente)
Poi He_k(mu*) / sqrt(k!)  senza ulteriore centratura.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
import numpy as np
from math import sqrt, factorial, comb

import config as CFG
import models.heston as heston_mod
from models.heston import _cumulants_to_raw_moments
from moments.hermite_moments import hermite_moments_from_raw, standardize_moments
from basis.hermite import He_coeffs, eval_hermite
from matrices.linear_system import build_A_tilde, build_A, build_b, solve_system
from matrices.basis_matrices import build_logistic_and_Q
from utils.quadrature import cumulant_domain, clr_domain, make_grid
from expansion.density import compute_C0, eval_density, exponent_func
from cos.cos_method import cos_density, benchmark_fourier_coeffs
from distances.metrics import all_distances
from scipy.linalg import solve as lu_solve
from basis.logistic import logistic_weight, eval_logistic_recurrence

N_MAX = 16
params = CFG.HESTON_PARAMS

kap = heston_mod.cumulants_numerical(params, max_order=2 * N_MAX + 2)
m1, var = kap[1], kap[2]
sigma = sqrt(var)
print(f"Heston: m1={m1:.6f}  sigma={sigma:.6f}  skew={kap[3]/sigma**3:.4f}  "
      f"exkurt={kap[4]/var**2:.4f}")

raw = _cumulants_to_raw_moments(kap, 2 * N_MAX + 2)
mh_old, _, _ = hermite_moments_from_raw(raw, K_max=2 * N_MAX)
std_mu_old, _, _ = standardize_moments(raw)

# cumulanti di X*: kappa*_1=0, kappa*_n = kappa_n / sigma^n
kap_star = np.zeros_like(kap)
kap_star[1] = 0.0
for n in range(2, len(kap)):
    kap_star[n] = kap[n] / (sigma ** n)
raw_star = _cumulants_to_raw_moments(kap_star, 2 * N_MAX + 2)  # = E[(X*)^k]
print(f"check X*: E[X*]={raw_star[1]:.2e}  E[(X*)^2]={raw_star[2]:.12f}  (deve essere 1)")

mh_star = np.zeros(2 * N_MAX + 1)
mh_star[0] = 1.0
for k in range(1, 2 * N_MAX + 1):
    coeffs = He_coeffs(k)
    he_e = sum(coeffs[j] * raw_star[j] for j in range(len(coeffs)))
    mh_star[k] = he_e / sqrt(factorial(k))

print("\n k | mu*_binomio (ora)     | mu* da kappa/sigma^n   | rel.diff")
for k in [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30]:
    d = abs(std_mu_old[k] - raw_star[k]) / max(abs(raw_star[k]), 1e-300)
    print(f"{k:3d}| {std_mu_old[k]: .12e} | {raw_star[k]: .12e} | {d:.2e}")

print("\n k | mh binomio            | mh kappa*              | rel.diff")
for k in [1, 2, 3, 4, 6, 8, 10, 14, 18, 22, 26, 30]:
    d = abs(mh_old[k] - mh_star[k]) / max(abs(mh_star[k]), 1e-300)
    print(f"{k:3d}| {mh_old[k]: .12e} | {mh_star[k]: .12e} | {d:.2e}")

# COS + Fourier
kdict = {"k1": kap[1], "k2": kap[2], "k4": kap[4]}
a, b = cumulant_domain(kdict, L=CFG.L)
x = make_grid(a, b, CFG.GRID_SIZE)
cf = lambda u: heston_mod.characteristic_function(u, params)
p = cos_density(x, cf, a, b, CFG.N_COS)
lp = np.log(np.maximum(p, 1e-300))
xs = (x - m1) / sigma
nu_g = np.exp(-xs**2 / 2) / (np.sqrt(2 * np.pi) * sigma)
nu_l = logistic_weight(xs) / sigma
alpha_L, beta_L, Q_log = build_logistic_and_Q(N_MAX)
cF_h = benchmark_fourier_coeffs(lp, x, lambda z: eval_hermite(z, N_MAX), nu_g, m1, sigma, N_MAX)
cF_l = benchmark_fourier_coeffs(
    lp, x, lambda z: eval_logistic_recurrence(z, N_MAX, alpha_L, beta_L), nu_l, m1, sigma, N_MAX)

print("\n== cond(A) e d2(c_hat, c_F)  Hermite / Logistic  (LU forzata, no lstsq) ==")
print(" N | cond H old  | cond H *   | d2H old  d2H *   | cond L old  | cond L *   | d2L old  d2L *")
for N in range(4, 17):
    row = []
    for mh in (mh_old, mh_star):
        At = build_A_tilde(N, mh)
        bN = build_b(N, mh)
        AH = build_A(N, np.eye(N_MAX + 1), At)
        AL = build_A(N, Q_log, At)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cH = lu_solve(AH, bN)
            cL = lu_solve(AL, bN)
        row.append((np.linalg.cond(AH), np.linalg.cond(AL),
                    np.linalg.norm(cH - cF_h[:N]), np.linalg.norm(cL - cF_l[:N]),
                    np.linalg.norm(cH), np.linalg.norm(cL)))
    (cHo, cLo, dHo, dLo, nHo, nLo) = row[0]
    (cHs, cLs, dHs, dLs, nHs, nLs) = row[1]
    print(f"{N:3d}| {cHo:.2e}   | {cHs:.2e}   | {dHo:7.3f} {dHs:7.3f} | "
          f"{cLo:.2e}   | {cLs:.2e}   | {dLo:7.3f} {dLs:7.3f}")

# Fig 11: densita' N=16 sul ristretto, mh_star + LU vs mh_old + lstsq
print("\n== Fig 11 N=16 restricted: vecchio (lstsq) vs kappa* (LU) ==")
a_r, b_r = clr_domain(
    lambda xx: np.log(np.maximum(cos_density(xx, cf, a, b, CFG.N_COS), 1e-300)),
    kdict, L_start=CFG.L, clr_tol=CFG.CLR_TOL)
x_r = make_grid(a_r, b_r, CFG.GRID_SIZE)
p_r = cos_density(x_r, cf, a, b, CFG.N_COS)
xs_r = (x_r - m1) / sigma
nu_gr = np.exp(-xs_r**2 / 2) / (np.sqrt(2 * np.pi) * sigma)

def dens_N16(mh, use_lstsq):
    At = build_A_tilde(16, mh)
    bN = build_b(16, mh)
    A = build_A(16, np.eye(N_MAX + 1), At)
    thr = 1e14 if use_lstsq else None
    c, cond = solve_system(A, bN, cond_threshold=thr)
    fn = lambda z: eval_hermite(np.asarray(z), 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C0 = compute_C0(c, fn, a_r, b_r, m1, sigma)
        ph = eval_density(x_r, c, C0, fn, m1, sigma)
    f = exponent_func(xs_r, c, fn)
    d = all_distances(x_r, ph, p_r, nu_gr)
    return c, cond, f, ph, d

c_old, cond_old, f_old, p_old, d_old = dens_N16(mh_old, True)
c_new, cond_new, f_new, p_new, d_new = dens_N16(mh_star, False)
print(f"old lstsq: cond={cond_old:.2e} ||c||={np.linalg.norm(c_old):.4f}  "
      f"f in [{f_old.min():.2f},{f_old.max():.2f}]  pmax={np.nanmax(p_old):.3f}  "
      f"L1={d_old['l1']:.4f}")
print(f"kappa* LU: cond={cond_new:.2e} ||c||={np.linalg.norm(c_new):.4f}  "
      f"f in [{f_new.min():.2f},{f_new.max():.2f}]  pmax={np.nanmax(p_new):.3f}  "
      f"L1={d_new['l1']:.4f}  COS pmax={p_r.max():.3f}")

# anche N=6, 10, 14 L1 restricted Hermite
print("\n== L1 Hermite restricted, LU, mh_star vs mh_old ==")
for N in [6, 8, 10, 12, 14, 16]:
    for lab, mh in [("old", mh_old), ("star", mh_star)]:
        At = build_A_tilde(N, mh); bN = build_b(N, mh)
        A = build_A(N, np.eye(N_MAX + 1), At)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = lu_solve(A, bN)
            fn = lambda z, NN=N: eval_hermite(np.asarray(z), NN)
            C0 = compute_C0(c, fn, a_r, b_r, m1, sigma)
            ph = eval_density(x_r, c, C0, fn, m1, sigma)
            L1 = all_distances(x_r, ph, p_r, nu_gr)["l1"]
        if lab == "old":
            L1o = L1
        else:
            print(f"N={N}: L1 old={L1o:.4f}  L1 star={L1:.4f}  finite star={np.all(np.isfinite(ph))}")
