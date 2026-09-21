"""
Diagnostica CGMY, parte 3 (pre-modifica):
 1. Base logistica CGMY: LU pura vs lstsq per ogni N (residui e distanze L1).
 2. Criterio dominio ristretto: proxy log p > -10 vs |clr| < 10 esatto.
 3. N=17..20 con LU pura: la pipeline sopravvive (niente crash)?
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from math import sqrt

import config as CFG
import models.cgmy as cgmy_mod
from utils.quadrature import cumulant_domain, make_grid
from basis.hermite import eval_hermite
from basis.logistic import logistic_weight, eval_logistic_recurrence
from moments.hermite_moments import hermite_moments_from_raw
from matrices.basis_matrices import build_logistic_and_Q
from matrices.linear_system import build_A_tilde, build_A, build_b
from expansion.density import compute_C0, eval_density
from cos.cos_method import cos_density
from distances.metrics import all_distances
from scipy.linalg import solve as lu_solve

params = CFG.CGMY_PARAMS
N_MAX = CFG.N_MAX

raw_mu = cgmy_mod.raw_moments(params, max_order=2 * N_MAX + 2)
m1, sigma = raw_mu[1], sqrt(raw_mu[2] - raw_mu[1] ** 2)
kap = cgmy_mod.cumulants(params, 4)
cumulants_dict = {"k1": kap[1], "k2": kap[2], "k4": kap[4]}
a, b = cumulant_domain(cumulants_dict, L=CFG.L)
cf = lambda u: cgmy_mod.characteristic_function(u, params)

x = make_grid(a, b, CFG.GRID_SIZE)
p_cos = cos_density(x, cf, a, b, N_cos=CFG.N_COS)
log_p = np.log(np.maximum(p_cos, 1e-300))
x_std = (x - m1) / sigma
nu_gauss = np.exp(-x_std**2 / 2) / (np.sqrt(2 * np.pi) * sigma)
nu_logis = logistic_weight(x_std) / sigma

mh, _, _ = hermite_moments_from_raw(raw_mu, K_max=2 * N_MAX)
alpha_L, beta_L, Q_logistic = build_logistic_and_Q(N_MAX)

print("== 1. LOGISTICA: LU vs lstsq per N=4..16 ==")
print(" N | cond(A_log)  | res LU    | res lstsq | L1 (LU)  | L1 (lstsq)")
for N in range(4, 17):
    At = build_A_tilde(N, mh)
    bN = build_b(N, mh)
    A = build_A(N, Q_logistic, At)
    cond = np.linalg.cond(A)
    c_lu = lu_solve(A, bN)
    c_ls, *_ = np.linalg.lstsq(A, bN, rcond=None)
    res_lu = np.linalg.norm(A @ c_lu - bN)
    res_ls = np.linalg.norm(A @ c_ls - bN)
    fn = lambda z: eval_logistic_recurrence(np.asarray(z), N, alpha_L, beta_L)
    l1 = {}
    for lab, c in [("lu", c_lu), ("ls", c_ls)]:
        C0 = compute_C0(c, fn, a, b, m1, sigma)
        ph = eval_density(x, c, C0, fn, m1, sigma)
        l1[lab] = all_distances(x, ph, p_cos, nu_logis)["l1"]
    print(f"{N:3d}| {cond:.3e}   | {res_lu:.2e} | {res_ls:.2e} | {l1['lu']:.5f}  | {l1['ls']:.5f}")

print("\n== 2. Criterio dominio: proxy vs |clr|<10 esatto ==")
E_log_gauss = np.trapz(log_p * nu_gauss, x)
E_log_logis = np.trapz(log_p * nu_logis, x)
print(f"E_nu[log p]: gauss={E_log_gauss:.4f}, logistic={E_log_logis:.4f}")
for label, cut in [("proxy  log p > -10        ", -10.0),
                   ("esatto clr>-10 (nu gauss) ", -10.0 + E_log_gauss),
                   ("esatto clr>-10 (nu logis) ", -10.0 + E_log_logis)]:
    valid = log_p > cut
    a_c, b_c = x[valid][0], x[valid][-1]
    print(f"{label}: [{a_c:.4f}, {b_c:.4f}]  std dx = {(b_c-m1)/sigma:.3f}")

print("\n== 3. N=17..20 con LU pura (sopravvivenza) ==")
import warnings
for N in [17, 18, 19, 20]:
    At = build_A_tilde(N, mh)
    bN = build_b(N, mh)
    for lab, Q in [("H", np.eye(N_MAX + 1)), ("L", Q_logistic)]:
        A = build_A(N, Q, At)
        c = lu_solve(A, bN)
        if lab == "H":
            fn = lambda z: eval_hermite(np.asarray(z), N)
            nu = nu_gauss
        else:
            fn = lambda z: eval_logistic_recurrence(np.asarray(z), N, alpha_L, beta_L)
            nu = nu_logis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            C0 = compute_C0(c, fn, a, b, m1, sigma)
            ph = eval_density(x, c, C0, fn, m1, sigma)
            d = all_distances(x, ph, p_cos, nu)
        print(f"N={N} {lab}: ||c||={np.linalg.norm(c):.3e}  L1={d['l1']:.4f}  "
              f"finite: {np.all(np.isfinite(ph))}")
