"""
Impatto della rimozione di lstsq su VG, NIG e Heston.

Per ogni modello e base (Hermite/Logistica), per N = 2..20:
  - cond(A_N) e se il vecchio codice avrebbe usato lstsq (cond > 1e14)
  - dove scattava: ||c_LU - c_lstsq||, residui dei due metodi
  - effetto sulle metriche delle figure (L1, Aitchison, d2 vs c Fourier)
    calcolato solo agli N interessati (N <= 16 = range dei grafici).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
import numpy as np
from math import sqrt

import config as CFG
import models.variance_gamma as vg_mod
import models.nig as nig_mod
import models.heston as heston_mod
from utils.quadrature import cumulant_domain, make_grid
from basis.hermite import eval_hermite
from basis.logistic import logistic_weight, eval_logistic_recurrence
from moments.hermite_moments import hermite_moments_from_raw
from matrices.basis_matrices import build_logistic_and_Q
from matrices.linear_system import build_A_tilde, build_A, build_b
from expansion.density import compute_C0, eval_density
from cos.cos_method import cos_density, benchmark_fourier_coeffs
from distances.metrics import all_distances
from scipy.linalg import solve as lu_solve

N_MAX = CFG.N_MAX
THRESH = 1e14

MODELS = [
    ("VG", vg_mod.characteristic_function, vg_mod.raw_moments, CFG.VG_PARAMS),
    ("NIG", nig_mod.characteristic_function, nig_mod.raw_moments, CFG.NIG_PARAMS),
    ("Heston", heston_mod.characteristic_function, heston_mod.raw_moments, CFG.HESTON_PARAMS),
]

alpha_L, beta_L, Q_logistic = build_logistic_and_Q(N_MAX)
Q_hermite = np.eye(N_MAX + 1)

for name, cf_func, rm_func, params in MODELS:
    print("=" * 72)
    print(f"MODELLO {name}")
    print("=" * 72)
    raw_mu = rm_func(params, max_order=2 * N_MAX + 2)
    m1, sigma = raw_mu[1], sqrt(raw_mu[2] - raw_mu[1] ** 2)
    kap = np.zeros(5)
    kap[1] = raw_mu[1]
    kap[2] = raw_mu[2] - raw_mu[1] ** 2
    kap[4] = (raw_mu[4] - 4 * raw_mu[3] * raw_mu[1] - 3 * raw_mu[2] ** 2
              + 12 * raw_mu[2] * raw_mu[1] ** 2 - 6 * raw_mu[1] ** 4)
    a, b = cumulant_domain({"k1": kap[1], "k2": kap[2], "k4": kap[4]}, L=CFG.L)
    x = make_grid(a, b, CFG.GRID_SIZE)
    cf = lambda u: cf_func(u, params)
    p_cos = cos_density(x, cf, a, b, N_cos=CFG.N_COS)
    log_p = np.log(np.maximum(p_cos, 1e-300))
    x_std = (x - m1) / sigma
    nu_g = np.exp(-x_std**2 / 2) / (np.sqrt(2 * np.pi) * sigma)
    nu_l = logistic_weight(x_std) / sigma

    mh, _, _ = hermite_moments_from_raw(raw_mu, K_max=2 * N_MAX)
    c_four_h = benchmark_fourier_coeffs(
        log_p, x, lambda z: eval_hermite(z, N_MAX), nu_g, m1, sigma, N_MAX)
    c_four_l = benchmark_fourier_coeffs(
        log_p, x, lambda z: eval_logistic_recurrence(z, N_MAX, alpha_L, beta_L),
        nu_l, m1, sigma, N_MAX)

    for basis, Q, nu, c_four in [("Hermite", Q_hermite, nu_g, c_four_h),
                                 ("Logistic", Q_logistic, nu_l, c_four_l)]:
        rows = []
        for N in range(2, N_MAX + 1):
            At = build_A_tilde(N, mh)
            bN = build_b(N, mh)
            A = build_A(N, Q, At)
            cond = np.linalg.cond(A)
            if cond <= THRESH:
                continue  # vecchio e nuovo codice coincidono (LU)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c_lu = lu_solve(A, bN)
                c_ls, *_ = np.linalg.lstsq(A, bN, rcond=None)
            res_lu = np.linalg.norm(A @ c_lu - bN)
            res_ls = np.linalg.norm(A @ c_ls - bN)
            diff = np.linalg.norm(c_lu - c_ls)
            row = {"N": N, "cond": cond, "res_lu": res_lu, "res_ls": res_ls,
                   "diff": diff}
            if N <= 16:  # range dei grafici: calcola anche le metriche
                if basis == "Hermite":
                    fn = (lambda NN: (lambda z: eval_hermite(np.asarray(z), NN)))(N)
                else:
                    fn = (lambda NN: (lambda z: eval_logistic_recurrence(
                        np.asarray(z), NN, alpha_L, beta_L)))(N)
                for lab, c in [("lu", c_lu), ("ls", c_ls)]:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        C0 = compute_C0(c, fn, a, b, m1, sigma)
                        ph = eval_density(x, c, C0, fn, m1, sigma)
                        d = all_distances(x, ph, p_cos, nu)
                    row[f"L1_{lab}"] = d["l1"]
                    row[f"ait_{lab}"] = d["aitchison"]
                    row[f"d2_{lab}"] = np.linalg.norm(c - c_four[:N])
            rows.append(row)

        if not rows:
            print(f"\n  {basis}: lstsq non scattava mai (cond <= 1e14 per ogni N).")
            continue
        print(f"\n  {basis}: lstsq scattava per N = {[r['N'] for r in rows]}")
        for r in rows:
            line = (f"   N={r['N']:2d} cond={r['cond']:.2e} "
                    f"||c_LU-c_ls||={r['diff']:.3e} "
                    f"res LU={r['res_lu']:.1e} ls={r['res_ls']:.1e}")
            if "L1_lu" in r:
                line += (f"\n         L1: LU={r['L1_lu']:.5f} ls={r['L1_ls']:.5f}"
                         f" | Aitch: LU={r['ait_lu']:.5f} ls={r['ait_ls']:.5f}"
                         f" | d2(c_Four): LU={r['d2_lu']:.5f} ls={r['d2_ls']:.5f}")
            print(line)
