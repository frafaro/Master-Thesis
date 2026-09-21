"""
Diagnostica CGMY, parte 2: verifica dei due rimedi.
 1. N=6: dominio ristretto (log p > -10, come Heston Fig 7/11) elimina il picco?
 2. N=16: la soluzione LU forzata (ignorando la soglia cond>1e14) e' buona?
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from math import sqrt

import config as CFG
import models.cgmy as cgmy_mod
from utils.quadrature import cumulant_domain, clr_domain, make_grid
from basis.hermite import eval_hermite
from moments.hermite_moments import hermite_moments_from_raw
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
a_full, b_full = cumulant_domain(cumulants_dict, L=CFG.L)
cf = lambda u: cgmy_mod.characteristic_function(u, params)

mh, _, _ = hermite_moments_from_raw(raw_mu, K_max=2 * N_MAX)
Q_hermite = np.eye(N_MAX + 1)

# dominio ristretto con lo stesso criterio di Heston: log p_COS > -CLR_TOL
a_r, b_r = clr_domain(
    lambda xx: np.log(np.maximum(cos_density(xx, cf, a_full, b_full, CFG.N_COS), 1e-300)),
    cumulants_dict, L_start=CFG.L, clr_tol=CFG.CLR_TOL)
print(f"Dominio pieno:     [{a_full:.4f}, {b_full:.4f}]  (std [-5.88, 5.88])")
print(f"Dominio ristretto: [{a_r:.4f}, {b_r:.4f}]  "
      f"(std [{(a_r-m1)/sigma:.2f}, {(b_r-m1)/sigma:.2f}])")

x_r = make_grid(a_r, b_r, CFG.GRID_SIZE)
p_cos_r = cos_density(x_r, cf, a_full, b_full, CFG.N_COS)
x_std_r = (x_r - m1) / sigma
nu_gauss_r = np.exp(-x_std_r**2 / 2) / (np.sqrt(2 * np.pi) * sigma)

x_f = make_grid(a_full, b_full, CFG.GRID_SIZE)
p_cos_f = cos_density(x_f, cf, a_full, b_full, CFG.N_COS)
x_std_f = (x_f - m1) / sigma
nu_gauss_f = np.exp(-x_std_f**2 / 2) / (np.sqrt(2 * np.pi) * sigma)

def density_from_c(c_h, N, x, a, b):
    fn = lambda z: eval_hermite(np.asarray(z), N)
    C0 = compute_C0(c_h, fn, a, b, m1, sigma)
    return eval_density(x, c_h, C0, fn, m1, sigma)

print("\n-- N=6: pieno vs ristretto (stessi c_hat, cambia solo il dominio) --")
At6 = build_A_tilde(6, mh); b6 = build_b(6, mh)
c6 = lu_solve(build_A(6, Q_hermite, At6), b6)
p6_full = density_from_c(c6, 6, x_f, a_full, b_full)
p6_restr = density_from_c(c6, 6, x_r, a_r, b_r)
d_full = all_distances(x_f, p6_full, p_cos_f, nu_gauss_f)
d_restr = all_distances(x_r, p6_restr, p_cos_r, nu_gauss_r)
print(f"pieno:     L1={d_full['l1']:.4f}  L2={d_full['l2']:.4f}  "
      f"logL2={d_full['log_l2']:.4f}  p_hat(b)={p6_full[-1]:.3f}")
print(f"ristretto: L1={d_restr['l1']:.4f}  L2={d_restr['l2']:.4f}  "
      f"logL2={d_restr['log_l2']:.4f}  p_hat(b_r)={p6_restr[-1]:.3f}")

print("\n-- N=16: lstsq (pipeline) vs LU forzato, dominio pieno --")
At16 = build_A_tilde(16, mh); b16 = build_b(16, mh)
A16 = build_A(16, Q_hermite, At16)
c16_lu = lu_solve(A16, b16)
c16_ls, *_ = np.linalg.lstsq(A16, b16, rcond=None)
for label, c in [("lstsq", c16_ls), ("LU   ", c16_lu)]:
    p16 = density_from_c(c, 16, x_f, a_full, b_full)
    d = all_distances(x_f, p16, p_cos_f, nu_gauss_f)
    print(f"{label}: L1={d['l1']:.4f}  L2={d['l2']:.4f}  logL2={d['log_l2']:.4f}  "
      f"aitchison={d['aitchison']:.4f}")

print("\n-- N=16 LU su dominio ristretto --")
p16_r = density_from_c(c16_lu, 16, x_r, a_r, b_r)
d16_r = all_distances(x_r, p16_r, p_cos_r, nu_gauss_r)
print(f"L1={d16_r['l1']:.4f}  L2={d16_r['l2']:.4f}  logL2={d16_r['log_l2']:.4f}")

# dove finisce il picco N=6: punto in cui f cambia segno / minimo di log p_hat
f6 = np.log(np.maximum(p6_full, 1e-300))
i_min = np.argmin(f6[len(f6)//2:]) + len(f6)//2
print(f"\nminimo di log p_hat (N=6) a x*={x_std_f[i_min]:.2f}; "
      f"bordo ristretto a x*={(b_r-m1)/sigma:.2f}")
