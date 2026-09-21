"""
Diagnostica CGMY (non fa parte della pipeline, non tocca le figure).

Domande:
 A. I momenti/CF CGMY sono coerenti con la teoria?  (check vs quadratura COS)
 B. Perché Hermite esplode a N=16?   (cond(A), momenti analitici vs troncati)
 C. Perché la coda destra diverge?   (clr al bordo, f(x*) al bordo, pendenze)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from math import factorial, sqrt

import config as CFG
import models.cgmy as cgmy_mod
import models.variance_gamma as vg_mod
import models.nig as nig_mod
from utils.quadrature import cumulant_domain, make_grid
from basis.hermite import eval_hermite
from basis.logistic import logistic_weight, eval_logistic_recurrence
from moments.hermite_moments import hermite_moments_from_raw
from matrices.basis_matrices import build_logistic_and_Q
from matrices.linear_system import build_A_tilde, build_A, build_b, solve_system
from expansion.density import compute_C0, eval_density, exponent_func
from cos.cos_method import cos_density, benchmark_fourier_coeffs
from scipy.linalg import solve as lu_solve

N_MAX = CFG.N_MAX
params = CFG.CGMY_PARAMS

print("=" * 70)
print("A. COERENZA TEORICA: momenti analitici vs densita' COS")
print("=" * 70)

raw_mu = cgmy_mod.raw_moments(params, max_order=2 * N_MAX + 2)
m1, sigma = raw_mu[1], sqrt(raw_mu[2] - raw_mu[1] ** 2)
kap = cgmy_mod.cumulants(params, 4)
print(f"m1={m1:.6f}  sigma={sigma:.6f}  skew={kap[3]/kap[2]**1.5:.4f}  "
      f"exkurt={kap[4]/kap[2]**2:.4f}")

cumulants_dict = {"k1": kap[1], "k2": kap[2], "k4": kap[4]}
a, b = cumulant_domain(cumulants_dict, L=CFG.L)
print(f"Dominio L=4: [a,b] = [{a:.4f}, {b:.4f}]   "
      f"in unita' std: [{(a-m1)/sigma:.3f}, {(b-m1)/sigma:.3f}]")

x = make_grid(a, b, CFG.GRID_SIZE)
cf = lambda u: cgmy_mod.characteristic_function(u, params)
p_cos = cos_density(x, cf, a, b, N_cos=CFG.N_COS)
log_p = np.log(np.maximum(p_cos, 1e-300))

# momenti grezzi dalla densita' COS (troncata su [a,b]) vs analitici (su R)
print("\n k | mu_k analitico (R)   | mu_k COS ([a,b])     | rapporto COS/analitico")
for k in [1, 2, 4, 8, 12, 16, 20, 24, 30]:
    mu_num = np.trapz(x**k * p_cos, x)
    print(f"{k:3d}| {raw_mu[k]: .12e} | {mu_num: .12e} | {mu_num/raw_mu[k]: .6f}")

# momenti di Hermite: analitici vs quadratura COS
mh, _, _ = hermite_moments_from_raw(raw_mu, K_max=2 * N_MAX)
x_std = (x - m1) / sigma
P_h = eval_hermite(x_std, 2 * N_MAX)
print("\n k | mh_k analitico (R)   | mh_k COS ([a,b])     | diff")
for k in [3, 4, 6, 8, 10, 14, 18, 22, 26, 30]:
    mh_num = np.trapz(P_h[k] * p_cos, x)
    print(f"{k:3d}| {mh[k]: .12e} | {mh_num: .12e} | {mh[k]-mh_num: .3e}")

print("\n" + "=" * 70)
print("B. PERCHE' HERMITE ESPLODE A N=16")
print("=" * 70)

nu_gauss = np.exp(-x_std**2 / 2) / (np.sqrt(2 * np.pi) * sigma)
c_fourier = benchmark_fourier_coeffs(
    log_p, x, lambda z: eval_hermite(z, N_MAX), nu_gauss, m1, sigma, N_MAX)

alpha_L, beta_L, Q_logistic = build_logistic_and_Q(N_MAX)
Q_hermite = np.eye(N_MAX + 1)

print("\n N | cond(A_herm) | metodo  | ||c_hat||   | d2(c_hat, c_Fourier)")
c_hats = {}
for N in range(4, N_MAX + 1):
    At = build_A_tilde(N, mh)
    bN = build_b(N, mh)
    A = build_A(N, Q_hermite, At)
    c_hat, cond = solve_system(A, bN)
    c_hats[N] = c_hat
    method = "lstsq" if cond > 1e14 else "LU"
    d2 = np.linalg.norm(c_hat - c_fourier[:N])
    print(f"{N:3d}| {cond:11.3e} | {method:7s} | {np.linalg.norm(c_hat):.6f} | {d2:.6f}")

# a N=16: LU forzato vs lstsq
print("\n-- N=16: LU forzato vs lstsq vs Fourier --")
At16 = build_A_tilde(16, mh)
b16 = build_b(16, mh)
A16 = build_A(16, Q_hermite, At16)
c_lu = lu_solve(A16, b16)
c_ls, *_ = np.linalg.lstsq(A16, b16, rcond=None)
print(f"||c_LU||={np.linalg.norm(c_lu):.4f}  residuo LU={np.linalg.norm(A16@c_lu-b16):.2e}")
print(f"||c_ls||={np.linalg.norm(c_ls):.4f}  residuo ls={np.linalg.norm(A16@c_ls-b16):.2e}")
print(f"d2(c_LU, c_Fourier)={np.linalg.norm(c_lu-c_fourier[:16]):.4f}")
print(f"d2(c_ls, c_Fourier)={np.linalg.norm(c_ls-c_fourier[:16]):.4f}")
print(f"\n j |  c_LU        |  c_lstsq     |  c_Fourier")
for j in range(16):
    print(f"{j+1:3d}| {c_lu[j]: .6f}   | {c_ls[j]: .6f}   | {c_fourier[j]: .6f}")

# sistema "ideale": A e b costruiti con i momenti della densita' COS troncata
print("\n-- N=15,16: sistema con momenti TRONCATI (da COS) vs momenti su R --")
mh_trunc = np.array([np.trapz(P_h[k] * p_cos, x) for k in range(2 * N_MAX + 1)])
for N in [14, 15, 16]:
    At_t = build_A_tilde(N, mh_trunc)
    b_t = build_b(N, mh_trunc)
    A_t = build_A(N, Q_hermite, At_t)
    cond_t = np.linalg.cond(A_t)
    c_t, _ = solve_system(A_t, b_t)
    d2_t = np.linalg.norm(c_t - c_fourier[:N])
    At_r = build_A_tilde(N, mh)
    A_r = build_A(N, Q_hermite, At_r)
    cond_r = np.linalg.cond(A_r)
    print(f"N={N}: cond(trunc)={cond_t:.3e} d2={d2_t:.4f}   vs   cond(R)={cond_r:.3e}")

# confronto crescita mh: CGMY vs VG vs NIG
print("\n-- crescita |mh_k|: CGMY vs VG vs NIG --")
raw_vg = vg_mod.raw_moments(CFG.VG_PARAMS, max_order=2 * N_MAX + 2)
mh_vg, _, _ = hermite_moments_from_raw(raw_vg, K_max=2 * N_MAX)
raw_nig = nig_mod.raw_moments(CFG.NIG_PARAMS, max_order=2 * N_MAX + 2)
mh_nig, _, _ = hermite_moments_from_raw(raw_nig, K_max=2 * N_MAX)
print(" k |  |mh| CGMY   |  |mh| VG     |  |mh| NIG")
for k in [6, 10, 14, 18, 22, 26, 30]:
    print(f"{k:3d}| {abs(mh[k]):.6e} | {abs(mh_vg[k]):.6e} | {abs(mh_nig[k]):.6e}")

# cond(A) VG e NIG per confronto
print("\n N | cond(A) CGMY | cond(A) VG   | cond(A) NIG")
for N in [10, 12, 14, 15, 16, 18, 20]:
    conds = []
    for mh_m in (mh, mh_vg, mh_nig):
        At_m = build_A_tilde(N, mh_m)
        A_m = build_A(N, np.eye(N_MAX + 1), At_m)
        conds.append(np.linalg.cond(A_m))
    print(f"{N:3d}| {conds[0]:.3e}   | {conds[1]:.3e}   | {conds[2]:.3e}")

print("\n" + "=" * 70)
print("C. CODA DESTRA")
print("=" * 70)

E_log_p = np.trapz(log_p * nu_gauss, x)
clr = log_p - E_log_p
print(f"E_nu[log p] = {E_log_p:.4f}")
print(f"log p(a) = {log_p[0]:.4f}   clr(a) = {clr[0]:.4f}")
print(f"log p(b) = {log_p[-1]:.4f}   clr(b) = {clr[-1]:.4f}")
print(f"CLR_TOL del paper = {CFG.CLR_TOL}  ->  |clr|<10 rispettato? "
      f"sx: {abs(clr[0])<10}, dx: {abs(clr[-1])<10}")

# pendenze teoriche delle code (log p ~ -M x a destra, ~ +G x a sinistra)
print(f"\nPendenze teoriche code (unita' std): dx = -M*sigma = {-params['M']*sigma:.3f},"
      f"  sx = +G*sigma = {params['G']*sigma:.3f}")
slope_dx = (log_p[-1] - log_p[-500]) / (x_std[-1] - x_std[-500])
slope_sx = (log_p[499] - log_p[0]) / (x_std[499] - x_std[0])
print(f"Pendenze COS misurate:            dx = {slope_dx:.3f},  sx = {slope_sx:.3f}")

# esponente f al bordo per N=6 e N=16
for N in [6, 16]:
    c_h = c_hats[N]
    fn = lambda z: eval_hermite(np.asarray(z), N)
    f_vals = exponent_func(x_std, c_h, fn)
    C0 = compute_C0(c_h, fn, a, b, m1, sigma)
    p_hat = eval_density(x, c_h, C0, fn, m1, sigma)
    mass_right = np.trapz(p_hat[x_std > 3], x[x_std > 3])
    print(f"\nN={N}: c_N (ultimo coeff) = {c_h[-1]: .6f}")
    print(f"  f(bordo sx {x_std[0]:.2f}) = {f_vals[0]: .3f}   "
          f"f(bordo dx {x_std[-1]:.2f}) = {f_vals[-1]: .3f}")
    print(f"  log p_hat(b) = {np.log(p_hat[-1]):.3f}  vs  log p_cos(b) = {log_p[-1]:.3f}")
    print(f"  massa di p_hat oltre x*=3: {mass_right:.4f}")

# quanto contribuisce il bordo destro all'errore dei coefficienti Fourier?
# c_j Fourier pesati con nu gaussiana: nu(5.88) e' minuscola
print(f"\nnu_gauss al bordo dx (x*={x_std[-1]:.2f}): "
      f"{np.exp(-x_std[-1]**2/2)/np.sqrt(2*np.pi):.3e}")
print(f"nu_logistic al bordo dx: {logistic_weight(x_std[-1]):.3e}")
