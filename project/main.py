"""
main.py — Full replication pipeline for Gambaro (2024).

Runs the complete experiment for VG, NIG, and Heston models,
producing Figures 1-12 and Table 2.

Usage:
    cd project/
    python main.py [--model VG|NIG|Heston|all] [--calibrate]

The --calibrate flag re-runs the Heston parameter calibration
(slow; results are cached in config.py HESTON_PARAMS after first run).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np

import config as CFG
from utils.quadrature import cumulant_domain, make_grid, clr_domain
from utils.timing import timed

# Models
import models.variance_gamma as vg_mod
import models.nig as nig_mod
import models.heston as heston_mod

# Basis
from basis.hermite import eval_hermite, gaussian_weight
from basis.logistic import logistic_weight, eval_logistic_recurrence

# Moments
from moments.hermite_moments import hermite_moments_from_raw, verify_hermite_moments

# Matrices
from matrices.basis_matrices import build_H, build_logistic_and_Q
from matrices.change_of_basis import build_Q_hermite
from matrices.linear_system import build_A_tilde, build_A, build_b, solve_system

# Expansion
from expansion.density import compute_C0, eval_density

# COS
from cos.cos_method import cos_density, benchmark_fourier_coeffs, verify_cos_density

# Distances
from distances.metrics import d2_coeff, all_distances

# Plots
from plots.figures import (
    fig_coeff_convergence, fig4_clr, fig_density_distances,
    fig_density_comparison, print_table2,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_eval_hermite_fn(N, m1, sigma):
    """Return callable x -> basis array for Hermite basis."""
    def fn(x):
        x_std = (np.asarray(x) - m1) / sigma
        return eval_hermite(x_std, N)
    return fn


def make_eval_hermite_std_fn(N):
    """Return callable x* -> basis array for Hermite basis (standardized input)."""
    def fn(x_std):
        return eval_hermite(np.asarray(x_std), N)
    return fn


def make_eval_logistic_std_fn(alpha_L, beta_L, N):
    """Return callable x* -> basis array for Logistic basis (standardized input)."""
    def fn(x_std):
        return eval_logistic_recurrence(np.asarray(x_std), N, alpha_L, beta_L)
    return fn


def gaussian_weight_original(x, m1, sigma):
    """Gaussian weight nu(x) = omega((x-m1)/sigma)/sigma (integrates to 1 over x)."""
    z = (x - m1) / sigma
    return np.exp(-z**2 / 2.0) / (np.sqrt(2.0 * np.pi) * sigma)


def logistic_weight_original(x, m1, sigma):
    """Logistic weight nu_L((x-m1)/sigma)/sigma (integrates to 1 over x)."""
    z = (x - m1) / sigma
    return logistic_weight(z) / sigma


# ── Core pipeline for one model ───────────────────────────────────────────────

def run_model(model_name: str, cf_func, raw_moments_func, params: dict,
              fig_nums: dict, timing_results: dict):
    """
    Full pipeline for a single model.

    Parameters
    ----------
    model_name  : 'VG', 'NIG', or 'Heston'
    cf_func     : characteristic function  phi(u, params)
    raw_moments_func: function(params, max_order) -> raw moment array
    params      : model parameter dict
    fig_nums    : {'coeff': int, 'dist_full': int, 'dist_restr': int (or None),
                   'dens_full': int, 'dens_restr': int (or None)}
    timing_results: dict to update with CPU times

    The function:
    1. Computes raw moments and Hermite moments
    2. Builds H_n, B_n (Hermite and Logistic), Q_n
    3. Builds and solves the linear system for each N in 1..N_MAX
    4. Evaluates the COS density on the domain
    5. Computes all distances vs N
    6. Generates all figures
    """
    N_MAX  = CFG.N_MAX
    N_COS  = CFG.N_COS
    L      = CFG.L
    GRID   = CFG.GRID_SIZE
    is_heston = (model_name == "Heston")

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    # ── 1. Raw moments ────────────────────────────────────────────────────────
    print("  [1] Computing raw moments...")
    raw_mu = raw_moments_func(params, max_order=2 * N_MAX + 2)
    m1    = raw_mu[1]
    m2    = raw_mu[2]
    sigma = np.sqrt(m2 - m1**2)
    print(f"      m1={m1:.4f}, sigma={sigma:.4f}")

    # ── 2. Cumulants for domain ───────────────────────────────────────────────
    kap = np.zeros(5)
    kap[1] = raw_mu[1]
    kap[2] = raw_mu[2] - raw_mu[1]**2
    kap[3] = raw_mu[3] - 3*raw_mu[2]*raw_mu[1] + 2*raw_mu[1]**3
    # 4th cumulant: k4 = mu4 - 4*mu3*mu1 - 3*mu2^2 + 12*mu2*mu1^2 - 6*mu1^4
    kap[4] = (raw_mu[4] - 4*raw_mu[3]*raw_mu[1]
              - 3*raw_mu[2]**2
              + 12*raw_mu[2]*raw_mu[1]**2
              - 6*raw_mu[1]**4)
    skewness = kap[3] / kap[2]**1.5
    ex_kurt  = kap[4] / kap[2]**2
    print(f"      skewness={skewness:.3f}, excess kurtosis={ex_kurt:.3f}")

    # ── 3. Domain (full, L=4) ─────────────────────────────────────────────────
    cumulants_dict = {"k1": kap[1], "k2": kap[2], "k4": kap[4]}
    a_full, b_full = cumulant_domain(cumulants_dict, L=L)
    x_full  = make_grid(a_full, b_full, GRID)
    print(f"      Domain [a,b] = [{a_full:.3f}, {b_full:.3f}]")

    # ── 4. COS density on full domain ─────────────────────────────────────────
    print("  [4] Computing COS density...")
    def cf(u):
        return cf_func(u, params)

    t_cos_start = __import__("time").perf_counter()
    p_cos_full = cos_density(x_full, cf, a_full, b_full, N_cos=N_COS)
    t_cos = __import__("time").perf_counter() - t_cos_start
    timing_results[model_name] = {"cos": t_cos}
    print(f"      COS time: {t_cos:.4f}s")

    log_p_cos_full = np.log(np.maximum(p_cos_full, 1e-300))

    # ── 5. Hermite moments ────────────────────────────────────────────────────
    print("  [5] Computing Hermite moments...")
    mh, _, _ = hermite_moments_from_raw(raw_mu, K_max=2 * N_MAX)
    verify_hermite_moments(mh[:3])

    # ── 6. Build basis matrices ───────────────────────────────────────────────
    print("  [6] Building basis matrices (Q_n)...")
    Q_hermite           = np.eye(N_MAX + 1)   # For Hermite basis, Q = I
    alpha_L, beta_L, Q_logistic = build_logistic_and_Q(N_MAX)
    print(f"      Q_logistic shape: {Q_logistic.shape}, max |diag|: {np.abs(np.diag(Q_logistic)).max():.3f}")

    # ── 7. Solve systems for all N ────────────────────────────────────────────
    print(f"  [7] Solving linear systems for N=1..{N_MAX}...")
    c_hats_hermite  = []
    c_hats_logistic = []
    t_hermite_N16  = 0.0
    t_logistic_N16 = 0.0

    for N in range(1, N_MAX + 1):
        At_N = build_A_tilde(N, mh)
        b_N  = build_b(N, mh)

        # Hermite
        A_herm = build_A(N, Q_hermite, At_N)
        c_h, _ = solve_system(A_herm, b_N)
        c_hats_hermite.append(c_h)

        # Logistic
        A_log = build_A(N, Q_logistic, At_N)
        c_l, _ = solve_system(A_log, b_N)
        c_hats_logistic.append(c_l)

        if N == 16:
            # Timed run for Table 2
            def _time_hermite():
                At = build_A_tilde(16, mh)
                b  = build_b(16, mh)
                A  = build_A(16, Q_hermite, At)
                return solve_system(A, b)
            _, t_hermite_N16 = timed(_time_hermite)

            def _time_logistic():
                At = build_A_tilde(16, mh)
                b  = build_b(16, mh)
                A  = build_A(16, Q_logistic, At)
                return solve_system(A, b)
            _, t_logistic_N16 = timed(_time_logistic)

    timing_results[model_name]["hermite"]  = t_hermite_N16
    timing_results[model_name]["logistic"] = t_logistic_N16

    # ── 8. Benchmark Fourier coefficients (from COS) ──────────────────────────
    print("  [8] Computing benchmark Fourier coefficients from COS...")

    def eval_h_std(x_std):
        return eval_hermite(x_std, N_MAX)
    def eval_l_std(x_std):
        return eval_logistic_recurrence(x_std, N_MAX, alpha_L, beta_L)

    nu_gauss = gaussian_weight_original(x_full, m1, sigma)
    nu_logis = logistic_weight_original(x_full, m1, sigma)

    c_exact_hermite  = benchmark_fourier_coeffs(
        log_p_cos_full, x_full, eval_h_std, nu_gauss, m1, sigma, N_MAX)
    c_exact_logistic = benchmark_fourier_coeffs(
        log_p_cos_full, x_full, eval_l_std, nu_logis, m1, sigma, N_MAX)

    # ── 9. Coefficient convergence distances ──────────────────────────────────
    print("  [9] Computing coefficient convergence distances...")
    N_vals = np.arange(1, N_MAX + 1)
    d2_h = np.array([d2_coeff(c_hats_hermite[n-1],  c_exact_hermite[:n])  for n in N_vals])
    d2_l = np.array([d2_coeff(c_hats_logistic[n-1], c_exact_logistic[:n]) for n in N_vals])

    fig_coeff_convergence(d2_h, d2_l, N_vals, model_name, fig_nums["coeff"])

    # ── 10. Evaluate densities and CLR ────────────────────────────────────────
    print("  [10] Evaluating exponential expansion densities...")

    def eval_hermite_N(N, x):
        x_std = (x - m1) / sigma
        return eval_hermite(x_std, N)

    def eval_logistic_N(N, x):
        x_std = (x - m1) / sigma
        return eval_logistic(x_std, N, B_L_mat)

    def get_density_hermite(N, x, a, b):
        c_h = c_hats_hermite[N - 1]
        fn  = make_eval_hermite_std_fn(N)
        C0  = compute_C0(c_h, fn, a, b, m1, sigma)
        return eval_density(x, c_h, C0, fn, m1, sigma)

    def get_density_logistic(N, x, a, b):
        c_l = c_hats_logistic[N - 1]
        fn  = make_eval_logistic_std_fn(alpha_L, beta_L, N)
        C0  = compute_C0(c_l, fn, a, b, m1, sigma)
        return eval_density(x, c_l, C0, fn, m1, sigma)

    # ── 11. Figure 4: CLR ─────────────────────────────────────────────────────
    # (CLR figure is assembled in the calling loop after all models are run)

    # ── 12. Density convergence distances (full domain) ───────────────────────
    print("  [12] Computing density convergence distances (full domain)...")
    dist_h_full  = {k: [] for k in ["aitchison", "log_l2", "l1", "l2"]}
    dist_l_full  = {k: [] for k in ["aitchison", "log_l2", "l1", "l2"]}

    for N in N_vals:
        ph = get_density_hermite(N,  x_full, a_full, b_full)
        pl = get_density_logistic(N, x_full, a_full, b_full)
        dh = all_distances(x_full, ph, p_cos_full, nu_gauss)
        dl = all_distances(x_full, pl, p_cos_full, nu_logis)
        for k in dh:
            dist_h_full[k].append(dh[k])
            dist_l_full[k].append(dl[k])

    for k in dist_h_full:
        dist_h_full[k] = np.array(dist_h_full[k])
        dist_l_full[k] = np.array(dist_l_full[k])

    # ── 13. Figures 5/6/8 (full domain) ──────────────────────────────────────
    if not is_heston:
        fig_density_distances(dist_h_full, dist_l_full, N_vals,
                              model_name, fig_nums["dist_full"])
    else:
        # Fig 8: Heston full domain, logistic only
        fig_density_distances(dist_h_full, dist_l_full, N_vals,
                              model_name + " (L=4)", fig_nums["dist_full"],
                              logistic_only=True)

    # ── 14. Figures 7/11 (Heston restricted domain) ──────────────────────────
    if is_heston and fig_nums.get("dist_restr") is not None:
        print("  [14] Computing restricted-domain distances for Heston...")
        a_restr, b_restr = clr_domain(
            lambda x: np.log(np.maximum(cos_density(x, cf, a_full, b_full, N_COS), 1e-300)),
            cumulants_dict, L_start=L, clr_tol=CFG.CLR_TOL)
        x_restr = make_grid(a_restr, b_restr, GRID)
        p_cos_restr = cos_density(x_restr, cf, a_full, b_full, N_COS)
        nu_gauss_restr = gaussian_weight_original(x_restr, m1, sigma)
        nu_logis_restr = logistic_weight_original(x_restr, m1, sigma)
        print(f"      Restricted domain: [{a_restr:.3f}, {b_restr:.3f}]")

        dist_h_restr = {k: [] for k in ["aitchison", "log_l2", "l1", "l2"]}
        dist_l_restr = {k: [] for k in ["aitchison", "log_l2", "l1", "l2"]}
        for N in N_vals:
            ph = get_density_hermite(N,  x_restr, a_restr, b_restr)
            pl = get_density_logistic(N, x_restr, a_restr, b_restr)
            dh = all_distances(x_restr, ph, p_cos_restr, nu_gauss_restr)
            dl = all_distances(x_restr, pl, p_cos_restr, nu_logis_restr)
            for k in dh:
                dist_h_restr[k].append(dh[k])
                dist_l_restr[k].append(dl[k])
        for k in dist_h_restr:
            dist_h_restr[k] = np.array(dist_h_restr[k])
            dist_l_restr[k] = np.array(dist_l_restr[k])

        fig_density_distances(dist_h_restr, dist_l_restr, N_vals,
                              model_name + " (restricted domain)",
                              fig_nums["dist_restr"])

    # ── 15. Density comparison figures (N=6 and N=16) ─────────────────────────
    print("  [15] Generating density comparison figures...")
    N6, N16 = CFG.N_FIXED

    ph6  = get_density_hermite(N6,   x_full, a_full, b_full)
    ph16 = get_density_hermite(N16,  x_full, a_full, b_full)
    pl6  = get_density_logistic(N6,  x_full, a_full, b_full)
    pl16 = get_density_logistic(N16, x_full, a_full, b_full)

    fig_density_comparison(x_full, p_cos_full,
                           ph6, ph16, pl6, pl16,
                           model_name, fig_nums["dens_full"])

    if is_heston and fig_nums.get("dens_restr") is not None:
        ph6_r  = get_density_hermite(N6,   x_restr, a_restr, b_restr)
        ph16_r = get_density_hermite(N16,  x_restr, a_restr, b_restr)
        pl6_r  = get_density_logistic(N6,  x_restr, a_restr, b_restr)
        pl16_r = get_density_logistic(N16, x_restr, a_restr, b_restr)
        fig_density_comparison(x_restr, p_cos_restr,
                               ph6_r, ph16_r, pl6_r, pl16_r,
                               model_name + " (restricted)", fig_nums["dens_restr"])

    # Return data needed for Figure 4
    x_std_full = (x_full - m1) / sigma
    clr_p = log_p_cos_full - np.trapezoid(log_p_cos_full * nu_gauss, x_full)
    return {
        "x_std": x_std_full,
        "clr":   clr_p,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    models_to_run = args.models

    timing = {}

    clr_data = {}

    if "VG" in models_to_run or "all" in models_to_run:
        clr_data["VG"] = run_model(
            model_name="VG",
            cf_func=vg_mod.characteristic_function,
            raw_moments_func=vg_mod.raw_moments,
            params=CFG.VG_PARAMS,
            fig_nums={"coeff": 1, "dist_full": 5, "dens_full": 9},
            timing_results=timing,
        )

    if "NIG" in models_to_run or "all" in models_to_run:
        clr_data["NIG"] = run_model(
            model_name="NIG",
            cf_func=nig_mod.characteristic_function,
            raw_moments_func=nig_mod.raw_moments,
            params=CFG.NIG_PARAMS,
            fig_nums={"coeff": 2, "dist_full": 6, "dens_full": 10},
            timing_results=timing,
        )

    if "Heston" in models_to_run or "all" in models_to_run:
        heston_params = CFG.HESTON_PARAMS
        if args.calibrate:
            print("\nCalibrating Heston parameters...")
            heston_params = heston_mod.calibrate_to_moments(
                target_skew=-1.2, target_kurt=2.5, x0=heston_params)
            print(f"  Calibrated: {heston_params}")

        clr_data["Heston"] = run_model(
            model_name="Heston",
            cf_func=heston_mod.characteristic_function,
            raw_moments_func=heston_mod.raw_moments,
            params=heston_params,
            fig_nums={"coeff": 3, "dist_full": 8, "dist_restr": 7,
                      "dens_full": 12, "dens_restr": 11},
            timing_results=timing,
        )

    # ── Figure 4: CLR overlay ────────────────────────────────────────────────
    if len(clr_data) == 3:
        print("\nGenerating Figure 4 (CLR overlay)...")
        fig4_clr(
            clr_data["VG"]["x_std"],     clr_data["VG"]["clr"],
            clr_data["NIG"]["x_std"],    clr_data["NIG"]["clr"],
            clr_data["Heston"]["x_std"], clr_data["Heston"]["clr"],
        )

    # ── Table 2 ─────────────────────────────────────────────────────────────
    if timing:
        print_table2(timing)

    print("\nDone. All figures saved to project/output/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gambaro (2024) replication")
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=["VG", "NIG", "Heston", "all"],
                        help="Which models to run (default: all)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Re-calibrate Heston parameters to target moments")
    args = parser.parse_args()
    main(args)
