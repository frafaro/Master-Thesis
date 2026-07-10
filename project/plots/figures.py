"""
Figure generation for Gambaro (2024) replication.

Figures 1-3:  Coefficient convergence  d2(c^_N, c) vs N  (Hermite & Logistic)
Figure 4:     CLR of three standardized PDFs on common domain
Figures 5-8:  Density convergence distances vs N  (4 subplots each)
Figures 9-12: Density comparison at N=6 and N=16 (PDF and log-PDF)

All figures are saved to the 'output/' subdirectory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for reproducibility
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style settings ────────────────────────────────────────────────────────────
COLORS = {
    "hermite":  "#1f77b4",   # matplotlib default blue
    "logistic": "#d62728",   # red
    "cos":      "black",
}
LS = {"hermite": "-", "logistic": "--", "cos": "-"}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figures 1-3: coefficient convergence ─────────────────────────────────────

def fig_coeff_convergence(d2_hermite_first6: np.ndarray,
                          d2_hermite_all:    np.ndarray,
                          d2_logistic_first6: np.ndarray,
                          d2_logistic_all:    np.ndarray,
                          N_vals: np.ndarray,
                          model_name: str,
                          fig_num: int):
    """
    Figure fig_num (1, 2, or 3) — layout matching Gambaro (2024) Fig. 1.

    Two panels side by side:
      Left  (a): Hermite polynomials
      Right (b): Logistic polynomials

    Each panel has TWO curves:
      - Solid blue  ("first 6 coefficients"):
            sqrt( Σ_{j=1}^{min(6,N)} (ĉj - cj)² )
      - Dashed orange ("all coefficients"):
            sqrt( Σ_{j=1}^{N} (ĉj - cj)² )

    X-axis: "number of moments", range 4–16, ticks at even values.
    Y-axis: linear scale ("square root distance" / "root square distance").
    Legend: upper right.
    """
    # Restrict to N = 4..16 as in the paper
    mask = (N_vals >= 4) & (N_vals <= 16)
    ns   = N_vals[mask]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    panels = [
        (axes[0],
         d2_hermite_first6[mask], d2_hermite_all[mask],
         "Hermite polynomials",
         "square root distance"),
        (axes[1],
         d2_logistic_first6[mask], d2_logistic_all[mask],
         "Logistic polynomials",
         "root square distance"),
    ]

    for ax, d_first6, d_all, title, ylabel in panels:
        ax.plot(ns, d_first6,
                color="tab:blue",   linestyle="-",  linewidth=1.5,
                label="first 6 coefficients")
        ax.plot(ns, d_all,
                color="tab:orange", linestyle="--", linewidth=1.5,
                label="all coefficients")
        ax.set_title(title)
        ax.set_xlabel("number of moments")
        ax.set_ylabel(ylabel)
        ax.set_xlim(4, 16)
        ax.set_xticks(range(4, 17, 2))
        ax.legend(loc="upper right", fontsize=9)
        ax.tick_params(direction="in", which="both")
        ax.grid(False)

    fig.tight_layout()
    _save(fig, f"figure_{fig_num:02d}_coeff_{model_name.lower()}.pdf")


# ── Figure 4: CLR of three PDFs ──────────────────────────────────────────────

def fig4_clr(x_vg: np.ndarray,  clr_vg: np.ndarray,
             x_nig: np.ndarray, clr_nig: np.ndarray,
             x_heston: np.ndarray, clr_heston: np.ndarray):
    """
    Figure 4: CLR of VG, NIG, Heston standardized PDFs on domain with L=4.
    Horizontal tolerance lines at y = ±10.
    x-axis: standardized x*.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_vg,     clr_vg,     label="VG",     color="blue")
    ax.plot(x_nig,    clr_nig,    label="NIG",     color="green")
    ax.plot(x_heston, clr_heston, label="Heston",  color="red")
    ax.axhline(10,  color="k", linestyle="--", linewidth=0.8)
    ax.axhline(-10, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$x^*$")
    ax.set_ylabel(r"$\mathrm{clr}(p)(x^*)$")
    ax.set_title("Figure 4: CLR of three standardized PDFs  (L=4)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    _save(fig, "figure_04_clr_three_pdfs.pdf")


# ── Figures 5-8: density convergence distances ───────────────────────────────

def fig_density_distances(dist_hermite: Dict[str, np.ndarray],
                          dist_logistic: Dict[str, np.ndarray],
                          N_vals: np.ndarray,
                          model_name: str,
                          fig_num: int,
                          logistic_only: bool = False):
    """
    4-subplot figure matching Gambaro (2024) Figs 5-8 layout exactly.

    Layout (2×2):
      (a) top-left:     Aitchison distance     y: "Aitchison distance"
      (b) top-right:    L2 log-pdf distance    y: r"$\mathcal{L}^2$ distance for log-pdf"
      (c) bottom-left:  L1 distance            y: r"$\mathcal{L}^1$ distance"
      (d) bottom-right: L2 distance            y: r"$\mathcal{L}^2$ distance"

    Each subplot:
      - Title: model_name ("VG", "NIG", "Heston …")
      - Hermite: dashed red,  label r"$\hat{p}_N$ Hermite"
      - Logistic: solid blue, label r"$\hat{p}_N$ Logistic"
      - Linear y-scale, no markers
      - X-axis "N", range 2-16
      - (a)/(b)/(c)/(d) centred below each subplot
    """
    keys    = ["aitchison", "log_l2", "l1", "l2"]
    ylabels = [
        "Aitchison distance",
        r"$\mathcal{L}^2$ distance for log-pdf",
        r"$\mathcal{L}^1$ distance",
        r"$\mathcal{L}^2$ distance",
    ]
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    # Restrict to N = 2..16
    mask = (N_vals >= 2) & (N_vals <= 16)
    ns   = N_vals[mask]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, key, ylabel, plabel in zip(axes.flatten(), keys, ylabels, panel_labels):
        if not logistic_only:
            ax.plot(ns, dist_hermite[key][mask],
                    color="tab:red",  linestyle="--", linewidth=1.5,
                    label=r"$\hat{p}_N$ Hermite")
        ax.plot(ns, dist_logistic[key][mask],
                color="tab:blue", linestyle="-",  linewidth=1.5,
                label=r"$\hat{p}_N$ Logistic")
        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel("$N$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(2, 16)
        ax.set_xticks(range(2, 17, 2))
        ax.legend(loc="upper right", fontsize=9)
        ax.tick_params(direction="in", which="both")
        ax.grid(False)
        # panel label centred below x-axis
        ax.text(0.5, -0.18, plabel, transform=ax.transAxes,
                ha="center", va="top", fontsize=12)

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    _save(fig, f"figure_{fig_num:02d}_distances_{model_name.lower()}.pdf")


# ── Figures 9-12: density comparison at fixed N ──────────────────────────────

def fig_density_comparison(x: np.ndarray,
                           p_cos: np.ndarray,
                           p_hermite_6:  np.ndarray,
                           p_hermite_16: np.ndarray,
                           p_logistic_6:  np.ndarray,
                           p_logistic_16: np.ndarray,
                           model_name: str,
                           fig_num: int):
    """
    4-subplot figure comparing true (COS) PDF with exponential expansion.
    Layout:
      (a) top-left:   PDF,     Hermite,  N=6 and N=16 vs COS
      (b) top-right:  log-PDF, Hermite,  N=6 and N=16 vs COS
      (c) bottom-left:  PDF,     Logistic, N=6 and N=16 vs COS
      (d) bottom-right: log-PDF, Logistic, N=6 and N=16 vs COS
    """
    eps = 1e-300
    log_cos  = np.log(np.maximum(p_cos, eps))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    panels = [
        (axes[0, 0], "PDF",     "Hermite",  p_hermite_6,  p_hermite_16,  False),
        (axes[0, 1], "log-PDF", "Hermite",  p_hermite_6,  p_hermite_16,  True),
        (axes[1, 0], "PDF",     "Logistic", p_logistic_6, p_logistic_16, False),
        (axes[1, 1], "log-PDF", "Logistic", p_logistic_6, p_logistic_16, True),
    ]

    for ax, ylabel_type, basis_name, p6, p16, use_log in panels:
        color = COLORS["hermite"] if basis_name == "Hermite" else COLORS["logistic"]
        if use_log:
            y_cos = log_cos
            y6    = np.log(np.maximum(p6,  eps))
            y16   = np.log(np.maximum(p16, eps))
            ax.set_ylabel(r"$\ln\, p(x)$")
        else:
            y_cos = p_cos
            y6    = p6
            y16   = p16
            ax.set_ylabel(r"$p(x)$")

        ax.plot(x, y_cos, color=COLORS["cos"],  lw=2,   linestyle="-",  label="COS (true)")
        ax.plot(x, y6,    color=color,           lw=1.5, linestyle="--", label=f"{basis_name} N=6",  alpha=0.85)
        ax.plot(x, y16,   color=color,           lw=1.5, linestyle=":",  label=f"{basis_name} N=16", alpha=0.85)
        ax.set_xlabel("x")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(f"Figure {fig_num}: Density comparison — {model_name}", y=1.01)
    fig.tight_layout()
    _save(fig, f"figure_{fig_num:02d}_density_{model_name.lower()}.pdf")


# ── Table 2: CPU times ────────────────────────────────────────────────────────

def print_table2(times: Dict[str, Dict[str, float]]):
    """
    Print and save Table 2 (CPU times in seconds, N=16).
    times: {'VG': {'cos': t, 'hermite': t, 'logistic': t}, 'NIG': ..., 'Heston': ...}
    """
    header = f"{'Model':<10} {'COS':>10} {'Hermite':>12} {'Logistic':>12}"
    rows = []
    for model in [m for m in ["VG", "NIG", "Heston"] if m in times]:
        t = times[model]
        rows.append(f"{model:<10} {t['cos']:>10.4f} {t['hermite']:>12.4f} {t['logistic']:>12.4f}")
    table_str = "\n".join(["", "Table 2 — CPU times (seconds), N=16", "-" * 48, header, "-" * 48] + rows + ["-" * 48, ""])
    print(table_str)
    path = os.path.join(OUTPUT_DIR, "table_2_cpu_times.txt")
    with open(path, "w") as f:
        f.write(table_str)
    print(f"  Saved: {path}")
