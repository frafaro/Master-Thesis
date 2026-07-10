# Gambaro (2024) — Full Replication: Pipeline Documentation

**Paper**: Anna Maria Gambaro, *"Exponential expansions for approximation of probability
distributions"*, Decisions in Economics and Finance (2024).
DOI: [10.1007/s10203-024-00460-2](https://doi.org/10.1007/s10203-024-00460-2)

**Goal**: Replicate all numerical experiments from Section 5 of the paper —
Figures 1–12 and Table 2 — for Variance Gamma (VG), Normal Inverse Gaussian (NIG),
and Heston log-return densities.

---

## 0. Quick Start

```bash
cd project/
pip install -r requirements.txt
python main.py --models all          # full run (~35 seconds)
python main.py --models VG           # only VG (fastest, ~8 s)
python main.py --models all --calibrate   # re-calibrate Heston (slower)
```

All output PDF figures and `table_2_cpu_times.txt` land in `project/output/`.

---

## 1. Mathematical Background

### 1.1 Bayesian Hilbert Space (Bayes Space)

The paper works in the **Bayes space** B²(I, ν), the space of strictly positive,
log-square-integrable functions on domain I with reference weight ν.  B² is
isometrically isomorphic to L²(I, ν) via the **centered log-ratio** (CLR)
transform:

```
clr(p)(x) = log p(x) − ∫_I log p(x) ν(x) dx       (eq. 2)
```

The Aitchison distance in B² equals the L²-distance between CLRs:

```
d_A(p̂_N, p) = √[ ∫_I (clr(p̂_N)(x) − clr(p)(x))² ν(x) dx ]   (eq. 18)
```

### 1.2 The Exponential Expansion (eq. 7 / eq. 16)

Given an orthonormal basis {φ_j} for L²(I, ν):

```
p̂_N(x) = Ĉ₀ · exp( Σ_{j=1}^N ĉʲ_N · φ_j(x*) )
```

where:
- `x* = (x − m₁)/σ` is the standardized variable, m₁=E[X], σ=√Var(X)
- `Ĉ₀ = ( ∫_I exp(Σ ĉ_j φ_j(x*)) dx )⁻¹` is the normalization constant (eq. 8)
- The coefficients `ĉ_j` are estimated by solving a linear system

The expansion always produces **strictly positive** densities (positivity guaranteed
by the exponential).

### 1.3 Coefficient Estimation — Linear System (eq. 15)

The coefficients ĉ_j satisfy the N×N linear system:

```
A_N · ĉ_N = b_N
```

where:

**A~_N (Hermite moment matrix)**:
```
Ã_N[i,j] = Σ_{k=0}^{i+j−2}  (1/k!) · Δ_{i−1, j−1, k} · mʰ_k
```

**Δ_{p,q,r} (triple Hermite integral coefficient)**:
```
Δ_{p,q,r} = p! q! r! / ((b−p)! (b−q)! (b−r)!)    if p+q+r even, b=(p+q+r)/2 ≥ max(p,q,r)
           = 0                                        otherwise
```
(Classical result: ∫ He_p He_q He_r ω dx with ω = Gaussian weight)

**A_N (full system matrix)**:
```
A_N[i,n] = Σ_{j=0}^{n} √j · q_{n,j} · Ã_N[i,j]
```

where q_{n,j} are the expansion coefficients of φ_n in the Hermite basis {h_j}:
φ_n(x*) = Σ_j q_{n,j} h_j(x*)  (eq. 13)

**b_N (right-hand side)**:
```
b_N[i] = −√(i−1) · mʰ_{i−1}    for i = 1,...,N
```

**Hermite moments mʰ_k**:
```
mʰ_k = E_p[h_k(X*)] = E_p[He_k(X*)] / √(k!)
     = (1/√k!) · Σ_{j=0}^{k} [coeff of xʲ in He_k] · μ*_j
```
where μ*_j = E[(X*)ʲ] are the standardized raw moments (Rompolis & Tzavalis 2008).

**Key simplification for Hermite basis**: when φ_j = h_j, the change-of-basis
matrix Q_n = I (identity), so:
```
A_N[i,n] = √n · Ã_N[i,n]
```

---

## 2. Project Structure

```
project/
├── config.py                 # All parameters, domain settings, grid sizes
├── main.py                   # Orchestrates the full pipeline for all models
│
├── models/
│   ├── variance_gamma.py     # VG: characteristic function, cumulants (CGF), moments
│   ├── nig.py                # NIG: characteristic function, CGF derivatives, moments
│   └── heston.py             # Heston: Gatheral CF, mpmath cumulants, calibration
│
├── basis/
│   ├── hermite.py            # Hermite polynomials: recurrence, normalization, eval
│   └── logistic.py           # Logistic polynomials: Stieltjes algorithm, recurrence eval
│
├── moments/
│   └── hermite_moments.py    # Standardize moments, compute mʰ_k from raw moments
│
├── matrices/
│   ├── basis_matrices.py     # Build H_n, call logistic builder + Q
│   ├── change_of_basis.py    # Compute Q_n via Gauss-Hermite inner products
│   └── linear_system.py      # Δ coefficients, Ã_N, A_N, b_N, solve A c = b
│
├── expansion/
│   └── density.py            # Evaluate exponent, compute Ĉ₀, evaluate p̂_N
│
├── cos/
│   └── cos_method.py         # COS density benchmark (Fang & Oosterlee 2009)
│                             # Benchmark Fourier coefficients c_j via eq. (9)
│
├── distances/
│   └── metrics.py            # d₂(ĉ,c), Aitchison, log-L2, L1, L2 distances
│
├── plots/
│   └── figures.py            # fig_coeff_convergence, fig4_clr, fig_density_distances,
│                             #   fig_density_comparison, print_table2
│
├── utils/
│   ├── quadrature.py         # cumulant_domain (eq. 22), clr_domain, make_grid
│   └── timing.py             # CPU timing for Table 2
│
├── output/                   # Generated PDF figures and table
│   ├── figure_01_coeff_vg.pdf     ... figure_12_density_heston.pdf
│   └── table_2_cpu_times.txt
│
└── requirements.txt
```

---

## 3. Chronological Pipeline (per model)

### STEP 1 — Model Parameters (`config.py`)

| Model | Parameters | Source |
|-------|-----------|--------|
| VG | sigma=0.2, nu=2/3, theta=0, mu=0 | Heston & Rossi (2016): skewness=0, excess kurtosis=3ν=2 |
| NIG | mu=0, theta=0.05, sigma=0.2, kappa=0.3, dt=1 | Gambaro (2024) Appendix A |
| Heston | kappa=2.0015, theta=0.04785, xi=0.40299, rho=−0.76635, v0=0.05879, T=1 | Calibrated to Rompolis & Tzavalis (2008): skewness=−1.2, excess kurtosis=2.5 |

NIG gives skewness≈0.22, excess kurtosis≈0.97 — approximately matching Table 1.

---

### STEP 2 — Raw Moments (`models/*.py`)

Each model exposes `raw_moments(params, max_order)` → array `μ[k] = E[Xᵏ]`.

**VG** (`variance_gamma.py`):
The Cumulant Generating Function is:
```
K(s) = μ·s − (1/ν)·log(1 − ν·θ·s − ν·σ²·s²/2)
```
Cumulants κ_k = K^(k)(0) are computed exactly via the recurrence
`d^k/ds^k log(f(s)) |_{s=0}` (Faà di Bruno formula), where f is a degree-2 polynomial.
Raw moments follow from the moment-cumulant recursion:
```
μ_n = Σ_{k=1}^n C(n−1,k−1) · κ_k · μ_{n−k}
```

**NIG** (`nig.py`):
```
K(s) = μ·s + (1/κ)·(1 − √(1 − 2θκs − σ²κs²))
```
Let g(s) = 1 − 2θκs − σ²κs². Derivatives of √g at s=0 are computed via the
Leibniz identity for (√g)² = g. Same moment-cumulant recursion then gives μ_k.

**Heston** (`heston.py`):
The Gatheral-form Characteristic Function (avoids branch-cut issues):
```
φ(u) = exp(A(u,T) + B(u,T)·v₀)
d    = √((κ−ρξiu)² + ξ²(iu+u²))
g    = (κ−ρξiu−d) / (κ−ρξiu+d)
A    = iu·r·T + (κθ/ξ²)·[(κ−ρξiu−d)·T − 2·log((1−g·e^{−dT})/(1−g))]
B    = (κ−ρξiu−d)/ξ² · (1−e^{−dT})/(1−g·e^{−dT})
```
Cumulants are computed by differentiating the log-MGF K(s) = log φ(−is) using
**mpmath** at 50-digit precision (prevents catastrophic cancellation in high-order
finite differences). κ_k = K^(k)(0) obtained via `mpmath.diff`.

Heston parameters were calibrated numerically (Nelder-Mead) to achieve
skewness=−1.2 and excess kurtosis=2.5 exactly.

---

### STEP 3 — Domain Truncation (`utils/quadrature.py`)

Domain I = [a, b] following eq. (22) with L=4:
```
a = k₁ − 4·√(k₂ + √k₄)
b = k₁ + 4·√(k₂ + √k₄)
```
where k₁, k₂, k₄ are the 1st, 2nd, 4th cumulants of the distribution.

For the **Heston restricted domain** (used in Figs. 7 and 11): the domain is
narrowed until log(p_COS(x)) > −10 everywhere on I (heuristic from the paper:
|clr(p)| < 10 for all x ∈ I, equivalently p(x) > 5×10⁻⁵).

---

### STEP 4 — COS Benchmark Density (`cos/cos_method.py`)

Following Fang & Oosterlee (2009) with N_COS = 2¹² = 4096 terms:
```
p(x) ≈ (2/(b−a)) · Σ_{k=0}^{N−1} ' Re[φ(kπ/(b−a)) · e^{−ikπa/(b−a)}] · cos(kπ(x−a)/(b−a))
```
where ' denotes that the k=0 term has weight 1/2.

The COS density is used as the "true" density benchmark.

---

### STEP 5 — Hermite Moments (`moments/hermite_moments.py`)

Following Rompolis & Tzavalis (2008), eqs. (2)–(3):

1. Standardize raw moments: `μ*_j = E[(X*)ʲ]` from raw moments via binomial expansion.
2. Express mʰ_k using the polynomial form of He_k:
   ```
   mʰ_k = (1/√k!) · Σ_{j=0}^{k} [He_k coefficient of xʲ] · μ*_j
   ```

**Verification** (automatically checked):
- mʰ₀ = 1  (E[h₀] = E[1] = 1)
- mʰ₁ = 0  (E[X*] = 0 by construction)
- mʰ₂ = 0  (E[X*²] = Var(X*) = 1, so E[He₂(X*)] = E[X*²−1] = 0)

---

### STEP 6 — Basis Construction

#### Hermite Basis (`basis/hermite.py`)

Normalized probabilist's Hermite polynomials:
```
h_j(x*) = He_j(x*) / √(j!)
```
Recurrence (stable, avoids cancellation):
```
He_0 = 1,  He_1 = x,  He_{n+1}(x) = x·He_n(x) − n·He_{n-1}(x)
```
In normalized form:
```
h_{k+1}(x) = (x·h_k(x) − √k·h_{k-1}(x)) / √(k+1)
```
Key derivative property (used in linear system derivation):
```
h'_j(x*) = √j · h_{j-1}(x*)
```

Monomial coefficient matrix `H_n[i,j]` = coefficient of xʲ in h_{i}(x)
is computed symbolically by expanding the recurrence.

#### Logistic Basis (`basis/logistic.py`)

Standard logistic weight:
```
ν_L(x) = exp(−x) / (1+exp(−x))² = sech²(x/2)/4    (mean=0, variance=π²/3)
```

**Algorithm: Stieltjes/Lanczos** (avoids the ill-conditioned Hankel moment matrix).
Starting from monic polynomials π_k:
```
π_0(x) = 1
π_{k+1}(x) = (x − α_k)·π_k(x) − β_k·π_{k-1}(x)
```
where (for symmetric ν_L: α_k = 0 for all k):
```
β_k = ‖π_k‖²_νL / ‖π_{k-1}‖²_νL
```
computed on a 3000-point Gauss-Legendre grid adapted to ν_L.

Normalized polynomials: `L_k = π_k / ‖π_k‖_νL`.

Evaluation recurrence (numerically stable for any N and any x):
```
L_0(x) = 1
L_1(x) = (x − α_0) · L_0(x) / √β₁
L_{k+1}(x) = [(x − α_k)·L_k(x) − √β_k·L_{k-1}(x)] / √β_{k+1}
```

---

### STEP 7 — Change-of-Basis Matrix Q_n (`matrices/change_of_basis.py`)

For each basis polynomial φ_n, we need:
```
q_{n,j} = ⟨φ_n, h_j⟩_ω = ∫ φ_n(x*) · h_j(x*) · ω(x*) dx*
```
(eq. 13, decomposition of φ_n in the Hermite basis)

**Hermite basis**: Q_n = I (identity — φ_j = h_j, so q_{j,j} = 1).

**Logistic basis**: computed via 200-point Gauss-Hermite quadrature:
```
Q_logistic[k,j] ≈ Σᵢ w_GH[i] · L_k(x_GH[i]) · h_j(x_GH[i])
```
This completely bypasses the monomial coefficient representation of L_k,
which is numerically unstable for degree ≥ 10 (catastrophic cancellation).

---

### STEP 8 — Linear System Construction and Solution (`matrices/linear_system.py`)

For each N = 1, 2, ..., N_MAX = 20:

1. **Build Ã_N** (N×N): nested loops over i,j,k with precomputed Δ_{p,q,r}.
2. **Build A_N** (N×N): weighted combination using Q_n and Ã_N.
3. **Build b_N** (N-vector): from Hermite moments.
4. **Solve**: `scipy.linalg.solve` (LU decomposition, O(N³)).
   Falls back to `numpy.linalg.lstsq` if cond(A_N) > 10¹⁴ (ill-conditioned
   logistic systems at high N).

**Verification**:
- For Hermite: A_N[i,n] = √n · Ã_N[i,n] (diagonal property — auto-verified)
- Residual ‖A_N ĉ − b_N‖ / ‖b_N‖ < 10⁻¹⁰ (checked internally)

**Note (footnote 4 of paper)**: the coefficients ĉ_N **do NOT depend** on the
truncated domain I — domain truncation only affects COS and distance computations.

---

### STEP 9 — Normalization Constant Ĉ₀ (`expansion/density.py`)

```
Ĉ₀ = 1 / ( σ · ∫_{I*} exp(Σ_{j=1}^N ĉ_j · φ_j(t)) dt )
```
where I* = [(a−m₁)/σ, (b−m₁)/σ] is the standardized domain.

**Numerical method**: trapezoidal rule on 20,000 uniform points in I*.
Log-sum-exp trick prevents overflow: subtract max(f) before exponentiating,
compensate afterwards.

**Verification**: ∫_I p̂_N(x) dx ≈ 1 (tolerance 10⁻⁴).

---

### STEP 10 — Benchmark Fourier Coefficients from COS (`cos/cos_method.py`)

The exact coefficients c_j (eq. 9) are estimated via:
```
c_j = ∫_I clr(p)(x) · φ_j(x*) · ν(x*) dx
```
where `clr(p)(x) = log(p_COS(x)) − E_ν[log(p_COS(X))]`.

Integration via the trapezoidal rule on the same density grid.
These serve as the "true" reference in the coefficient distance d₂(ĉ_N, c).

---

### STEP 11 — Distance Metrics (`distances/metrics.py`)

All distances computed via trapezoidal rule on the truncated domain I.

| Equation | Name | Formula |
|----------|------|---------|
| (17) | Coefficient distance | d₂(ĉ_N,c) = Σ_{j=1}^N (ĉʲ_N − c_j)² |
| (18) | Aitchison distance | d_A = √[ ∫ (clr(p̂_N)−clr(p))² ν dx ] |
| (19) | L²-log distance | d₂(log) = √[ ∫ (log p̂_N−log p)² dx ] |
| (20) | L¹ distance | d₁ = ∫ |p̂_N−p| dx |
| (21) | L² distance | d₂ = √[ ∫ (p̂_N−p)² dx ] |

For the Aitchison distance:
```
clr(p̂_N)(x) = Σ_{j=1}^N ĉ_j φ_j(x*) + log Ĉ₀ − E_ν[Σ ĉ_j φ_j(X*) + log Ĉ₀]
```
The E_ν[·] term is computed numerically via trapezoidal integration.

---

### STEP 12 — Figure Generation (`plots/figures.py`)

| Figure | Content | Function | Key Parameters |
|--------|---------|----------|----------------|
| 1 (VG) | Coefficient d₂ vs N, Hermite & Logistic | `fig_coeff_convergence` | N=1..20 |
| 2 (NIG) | Same for NIG | `fig_coeff_convergence` | — |
| 3 (Heston) | Same for Heston | `fig_coeff_convergence` | — |
| 4 | CLR of 3 PDFs, tolerance ±10 | `fig4_clr` | x-axis = x* |
| 5 (VG) | 4 distances vs N, Hermite & Logistic | `fig_density_distances` | L=4 domain |
| 6 (NIG) | Same for NIG | `fig_density_distances` | — |
| 7 (Heston) | 4 distances vs N, Hermite & Logistic | `fig_density_distances` | restricted domain |
| 8 (Heston) | 4 distances vs N, Logistic only | `fig_density_distances` | L=4, logistic_only=True |
| 9 (VG) | PDF & log-PDF, N=6 & N=16 | `fig_density_comparison` | — |
| 10 (NIG) | Same for NIG | `fig_density_comparison` | — |
| 11 (Heston) | Same for Heston restricted | `fig_density_comparison` | restricted domain |
| 12 (Heston) | Same for Heston L=4 | `fig_density_comparison` | full domain |

**Table 2**: CPU times for COS, Hermite N=16, Logistic N=16, per model.

All figures saved as PDF in `project/output/`.

---

## 4. Module Interaction Diagram

```
config.py  ─────────────────────────────────────────────────────────┐
                                                                     │
models/                                                              │
  *.characteristic_function(u)  ──────────────────────── cos/        │
  *.raw_moments(params, K)  ──────────────────────────── moments/    │
                                      │                              │
                                      ▼                              │
                              moments/hermite_moments.py             │
                                  mh[k] = E[h_k(X*)]                │
                                      │                              │
                    ┌─────────────────┼─────────────────────┐        │
                    ▼                 ▼                     ▼        │
              basis/hermite.py  basis/logistic.py    utils/quadrature.py
              H_n matrix        Stieltjes α,β         domain [a,b]
                    │                 │                     │
                    └────────┬────────┘                     │
                             ▼                              │
                     matrices/                              │
                       change_of_basis.py ──── Q_n          │
                       linear_system.py  ──── Ã_N, A_N,     │
                                               b_N, solve   │
                                    │                       │
                                    ▼                       │
                             expansion/density.py ──────────┤
                               C0 via quadrature            │
                               p̂_N(x) = C0·exp(Σ c_j φ_j)  │
                                    │                       │
                    ┌───────────────┼──────────────┐        │
                    │               │              │        │
                    ▼               ▼              ▼        │
             cos/cos_method.py   distances/     utils/     │
               p_COS(x)          metrics.py    timing.py   │
               c_j benchmark      dA, d2log,               │
                                   d1, d2                  │
                                    │                       │
                                    ▼                       │
                              plots/figures.py              │
                                Figures 1–12, Table 2       │
                                    │                       │
                                    ▼                       │
                              output/*.pdf  ─────────────────┘
```

---

## 5. How to Retrieve Individual Plots

### Run everything (all 12 figures + Table 2):
```bash
python main.py --models all
```

### Run only one model:
```bash
python main.py --models VG      # Figures 1, 5, 9  +  partial Table 2
python main.py --models NIG     # Figures 2, 6, 10 +  partial Table 2
python main.py --models Heston  # Figures 3, 7, 8, 11, 12 + partial Table 2
```

### Output files:
```
output/
├── figure_01_coeff_vg.pdf              # Fig 1:  VG coefficient convergence
├── figure_02_coeff_nig.pdf             # Fig 2:  NIG coefficient convergence
├── figure_03_coeff_heston.pdf          # Fig 3:  Heston coefficient convergence
├── figure_04_clr_three_pdfs.pdf        # Fig 4:  CLR of all 3 PDFs
├── figure_05_distances_vg.pdf          # Fig 5:  VG density distances (L=4)
├── figure_06_distances_nig.pdf         # Fig 6:  NIG density distances (L=4)
├── figure_07_distances_heston (restricted domain).pdf  # Fig 7: Heston restricted
├── figure_08_distances_heston (l=4).pdf               # Fig 8: Heston L=4, logistic
├── figure_09_density_vg.pdf            # Fig 9:  VG density comparison N=6,16
├── figure_10_density_nig.pdf           # Fig 10: NIG density comparison N=6,16
├── figure_11_density_heston (restricted).pdf  # Fig 11: Heston restricted domain
├── figure_12_density_heston.pdf        # Fig 12: Heston L=4 domain
└── table_2_cpu_times.txt               # Table 2: CPU timing
```

### Regenerate individual figures programmatically:
```python
from main import run_model
import config as CFG
import models.variance_gamma as vg_mod

timing = {}
run_model("VG", vg_mod.characteristic_function, vg_mod.raw_moments,
          CFG.VG_PARAMS, {"coeff": 1, "dist_full": 5, "dens_full": 9},
          timing_results=timing)
# Saves: figure_01_coeff_vg.pdf, figure_05_distances_vg.pdf, figure_09_density_vg.pdf
```

---

## 6. Formula Verification (audit results)

The following values were verified numerically against the paper's formulas:

| Check | Expected | Computed | Status |
|-------|----------|---------|--------|
| Δ_{0,0,0} | 1 | 1.000000 | ✓ |
| Δ_{1,1,0} | 1 | 1.000000 | ✓ |
| Δ_{2,2,0} | 2 | 2.000000 | ✓ |
| Δ_{1,1,2} | 2 | 2.000000 | ✓ |
| A_N[i,n] = √n·Ã_N[i,n] (Hermite, all N≤20) | — | — | ✓ |
| b_N[0]=b_N[1]=b_N[2] = 0 | 0 | 0 | ✓ |
| mʰ₀ = 1 | 1 | 1.000000 | ✓ |
| mʰ₁ = 0 (standardized) | 0 | ~0 | ✓ |
| mʰ₂ = 0 (unit variance) | 0 | ~0 | ✓ |
| VG skewness = 0 | 0 | 0.000000 | ✓ |
| VG excess kurtosis = 2 | 2 | 2.000000 | ✓ |
| Heston skewness = −1.2 | −1.2 | −1.2000 | ✓ |
| Heston excess kurtosis = 2.5 | 2.5 | 2.5000 | ✓ |
| ∫ p_COS(x) dx | 1 | 0.99999999 | ✓ |
| ∫ p̂_N(x) dx (N=8, VG) | 1 | 1.00000039 | ✓ |
| Hermite orthonormality max error | 0 | 2.5e-12 | ✓ |
| Logistic Stieltjes Gram-matrix diag (GL grid) | 1 | 1.0±4e-16 | ✓ |

---

## 7. Known Differences from the Paper

1. **CPU times (Table 2)**: Our timings (COS≈0.61s, Hermite≈0.006s, Logistic≈0.006s)
   differ from the paper's (COS≈0.15–0.66s, Hermite≈0.015–0.020s, Logistic≈0.049–0.093s)
   due to different hardware (Intel i3-9100F @ 3.60 GHz in the paper vs Apple Silicon here)
   and different Python/NumPy versions.

2. **NIG moments**: With the Appendix A parameters (mu=0, theta=0.05, sigma=0.2, kappa=0.3),
   we obtain skewness≈0.223 and excess kurtosis≈0.966, vs Table 1 values of 0.2 and 1.
   This slight discrepancy is expected — the paper rounds to 1 decimal place.

3. **Convergence at high N for Logistic**: The linear system becomes ill-conditioned
   for the Logistic basis at N≥17 (cond(A) > 10¹⁴). We fall back to least-squares
   in these cases. The paper does not address this explicitly.

4. **VG Parameters**: The exact numerical values of sigma=0.2, nu=2/3, theta=0, mu=0
   are inferred from the constraint "skewness=0, excess kurtosis=2" (since
   excess kurtosis = 3ν → ν=2/3; theta=0 for skewness=0). The paper cites
   Heston & Rossi (2016) for these parameters without listing them explicitly.

---

## 8. References

- Gambaro, A.M. (2024). Exponential expansions for approximation of probability distributions. *Decisions in Economics and Finance*. DOI: 10.1007/s10203-024-00460-2
- Fang, F., Oosterlee, C.W. (2009). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM J. Sci. Comput.* 31(2), 826–848.
- Muscolino, G., Ricciardi, G. (1999). Probability density function of MDOF structural systems under non-normal delta-correlated inputs. *Comput. Methods Appl. Mech. Eng.* 168(1), 121–133.
- Rompolis, L.S., Tzavalis, E. (2008). Recovering risk neutral densities from option prices: a new approach. *J. Financ. Quant. Anal.* 43(4), 1037–1053.
- Heston, S.L., Rossi, A.G. (2016). A spanning series approach to options. *Rev. Asset Pricing Stud.* 7(1), 2–42.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.
