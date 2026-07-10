# Master Thesis — Replication of Gambaro (2024)

**Thesis**: Francesco Farolfi — Quantitative Finance, Master's Degree  
**Paper replicated**: Anna Maria Gambaro, *"Exponential expansions for approximation of probability distributions"*, Decisions in Economics and Finance (2024). DOI: [10.1007/s10203-024-00460-2](https://doi.org/10.1007/s10203-024-00460-2)

---

## What This Project Does

This repository replicates **all numerical experiments from Section 5** of Gambaro (2024):
**Figures 1–12** and **Table 2**, for three financial log-return models:

| Model | Skewness | Excess Kurtosis | Reference |
|---|---|---|---|
| Variance Gamma (VG) | 0 | 2 | Heston & Rossi (2016) |
| Normal Inverse Gaussian (NIG) | ≈0.22 | ≈0.97 | Gambaro (2024) Appendix A |
| Heston (log-return) | −1.2 | 2.5 | Rompolis & Tzavalis (2008) |

---

## Core Method: Exponential Expansion in Bayesian Hilbert Space

The paper approximates a probability density `p(x)` by an **exponential family** built on an orthonormal basis:

```
p̂_N(x) = Ĉ₀ · exp( Σ_{j=1}^N ĉⱼ · φⱼ(x*) )
```

where `x* = (x − m₁)/σ` is the standardized variable, `Ĉ₀` is a normalization constant, and `{φⱼ}` is one of two bases tested:

- **Hermite basis**: `φⱼ = hⱼ` (normalized probabilist's Hermite polynomials), orthonormal w.r.t. the Gaussian weight `ω(x*)`
- **Logistic basis**: `{Lⱼ}` (polynomials orthonormal w.r.t. the logistic weight `ν_L(x*) = sech²(x*/2)/4`)

The key mathematical framework is the **Bayesian Hilbert Space** (Bayes Space) `B²(I, ν)`, where densities live and the geometry is defined by the **centered log-ratio (CLR) transform**:

```
clr(p)(x) = log p(x) − ∫_I log p(x) ν(x) dx
```

The Fourier coefficients of `clr(p)` in the basis `{φⱼ}` are the target `cⱼ`, and the expansion approximates them by solving a **linear system** derived via integration by parts.

---

## How the Coefficients Are Estimated

The coefficients `ĉⱼ` are obtained by solving the `N×N` linear system:

```
A_N · ĉ_N = b_N
```

**System components (eq. 15 of the paper):**

**Ã_N** (inner matrix):
```
Ã_N[i,j] = Σ_{k=0}^{i+j−2} (1/k!) · Δ_{i−1,j−1,k} · mʰₖ
```

where `Δ_{p,q,r} = p!q!r! / ((b−p)!(b−q)!(b−r)!)` with `b=(p+q+r)/2` is the **triple Hermite integral coefficient** (classical result: `∫ Heₚ Heq Her ω dx`).

**A_N** (full matrix via change-of-basis `Q_n`):
```
A_N[i,n] = Σ_{j=0}^{n} √j · q_{n,j} · Ã_N[i,j]
```

For the Hermite basis `Q_n = I`, this simplifies to `A[i,n] = √n · Ã[i,n]`.

**b_N** (right-hand side — key formula):
```
b_N[i] = −√(i−1) · mʰ_{i−2}    for i = 1,...,N
```

The index `i−2` (not `i−1`) comes from the derivative property `h'_{i−1}(x*) = √(i−1) · h_{i−2}(x*)`. This was **confirmed by three independent checks**:
1. Gaussian test: gives ĉ₂ = −1/√2 exactly, all others 0 ✓
2. Hand derivation of the integration-by-parts formula ✓
3. Appendix B of the paper (2D analogue: `b_{i,j} = −√(i−1) · mʰ_{i−2,j−1}`, reduces to 1D at j=1) ✓

The **Hermite moments** `mʰₖ = E_p[hₖ(X*)]` are computed analytically from raw moments via the polynomial representation of `Heₖ`.

---

## Pipeline Summary

```
Model params (config.py)
        ↓
Raw moments E[Xᵏ]  (models/)
        ↓
Standardize → Hermite moments mʰₖ  (moments/)
        ↓
Basis construction: h_j (Hermite) or L_j (Logistic via Stieltjes)  (basis/)
        ↓
Change-of-basis Q_n via Gauss-Hermite inner products  (matrices/)
        ↓
Build Ã_N, A_N, b_N → solve A ĉ = b  (matrices/linear_system.py)
        ↓
Normalization constant Ĉ₀ via quadrature  (expansion/)
        ↓
COS benchmark density p_COS(x) and exact coefficients cⱼ  (cos/)
        ↓
Distance metrics: d₂(ĉN,c), Aitchison, log-L2, L1, L2  (distances/)
        ↓
Figures 1–12, Table 2  (plots/)
```

---

## Quick Start

```bash
cd project/
pip install -r requirements.txt

python main.py --models all           # full run (~35 s), all 12 figures + Table 2
python main.py --models VG            # VG only: Figures 1, 5, 9
python main.py --models NIG           # NIG only: Figures 2, 6, 10
python main.py --models Heston        # Heston: Figures 3, 7, 8, 11, 12
python main.py --models all --calibrate   # re-calibrate Heston parameters
```

Output PDFs and PNGs land in `project/output/`.

---

## Figure Index

| Figure | Content | Models |
|---|---|---|
| 1–3 | Coefficient convergence `d₂(ĉN, c)` vs N (Hermite & Logistic) | VG, NIG, Heston |
| 4 | CLR of three PDFs on standardized domain | All three |
| 5–6 | 4 density distances vs N (Aitchison, log-L2, L1, L2) | VG, NIG |
| 7 | Density distances on Heston restricted domain | Heston |
| 8 | Density distances on Heston L=4 domain, Logistic only | Heston |
| 9–10 | PDF & log-PDF comparison at N=6 and N=16 | VG, NIG |
| 11–12 | PDF & log-PDF comparison, restricted and full domain | Heston |
| Table 2 | CPU times for COS, Hermite N=16, Logistic N=16 | All three |

---

## Repository Structure

```
project/
├── config.py                  # Model parameters, N_MAX=20, L=4, N_COS=4096
├── main.py                    # Full pipeline orchestrator
│
├── models/
│   ├── variance_gamma.py      # VG CF, cumulants, raw moments
│   ├── nig.py                 # NIG CF, CGF derivatives, raw moments
│   └── heston.py              # Heston Gatheral CF, mpmath cumulants, calibration
│
├── basis/
│   ├── hermite.py             # Normalized He polynomials via 3-term recurrence
│   └── logistic.py            # Logistic polynomials via Stieltjes/Lanczos algorithm
│
├── moments/
│   └── hermite_moments.py     # mʰₖ = E[hₖ(X*)] from raw moments (Rompolis & Tzavalis 2008)
│
├── matrices/
│   ├── basis_matrices.py      # Entry point: build H, logistic recurrence + Q
│   ├── change_of_basis.py     # Q_n via 200-pt Gauss-Hermite inner products
│   └── linear_system.py       # Δ_{p,q,r}, Ã_N, A_N, b_N, solve + residual check
│
├── expansion/
│   └── density.py             # Ĉ₀ via log-sum-exp quadrature, evaluate p̂_N
│
├── cos/
│   └── cos_method.py          # COS density (Fang & Oosterlee 2009), benchmark cⱼ
│
├── distances/
│   └── metrics.py             # d₂(ĉN,c) with truncation, Aitchison, log-L2, L1, L2
│
├── plots/
│   └── figures.py             # All 12 figure functions + print_table2
│
├── utils/
│   ├── quadrature.py          # Domain via eq. (22), clr_domain for Heston, grid
│   └── timing.py              # perf_counter wrapper for Table 2
│
├── output/                    # Generated figures (PDF + PNG) and table
└── PIPELINE_DOCUMENTATION.md  # Detailed technical documentation of every module
```

---

## Key Implementation Details

### Logistic Basis: Stieltjes Algorithm
The logistic polynomials `Lₖ` are built via the **Stieltjes/Lanczos algorithm** on a 3000-point Gauss-Legendre grid adapted to the logistic weight. This avoids the numerically unstable Gram-Schmidt on the Hankel moment matrix. The three-term recurrence `(α, β)` coefficients are then used for stable evaluation at any point.

### Change-of-Basis Q_n: Inner Products
`Q_n[k,j] = ⟨Lₖ, hⱼ⟩_ω` is computed via 200-point Gauss-Hermite quadrature, completely bypassing the monomial coefficient representation which suffers catastrophic cancellation for degree ≥ 10.

The Logistic–Hermite overlap decays rapidly: `Q[1,1] ≈ 0.551`, `Q[2,2] ≈ 0.240` (analytically confirmed: `Q[2,2] = √90/(4π²) ≈ 0.240`). This makes the Logistic A matrix poorly conditioned at higher N, explaining the larger coefficient distances in Figures 1–3 (right panels).

### Heston Cumulants: mpmath
Heston's characteristic function does not have a simple closed-form CGF. Cumulants are computed by numerically differentiating `log φ(−is)` using **mpmath** at 50-digit precision, avoiding catastrophic cancellation that occurs in standard float64 finite differences.

### Normalization Constant: Log-Sum-Exp
`Ĉ₀` is computed as `1 / (σ · ∫ exp(Σ ĉⱼ φⱼ(t)) dt)`. To avoid overflow, the exponent is shifted by its maximum before exponentiating (log-sum-exp trick).

### COS Method
The COS benchmark uses `N_COS = 2¹² = 4096` terms following Fang & Oosterlee (2009). The Heston CF uses the Gatheral form to avoid branch-cut issues for large `|u|`.

### Distance d₂(ĉN, c) — Full Formula (eq. 17)
```
d₂(ĉN, c) = sqrt( Σ_{j=1}^N (ĉⱼ − cⱼ)² + Σ_{j>N} cⱼ² )
```
Both the **estimation error** (first term) and the **truncation error** (second term, tail of the exact series) are included. The benchmark `cⱼ` are computed from the COS density via numerical integration of eq. (9).

---

## Mathematical Notes

### Why kurtosis enters b_N at index 6, not 4
For a standardized distribution, `mʰₖ = E_p[hₖ(X*)]` satisfies `mʰ₀=1`, `mʰ₁=0`, `mʰ₂=0` always (by definition of mean=0 and variance=1). The first non-trivially non-zero higher moment for a distribution with excess kurtosis `κ` is `mʰ₄ = κ/(2√6)`, which enters `b_N[6] = −√5 · mʰ₄`. Therefore **kurtosis drives ĉ₆ directly** and ĉ₄ only through off-diagonal coupling.

### Why Logistic distances are larger than Hermite
The Logistic basis has `Q[n,n] → 0` rapidly as `n` grows (the logistic polynomials project weakly onto the Gaussian Hermite basis). This makes the A matrix diagonal entries small, leading to larger estimation errors in ĉ_N for the Logistic case. This is a mathematical property of the Logistic–Hermite overlap, not an implementation issue.

### Appendix B (2D formula) cross-check
The paper's Appendix B gives the 2D extension: `b_{i,j} = −√(i−1) · mʰ_{i−2,j−1}`. Setting `j=1` (no second dimension, `mʰ_{k,0} = mʰₖ`) exactly recovers the 1D formula `b_i = −√(i−1) · mʰ_{i−2}`, confirming the index. The 2D A matrix also has the `√n` factor outside the sum, matching our 1D `A[i,n] = √n · Ã[i,n]` for the Hermite basis.

---

## References

- Gambaro, A.M. (2024). Exponential expansions for approximation of probability distributions. *Decisions in Economics and Finance*. DOI: 10.1007/s10203-024-00460-2
- Fang, F., Oosterlee, C.W. (2009). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM J. Sci. Comput.* 31(2), 826–848.
- Muscolino, G., Ricciardi, G. (1999). Probability density function of MDOF structural systems under non-normal delta-correlated inputs. *Comput. Methods Appl. Mech. Eng.* 168(1), 121–133.
- Rompolis, L.S., Tzavalis, E. (2008). Recovering risk neutral densities from option prices: a new approach. *J. Financ. Quant. Anal.* 43(4), 1037–1053.
- Heston, S.L., Rossi, A.G. (2016). A spanning series approach to options. *Rev. Asset Pricing Stud.* 7(1), 2–42.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.
