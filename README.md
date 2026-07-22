# Master Thesis — Replication of Gambaro (2024)

**Thesis**: Francesco Farolfi — Quantitative Finance, Master's Degree  
**Paper replicated**: Anna Maria Gambaro, *"Exponential expansions for approximation of probability distributions"*, Decisions in Economics and Finance (2024). DOI: [10.1007/s10203-024-00460-2](https://doi.org/10.1007/s10203-024-00460-2)

---

## 1. Obiettivo

Questo repository replica **tutti gli esperimenti numerici della Sezione 5** di Gambaro (2024): **Figure 1–12** e **Tabella 2**, per tre modelli di log-return finanziari:

| Modello | Skewness | Kurtosi eccesso | Riferimento parametri |
|---------|----------|-----------------|-----------------------|
| Variance Gamma (VG) | 0 | 2 | Heston & Rossi (2016) |
| Normal Inverse Gaussian (NIG) | ≈0.22 | ≈0.97 | Gambaro (2024), Appendice A |
| Heston (log-return) | −1.2 | 2.5 | Rompolis & Tzavalis (2008) |

---

## 2. Metodo: Espansione Esponenziale in Bayesian Hilbert Space

Il paper approssima una densità di probabilità `p(x)` con una famiglia esponenziale costruita su una base ortonormale [Gambaro 2024, eq. 16]:

```
p̂_N(x) = Ĉ₀ · exp( Σ_{j=1}^N ĉⱼ · φⱼ(x*) )

Ĉ₀ = ( ∫_I exp( Σ_{j=1}^N ĉⱼ · φⱼ(x*) ) dx )⁻¹
```

dove `x* = (x − m₁)/σ` è la variabile standardizzata e `{φⱼ}` è una delle due basi testate:

- **Base Hermite**: `φⱼ = hⱼ` (polinomi di Hermite probabilistici normalizzati), ortonormali rispetto al peso gaussiano `ω(x*) = e^{−x*²/2}/√(2π)`
- **Base Logistica**: `{Lⱼ}` (polinomi ortonormali rispetto al peso logistico `ν_L(x*) = sech²(x*/2)/4`, con varianza π²/3) [Heston & Rossi 2016]

Il framework matematico è il **Bayesian Hilbert Space** B²(I, ν) [Gambaro 2024, §2], dove la geometria è definita dalla **CLR transform** (centered log-ratio):

```
clr(p)(x) = log p(x) − ∫_I log p(x) ν(x) dx     [eq. 2]
```

I coefficienti di Fourier di `clr(p)` nella base `{φⱼ}` sono i target `cⱼ` [eq. 9]; il sistema lineare li stima dai momenti della distribuzione.

---

## 3. Quick Start

```bash
cd project/
pip install -r requirements.txt

python main.py --models all           # run completo (~35 s), tutte le 12 figure + Tabella 2
python main.py --models VG            # solo VG: Figure 1, 5, 9
python main.py --models NIG           # solo NIG: Figure 2, 6, 10
python main.py --models Heston        # Heston: Figure 3, 7, 8, 11, 12
python main.py --models all --calibrate   # ri-calibra i parametri Heston
```

PDF e PNG vengono salvati in `project/output/`.

---

## 4. Indice Figure

| Figura | Contenuto | Modelli |
|--------|-----------|---------|
| 1–3 | Convergenza coefficienti `d₂(ĉN, c)` vs N (Hermite & Logistica) | VG, NIG, Heston |
| 4 | CLR delle tre PDF sul dominio standardizzato | Tutti e tre |
| 5–6 | 4 distanze densità vs N (Aitchison, log-L2, L1, L2) | VG, NIG |
| 7 | Distanze densità, dominio ristretto Heston | Heston |
| 8 | Distanze densità, dominio L=4, solo Logistica | Heston |
| 9–10 | PDF & log-PDF a confronto, N=6 e N=16 | VG, NIG |
| 11–12 | PDF & log-PDF, dominio ristretto e completo | Heston |
| Tabella 2 | Tempi CPU: COS, Hermite N=16, Logistica N=16 | Tutti e tre |

---

## 5. Struttura del Repository

```
project/
├── config.py                  # Parametri modelli, N_MAX=20, L=4, N_COS=4096
├── main.py                    # Orchestratore della pipeline completa
│
├── models/
│   ├── variance_gamma.py      # VG: CF, cumulanti (CGF), momenti
│   ├── nig.py                 # NIG: CF, derivate CGF, momenti
│   └── heston.py              # Heston: CF (forma Gatheral), cumulanti mpmath, calibrazione
│
├── basis/
│   ├── hermite.py             # Polinomi He normalizzati via ricorrenza a 3 termini
│   └── logistic.py            # Polinomi logistici via algoritmo di Stieltjes/Lanczos
│
├── moments/
│   └── hermite_moments.py     # mʰₖ = E[hₖ(X*)] da momenti grezzi [Rompolis & Tzavalis 2008]
│
├── matrices/
│   ├── basis_matrices.py      # Entry point: costruisce H, ricorrenza logistica + Q
│   ├── change_of_basis.py     # Q_n via inner products Gauss-Hermite a 200 punti
│   └── linear_system.py       # Δ_{p,q,r}, Ã_N, A_N, b_N, solve + verifica residuo
│
├── expansion/
│   └── density.py             # Ĉ₀ via log-sum-exp quadratura, valuta p̂_N
│
├── cos/
│   └── cos_method.py          # Densità COS [Fang & Oosterlee 2009], coefficienti cⱼ benchmark
│
├── distances/
│   └── metrics.py             # d₂(ĉN,c) con troncamento, Aitchison, log-L2, L1, L2
│
├── plots/
│   └── figures.py             # Tutte le 12 funzioni figura + print_table2
│
├── utils/
│   ├── quadrature.py          # Dominio via eq. (22), clr_domain per Heston, griglia
│   └── timing.py              # Wrapper perf_counter per Tabella 2
│
└── output/                    # Figure generate (PDF + PNG) e tabella
```

---

## 6. Background Matematico

### 6.1 Bayesian Hilbert Space [Gambaro 2024, §2]

Il paper lavora nello spazio B²(I, ν), lo spazio delle funzioni strettamente positive log-quadrato-integrabili con peso di riferimento ν. B² è isometricamente isomorfo a L²(I, ν) tramite la trasformazione CLR [eq. 2]. La distanza di Aitchison in B² coincide con la distanza L² fra le CLR:

```
d_A(p̂_N, p) = √[ ∫_I (clr(p̂_N)(x) − clr(p)(x))² ν(x) dx ]   [eq. 18]
```

### 6.2 Stima dei Coefficienti — Sistema Lineare [Gambaro 2024, eq. 15]

I coefficienti `ĉⱼ` si ottengono risolvendo il sistema N×N:

```
A_N · ĉ_N = b_N
```

**Matrice Ã_N** (momento di Hermite):
```
Ã_N[i,j] = Σ_{k=0}^{i+j−2}  Δ_{i−1,j−1,k} / √((i−1)!(j−1)!k!)  ·  mʰₖ
          = E_p[ h_{i−1}(X*) · h_{j−1}(X*) ]
```

**Coefficiente Δ_{p,q,r}** (integrale triplo di Hermite) [Erdelyi et al. 1953; Grad 1949]:
```
Δ_{p,q,r} = p! q! r! / ((b−p)!(b−q)!(b−r)!)   se p+q+r pari, b=(p+q+r)/2 ≥ max(p,q,r)
           = 0                                    altrimenti
```
Interpretazione: `∫ He_p He_q He_r ω dx`. La formula nel codice usa `Δ/√(p!q!r!)` con momenti normalizzati mʰ per ottenere correttamente E_p[h_i h_j].

**Matrice A_N** (con cambio di base Q_n, eq. 13–14):
```
A_N[i,n] = Σ_{j=1}^{n} √j · q_{n,j} · Ã_N[i,j]
```
dove `q_{n,j} = ⟨φ_n, h_j⟩_ω` è l'elemento della matrice di cambio base. Per la base Hermite: Q_n = I, quindi `A[i,n] = √n · Ã[i,n]`.

**Vettore b_N**:
```
b_N[i] = −√(i−1) · mʰ_{i−2}    per i = 1,...,N
```
L'indice è `i−2` (non `i−1`): viene dalla proprietà di derivazione `h'_{i−1}(x*) = √(i−1) · h_{i−2}(x*)` [Muscolino & Ricciardi 1999]. Verificato da: (1) test Gaussiano: dà ĉ₂ = −1/√2 esatto; (2) formula 2D Appendice B: `b_{i,j} = −√(i−1) · mʰ_{i−2,j−1}`, che per j=1 riduce alla 1D.

**Momenti di Hermite** `mʰₖ = E_p[hₖ(X*)]` [Rompolis & Tzavalis 2008, eqs. 2–3]:
```
mʰₖ = (1/√k!) · Σ_{j=0}^{k} [coeff. di xʲ in He_k] · μ*_j
```
dove `μ*_j = E[(X*)ʲ]` sono i momenti grezzi standardizzati.

---

## 7. Pipeline Cronologica (per modello)

### Step 1 — Parametri Modello (`config.py`)

| Modello | Parametri | Fonte |
|---------|-----------|-------|
| VG | sigma=0.2, nu=2/3, theta=0, mu=0 | Heston & Rossi (2016): skewness=0, kurtosi eccesso=3ν=2 |
| NIG | mu=0, theta=0.05, sigma=0.2, kappa=0.3, dt=1 | Gambaro (2024), Appendice A |
| Heston | kappa=2.0015, theta=0.04785, xi=0.40299, rho=−0.76635, v0=0.05879, T=1 | Calibrati su Rompolis & Tzavalis (2008): skewness=−1.2, kurtosi=2.5 |

**Nota (footnote 4 del paper):** i coefficienti ĉ_N **non dipendono** dal dominio troncato I — la troncatura influenza solo COS e le metriche di distanza.

---

### Step 2 — Momenti Grezzi (`models/*.py`)

Ogni modello espone `raw_moments(params, max_order)` → array `μ[k] = E[Xᵏ]`.

**VG** — CGF: `K(s) = μ·s − (1/ν)·log(1 − νθs − νσ²s²/2)`. Cumulanti calcolati esattamente via formula di Faà di Bruno; momenti grezzi dalla ricorrenza momento-cumulante: `μ_n = Σ_{k=1}^n C(n−1,k−1) · κ_k · μ_{n−k}`.

**NIG** — CGF: `K(s) = μ·s + (1/κ)·(1 − √(1 − 2θκs − σ²κs²))`. Derivate di √g calcolate via identità di Leibniz; stessa ricorrenza momento-cumulante.

**Heston** [Gatheral 2006] — CF (forma di Gatheral, evita problemi di branch cut):
```
φ(u) = exp(A(u,T) + B(u,T)·v₀)
d = √((κ−ρξiu)² + ξ²(iu+u²))
g = (κ−ρξiu−d) / (κ−ρξiu+d)
```
Cumulanti calcolati differenziando numericamente `log φ(−is)` con **mpmath** a 50 cifre di precisione [mpmath library], evitando cancellazione catastrofica nelle differenze finite standard.

---

### Step 3 — Troncatura del Dominio (`utils/quadrature.py`)

Dominio I = [a, b] seguendo eq. (22) con L=4 [Gambaro 2024]:
```
a = k₁ − 4·√(k₂ + √k₄)
b = k₁ + 4·√(k₂ + √k₄)
```
dove k₁, k₂, k₄ sono il 1°, 2° e 4° cumulante della distribuzione.

**Dominio ristretto Heston** (usato in Fig. 7 e 11): il dominio viene ristretto finché `log(p_COS(x)) > −10` ovunque su I (euristica del paper: `|clr(p)| < 10`, ovvero `p(x) > 5×10⁻⁵`).

---

### Step 4 — Densità Benchmark COS (`cos/cos_method.py`)

Seguendo Fang & Oosterlee (2009) con N_COS = 2¹² = 4096 termini:
```
p(x) ≈ (2/(b−a)) · Σ_{k=0}^{N−1} ' Re[φ(kπ/(b−a)) · e^{−ikπa/(b−a)}] · cos(kπ(x−a)/(b−a))
```
La densità COS è usata come densità "vera" di riferimento.

---

### Step 5 — Momenti di Hermite (`moments/hermite_moments.py`)

Seguendo Rompolis & Tzavalis (2008), eqs. (2)–(3):
1. Standardizzare i momenti grezzi: `μ*_j = E[(X*)ʲ]` tramite espansione binomiale.
2. Calcolare `mʰₖ = (1/√k!) · Σ_{j=0}^{k} [coeff. di xʲ in He_k] · μ*_j`.

**Verifiche automatiche:** mʰ₀ = 1, mʰ₁ = 0, mʰ₂ = 0 (conseguenze della standardizzazione).

---

### Step 6 — Costruzione della Base

#### Base Hermite (`basis/hermite.py`)

Polinomi di Hermite probabilistici normalizzati: `h_j(x*) = He_j(x*) / √(j!)`.

Ricorrenza (stabile numericamente):
```
He_0 = 1,  He_1 = x,  He_{n+1}(x) = x·He_n(x) − n·He_{n-1}(x)
→ forma normalizzata: h_{k+1}(x) = (x·h_k − √k·h_{k-1}) / √(k+1)
```

Proprietà di derivazione (chiave per la derivazione del sistema lineare) [Muscolino & Ricciardi 1999]:
```
h'_j(x*) = √j · h_{j-1}(x*)
```

#### Base Logistica (`basis/logistic.py`)

Peso logistico standard [Heston & Rossi 2016]:
```
ν_L(x) = sech²(x/2)/4    (media=0, varianza=π²/3)
```

**Algoritmo: Stieltjes/Lanczos** [Gautschi 1982] — evita la matrice di Hankel mal condizionata. Fornisce la ricorrenza a tre termini (α_k, β_k) su una griglia di 3000 punti Gauss-Legendre adattata a ν_L:
```
L_{k+1}(x) = [(x − αₖ)·Lₖ(x) − √βₖ·L_{k-1}(x)] / √β_{k+1}
```
Per simmetria di ν_L: αₖ = 0 per ogni k. Verificato: β₁ = π²/3 ≈ 3.2899 (varianza della distribuzione logistica).

---

### Step 7 — Matrice di Cambio Base Q_n (`matrices/change_of_basis.py`)

Per ogni polinomio φ_n della base si calcola [Gambaro 2024, eq. 13–14]:
```
q_{n,j} = ⟨φ_n, h_j⟩_ω = ∫ φ_n(x*) · h_j(x*) · ω(x*) dx*
```

**Base Hermite:** Q_n = I (identità).

**Base Logistica:** calcolato via quadratura Gauss-Hermite a 200 punti:
```
Q_logistic[k,j] ≈ Σᵢ w_GH[i] · L_k(x_GH[i]) · h_j(x_GH[i])
```
Questo bypassa completamente la rappresentazione monomiale di L_k, numericamente instabile per grado ≥ 10 (cancellazione catastrofica). Equivalente algebrico: `Q = B · H⁻¹` dove B e H sono le matrici dei coefficienti monomiali (entrambe lower-triangolari).

---

### Step 8 — Costruzione e Soluzione del Sistema Lineare (`matrices/linear_system.py`)

Per ogni N = 1, 2, ..., N_MAX = 20:

1. **Costruisce Ã_N** (N×N): cicli annidati su i,j,k con Δ_{p,q,r} precalcolati.
2. **Costruisce A_N** (N×N): combinazione pesata via Q_n e Ã_N.
3. **Costruisce b_N** (vettore N): dai momenti di Hermite.
4. **Risolve**: `scipy.linalg.solve` (decomposizione LU). Fallback a `numpy.linalg.lstsq` se cond(A_N) > 10¹⁴ (sistemi logistici mal condizionati ad alto N).

---

### Step 9 — Costante di Normalizzazione Ĉ₀ (`expansion/density.py`)

```
Ĉ₀ = 1 / ( σ · ∫_{I*} exp(Σ_{j=1}^N ĉⱼ · φⱼ(t)) dt )
```
**Metodo numerico:** regola dei trapezi su 20.000 punti uniformi in I*. Trick log-sum-exp per evitare overflow: sottrae max(f) prima di esponenziare, compensa poi.

---

### Step 10 — Coefficienti Fourier Benchmark da COS (`cos/cos_method.py`)

I coefficienti esatti cⱼ (eq. 9) sono stimati via:
```
cⱼ = ∫_I clr(p)(x) · φⱼ(x*) · ν(x*) dx
```
dove `clr(p)(x) = log(p_COS(x)) − E_ν[log(p_COS(X))]`. Integrazione con la regola dei trapezi sulla griglia della densità.

---

### Step 11 — Metriche di Distanza (`distances/metrics.py`)

Tutte le distanze sono calcolate via regola dei trapezi sul dominio troncato I.

| Eq. | Nome | Formula |
|-----|------|---------|
| (17) | Distanza coefficienti | d₂(ĉN,c) = √[ Σ_{j=1}^N (ĉⱼ−cⱼ)² + Σ_{j>N} cⱼ² ] |
| (18) | Distanza Aitchison | d_A = √[ ∫ (clr(p̂N)−clr(p))² ν dx ] |
| (19) | Distanza L²-log | d₂(log) = √[ ∫ (log p̂N−log p)² dx ] |
| (20) | Distanza L¹ | d₁ = ∫ \|p̂N−p\| dx |
| (21) | Distanza L² | d₂ = √[ ∫ (p̂N−p)² dx ] |

**Distanza d₂(ĉN, c) completa [eq. 17]:** include sia l'errore di stima (primo termine) sia l'errore di troncamento (secondo termine, coda della serie esatta). I coefficienti benchmark cⱼ sono calcolati dalla densità COS via integrazione numerica di eq. (9).

---

### Step 12 — Generazione Figure (`plots/figures.py`)

| Figura | Contenuto | Funzione | Parametri chiave |
|--------|-----------|----------|-----------------|
| 1 (VG) | d₂ coefficienti vs N, Hermite & Logistica | `fig_coeff_convergence` | N=1..20 |
| 2 (NIG) | Stesso per NIG | `fig_coeff_convergence` | — |
| 3 (Heston) | Stesso per Heston | `fig_coeff_convergence` | — |
| 4 | CLR 3 PDF, tolleranza ±10 | `fig4_clr` | asse x = x* |
| 5 (VG) | 4 distanze vs N | `fig_density_distances` | dominio L=4 |
| 6 (NIG) | Stesso per NIG | `fig_density_distances` | — |
| 7 (Heston) | 4 distanze vs N | `fig_density_distances` | dominio ristretto |
| 8 (Heston) | 4 distanze vs N, solo Logistica | `fig_density_distances` | L=4, logistic_only=True |
| 9 (VG) | PDF & log-PDF, N=6 & N=16 | `fig_density_comparison` | — |
| 10 (NIG) | Stesso per NIG | `fig_density_comparison` | — |
| 11 (Heston) | Stesso, dominio ristretto | `fig_density_comparison` | — |
| 12 (Heston) | Stesso, dominio L=4 | `fig_density_comparison` | senza curva Hermite |

Per rigenerare individualmente:
```python
from main import run_model
import config as CFG, models.variance_gamma as vg_mod
timing = {}
run_model("VG", vg_mod.characteristic_function, vg_mod.raw_moments,
          CFG.VG_PARAMS, {"coeff": 1, "dist_full": 5, "dens_full": 9}, timing_results=timing)
```

---

## 8. Diagramma di Interazione Moduli

```
config.py ──────────────────────────────────────────────────────────┐
                                                                    │
models/                                                             │
  *.characteristic_function(u) ──────────────────────── cos/        │
  *.raw_moments(params, K) ───────────────────────────── moments/   │
                                     │                              │
                                     ▼                              │
                             moments/hermite_moments.py             │
                                 mh[k] = E[h_k(X*)]                │
                                     │                              │
                   ┌─────────────────┼────────────────────┐        │
                   ▼                 ▼                    ▼        │
             basis/hermite.py  basis/logistic.py   utils/quadrature.py
             matrice H^n        Stieltjes α,β        dominio [a,b]
                   │                 │                    │
                   └────────┬────────┘                    │
                            ▼                             │
                    matrices/                             │
                      change_of_basis.py ── Q_n           │
                      linear_system.py  ── Ã_N, A_N, b_N  │
                                   │                      │
                                   ▼                      │
                            expansion/density.py ─────────┤
                              Ĉ₀ via quadratura           │
                              p̂_N(x) = Ĉ₀·exp(Σ cⱼ φⱼ) │
                                   │                      │
                   ┌───────────────┼─────────────┐        │
                   ▼               ▼             ▼        │
            cos/cos_method.py  distances/    utils/       │
              p_COS(x)          metrics.py  timing.py     │
              cⱼ benchmark       dA, d₂log, d₁, d₂       │
                                   │                      │
                                   ▼                      │
                             plots/figures.py             │
                               Figure 1–12, Tabella 2     │
                                   ▼                      │
                             output/*.pdf ─────────────────┘
```

---

## 9. Dettagli Implementativi Chiave

### Correzione formula Ã_N
La formula corretta usa `Δ/√(p!q!r!)` con momenti **normalizzati** mʰ (non `(1/k!)·Δ`). La versione errata gonfiava le entrate diagonali (es. Ã[3,3] = 2.408 invece del corretto 2.000 per VG) e produceva coefficienti ĉ più piccoli del vero. La correzione ha ridotto l'errore ‖ĉ − c_true‖ di **1.8× per VG**.

### Logistica: perché Stieltjes e non Gram-Schmidt
La matrice di Hankel dei momenti è estremamente mal condizionata per grado ≥ 10 (cancellazione catastrofica). L'algoritmo di Stieltjes [Gautschi 1982] opera direttamente sulla griglia di quadratura e rimane stabile per qualsiasi N.

### Cumulanti Heston: mpmath
La CF di Heston non ha una CGF in forma chiusa. I cumulanti vengono calcolati differenziando numericamente `log φ(−is)` con mpmath a 50 cifre di precisione, evitando la cancellazione catastrofica nelle differenze finite standard in float64.

### Costante Ĉ₀: log-sum-exp
Per evitare overflow, l'esponente viene traslato del suo massimo prima di esponenziare, compensando poi: `integral_x = σ · ∫exp(f(t)−fmax)dt · exp(fmax)`.

### Metodo COS
Il benchmark COS usa N_COS = 2¹² = 4096 termini [Fang & Oosterlee 2009]. La CF di Heston usa la forma di Gatheral per evitare problemi di branch cut per grandi `|u|`.

### Perché la kurtosi entra in b_N all'indice 6 (non 4)
Per una distribuzione standardizzata: mʰ₀=1, mʰ₁=0, mʰ₂=0 sempre (per definizione di media=0 e varianza=1). Il primo momento superiore non banalmente non nullo per kurtosi κ è `mʰ₄ = κ/(2√6)`, che entra in `b_N[6] = −√5 · mʰ₄`. Quindi la **kurtosi guida ĉ₆ direttamente** e ĉ₄ solo tramite accoppiamento fuori-diagonale.

### Perché le distanze Logistiche sono maggiori di Hermite
La base Logistica ha `Q[n,n] → 0` rapidamente all'aumentare di n (i polinomi logistici si proiettano debolmente sulla base Hermite). Questo rende le entrate diagonali della matrice A piccole, causando errori di stima maggiori per ĉ_N nel caso Logistico. È una proprietà matematica dell'overlap Logistica–Hermite, non un problema implementativo.

---

## 10. Verifiche Numeriche (audit)

| Check | Atteso | Calcolato | Stato |
|-------|--------|-----------|-------|
| Δ_{0,0,0} | 1 | 1.000000 | ✓ |
| Δ_{1,1,2} | 2 | 2.000000 | ✓ |
| A_N[i,n] = √n·Ã_N[i,n] (Hermite, tutti N≤20) | — | — | ✓ |
| Test Gaussiano: ĉ₂=−1/√2, tutti gli altri ĉⱼ=0 | esatto | esatto | ✓ |
| mʰ₀ = 1 | 1 | 1.000000 | ✓ |
| mʰ₁ = mʰ₂ = 0 (standardizzato) | 0 | ~0 | ✓ |
| VG skewness = 0, kurtosi eccesso = 2 | (0, 2) | (0.000, 2.000) | ✓ |
| Heston skewness = −1.2, kurtosi = 2.5 | (−1.2, 2.5) | (−1.200, 2.500) | ✓ |
| ∫ p_COS(x) dx | 1 | 0.99999999 | ✓ |
| ∫ p̂_N(x) dx (N=8, VG) | 1 | 1.00000039 | ✓ |
| Ortonormalità Hermite, errore max | 0 | 2.5×10⁻¹² | ✓ |
| Stieltjes logistico, diag. matrice Gram | 1 | 1.0±4×10⁻¹⁶ | ✓ |
| Cambio base: max\|Q_algebrico − Q_inner\| | 0 | 8.9×10⁻¹⁶ | ✓ |
| Ricostruzione φ₃(x) da Q (sanity check) | identità | errore 1.1×10⁻¹⁶ | ✓ |

---

## 11. Cross-Check Appendice B (formula 2D)

Gambaro (2024), Appendice B estende la stima a **PDF bivariate p(x₁,x₂)**. Il sistema 2D è `ΣΣ A_{i,j,n,m} ĉ_{n,m} = b_{i,j}` con:

```
b_{i,j} = −√(i−1) · mʰ_{i−2, j−1}
```

Impostando j=1 (caso 1D, `mʰ_{k,0} = mʰₖ`) si ottiene esattamente `b_i = −√(i−1) · mʰ_{i−2}`, **confermando l'indice i−2** usato nel codice. La matrice A 2D ha anch'essa il fattore √n fuori dalla somma, coerentemente con `A[i,n] = √n · Ã[i,n]` per la base Hermite.

---

## 12. Differenze Note rispetto al Paper

1. **Tempi CPU (Tabella 2):** i nostri tempi (COS≈0.61s, Hermite≈0.006s, Logistica≈0.006s) differiscono da quelli del paper (Intel i3-9100F @ 3.60 GHz) per differenze hardware e versioni Python/NumPy.

2. **Momenti NIG:** con i parametri dell'Appendice A otteniamo skewness≈0.223 e kurtosi≈0.966 vs i valori arrotondati 0.2 e 1 della Tabella 1 del paper. Discrepanza attesa: il paper arrotonda a 1 decimale.

3. **Convergenza Logistica ad alto N:** il sistema lineare diventa mal condizionato per la base Logistica a N≥17 (cond(A) > 10¹⁴). Si usa least-squares come fallback. Il paper non affronta esplicitamente questo caso.

4. **Parametri VG:** i valori sigma=0.2, nu=2/3, theta=0, mu=0 sono inferiti dal vincolo "skewness=0, kurtosi eccesso=2" (poiché kurtosi eccesso = 3ν → ν=2/3). Il paper cita Heston & Rossi (2016) senza listare i parametri esplicitamente.

---

## 13. Cosa È Esplicito nel Paper vs. Cosa Viene dalle Referenze

| Elemento | Fonte | Dove |
|----------|-------|------|
| Parametri NIG | Gambaro (2024) | Appendice A |
| Tutte le formule eqs. (7)–(22) | Gambaro (2024) | §2–§4 |
| Griglia COS 2¹² | Gambaro (2024) | §5 |
| Regola dominio eq. (22) con L=4 | Gambaro (2024) | §4 |
| Tolleranza CLR \|clr(p)\| < 10 | Gambaro (2024) | p. 13 |
| Parametri VG (sigma, nu, theta, mu) | Heston & Rossi (2016) | §2 o appendice numerica |
| Parametri Heston (kappa, theta, xi, rho, v0, T) | Rompolis & Tzavalis (2008) | §3 o tabella parametri |
| Costruzione base Logistica (peso ν_L) | Heston & Rossi (2016) | Appendice o §2 |
| Derivazione sistema lineare (integrazione per parti) | Muscolino & Ricciardi (1999) | Appendice |
| Formula momenti Hermite da momenti grezzi | Rompolis & Tzavalis (2008) | Eqs. (2)–(3) |
| Integrale triplo Hermite (coefficienti Δ) | Erdelyi et al. (1953); Grad (1949) | Risultato classico |
| Algoritmo Stieltjes per OPS | Gautschi (1982) | SIAM J. Sci. Stat. Comput. |
| Implementazione COS | Fang & Oosterlee (2009) | Algorithm 1 |
| CF Heston (forma Gatheral, branch cut) | Gatheral (2006) | Cap. 2 |

---

## 14. Riferimenti

- **Gambaro, A.M.** (2024). Exponential expansions for approximation of probability distributions. *Decisions in Economics and Finance*. DOI: 10.1007/s10203-024-00460-2
- **Fang, F., Oosterlee, C.W.** (2009). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM J. Sci. Comput.* 31(2), 826–848.
- **Muscolino, G., Ricciardi, G.** (1999). Probability density function of MDOF structural systems under non-normal delta-correlated inputs. *Comput. Methods Appl. Mech. Eng.* 168(1), 121–133.
- **Rompolis, L.S., Tzavalis, E.** (2008). Recovering risk neutral densities from option prices: a new approach. *J. Financ. Quant. Anal.* 43(4), 1037–1053.
- **Heston, S.L., Rossi, A.G.** (2016). A spanning series approach to options. *Rev. Asset Pricing Stud.* 7(1), 2–42.
- **Gatheral, J.** (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.
- **Gautschi, W.** (1982). On generating orthogonal polynomials. *SIAM J. Sci. Stat. Comput.* 3(3), 289–317.
- **Erdelyi, A. et al.** (1953). *Higher Transcendental Functions*, Vol. II. McGraw-Hill.
- **Grad, H.** (1949). Note on N-dimensional Hermite polynomials. *Commun. Pure Appl. Math.* 2(4), 325–330.
