
# Gambaro (2024) — Full Replication Roadmap

---

## Implementation Status

This roadmap has been **fully implemented** in `project/`. All 12 figures and Table 2 from
Section 5 of Gambaro (2024) are reproduced. The key corrections applied during implementation:

| Issue | Wrong (original plan) | Correct (implemented) |
|---|---|---|
| b vector formula | `b_N[i] = -sqrt(i-1) * m^h_{i-1}` | `b_N[i] = -sqrt(i-1) * m^h_{i-2}` |
| d₂ coefficient distance | `Σ(ĉⱼ-cⱼ)²` (no sqrt, no truncation) | `sqrt(Σ(ĉⱼ-cⱼ)² + Σ_{j>N}cⱼ²)` |
| NumPy compatibility | `np.trapezoid` (NumPy ≥ 2.0 only) | `np.trapz` (NumPy 1.26 compatible) |
| Q_n computation | Monomial coefficient matrix (unstable) | Gauss-Hermite inner products (stable) |

See `project/PIPELINE_DOCUMENTATION.md` for the full technical description of every module.

---

## STEP 1 — Complete Computational Pipeline

```
Raw moments of p(x)  [up to order 2N-2]
         ↓
Standardize: x* = (x - m1) / sigma,  sigma = sqrt(m2 - m1^2)
         ↓
Hermite moments  m^h_k = E[h_k(X*)]  from raw moments / cumulants
         ↓
Build H_n  (Hermite coefficient matrix, upper-triangular)
         ↓
Build B_n  (basis {phi_j} coefficient matrix, upper-triangular)
         ↓
Solve B_n = H_n Q_n  →  Q_n  (upper-triangular, efficient back-substitution)
         ↓
Build A~_N  (Hermite moment matrix, eq. (15) inner sum)
Build A_N   (full system matrix, outer sum using Q_n)
Build b_N   (vector of Hermite moments)
         ↓
Solve  A_N c^ = b_N  →  c^_N  (coefficients)
         ↓
Build exponential expansion:
  p^_N(x) = C^_0 * exp( sum_{j=1}^{N} c^_j phi_j(x*) )
         ↓
Compute C^_0 numerically via quadrature on truncated domain I
         ↓
Verify  integral p^_N = 1
         ↓
COS benchmark: evaluate true p(x) via CF inversion (Fang & Oosterlee 2009)
         ↓
Benchmark coefficients c_j via eq. (9) using COS-estimated clr(p)
         ↓
Compute distances d2(c^_N, c), d_A, d2-log, d1, d2  as N varies
         ↓
Generate Figures 1–12 and Table 2
```

**Key constraint from the paper:** coefficients c^_N do NOT depend on the truncated domain I (footnote 4, p. 13). Domain truncation only affects the COS benchmark and distance calculations.

---

## STEP 2 — Models, Parameters, Characteristic Functions, Moments

### 2.1 Variance Gamma (VG)

**Source of parameters:** Heston & Rossi (2016), *Review of Asset Pricing Studies* 7(1), 2–42. The paper states "we use the parameters defined in Heston and Rossi (2016), that gives skewness equal to zero and excess kurtosis of 2."

**Action required:** Read Heston & Rossi (2016) Section 2 or appendix to extract the exact numerical values of (sigma, nu, theta, mu) used. Do not guess — parameters must be taken verbatim.

**Standard VG characteristic function** (Carr-Madan parametrization):
```
phi_VG(u) = exp(i*u*mu) * ( 1 / (1 - i*u*theta*nu + u^2*sigma^2*nu/2) )^(1/nu)
```
where mu (drift), sigma (volatility), nu (variance rate), theta (asymmetry). Setting dt=1.

**Cumulants of VG** (needed for moments and domain rule, derivable analytically):
- kappa_1 = mu + theta
- kappa_2 = sigma^2 + theta^2 * nu
- kappa_3 = 2*theta^3*nu^2 + 3*sigma^2*theta*nu
- kappa_4 = 3*nu*(2*theta^4*nu^2 + 4*sigma^2*theta^2*nu + sigma^4)

For skewness=0 (theta=0): kappa_3=0, excess kurtosis = kappa_4/kappa_2^2 = 3*nu*sigma^4/sigma^4 = 3*nu = 2, hence nu=2/3. sigma and mu from Heston & Rossi (2016).

**Moments:** obtained analytically from cumulants via the moment-cumulant formula. No numerical differentiation needed.
- Raw moment mu_k = f(kappa_1, ..., kappa_k)
- Use the standard recursive moment-cumulant relation

**Orders needed:** up to 2N-2 = 30 (for N=16, the maximum tested).

### 2.2 Normal Inverse Gaussian (NIG)

**Source:** Appendix A of Gambaro (2024), fully explicit.

**Characteristic function:**
```
phi_NIG(u) = exp( i*u*mu*dt + (dt/kappa) * (1 - sqrt(1 + u^2*sigma^2*kappa - 2*i*u*theta*kappa)) )
```
Parameters: mu=0, theta=0.05, sigma=0.2, kappa=0.3, dt=1.

**Cumulants of NIG** (via CF derivatives or known formulas for NIG):
The k-th cumulant of NIG(mu, theta, sigma, kappa) at dt=1:
- kappa_1 = mu + theta
- kappa_2 = sigma^2 + theta^2*kappa
- kappa_3 = 3*theta*(sigma^2 + theta^2*kappa)*kappa / (sigma^2 + theta^2*kappa)^(1/2) ... 

**Recommended approach:** compute cumulants/moments by taking derivatives of log(phi(u)) numerically (with very high precision, e.g. mpmath) or by using the known closed-form cumulant generating function of NIG:

CGF of NIG: K(s) = mu*s + (1/kappa)*(1 - sqrt(1 - 2*theta*kappa*s - sigma^2*kappa*s^2))
- kappa_k = d^k K(s)/ds^k |_{s=0}

This is the recommended analytical route.

### 2.3 Heston Model

**Source of parameters:** Rompolis & Tzavalis (2008), *Journal of Financial and Quantitative Analysis* 43(4), 1037–1053. The paper states "we use parameters defined in Rompolis and Tzavalis (2008), that produce a negative skewness of -1.2 and excess kurtosis of 2.5."

**Action required:** Read Rompolis & Tzavalis (2008) to find the exact values of (kappa_H, theta_H, xi, rho, v0, T).

**Heston characteristic function** (standard Heston 1993 formula for log-returns X = log(S_T/S_0)):
```
phi_Heston(u) = exp( i*u*(r*T) + v0 * A(u,T) + (kappa_H*theta_H/xi^2) * B(u,T) )
```
where:
- d(u) = sqrt( (kappa_H - i*rho*xi*u)^2 + xi^2*(i*u + u^2) )
- A(u,T) = (kappa_H - i*rho*xi*u - d) / xi^2 * (1 - exp(-d*T)) / (1 - g*exp(-d*T))
- B(u,T) = (kappa_H - i*rho*xi*u - d)*T - 2*log( (1 - g*exp(-d*T)) / (1 - g) )
- g = (kappa_H - i*rho*xi*u - d) / (kappa_H - i*rho*xi*u + d)

The Feller condition: 2*kappa_H*theta_H >= xi^2 is desirable for numerical stability.

**Moments:** Heston does NOT have an easily-closed-form CGF. Recommended approach:
1. Compute derivatives of log(phi_Heston(u)) at u=0 numerically (complex-step differentiation or symbolic differentiation).
2. Use mpmath for arbitrary-precision numerical differentiation if needed.
3. Alternatively use the known semi-closed-form cumulants from Rompolis & Tzavalis (2008) which are polynomial functions of the parameters.

**Action required:** Read Rompolis & Tzavalis (2008) equations for the first four cumulants of Heston model log-returns, then extend to higher orders if needed via numerical CF differentiation.

---

## STEP 3 — Basis Construction

### 3.1 Hermite Basis (Probabilist's convention)

**Definition:** The probabilist's Hermite polynomials He_j(x) satisfy orthogonality with respect to the standard Gaussian weight omega(x) = (1/sqrt(2*pi))*exp(-x^2/2):
```
integral He_m(x) He_n(x) omega(x) dx = n! * delta_{m,n}
```

**Recurrence (from the paper, standard):**
```
He_0(x) = 1
He_1(x) = x
He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x)
```

**Normalized Hermite polynomials** (used in the paper):
```
h_j(x*) = (-1)^j / sqrt(j!) * (1/eta(x*)) * d^j eta(x*)/d(x*)^j
```
where eta is the standard normal PDF. Equivalently: h_j(x*) = He_j(x*) / sqrt(j!).

Orthonormality: integral h_m(x*) h_n(x*) omega(x*) dx* = delta_{m,n}.

**Implementation:** Evaluate He_n via the three-term recurrence. Never use the explicit formula for large n due to catastrophic cancellation. Maintain an array of evaluated polynomial values at each grid point.

**Derivative property used in the linear system:**
```
h'_j(x*) = sqrt(j) * h_{j-1}(x*)
```
This is the key identity used by Muscolino & Ricciardi (1999) to derive the linear system.

**Numerical stability concern:** For large n (say n > 20), the polynomial values can be large before normalization. Use the recurrence in double precision; only switch to mpmath if loss of significance is detected.

### 3.2 Logistic Basis

**Source:** Heston & Rossi (2016), *Review of Asset Pricing Studies* 7(1), 2–42. Gambaro explicitly cites this paper for the Logistic polynomial basis. This is the primary external reference to consult.

**What to extract from Heston & Rossi (2016):**
- The exact definition of the logistic weight function nu_L(x)
- The explicit recurrence relation for the logistic polynomials L_j(x)
- The normalization convention used
- Whether the standardized logistic is used (mean 0, variance pi^2/3)

**Likely definition:** the standardized logistic density is:
```
nu_L(x) = e^(-x) / (1 + e^(-x))^2   [location=0, scale=1]
```
with mean 0 and variance pi^2/3.

**Construction via Gram-Schmidt:** Apply Gram-Schmidt orthonormalization to the monomial basis {1, x, x^2, ...} with respect to the inner product:
```
<f, g>_L = integral f(x) g(x) nu_L(x) dx
```
The integrals int x^k nu_L(x) dx are the moments of the logistic distribution, computable analytically:
- All odd moments = 0 (symmetric distribution)
- Even moments: E[X^{2k}] = 2*(1 - 2^{1-2k}) * (2k)! * |B_{2k}| / (2k)! ... (Bernoulli numbers)

**Numerical implementation of Gram-Schmidt:**
1. Set L_0(x) = 1 / sqrt(<1,1>_L)
2. For each j >= 1: compute the "raw" p_j = x^j - sum_{k=0}^{j-1} <x^j, L_k>_L * L_k
3. Normalize: L_j = p_j / sqrt(<p_j, p_j>_L)

**Important:** use high-precision quadrature (Gauss-Legendre or Gauss-Hermite adapted to logistic weight, or mpmath's quad) to compute inner products. The logistic distribution has heavier tails than Gaussian — integrate over a sufficiently wide range (e.g., [-50, 50]).

**Three-term recurrence:** all orthogonal polynomials with respect to a symmetric measure (mean=0) satisfy a recurrence of the form:
```
L_{n+1}(x) = (x - alpha_n) * L_n(x) - beta_n * L_{n-1}(x)
```
For a symmetric distribution: alpha_n = 0. The beta_n coefficients are determined from the Cholesky/Gram-Schmidt process. **Action required:** extract these recurrence coefficients from Heston & Rossi (2016), or compute them from the moment matrix using the Lanczos algorithm.

**Derivative property:** For logistic polynomials, there is no derivative shortcut analogous to Hermite. This is why the paper extends the linear system via the change-of-basis Q_n.

---

## STEP 4 — Matrices B_n, H_n, Q_n

All three matrices are (N+1) x (N+1) and upper triangular.

### 4.1 Matrix H_n

**Definition:** H_n[i,j] = coefficient of x^{j-1} in the (i-1)-th normalized Hermite polynomial h_{i-1}(x*), for i,j = 1,...,N+1.

**Structure:** upper triangular because h_k is a polynomial of degree k. Diagonal entries are nonzero; entries below diagonal are zero.

**Algorithm:**
1. For k = 0,...,N: compute coefficients of He_k by expanding the three-term recurrence symbolically (coefficient-by-coefficient).
2. Divide each coefficient vector by sqrt(k!) to get h_k.
3. Assemble row k+1 of H_n as this coefficient vector (zero-padded to length N+1).

**Explicit small example:**
```
He_0 = 1                     → h_0 = [1, 0, 0, ...]
He_1 = x                     → h_1 = [0, 1, 0, ...]
He_2 = x^2 - 1               → h_2 = [-1/sqrt(2), 0, 1/sqrt(2), ...]
He_3 = x^3 - 3x              → h_3 = [0, -3/sqrt(6), 0, 1/sqrt(6), ...]
```

**Verification:** compute H_n * e_j where e_j = standard basis vector; the result should be the polynomial values at test points x = {-2,-1,0,1,2} matching the known Hermite polynomial values.

### 4.2 Matrix B_n

**Definition:** B_n[i,j] = coefficient of x^{j-1} in the (i-1)-th basis polynomial phi_{i-1}(x*), for i,j = 1,...,N+1.

- For Hermite basis: B_n = H_n (identical matrix)
- For Logistic basis: B_n is the coefficient matrix of L_0,...,L_N in the monomial basis

**Algorithm for Logistic:** same as H_n but starting from logistic polynomials. Extract coefficient vectors from the Gram-Schmidt construction (or recurrence).

### 4.3 Matrix Q_n

**Definition:** Q_n[i,j] = q_{i-1, j-1}, where phi_{i-1}(x*) = sum_{k=0}^{i-1} q_{i-1,k} h_k(x*).

**Algorithm:** from eq. (14), B_n = H_n * Q_n, so Q_n = H_n^{-1} * B_n.

Since H_n is upper triangular, H_n^{-1} * B_n is computed by back-substitution column by column. This is O(N^3) but N is at most ~20 in the paper so it is fast.

**Equivalently**, solving H_n * Q_n = B_n means: for each column j of Q_n, solve the upper triangular system H_n * q_j = b_j where b_j is column j of B_n. This is back-substitution.

**Verification:** compute H_n * Q_n and check it equals B_n element-by-element (up to machine precision).

---

## STEP 5 — Building A~_N, A_N, b_N

All indices run from 1 to N; polynomial indices run from 0.

### 5.1 Hermite Moments m^h_k

**Definition (paper, p. 9):**
```
m^h_k = integral h_k(x*) p(x) dx = E_p[h_k(X*)] = E_p[He_k(X*)] / sqrt(k!)
```
where X* = (X - m1)/sigma, m1 = E[X], sigma = sqrt(Var(X)).

**Connection to raw moments:** The (non-normalized) Hermite moment sqrt(k!) * m^h_k = E[He_k(X*)] can be written in terms of the central moments of X* (or equivalently the standardized cumulants). The paper references Rompolis & Tzavalis (2008) equations (2) and (3) for this.

**Action required:** Read Rompolis & Tzavalis (2008) eqs. (2)–(3) for the recursive formula. The essential formula is:
```
E[He_n(X*)] = sum over partitions of n of multinomial coefficients * product of cumulants of X*
```
In practice, the simplest route for moderate n is to:
1. Compute the raw moments mu_k = E[X^k] analytically from the model.
2. Compute central moments by binomial expansion.
3. Standardize: compute moments of X* = (X - m1)/sigma from central moments.
4. Express E[He_k(X*)] using the moment-Hermite recursion:
   ```
   E[He_0] = 1,  E[He_1] = 0 (by standardization),  E[He_2] = 0 (since Var(X*)=1)
   E[He_{n+1}(X*)] = E[X* He_n(X*)] - n * E[He_{n-1}(X*)]
                    = E[X* He_n(X*)] - n * m^h_{n-1}*sqrt((n-1)!)
   ```
   This still requires E[X* He_n(X*)], which involves E[X* He_n(X*)] = E[He_{n+1}(X*)] + n*E[He_{n-1}(X*)] by the Hermite recurrence, creating a dependency. Better to use the direct formula via cumulants.

**Recommended formula:** use the known relation between Hermite polynomials and cumulants. For a zero-mean unit-variance random variable X* with cumulants kappa_1*=0, kappa_2*=1, kappa_j* for j>=3:
```
E[He_n(X*)] = sum_{k=0}^{floor(n/2)} T(n,k) * kappa_n_combination
```
The exact formula is in Shiryaev (1996) or Lukacs (1970). Practically: compute moments of X* to order n, then evaluate He_n at each moment via the three-term recurrence applied to the moment sequence (polynomial of moments).

**Simplest stable implementation:** for each k, evaluate:
```
m^h_k = (1/sqrt(k!)) * sum_{j=0}^{k} coeff_j * mu*_j
```
where coeff_j are the coefficients of He_k (the j-th monomial coefficient of the k-th Hermite polynomial), and mu*_j = E[(X*)^j] = raw moment of standardized variable.

This is the most numerically direct approach: express He_k as a polynomial, then take expectation term by term.

### 5.2 Combinatorial coefficients Delta_{p,q,r}

**Definition from paper (eq. 15, p. 9):**
```
Delta_{p,q,r} = product_{a in {p,q,r}} (a+1)! / (b-a+1)!   if p+q+r is even, b = (p+q+r)/2 >= max(p,q,r)
Delta_{p,q,r} = 0                                            otherwise
```
where p, q, r >= 0 are integer indices.

**Interpretation:** This is a triple-product of binomial-like coefficients arising from the integration formula for products of three Hermite polynomials:
```
integral He_p(x) He_q(x) He_r(x) omega(x) dx = p! q! r! / (b-p)! (b-q)! (b-r)!  if p+q+r even
```
This is the classical "triple Hermite integral" formula — see Erdelyi et al. (1953) *Higher Transcendental Functions* or Grad (1949) "Note on N-dimensional Hermite polynomials".

**Algorithm:**
```python
def delta(p, q, r):
    s = p + q + r
    if s % 2 != 0:
        return 0
    b = s // 2
    if b < p or b < q or b < r:
        return 0
    return factorial(b) / (factorial(b-p) * factorial(b-q) * factorial(b-r))
    # equivalently: factorial(p)*factorial(q)*factorial(r) / ((b-p)!(b-q)!(b-r)!)
```
Note: the factorial representation can overflow for large indices. Use log-gamma / scipy.special.gammaln for numerical evaluation.

### 5.3 Matrix A~_N (Hermite moment matrix)

**From eq. (15) in the paper:**
```
A~_N[i,j] = sum_{k=0}^{i+j-2} (1/k!) * Delta_{i-1, j-1, k} * m^h_k
```
for i, j = 1,...,N (note: indices 1-based in paper, 0-based in code).

**Dimensions:** N x N.

**Algorithm:**
1. Pre-compute m^h_k for k = 0,...,2N-2.
2. Pre-compute Delta_{p,q,r} for p,q,r in {0,...,N-1} and k in {0,...,2N-2}.
3. For each (i,j) pair: loop k from 0 to i+j-2, accumulate sum.

**Numerical concern:** m^h_k for large k can be very large (growing like (k-1)!! for Gaussian-like distributions). The factorial 1/k! partially compensates. For N=16, k goes up to 30 — values are still manageable in float64 but borderline. Consider using Decimal or mpmath for N > 20.

### 5.4 Matrix A_N (full system matrix)

**From eq. (15) in the paper:**
```
A_N[i,n] = sum_{j=0}^{n} sqrt(j) * q_{n,j} * A~_N[i,j]
```
for i, n = 1,...,N. Note: the index j=0 term contributes sqrt(0)=0, so the sum effectively starts at j=1.

**Algorithm:**
1. Have A~_N (N x N).
2. Have Q_n (N+1 x N+1): extract the relevant submatrix.
3. For each (i,n): A_N[i,n] = sum_{j=1}^{n} sqrt(j) * Q_n[n,j] * A~_N[i,j]

**Note on index alignment:** Q_n[n,j] is q_{n,j} in the paper notation — the (n+1, j+1) element of Q_n in 1-based indexing.

**For Hermite basis:** Q_n = identity, so q_{n,j} = delta_{n,j}, so:
```
A_N[i,n] = sqrt(n) * A~_N[i,n]
```
This simplifies to a diagonal scaling.

### 5.5 Vector b_N

**From eq. (15) in the paper (confirmed via hand-derivation and Appendix B 2D formula):**
```
b_N[i] = -sqrt(i-1) * m^h_{i-2},   i = 1,...,N
```
where the index is i-2 (NOT i-1). Special case: i=1 gives sqrt(0)=0 so b_N[1]=0 regardless.

**Derivation:** Integration by parts uses h'_{i-1}(x*) = sqrt(i-1) * h_{i-2}(x*).
The resulting integral is E_p[h_{i-2}(X*)] = m^h_{i-2}.

**Verification via Gaussian test:** for X* ~ N(0,1), m^h_k = delta_{k,0}:
- b_N[1] = 0 (sqrt(0)=0)
- b_N[2] = -1 * m^h_0 = -1
- b_N[3] = -sqrt(2) * m^h_1 = 0
- b_N[4] = -sqrt(3) * m^h_2 = 0
Solving A c = b gives ĉ_2 = -1/sqrt(2), all others 0 — exactly the Gaussian CLR coefficients. ✓

**Cross-check via Appendix B (2D formula):** the 2D analogue is b_{i,j} = -sqrt(i-1) * m^h_{i-2, j-1},
which reduces to b_i = -sqrt(i-1) * m^h_{i-2} when j=1 (1D). ✓

**Signal location for non-Gaussian distributions:** for a distribution with excess kurtosis κ,
the first non-trivially non-zero higher moment is m^h_4 = κ/(2*sqrt(6)), which enters b_N[6]:
```
b_N[6] = -sqrt(5) * m^h_4
```
The kurtosis effect flows directly into ĉ_6, and into ĉ_4 only via off-diagonal coupling.

**Derivation source:** Muscolino & Ricciardi (1999), *Computer Methods in Applied Mechanics and Engineering* 168(1), 121–133. Read this paper for the full derivation of the linear system from integration by parts of the Hermite derivative property.

---

## STEP 6 — Solving the Linear System

**System:** A_N * c^_N = b_N, where A_N is N x N, generally dense and asymmetric.

**Numerical method:** LU decomposition with partial pivoting (scipy.linalg.solve or numpy.linalg.solve). For N <= 20 this is entirely stable.

**Conditioning analysis:**
- Compute cond(A_N) = ||A_N|| * ||A_N^{-1}||
- If cond(A_N) > 10^8 in float64 (epsilon ~10^{-16}), the solution will lose ~8 digits. For N=16, report the condition number.
- The paper does not discuss conditioning explicitly, but the Heston coefficients "struggle to converge", suggesting the system may be poorly conditioned for extreme skewness/kurtosis.

**Alternative for ill-conditioned systems:** use scipy.linalg.lstsq (minimum-norm least-squares) with a regularization threshold, or Tikhonov regularization. However, the paper does not mention regularization — stick to standard solve for reproducibility.

**Verification:**
1. Check residual ||A_N * c^_N - b_N|| / ||b_N|| < 10^{-10}
2. For Hermite basis (A_N diagonal): c^_N[i] = b_N[i] / A_N[i,i]. Verify this analytically.

**For Hermite basis only:** Since Q_n = I, A_N[i,n] = sqrt(n) * A~_N[i,n]. The matrix A_N is
NOT diagonal in general — off-diagonal terms couple different orders. For the Gaussian reference
distribution (all m^h_k = 0 for k>=1), A~_N IS diagonal (A~_N[i,j] = (i-1)! * delta_{ij}),
but for non-Gaussian distributions the off-diagonal mh contributions make A dense.

---

## STEP 7 — Exponential Expansion and Normalization

**Approximated PDF from eq. (16):**
```
p^_N(x) = C^_0 * exp( sum_{j=1}^{N} c^_j * phi_j(x*) ),   x* = (x - m1)/sigma
```

**Normalization constant:**
```
C^_0 = 1 / integral_I exp( sum_{j=1}^{N} c^_j * phi_j(x*) ) dx
```

**Note on variable change:** the integral is over x (not x*). Since x* = (x-m1)/sigma, we have dx = sigma * dx*, so:
```
C^_0 = 1 / ( sigma * integral_{I*} exp( sum_{j=1}^{N} c^_j * phi_j(t) ) dt )
```
where I* = (I - m1)/sigma is the standardized domain.

**Numerical quadrature for C^_0:**
- Use Gaussian quadrature (scipy.integrate.quad or numpy's Gauss-Legendre) on the domain I.
- The integrand can be extremely large or small depending on c^_j values; work in log-space:
  - Compute f(x) = sum_{j=1}^{N} c^_j * phi_j(x*) at each quadrature point
  - Use scipy.integrate.quad with the function exp(f(x)); let scipy handle the scaling
  - If exp(f) overflows: subtract max(f) before exponentiating (importance: this changes C^_0 but not the normalization check)
  - A grid of 10^4 uniformly spaced points is sufficient for moderate N; use scipy.integrate.quad for accuracy.
- Alternative: discretize I on a fine grid (same as COS grid), compute exp(f) at each point, integrate via trapezoidal rule.

**Positivity heuristic (from paper, p. 13):** verify |clr(p)(x)| < 10 for all x in I, which means p(x) > 5e-5 everywhere on I.

**Verification:** after computing C^_0, numerically integrate p^_N over I. The result should be 1.0 up to quadrature error (< 10^{-6}).

---

## STEP 8 — COS Method Benchmark

**Reference:** Fang & Oosterlee (2009), *SIAM Journal on Scientific Computing* 31(2), 826–848. Read this paper for full implementation details including truncation error bounds.

**Grid:** 2^12 = 4096 terms (stated in paper Section 5).

**Domain:** use eq. (22) with L=4:
```
I = [k1 - L*sqrt(k2 + sqrt(k4)),  k1 + L*sqrt(k2 + sqrt(k4))]
```
where k1,k2,k4 are the first, second, and fourth cumulants of the distribution.

**COS density formula:**
```
p(x) ≈ (2/(b-a)) * sum_{k=0}^{N_cos-1} ' Re[ phi(k*pi/(b-a)) * exp(-i*k*pi*a/(b-a)) ] * cos(k*pi*(x-a)/(b-a))
```
where ' means the k=0 term has weight 1/2, phi is the characteristic function, N_cos = 2^12, [a,b] = I.

**Key steps:**
1. Evaluate phi(u_k) for u_k = k*pi/(b-a), k=0,...,N_cos-1, using the CF of the model.
2. Multiply by exp(-i*u_k*a): this is the standard COS coefficient formula.
3. Multiply by 2/(b-a).
4. Evaluate at target x points using the cosine series.

**Evaluation grid for density plots:** use a fine grid of x values within I (e.g., 10^3 to 10^4 uniform points). The COS formula is evaluated via vectorized operations over this grid.

**Numerical concern for NIG and Heston:** the CF can be complex-valued with branch cut issues (square root). For NIG: sqrt(1 + u^2*sigma^2*kappa - 2*i*u*theta*kappa) — use principal branch (numpy.sqrt of complex numbers). For Heston: same for sqrt in d(u). In Heston, the Gatheral form of the CF (equivalent to standard Heston but avoids branch cuts) may be needed for large u. Read Fang & Oosterlee (2009) and Lord & Kahl (2010) for Heston CF stability.

**Benchmark coefficients c_j:** after computing the COS density, estimate the exact Fourier coefficients c_j via numerical integration of eq. (9):
```
c_j = integral clr(p)(x) * phi_j(x*) * nu(x*) dx
```
where clr(p)(x) = log(p_COS(x)) - integral log(p_COS(x)) * nu(x*) dx.

The inner integral (mean of log(p) under nu) is computed numerically. Then c_j is a one-dimensional integral over I, computed via trapezoidal rule or Gauss quadrature on the COS grid.

---

## STEP 9 — Distance Metrics

All distances are computed numerically via trapezoidal integration over the domain I.

**Grid for integration:** use the same fine grid as for the COS evaluation (e.g., 10^4 points). This gives integrals accurate to ~10^{-6}.

### 9.1 Coefficient distance (eq. 17)
```
d2(c^_N, c) = sqrt( sum_{j=1}^{N} (c^_j - c_j)^2  +  sum_{j>N} c_j^2 )
```
The first term is the estimation error (how well ĉ_N approximates c_1,...,c_N).
The second term is the truncation error (the tail of the exact series c_{N+1}, c_{N+2}, ...).
Both terms are needed for eq. (17). The full benchmark series c_j is estimated from COS.
Computed for N = 1, 2, ..., N_max (e.g., 20).

### 9.2 Aitchison distance (eq. 18)
```
d_A(p^_N, p) = sqrt( integral_I (clr(p^_N)(x) - clr(p)(x))^2 * nu(x*) dx )
```
where nu(x*) is the reference weight (Gaussian for Hermite, Logistic for Logistic basis).

**Important:** the reference measure nu depends on which basis is used. For Hermite: nu(x) = omega(x*)/sigma (Gaussian centered at m1). For Logistic: nu(x) = nu_L(x*)/sigma.

**Computing clr:** 
- clr(p^_N)(x) = log(p^_N(x)) - integral_I log(p^_N(x)) * nu(x*) * sigma dx (= log(C^_0) + sum c_j phi_j(x*) minus its nu-expectation)
- clr(p)(x) = log(p_COS(x)) - integral_I log(p_COS(x)) * nu(x*) * sigma dx

**Numerical pitfall:** log(p(x)) diverges as p(x)→0 at the boundaries. Ensure domain I is truncated so p(x) > 5e-5 everywhere (the |clr| < 10 rule from the paper).

### 9.3 L2-log distance (eq. 19)
```
d2_log(p^_N, p) = sqrt( integral_I (log(p^_N(x)) - log(p(x)))^2 dx )
```
Note: this is NOT weighted by nu (unlike Aitchison). This allows comparison across Hermite and Logistic bases on equal footing.

### 9.4 L1 distance (eq. 20)
```
d1(p^_N, p) = integral_I |p^_N(x) - p(x)| dx
```

### 9.5 L2 distance (eq. 21)
```
d2(p^_N, p) = sqrt( integral_I (p^_N(x) - p(x))^2 dx )
```

**All distances computed for N = 1, 2, ..., N_max, keeping the domain I fixed.**

---

## STEP 10 — Figures Specifications

### Figures 1, 2, 3 — Coefficient convergence

- **What:** d2(c^_N, c) vs N, for each model
- **x-axis:** N = 1, 2, ..., N_max (paper uses up to ~16-20)
- **y-axis:** d2(c^_N, c) in log scale (log10)
- **Left panel (a):** Hermite basis
- **Right panel (b):** Logistic basis
- **Each figure:** one model (Fig 1=VG, Fig 2=NIG, Fig 3=Heston)
- **Domain for c_j benchmark:** L=4 (paper uses same domain as density plots for VG/NIG; for Heston, use both domains)
- **Input:** c^_N from solver, c_j from numerical integration of eq. (9) using COS

### Figure 4 — clr of three PDFs

- **What:** clr(p)(x) plotted for VG, NIG, Heston on the same axes
- **Domain:** I from eq. (22) with L=4 (standardized domain I*)
- **x-axis:** standardized x* in I*
- **y-axis:** clr(p)(x*) — no units
- **Reference weight nu:** for comparability, the paper uses a single nu (likely the Gaussian weight)
- **Horizontal lines:** at y = +10 and y = -10 (tolerance thresholds)
- **Input:** p from COS, compute clr numerically

### Figures 5, 6, 7, 8 — Density convergence distances

- **Each figure:** 4 subplots arranged 2x2
  - (a) top-left: Aitchison distance d_A vs N
  - (b) top-right: log-L2 distance d2_log vs N
  - (c) bottom-left: L1 distance d1 vs N
  - (d) bottom-right: L2 distance d2 vs N
- **x-axis:** N = 1, 2, ..., N_max
- **y-axis:** distance value (likely log scale)
- **Each subfigure:** two curves — Hermite (solid?) and Logistic (dashed?)
- **Figure 5:** VG, domain L=4
- **Figure 6:** NIG, domain L=4
- **Figure 7:** Heston, restricted domain (clr > -10)
- **Figure 8:** Heston, domain L=4, Logistic only

### Figures 9, 10, 11, 12 — Density comparison at fixed N

- **Each figure:** 4 subplots
  - (a) top-left: PDF p(x) vs x, COS vs Hermite N=6
  - (b) top-right: PDF p(x) vs x, COS vs Logistic N=6
  - (c) bottom-left: PDF p(x) vs x, COS vs Hermite N=16
  - (d) bottom-right: PDF p(x) vs x, COS vs Logistic N=16
  OR alternatively: (a) PDF Hermite, (b) log-PDF Hermite, (c) PDF Logistic, (d) log-PDF Logistic
- **Caption says "true PDF (or log-PDF)"** suggesting both PDF and log-PDF appear. Likely the subplots alternate PDF and log-PDF for each basis.
- **Read the figures carefully from the PDF:** the exact subplot arrangement is visible in the PDF images.
- **Figure 9:** VG, domain L=4
- **Figure 10:** NIG, domain L=4
- **Figure 11:** Heston, restricted domain
- **Figure 12:** Heston, domain L=4

### Table 2 — CPU times

Reproduce by timing the three methods (COS, Hermite N=16, Logistic N=16) using Python's `time.perf_counter()`. CPU times will differ from the paper's Intel i3-9100F; report both for transparency.

---

## STEP 11 — Verification Checklist

Verify each block independently before integrating.

### 11.1 Basis orthonormality
```
integral phi_i(x*) phi_j(x*) nu(x*) dx* ≈ delta_{i,j}
```
Compute this for i,j = 0,...,N_max. Maximum off-diagonal element should be < 10^{-10}.

### 11.2 Matrix factorization
After solving B_n = H_n * Q_n:
```
||B_n - H_n @ Q_n||_F < 10^{-12}
```

### 11.3 Moment consistency
- For each model, verify that the first 4 raw moments match the known analytical values (mean, variance, skewness, kurtosis from Table 1).
- Verify m^h_0 = 1, m^h_1 = 0, m^h_2 = 0 (consequences of standardization).

### 11.4 Linear system residual
```
||A_N @ c^_N - b_N|| / ||b_N|| < 10^{-10}
```

### 11.5 Normalization of p^_N
```
|integral_I p^_N(x) dx - 1| < 10^{-6}
```

### 11.6 Positivity
```
p^_N(x) > 0 for all x in I
```
Always true by construction (exponential form). Verify numerically.

### 11.7 COS accuracy
Verify COS density integrates to 1 and matches known moments:
```
|integral_I p_COS(x) dx - 1| < 10^{-6}
|integral_I x * p_COS(x) dx - m1| < 10^{-5}
|integral_I x^2 * p_COS(x) dx - m2| < 10^{-5}
```

### 11.8 Coefficient convergence direction
For small N: c^_N should roughly agree with c_j for j <= N. For large N: the distance d2(c^_N, c) should be non-increasing (not guaranteed for all models but expected for VG and NIG).

---

## STEP 12 — Modular Software Architecture

```
project/
├── requirements.txt          # numpy, scipy, matplotlib, mpmath
├── config.py                 # all model parameters, domain settings, N_max
├── main.py                   # orchestrates full pipeline, calls submodules
│
├── models/
│   ├── variance_gamma.py     # CF phi_VG(u), cumulants, raw moments up to order K
│   ├── nig.py                # CF phi_NIG(u), cumulants from CGF, raw moments
│   └── heston.py             # CF phi_Heston(u), numerical cumulants via CF derivatives
│
├── basis/
│   ├── hermite.py            # He_n recurrence, h_n normalization, derivative property
│   └── logistic.py           # logistic weight, Gram-Schmidt, recurrence coefficients
│
├── moments/
│   └── hermite_moments.py    # m^h_k from raw moments; moment-cumulant conversion
│
├── matrices/
│   ├── basis_matrices.py     # H_n, B_n construction (coefficient matrices)
│   ├── change_of_basis.py    # solve B_n = H_n Q_n → Q_n
│   └── linear_system.py      # A~_N (Delta coefficients), A_N, b_N; solve A c = b
│
├── expansion/
│   └── density.py            # evaluate sum c_j phi_j(x*), compute C^_0 by quadrature
│
├── cos/
│   └── cos_method.py         # COS density, CF evaluation, clr computation, Fourier c_j
│
├── distances/
│   └── metrics.py            # d2 coefficients, Aitchison, log-L2, L1, L2; all via quadrature
│
├── plots/
│   └── figures.py            # one function per figure (fig1_vg_coeff, fig5_vg_dist, etc.)
│
└── utils/
    ├── quadrature.py         # uniform grid, trapezoidal rule, domain construction (eq. 22)
    └── timing.py             # CPU time utilities for Table 2
```

**config.py** should expose:
- VG parameters (from Heston & Rossi 2016)
- NIG parameters (mu=0, theta=0.05, sigma=0.2, kappa=0.3)
- Heston parameters (from Rompolis & Tzavalis 2008)
- N_max = 20 (or higher for convergence plots)
- N_fixed = [6, 16] (for density comparison figures)
- N_cos = 4096 (= 2^12)
- L = 4 (domain parameter, eq. 22)
- grid_size = 10000 (quadrature grid for distances)

---

## What Is Explicit in the Paper vs. What Must Be Sourced

### Explicit in the paper
- NIG parameters (Appendix A)
- All formulas: eqs. (7)–(22)
- COS grid size 2^12
- Domain rule eq. (22) with L=4
- clr tolerance |clr(p)| < 10
- Table 2 values
- Qualitative results for all figures

### Must be obtained from references
| Item | Reference | Location |
|---|---|---|
| VG parameters (sigma, nu, theta, mu) | Heston & Rossi (2016) | Section 2 or numerical section |
| Heston parameters (kappa, theta, xi, rho, v0, T) | Rompolis & Tzavalis (2008) | Section 3 or table of parameters |
| Logistic polynomial recurrence / construction | Heston & Rossi (2016) | Appendix or Section 2 |
| Linear system derivation (integration by parts) | Muscolino & Ricciardi (1999) | Appendix |
| Hermite moment from raw moment formulas | Rompolis & Tzavalis (2008) | Eqs. (2)–(3) |
| Triple Hermite integral (Delta coefficients) | Erdelyi et al. (1953) or Grad (1949) | Standard result |
| COS implementation details | Fang & Oosterlee (2009) | Algorithm 1 |
| Heston CF branch cut fix | Lord & Kahl (2010) | Full paper |
